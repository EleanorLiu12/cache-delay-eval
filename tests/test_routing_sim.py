import unittest

import json
import tempfile

from cache_delay_eval.routing_sim import (BLOCK, CLASS_BLOCKS, Cost, Engine, EngineConfig, HotspotDetector,
                                          RouterIndex, SimRequest, SMetric, load_trace, simulate)


def req(rid, keys, arrival_ms=0.0, extra=0, output_len=4):
    return SimRequest(rid, arrival_ms, list(keys), len(keys) * BLOCK + extra, output_len)


class EngineTests(unittest.TestCase):
    def test_second_request_hits_all_but_last_block_and_pool_is_restored(self):
        a = req("a", range(8), 0)
        b = req("b", range(8), 1000)
        simulate([a, b], 1, "load", "exact", Cost(), EngineConfig(num_blocks=64))
        self.assertEqual(a.engine_hit, 0)
        self.assertEqual(b.engine_hit, 7 * BLOCK)   # vLLM recomputes the last full block

    def test_blocks_return_to_free_queue(self):
        cfg = EngineConfig(num_blocks=64)
        engine = Engine(cfg, Cost())
        r = req("a", range(10), output_len=20)
        engine.waiting.append(r)
        while engine.waiting or engine.running:
            plan, _ = engine.schedule()
            engine.finish_step(plan, 0.0, lambda _: None, lambda _: None)
        self.assertEqual(len(engine.free), cfg.num_blocks)
        self.assertEqual(r.generated, 20)

    def test_preemption_when_blocks_run_out(self):
        reqs = [req(f"r{i}", range(100 * i, 100 * i + 6), 0, output_len=64) for i in range(4)]
        simulate(reqs, 1, "load", "exact", Cost(), EngineConfig(num_blocks=30))
        self.assertTrue(all(r.done_ms is not None for r in reqs))
        self.assertGreater(sum(r.preempted for r in reqs), 0)


class IndexTests(unittest.TestCase):
    def test_exact_index_forgets_evicted_prefix_but_dispatch_index_does_not(self):
        cfg = EngineConfig(num_blocks=12)
        engine = Engine(cfg, Cost())
        first = req("a", range(4))
        flood = [req(f"f{i}", range(1000 + 10 * i, 1000 + 10 * i + 4)) for i in range(10)]
        indexes = {mode: RouterIndex(mode, engine) for mode in ("exact", "dispatch", "lru")}
        for r in [first] + flood:
            for index in indexes.values():
                index.dispatched(r.keys)
            engine.waiting.append(r)
            while engine.waiting or engine.running:
                plan, _ = engine.schedule()
                engine.finish_step(plan, 0.0, lambda _: None, lambda _: None)
        self.assertEqual(indexes["exact"].hits(first.keys), 0)
        self.assertEqual(indexes["dispatch"].hits(first.keys), 4)
        self.assertEqual(indexes["lru"].hits(first.keys), 0)

    def test_lmetric_prefers_cached_instance_when_load_equal(self):
        a = req("a", range(32), 0)
        b = req("b", range(32), 5000)
        simulate([a, b], 2, "lmetric", "exact", Cost(), EngineConfig(num_blocks=200))
        self.assertEqual(a.instance, b.instance)

    def test_inflight_index_counts_prefix_until_prefill_finishes(self):
        engine = Engine(EngineConfig(num_blocks=64), Cost())
        exact, inflight = RouterIndex("exact", engine), RouterIndex("inflight", engine)
        keys = list(range(8))
        inflight.dispatched(keys)
        self.assertEqual((exact.hits(keys), inflight.hits(keys)), (0, 8))
        inflight.prefilled(keys)
        self.assertEqual(inflight.hits(keys), 0)


class DetectorTests(unittest.TestCase):
    def hot(self, rid, now, detector, hits, bs, queued):
        r = SimRequest(rid, now, list(range(CLASS_BLOCKS)), CLASS_BLOCKS * BLOCK + 8, 4, "hot")
        r.holders = sum(h >= CLASS_BLOCKS for h in hits)
        detector.observe(r, now)
        return r, detector.allowed(r, hits, bs, queued)

    def test_filters_holders_only_after_2m_consecutive_hotspot_wins(self):
        d = HotspotDetector(4, window_ms=60_000, min_count=1)
        hits, bs, queued = [CLASS_BLOCKS, 0, 0, 0], [3, 3, 3, 3], [0, 0, 0, 0]
        outcomes = [self.hot(f"r{k}", k, d, hits, bs, queued) for k in range(3)]
        # x = 1 > |M|/N = 1/4: every request alarms; the second one completes 2|M| = 2 wins.
        self.assertTrue(all(r.alarm for r, _ in outcomes))
        self.assertEqual([a for _, a in outcomes], [None, [1, 2, 3], [1, 2, 3]])

    def test_streak_resets_when_lmetric_already_prefers_a_cold_instance(self):
        d = HotspotDetector(4, window_ms=60_000, min_count=1)
        hits, queued = [CLASS_BLOCKS, 0, 0, 0], [0, 0, 0, 0]
        self.hot("a", 0, d, hits, [3, 3, 3, 3], queued)
        _, allowed = self.hot("b", 1, d, hits, [300, 0, 0, 0], queued)
        self.assertIsNone(allowed)
        self.assertEqual(d.streak["hot"], 0)

    def test_no_alarm_when_share_within_coverage(self):
        d = HotspotDetector(4, window_ms=60_000, min_count=1)
        for k in range(3):
            d.observe(SimRequest(f"o{k}", k, [1000 + k], BLOCK, 1, f"other{k}"), k)
        r, allowed = self.hot("a", 3, d, [CLASS_BLOCKS, 0, 0, 0], [0] * 4, [0] * 4)
        self.assertFalse(r.alarm)    # x = 1/4, not above |M|/N = 1/4
        self.assertIsNone(allowed)

    def test_detector_policy_spreads_a_dominant_class(self):
        reqs = [SimRequest(f"r{k}", 50.0 * k, list(range(CLASS_BLOCKS)), CLASS_BLOCKS * BLOCK + 8, 64, "hot")
                for k in range(40)]
        simulate(reqs, 4, "lmetric_detector", "exact", Cost(), EngineConfig(num_blocks=2000))
        self.assertTrue(any(r.filtered for r in reqs))
        self.assertEqual(len({r.instance for r in reqs}), 4)


class SessionTests(unittest.TestCase):
    def test_parent_is_longest_covering_prefix(self):
        rows = [dict(timestamp=0, input_length=64, output_length=1, hash_ids=[1, 2, 3, 4]),
                dict(timestamp=1, input_length=96, output_length=1, hash_ids=[1, 2, 3, 4, 5, 6]),
                dict(timestamp=2, input_length=96, output_length=1, hash_ids=[1, 9, 9, 9, 9, 9])]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
        a, b, c = load_trace(f.name, 0, 10, 1.0)
        self.assertEqual((a.parent, b.parent, c.parent), (None, a.rid, None))   # c shares 1 of 6 blocks
        self.assertEqual(b.est_hit, 4 * BLOCK)

    def test_pin_follows_parent_even_when_another_instance_is_idle(self):
        a = req("a", range(32), 0, output_len=2000)
        b = req("b", range(32, 64), 1, output_len=2000)
        c = req("c", range(40), 2)
        c.parent, c.est_hit = "a", 32 * BLOCK
        simulate([a, b, c], 3, "pin", "exact", Cost(), EngineConfig(num_blocks=2000))
        self.assertEqual(c.instance, a.instance)


class SMetricTests(unittest.TestCase):
    def test_sticks_to_session_within_slo_and_migrates_beyond_it(self):
        r = SimRequest("r", 0, list(range(64)), 64 * BLOCK, 4)
        r.parent, r.est_hit = "p", 60 * BLOCK
        hits, bs = [60, 0], [5, 0]
        self.assertEqual(SMetric().choose(Cost(), bs, [0.0, 0.0], hits, r, 0), 0)     # LMetric would pick 1
        self.assertEqual(SMetric().choose(Cost(), bs, [5000.0, 0.0], hits, r, 0), 1)  # queue misses the SLO

    def test_first_turn_and_evicted_session_are_balanced(self):
        r = SimRequest("r", 0, list(range(64)), 64 * BLOCK, 4)
        hits, bs = [60, 0], [40, 0]    # balance branch is still cache-aware, so make 0 busy
        self.assertEqual(SMetric().choose(Cost(), bs, [0.0, 0.0], hits, r, 0), 1)
        r.parent, r.est_hit = "p", 60 * BLOCK
        self.assertEqual(SMetric().choose(Cost(), bs, [0.0, 0.0], [20, 0], r, 0), 1)  # 20 < 0.5 * 60


if __name__ == "__main__":
    unittest.main()
