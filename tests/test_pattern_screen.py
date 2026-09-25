import unittest

from cache_delay_eval.pattern_screen import serial_lru, select_cells
from cache_delay_eval.trace import TraceRequest


def request(blocks):
    return TraceRequest("r", 0, 1, block_hashes=tuple(blocks), prompt_tokens=16 * len(blocks))


class PatternScreenTests(unittest.TestCase):
    def test_cached_suffix_cannot_hit_after_prefix_eviction(self):
        result = serial_lru([request([1, 2, 3]), request([1, 2, 3])], 2)
        self.assertEqual(result["request_mean_hit_rate"], 0)
        self.assertEqual(result["prompts_larger_than_capacity"], 2)

    def test_unbounded_serial_shared_prefix(self):
        result = serial_lru([request([1, 2]), request([1, 3])], 10)
        self.assertEqual(result["request_mean_hit_rate"], .25)
        self.assertEqual(result["evicted_blocks"], 0)

    def test_capacity_pressure_removes_reuse(self):
        trace = [request([1, 2]), request([3, 4]), request([1, 2])]
        self.assertEqual(serial_lru(trace, 2)["request_mean_hit_rate"], 0)
        self.assertAlmostEqual(serial_lru(trace, 4)["request_mean_hit_rate"], 1 / 3)

    def test_selection_uses_mean_and_excludes_single_turn_variant(self):
        rows = [dict(P=512, shallow_turns=t, rho_mix=m, rho=r, rho_spearman=r)
                for t, m, r in [(3, 0, .5), (3, .5, -.2), (3, .5, .2),
                                (3, .8, -.3), (1, .9, -.99)]]
        selected = {r["label"]: r["rho_mix"] for r in select_cells(rows)}
        self.assertEqual(selected, {"positive": 0, "near-zero": .5, "negative": .8})


if __name__ == "__main__":
    unittest.main()
