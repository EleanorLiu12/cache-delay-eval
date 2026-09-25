import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aiohttp import web

from cache_delay_eval.analyze_patterns import load_run, main as analyze_main, pair
from cache_delay_eval.live_replay import replay
from cache_delay_eval.materialize import materialize
from cache_delay_eval.paired_pilot import build_plan, command
from cache_delay_eval.session_gen import GeneratorConfig, WorkloadKnobs, build_trace
from cache_delay_eval.trace import TraceHeader, TraceRequest, read_trace, write_trace


class MaterializeTests(unittest.TestCase):
    def test_preserves_complete_block_identity_and_partial_session_extension(self):
        h, rs, _ = build_trace(WorkloadKnobs(P=32, rho_mix=.5), GeneratorConfig(requests=65, seed=2))
        mh, ms = materialize(h, rs, list(range(100, 116)), "test-model")
        identities = {}
        encoded = {}
        previous = {}
        for raw, actual in zip(rs, ms):
            self.assertEqual(actual.prompt_tokens, len(actual.token_ids))
            self.assertEqual(actual.arrival_time_ms, raw.arrival_time_ms)
            self.assertEqual(actual.output_tokens, raw.output_tokens)
            for i, identity in enumerate(raw.block_hashes):
                block = actual.token_ids[i * 16:(i + 1) * 16]
                self.assertEqual(identities.setdefault(identity, block), block)
                self.assertEqual(encoded.setdefault(block, identity), identity)
            if actual.session_id in previous:
                parent = previous[actual.session_id]
                self.assertEqual(actual.token_ids[:len(parent)], parent)
            previous[actual.session_id] = actual.token_ids
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            write_trace(path, mh, ms)
            self.assertEqual(read_trace(path), (mh, ms))

    def test_rejects_wrong_source_and_noninjective_alphabet(self):
        h, rs, _ = build_trace(WorkloadKnobs(P=32), GeneratorConfig(requests=1))
        with self.assertRaises(ValueError):
            materialize(h, rs, [1, 2], "m")
        with self.assertRaises(ValueError):
            materialize(replace(h, source="unsupported-source"), rs, list(range(16)), "m")


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.condition = "off"
        self.bad_usage = False
        self.active = self.peak = 0
        self.bodies = []
        app = web.Application()
        app.router.add_post("/v1/completions", self.serve)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        self.trace = self.root / "trace.jsonl"
        h = TraceHeader("test", "2026-09-22", "synthetic", model_id="test", block_size=16)
        rs = [TraceRequest("a", 0, 2, token_ids=tuple(range(32)), prompt_tokens=32, session_id="s"),
              TraceRequest("b", 5, 2, token_ids=tuple(range(48)), prompt_tokens=48,
                           session_id="s", parent_request_id="a")]
        write_trace(self.trace, h, rs)

    async def asyncTearDown(self):
        await self.runner.cleanup()
        self.directory.cleanup()

    async def serve(self, request):
        body = await request.json()
        self.bodies.append(body)
        self.active += 1
        self.peak = max(self.peak, self.active)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await asyncio.sleep(.03)
        # Empty decoded text still carries a generated token ID.
        events = [dict(choices=[dict(text="", token_ids=[100], finish_reason=None)]),
                  dict(choices=[dict(text="x", token_ids=[101], finish_reason="length")]),
                  dict(choices=[], usage=dict(prompt_tokens=len(body["prompt"]),
                       completion_tokens=1 if self.bad_usage else 2,
                       prompt_tokens_details=dict(cached_tokens=16 if self.condition == "on" else 0)))]
        for event in events:
            data = ("data: " + json.dumps(event) + "\n\n").encode()
            await response.write(data[:8])
            await response.write(data[8:])
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        self.active -= 1
        return response

    async def run_one(self, repeat=0):
        folder = self.root / f"run-{repeat}-{self.condition}"
        folder.mkdir()
        meta = dict(trace="trace.jsonl", condition=self.condition, repeat=repeat,
                    capacity_blocks=2048, engine_config={"model_revision": "test"})
        summary = await replay(self.trace, self.base, "test", folder / "requests.jsonl", meta,
                               max_dispatch_lag_ms=2000)
        return folder / "requests.jsonl", summary

    async def test_concurrent_open_loop_and_exact_output(self):
        path, summary = await self.run_one()
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(self.peak, 2)
        self.assertEqual(summary["parent_overlap_requests"], 1)
        _, rows = load_run(path)
        self.assertLess(rows["b"]["dispatched_ms"], rows["a"]["completed_ms"])
        self.assertTrue(all(b["ignore_eos"] and b["min_tokens"] == 2 for b in self.bodies))

    async def test_wrong_output_length_invalidates_run(self):
        self.bad_usage = True
        path, summary = await self.run_one()
        self.assertEqual(summary["errors"], 2)
        with self.assertRaises(ValueError):
            load_run(path)

    async def test_full_pair_analysis_and_mismatch_rejection(self):
        plans = []
        for repeat in (0, 1):
            pair_data = {}
            for condition in ("on", "off"):
                self.condition = condition
                path, _ = await self.run_one(repeat)
                pair_data[condition] = load_run(path)
                plans.append(dict(trace="trace.jsonl", capacity_blocks=2048, repeat=repeat, condition=condition))
            rows = pair(pair_data["on"], pair_data["off"])
            self.assertEqual(len(rows), 2)
            pair_data["off"][0]["trace_sha256"] = "wrong"
            with self.assertRaises(ValueError):
                pair(pair_data["on"], pair_data["off"])
        (self.root / "plan.json").write_text(json.dumps(dict(runs=plans)))
        output = self.root / "analysis.json"
        self.assertEqual(analyze_main([str(self.root), "--output", str(output)]), 0)
        analysis = json.loads(output.read_text())
        self.assertEqual(len(analysis["requests"]), 4)
        self.assertFalse(analysis["patterns"][0]["positive_pearson_in_all_repetitions"])
        (self.root / "run-1-off" / "requests.jsonl").unlink()
        with self.assertRaises(ValueError):
            analyze_main([str(self.root), "--output", str(self.root / "bad.json")])


class PlanTests(unittest.TestCase):
    def test_real_plan_has_paired_conditions_and_only_cache_flag_changes(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "config/pattern-pilot.json").read_text())
        runs = build_plan(config, root)
        self.assertEqual(len(runs), 16)
        self.assertEqual([r["condition"] for r in runs[:4]], ["off", "on", "on", "off"])
        a, b = [command(config, r, "fixed-revision") for r in runs[:2]]
        self.assertEqual(a[:-1], b[:-1])
        config["max_model_len"] = 1024
        with self.assertRaises(ValueError):
            build_plan(config, root)


if __name__ == "__main__":
    unittest.main()
