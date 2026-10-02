"""Independent CPU integration checks for dependency-correct chat replay.

The server observes actual request arrival times and submitted prompts; these
checks do not infer dependency correctness from the replay's own summary.
"""

import asyncio
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from aiohttp import web

from cache_delay_eval.chat_replay import ReplayConfig, SessionScript, replay_sessions


class CharacterBuilder:
    """A reversible template whose token IDs expose the submitted messages."""

    metadata = {"builder": "test-character-template", "terminal_eos_id": 0}

    def __init__(self):
        self.rendered = []

    def render(self, messages):
        snapshot = [dict(message) for message in messages]
        self.rendered.append((time.monotonic(), snapshot))
        return list(map(ord, json.dumps(snapshot, ensure_ascii=False)))

    def decode_response(self, ids):
        ids = list(ids)
        if ids and ids[-1] == 0:
            ids.pop()
        return "".join(map(chr, ids))


class ChatReplayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.builder = CharacterBuilder()
        self.scenarios = {}
        self.received = []
        self.active = self.peak = 0
        self.started = asyncio.Event()
        self.serial = 0
        app = web.Application()
        app.router.add_post("/v1/completions", self.serve)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.config = ReplayConfig(
            model="mock-model", run_id="cpu-test", max_tokens=128,
            max_model_len=32768, think_delay_s=0.025, timeout_s=2,
            require_cached_tokens=True, max_dispatch_lag_ms=2000, seed=699,
        )

    async def asyncTearDown(self):
        await self.runner.cleanup()
        self.temporary.cleanup()

    async def emit(self, response, event):
        data = ("data: " + json.dumps(event, ensure_ascii=False) + "\r\n\r\n").encode()
        # Each byte is a separate write, including bytes inside UTF-8 characters.
        for value in data:
            await response.write(bytes([value]))

    async def serve(self, request):
        loop = asyncio.get_running_loop()
        body = await request.json()
        messages = json.loads("".join(map(chr, body["prompt"])))
        user = [m["content"] for m in messages if m["role"] == "user"][-1]
        scenario = self.scenarios.get(user, {})
        seen = dict(body=body, messages=messages, received=loop.time(),
                    header_id=request.headers.get("X-Request-Id"), user=user)
        self.received.append(seen)
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.set()
        try:
            if "http_status" in scenario:
                return web.Response(status=scenario["http_status"], text="mock failure")
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await asyncio.sleep(scenario.get("first_delay", 0.005))
            tokens = scenario.get("tokens", list(map(ord, "reply:" + user)) + [0])
            await self.emit(response, {"choices": [{"index": 0, "text": "",
                                                    "token_ids": tokens[:1],
                                                    "finish_reason": None}]})
            seen["first_token_sent"] = loop.time()
            await asyncio.sleep(scenario.get("text_delay", 0.015))
            if scenario.get("stream_error"):
                await self.emit(response, {"error": {"message": "mock stream error"}})
                await response.write_eof()
                return response
            text = scenario.get("text", "reply:" + user)
            seen["text_sent"] = loop.time()
            await self.emit(response, {"choices": [{"index": 0, "text": text,
                                                    "token_ids": tokens[1:],
                                                    "finish_reason": scenario.get("finish", "stop")}]})
            usage = dict(prompt_tokens=len(body["prompt"]), completion_tokens=len(tokens),
                         prompt_tokens_details={"cached_tokens": 0})
            usage.update(scenario.get("usage_patch", {}))
            if not scenario.get("omit_usage"):
                await self.emit(response, {"choices": [], "usage": usage})
            if not scenario.get("omit_done"):
                await response.write(b"data: [DONE]\r\n\r\n")
            await response.write_eof()
            seen["stream_completed"] = loop.time()
            return response
        except (ConnectionResetError, RuntimeError):
            # Client cancellation/timeout may close the socket mid-stream.
            return response
        finally:
            self.active -= 1

    async def run_replay(self, scripts, config=None):
        self.serial += 1
        output = self.root / f"run-{self.serial}"
        summary = await replay_sessions(
            scripts, self.builder, self.base_url, output, config or self.config,
            run_meta={"test_backend": "local mock HTTP; no model execution"},
        )
        rows = [json.loads(line) for line in (output / "turns.jsonl").read_text().splitlines()]
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        self.assertTrue(events)
        self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
        self.assertIsInstance(json.loads((output / "run.json").read_text()), dict)
        return output, rows, summary

    async def test_dependency_waits_for_actual_parent_and_reuses_generated_answer(self):
        self.scenarios["first"] = {"tokens": list(map(ord, "actual answer")) + [0],
                                   "text": "a deliberately different streamed rendering"}
        _, rows, _ = await self.run_replay([
            SessionScript("session", ("first", "second", "third")),
        ])
        rows.sort(key=lambda row: row["turn_index"])
        self.assertEqual([row["status"] for row in rows], ["ok"] * 3)
        self.assertEqual([row["turn_index"] for row in rows], [1, 2, 3])
        self.assertEqual(rows[0]["assistant_content"], "actual answer")
        self.assertEqual(rows[0]["output_token_ids"], list(map(ord, "actual answer")) + [0])
        self.assertEqual(rows[0]["finish_reason"], "stop")
        self.assertEqual(self.received[1]["messages"], [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "actual answer"},
            {"role": "user", "content": "second"},
        ])
        self.assertEqual(self.received[2]["messages"][-2:], [
            {"role": "assistant", "content": "reply:second"},
            {"role": "user", "content": "third"},
        ])
        self.assertIsNone(rows[0]["parent_request_id"])
        for parent, child, observed_parent, observed_child in zip(
            rows, rows[1:], self.received, self.received[1:],
        ):
            self.assertEqual(child["parent_request_id"], parent["request_id"])
            self.assertAlmostEqual(child["release_s"],
                                   parent["completed_s"] + self.config.think_delay_s, places=6)
            self.assertGreaterEqual(child["dispatched_s"], child["release_s"])
            self.assertGreaterEqual(observed_child["received"] - observed_parent["stream_completed"],
                                    self.config.think_delay_s - 0.002)
        for (_, messages), observed in zip(self.builder.rendered[1:], self.received[:-1]):
            # Rendering dependent prompts must happen after the parent is done.
            rendered_time = next(t for t, m in self.builder.rendered if m == messages)
            self.assertGreaterEqual(rendered_time, observed["stream_completed"])

    async def test_independent_sessions_overlap_and_delayed_root_respects_release(self):
        self.scenarios["slow-a"] = {"text_delay": 0.12}
        self.scenarios["slow-b"] = {"text_delay": 0.12}
        _, rows, _ = await self.run_replay([
            SessionScript("a", ("slow-a",)), SessionScript("b", ("slow-b",)),
            SessionScript("later", ("later",), root_arrival_s=0.08),
        ])
        self.assertEqual([row["status"] for row in rows], ["ok"] * 3)
        self.assertGreaterEqual(self.peak, 2)
        observed = {entry["user"]: entry for entry in self.received}
        self.assertLess(observed["slow-b"]["received"], observed["slow-a"]["stream_completed"])
        self.assertGreaterEqual(observed["later"]["received"] - observed["slow-a"]["received"], 0.065)
        later = next(row for row in rows if row["session_id"] == "later")
        self.assertEqual(later["release_s"], 0.08)
        self.assertGreaterEqual(later["dispatched_s"], 0.08)

    async def test_fragmented_unicode_stream_ttft_uses_token_ids_before_text(self):
        self.scenarios["unicode"] = {"tokens": [ord("λ"), ord("🙂"), 0],
                                     "text": "λ🙂", "text_delay": 0.12}
        _, rows, _ = await self.run_replay([SessionScript("s", ("unicode",))])
        row = rows[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["assistant_content"], "λ🙂")
        self.assertEqual(row["streamed_text"], "λ🙂")
        self.assertEqual(row["output_token_ids"], [ord("λ"), ord("🙂"), 0])
        stream_events = row["stream_events"]
        self.assertEqual([token for event in stream_events for token in event["token_ids"]],
                         row["output_token_ids"])
        self.assertEqual([event["at"] for event in stream_events],
                         sorted(event["at"] for event in stream_events))
        self.assertEqual(stream_events[0]["text"], "")
        self.assertEqual(stream_events[0]["at"], row["first_token_s"])
        e2e_ms = (row["completed_s"] - row["dispatched_s"]) * 1000
        self.assertGreater(e2e_ms - row["client_ttft_ms"], 80)
        self.assertGreaterEqual(row["client_ttft_ms"], 0)
        self.assertGreaterEqual(row["release_to_first_token_ms"], row["client_ttft_ms"])

    async def test_payload_permits_eos_and_correlates_request_ids(self):
        self.scenarios["short"] = {"tokens": [ord("x"), 0], "text": "x"}
        _, rows, _ = await self.run_replay([SessionScript("s", ("short",))])
        self.assertEqual(rows[0]["status"], "ok")
        observed = self.received[0]
        body = observed["body"]
        self.assertEqual(body["model"], "mock-model")
        self.assertEqual(body["n"], 1)
        self.assertEqual(body["max_tokens"], 128)
        self.assertIs(body["ignore_eos"], False)
        self.assertNotIn("min_tokens", body)
        self.assertIs(body["stream"], True)
        self.assertIs(body["return_token_ids"], True)
        self.assertIs(body["skip_special_tokens"], False)
        self.assertIs(body["spaces_between_special_tokens"], False)
        self.assertEqual(body["request_id"], observed["header_id"])
        self.assertEqual(body["request_id"], rows[0]["request_id"])

    async def test_response_at_generation_cap_is_retained_without_eos(self):
        self.scenarios["capped"] = {"tokens": [ord("x")] * 128,
                                    "text": "x" * 128, "finish": "length"}
        _, rows, _ = await self.run_replay([SessionScript("s", ("capped", "next"))])
        rows.sort(key=lambda row: row["turn_index"])
        self.assertEqual([row["status"] for row in rows], ["ok", "ok"])
        self.assertEqual(rows[0]["finish_reason"], "length")
        self.assertEqual(rows[0]["assistant_content"], "x" * 128)
        self.assertEqual(self.received[1]["messages"][1], {"role": "assistant", "content": "x" * 128})

    async def test_http_failure_blocks_only_its_own_descendants(self):
        self.scenarios["fail"] = {"http_status": 503}
        _, rows, _ = await self.run_replay([
            SessionScript("failed", ("fail", "never-1", "never-2")),
            SessionScript("healthy", ("healthy", "healthy-next")),
        ])
        failed = sorted((r for r in rows if r["session_id"] == "failed"), key=lambda r: r["turn_index"])
        self.assertEqual([r["status"] for r in failed], ["error", "blocked", "blocked"])
        self.assertTrue(all(r["status"] == "ok" for r in rows if r["session_id"] == "healthy"))
        self.assertEqual(set(r["user"] for r in self.received), {"fail", "healthy", "healthy-next"})
        self.assertEqual(len(self.builder.rendered), 3)

    async def test_context_overflow_preserves_full_prompt_and_sends_nothing(self):
        config = replace(self.config, max_model_len=140, max_tokens=128)
        _, rows, _ = await self.run_replay([SessionScript("s", ("too-long", "blocked"))], config)
        rows.sort(key=lambda r: r["turn_index"])
        self.assertEqual([r["status"] for r in rows], ["context_overflow", "blocked"])
        self.assertEqual(self.received, [])
        self.assertEqual(len(self.builder.rendered), 1)
        expected = [{"role": "user", "content": "too-long"}]
        self.assertEqual(json.loads("".join(map(chr, rows[0]["prompt_token_ids"]))), expected)

    async def test_context_exactly_fits_prompt_plus_output_budget(self):
        user = "fits"
        tokens = list(map(ord, json.dumps([{"role": "user", "content": user}])))
        config = replace(self.config, max_model_len=len(tokens) + self.config.max_tokens)
        _, rows, _ = await self.run_replay([SessionScript("s", (user,))], config)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(self.received[0]["body"]["prompt"], tokens)

    async def test_invalid_usage_or_output_counts_fail_and_preserve_partial_response(self):
        invalid = {
            "wrong-completion-count": {"usage_patch": {"completion_tokens": 99}},
            "float-completion-count": {"usage_patch": {"completion_tokens": 2.0}},
            "wrong-prompt-count": {"usage_patch": {"prompt_tokens": 1}},
            "bool-prompt-count": {"usage_patch": {"prompt_tokens": True}},
            "missing-cache": {"usage_patch": {"prompt_tokens_details": {}}},
            "bool-cache": {"usage_patch": {"prompt_tokens_details": {"cached_tokens": True}}},
            "negative-cache": {"usage_patch": {"prompt_tokens_details": {"cached_tokens": -1}}},
            "excess-cache": {"usage_patch": {"prompt_tokens_details": {"cached_tokens": 100000}}},
            "absent-usage": {"omit_usage": True},
            "too-many-output-tokens": {"tokens": [ord("x")] * 129},
        }
        for label, scenario in invalid.items():
            with self.subTest(case=label):
                self.scenarios[label] = {"tokens": [ord("x"), 0], "text": "x", "text_delay": 0, **scenario}
                before = len(self.received)
                _, rows, _ = await self.run_replay([SessionScript(label, (label, "never"))])
                rows.sort(key=lambda r: r["turn_index"])
                self.assertEqual([r["status"] for r in rows], ["error", "blocked"])
                self.assertEqual(len(self.received), before + 1)
                self.assertEqual(rows[0]["output_token_ids"], self.scenarios[label]["tokens"])
                self.assertEqual(rows[0]["streamed_text"], "x")

    async def test_incomplete_stream_is_rejected(self):
        for label, scenario in (("missing-done", {"omit_done": True}),
                                ("missing-finish", {"finish": None})):
            with self.subTest(case=label):
                self.scenarios[label] = scenario
                _, rows, _ = await self.run_replay([SessionScript(label, (label,))])
                self.assertEqual(rows[0]["status"], "error")
                self.assertTrue(rows[0]["output_token_ids"])

    async def test_missing_cache_count_is_only_allowed_when_explicitly_configured(self):
        self.scenarios["no-cache"] = {"usage_patch": {"prompt_tokens_details": {}}}
        _, rows, summary = await self.run_replay(
            [SessionScript("s", ("no-cache",))], replace(self.config, require_cached_tokens=False),
        )
        self.assertEqual(rows[0]["status"], "ok")
        self.assertIsNone(rows[0]["cached_tokens"])
        self.assertEqual(summary["missing_cache_usage"], 1)

    async def test_stream_error_keeps_received_tokens_and_blocks_child(self):
        self.scenarios["broken"] = {"tokens": [ord("z"), 0], "stream_error": True}
        _, rows, _ = await self.run_replay([SessionScript("s", ("broken", "never"))])
        rows.sort(key=lambda r: r["turn_index"])
        self.assertEqual([r["status"] for r in rows], ["error", "blocked"])
        self.assertEqual(rows[0]["output_token_ids"], [ord("z")])
        self.assertEqual(len(self.received), 1)

    async def test_timeout_keeps_received_tokens_and_blocks_child(self):
        self.scenarios["timeout"] = {"tokens": [ord("t"), 0], "text_delay": 0.2}
        _, rows, _ = await self.run_replay(
            [SessionScript("s", ("timeout", "never"))], replace(self.config, timeout_s=0.06),
        )
        rows.sort(key=lambda r: r["turn_index"])
        self.assertEqual([r["status"] for r in rows], ["error", "blocked"])
        self.assertEqual(rows[0]["output_token_ids"], [ord("t")])

    async def test_existing_output_directory_is_never_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "turns.jsonl"
        marker.write_text("original evidence\n")
        with self.assertRaises(FileExistsError):
            await replay_sessions([SessionScript("s", ("hello",))], self.builder,
                                  self.base_url, output, self.config)
        self.assertEqual(marker.read_text(), "original evidence\n")
        self.assertEqual(self.received, [])

    async def test_cancellation_retains_partial_tokens_and_every_planned_turn(self):
        self.scenarios["cancel-me"] = {"tokens": [ord("c"), 0], "text_delay": 0.3}
        output = self.root / "cancelled"
        task = asyncio.create_task(replay_sessions(
            [SessionScript("active", ("cancel-me", "never")),
             SessionScript("future", ("later",), root_arrival_s=5)],
            self.builder, self.base_url, output, self.config,
        ))
        await asyncio.wait_for(self.started.wait(), timeout=2)
        # Wait until the first token has crossed the socket before cancelling.
        await asyncio.sleep(0.04)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        rows = [json.loads(line) for line in (output / "turns.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        active = next(row for row in rows if row["session_id"] == "active" and row["turn_index"] == 1)
        child = next(row for row in rows if row["session_id"] == "active" and row["turn_index"] == 2)
        future = next(row for row in rows if row["session_id"] == "future")
        self.assertEqual(active["status"], "cancelled")
        self.assertEqual(active["output_token_ids"], [ord("c")])
        self.assertEqual(child["status"], "blocked")
        self.assertEqual(future["status"], "cancelled")
        self.assertIsNone(future["dispatched_s"])
        summary = json.loads((output / "summary.json").read_text())
        self.assertIs(summary["interrupted"], True)
        self.assertIs(summary["valid"], False)
        self.assertEqual(len(self.received), 1)


if __name__ == "__main__":
    unittest.main()
