import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from aiohttp import web

from cache_delay_eval.routing import (BLOCK, InstanceView, Request, TokenBook, choose,
                                      load_trace, replay, synth_burst)


def request(rid, blocks, input_len, arrival_ms=0.0, output_len=4):
    return Request(rid, arrival_ms, blocks, input_len, output_len)


class TokenBookTests(unittest.TestCase):
    def test_shared_blocks_give_shared_token_prefixes(self):
        book = TokenBook("t")
        a, b = request("a", [1, 2, 3], 40), request("b", [1, 2, 9], 48)
        book.materialize(a)
        book.materialize(b)
        self.assertEqual(len(a.tokens), 40)
        self.assertEqual(a.tokens[:32], b.tokens[:32])
        self.assertNotEqual(a.tokens[32:40], b.tokens[32:40])
        self.assertEqual(TokenBook("t").block(1), book.block(1))
        self.assertEqual(book.by_tokens[book.block(2)], 2)


class TraceTests(unittest.TestCase):
    def test_slice_scales_time_and_truncates_to_prefix_blocks(self):
        rows = [dict(chat_id=0, timestamp=10.0, input_length=70, output_length=900, hash_ids=[5, 6, 7, 8, 9]),
                dict(chat_id=1, timestamp=12.0, input_length=20, output_length=3, hash_ids=[5, 11]),
                dict(chat_id=2, timestamp=30.0, input_length=20, output_length=3, hash_ids=[5, 12])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            reqs = load_trace(path, 10.0, 10.0, 2.0, 40, 256, TokenBook("t"))
        self.assertEqual([r.rid for r in reqs], ["c0", "c1"])
        self.assertEqual(reqs[0].blocks, [5, 6, 7])
        self.assertEqual((reqs[0].input_len, reqs[0].output_len, len(reqs[0].tokens)), (40, 256, 40))
        self.assertEqual(reqs[1].arrival_ms, 1000.0)

    def test_burst_shares_only_the_hot_prefix(self):
        reqs = synth_burst(30, 1.0, [64], 5.0, 5, 10, 64, 16, 4, 1, TokenBook("t"))
        hot = [r for r in reqs if r.rid.startswith("h")]
        self.assertTrue(hot and all(5000 <= r.arrival_ms < 15000 for r in hot))
        self.assertEqual(len({tuple(r.tokens[:64]) for r in hot}), 1)
        self.assertEqual(len({tuple(r.tokens[64:]) for r in hot}), len(hot))


class PolicyTests(unittest.TestCase):
    def views(self):
        a, b = InstanceView("http://a", None), InstanceView("http://b", None)
        a.resident.update({1: 1, 2: 1, 3: 1})
        return a, b

    def test_hits_stop_at_first_missing_block(self):
        a, _ = self.views()
        self.assertEqual(a.hit_blocks(request("r", [1, 2, 9, 3], 64)), 2)
        self.assertEqual(a.hit_blocks(request("r", [1, 2, 3], 40)), 2)  # partial last block

    def test_lmetric_multiplies_prefill_tokens_by_batch_plus_one(self):
        a, b = self.views()
        r = request("r", [1, 2, 3, 4], 64)
        a.bs, b.bs = 3, 0
        index, hits, scores = choose("lmetric", [a, b], r, 0)
        self.assertEqual(hits, [3, 0])
        self.assertEqual(scores, [(16 * 4,), (64 * 1,)])
        self.assertEqual(index, 0)  # tie broken by turn
        self.assertEqual(choose("lmetric", [a, b], r, 1)[0], 1)
        a.bs = 4
        self.assertEqual(choose("lmetric", [a, b], r, 0)[0], 1)

    def test_load_and_affinity(self):
        a, b = self.views()
        r = request("r", [1, 2, 3, 4], 64)
        a.bs = 2
        self.assertEqual(choose("load", [a, b], r, 0)[0], 1)
        self.assertEqual(choose("affinity", [a, b], r, 0)[0], 0)

    def test_events_track_store_and_remove(self):
        book = TokenBook("t")
        view = InstanceView("http://a", "tcp://a")
        tokens = list(book.block(1)) + list(book.block(2)) + [7] * BLOCK
        view.apply(dict(type="BlockStored", block_hashes=[11, 12, 13], token_ids=tokens), book)
        self.assertEqual((view.resident[1], view.resident[2], view.unknown_blocks), (1, 1, 1))
        view.apply(dict(type="BlockRemoved", block_hashes=[11]), book)
        self.assertNotIn(1, view.resident)
        view.apply(dict(type="AllBlocksCleared"), book)
        self.assertFalse(view.resident)


class ReplayTests(unittest.TestCase):
    def test_mock_cluster_counts_and_restores_indicators(self):
        async def run():
            seen = []

            async def completions(req):
                body = await req.json()
                seen.append(req.app["name"])
                resp = web.StreamResponse()
                await resp.prepare(req)
                for i in range(body["max_tokens"]):
                    chunk = dict(choices=[dict(token_ids=[1], finish_reason="length" if i == body["max_tokens"] - 1 else None)])
                    await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    await asyncio.sleep(0.005)
                usage = dict(prompt_tokens=len(body["prompt"]), completion_tokens=body["max_tokens"],
                             prompt_tokens_details=dict(cached_tokens=0))
                await resp.write(f"data: {json.dumps(dict(choices=[], usage=usage))}\n\ndata: [DONE]\n\n".encode())
                return resp

            runners, urls = [], []
            for name in ("a", "b"):
                app = web.Application()
                app["name"] = name
                app.router.add_post("/v1/completions", completions)
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, "127.0.0.1", 0)
                await site.start()
                runners.append(runner)
                urls.append(f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}")
            book = TokenBook("t")
            reqs = [request(f"r{i}", [1, 2, 100 + i], 40, arrival_ms=i * 5.0) for i in range(8)]
            for r in reqs:
                book.materialize(r)
            views = [InstanceView(u, None) for u in urls]
            with tempfile.TemporaryDirectory() as tmp:
                summary = await replay(reqs, views, book, "affinity", "dispatch", "m",
                                       Path(tmp) / "run.jsonl", {}, reset=False, snapshot_s=0.01)
                rows = [json.loads(l) for l in (Path(tmp) / "run.jsonl").read_text().splitlines()]
            for runner in runners:
                await runner.cleanup()
            return summary, rows, views

        summary, rows, views = asyncio.run(run())
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(sum(summary["per_instance"]), 8)
        self.assertEqual(max(summary["per_instance"]), 8)  # dispatch index keeps the shared prefix sticky
        self.assertEqual([(v.bs, v.queued) for v in views], [(0, 0), (0, 0)])
        self.assertEqual(rows[0]["type"], "run_meta")
        self.assertEqual(rows[-1]["type"], "run_summary")


if __name__ == "__main__":
    unittest.main()
