"""Client-side multi-instance routing replay for the LMetric failure-case study.

The client is the router. It keeps LMetric's per-instance indicators the way
blitz-router does: batch size and queued prefill tokens change at dispatch,
first token and completion; the prefix index follows engine block events
(``events`` mode) or assumes every dispatched prefix stays cached
(``dispatch`` mode, the approximate-tree behavior of open-source routers).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import statistics
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

BLOCK = 16
VOCAB = (1000, 100000)
POLICIES = ("load", "lmetric", "affinity")
INDEX_MODES = ("events", "dispatch")
CLASS_BLOCKS = 32


@dataclass
class Request:
    rid: str
    arrival_ms: float
    blocks: list[int]
    input_len: int
    output_len: int
    tokens: list[int] = field(default_factory=list, repr=False)

    @property
    def full_blocks(self) -> list[int]:
        return self.blocks[: self.input_len // BLOCK]

    @property
    def cls(self) -> str:
        key = ",".join(map(str, self.blocks[:CLASS_BLOCKS]))
        return hashlib.sha1(key.encode()).hexdigest()[:12]


class TokenBook:
    """Deterministic 16-token content per trace block id."""

    def __init__(self, salt: str):
        self.salt = salt
        self.content: dict[int, tuple[int, ...]] = {}
        self.by_tokens: dict[tuple[int, ...], int] = {}

    def block(self, block_id: int) -> tuple[int, ...]:
        tokens = self.content.get(block_id)
        if tokens is None:
            rng = random.Random(f"{self.salt}:{block_id}")
            tokens = tuple(rng.randrange(*VOCAB) for _ in range(BLOCK))
            self.content[block_id] = tokens
            self.by_tokens[tokens] = block_id
        return tokens

    def materialize(self, request: Request) -> None:
        tokens: list[int] = []
        for block_id in request.blocks:
            tokens.extend(self.block(block_id))
        request.tokens = tokens[: request.input_len]


def load_trace(path, start_s, duration_s, time_scale, max_input, max_output, book):
    """Slice a Qwen-Bailian trace; long inputs keep their prefix blocks."""
    requests = []
    with open(path) as stream:
        for line in stream:
            row = json.loads(line)
            t = row["timestamp"] - start_s
            if not 0 <= t < duration_s:
                continue
            input_len = min(row["input_length"], max_input)
            request = Request(rid=f"c{row['chat_id']}", arrival_ms=t / time_scale * 1000,
                              blocks=row["hash_ids"][: math.ceil(input_len / BLOCK)],
                              input_len=input_len,
                              output_len=max(1, min(row["output_length"], max_output)))
            book.materialize(request)
            requests.append(request)
    return sorted(requests, key=lambda r: r.arrival_ms)


def synth_burst(duration_s, bg_rate, bg_inputs, hot_rate, hot_start_s, hot_duration_s,
                hot_prefix, hot_suffix, output_len, seed, book):
    """Poisson background with unique prefixes plus one hot shared-prefix burst."""
    rng = random.Random(seed)
    next_id = [10**9]

    def fresh(n):
        ids = list(range(next_id[0], next_id[0] + n))
        next_id[0] += n
        return ids

    hot_blocks = list(range(2 * 10**9, 2 * 10**9 + math.ceil(hot_prefix / BLOCK)))
    requests = []
    t = rng.expovariate(bg_rate)
    while t < duration_s:
        n = rng.choice(bg_inputs)
        requests.append(Request(f"b{len(requests)}", t * 1000, fresh(math.ceil(n / BLOCK)), n, output_len))
        t += rng.expovariate(bg_rate)
    t = hot_start_s + rng.expovariate(hot_rate)
    while t < min(duration_s, hot_start_s + hot_duration_s):
        n = hot_prefix + hot_suffix
        requests.append(Request(f"h{len(requests)}", t * 1000,
                                hot_blocks + fresh(math.ceil(hot_suffix / BLOCK)), n, output_len))
        t += rng.expovariate(hot_rate)
    for request in requests:
        book.materialize(request)
    return sorted(requests, key=lambda r: r.arrival_ms)


class InstanceView:
    def __init__(self, url: str, events: str | None):
        self.url = url.rstrip("/")
        self.events = events
        self.bs = 0
        self.queued = 0
        self.resident: Counter[int] = Counter()
        self.engine_hash: dict[object, int] = {}
        self.last_seq: int | None = None
        self.seq_gaps = 0
        self.unknown_blocks = 0

    def hit_blocks(self, request: Request) -> int:
        hits = 0
        for block_id in request.full_blocks:
            if self.resident[block_id] <= 0:
                break
            hits += 1
        return hits

    def apply(self, event: dict, book: TokenBook) -> None:
        kind = event.get("type")
        if kind == "BlockStored":
            tokens = event["token_ids"]
            for i, engine_hash in enumerate(event["block_hashes"]):
                block_id = book.by_tokens.get(tuple(tokens[i * BLOCK:(i + 1) * BLOCK]))
                if block_id is None:
                    self.unknown_blocks += 1  # generated-output blocks
                    continue
                self.engine_hash[engine_hash] = block_id
                self.resident[block_id] += 1
        elif kind == "BlockRemoved":
            for engine_hash in event["block_hashes"]:
                block_id = self.engine_hash.pop(engine_hash, None)
                if block_id is not None:
                    self.resident[block_id] -= 1
                    if self.resident[block_id] <= 0:
                        del self.resident[block_id]
        elif kind == "AllBlocksCleared":
            self.resident.clear()
            self.engine_hash.clear()


def choose(policy: str, views: list[InstanceView], request: Request, turn: int):
    hits = [v.hit_blocks(request) for v in views]
    if policy == "load":
        scores = [(v.bs, v.queued) for v in views]
    elif policy == "lmetric":
        # blitz-router lmetric-q: prefill_tokens(req) * (bs + 1)
        scores = [((v.queued + request.input_len - h * BLOCK) * (v.bs + 1),) for v, h in zip(views, hits)]
    elif policy == "affinity":
        scores = [(-h, v.bs) for v, h in zip(views, hits)]
    else:
        raise ValueError(policy)
    best = min(scores)
    tied = [i for i, s in enumerate(scores) if s == best]
    return tied[turn % len(tied)], hits, scores


async def stream(session, url, payload, on_first):
    loop = asyncio.get_running_loop()
    start = loop.time()
    first = finish = None
    usage: dict = {}
    async with session.post(url + "/v1/completions", json=payload) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:500]}")
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            event = json.loads(line[5:])
            if event.get("error"):
                raise RuntimeError(str(event["error"])[:500])
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if first is None and choice.get("token_ids"):
                    first = loop.time()
                    on_first()
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
    if first is None or finish is None:
        raise RuntimeError("stream ended without first token or finish reason")
    if usage.get("completion_tokens") != payload["max_tokens"]:
        raise RuntimeError(f"output length {usage.get('completion_tokens')} != {payload['max_tokens']}")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return dict(ttft_ms=(first - start) * 1000, e2e_ms=(loop.time() - start) * 1000,
                engine_cached_tokens=cached, prompt_tokens=usage.get("prompt_tokens"))


async def subscribe(view: InstanceView, book: TokenBook, stop: asyncio.Event):
    import msgspec
    import zmq
    import zmq.asyncio

    socket = zmq.asyncio.Context.instance().socket(zmq.SUB)
    socket.connect(view.events)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    decode = msgspec.msgpack.Decoder()
    try:
        while not stop.is_set():
            if not await socket.poll(100):
                continue
            _topic, seq_bytes, payload = await socket.recv_multipart()
            seq = int.from_bytes(seq_bytes, "big")
            if view.last_seq is not None and seq != view.last_seq + 1:
                view.seq_gaps += 1
            view.last_seq = seq
            for event in decode.decode(payload)[1]:
                view.apply(event, book)
    finally:
        socket.close(0)


def post(url: str, path: str) -> None:
    request = urllib.request.Request(url + path, data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status >= 300:
            raise RuntimeError(f"{url}{path} returned HTTP {response.status}")


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


async def replay(requests, views, book, policy, index_mode, model, output, meta,
                 timeout_s=1800, reset=True, snapshot_s=1.0):
    import aiohttp

    if reset:
        for view in views:
            post(view.url, "/reset_prefix_cache")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stop = asyncio.Event()
    subscribers = []
    if index_mode == "events":
        subscribers = [asyncio.create_task(subscribe(v, book, stop)) for v in views]
        await asyncio.sleep(1.0)  # let SUB sockets connect before traffic starts
    loop = asyncio.get_running_loop()
    payloads = [dict(model=model, prompt=r.tokens, max_tokens=r.output_len,
                     ignore_eos=True, temperature=0, stream=True,
                     stream_options={"include_usage": True}, return_token_ids=True,
                     add_special_tokens=False) for r in requests]
    turn = [0]
    with output.open("x") as log:
        log.write(json.dumps(dict(meta, type="run_meta", policy=policy, index_mode=index_mode,
                                  model=model, requests=len(requests),
                                  instances=[v.url for v in views], wall_start=time.time())) + "\n")

        async def snapshots():
            while not stop.is_set():
                log.write(json.dumps(dict(type="snapshot", t_ms=(loop.time() - epoch) * 1000,
                                          bs=[v.bs for v in views], queued=[v.queued for v in views],
                                          resident=[len(v.resident) for v in views])) + "\n")
                await asyncio.sleep(snapshot_s)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s),
                                         connector=aiohttp.TCPConnector(limit=0)) as session:
            epoch = loop.time() + 0.2
            snapper = asyncio.create_task(snapshots())

            async def send(i, request):
                await asyncio.sleep(max(0, epoch + request.arrival_ms / 1000 - loop.time()))
                dispatched = (loop.time() - epoch) * 1000
                chosen, hits, scores = choose(policy, views, request, turn[0])
                turn[0] += 1
                view = views[chosen]
                new_tokens = request.input_len - hits[chosen] * BLOCK
                view.bs += 1
                view.queued += new_tokens
                if index_mode == "dispatch":
                    for block_id in request.full_blocks:
                        view.resident[block_id] = max(view.resident[block_id], 1)
                pending = [True]

                def first_token():
                    pending[0] = False
                    view.queued -= new_tokens

                row = dict(type="request", rid=request.rid, cls=request.cls,
                           scheduled_ms=request.arrival_ms, dispatched_ms=dispatched,
                           instance=chosen, hits=hits, bs=[v.bs for v in views],
                           queued=[v.queued for v in views], score=[list(s) for s in scores],
                           input_len=request.input_len, output_len=request.output_len,
                           predicted_hit_tokens=hits[chosen] * BLOCK)
                try:
                    row.update(await stream(session, view.url, payloads[i], first_token))
                    row["status"] = "ok"
                except Exception as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                finally:
                    if pending[0]:
                        view.queued -= new_tokens
                    view.bs -= 1
                row["completed_ms"] = (loop.time() - epoch) * 1000
                log.write(json.dumps(row) + "\n")
                return row

            rows = await asyncio.gather(*(send(i, r) for i, r in enumerate(requests)))
            stop.set()
            await snapper
        for task in subscribers:
            await task
        ok = [r for r in rows if r["status"] == "ok"]
        ttft = [r["ttft_ms"] for r in ok]
        checked = [r for r in ok if r.get("engine_cached_tokens") is not None]
        summary = dict(
            type="run_summary", policy=policy, index_mode=index_mode,
            requests=len(rows), errors=len(rows) - len(ok),
            late_dispatches_50ms=sum(r["dispatched_ms"] - r["scheduled_ms"] > 50 for r in rows),
            per_instance=[sum(r["instance"] == i for r in rows) for i in range(len(views))],
            ttft_ms=dict(mean=statistics.fmean(ttft) if ttft else None,
                         p50=percentile(ttft, .5), p90=percentile(ttft, .9), p99=percentile(ttft, .99)),
            engine_cached_fraction=(sum(r["engine_cached_tokens"] for r in checked)
                                    / max(1, sum(r["input_len"] for r in checked))) if checked else None,
            index_error_tokens_mean=(statistics.fmean(abs(r["predicted_hit_tokens"] - r["engine_cached_tokens"])
                                                      for r in checked) if checked else None),
            missing_engine_cached=len(ok) - len(checked),
            event_seq_gaps=[v.seq_gaps for v in views],
            unknown_event_blocks=[v.unknown_blocks for v in views])
        log.write(json.dumps(summary) + "\n")
    return summary


def build_workload(args, book):
    if args.workload == "trace":
        return load_trace(args.trace, args.start_s, args.duration_s, args.time_scale,
                          args.max_input, args.max_output, book)
    return synth_burst(args.duration_s, args.bg_rate, [int(x) for x in args.bg_inputs.split(",")],
                       args.hot_rate, args.hot_start_s, args.hot_duration_s, args.hot_prefix,
                       args.hot_suffix, args.max_output, args.seed, book)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instances", required=True,
                   help="comma-separated http://host:port=tcp://host:eventport")
    p.add_argument("--policy", choices=POLICIES, required=True)
    p.add_argument("--index", choices=INDEX_MODES, default="events")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--output", required=True)
    p.add_argument("--workload", choices=("trace", "burst"), required=True)
    p.add_argument("--trace")
    p.add_argument("--start-s", type=float, default=0.0)
    p.add_argument("--duration-s", type=float, default=600.0)
    p.add_argument("--time-scale", type=float, default=1.0, help="trace seconds per replay second")
    p.add_argument("--max-input", type=int, default=8192)
    p.add_argument("--max-output", type=int, default=256)
    p.add_argument("--bg-rate", type=float, default=4.0)
    p.add_argument("--bg-inputs", default="512,1024,2048,4096")
    p.add_argument("--hot-rate", type=float, default=4.0)
    p.add_argument("--hot-start-s", type=float, default=60.0)
    p.add_argument("--hot-duration-s", type=float, default=120.0)
    p.add_argument("--hot-prefix", type=int, default=4096)
    p.add_argument("--hot-suffix", type=int, default=256)
    p.add_argument("--seed", type=int, default=699)
    p.add_argument("--salt", default="a30-2026-10-02")
    p.add_argument("--no-reset", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="build the workload and print its shape")
    args = p.parse_args(argv)
    book = TokenBook(args.salt)
    requests = build_workload(args, book)
    shape = dict(requests=len(requests),
                 span_s=requests[-1].arrival_ms / 1000 if requests else 0,
                 input_tokens=sum(r.input_len for r in requests),
                 output_tokens=sum(r.output_len for r in requests),
                 classes=len({r.cls for r in requests}))
    print(json.dumps(shape))
    if args.dry_run:
        return
    views = []
    for spec in args.instances.split(","):
        url, _, events = spec.partition("=")
        views.append(InstanceView(url, events or None))
    if args.index == "events" and any(v.events is None for v in views):
        p.error("events index needs url=tcp://host:port for every instance")
    meta = dict(vars(args), workload_shape=shape)
    summary = asyncio.run(replay(requests, views, book, args.policy, args.index, args.model,
                                 args.output, meta, reset=not args.no_reset))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
