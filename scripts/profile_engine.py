"""Measure one vLLM instance's prefill and decode costs for the routing simulator.

Prefill: one request at a time on an idle engine, max_tokens=1, with a warm
request first when part of the prompt should be cached. Decode: B concurrent
requests with unique prompts; the step time is the median gap between
streamed chunks after the first few tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

VOCAB = (1000, 100000)


def tokens(rng, n):
    return [rng.randrange(*VOCAB) for _ in range(n)]


async def run_one(session, url, model, prompt, max_tokens):
    loop = asyncio.get_running_loop()
    start = loop.time()
    stamps = []
    usage = {}
    payload = dict(model=model, prompt=prompt, max_tokens=max_tokens,
                   ignore_eos=True, temperature=0, stream=True, return_token_ids=True,
                   stream_options={"include_usage": True}, add_special_tokens=False)
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
                n = len(choice.get("token_ids") or [])
                if n:
                    stamps.append(((loop.time() - start) * 1000, n))
    if usage.get("completion_tokens") != max_tokens:
        raise RuntimeError(f"completion_tokens {usage.get('completion_tokens')} != {max_tokens}")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return dict(ttft_ms=stamps[0][0], stamps=stamps, cached_tokens=cached,
                prompt_tokens=usage.get("prompt_tokens"))


def gaps(stamps, skip):
    out = []
    for (t0, _), (t1, n) in zip(stamps[skip:], stamps[skip + 1:]):
        out.append((t1 - t0) / n)
    return out


def fit(xs, ys):
    """Ordinary least squares y = a + b x."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return dict(intercept_ms=my - b * mx, slope_ms=b)


async def profile(args):
    import aiohttp

    url = args.url.rstrip("/")
    rng = random.Random(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("x") as log:
        def write(row):
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()

        write(dict(type="meta", url=url, model=args.model, wall_start=time.time(), args=vars(args)))
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            for _ in range(args.warmup):
                await run_one(session, url, args.model, tokens(rng, 256), 8)
            for length in args.prefill_lengths:
                for fraction in args.cached_fractions:
                    for rep in range(args.repeats):
                        prompt = tokens(rng, length)
                        cached_target = int(length * fraction) // 16 * 16
                        if cached_target:
                            await run_one(session, url, args.model, prompt[:cached_target], 1)
                        result = await run_one(session, url, args.model, prompt, 1)
                        write(dict(type="prefill", length=length, cached_target=cached_target, rep=rep,
                                   ttft_ms=result["ttft_ms"], cached_tokens=result["cached_tokens"]))
                        print(f"prefill L={length} cached={cached_target} ttft={result['ttft_ms']:.1f}ms "
                              f"engine_cached={result['cached_tokens']}", flush=True)
            for context in args.decode_contexts:
                for batch in args.batch_sizes:
                    if batch * (context + args.decode_tokens) > args.kv_token_budget:
                        continue  # would exceed the cache and measure preemption instead
                    for rep in range(args.repeats):
                        results = await asyncio.gather(*(
                            run_one(session, url, args.model, tokens(rng, context), args.decode_tokens)
                            for _ in range(batch)))
                        step = [g for r in results for g in gaps(r["stamps"], args.skip_tokens)]
                        write(dict(type="decode", context=context, batch=batch, rep=rep,
                                   step_ms_median=statistics.median(step),
                                   step_ms_p90=sorted(step)[int(.9 * (len(step) - 1))],
                                   samples=len(step)))
                        print(f"decode ctx={context} B={batch} step={statistics.median(step):.2f}ms", flush=True)
        prefill = [r for r in rows if r["type"] == "prefill"]
        cold = [r for r in prefill if r["cached_target"] == 0]
        decode = [r for r in rows if r["type"] == "decode" and r["context"] == args.decode_contexts[0]]
        summary = dict(
            type="summary",
            prefill_cold_vs_tokens=fit([r["length"] for r in cold], [r["ttft_ms"] for r in cold]),
            prefill_vs_new_tokens=fit([r["length"] - (r["cached_tokens"] or 0) for r in prefill],
                                      [r["ttft_ms"] for r in prefill]),
            decode_step_vs_batch=fit([r["batch"] for r in decode], [r["step_ms_median"] for r in decode]),
            cache_mismatch=sum(r["cached_tokens"] is not None and r["cached_tokens"] < r["cached_target"]
                               for r in prefill),
            missing_cached=sum(r["cached_tokens"] is None for r in prefill))
        try:
            with urllib.request.urlopen(url + "/metrics", timeout=10) as response:
                summary["cache_config"] = [l for l in response.read().decode().splitlines()
                                           if l.startswith("vllm:cache_config_info")]
        except OSError as exc:
            summary["cache_config"] = f"unavailable: {exc}"
        write(summary)
        print(json.dumps(summary, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--output", required=True)
    p.add_argument("--prefill-lengths", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096, 8192])
    p.add_argument("--cached-fractions", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--decode-contexts", type=int, nargs="+", default=[256, 2048])
    p.add_argument("--decode-tokens", type=int, default=128)
    p.add_argument("--kv-token-budget", type=int, default=30000,
                   help="skip decode points whose total context exceeds this many cached tokens")
    p.add_argument("--skip-tokens", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=699)
    asyncio.run(profile(p.parse_args(argv)))


if __name__ == "__main__":
    main()
