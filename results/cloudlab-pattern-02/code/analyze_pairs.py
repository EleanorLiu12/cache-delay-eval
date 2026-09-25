"""Strict pairing and descriptive analysis of live caching on/off runs."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .session_gen import _pearson


def load_run(path):
    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not values or values[0].get("type") != "run_meta" or values[-1].get("type") != "run_summary":
        raise ValueError(f"{path}: incomplete run")
    meta, summary = values[0], values[-1]
    requests = [r for r in values if r.get("type") == "request"]
    by_id = {r["request_id"]: r for r in requests}
    if len(by_id) != len(requests) or len(requests) != meta["requests"]:
        raise ValueError(f"{path}: duplicate or missing requests")
    if any(summary[k] for k in ("errors", "late_dispatches", "missing_cache_usage")):
        raise ValueError(f"{path}: invalid run summary {summary}")
    for r in requests:
        if r["status"] != "ok" or r["cached_tokens"] is None or r["dispatch_lag_ms"] > meta["max_dispatch_lag_ms"]:
            raise ValueError(f"{path}: invalid request {r['request_id']}")
        if meta["condition"] == "off" and r["cached_tokens"] != 0:
            raise ValueError(f"{path}: cache-off run reports cached tokens")
    return meta, by_id


def pair(on, off):
    on_meta, on_rows = on
    off_meta, off_rows = off
    for key in ("trace_sha256", "model", "arrival_scale", "capacity_blocks", "repeat", "engine_config"):
        if on_meta[key] != off_meta[key]:
            raise ValueError(f"paired run mismatch: {key}")
    if on_rows.keys() != off_rows.keys():
        raise ValueError("paired request IDs differ")
    result = []
    for request_id, a in on_rows.items():
        b = off_rows[request_id]
        for key in ("scheduled_ms", "prompt_tokens", "output_tokens"):
            if a[key] != b[key]:
                raise ValueError(f"{request_id}: paired workload mismatch: {key}")
        result.append(dict(request_id=request_id, prompt_tokens=a["prompt_tokens"],
                           cached_token_fraction=a["cached_tokens"] / a["prompt_tokens"],
                           ttft_on_ms=a["ttft_ms"], ttft_off_ms=b["ttft_ms"],
                           delta_ttft_ms=b["ttft_ms"] - a["ttft_ms"],
                           archetype=a.get("archetype")))
    return result


def correlation(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return _pearson(xs, ys)


def residual(values, lengths):
    mean_x, mean_y = statistics.mean(lengths), statistics.mean(values)
    variance = sum((x - mean_x)**2 for x in lengths)
    slope = sum((x - mean_x)*(y - mean_y) for x, y in zip(lengths, values)) / variance if variance else 0
    return [y - mean_y - slope * (x - mean_x) for x, y in zip(lengths, values)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilon-ms", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.epsilon_ms < 0:
        parser.error("epsilon must be nonnegative")
    groups = defaultdict(dict)
    plan = json.loads((args.suite / "plan.json").read_text())
    expected = {(r["trace"], r["capacity_blocks"], r["repeat"], r["condition"]) for r in plan["runs"]}
    observed = set()
    for path in sorted(args.suite.glob("run-*/requests.jsonl")):
        data = load_run(path)
        m = data[0]
        key = (m["trace"], m["capacity_blocks"], m["repeat"])
        full_key = key + (m["condition"],)
        if full_key in observed:
            raise ValueError("duplicate run")
        observed.add(full_key)
        groups[key][m["condition"]] = data
    if observed != expected:
        raise ValueError(f"suite incomplete or unexpected runs: missing={expected - observed}, extra={observed - expected}")
    summaries, paired_rows = [], []
    repeat_deltas = defaultdict(list)
    for (trace, capacity, repeat), conditions in sorted(groups.items()):
        rows = pair(conditions["on"], conditions["off"])
        lengths = [r["prompt_tokens"] for r in rows]
        hits = [r["cached_token_fraction"] for r in rows]
        ttfts = [r["ttft_on_ms"] for r in rows]
        summaries.append(dict(trace=trace, capacity_blocks=capacity, repeat=repeat,
                              requests=len(rows),
                              fraction_delta_negative=sum(r["delta_ttft_ms"] < 0 for r in rows) / len(rows),
                              fraction_slower_beyond_epsilon=sum(r["delta_ttft_ms"] < -args.epsilon_ms for r in rows) / len(rows),
                              median_delta_ms=statistics.median(r["delta_ttft_ms"] for r in rows),
                              pooled_hit_ttft_pearson=correlation(hits, ttfts),
                              prompt_length_partial_pearson=correlation(residual(hits, lengths), residual(ttfts, lengths))))
        for r in rows:
            paired_rows.append(dict(r, trace=trace, capacity_blocks=capacity, repeat=repeat))
            repeat_deltas[(trace, capacity, r["request_id"])].append(r["delta_ttft_ms"])
    repeated = [dict(trace=t, capacity_blocks=c, request_id=i, repetitions=len(ds),
                     median_delta_ms=statistics.median(ds))
                for (t, c, i), ds in repeat_deltas.items()
                if len(ds) >= 2 and all(d < -args.epsilon_ms for d in ds)]
    with args.output.open("x") as stream:
        json.dump(dict(epsilon_ms=args.epsilon_ms, summaries=summaries,
                       repeatedly_slower_requests=repeated, pairs=paired_rows,
                       interpretation="Descriptive pilot only. Repeated negative deltas flag candidates; they do not establish scheduler causality. TTFT includes transport. No Y1-prime without scheduler instrumentation."), stream, indent=2)
        stream.write("\n")
    print(f"Analyzed {len(summaries)} complete pairs; {len(repeated)} requests slower beyond epsilon in every repetition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
