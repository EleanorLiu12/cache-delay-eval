"""Find trace patterns with higher actual cache reuse and higher TTFT.

Analyze each cache-on run separately. Cache-off measurements remain a paired
control; their treatment effect does not define a hit/TTFT pattern.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .session_gen import _pearson, _ranks


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
        result.append(dict(request_id=request_id, scheduled_ms=a["scheduled_ms"],
                           session_id=a.get("session_id"), prompt_tokens=a["prompt_tokens"],
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


def hit_ttft_summary(rows):
    """Describe a single run or archetype; constant variables are undefined."""
    lengths = [r["prompt_tokens"] for r in rows]
    hits = [r["cached_token_fraction"] for r in rows]
    ttfts = [r["ttft_on_ms"] for r in rows]
    return dict(
        requests=len(rows),
        pooled_hit_ttft_pearson=correlation(hits, ttfts),
        pooled_hit_ttft_spearman=correlation(_ranks(hits), _ranks(ttfts)),
        prompt_length_partial_pearson=correlation(
            residual(hits, lengths), residual(ttfts, lengths)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
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
    summaries, request_rows = [], []
    pattern_groups = defaultdict(list)
    for (trace, capacity, repeat), conditions in sorted(groups.items()):
        rows = pair(conditions["on"], conditions["off"])
        archetypes = defaultdict(list)
        for r in rows:
            archetypes[r["archetype"] or "unknown"].append(r)
            request_rows.append(dict(r, trace=trace, capacity_blocks=capacity, repeat=repeat))
        summary = dict(
            trace=trace, capacity_blocks=capacity, repeat=repeat,
            **hit_ttft_summary(rows),
            within_archetype={name: hit_ttft_summary(rs)
                              for name, rs in sorted(archetypes.items())},
            paired_control=dict(median_delta_ms=statistics.median(
                r["delta_ttft_ms"] for r in rows)),
        )
        summaries.append(summary)
        pattern_groups[(trace, capacity)].append(summary)
    patterns = []
    for (trace, capacity), runs in sorted(pattern_groups.items()):
        positive = sum(r["pooled_hit_ttft_pearson"] is not None and
                       r["pooled_hit_ttft_pearson"] > 0 for r in runs)
        patterns.append(dict(
            trace=trace, capacity_blocks=capacity, repetitions=len(runs),
            positive_pearson_repetitions=positive,
            positive_pearson_in_all_repetitions=len(runs) >= 2 and positive == len(runs),
        ))
    with args.output.open("x") as stream:
        json.dump(dict(
            objective="Identify trace patterns with higher actual cached-token fraction and higher TTFT.",
            hit_definition="Per-request cached_tokens / prompt_tokens in the cache-on run.",
            patterns=patterns, summaries=summaries, requests=request_rows,
            interpretation="Correlations are computed within each cache-on run. Positive Pearson is a descriptive candidate, not statistical or causal confirmation. Inspect Spearman, archetypes and arrival timelines; prompt-length adjustment does not remove queue or session dependence. Cache-off deltas are supporting controls. TTFT includes transport.",
        ), stream, indent=2)
        stream.write("\n")
    count = sum(p["positive_pearson_in_all_repetitions"] for p in patterns)
    print(f"Analyzed {len(summaries)} cache-on runs with paired controls; "
          f"{count}/{len(patterns)} trace/capacity groups have positive hit/TTFT Pearson in every repetition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
