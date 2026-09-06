"""Aggregate raw calibration JSONL into a tidy condition-level CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def summarize(path: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("type") != "calibration_result":
                raise ValueError(f"line {line_no}: not a calibration_result")
            key = (
                int(row["prompt_tokens_target"]),
                int(row["cached_prefix_tokens_target"]),
                int(row["queue_depth_target"]),
            )
            groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        rows = groups[key]
        ok = [row for row in rows if row.get("status") == "ok"]
        ttft = [float(row["ttft_ms"]) for row in ok]
        e2e = [float(row["e2e_ms"]) for row in ok]
        observed = [
            float(row["queue_depth_observed"])
            for row in ok
            if row.get("queue_depth_observed") is not None
        ]
        hits = [
            float(row["prefix_cache_hit_tokens_batch"])
            for row in ok
            if row.get("prefix_cache_hit_tokens_batch") is not None
        ]
        output.append(
            {
                "prompt_tokens": key[0],
                "cached_prefix_tokens": key[1],
                "queue_depth_target": key[2],
                "trials_total": len(rows),
                "trials_ok": len(ok),
                "trials_error": len(rows) - len(ok),
                "ttft_median_ms": median(ttft) if ttft else math.nan,
                "ttft_p95_ms": percentile(ttft, 0.95),
                "e2e_median_ms": median(e2e) if e2e else math.nan,
                "e2e_p95_ms": percentile(e2e, 0.95),
                "queue_depth_observed_median": median(observed) if observed else math.nan,
                "prefix_cache_hit_tokens_batch_median": median(hits) if hits else math.nan,
            }
        )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = summarize(args.input)
    if not rows:
        raise SystemExit("no result rows found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} condition summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

