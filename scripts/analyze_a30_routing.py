"""Compare paired A30 routing runs and screen LMetric's Equation 2 condition."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_run(path: Path) -> tuple[dict[str, dict], dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    requests = [row for row in rows if row.get("type") == "request"]
    by_id = {row["rid"]: row for row in requests}
    if len(by_id) != len(requests):
        raise ValueError(f"duplicate request IDs in {path}")
    if not rows or rows[-1].get("type") != "run_summary":
        raise ValueError(f"incomplete run: {path}")
    snapshots = [row for row in rows if row.get("type") == "snapshot"]
    summary = dict(rows[-1])
    summary["mean_batch_spread"] = mean([max(row["bs"]) - min(row["bs"])
                                         for row in snapshots])
    summary["max_batch_spread"] = max((max(row["bs"]) - min(row["bs"])
                                        for row in snapshots), default=None)
    return by_id, summary


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def eq2_violations(requests: dict[str, dict], scale: int) -> tuple[set[str], dict]:
    """Use 10 trace-second windows and each dispatch's 512-token prefix coverage."""
    totals: Counter[int] = Counter()
    groups: Counter[tuple[int, str]] = Counter()
    for row in requests.values():
        window = int(row["scheduled_ms"] * scale / 10000)
        totals[window] += 1
        groups[window, row["cls"]] += 1

    flagged: set[str] = set()
    flagged_windows: set[int] = set()
    max_gap = 0.0
    for rid, row in requests.items():
        if row["input_len"] < 512:
            continue
        window = int(row["scheduled_ms"] * scale / 10000)
        class_count = groups[window, row["cls"]]
        if class_count < 5:
            continue
        share = class_count / totals[window]
        coverage = sum(hit >= 32 for hit in row["hits"]) / len(row["hits"])
        if 0 < coverage < 1 and share > coverage:
            flagged.add(rid)
            flagged_windows.add(window)
            max_gap = max(max_gap, share - coverage)
    return flagged, dict(
        ten_second_windows=len(totals), violating_windows=len(flagged_windows),
        violating_requests=len(flagged), max_share_minus_coverage=max_gap,
    )


def compare(root: Path, tag: str, scale: int) -> dict:
    runs = {}
    for policy in ("lmetric", "load", "affinity"):
        runs[policy] = read_run(root / f"{tag}-x{scale}-{policy}-events.jsonl")
    ids = set(runs["lmetric"][0])
    if any(set(requests) != ids for requests, _ in runs.values()):
        raise ValueError(f"request sets differ for {tag} x{scale}")
    flagged, condition = eq2_violations(runs["lmetric"][0], scale)
    lmetric, load = runs["lmetric"][0], runs["load"][0]
    flagged_by_window: dict[int, list[str]] = defaultdict(list)
    for rid in flagged:
        flagged_by_window[int(lmetric[rid]["scheduled_ms"] * scale / 10000)].append(rid)
    window_differences = [statistics.fmean(lmetric[rid]["ttft_ms"] - load[rid]["ttft_ms"]
                                           for rid in rids)
                          for rids in flagged_by_window.values()]
    condition["violating_windows_lmetric_mean_slower"] = sum(d > 0 for d in window_differences)
    condition["median_window_mean_difference_ms"] = (statistics.median(window_differences)
                                                       if window_differences else None)

    def paired(subset: set[str]) -> dict:
        good = [rid for rid in subset if lmetric[rid]["status"] == load[rid]["status"] == "ok"]
        differences = [lmetric[rid]["ttft_ms"] - load[rid]["ttft_ms"] for rid in good]
        return dict(requests=len(good), lmetric_mean_ms=mean([lmetric[rid]["ttft_ms"] for rid in good]),
                    load_mean_ms=mean([load[rid]["ttft_ms"] for rid in good]),
                    mean_paired_difference_ms=mean(differences),
                    median_paired_difference_ms=(statistics.median(differences)
                                                 if differences else None),
                    fraction_lmetric_slower=(sum(d > 0 for d in differences) / len(differences)
                                             if differences else None))

    return dict(tag=tag, time_scale=scale, condition=condition,
                all_requests=paired(ids), violation_requests=paired(flagged),
                other_requests=paired(ids - flagged),
                policies={name: summary for name, (_, summary) in runs.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing routing JSONL runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = [compare(args.root, tag, scale) for scale in (2, 3)
              for tag in ("hot", "typical")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result:
        print(row["tag"], f"x{row['time_scale']}",
              "violating requests", row["condition"]["violating_requests"],
              "overall paired TTFT difference (ms)",
              row["all_requests"]["mean_paired_difference_ms"],
              "violation paired difference (ms)",
              row["violation_requests"]["mean_paired_difference_ms"])


if __name__ == "__main__":
    main()
