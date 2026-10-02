"""CPU characterization of original Qwen-Bailian records; no cache simulation."""
import argparse
from bisect import bisect_left, insort
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import statistics
import sys
import time


BLOCK_SIZE = 16
FIELDS = {"chat_id", "parent_chat_id", "timestamp", "input_length", "output_length",
          "type", "turn", "hash_ids"}


def quantile(values, probability):
    """Nearest-rank quantile: sorted_values[ceil(p*n)-1], clipped at zero."""
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(probability * len(values)) - 1)]


def describe(values):
    values = list(values)
    if not values:
        return {"count": 0}
    return dict(count=len(values), mean=statistics.mean(values), minimum=min(values),
                p25=quantile(values, .25), p50=quantile(values, .5),
                p75=quantile(values, .75), p90=quantile(values, .9),
                p95=quantile(values, .95), p99=quantile(values, .99), maximum=max(values))


def fraction(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                fraction=numerator / denominator if denominator else None)


def lcp(a, b):
    """Match only a leading sequence; repeated nonprefix blocks do not count."""
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def read_records(path):
    with Path(path).open() as stream:
        return _read_records(stream)


def _read_records(stream):
    rows, issues = [], Counter()
    invalid_examples = []
    seen = set()
    previous_time = None
    for line_no, line in enumerate(stream, 1):
        if not line.strip():
            issues["blank_lines"] += 1
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != FIELDS:
                raise ValueError("unexpected fields")
            for name in ("chat_id", "parent_chat_id", "input_length", "output_length", "turn"):
                if type(row[name]) is not int:
                    raise ValueError(f"invalid integer: {name}")
            if row["chat_id"] < 0 or row["parent_chat_id"] < -1 or row["input_length"] < 1:
                raise ValueError("invalid identifier or input length")
            if row["output_length"] < 0 or row["turn"] < 1:
                raise ValueError("invalid output length or turn")
            if type(row["timestamp"]) not in (int, float) or not math.isfinite(row["timestamp"]) or row["timestamp"] < 0:
                raise ValueError("invalid timestamp")
            if not isinstance(row["type"], str) or not row["type"]:
                raise ValueError("invalid request type")
            if not isinstance(row["hash_ids"], list) or any(type(x) is not int or x < 0 for x in row["hash_ids"]):
                raise ValueError("invalid block hashes")
            if len(row["hash_ids"]) != math.ceil(row["input_length"] / BLOCK_SIZE):
                raise ValueError("hash count differs from ceiling(input_length/16)")
        except (ValueError, TypeError) as exc:
            issues["invalid_records"] += 1
            if len(invalid_examples) < 10:
                invalid_examples.append(dict(line=line_no, error=str(exc)))
            continue
        if row["chat_id"] in seen:
            raise ValueError(f"line {line_no}: duplicate chat_id; cannot join lineage unambiguously")
        seen.add(row["chat_id"])
        if previous_time is not None:
            issues["out_of_order_adjacent_arrivals"] += row["timestamp"] < previous_time
            issues["equal_adjacent_arrivals"] += row["timestamp"] == previous_time
        previous_time = row["timestamp"]
        row["source_line"] = line_no
        row["blocks"] = tuple(row["hash_ids"][:row["input_length"] // BLOCK_SIZE])
        rows.append(row)
    if not rows:
        raise ValueError("No valid input records")
    return rows, dict(issues), invalid_examples


def attach_lineage(rows):
    by_id = {r["chat_id"]: r for r in rows}
    children, groups = defaultdict(list), defaultdict(list)
    audit = Counter()
    roots = {}
    for row in rows:
        chain, visiting = [], set()
        current = row["chat_id"]
        while current not in roots:
            if current in visiting:
                raise ValueError(f"Cycle in parent graph at {current}")
            visiting.add(current)
            node = by_id[current]
            chain.append(current)
            parent = node["parent_chat_id"]
            if parent == -1:
                root, observed = current, True
                break
            if parent not in by_id:
                root, observed = parent, False
                break
            current = parent
        else:
            root, observed = roots[current]
        for item in chain:
            roots[item] = (root, observed)
    for row in rows:
        row["component_id"], row["root_observed"] = roots[row["chat_id"]]
        groups[row["component_id"]].append(row)
        parent = by_id.get(row["parent_chat_id"])
        if row["parent_chat_id"] == -1:
            audit["roots"] += 1
            audit["root_turn_not_one"] += row["turn"] != 1
        elif parent is None:
            audit["missing_parent_edges"] += 1
        else:
            children[parent["chat_id"]].append(row["chat_id"])
            audit["observed_edges"] += 1
            audit["parent_not_earlier_in_file"] += parent["source_line"] >= row["source_line"]
            audit["parent_not_strictly_earlier_in_time"] += parent["timestamp"] >= row["timestamp"]
            audit["nonconsecutive_turn_edges"] += row["turn"] != parent["turn"] + 1
    audit["branching_parents"] = sum(len(v) > 1 for v in children.values())
    audit["mixed_type_components"] = sum(len({r["type"] for r in group}) > 1 for group in groups.values())
    audit["components_without_observed_root"] = sum(not group[0]["root_observed"] for group in groups.values())
    return by_id, groups, dict(audit)


def prefix_metrics(rows):
    """Exact LCP to earlier inputs, excluding simultaneous arrivals and outputs.

    In lexicographic order, a query's maximum LCP is attained by a neighbor.
    For cross-component LCP, skip neighbors belonging to the query component.
    This uses observed history with unlimited retention, not resident KV state.
    """
    ordered = sorted(rows, key=lambda r: (r["timestamp"], r["chat_id"]))
    by_id = {r["chat_id"]: r for r in ordered}
    index, histories, output = [], defaultdict(list), []
    for timestamp, batch_iter in groupby(ordered, key=lambda r: r["timestamp"]):
        batch = list(batch_iter)
        for row in batch:
            blocks = row["blocks"]
            position = bisect_left(index, (blocks, -1))
            candidates = index[max(0, position - 1):position + 1]
            any_match = max((lcp(blocks, key) for key, _ in candidates), default=0)
            foreign = []
            for start, step in ((position - 1, -1), (position, 1)):
                j = start
                while 0 <= j < len(index):
                    key, identifier = index[j]
                    if by_id[identifier]["component_id"] != row["component_id"]:
                        foreign.append(lcp(blocks, key))
                        break
                    j += step
            within = max((lcp(blocks, prior) for prior in histories[row["component_id"]]), default=0)
            parent = by_id.get(row["parent_chat_id"])
            valid_parent = parent is not None and parent["timestamp"] < timestamp
            parent_match = lcp(blocks, parent["blocks"]) if valid_parent else None
            record = {k: row[k] for k in ("chat_id", "parent_chat_id", "component_id", "root_observed",
                                          "source_line", "timestamp", "type", "turn", "input_length", "output_length")}
            record.update(full_blocks=len(blocks), historical_prefix_tokens=any_match * BLOCK_SIZE,
                          within_component_prefix_tokens=within * BLOCK_SIZE,
                          cross_component_prefix_tokens=max(foreign, default=0) * BLOCK_SIZE,
                          parent_prefix_tokens=parent_match * BLOCK_SIZE if parent_match is not None else None,
                          potential_reuse_fraction=any_match * BLOCK_SIZE / row["input_length"],
                          within_component_reuse_fraction=within * BLOCK_SIZE / row["input_length"],
                          cross_component_reuse_fraction=max(foreign, default=0) * BLOCK_SIZE / row["input_length"])
            if valid_parent:
                record.update(parent_arrival_gap_s=timestamp - parent["timestamp"],
                              input_growth_tokens=row["input_length"] - parent["input_length"],
                              input_minus_parent_input_and_output=row["input_length"] - parent["input_length"] - parent["output_length"],
                              parent_full_blocks_retained=parent_match == len(parent["blocks"]))
            output.append(record)
        # Equal-time records cannot serve as each other's earlier history.
        for row in batch:
            insort(index, (row["blocks"], row["chat_id"]))
            histories[row["component_id"]].append(row["blocks"])
    return output


def recent_other_component(last_by_component, component):
    # Values are per-component latest arrivals. Only the two newest can matter.
    return next((timestamp for owner, timestamp in reversed(last_by_component.items()) if owner != component), None)


def add_context_flags(records, long_threshold):
    recent_long, recent_low_reuse_long = {}, {}
    for _, batch_iter in groupby(records, key=lambda r: r["timestamp"]):
        batch = list(batch_iter)
        for row in batch:
            for label, recent in (("long", recent_long), ("low_reuse_long", recent_low_reuse_long)):
                timestamp = recent_other_component(recent, row["component_id"])
                row[f"seconds_since_other_component_{label}"] = row["timestamp"] - timestamp if timestamp is not None else None
        for row in batch:
            if row["input_length"] >= long_threshold:
                for recent in [recent_long] + ([recent_low_reuse_long] if row["potential_reuse_fraction"] < .25 else []):
                    recent.pop(row["component_id"], None)
                    recent[row["component_id"]] = row["timestamp"]


def prevalence(records, reuse_threshold, window_s, short_threshold, low_reuse_long=False):
    group_ids = {r["component_id"] for r in records}
    label = "low_reuse_long" if low_reuse_long else "long"
    context = [r for r in records if r[f"seconds_since_other_component_{label}"] is not None
               and 0 < r[f"seconds_since_other_component_{label}"] <= window_s]
    high = [r for r in records if r["potential_reuse_fraction"] >= reuse_threshold]
    high_ids = {r["chat_id"] for r in high}
    event = [r for r in context if r["chat_id"] in high_ids]
    short_high = [r for r in high if r["input_length"] <= short_threshold]
    short_event = [r for r in event if r["input_length"] <= short_threshold]
    continuation_high = [r for r in high if r["parent_chat_id"] != -1]
    continuation_event = [r for r in event if r["parent_chat_id"] != -1]
    within_high = [r for r in records if r["within_component_reuse_fraction"] >= reuse_threshold]
    within_ids = {r["chat_id"] for r in within_high}
    within_event = [r for r in context if r["chat_id"] in within_ids]
    return dict(reuse_threshold=reuse_threshold, window_seconds=window_s,
                preceding_long_requires_reuse_below_25pct=low_reuse_long,
                long_context_all_requests=fraction(len(context), len(records)),
                high_reuse=fraction(len(high), len(records)),
                event_requests=fraction(len(event), len(records)),
                event_given_high_reuse=fraction(len(event), len(high)),
                event_components=fraction(len({r["component_id"] for r in event}), len(group_ids)),
                short_high_reuse_requests=fraction(len(short_high), len(records)),
                short_event_requests=fraction(len(short_event), len(records)),
                short_event_given_short_high_reuse=fraction(len(short_event), len(short_high)),
                continuation_high_reuse_requests=fraction(len(continuation_high), len(records)),
                continuation_event_requests=fraction(len(continuation_event), len(records)),
                event_given_high_reuse_continuation=fraction(len(continuation_event), len(continuation_high)),
                within_component_high_reuse_requests=fraction(len(within_high), len(records)),
                within_component_event_requests=fraction(len(within_event), len(records)),
                event_given_within_component_high_reuse=fraction(len(within_event), len(within_high)))


def arrival_bins(records, width, end_s):
    bins = [dict(start_s=i * width, duration_s=min(width, end_s - i * width), requests=0,
                 roots=0, input_tokens=0, output_tokens=0, high_reuse_requests=0)
            for i in range(math.ceil(end_s / width))]
    for row in records:
        b = bins[int(row["timestamp"] // width)]
        b["requests"] += 1
        b["roots"] += row["parent_chat_id"] == -1
        b["input_tokens"] += row["input_length"]
        b["output_tokens"] += row["output_length"]
        b["high_reuse_requests"] += row["potential_reuse_fraction"] >= .5
    return bins


def summarize_cohort(rows, end_s):
    records = prefix_metrics(rows)
    lengths = [r["input_length"] for r in records]
    long_threshold, short_threshold = quantile(lengths, .9), quantile(lengths, .5)
    add_context_flags(records, long_threshold)
    components = defaultdict(list)
    for row in records:
        components[row["component_id"]].append(row)
    edges = [r for r in records if "input_growth_tokens" in r]
    prefix = {}
    for field in ("potential_reuse_fraction", "within_component_reuse_fraction", "cross_component_reuse_fraction"):
        prefix[field] = dict(distribution=describe(r[field] for r in records),
                             thresholds={str(p): fraction(sum(r[field] >= p for r in records), len(records))
                                         for p in (.25, .5, .75)},
                             any_match=fraction(sum(r[field] > 0 for r in records), len(records)))
    leading_prefixes = []
    for depth in (1, 4, 16, 32, 64):
        counts = Counter(r["blocks"][:depth] for r in rows if len(r["blocks"]) >= depth)
        leading_prefixes.append(dict(prefix_tokens=depth * BLOCK_SIZE,
                                     eligible_requests=sum(counts.values()), distinct_prefixes=len(counts),
                                     most_common_prefix_request_count=max(counts.values(), default=0),
                                     requests_in_prefix_groups_with_multiple_records=sum(n for n in counts.values() if n > 1)))
    rates, bins_by_width = {}, {}
    for width in (1, 10, 60):
        bins = arrival_bins(records, width, end_s)
        complete_counts = [b["requests"] for b in bins if b["duration_s"] == width]
        mean = statistics.mean(complete_counts) if complete_counts else 0
        rates[str(width)] = dict(full_bin_request_counts=describe(complete_counts),
                                 variance_to_mean=statistics.pvariance(complete_counts) / mean if mean else None,
                                 peak_to_mean=max(complete_counts) / mean if mean else None,
                                 empty_full_bins=sum(x == 0 for x in complete_counts))
        bins_by_width[str(width)] = bins
    interarrivals = [b["timestamp"] - a["timestamp"] for a, b in zip(records, records[1:])]
    mean_gap = statistics.mean(interarrivals) if interarrivals else 0
    grid = [prevalence(records, p, window, short_threshold, low)
            for low in (False, True) for p in (.25, .5, .75) for window in (1, 10, 60)]
    windows = []
    for i in range(3):
        start, end = end_s * i / 3, end_s * (i + 1) / 3
        sample = [r for r in records if start <= r["timestamp"] < end]
        windows.append(dict(start_s=start, end_s=end, requests=len(sample),
                            historical_reuse=describe(r["potential_reuse_fraction"] for r in sample),
                            event=prevalence(sample, .5, 10, short_threshold)))
    overflow = [r for r in records if r["input_length"] + r["output_length"] > 32768]
    excluded_components = {r["component_id"] for r in overflow}
    summary = dict(requests=len(records), components=len(components),
                   types=dict(Counter(r["type"] for r in records)),
                   singleton_components=sum(len(g) == 1 for g in components.values()),
                   multi_request_components=sum(len(g) > 1 for g in components.values()),
                   component_size=describe(len(g) for g in components.values()),
                   component_max_turn=describe(max(r["turn"] for r in g) for g in components.values()),
                   component_observed_arrival_span_s=describe(g[-1]["timestamp"] - g[0]["timestamp"] for g in components.values()),
                   input_tokens=describe(lengths), output_tokens=describe(r["output_length"] for r in records),
                   parent_edges=len(edges), parent_full_blocks_retained=fraction(sum(r["parent_full_blocks_retained"] for r in edges), len(edges)),
                   growth=fraction(sum(r["input_growth_tokens"] > 0 for r in edges), len(edges)),
                   contraction=fraction(sum(r["input_growth_tokens"] < 0 for r in edges), len(edges)),
                   parent_arrival_gap_s=describe(r["parent_arrival_gap_s"] for r in edges),
                   input_growth_tokens=describe(r["input_growth_tokens"] for r in edges),
                   input_minus_parent_input_and_output=describe(r["input_minus_parent_input_and_output"] for r in edges),
                   prefix_opportunity=prefix, leading_prefix_groups=leading_prefixes,
                   thresholds=dict(long_input_at_least=long_threshold, short_input_at_most=short_threshold),
                   prevalence_grid=grid, temporal_thirds=windows,
                   arrivals=dict(interval_start_s=0, interval_end_s=end_s, requests_per_second=len(records) / end_s,
                                 interarrival_s=describe(interarrivals),
                                 interarrival_coefficient_of_variation=statistics.pstdev(interarrivals) / mean_gap if mean_gap else None,
                                 bins=rates),
                   pilot_context_limit_audit=dict(limit_tokens=32768, exceeding_requests=len(overflow),
                                                  affected_components=len(excluded_components),
                                                  requests_in_affected_components=sum(r["component_id"] in excluded_components for r in records)))
    return summary, records, bins_by_width


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if digest != manifest["expected_trace_sha256"] or args.input.stat().st_size != manifest["expected_trace_bytes"]:
        raise ValueError("Input does not match source manifest")
    rows, schema_issues, invalid_examples = read_records(args.input)
    _, groups, lineage = attach_lineage(rows)
    text = [r for group in groups.values() if group[0]["root_observed"] and all(r["type"] == "text" for r in group) for r in group]
    # All records determine the same observation interval for every cohort.
    end_s = math.floor(max(r["timestamp"] for r in rows)) + 1
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(analysis="Observed production request structure; no measured cache hits or latency",
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  input=str(args.input), input_sha256=digest,
                  source_manifest=str(args.manifest), source_revision=manifest["revision"],
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  python_version=sys.version, command=sys.argv,
                  methods=dict(block_size=BLOCK_SIZE, final_partial_block="excluded",
                               history="only strictly earlier arrivals; no generated-output hashes or completion assumptions",
                               historical_retention="unbounded observed input history; structural opportunity only",
                               session_unit="connected parent component; no observed terminal status",
                               quantiles="nearest rank, ceil(p*n)-1",
                               observation_interval="[0, floor(max_timestamp)+1); final partial bins excluded from count dispersion",
                               cooccurrence="current reuse >= threshold; earlier long input >= cohort p90 in a different component within window",
                               short_variant="also requires current input <= cohort median",
                               temporal_thirds="fixed thirds; retain pre-window observed history; components may cross boundaries",
                               population="entire pinned Trace A; no random sampling or synthetic records"),
                  quality=dict(valid_records=len(rows), schema_issues=schema_issues,
                               invalid_examples=invalid_examples, lineage=lineage,
                               partial_final_block_records=sum(r["input_length"] % BLOCK_SIZE != 0 for r in rows),
                               zero_output_records=sum(r["output_length"] == 0 for r in rows)), cohorts={})
    for name, cohort in [("all_types", rows), ("observed_root_text", text)]:
        summary, records, bins = summarize_cohort(cohort, end_s)
        result["cohorts"][name] = summary
        with (args.output_dir / f"requests-{name}.jsonl").open("x") as stream:
            for row in records:
                stream.write(json.dumps(row) + "\n")
        (args.output_dir / f"arrival-bins-{name}.json").write_text(json.dumps(bins, indent=2) + "\n")
        print(name, json.dumps(dict(requests=summary["requests"], components=summary["components"],
                                    thresholds=summary["thresholds"])), flush=True)
    result["analysis_wall_seconds"] = time.perf_counter() - started
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"CPU analysis saved to {args.output_dir}; no model or GPU invoked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
