"""CPU-only prefix-history sensitivity and descriptive mixed-arrival comparisons.

This module deliberately leaves the original Trace A analyzer and results intact.
It measures published input-block structure, never computed or resident KV state.
"""
import argparse
from bisect import bisect_left, insort
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import sys
import time

from .analyze_qwen_trace import (
    BLOCK_SIZE, add_context_flags, attach_lineage, describe, fraction, lcp,
    prefix_metrics, prevalence, quantile, read_records,
)


WINDOWS = (1, 10, 60, 300, None)
SCOPES = ("all", "within", "cross")
COMMON_LONG_TOKENS = 4096
TIME_SLICE_SECONDS = 2400


def window_label(window):
    return "unlimited" if window is None else str(window)


class ActivePrefixes:
    """Exact lexicographic nearest-neighbor index with explicit expiry.

    The maximum common prefix is attained by the predecessor or successor in
    lexicographic order. Removing a component leaves this fact unchanged, so
    the first foreign neighbor on each side suffices for cross-chain queries.
    Witness ties use the smaller ID among those neighbors, not the globally
    smallest ID among all maximizers. A zero-length match has no witness.
    """

    def __init__(self, by_id):
        self.by_id = by_id
        self.times = {identifier: Decimal(str(row["timestamp"])) for identifier, row in by_id.items()}
        self.all = []
        self.within = defaultdict(list)

    def add(self, row):
        key = (row["blocks"], row["chat_id"])
        insort(self.all, key)
        insort(self.within[row["component_id"]], key)

    def remove(self, row):
        key = (row["blocks"], row["chat_id"])
        for index in (self.all, self.within[row["component_id"]]):
            position = bisect_left(index, key)
            if position == len(index) or index[position] != key:
                raise AssertionError("Expired prefix missing from active index")
            index.pop(position)

    def query(self, row, scope):
        index = self.within[row["component_id"]] if scope == "within" else self.all
        position = bisect_left(index, (row["blocks"], -1))
        candidates = []
        for start, step in ((position - 1, -1), (position, 1)):
            j = start
            while 0 <= j < len(index):
                blocks, identifier = index[j]
                if scope != "cross" or self.by_id[identifier]["component_id"] != row["component_id"]:
                    candidates.append((lcp(row["blocks"], blocks), identifier))
                    break
                j += step
        matched, identifier = max(candidates, key=lambda pair: (pair[0], -pair[1]), default=(0, None))
        if matched == 0:
            return dict(prefix_tokens=0, witness_chat_id=None, age_seconds=None)
        return dict(prefix_tokens=matched * BLOCK_SIZE, witness_chat_id=identifier,
                    age_seconds=float(self.times[row["chat_id"]] - self.times[identifier]))


def prefix_recency(rows, windows=WINDOWS):
    """Return exact per-ID metrics for 0 < current_time - prior_time <= window.

    Subtract timestamps for both expiry and witness age. Query every member of
    an equal-time batch before inserting any member. Inputs have already had
    their partial final blocks excluded by the original validated reader.
    """
    ordered = sorted(rows, key=lambda row: (row["timestamp"], row["chat_id"]))
    by_id = {row["chat_id"]: row for row in ordered}
    result = {identifier: {} for identifier in by_id}
    for window in windows:
        if window is not None and window <= 0:
            raise ValueError("History windows must be positive")
        index, active = ActivePrefixes(by_id), deque()
        for timestamp, iterator in groupby(ordered, key=lambda row: row["timestamp"]):
            batch = list(iterator)
            if window is not None:
                current = Decimal(str(timestamp))
                limit = Decimal(str(window))
                while active and current - index.times[active[0]["chat_id"]] > limit:
                    index.remove(active.popleft())
            for row in batch:
                result[row["chat_id"]][window_label(window)] = {
                    scope: index.query(row, scope) for scope in SCOPES
                }
            for row in batch:
                index.add(row)
                if window is not None:
                    active.append(row)
    return result


def add_context_witnesses(records, thresholds):
    """Latest strictly earlier qualifying request in a different component.

    Low-overlap predecessors use their own unlimited-history overlap <25%.
    The comparator for the current request instead splits at 50%.
    """
    ordered = sorted(records, key=lambda row: (row["timestamp"], row["chat_id"]))
    recent = {name: {kind: {} for kind in ("any_long", "low_overlap_long")} for name in thresholds}
    for timestamp, iterator in groupby(ordered, key=lambda row: row["timestamp"]):
        batch = list(iterator)
        for row in batch:
            row["contexts"] = {}
            for name, variants in recent.items():
                row["contexts"][name] = {}
                for kind, owners in variants.items():
                    prior = next((other for component, other in reversed(owners.items())
                                  if component != row["component_id"]), None)
                    row["contexts"][name][kind] = dict(
                        witness_chat_id=prior["chat_id"] if prior else None,
                        age_seconds=float(Decimal(str(timestamp)) - Decimal(str(prior["timestamp"]))) if prior else None)
        for row in batch:
            for name, threshold in thresholds.items():
                if row["input_length"] < threshold:
                    continue
                kinds = ["any_long"] + (["low_overlap_long"] if row["potential_reuse_fraction"] < .25 else [])
                for kind in kinds:
                    owners = recent[name][kind]
                    owners.pop(row["component_id"], None)
                    owners[row["component_id"]] = row


def prompt_bin(length):
    if length <= 512:
        return "<=512"
    if length <= 2048:
        return "513-2048"
    if length <= 8192:
        return "2049-8192"
    return ">8192"


def turn_group(row):
    return "initial" if row["parent_chat_id"] == -1 else "continuation"


def difference_pp(high, low):
    if high["fraction"] is None or low["fraction"] is None:
        return None
    return 100 * (high["fraction"] - low["fraction"])


def mixed_comparison(records, long_definition, threshold, precursor, window):
    """Disjoint request groups; a chain may contain requests from both groups."""
    high = [row for row in records if row["potential_reuse_fraction"] >= .5]
    low = [row for row in records if row["potential_reuse_fraction"] < .5]

    def measure(group):
        event = [row for row in group
                 if (age := row["contexts"][long_definition][precursor]["age_seconds"]) is not None
                 and 0 < age <= window]
        return dict(requests=fraction(len(event), len(group)),
                    chains=fraction(len({row["component_id"] for row in event}),
                                    len({row["component_id"] for row in group})))

    high_result, low_result = measure(high), measure(low)
    return dict(long_definition=long_definition, long_input_at_least=threshold,
                precursor=precursor, window_seconds=window,
                high=high_result, low=low_result, all=measure(records),
                request_difference_pp=difference_pp(high_result["requests"], low_result["requests"]),
                chain_difference_pp=difference_pp(high_result["chains"], low_result["chains"]),
                chains_in_both_groups=len({row["component_id"] for row in high}
                                         & {row["component_id"] for row in low}))


def mixed_grid(records, thresholds):
    return [mixed_comparison(records, name, threshold, kind, window)
            for name, threshold in thresholds.items()
            for kind in ("any_long", "low_overlap_long") for window in (1, 10, 60)]


def stratified_mixed(records, thresholds):
    """Fixed one-way and joint strata; preceding context remains cohort-wide."""
    groups = defaultdict(list)
    for row in records:
        slice_number = math.floor(row["timestamp"] / TIME_SLICE_SECONDS)
        time_label = f"{slice_number * TIME_SLICE_SECONDS}-{(slice_number + 1) * TIME_SLICE_SECONDS}"
        length_label, turn_label = prompt_bin(row["input_length"]), turn_group(row)
        for kind, key in (("time_slice", time_label), ("prompt_length", length_label),
                          ("turn", turn_label), ("joint", f"{time_label}|{length_label}|{turn_label}")):
            groups[kind, key].append(row)
    result = []
    for (kind, key), sample in sorted(groups.items()):
        for comparison in mixed_grid(sample, thresholds):
            comparison["stratum"] = dict(kind=kind, key=key, requests=len(sample),
                                         components=len({row["component_id"] for row in sample}))
            result.append(comparison)
    return result


def summarize_recency(records):
    result = []
    for window in WINDOWS:
        label = window_label(window)
        scopes = {}
        for scope in SCOPES:
            tokens = [row["recency"][label][scope]["prefix_tokens"] for row in records]
            ratios = [count / row["input_length"] for count, row in zip(tokens, records)]
            scopes[scope] = dict(fraction_distribution=describe(ratios),
                                 prefix_token_distribution=describe(tokens),
                                 at_least_50pct=fraction(sum(value >= .5 for value in ratios), len(records)),
                                 any_match=fraction(sum(count > 0 for count in tokens), len(records)))
        result.append(dict(window_seconds=window, window_label=label, scopes=scopes))
    return result


def standardized_mixed(strata, total_requests, minimum_group_size=30):
    """Descriptive common-composition contrast, with explicit overlap support.

    Weight each supported joint cell by its combined high/low request count.
    This controls only the recorded coarse strata and is not a causal estimate.
    """
    grouped = defaultdict(list)
    for row in strata:
        if row["stratum"]["kind"] == "joint":
            grouped[row["long_definition"], row["precursor"], row["window_seconds"]].append(row)
    result = []
    for (definition, precursor, window), cells in grouped.items():
        supported = [r for r in cells if min(r[g]["requests"]["denominator"] for g in ("high", "low")) >= minimum_group_size]
        n = sum(r["stratum"]["requests"] for r in supported)
        rates = {g: sum(r["stratum"]["requests"] * r[g]["requests"]["fraction"] for r in supported) / n if n else None for g in ("high", "low")}
        result.append(dict(long_definition=definition, precursor=precursor, window_seconds=window,
                           minimum_requests_per_group_per_cell=minimum_group_size,
                           supported_cells=len(supported), total_cells=len(cells),
                           coverage=fraction(n, total_requests), excluded_requests=total_requests-n,
                           standardized_high_rate=rates["high"], standardized_low_rate=rates["low"],
                           standardized_difference_pp=100*(rates["high"]-rates["low"]) if n else None))
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compare_retained(records, result, path, cohort_name):
    """Check all retained A request metrics and the complete old prevalence grid."""
    directory = Path(path)
    retained = {}
    with (directory / f"requests-{cohort_name}.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            retained[row["chat_id"]] = row
    if set(retained) != {row["chat_id"] for row in records}:
        raise AssertionError("Cohort differs from retained analysis")
    # Compare every old field; new evidence only adds fields.
    comparisons = 0
    for row in records:
        for field, value in retained[row["chat_id"]].items():
            if row[field] != value:
                raise AssertionError(f"Retained metric differs: {row['chat_id']} {field}")
            comparisons += 1
    retained_analysis = json.loads((directory / "analysis.json").read_text())
    if result["legacy_prevalence_grid"] != retained_analysis["cohorts"][cohort_name]["prevalence_grid"]:
        raise AssertionError("Legacy prevalence grid differs from retained result")
    return dict(requests_checked=len(retained), request_field_comparisons=comparisons,
                legacy_prevalence_grid_equal=True, source_directory=str(directory),
                retained_analysis_sha256=sha256_file(directory / "analysis.json"),
                retained_requests_sha256=sha256_file(directory / f"requests-{cohort_name}.jsonl"))


def format_fraction(value):
    if value["fraction"] is None:
        return f"{value['numerator']}/{value['denominator']} (undefined)"
    return f"{value['numerator']:,}/{value['denominator']:,} ({100 * value['fraction']:.2f}%)"


def write_report(result, directory):
    lines = [f"# {result['dataset']}: temporal CPU analysis", "",
             f"Cohort: {result['cohort']['rule']}. Requests: {result['cohort']['requests']:,}; "
             f"observed parent components: {result['cohort']['components']:,}.", "",
             "## Prefix-history sensitivity", "",
             "Maximum full-block prefix overlap uses strictly earlier inputs from this cohort. "
             "A finite window includes a predecessor exactly at its age boundary. "
             "The ratio divides matching full-block tokens by the original full input length. "
             "The partial final block is excluded. Within and cross-component overlap are "
             "alternative maxima and must not be added. All request denominators include initial requests.", "",
             "| History window | Any component: overlap >=50% | Within component | Cross component |",
             "| --- | ---: | ---: | ---: |"]
    for row in result["recency"]:
        lines.append("| " + " | ".join([row["window_label"]] +
                     [format_fraction(row["scopes"][scope]["at_least_50pct"]) for scope in SCOPES]) + " |")
    lines += ["", "## Mixed-arrival comparison", "",
              "Current requests form disjoint groups: unlimited-history overlap >=50% (high) and <50% (low). "
              "A predecessor must be from another observed component, arrive strictly earlier, and have age <= "
              "the stated window. The low-overlap-long variant additionally requires the predecessor's own "
              "unlimited-history overlap <25%. Long means input >= the cohort p90 or the fixed common "
              "4,096-token threshold, declared before computing temporal results. The original descriptive "
              "25/50/75% grid is retained in the aggregate JSON.", "",
              "| Long definition | Predecessor | Window (s) | High requests with context | Low requests with context | High minus low (pp) |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in result["mixed_arrival"]:
        delta = row["request_difference_pp"]
        lines.append(f"| {row['long_definition']} ({row['long_input_at_least']:,}) | {row['precursor']} | "
                     f"{row['window_seconds']} | {format_fraction(row['high']['requests'])} | "
                     f"{format_fraction(row['low']['requests'])} | " +
                     (f"{delta:+.2f}" if delta is not None else "undefined") + " |")
    lines += ["", "### Chain incidence", "",
              "Chain incidence means at least one qualifying request in that group and stratum, divided by "
              "chains containing any request in that group and stratum. Request groups are disjoint, but a "
              "chain can contain both high and low requests. Chain denominators therefore overlap; incidence "
              "is also sensitive to the number of observed requests per chain.", "",
              "| Long definition | Predecessor | Window (s) | High chain incidence | Low chain incidence | High minus low (pp) |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in result["mixed_arrival"]:
        delta = row["chain_difference_pp"]
        lines.append(f"| {row['long_definition']} | {row['precursor']} | {row['window_seconds']} | "
                     f"{format_fraction(row['high']['chains'])} | {format_fraction(row['low']['chains'])} | " +
                     (f"{delta:+.2f}" if delta is not None else "undefined") + " |")
    lines += ["", "## Strata, interpretation, and evidence", "",
              "The aggregate JSON contains the same numerator/denominator comparisons within fixed "
              "40-minute slices anchored at zero, prompt bins <=512 / 513-2048 / 2049-8192 / >8192 tokens, "
              "explicit initial requests (parent=-1) versus continuations, and all observed joint strata. "
              "History and predecessor context remain cohort-wide across stratum boundaries; thresholds "
              "are fixed for the full cohort. Empty groups have undefined rates. Small strata are retained "
              "for audit and should not be treated as independent evidence. Chains can span time slices.", "",
              "This is a history-window sensitivity analysis, not a TTL-cache or finite-memory simulation. "
              "Earlier inputs may be incomplete, evicted, or on a different replica. The release supplies "
              "no actual cache-hit, TTFT, or completion observations. These descriptive differences do not "
              "identify a shared queue, bottleneck, causation, or an association beyond confounding. No "
              "randomized reference, significance test, or independent-request uncertainty interval was used.", "",
              "[Aggregate results](analysis.json) include methods, hashes, comparisons, strata, and validation. "
              "[Request evidence](requests.jsonl.gz) retains source line, original IDs, per-window matching "
              "prefix lengths with witness request IDs and ages, and latest qualifying predecessor IDs. "
              "A zero prefix has a null witness. A positive prefix witness is one exact maximizing input, "
              "not necessarily the latest maximizer. The original raw hashes allow every match to be checked.", ""]
    (directory / "report.md").write_text("\n".join(lines))


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cohort", choices=("observed_root_text", "observed_root_api", "all_records"), required=True)
    parser.add_argument("--compare-retained", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = json.loads(args.manifest.read_text())
    digest = sha256_file(args.input)
    if digest != manifest["expected_trace_sha256"] or args.input.stat().st_size != manifest["expected_trace_bytes"]:
        raise ValueError("Original input does not match source manifest")
    rows, issues, invalid = read_records(args.input)
    if invalid or issues.get("invalid_records", 0):
        raise ValueError("Temporal analysis requires a separately audited valid schema")
    _, groups, lineage = attach_lineage(rows)
    if args.cohort.startswith("observed_root_"):
        kind = args.cohort.removeprefix("observed_root_")
        cohort = [row for group in groups.values()
                  if group[0]["root_observed"] and all(row["type"] == kind for row in group)
                  for row in group]
        rule = f"all requests in observed-root parent components containing only type={kind}"
    else:
        cohort = rows
        rule = "all released valid records, retaining the observed type labels and parent components"
    if not cohort:
        raise ValueError("Empty cohort")
    print(f"{args.dataset}: computing {len(cohort):,} requests in {len({r['component_id'] for r in cohort}):,} components", flush=True)
    records = prefix_metrics(cohort)
    lengths = [row["input_length"] for row in records]
    thresholds = dict(cohort_p90=quantile(lengths, .9), absolute_4096=COMMON_LONG_TOKENS)
    add_context_flags(records, thresholds["cohort_p90"])
    recency = prefix_recency(cohort)
    print("Exact recency computation complete; building comparisons and retained-output checks", flush=True)
    for row in records:
        row["recency"] = recency[row["chat_id"]]
        for scope, old_field in (("all", "historical_prefix_tokens"),
                                 ("within", "within_component_prefix_tokens"),
                                 ("cross", "cross_component_prefix_tokens")):
            if row["recency"]["unlimited"][scope]["prefix_tokens"] != row[old_field]:
                raise AssertionError(f"Unlimited-history implementation disagreement: {row['chat_id']} {scope}")
    add_context_witnesses(records, thresholds)
    precursor_rounding_differences = 0
    for row in records:
        for new, old in (("any_long", "long"), ("low_overlap_long", "low_reuse_long")):
            fresh, legacy = row["contexts"]["cohort_p90"][new]["age_seconds"], row[f"seconds_since_other_component_{old}"]
            if fresh != legacy:
                if fresh is None or legacy is None or not math.isclose(fresh, legacy, abs_tol=1e-9):
                    raise AssertionError("Legacy precursor flag disagreement")
                precursor_rounding_differences += 1
    result = dict(dataset=args.dataset, analysis="Observed input prefix and arrival structure; CPU only",
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  source=dict(input=str(args.input), input_sha256=digest, input_bytes=args.input.stat().st_size,
                              manifest=str(args.manifest), manifest_sha256=sha256_file(args.manifest),
                              revision=manifest["revision"]),
                  provenance=dict(command=sys.argv, python_version=sys.version,
                                  analyzer_sha256=sha256_file(__file__),
                                  imported_analyzer_sha256=sha256_file(Path(__file__).with_name("analyze_qwen_trace.py"))),
                  cohort=dict(name=args.cohort, rule=rule, requests=len(records),
                              components=len({row["component_id"] for row in records}),
                              type_counts=dict(Counter(row["type"] for row in records)),
                              excluded_requests=len(rows) - len(records)),
                  quality=dict(source_valid_records=len(rows), schema_issues=issues, lineage=lineage),
                  methods=dict(block_size=BLOCK_SIZE, final_partial_block="excluded",
                               history="strictly earlier cohort inputs; 0 < age <= window; unlimited has no age cap",
                               time_arithmetic="Decimal(str(source timestamp)) subtraction for window boundaries and reported ages; original legacy fields retain historical float arithmetic",
                               fraction_denominator="original input token count, including partial-block tokens",
                               scopes="all / within same observed parent component / another component; maxima not additive",
                               high_low_groups="current unlimited-history overlap >=0.5 versus <0.5; disjoint requests",
                               long_thresholds=thresholds, common_absolute_threshold_predeclared=True,
                               low_overlap_precursor="predecessor unlimited-history overlap <0.25",
                               chain_incidence="event chains / chains with any request in the same group and stratum; groups can share chains",
                               time_slices_seconds=TIME_SLICE_SECONDS, time_slice_origin_seconds=0,
                               prompt_bins=["<=512", "513-2048", "2049-8192", ">8192"],
                               initial_definition="parent_chat_id == -1; other requests are continuations",
                               strata="time, prompt length, turn, and joint; context/history remain full-cohort",
                               quantiles="nearest rank; ceil(p*n)-1",
                               significance="none; dependencies and confounding prevent independent-request interpretation",
                               completion_cache_and_latency="unobserved; no inference or cache simulation"),
                  recency=summarize_recency(records), mixed_arrival=mixed_grid(records, thresholds),
                  strata=stratified_mixed(records, thresholds),
                  legacy_prevalence_grid=[prevalence(records, reuse, window, quantile(lengths, .5), low)
                                          for low in (False, True) for reuse in (.25, .5, .75) for window in (1, 10, 60)],
                  validation=dict(unlimited_equals_original_implementation_requests=len(records),
                                  precursor_age_matches_original_within_1e_minus_9_requests=len(records),
                                  precursor_age_float_representation_differences=precursor_rounding_differences))
    if args.compare_retained:
        result["validation"]["retained_output"] = compare_retained(records, result, args.compare_retained, args.cohort)
    result["standardized_mixed_arrival"] = standardized_mixed(result["strata"], len(records))
    result["methods"]["standardization"] = "Descriptive high-minus-low contrast weighted to the combined request distribution of joint time/length/turn strata with >=30 requests per group. Coverage is explicit; coarse adjustment cannot eliminate confounding."
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with gzip.GzipFile(filename=str(args.output_dir / "requests.jsonl.gz"), mode="xb", mtime=0) as stream:
        for row in records:
            stream.write((json.dumps(row, separators=(",", ":")) + "\n").encode())
    result["provenance"]["request_evidence_sha256"] = sha256_file(args.output_dir / "requests.jsonl.gz")
    result["analysis_wall_seconds"] = time.perf_counter() - started
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(result, args.output_dir)
    print(f"Temporal CPU results saved to {args.output_dir} in {result['analysis_wall_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
