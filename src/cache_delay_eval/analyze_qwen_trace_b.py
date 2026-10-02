"""Audit the complete pinned Trace B without assuming request types or topology."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

from .analyze_qwen_trace import (
    BLOCK_SIZE, FIELDS, attach_lineage, describe, fraction, read_records, summarize_cohort,
)


def topology_audit(rows, by_id, groups, lineage):
    """Count edges, branches and path depth separately from declared turn numbers."""
    children = defaultdict(list)
    for row in rows:
        if row["parent_chat_id"] in by_id:
            children[row["parent_chat_id"]].append(row["chat_id"])
    depth = {}
    for row in rows:
        path, current = [], row["chat_id"]
        while current not in depth:
            path.append(current)
            parent = by_id[current]["parent_chat_id"]
            if parent not in by_id:
                base = 0
                break
            current = parent
        else:
            base = depth[current]
        for identifier in reversed(path):
            base += 1
            depth[identifier] = base
    for row in rows:
        row["observed_path_depth"] = depth[row["chat_id"]]
        row["observed_children"] = len(children[row["chat_id"]])
    keys = ("roots", "root_turn_not_one", "missing_parent_edges", "observed_edges",
            "parent_not_earlier_in_file", "parent_not_strictly_earlier_in_time",
            "nonconsecutive_turn_edges", "branching_parents", "mixed_type_components",
            "components_without_observed_root")
    audit = {key: lineage.get(key, 0) for key in keys}
    audit.update(
        components=len(groups), singleton_components=sum(len(g) == 1 for g in groups.values()),
        branching_components=len({by_id[p]["component_id"] for p, c in children.items() if len(c) > 1}),
        maximum_children=max((len(c) for c in children.values()), default=0),
        declared_turn_counts=dict(sorted(Counter(r["turn"] for r in rows).items())),
        observed_request_path_depth=describe(depth.values()),
        component_maximum_path_depth=describe(max(depth[r["chat_id"]] for r in g) for g in groups.values()),
        observed_children_distribution=dict(sorted(Counter(len(children[r["chat_id"]]) for r in rows).items())),
        branch_examples=[dict(chat_id=p, component_id=by_id[p]["component_id"], child_ids=sorted(c))
                         for p, c in sorted(children.items()) if len(c) > 1][:20],
        depth_definition="1 for a node whose parent is absent or -1; one plus observed parent's depth otherwise",
        cycles=0,
    )
    return audit


def cohort_turn_summary(records):
    """Make root/continuation denominators explicit, including an empty group."""
    result = {}
    for label, selected in (
        ("initial", [r for r in records if r["parent_chat_id"] == -1]),
        ("continuation", [r for r in records if r["parent_chat_id"] != -1]),
    ):
        result[label] = dict(
            requests=len(selected), components=len({r["component_id"] for r in selected}),
            input_tokens=describe(r["input_length"] for r in selected),
            output_tokens=describe(r["output_length"] for r in selected),
            historical_overlap_at_least_50pct=fraction(sum(r["potential_reuse_fraction"] >= .5 for r in selected), len(selected)),
            within_component_overlap_at_least_50pct=fraction(sum(r["within_component_reuse_fraction"] >= .5 for r in selected), len(selected)),
            cross_component_overlap_at_least_50pct=fraction(sum(r["cross_component_reuse_fraction"] >= .5 for r in selected), len(selected)),
        )
    return result


def make_cohorts(rows, groups):
    """Discover exact type labels; never silently interpret API requests as text."""
    cohorts = {"all_types": rows}
    for kind in sorted({r["type"] for r in rows}):
        if not kind.replace("_", "").isalnum():
            raise ValueError(f"Request type requires an explicit safe output name: {kind!r}")
        selected = [r for group in groups.values()
                    if group[0]["root_observed"] and all(r["type"] == kind for r in group) for r in group]
        if selected:
            cohorts[f"observed_root_{kind}"] = selected
    return cohorts


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    digest = hashlib.file_digest(args.input.open("rb"), "sha256").hexdigest()
    if digest != manifest["expected_trace_sha256"] or args.input.stat().st_size != manifest["expected_trace_bytes"]:
        raise ValueError("Input differs from publisher LFS object in source manifest")
    if not manifest["trace_verified_against_lfs_pointer"]:
        raise ValueError("Source acquisition did not verify the pinned publisher LFS pointer")
    rows, schema_issues, invalid_examples = read_records(args.input)
    by_id, groups, lineage = attach_lineage(rows)
    topology = topology_audit(rows, by_id, groups, lineage)
    timestamps = Counter(r["timestamp"] for r in rows)
    end_s = math.floor(max(timestamps)) + 1
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(
        analysis="Complete Trace B input structure; no measured cache hits or latency",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        input=str(args.input), input_sha256=digest, source_manifest=str(args.manifest),
        source_revision=manifest["revision"], python_version=sys.version, command=sys.argv,
        methods=dict(
            block_size=BLOCK_SIZE, final_partial_block="excluded",
            history="cohort-local inputs at strictly earlier arrivals; simultaneous inputs excluded",
            historical_retention="unlimited observed input history; no residency or completion assumptions",
            session_unit="connected observed parent component; singleton roots need not be independent user sessions",
            depth="observed parent-path node count, reported separately from source turn",
            quantiles="nearest rank, ceil(p*n)-1",
            observation_interval="[0, floor(max_timestamp)+1); partial end bins excluded from dispersion",
            population="entire pinned Trace B; no sampling; cohorts kept separate",
            cohort_selection="all records, plus observed-root components uniform in each discovered exact type label",
            failures="zero outputs audited; no error/status field, so successful completion cannot be established",
            cooccurrence="legacy descriptive grid retained for comparability; improved disjoint comparisons are separate temporal results",
        ),
        quality=dict(
            valid_records=len(rows), schema_issues=schema_issues, invalid_examples=invalid_examples,
            fields=sorted(FIELDS), actual_request_types=dict(Counter(r["type"] for r in rows)),
            unique_ids=len(by_id), duplicate_ids=0, minimum_id=min(by_id), maximum_id=max(by_id),
            lineage=topology, partial_final_block_records=sum(r["input_length"] % BLOCK_SIZE != 0 for r in rows),
            full_hash_count_consistent_records=len(rows),
            zero_output_records=sum(r["output_length"] == 0 for r in rows),
            observed_timestamp_minimum=min(timestamps), observed_timestamp_maximum=max(timestamps),
            distinct_timestamps=len(timestamps), tied_timestamp_groups=sum(n > 1 for n in timestamps.values()),
            requests_at_tied_timestamps=sum(n for n in timestamps.values() if n > 1),
            maximum_timestamp_multiplicity=max(timestamps.values()),
            source_has_completion_or_failure_status=False,
        ), cohorts={},
    )
    for name, cohort in make_cohorts(rows, groups).items():
        summary, records, bins = summarize_cohort(cohort, end_s)
        for record in records:
            source = by_id[record["chat_id"]]
            record.update(observed_path_depth=source["observed_path_depth"], observed_children=source["observed_children"])
        summary.update(
            included_requests=fraction(len(cohort), len(rows)),
            excluded_requests=len(rows) - len(cohort),
            initial_vs_continuation=cohort_turn_summary(records),
            observed_path_depth=describe(r["observed_path_depth"] for r in records),
        )
        result["cohorts"][name] = summary
        with (args.output_dir / f"requests-{name}.jsonl").open("x") as stream:
            for row in records:
                stream.write(json.dumps(row) + "\n")
        write_json(args.output_dir / f"arrival-bins-{name}.json", bins)
        print(name, json.dumps(dict(requests=len(records), components=summary["components"], thresholds=summary["thresholds"])), flush=True)
    result["analysis_wall_seconds"] = time.perf_counter() - started
    write_json(args.output_dir / "analysis.json", result)
    shutil.copyfile(args.manifest, args.output_dir / "source-manifest.json")
    print(f"CPU structural analysis saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
