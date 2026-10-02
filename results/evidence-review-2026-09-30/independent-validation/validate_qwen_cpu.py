"""Independent raw-data checks for the Qwen CPU analyses (no analyzer imports).

The production analyzers use ordered prefix indexes. This checker scans every
eligible raw candidate for a deterministic request sample and reconstructs
lineage and structural summaries directly from the publisher's JSONL bytes.
It also recounts the complete temporal evidence, and checks a pre-run file
fingerprint baseline. Sampling limits are stated in validation.json.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def check_equal(actual, expected, label):
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise AssertionError(f"{label}: observed {actual!r}, independently expected {expected!r}")
    elif actual != expected:
        raise AssertionError(f"{label}: observed {actual!r}, independently expected {expected!r}")


def describe(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    result = {"count": len(ordered), "mean": statistics.mean(ordered),
              "minimum": ordered[0], "maximum": ordered[-1]}
    for percentile in (25, 50, 75, 90, 95, 99):
        result[f"p{percentile}"] = ordered[max(0, math.ceil(percentile * len(ordered) / 100) - 1)]
    return result


def check_description(actual, values, label):
    expected = describe(values)
    for key, value in expected.items():
        check_equal(actual[key], value, f"{label}.{key}")


def frac(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "fraction": numerator / denominator if denominator else None}


def check_fraction(actual, numerator, denominator, label):
    for key, value in frac(numerator, denominator).items():
        check_equal(actual[key], value, f"{label}.{key}")


def read_raw(path):
    rows = list(lines(path))
    expected_fields = {"chat_id", "parent_chat_id", "timestamp", "input_length", "output_length", "type", "turn", "hash_ids"}
    by_id = {}
    children = Counter()
    for source_line, row in enumerate(rows, 1):
        check_equal(set(row), expected_fields, f"raw line {source_line} fields")
        assert row["chat_id"] not in by_id, "Duplicate raw request ID"
        for key in ("chat_id", "parent_chat_id", "input_length", "output_length", "turn"):
            assert type(row[key]) is int, (source_line, key)
        assert row["chat_id"] >= 0 and row["parent_chat_id"] >= -1
        assert row["input_length"] > 0 and row["output_length"] >= 0 and row["turn"] >= 1
        assert math.isfinite(row["timestamp"]) and row["timestamp"] >= 0
        assert type(row["type"]) is str and row["type"]
        assert all(type(x) is int and x >= 0 for x in row["hash_ids"])
        check_equal(len(row["hash_ids"]), (row["input_length"] + 15) // 16, f"raw {source_line} hash count")
        row["source_line"] = source_line
        row["arrival_exact"] = Decimal(str(row["timestamp"]))
        row["blocks"] = tuple(row.pop("hash_ids")[:row["input_length"] // 16])
        by_id[row["chat_id"]] = row
        if row["parent_chat_id"] != -1:
            children[row["parent_chat_id"]] += 1
    groups = defaultdict(list)
    for row in rows:
        node = row
        visited = set()
        depth = 1
        while node["parent_chat_id"] in by_id:
            assert node["chat_id"] not in visited, "Cycle in raw parent graph"
            visited.add(node["chat_id"])
            node = by_id[node["parent_chat_id"]]
            depth += 1
        row["root_observed"] = node["parent_chat_id"] == -1
        row["component_id"] = node["chat_id"] if row["root_observed"] else node["parent_chat_id"]
        row["observed_depth"] = depth
        groups[row["component_id"]].append(row)
    edges = [(row, by_id[row["parent_chat_id"]]) for row in rows if row["parent_chat_id"] in by_id]
    roots = [row for row in rows if row["parent_chat_id"] == -1]
    audit = {
        "valid_records": len(rows), "types": dict(Counter(row["type"] for row in rows)),
        "schema_issues": {"out_of_order_adjacent_arrivals": sum(b["timestamp"] < a["timestamp"] for a, b in zip(rows, rows[1:])),
                          "equal_adjacent_arrivals": sum(b["timestamp"] == a["timestamp"] for a, b in zip(rows, rows[1:]))},
        "lineage": {"roots": len(roots), "root_turn_not_one": sum(row["turn"] != 1 for row in roots),
                    "observed_edges": len(edges), "missing_parent_edges": len(rows) - len(edges) - len(roots),
                    "parent_not_earlier_in_file": sum(p["source_line"] >= r["source_line"] for r, p in edges),
                    "parent_not_strictly_earlier_in_time": sum(p["timestamp"] >= r["timestamp"] for r, p in edges),
                    "nonconsecutive_turn_edges": sum(r["turn"] != p["turn"] + 1 for r, p in edges),
                    "branching_parents": sum(count > 1 and identifier in by_id for identifier, count in children.items()),
                    "mixed_type_components": sum(len({r["type"] for r in group}) > 1 for group in groups.values()),
                    "components_without_observed_root": sum(not group[0]["root_observed"] for group in groups.values())},
        "partial_final_block_records": sum(row["input_length"] % 16 != 0 for row in rows),
        "zero_output_records": sum(row["output_length"] == 0 for row in rows),
        "observed_depth": describe(row["observed_depth"] for row in rows),
    }
    return rows, by_id, groups, audit


def cohort_rows(name, rows, groups):
    if name == "all_types":
        return rows
    prefix = "observed_root_"
    assert name.startswith(prefix), name
    request_type = name[len(prefix):]
    return [r for group in groups.values() if group[0]["root_observed"] and all(r["type"] == request_type for r in group) for r in group]


def verify_structural(raw, groups, audit, directory):
    analysis = json.loads((directory / "analysis.json").read_text())
    for key in ("valid_records", "partial_final_block_records", "zero_output_records"):
        check_equal(analysis["quality"][key], audit[key], f"quality.{key}")
    for key in ("schema_issues", "lineage"):
        for name, value in audit[key].items():
            check_equal(analysis["quality"][key].get(name, 0), value, f"quality.{key}.{name}")
    checked = {}
    for cohort, summary in analysis["cohorts"].items():
        rows = cohort_rows(cohort, raw, groups)
        by_id = {r["chat_id"]: r for r in rows}
        components = defaultdict(list)
        for row in rows:
            components[row["component_id"]].append(row)
        checks = {"requests": len(rows), "components": len(components),
                  "types": dict(Counter(r["type"] for r in rows)),
                  "singleton_components": sum(len(g) == 1 for g in components.values()),
                  "multi_request_components": sum(len(g) > 1 for g in components.values())}
        for key, expected in checks.items():
            check_equal(summary[key], expected, f"{cohort}.{key}")
        for key, values in [
            ("input_tokens", [r["input_length"] for r in rows]),
            ("output_tokens", [r["output_length"] for r in rows]),
            ("component_size", [len(g) for g in components.values()]),
            ("component_max_turn", [max(r["turn"] for r in g) for g in components.values()]),
            ("component_observed_arrival_span_s", [max(r["timestamp"] for r in g) - min(r["timestamp"] for r in g) for g in components.values()]),
        ]:
            check_description(summary[key], values, f"{cohort}.{key}")
        seen = set()
        scalars = {field: [] for field in ("potential_reuse_fraction", "within_component_reuse_fraction", "cross_component_reuse_fraction")}
        for evidence in lines(directory / f"requests-{cohort}.jsonl"):
            identifier = evidence["chat_id"]
            assert identifier not in seen, "Duplicate evidence request ID"
            seen.add(identifier)
            row = by_id[identifier]
            for key in ("parent_chat_id", "timestamp", "input_length", "output_length", "type", "turn", "component_id", "root_observed", "source_line"):
                check_equal(evidence[key], row[key], f"{cohort}.{identifier}.{key}")
            check_equal(evidence["full_blocks"], len(row["blocks"]), f"{cohort}.{identifier}.full_blocks")
            parent = by_id.get(row["parent_chat_id"])
            if parent and parent["timestamp"] < row["timestamp"]:
                for key, expected in {
                    "parent_arrival_gap_s": row["timestamp"] - parent["timestamp"],
                    "input_growth_tokens": row["input_length"] - parent["input_length"],
                    "input_minus_parent_input_and_output": row["input_length"] - parent["input_length"] - parent["output_length"],
                }.items():
                    check_equal(evidence[key], expected, f"{cohort}.{identifier}.{key}")
            for field in scalars:
                scalars[field].append(evidence[field])
        check_equal(seen, set(by_id), f"{cohort} evidence coverage")
        for field, values in scalars.items():
            prefix = summary["prefix_opportunity"][field]
            check_description(prefix["distribution"], values, f"{cohort}.{field}")
            for threshold in (.25, .5, .75):
                check_fraction(prefix["thresholds"][str(threshold)], sum(v >= threshold for v in values), len(values), f"{cohort}.{field}.{threshold}")
            check_fraction(prefix["any_match"], sum(v > 0 for v in values), len(values), f"{cohort}.{field}.any_match")
        checked[cohort] = checks
    return checked


def common_tokens(left, right):
    """Count a leading run, comparing only independently sliced full blocks."""
    for index in range(min(len(left), len(right))):
        if left[index] != right[index]:
            return index * 16
    return min(len(left), len(right)) * 16


def exhaustive_prefix(row, candidates):
    result = {window: {scope: 0 for scope in ("all", "within", "cross")} for window in ("1", "10", "60", "300", "unlimited")}
    for prior in candidates:
        age = float(row["arrival_exact"] - prior["arrival_exact"])
        if age <= 0:
            continue
        tokens = common_tokens(row["blocks"], prior["blocks"])
        if not tokens:
            continue
        scope = "within" if row["component_id"] == prior["component_id"] else "cross"
        for window in result:
            if window == "unlimited" or age <= int(window):
                result[window]["all"] = max(result[window]["all"], tokens)
                result[window][scope] = max(result[window][scope], tokens)
    return result


def choose_sample(rows, evidence_by_id, count):
    """Fixed hash ordering, with coverage of boundaries, ties and key groups."""
    ordered = sorted(rows, key=lambda r: hashlib.sha256(f"qwen-independent-validation-v1:{r['chat_id']}".encode()).digest())
    ties = Counter(r["timestamp"] for r in rows)
    categories = {
        "equal_arrival_time": lambda r: ties[r["timestamp"]] > 1,
        "partial_final_block": lambda r: r["input_length"] % 16 != 0,
        "high_overlap": lambda r: evidence_by_id[r["chat_id"]]["historical_prefix_tokens"] / r["input_length"] >= .5,
        "low_overlap": lambda r: evidence_by_id[r["chat_id"]]["historical_prefix_tokens"] / r["input_length"] < .5,
        "initial": lambda r: r["parent_chat_id"] == -1,
        "continuation": lambda r: r["parent_chat_id"] != -1,
        "sub_block_input": lambda r: r["input_length"] < 16,
    }
    selected = {}
    for label, row in [
        ("earliest", min(rows, key=lambda r: r["timestamp"])),
        ("latest", max(rows, key=lambda r: r["timestamp"])),
        ("shortest", min(rows, key=lambda r: r["input_length"])),
        ("longest", max(rows, key=lambda r: r["input_length"])),
    ]:
        selected.setdefault(row["chat_id"], []).append(label)
    for label, predicate in categories.items():
        for row in [r for r in ordered if predicate(r)][:10]:
            selected.setdefault(row["chat_id"], []).append(label)
    for row in ordered:
        if len(selected) >= count:
            break
        selected.setdefault(row["chat_id"], []).append("hash_sample")
    return {identifier: labels for identifier, labels in list(selected.items())[:count]}


def verify_temporal(raw, directory, sample_count):
    analysis = json.loads((directory / "analysis.json").read_text())
    groups = defaultdict(list)
    for row in raw:
        groups[row["component_id"]].append(row)
    rows = cohort_rows(analysis["cohort"]["name"], raw, groups)
    by_id = {r["chat_id"]: r for r in rows}
    evidence = list(lines(directory / "requests.jsonl.gz"))
    metrics = {r["chat_id"]: r for r in evidence}
    check_equal(len(metrics), len(evidence), "unique temporal evidence")
    check_equal(set(metrics), set(by_id), "temporal cohort coverage")
    thresholds = analysis["methods"]["long_thresholds"]
    lengths = sorted(r["input_length"] for r in rows)
    check_equal(thresholds["cohort_p90"], lengths[math.ceil(.9*len(lengths))-1], "p90 threshold")
    check_equal(thresholds["absolute_4096"], 4096, "common absolute threshold")
    witness_checks = 0
    for item in evidence:
        r = by_id[item["chat_id"]]
        for field in ("source_line", "parent_chat_id", "component_id", "timestamp", "input_length", "type"):
            check_equal(item[field], r[field], f"temporal metadata {r['chat_id']} {field}")
        for window, scopes in item["recency"].items():
            for scope, match in scopes.items():
                tokens = match["prefix_tokens"]
                if tokens == 0:
                    check_equal(match["witness_chat_id"], None, "zero prefix witness")
                    continue
                p = by_id[match["witness_chat_id"]]
                age = float(r["arrival_exact"] - p["arrival_exact"])
                assert age > 0 and (window == "unlimited" or age <= int(window))
                if scope != "all":
                    assert (scope == "within") == (r["component_id"] == p["component_id"])
                check_equal(common_tokens(r["blocks"], p["blocks"]), tokens, "prefix witness length")
                check_equal(match["age_seconds"], age, "prefix witness age")
                witness_checks += 1
    for aggregate in analysis["recency"]:
        label = aggregate["window_label"]
        for scope, summary in aggregate["scopes"].items():
            tokens = [r["recency"][label][scope]["prefix_tokens"] for r in evidence]
            ratios = [t/r["input_length"] for t,r in zip(tokens,evidence)]
            check_description(summary["fraction_distribution"], ratios, f"{label}.{scope}.ratios")
            check_description(summary["prefix_token_distribution"], tokens, f"{label}.{scope}.tokens")
            check_fraction(summary["at_least_50pct"], sum(v>=.5 for v in ratios),len(rows),"recency high prevalence")
            check_fraction(summary["any_match"], sum(v>0 for v in tokens),len(rows),"recency any prevalence")

    def check_mixed(items, result):
        def measure(selected):
            event = [r for r in selected if (age := r["contexts"][result["long_definition"]][result["precursor"]]["age_seconds"]) is not None and 0 < age <= result["window_seconds"]]
            return frac(len(event),len(selected)), frac(len({r['component_id'] for r in event}),len({r['component_id'] for r in selected}))
        subgroups = {"high":[r for r in items if 2*r["historical_prefix_tokens"]>=r["input_length"]],
                     "low":[r for r in items if 2*r["historical_prefix_tokens"]<r["input_length"]], "all":items}
        for name, selected in subgroups.items():
            req, chain = measure(selected)
            check_equal(result[name]["requests"],req,"mixed request fraction")
            check_equal(result[name]["chains"],chain,"mixed chain fraction")
        for unit,field in (("requests","request_difference_pp"),("chains","chain_difference_pp")):
            h,l = [result[g][unit]["fraction"] for g in ("high","low")]
            delta = None if h is None or l is None else 100*(h-l)
            check_equal(result[field],delta,"mixed delta")

    strata = defaultdict(list)
    for r in evidence:
        start = int(r["timestamp"]//2400)*2400
        t = f"{start}-{start+2400}"
        p = "<=512" if r["input_length"]<=512 else "513-2048" if r["input_length"]<=2048 else "2049-8192" if r["input_length"]<=8192 else ">8192"
        u = "initial" if r["parent_chat_id"] == -1 else "continuation"
        for k,v in (("time_slice",t),("prompt_length",p),("turn",u),("joint",f"{t}|{p}|{u}")):
            strata[k,v].append(r)
    for result in analysis["mixed_arrival"]:
        check_mixed(evidence,result)
    for result in analysis["strata"]:
        key = result["stratum"]["kind"], result["stratum"]["key"]
        check_equal(len(strata[key]), result["stratum"]["requests"], "stratum denominator")
        check_mixed(strata[key],result)
    for result in analysis.get("standardized_mixed_arrival",[]):
        cells=[r for r in analysis["strata"] if r["stratum"]["kind"]=="joint" and
               all(r[k]==result[k] for k in ("long_definition","precursor","window_seconds")) and
               all(r[g]["requests"]["denominator"]>=30 for g in ("high","low"))]
        n=sum(r["stratum"]["requests"] for r in cells)
        check_fraction(result["coverage"],n,len(rows),"standardization coverage")
        rates={g:sum(r["stratum"]["requests"]*r[g]["requests"]["fraction"] for r in cells)/n if n else None for g in ("high","low")}
        for g in rates:
            check_equal(result[f"standardized_{g}_rate"],rates[g],"standardized rate")
        check_equal(result["standardized_difference_pp"],None if not n else 100*(rates["high"]-rates["low"]),"standardized difference")

    selected = choose_sample(rows,metrics,sample_count)
    samples = []
    for identifier, labels in selected.items():
        r, m = by_id[identifier], metrics[identifier]
        exact = exhaustive_prefix(r, rows)
        for window, scopes in exact.items():
            for scope, tokens in scopes.items():
                check_equal(m["recency"][window][scope]["prefix_tokens"],tokens,f"exhaustive {identifier} {window} {scope}")
        for definition, threshold in thresholds.items():
            for precursor in ("any_long","low_overlap_long"):
                candidates = [p for p in rows if p["timestamp"]<r["timestamp"] and p["component_id"]!=r["component_id"] and p["input_length"]>=threshold and (precursor=="any_long" or 4*metrics[p["chat_id"]]["historical_prefix_tokens"]<p["input_length"])]
                expected_time = max((p["timestamp"] for p in candidates),default=None)
                expected_age = None if expected_time is None else float(r["arrival_exact"]-Decimal(str(expected_time)))
                ctx = m["contexts"][definition][precursor]
                check_equal(ctx["age_seconds"],expected_age,f"exhaustive context {identifier}")
                if expected_time is not None:
                    assert ctx["witness_chat_id"] in {p["chat_id"] for p in candidates if p["timestamp"]==expected_time}
        samples.append(dict(chat_id=identifier, source_line=r["source_line"], selection_reasons=labels,
                            exhaustive_prefix_and_context_equal=True))
    return dict(requests_checked=len(rows), positive_prefix_witness_checks=witness_checks,
                aggregate_and_stratum_comparisons_checked=len(analysis["mixed_arrival"])+len(analysis["strata"]),
                exhaustive_queries=len(samples), exhaustive_request_evidence=samples,
                all_checks_passed=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, help="Label,raw JSONL,structural directory,temporal directory (comma separated)")
    parser.add_argument("--historical-baseline", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(args.historical_baseline.read_text())
    for path, expected in baseline["files"].items():
        check_equal(digest(path), expected, f"historical preservation: {path}")
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "validator_sha256": digest(__file__),
        "methods": {
            "independence": "No imports from production analyzers. Raw parent traversal, raw scalar distributions and exhaustive candidate prefix scans.",
            "prefix_sampling": "Up to 80 deterministic requests per temporal cohort, covering equal timestamps, partial blocks, high/low overlap, initial/continuation, time/length boundaries; exhaustive raw candidates per sampled query.",
            "limits": "Exhaustive prefix correctness is checked for the deterministic sample, not every request. Complete evidence counts and structural metadata are checked for every request.",
        },
        "historical_preservation": {"files_checked": len(baseline["files"]), "all_unchanged": True},
        "traces": {},
    }
    for spec in args.trace:
        label, raw_path, structural_dir, temporal_dir = spec.split(",")
        print(f"Checking {label} raw schema and structural evidence", flush=True)
        rows, _, groups, audit = read_raw(raw_path)
        trace = {"input": raw_path, "input_sha256": digest(raw_path), "raw_audit": audit,
                 "structural": verify_structural(rows, groups, audit, Path(structural_dir))}
        if temporal_dir:
            trace["temporal"] = verify_temporal(rows, Path(temporal_dir), args.sample_count)
        result["traces"][label] = trace
    for path, expected in baseline["files"].items():
        check_equal(digest(path), expected, f"historical preservation after checks: {path}")
    result["passed"] = True
    (args.output_dir / "historical-baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    (args.output_dir / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"passed": True, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
