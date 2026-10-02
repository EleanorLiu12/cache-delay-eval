"""Count a declared mixed-session structure in retained records, without inference."""

from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import itertools
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PROTOCOL_HASH = hashlib.sha256((OUT / "protocol.md").read_bytes()).hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + "\n")


def read_qwen(path):
    rows = []
    for line_no, line in enumerate(path.open(), 1):
        r = json.loads(line, parse_float=Decimal)
        assert len(r["hash_ids"]) == (r["input_length"] + 15) // 16
        rows.append(dict(id=str(r["chat_id"]), parent=None if r["parent_chat_id"] == -1 else str(r["parent_chat_id"]),
                         time=Decimal(r["timestamp"]), tokens=r["input_length"], turn=r["turn"],
                         label=r["type"], blocks=tuple(r["hash_ids"][:r["input_length"] // 16]), line=line_no))
    return rows


def read_synthetic(path):
    rows = []
    for line_no, line in enumerate(path.open(), 1):
        r = json.loads(line)
        if r["type"] != "request":
            continue
        rows.append(dict(id=r["request_id"], parent=r.get("parent_request_id"),
                         time=Decimal(r["arrival_time_ms"]) / 1000, tokens=r["prompt_tokens"],
                         turn=r["metadata"]["turn_index"] + 1, label="synthetic",
                         blocks=tuple(r["block_hashes"]), line=line_no))
    return rows


def chains(rows):
    by_id = {r["id"]: r for r in rows}
    assert len(by_id) == len(rows)
    roots, groups = {}, defaultdict(list)
    for row in rows:
        trail, visited, key = [], set(), row["id"]
        while key not in roots and key in by_id:
            assert key not in visited, "Cycle in observed lineage"
            visited.add(key)
            trail.append(key)
            parent = by_id[key]["parent"]
            if parent is None:
                break
            key = parent
        root = roots.get(key, key)
        for key in trail:
            roots[key] = root
        groups[root].append(row)
    eligible, excluded = [], []
    for key, group in groups.items():
        group.sort(key=lambda r: (r["turn"], r["time"], r["id"]))
        why = []
        if key not in by_id or group[0]["parent"] is not None or group[0]["turn"] != 1:
            why.append("missing_or_invalid_root")
        if len({r["label"] for r in group}) != 1:
            why.append("mixed_source_labels")
        child_counts = Counter(r["parent"] for r in group if r["parent"] is not None)
        if any(n > 1 for n in child_counts.values()):
            why.append("branching")
        for parent, child in zip(group, group[1:]):
            if child["parent"] != parent["id"] or child["turn"] != parent["turn"] + 1:
                why.append("nonconsecutive_lineage")
            if child["time"] <= parent["time"]:
                why.append("nonincreasing_arrivals")
        if why:
            excluded.append(dict(root=key, labels=sorted({r["label"] for r in group}),
                                 requests=len(group), reasons=sorted(set(why))))
            continue
        # Only the uninterrupted growing-prefix sequence beginning at the root is usable.
        prefix = [group[0]]
        for parent, child in zip(group, group[1:]):
            if child["tokens"] <= parent["tokens"] or child["blocks"][:len(parent["blocks"])] != parent["blocks"]:
                break
            prefix.append(child)
        eligible.append(dict(root=key, label=group[0]["label"], rows=group, prefix=prefix))
    return eligible, excluded


def window_count(rows, eligible, origin, stop, width, offset, long_tokens, depth, witnesses=False):
    start = origin + offset
    n = int((stop - start) // width)
    windows = [dict(index=i, start_s=str(start + i * width), end_s=str(start + (i + 1) * width),
                    requests=0, eligible_roots=0, long_roots=0, short_chains=0, match=False) for i in range(n)]
    long_by_window, short_by_window = defaultdict(list), defaultdict(list)
    covered_requests = 0
    for row in rows:
        i = int((row["time"] - start) // width) if row["time"] >= start else -1
        if 0 <= i < n:
            windows[i]["requests"] += 1
            covered_requests += 1
    for chain in eligible:
        root = chain["rows"][0]
        i = int((root["time"] - start) // width) if root["time"] >= start else -1
        if not 0 <= i < n:
            continue
        windows[i]["eligible_roots"] += 1
        if root["tokens"] >= long_tokens and len(chain["rows"]) <= 3:
            long_by_window[i].append(chain)
        if root["tokens"] <= 2048:
            in_window = [r for r in chain["prefix"] if r["time"] < start + (i + 1) * width]
            if len(in_window) >= depth:
                short_by_window[i].append((chain, in_window))
    for i, window in enumerate(windows):
        longs = sorted(long_by_window[i], key=lambda c: (c["rows"][0]["time"], c["root"]))
        shorts = sorted(short_by_window[i], key=lambda item: (item[0]["rows"][0]["time"], item[0]["root"]))
        window["long_roots"] = len(longs)
        window["short_chains"] = len(shorts)
        if not longs:
            continue
        long_chain = longs[0]
        anchor = long_chain["rows"][0]
        for chain, seq in shorts:
            assert chain["root"] != long_chain["root"]
            later = [(p, c) for p, c in zip(seq, seq[1:])
                     if c["time"] > anchor["time"] and len(p["blocks"]) * 16 * 2 >= c["tokens"]]
            if len(later) < 2:
                continue
            window["match"] = True
            if witnesses:
                window["witness"] = dict(
                    long_root=anchor["id"], long_root_line=anchor["line"], long_root_tokens=anchor["tokens"],
                    long_root_time_s=str(anchor["time"]), long_observed_requests=len(long_chain["rows"]),
                    short_root=seq[0]["id"], short_root_line=seq[0]["line"], short_root_tokens=seq[0]["tokens"],
                    short_sequence=[r["id"] for r in seq], short_sequence_lines=[r["line"] for r in seq],
                    later_continuations=[dict(parent=p["id"], child=c["id"], child_line=c["line"],
                                              time_s=str(c["time"]), retained_parent_tokens=len(p["blocks"])*16,
                                              child_tokens=c["tokens"]) for p, c in later[:2]])
            break
    positives = sum(w["match"] for w in windows)
    result = dict(window_seconds=width, offset_seconds=offset, long_root_min_tokens=long_tokens,
                  short_root_max_tokens=2048, min_short_depth=depth, positive_windows=positives,
                  total_windows=n, percent=positives / n * 100 if n else None,
                  covered_requests=covered_requests, excluded_boundary_requests=len(rows)-covered_requests,
                  nonempty_windows=sum(w["requests"] > 0 for w in windows),
                  eligible_roots=sum(w["eligible_roots"] for w in windows),
                  long_roots=sum(w["long_roots"] for w in windows),
                  short_chains=sum(w["short_chains"] for w in windows))
    return result, windows


def analyze_source(name, path, synthetic=False):
    rows = read_synthetic(path) if synthetic else read_qwen(path)
    eligible, excluded = chains(rows)
    origin = min(r["time"] for r in rows)
    stop = origin + 300 if synthetic else max(r["time"] for r in rows)
    cohorts, sensitivity, evidence = [], [], {}
    for label in sorted({r["label"] for r in rows}):
        cohort_rows = [r for r in rows if r["label"] == label]
        cohort_chains = [c for c in eligible if c["label"] == label]
        primary, windows = window_count(cohort_rows, cohort_chains, origin, stop, 300, 0, 4096, 4, True)
        primary.update(cohort=label, source_requests=len(cohort_rows), eligible_chains=len(cohort_chains),
                       excluded_chains=sum(label in c["labels"] for c in excluded),
                       full_high_hit_high_ttft_pattern_percent=None)
        cohorts.append(primary)
        evidence[label] = windows
        for width, half_shift, threshold, depth in itertools.product((60, 300), (False, True), (4096, 8192, 16384), (4, 8, 30)):
            result, _ = window_count(cohort_rows, cohort_chains, origin, stop, width, width//2 if half_shift else 0,
                                     threshold, depth)
            result["cohort"] = label
            sensitivity.append(result)
    result = dict(source=name, path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  synthetic=synthetic, source_requests=len(rows), observation_start_s=str(origin),
                  observation_stop_s=str(stop), excluded_lineage=excluded, primary=cohorts)
    save(name + "-windows.json", evidence)
    save(name + "-sensitivity.json", sensitivity)
    return result


def main():
    save("run-manifest.json", dict(protocol_sha256=PROTOCOL_HASH, status="started",
                                   meaning="New structural definition; not actual-hit/TTFT prevalence"))
    sources = []
    for name, pattern in (("qwen-a", "qwen-bailian/*/qwen_traceA_blksz_16.jsonl"),
                          ("qwen-b", "qwen-bailian-trace-b/*/qwen_traceB_blksz_16.jsonl")):
        paths = list((ROOT / "data").glob(pattern))
        assert len(paths) == 1
        sources.append(analyze_source(name, paths[0]))
    for name in ("negative", "positive"):
        sources.append(analyze_source("synthetic-" + name, ROOT / f"results/pattern-screen/traces/P512-{name}-seed100.jsonl", True))
    summary = dict(protocol_sha256=PROTOCOL_HASH, sources=sources,
                   unmeasurable={"WildChat": "All 606 user arrivals missing in the fixed 200-conversation sample",
                                 "BurstGPT": "No completed local corpus analysis",
                                 "ServeGen": "No completed local corpus analysis"},
                   full_real_source_pattern_prevalence="Unknown: actual hit and TTFT unavailable")
    save("summary.json", summary)
    save("run-manifest.json", dict(protocol_sha256=PROTOCOL_HASH, status="complete",
                                   analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps([dict(source=s["source"], primary=s["primary"]) for s in sources], indent=2))


if __name__ == "__main__":
    main()
