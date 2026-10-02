"""Independently enumerate window witnesses from the original Qwen releases."""

from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
summary = json.loads((OUT / "summary.json").read_text())
assert hashlib.sha256((OUT / "protocol.md").read_bytes()).hexdigest() == summary["protocol_sha256"]
checked = 0
witnesses_checked = 0

for source in summary["sources"]:
    if source["synthetic"]:
        continue
    path = ROOT / source["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    records = [json.loads(line, parse_float=Decimal) for line in path.open()]
    by_id = {r["chat_id"]: r for r in records}
    groups = defaultdict(list)
    for row in records:
        root = row
        visited = set()
        while root["parent_chat_id"] != -1:
            assert root["chat_id"] not in visited
            visited.add(root["chat_id"])
            root = by_id[root["parent_chat_id"]]
        groups[root["chat_id"]].append(row)
    typed = defaultdict(list)
    for group in groups.values():
        group.sort(key=lambda r: r["turn"])
        assert group[0]["turn"] == 1
        assert len({r["type"] for r in group}) == 1
        for i, row in enumerate(group):
            assert row["turn"] == i + 1
            if i:
                assert row["parent_chat_id"] == group[i-1]["chat_id"]
                assert row["timestamp"] > group[i-1]["timestamp"]
        typed[group[0]["type"]].append(group)
    all_results = json.loads((OUT / (source["source"] + "-sensitivity.json")).read_text())
    window_evidence = json.loads((OUT / (source["source"] + "-windows.json")).read_text())
    for config in all_results:
        width, offset = config["window_seconds"], config["offset_seconds"]
        start = min(r["timestamp"] for r in records) + offset
        stop = max(r["timestamp"] for r in records)
        n = 0
        while start + (n + 1) * width <= stop:
            n += 1
        assert n == config["total_windows"]
        candidates = [[] for _ in range(n)]
        for group in typed[config["cohort"]]:
            root_time = group[0]["timestamp"]
            if start <= root_time < start + n * width:
                idx = int((root_time-start) // width)
                candidates[idx].append(group)
        detected = []
        for i, groups_here in enumerate(candidates):
            end = start + (i + 1) * width
            longs = [g for g in groups_here if len(g) <= 3 and g[0]["input_length"] >= config["long_root_min_tokens"]]
            found = False
            for group in groups_here:
                if group[0]["input_length"] > 2048:
                    continue
                valid = [group[0]]
                for parent, child in zip(group, group[1:]):
                    if child["timestamp"] >= end:
                        break
                    blocks = parent["input_length"] // 16
                    if child["input_length"] <= parent["input_length"] or parent["hash_ids"][:blocks] != child["hash_ids"][:blocks]:
                        break
                    valid.append(child)
                if len(valid) < config["min_short_depth"]:
                    continue
                # Enumerate all potential long anchors, independently of the main script's earliest-anchor shortcut.
                for long in longs:
                    qualifying = []
                    for parent, child in zip(valid, valid[1:]):
                        retained = parent["input_length"] // 16 * 16
                        if long[0]["timestamp"] < child["timestamp"] and Decimal(retained) / child["input_length"] >= Decimal("0.5"):
                            qualifying.append(child)
                    if len(qualifying) >= 2:
                        found = True
                        break
                if found:
                    break
            detected.append(found)
        assert sum(detected) == config["positive_windows"], config
        checked += 1
        if (width, offset, config["long_root_min_tokens"], config["min_short_depth"]) == (300, 0, 4096, 4):
            expected = window_evidence[config["cohort"]]
            assert detected == [w["match"] for w in expected]
            for window in expected:
                if not window["match"]:
                    continue
                w = window["witness"]
                assert by_id[int(w["long_root"])]["input_length"] == w["long_root_tokens"]
                for item in w["later_continuations"]:
                    child, parent = by_id[int(item["child"])], by_id[int(item["parent"])]
                    assert child["parent_chat_id"] == parent["chat_id"]
                    assert child["input_length"] == item["child_tokens"]
                    assert child["timestamp"] == Decimal(item["time_s"])
                witnesses_checked += 1

synthetic = {s["source"]: s["primary"][0] for s in summary["sources"] if s["synthetic"]}
assert synthetic["synthetic-negative"]["positive_windows"] == 1
assert synthetic["synthetic-positive"]["positive_windows"] == 0
assert all(r["full_high_hit_high_ttft_pattern_percent"] is None for s in summary["sources"] for r in s["primary"])
baseline = json.loads((ROOT / "results/evidence-review-2026-09-30/preserved-file-hashes.json").read_text())["files"]
for name, digest in baseline.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
result = dict(status="passed", real_source_configurations_independently_recomputed=checked,
              positive_primary_window_witnesses_checked=witnesses_checked,
              synthetic_diagnostic="Mixed trace matched; short-only trace did not",
              historical_files_unchanged=len(baseline), protocol_sha256=summary["protocol_sha256"],
              scope="Validation of the declared structural statistic; not validation of production prevalence")
(OUT / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
