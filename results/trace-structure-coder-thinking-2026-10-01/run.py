"""Apply the frozen October 1 structural rule, unchanged, to the Qwen-Bailian Coder and Thinking traces."""

import hashlib
import json
import sys
from math import sqrt
from pathlib import Path

OUT = Path(__file__).resolve().parent
RULE = OUT.parent / "trace-structure-2026-10-01"
sys.path.insert(0, str(RULE))
import analyze  # frozen rule; imported, not modified

analyze.OUT = OUT  # write per-trace evidence here, not into the October 1 folder

# Trace file versus the usage scenario the publisher's README assigns to it.
TRACES = [
    dict(trace="Qwen-Bailian Trace A", scenario="To-C: chat-style interactive services", source="qwen-a", file=None),
    dict(trace="Qwen-Bailian Trace B", scenario="To-B: API-driven task automation", source="qwen-b", file=None),
    dict(trace="Qwen-Bailian Coder trace", scenario="Code generation", source="qwen-coder",
         file="qwen-bailian-coder/*/qwen_coder_blksz_16.jsonl"),
    dict(trace="Qwen-Bailian Thinking trace", scenario="Reasoning-intensive chat", source="qwen-thinking",
         file="qwen-bailian-thinking/*/qwen_thinking_blksz_16.jsonl"),
]


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h) * 100, min(1.0, c + h) * 100


def main():
    frozen = {s["source"]: s for s in json.loads((RULE / "summary.json").read_text())["sources"]}
    rows = []
    for t in TRACES:
        if t["file"]:
            paths = list((analyze.ROOT / "data").glob(t["file"]))
            assert len(paths) == 1
            source = analyze.analyze_source(t["source"], paths[0])
            origin = "computed in this run"
        else:
            source = frozen[t["source"]]
            origin = "copied from ../trace-structure-2026-10-01/summary.json"
        for c in source["primary"]:
            low, high = wilson(c["positive_windows"], c["total_windows"])
            rows.append(dict(trace=t["trace"], scenario=t["scenario"], type_label=c["cohort"],
                             positive_windows=c["positive_windows"], total_windows=c["total_windows"],
                             percent=c["percent"], wilson95_percent=[low, high],
                             source_requests=c["source_requests"], eligible_chains=c["eligible_chains"],
                             excluded_chains=c["excluded_chains"], trace_sha256=source["sha256"], origin=origin))
    summary = dict(rule_protocol_sha256=analyze.PROTOCOL_HASH,
                   rule_analysis_sha256=hashlib.sha256((RULE / "analyze.py").read_bytes()).hexdigest(),
                   scenario_source="Publisher README scenario table; not inferred from request content",
                   rows=rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for r in rows:
        print(f"{r['trace']:28} {r['scenario']:40} {r['type_label']:9} "
              f"{r['positive_windows']:2}/{r['total_windows']} {r['percent']:5.1f}%  "
              f"[{r['wilson95_percent'][0]:.0f}%, {r['wilson95_percent'][1]:.0f}%]  chains {r['eligible_chains']} excl {r['excluded_chains']}")


if __name__ == "__main__":
    main()
