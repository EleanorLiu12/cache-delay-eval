"""Summarize routing replays and profiles under a results directory and flag failed gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def last_row(rows: list[dict], kind: str):
    return next((r for r in reversed(rows) if r.get("type") == kind), None)


def index_agreement(rows: list[dict]) -> str:
    """Router-predicted vs engine-reported cached tokens. Under-prediction is the
    expected lag (a queued request's prefix is not yet in the engine's cache)."""
    diffs = [r["predicted_hit_tokens"] - r["engine_cached_tokens"] for r in rows
             if r.get("type") == "request" and r.get("engine_cached_tokens") is not None]
    if not diffs:
        return "index -"
    under = [d for d in diffs if d < 0]
    over = [d for d in diffs if d > 0]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0
    return (f"index exact {diffs.count(0)}/{len(diffs)}, under {len(under)} (mean {-mean(under):.0f}), "
            f"over {len(over)} (mean {mean(over):.0f})")


def run_problems(s: dict) -> list[str]:
    problems = []
    if s["errors"]:
        problems.append(f"{s['errors']} errors")
    if any(s["event_seq_gaps"]):
        problems.append(f"event gaps {s['event_seq_gaps']}")
    if s["missing_engine_cached"]:
        problems.append(f"{s['missing_engine_cached']} without engine cached tokens")
    if s["late_dispatches_50ms"]:
        problems.append(f"{s['late_dispatches_50ms']} dispatches >50 ms late")
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    args = p.parse_args(argv)
    failed = 0
    for path in sorted(Path(args.root).rglob("*.jsonl")):
        name = path.relative_to(args.root)
        rows = load(path)
        summary = last_row(rows, "run_summary")
        if summary:
            problems = run_problems(summary)
            ttft = summary["ttft_ms"]
            mean = f"{ttft['mean']:.0f}" if ttft["mean"] is not None else "-"
            p99 = f"{ttft['p99']:.0f}" if ttft["p99"] is not None else "-"
            cached = summary["engine_cached_fraction"]
            print(f"{'FAIL' if problems else 'ok  '} {name}: n={summary['requests']} "
                  f"per_instance={summary['per_instance']} ttft_mean={mean}ms p99={p99}ms "
                  f"cached={'-' if cached is None else f'{cached:.2f}'} {index_agreement(rows)} {'; '.join(problems)}")
            failed += bool(problems)
            continue
        summary = last_row(rows, "summary")
        if summary:
            problems = []
            if summary["cache_mismatch"] or summary["missing_cached"]:
                problems.append(f"cache mismatch {summary['cache_mismatch']}, missing {summary['missing_cached']}")
            cold, decode = summary["prefill_cold_vs_tokens"], summary["decode_step_vs_batch"]
            print(f"{'FAIL' if problems else 'ok  '} {name}: prefill {cold['intercept_ms']:.1f}+"
                  f"{cold['slope_ms'] * 1000:.1f}ms/1k tokens, decode step {decode['intercept_ms']:.1f}+"
                  f"{decode['slope_ms']:.2f}ms per request {'; '.join(problems)}")
            failed += bool(problems)
        else:
            print(f"FAIL {name}: no summary row (run interrupted?)")
            failed += 1
    print(f"{failed} file(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
