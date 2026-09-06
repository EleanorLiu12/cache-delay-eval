"""Audit calibration rows for the experimental invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(path: Path) -> dict[str, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {
        "rows": len(rows),
        "errors": sum(row.get("status") != "ok" for row in rows),
        "prompt_length_mismatches": sum(
            row.get("status") == "ok"
            and row.get("prompt_tokens_reported") != row.get("prompt_tokens_target")
            for row in rows
        ),
        "cache_hit_mismatches": sum(
            row.get("status") == "ok"
            and row.get("prefix_cache_hit_tokens_batch") is not None
            and row.get("prefix_cache_hit_tokens_batch")
            != row.get("cached_prefix_tokens_target")
            for row in rows
        ),
        "queue_depth_mismatches": sum(
            row.get("status") == "ok"
            and row.get("queue_depth_observed") is not None
            and row.get("queue_depth_observed") != row.get("queue_depth_target")
            for row in rows
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    checks = audit(args.input)
    print(json.dumps(checks, indent=2, sort_keys=True))
    failures = sum(value for key, value in checks.items() if key != "rows")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())

