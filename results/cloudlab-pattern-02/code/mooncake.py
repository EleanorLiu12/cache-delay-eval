"""Convert a Mooncake FAST'25 trace to the request-trace-v1 format.

Each Mooncake line holds `timestamp` (ms offset), `input_length` and
`output_length` (tokens), and `hash_ids` (remapped 512-token prefix block
hashes). The release has no session, user, or tool/human fields, so converted
requests carry none of them.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .trace import TraceHeader, TraceRequest, TraceValidationError, _json_records, write_trace


MOONCAKE_BLOCK_SIZE = 512


def _integer(row: dict[str, Any], name: str, minimum: int, line: int) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TraceValidationError(f"line {line}: {name} must be an integer >= {minimum}")
    return value


def convert_records(records: Iterable[tuple[int, Any]]) -> tuple[list[TraceRequest], dict[str, Any]]:
    """Return requests ordered by arrival plus counts of every adjustment made."""
    requests: list[TraceRequest] = []
    clamped = 0
    for index, (line, row) in enumerate(records):
        if not isinstance(row, dict):
            raise TraceValidationError(f"line {line}: record must be an object")
        hash_ids = row.get("hash_ids")
        if (
            not isinstance(hash_ids, list)
            or not hash_ids
            or any(isinstance(h, bool) or not isinstance(h, int) or h < 0 for h in hash_ids)
        ):
            raise TraceValidationError(
                f"line {line}: hash_ids must be a non-empty array of non-negative integers"
            )
        output_length = _integer(row, "output_length", 0, line)
        metadata: dict[str, Any] = {}
        # request-trace-v1 requires output_tokens >= 1; keep the raw value visible.
        if output_length == 0:
            clamped += 1
            metadata["source_output_length"] = 0
        requests.append(
            TraceRequest(
                request_id=f"mooncake-{index:06d}",
                arrival_time_ms=_integer(row, "timestamp", 0, line),
                output_tokens=max(1, output_length),
                block_hashes=tuple(hash_ids),
                prompt_tokens=_integer(row, "input_length", 1, line),
                metadata=metadata,
            )
        )
    if not requests:
        raise TraceValidationError("trace must contain at least one request")
    reordered = any(
        later.arrival_time_ms < earlier.arrival_time_ms
        for earlier, later in zip(requests, requests[1:])
    )
    # Stable sort: same-timestamp requests keep their source order.
    requests.sort(key=lambda request: request.arrival_time_ms)
    return requests, {"clamped_output_tokens": clamped, "reordered": reordered}


def convert(source: Path, output: Path) -> int:
    requests, adjustments = convert_records(_json_records(source))
    header = TraceHeader(
        trace_id=f"mooncake-{source.stem}",
        created_at=datetime.now(timezone.utc).isoformat(),
        source="mooncake-fast25",
        block_size=MOONCAKE_BLOCK_SIZE,
        metadata={"source_file": source.name, **adjustments},
    )
    write_trace(output, header, requests)
    return len(requests)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a Mooncake trace to request-trace-v1")
    parser.add_argument("source", type=Path, help="Mooncake JSONL, e.g. toolagent_trace.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        count = convert(args.source, args.output)
    except (OSError, TraceValidationError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {count} requests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
