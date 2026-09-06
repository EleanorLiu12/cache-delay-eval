"""Versioned JSONL trace I/O and semantic validation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA_VERSION = "1.1"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0", "1.1"})


class TraceValidationError(ValueError):
    """Raised when a trace violates the v1 format or semantic constraints."""


@dataclass(frozen=True)
class TraceHeader:
    trace_id: str
    created_at: str
    source: str
    model_id: str | None = None
    tokenizer_id: str | None = None
    block_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "type": "trace_meta",
            "schema_version": SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "time_unit": "ms",
            "source": self.source,
            "metadata": self.metadata,
        }
        if self.model_id is not None:
            value["model_id"] = self.model_id
        if self.tokenizer_id is not None:
            value["tokenizer_id"] = self.tokenizer_id
        if self.block_size is not None:
            value["block_size"] = self.block_size
        return value


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    arrival_time_ms: int
    output_tokens: int
    prompt: str | None = None
    token_ids: tuple[int, ...] | None = None
    block_hashes: tuple[int | str, ...] | None = None
    prompt_tokens: int | None = None
    session_id: str | None = None
    user_id: str | None = None
    parent_request_id: str | None = None
    prefix_group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "type": "request",
            "request_id": self.request_id,
            "arrival_time_ms": self.arrival_time_ms,
            "output_tokens": self.output_tokens,
            "metadata": self.metadata,
        }
        if self.prompt is not None:
            value["prompt"] = self.prompt
        if self.token_ids is not None:
            value["token_ids"] = list(self.token_ids)
        if self.block_hashes is not None:
            value["block_hashes"] = list(self.block_hashes)
        if self.prompt_tokens is not None:
            value["prompt_tokens"] = self.prompt_tokens
        for key in ("session_id", "user_id", "parent_request_id", "prefix_group"):
            item = getattr(self, key)
            if item is not None:
                value[key] = item
        return value


def _nonempty_string(value: Any, name: str, line: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TraceValidationError(f"line {line}: {name} must be a non-empty string")
    return value


def _is_block_hash(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    return isinstance(value, str) and bool(value)


def _metadata(value: Any, line: int) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TraceValidationError(f"line {line}: metadata must be an object")
    return value


def parse_header(value: Any, line: int = 1) -> TraceHeader:
    if not isinstance(value, dict) or value.get("type") != "trace_meta":
        raise TraceValidationError(f"line {line}: first record must have type='trace_meta'")
    if value.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise TraceValidationError(
            f"line {line}: unsupported schema_version {value.get('schema_version')!r}; "
            f"supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    if value.get("time_unit") != "ms":
        raise TraceValidationError(f"line {line}: time_unit must be 'ms'")
    return TraceHeader(
        trace_id=_nonempty_string(value.get("trace_id"), "trace_id", line),
        created_at=_nonempty_string(value.get("created_at"), "created_at", line),
        source=_nonempty_string(value.get("source"), "source", line),
        model_id=value.get("model_id"),
        tokenizer_id=value.get("tokenizer_id"),
        block_size=value.get("block_size"),
        metadata=_metadata(value.get("metadata"), line),
    )


def parse_request(value: Any, line: int) -> TraceRequest:
    if not isinstance(value, dict) or value.get("type") != "request":
        raise TraceValidationError(f"line {line}: record must have type='request'")
    request_id = _nonempty_string(value.get("request_id"), "request_id", line)
    arrival = value.get("arrival_time_ms")
    if isinstance(arrival, bool) or not isinstance(arrival, int) or arrival < 0:
        raise TraceValidationError(f"line {line}: arrival_time_ms must be a non-negative integer")
    output_tokens = value.get("output_tokens")
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens < 1:
        raise TraceValidationError(f"line {line}: output_tokens must be a positive integer")

    prompt = value.get("prompt")
    token_ids = value.get("token_ids")
    block_hashes = value.get("block_hashes")
    supplied = [
        name
        for name, item in (
            ("prompt", prompt),
            ("token_ids", token_ids),
            ("block_hashes", block_hashes),
        )
        if item is not None
    ]
    if len(supplied) != 1:
        raise TraceValidationError(
            f"line {line}: provide exactly one of prompt, token_ids, or block_hashes "
            f"(found {supplied or 'none'})"
        )
    if prompt is not None and (not isinstance(prompt, str) or not prompt):
        raise TraceValidationError(f"line {line}: prompt must be a non-empty string")
    parsed_tokens: tuple[int, ...] | None = None
    if token_ids is not None:
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in token_ids)
        ):
            raise TraceValidationError(
                f"line {line}: token_ids must be a non-empty array of non-negative integers"
            )
        parsed_tokens = tuple(token_ids)
    parsed_hashes: tuple[int | str, ...] | None = None
    if block_hashes is not None:
        if (
            not isinstance(block_hashes, list)
            or not block_hashes
            or any(not _is_block_hash(h) for h in block_hashes)
        ):
            raise TraceValidationError(
                f"line {line}: block_hashes must be a non-empty array of non-negative "
                "integers or non-empty strings"
            )
        parsed_hashes = tuple(block_hashes)

    prompt_tokens = value.get("prompt_tokens")
    if prompt_tokens is not None and (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens < 1
    ):
        raise TraceValidationError(f"line {line}: prompt_tokens must be a positive integer")
    if block_hashes is not None and prompt_tokens is None:
        raise TraceValidationError(
            f"line {line}: prompt_tokens is required alongside block_hashes, because a "
            "hash-only record carries no text from which to derive the prompt length"
        )

    optional: dict[str, str | None] = {}
    for key in ("session_id", "user_id", "parent_request_id", "prefix_group"):
        item = value.get(key)
        optional[key] = None if item is None else _nonempty_string(item, key, line)
    return TraceRequest(
        request_id=request_id,
        arrival_time_ms=arrival,
        output_tokens=output_tokens,
        prompt=prompt,
        token_ids=parsed_tokens,
        block_hashes=parsed_hashes,
        prompt_tokens=prompt_tokens,
        metadata=_metadata(value.get("metadata"), line),
        **optional,
    )


def _json_records(path: Path) -> Iterator[tuple[int, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                yield line_no, json.loads(raw)
            except json.JSONDecodeError as exc:
                raise TraceValidationError(f"line {line_no}: invalid JSON: {exc.msg}") from exc


def read_trace(path: str | Path) -> tuple[TraceHeader, list[TraceRequest]]:
    records = _json_records(Path(path))
    try:
        line, raw_header = next(records)
    except StopIteration as exc:
        raise TraceValidationError("trace is empty") from exc
    header = parse_header(raw_header, line)
    requests: list[TraceRequest] = []
    seen: dict[str, TraceRequest] = {}
    session_users: dict[str, str] = {}
    last_arrival = -1
    for line, raw in records:
        request = parse_request(raw, line)
        if request.request_id in seen:
            raise TraceValidationError(f"line {line}: duplicate request_id {request.request_id!r}")
        if request.arrival_time_ms < last_arrival:
            raise TraceValidationError(
                f"line {line}: arrival_time_ms is earlier than the preceding request"
            )
        if request.parent_request_id is not None:
            parent = seen.get(request.parent_request_id)
            if parent is None:
                raise TraceValidationError(
                    f"line {line}: parent_request_id must refer to an earlier request"
                )
            if request.session_id is None or request.session_id != parent.session_id:
                raise TraceValidationError(
                    f"line {line}: child and parent must have the same session_id"
                )
        if request.session_id is not None and request.user_id is not None:
            owner = session_users.setdefault(request.session_id, request.user_id)
            if owner != request.user_id:
                raise TraceValidationError(
                    f"line {line}: session {request.session_id!r} is already attributed to "
                    f"user {owner!r}, not {request.user_id!r}"
                )
        seen[request.request_id] = request
        requests.append(request)
        last_arrival = request.arrival_time_ms
    if not requests:
        raise TraceValidationError("trace must contain at least one request")
    return header, requests


def write_trace(
    path: str | Path, header: TraceHeader, requests: Iterable[TraceRequest]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(header.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        for request in requests:
            stream.write(json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a cache-delay request trace")
    parser.add_argument("trace", type=Path)
    args = parser.parse_args(argv)
    try:
        header, requests = read_trace(args.trace)
    except (OSError, TraceValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    sessions = {r.session_id for r in requests if r.session_id is not None}
    users = {r.user_id for r in requests if r.user_id is not None}
    print(
        f"VALID trace_id={header.trace_id} "
        f"requests={len(requests)} sessions={len(sessions)} users={len(users)} "
        f"duration_ms={requests[-1].arrival_time_ms - requests[0].arrival_time_ms}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

