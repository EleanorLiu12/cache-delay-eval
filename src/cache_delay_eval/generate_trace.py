"""Generate a deterministic synthetic trace with controlled prefix reuse."""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path

from .trace import TraceHeader, TraceRequest, write_trace


TOPICS = (
    "distributed systems observability scheduling cache consistency",
    "machine learning inference batching latency throughput",
    "database indexing transactions isolation recovery",
    "computer networks routing congestion reliability transport",
    "operating systems memory processes virtualization storage",
    "software engineering testing maintenance architecture review",
    "security authentication authorization encryption auditing",
    "data science sampling estimation regression visualization",
)


def generate_requests(
    count: int,
    request_rate: float,
    reuse_probability: float,
    prefix_groups: int,
    prompt_words: int,
    output_tokens: int,
    seed: int,
) -> list[TraceRequest]:
    if count < 1 or request_rate <= 0 or not 0 <= reuse_probability <= 1:
        raise ValueError("count and request_rate must be positive; reuse_probability must be in [0,1]")
    if prefix_groups < 1 or prompt_words < 8 or output_tokens < 1:
        raise ValueError("prefix_groups/output_tokens must be positive and prompt_words must be >= 8")
    rng = random.Random(seed)
    prefixes = [
        (f"Shared context {group}: " + TOPICS[group % len(TOPICS)] + " ") * 12
        for group in range(prefix_groups)
    ]
    arrival_s = 0.0
    previous_by_session: dict[str, str] = {}
    requests: list[TraceRequest] = []
    for index in range(count):
        if index:
            arrival_s += rng.expovariate(request_rate)
        reused = rng.random() < reuse_probability
        group = rng.randrange(prefix_groups) if reused else None
        session_id = f"session-{group:03d}" if group is not None else f"singleton-{index:06d}"
        prefix = prefixes[group] if group is not None else f"Unique context for request {index}. "
        suffix_words = [
            f"item{rng.randrange(100000):05d}" for _ in range(max(1, prompt_words - len(prefix.split())))
        ]
        prompt = (prefix + "Question " + " ".join(suffix_words)).strip()
        request_id = f"req-{index:06d}"
        parent = previous_by_session.get(session_id) if group is not None else None
        requests.append(
            TraceRequest(
                request_id=request_id,
                arrival_time_ms=round(arrival_s * 1000),
                output_tokens=output_tokens,
                prompt=prompt,
                session_id=session_id,
                parent_request_id=parent,
                prefix_group=None if group is None else f"prefix-{group:03d}",
                metadata={"synthetic": True, "intended_prefix_reuse": reused},
            )
        )
        previous_by_session[session_id] = request_id
    return requests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--request-rate", type=float, default=2.0, help="Poisson arrivals/second")
    parser.add_argument("--reuse-probability", type=float, default=0.7)
    parser.add_argument("--prefix-groups", type=int, default=16)
    parser.add_argument("--prompt-words", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=699)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    requests = generate_requests(
        args.requests,
        args.request_rate,
        args.reuse_probability,
        args.prefix_groups,
        args.prompt_words,
        args.output_tokens,
        args.seed,
    )
    header = TraceHeader(
        trace_id=f"synthetic-seed-{args.seed}",
        created_at=datetime.now(timezone.utc).isoformat(),
        source="synthetic-controlled-prefix-reuse",
        metadata={
            "seed": args.seed,
            "request_rate_per_s": args.request_rate,
            "reuse_probability": args.reuse_probability,
            "prefix_groups": args.prefix_groups,
        },
    )
    write_trace(args.output, header, requests)
    print(f"wrote {len(requests)} requests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

