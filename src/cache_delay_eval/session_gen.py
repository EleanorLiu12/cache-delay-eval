"""Synthetic agent-session generator for prefix-cache trace patterns.

Three workload controls are implemented here:

    P    shared prefix length in tokens (system prompt + tool defs + few-shot)
    W    whole-trace distinct blocks / nominal KV capacity (metadata only)
    rho  length-reuse coupling, corr(prompt_tokens, reuse), swept by mixing
         two session archetypes

Output is request-trace-v1 JSONL in its ``block_hashes`` form, which is the
form a block-level replay needs and the only one that does not require a
tokenizer.


Block-hash semantics
--------------------
vLLM hashes a block as ``H(parent_block_hash, token_ids_of_this_block)``, so
two requests share a leading block hash exactly when they share that entire
leading token sequence, and a shared run of blocks is always a contiguous
prefix.  This generator reproduces that property without materialising token
ids: every session owns one monotone token stream, blocks are cut from it at
fixed ``block_size`` boundaries, and a block is given a shared identity only
when it lies wholly inside the shared prefix and the two sessions draw the
same prefix group.  Turn k of a session prompts with ``stream[:L_k]``, so its
full blocks are stream blocks ``0 .. L_k // block_size - 1`` and turn k's
hashes are a prefix of turn k+1's, exactly as real multi-turn chat behaves.

A block that straddles the end of the shared prefix (when P is not a multiple
of block_size) mixes shared and session-private tokens and is therefore *not*
shared, which is also what vLLM does.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .trace import TraceHeader, TraceRequest, write_trace


DEFAULT_BLOCK_SIZE = 16


@dataclass(frozen=True)
class Archetype:
    """One session shape.  Mixing two of these is how `rho` is controlled."""

    name: str
    turns: int
    first_user_tokens: int
    user_tokens: int
    output_tokens: int


# Short-turn deep session: ~200 tokens per turn, ~30 turns -> short prompts,
# high reuse.  Long-turn shallow session: ~16K first turn, ~3 turns -> long
# prompts, low reuse on the turn that carries almost all of the tokens.
DEEP_SHORT = Archetype("deep_short", turns=30, first_user_tokens=150, user_tokens=150, output_tokens=50)
SHALLOW_LONG = Archetype("shallow_long", turns=3, first_user_tokens=16000, user_tokens=300, output_tokens=300)


@dataclass(frozen=True)
class WorkloadKnobs:
    """Prefix length, nominal capacity metadata and session mixture."""

    # --- implemented ---------------------------------------------------
    P: int = 512
    """Shared prefix length in tokens.  Design levels: 512 (low), 8192 (high)."""

    W: float = 0.3
    """Whole-trace distinct blocks / nominal KV capacity. Levels: 0.3, 3.0.

    The generator does not itself evict; it records the capacity in blocks
    that realises this ratio for the trace it just produced, under
    ``metadata.capacity_blocks``. This is not the concurrent working set;
    the live pilot uses its own explicit capacity instead."""

    rho_mix: float = 0.0
    """Fraction of *sessions* drawn from SHALLOW_LONG rather than DEEP_SHORT.

    Not equal to rho.  rho is measured from the finished trace by
    ``measure_rho``; use ``rho_profile`` to map mix -> rho."""

    def check(self) -> None:
        if self.P < 1:
            raise ValueError("P must be a positive token count")
        if self.W <= 0:
            raise ValueError("W must be positive")
        if not 0.0 <= self.rho_mix <= 1.0:
            raise ValueError("rho_mix must be in [0, 1]")


@dataclass(frozen=True)
class GeneratorConfig:
    """Session shapes, fixed arrival generation and reproducibility settings."""

    requests: int = 400
    """Target request count.  Sessions are drawn until this is reached, so the
    trace stays the same size as rho_mix moves the session mixture."""

    prefix_groups: int = 4
    block_size: int = DEFAULT_BLOCK_SIZE
    seed: int = 699
    session_rate_per_s: float = 2.0
    """Poisson session starts per second."""

    think_time_s: float = 5.0
    """Fixed within-session arrival gap; independent of completion time."""

    length_jitter: float = 0.15
    """Lognormal sigma applied to every per-turn token count."""

    deep: Archetype = DEEP_SHORT
    shallow: Archetype = SHALLOW_LONG
    """The two session shapes mixed by rho_mix; overrides are recorded in metadata."""


def _block_hash(key: tuple[Any, ...]) -> int:
    """A 64-bit opaque block identifier; equality is the only thing that matters."""
    return int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "big")


def _jittered(rng: random.Random, mean: int, sigma: float) -> int:
    if sigma <= 0:
        return mean
    # Lognormal with the requested median; clipped so no turn can vanish.
    return max(1, round(mean * math.exp(rng.gauss(0.0, sigma))))


def _session_turn_lengths(
    rng: random.Random, archetype: Archetype, sigma: float
) -> list[tuple[int, int]]:
    """Return [(user_tokens, output_tokens)] for each turn of one session."""
    turns = []
    for index in range(archetype.turns):
        mean_user = archetype.first_user_tokens if index == 0 else archetype.user_tokens
        turns.append(
            (_jittered(rng, mean_user, sigma), _jittered(rng, archetype.output_tokens, sigma))
        )
    return turns


def _stream_block_hashes(
    count: int, shared_blocks: int, prefix_group: int, session_uid: int
) -> list[int]:
    """Identities of the first `count` blocks of one session's token stream.

    Blocks wholly inside the shared prefix take a per-(group, index) identity
    and are therefore shared with every other session in the same group; every
    later block is session-private.  Because the shared run is always the
    leading run, prefix contiguity holds by construction.
    """
    hashes = []
    for index in range(count):
        if index < shared_blocks:
            hashes.append(_block_hash(("prefix", prefix_group, index)))
        else:
            hashes.append(_block_hash(("session", session_uid, index)))
    return hashes


def generate_requests(
    knobs: WorkloadKnobs | None = None,
    config: GeneratorConfig | None = None,
) -> tuple[list[TraceRequest], dict[str, Any]]:
    """Generate the request list plus the derived trace statistics."""
    knobs = knobs or WorkloadKnobs()
    config = config or GeneratorConfig()
    knobs.check()
    if config.requests < 1 or config.prefix_groups < 1 or config.block_size < 1:
        raise ValueError("requests, prefix_groups and block_size must be positive")
    if config.session_rate_per_s <= 0 or config.think_time_s < 0:
        raise ValueError("session_rate_per_s must be positive and think_time_s non-negative")

    rng = random.Random(config.seed)
    block_size = config.block_size
    shared_blocks = knobs.P // block_size

    events: list[dict[str, Any]] = []
    session_start_s = 0.0
    session_uid = 0
    emitted = 0
    archetype_counts = {config.deep.name: 0, config.shallow.name: 0}

    while emitted < config.requests:
        archetype = config.shallow if rng.random() < knobs.rho_mix else config.deep
        archetype_counts[archetype.name] += 1
        prefix_group = rng.randrange(config.prefix_groups)  # uniform until `s` lands
        turn_lengths = _session_turn_lengths(rng, archetype, config.length_jitter)

        # Prompt of turn k is the stream up to and including turn k's user text.
        prompt_lengths: list[int] = []
        cursor = knobs.P
        for user_tokens, output_tokens in turn_lengths:
            cursor += user_tokens
            prompt_lengths.append(cursor)
            cursor += output_tokens
        block_hashes = _stream_block_hashes(
            prompt_lengths[-1] // block_size, shared_blocks, prefix_group, session_uid
        )

        arrival_s = session_start_s
        for turn_index, (prompt_tokens, (_, output_tokens)) in enumerate(
            zip(prompt_lengths, turn_lengths)
        ):
            full_blocks = prompt_tokens // block_size
            events.append(
                {
                    "arrival_ms": round(arrival_s * 1000),
                    "session_uid": session_uid,
                    "prefix_group": prefix_group,
                    "turn_index": turn_index,
                    "archetype": archetype.name,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "block_hashes": tuple(block_hashes[:full_blocks]),
                }
            )
            emitted += 1
            arrival_s += rng.expovariate(1.0 / config.think_time_s) if config.think_time_s else 0.0

        session_uid += 1
        session_start_s += rng.expovariate(config.session_rate_per_s)

    # Global arrival order; ties keep session/turn order so parents stay earlier.
    events.sort(key=lambda event: (event["arrival_ms"], event["session_uid"], event["turn_index"]))

    seen_blocks: set[int] = set()
    previous_in_session: dict[int, str] = {}
    requests: list[TraceRequest] = []
    for index, event in enumerate(events):
        hashes = event["block_hashes"]
        # Oracle reuse: the share of this prompt's full blocks that some
        # earlier request already introduced, i.e. the hit rate an unbounded
        # cache would give.  Ground truth for measuring rho; the simulator
        # computes its own hit rate under a finite cache.
        repeated = sum(1 for h in hashes if h in seen_blocks)
        oracle_hit_rate = repeated / len(hashes) if hashes else 0.0
        seen_blocks.update(hashes)
        session_id = f"session-{event['session_uid']:05d}"
        request_id = f"req-{index:06d}"
        requests.append(
            TraceRequest(
                request_id=request_id,
                arrival_time_ms=event["arrival_ms"],
                output_tokens=event["output_tokens"],
                block_hashes=hashes,
                prompt_tokens=event["prompt_tokens"],
                session_id=session_id,
                user_id=f"user-{event['session_uid']:05d}",
                parent_request_id=previous_in_session.get(event["session_uid"]),
                prefix_group=f"prefix-{event['prefix_group']:03d}",
                metadata={
                    "archetype": event["archetype"],
                    "turn_index": event["turn_index"],
                    "oracle_hit_rate": oracle_hit_rate,
                    "prompt_blocks": len(hashes),
                },
            )
        )
        previous_in_session[event["session_uid"]] = request_id

    stats = derive_stats(requests, knobs, config, archetype_counts)
    return requests, stats


def derive_stats(
    requests: Sequence[TraceRequest],
    knobs: WorkloadKnobs,
    config: GeneratorConfig,
    archetype_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Working set, the capacity that realises W, and the achieved rho."""
    distinct: set[int] = set()
    for request in requests:
        distinct.update(request.block_hashes or ())
    working_set_blocks = len(distinct)
    # W = working set / capacity, so capacity = working set / W.  Defining the
    # working set over the whole trace (not a sliding window) makes W depend on
    # trace length; keep `requests` fixed when comparing cells.
    capacity_blocks = max(1, round(working_set_blocks / knobs.W))
    rho, rho_spearman = measure_rho(requests)
    return {
        "P": knobs.P,
        "W": knobs.W,
        "rho_mix": knobs.rho_mix,
        "rho": rho,
        "rho_spearman": rho_spearman,
        "block_size": config.block_size,
        "prefix_groups": config.prefix_groups,
        "seed": config.seed,
        "requests": len(requests),
        "sessions": len({r.session_id for r in requests}),
        "archetype_sessions": archetype_counts or {},
        "deep_archetype": config.deep.__dict__,
        "shallow_archetype": config.shallow.__dict__,
        "working_set_blocks": working_set_blocks,
        "working_set_tokens": working_set_blocks * config.block_size,
        "capacity_blocks": capacity_blocks,
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denominator = math.sqrt(sum(d * d for d in dx) * sum(d * d for d in dy))
    if denominator == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        average = (index + stop) / 2.0
        for position in range(index, stop + 1):
            ranks[order[position]] = average
        index = stop + 1
    return ranks


def measure_rho(requests: Sequence[TraceRequest]) -> tuple[float, float]:
    """corr(prompt_tokens, oracle hit rate) over the trace, Pearson and Spearman."""
    lengths = [float(r.prompt_tokens or 0) for r in requests]
    hits = [float(r.metadata.get("oracle_hit_rate", 0.0)) for r in requests]
    return _pearson(lengths, hits), _pearson(_ranks(lengths), _ranks(hits))


def rho_profile(
    mixes: Sequence[float],
    knobs: WorkloadKnobs | None = None,
    config: GeneratorConfig | None = None,
    seeds: Sequence[int] = (0,),
) -> list[dict[str, Any]]:
    """Measure rho across a grid of mixture weights, averaged over seeds.

    rho(mix) is not monotone: both pure archetypes have a positive within-
    session slope (later turns are both longer and better cached), and the
    negative values come from the between-archetype contrast, which is
    strongest at intermediate mixtures.  So scan, do not bisect.
    """
    base_knobs = knobs or WorkloadKnobs()
    base_config = config or GeneratorConfig()
    rows = []
    for mix in mixes:
        per_seed = []
        for seed in seeds:
            requests, stats = generate_requests(
                WorkloadKnobs(P=base_knobs.P, W=base_knobs.W, rho_mix=mix),
                GeneratorConfig(**{**base_config.__dict__, "seed": seed}),
            )
            per_seed.append(stats)
        rows.append(
            {
                "rho_mix": mix,
                "rho_mean": sum(s["rho"] for s in per_seed) / len(per_seed),
                "rho_min": min(s["rho"] for s in per_seed),
                "rho_max": max(s["rho"] for s in per_seed),
                "rho_spearman_mean": sum(s["rho_spearman"] for s in per_seed) / len(per_seed),
                "requests_mean": sum(s["requests"] for s in per_seed) / len(per_seed),
                "working_set_blocks_mean": sum(s["working_set_blocks"] for s in per_seed) / len(per_seed),
            }
        )
    return rows


def nearest_mix(profile: Sequence[dict[str, Any]], target_rho: float) -> float:
    """Pick the grid mixture whose mean rho is closest to `target_rho`."""
    if not profile:
        raise ValueError("profile is empty")
    return min(profile, key=lambda row: abs(row["rho_mean"] - target_rho))["rho_mix"]


def build_trace(
    knobs: WorkloadKnobs | None = None,
    config: GeneratorConfig | None = None,
    created_at: str | None = None,
) -> tuple[TraceHeader, list[TraceRequest], dict[str, Any]]:
    knobs = knobs or WorkloadKnobs()
    config = config or GeneratorConfig()
    requests, stats = generate_requests(knobs, config)
    header = TraceHeader(
        trace_id=f"sessions-P{knobs.P}-W{knobs.W}-mix{knobs.rho_mix}-seed{config.seed}",
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        source="synthetic-agent-sessions",
        block_size=config.block_size,
        metadata=stats,
    )
    return header, requests, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-tokens", "-P", type=int, default=512, help="knob P")
    parser.add_argument("--working-set-ratio", "-W", type=float, default=0.3, help="knob W")
    parser.add_argument("--rho-mix", type=float, default=0.0, help="knob rho: archetype mixture")
    parser.add_argument("--requests", type=int, default=GeneratorConfig.requests)
    parser.add_argument("--prefix-groups", type=int, default=GeneratorConfig.prefix_groups)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    header, requests, stats = build_trace(
        WorkloadKnobs(P=args.prefix_tokens, W=args.working_set_ratio, rho_mix=args.rho_mix),
        GeneratorConfig(
            requests=args.requests,
            prefix_groups=args.prefix_groups,
            block_size=args.block_size,
            seed=args.seed,
        ),
    )
    write_trace(args.output, header, requests)
    print(
        f"wrote {len(requests)} requests to {args.output}\n"
        f"  rho={stats['rho']:+.3f} (spearman {stats['rho_spearman']:+.3f})  "
        f"working_set={stats['working_set_blocks']} blocks  "
        f"capacity_for_W={stats['capacity_blocks']} blocks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
