"""Exact waiting-order search for a deliberately small, uncalibrated service model.

This is NOT a vLLM simulator. One token occupies one block. Full prompt AND
scripted output capacity is reserved at admission; no preemption is needed.
Running requests receive service in admission order; only the waiting order
can change. Every service iteration costs positive integer toy ticks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from itertools import permutations
import math
from typing import Iterable


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt: tuple[int, ...]
    output: tuple[int, ...]
    arrival: int = 0


@dataclass(frozen=True)
class Instance:
    name: str
    requests: tuple[Request, ...]
    kv_blocks: int
    max_sequences: int = 2
    token_budget: int = 3
    prefill_chunk: int = 2
    overhead_ticks: int = 1
    prefill_token_ticks: int = 1
    decode_token_ticks: int = 1
    slack_percent: int = 5
    epsilon_ticks: int = 1

    def validate(self) -> None:
        if not 1 <= len(self.requests) <= 8:
            raise ValueError("The bounded model supports 1 to 8 single-turn requests")
        if len({r.request_id for r in self.requests}) != len(self.requests):
            raise ValueError("Request IDs must be unique")
        for name in ("kv_blocks", "max_sequences", "token_budget", "prefill_chunk",
                     "overhead_ticks", "prefill_token_ticks", "decode_token_ticks"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if any(type(v) is not int or v < 0 for v in (self.slack_percent, self.epsilon_ticks)):
            raise ValueError("Progress tolerances must be nonnegative integers")
        for request in self.requests:
            if not request.request_id or type(request.arrival) is not int or request.arrival < 0:
                raise ValueError("Each request needs an ID and nonnegative integer arrival")
            if not request.prompt or not request.output:
                raise ValueError("Prompts and fixed response scripts must be nonempty")
            if any(type(t) is not int for t in request.prompt + request.output):
                raise ValueError("Token identities must be integers")
            if len(request.prompt) + len(request.output) > self.kv_blocks:
                raise ValueError("Each request must fit alone, including its output reservation")


@dataclass(frozen=True)
class Block:
    block_id: int
    identity: tuple[int, ...] | None
    computed: bool
    refs: tuple[int, ...]
    touched: int


@dataclass(frozen=True)
class Progress:
    admitted_at: int | None = None
    prefill_done: int = 0
    block_ids: tuple[int, ...] = ()
    output_times: tuple[int, ...] = ()
    completed_at: int | None = None
    cached_tokens: int = 0


@dataclass(frozen=True)
class State:
    time: int
    progress: tuple[Progress, ...]
    running: tuple[int, ...] = ()
    blocks: tuple[Block, ...] = ()
    touch_clock: int = 0
    next_block_id: int = 0


def initial_state(instance: Instance) -> State:
    instance.validate()
    return State(0, tuple(Progress() for _ in instance.requests))


def eligible(instance: Instance, state: State) -> tuple[int, ...]:
    return tuple(sorted((i for i, r in enumerate(instance.requests)
                         if r.arrival <= state.time and state.progress[i].admitted_at is None),
                        key=lambda i: (instance.requests[i].arrival, i)))


def finished(state: State) -> bool:
    return all(p.completed_at is not None for p in state.progress)


def prefix_blocks(request: Request, blocks: Iterable[Block]) -> tuple[int, ...]:
    """Read-only lookup: full token-prefix identity, never token suffix identity."""
    available: dict[tuple[int, ...], int] = {}
    for block in sorted(blocks, key=lambda b: b.block_id):
        if block.computed and block.identity is not None:
            available.setdefault(block.identity, block.block_id)
    hits = []
    for end in range(1, len(request.prompt) + 1):
        identity = request.prompt[:end]
        if identity not in available:
            break
        hits.append(available[identity])
    return tuple(hits)


def fit_snapshot(instance: Instance, index: int, blocks: Iterable[Block]) -> dict:
    """Read-only fit result; referenced hits are protected from eviction."""
    blocks = tuple(blocks)
    request = instance.requests[index]
    hits = prefix_blocks(request, blocks)
    active = {b.block_id for b in blocks if b.refs}
    protected = set(hits) - active
    available = instance.kv_blocks - len(active) - len(protected)
    prompt_needed = len(request.prompt) - len(hits)
    lifetime_needed = prompt_needed + len(request.output)
    reason = ("full_prompt_kv" if prompt_needed > available else
              "scripted_output_reservation" if lifetime_needed > available else None)
    return dict(request_id=request.request_id, cached_tokens=len(hits),
                hit_block_ids=list(hits), active_blocks=len(active),
                free_blocks=instance.kv_blocks - len(blocks),
                evictable_blocks=sum(not b.refs for b in blocks),
                protected_inactive_hit_blocks=len(protected), allocatable_blocks=available,
                missing_prompt_blocks=prompt_needed, output_reservation_blocks=len(request.output),
                required_new_blocks=lifetime_needed, fits=reason is None, failure_reason=reason)


def check_state(instance: Instance, state: State) -> None:
    """Assert physical capacity, ownership, identity, and scripted progress invariants."""
    assert len(state.blocks) <= instance.kv_blocks
    assert len(state.running) <= instance.max_sequences
    assert len(set(state.running)) == len(state.running)
    blocks = {b.block_id: b for b in state.blocks}
    assert len(blocks) == len(state.blocks)
    for i, (request, p) in enumerate(zip(instance.requests, state.progress)):
        assert 0 <= p.prefill_done <= len(request.prompt)
        assert 0 <= len(p.output_times) <= len(request.output)
        assert all(a < b for a, b in zip(p.output_times, p.output_times[1:]))
        if p.admitted_at is None:
            assert not p.block_ids and not p.output_times and i not in state.running
            continue
        assert p.admitted_at >= request.arrival
        assert len(p.block_ids) == len(request.prompt) + len(request.output)
        assert len(set(p.block_ids)) == len(p.block_ids)
        if p.completed_at is not None:
            assert i not in state.running and len(p.output_times) == len(request.output)
            assert p.completed_at == p.output_times[-1]
        else:
            assert i in state.running
            for pos, block_id in enumerate(p.block_ids):
                block = blocks[block_id]
                assert i in block.refs
                if pos < p.prefill_done:
                    assert block.computed and block.identity == request.prompt[:pos + 1]
                elif pos < len(request.prompt):
                    assert not block.computed and block.identity == request.prompt[:pos + 1]
                elif pos - len(request.prompt) < len(p.output_times):
                    assert block.computed
                    assert block.identity == request.prompt + request.output[:pos - len(request.prompt) + 1]
                else:
                    assert not block.computed and block.identity is None
    for block in state.blocks:
        assert len(set(block.refs)) == len(block.refs)
        for index in block.refs:
            assert index in state.running and block.block_id in state.progress[index].block_ids
        if not block.refs:
            assert block.computed


def transition(instance: Instance, state: State, order: tuple[int, ...]) -> tuple[State, dict]:
    """One fixed engine step. Only `order` is controlled by the Oracle."""
    waiting = eligible(instance, state)
    if len(order) != len(waiting) or set(order) != set(waiting):
        raise ValueError("Action must permute exactly the currently eligible waiting requests")
    if finished(state):
        raise ValueError("Cannot advance a complete workload")
    blocks = {b.block_id: b for b in state.blocks}
    progress = list(state.progress)
    running = list(state.running)
    clock, next_id = state.touch_clock, state.next_block_id
    budget = instance.token_budget
    services: list[tuple[int, str, int]] = []
    admissions, evictions = [], []
    stop = None

    def plan(index: int) -> None:
        nonlocal budget
        request, p = instance.requests[index], progress[index]
        if budget == 0:
            return
        if p.prefill_done < len(request.prompt):
            count = min(instance.prefill_chunk, budget, len(request.prompt) - p.prefill_done)
            services.append((index, "prefill", count))
        else:
            count = 1
            services.append((index, "decode", count))
        budget -= count

    # Running service always has priority; completed work is released at iteration end.
    for index in running:
        plan(index)
    for position, index in enumerate(order):
        if len(running) == instance.max_sequences or budget == 0:
            stop = dict(reason="sequence_slots" if len(running) == instance.max_sequences
                        else "token_budget", evaluated_request=None,
                        unexamined=[instance.requests[j].request_id for j in order[position:]])
            break
        fit = fit_snapshot(instance, index, blocks.values())
        admissions.append(fit)
        if not fit["fits"]:
            stop = dict(reason=fit["failure_reason"], evaluated_request=instance.requests[index].request_id,
                        unexamined=[instance.requests[j].request_id for j in order[position + 1:]],
                        alternatives=[fit_snapshot(instance, j, blocks.values()) for j in order[position + 1:]])
            break
        request = instance.requests[index]
        ids = list(fit["hit_block_ids"])
        for block_id in ids:
            clock += 1
            block = blocks[block_id]
            blocks[block_id] = replace(block, refs=tuple(sorted(block.refs + (index,))), touched=clock)
        needed = fit["required_new_blocks"]
        while instance.kv_blocks - len(blocks) < needed:
            victim = min((b for b in blocks.values() if not b.refs),
                         key=lambda b: (b.touched, b.block_id))
            evictions.append(dict(block_id=victim.block_id, identity=list(victim.identity or ())))
            del blocks[victim.block_id]
        for offset in range(needed):
            position_in_prompt = len(ids)
            identity = request.prompt[:position_in_prompt + 1] if position_in_prompt < len(request.prompt) else None
            clock += 1
            blocks[next_id] = Block(next_id, identity, False, (index,), clock)
            ids.append(next_id)
            next_id += 1
        progress[index] = Progress(state.time, fit["cached_tokens"], tuple(ids), cached_tokens=fit["cached_tokens"])
        running.append(index)
        plan(index)

    if not services:
        future = [r.arrival for i, r in enumerate(instance.requests)
                  if progress[i].admitted_at is None and r.arrival > state.time]
        if running or waiting or not future:
            raise AssertionError("No legal progress: each request must fit alone")
        advanced = replace(state, time=min(future))
        return advanced, dict(start=state.time, end=advanced.time, action=[],
                              forced_idle=True, admissions=[], services=[], evictions=[], stop=None)

    # Parallel toy cost: overhead + largest per-request service cost in this batch.
    duration = instance.overhead_ticks + max(count * instance.prefill_token_ticks if kind == "prefill"
                                             else instance.decode_token_ticks
                                             for _, kind, count in services)
    end = state.time + duration
    for index, kind, count in services:
        request, p = instance.requests[index], progress[index]
        if kind == "prefill":
            for pos in range(p.prefill_done, p.prefill_done + count):
                block_id = p.block_ids[pos]
                blocks[block_id] = replace(blocks[block_id], computed=True)
            progress[index] = replace(p, prefill_done=p.prefill_done + count)
        else:
            output_count = len(p.output_times) + 1
            block_id = p.block_ids[len(request.prompt) + output_count - 1]
            blocks[block_id] = replace(blocks[block_id], computed=True,
                                       identity=request.prompt + request.output[:output_count])
            progress[index] = replace(p, output_times=p.output_times + (end,),
                                      completed_at=end if output_count == len(request.output) else None)
    for index in tuple(running):
        if progress[index].completed_at is None:
            continue
        running.remove(index)
        for block_id in progress[index].block_ids:
            clock += 1
            block = blocks[block_id]
            blocks[block_id] = replace(block, refs=tuple(j for j in block.refs if j != index), touched=clock)
    advanced = State(end, tuple(progress), tuple(running),
                     tuple(sorted(blocks.values(), key=lambda b: b.block_id)), clock, next_id)
    check_state(instance, advanced)
    evidence = dict(start=state.time, end=end,
                    action=[instance.requests[i].request_id for i in order], forced_idle=False,
                    admissions=admissions, evictions=evictions, stop=stop,
                    services=[dict(request_id=instance.requests[i].request_id, kind=k, tokens=n)
                              for i, k, n in services],
                    physical_blocks=len(blocks), active_blocks=sum(bool(b.refs) for b in blocks.values()),
                    shared_active_blocks=sum(len(b.refs) > 1 for b in blocks.values()),
                    block_state=[asdict(b) for b in advanced.blocks])
    return advanced, evidence


def replay(instance: Instance, actions: tuple[tuple[int, ...], ...] | None = None) -> tuple[State, list[dict]]:
    state, events = initial_state(instance), []
    # With no preemption, at least one scripted token advances each service step.
    step_limit = sum(len(r.prompt) + len(r.output) for r in instance.requests) + len(instance.requests)
    while not finished(state):
        if len(events) >= step_limit:
            raise AssertionError("Finite service bound exceeded")
        action = eligible(instance, state) if actions is None else actions[len(events)]
        state, event = transition(instance, state, action)
        events.append(event)
    if actions is not None and len(events) != len(actions):
        raise ValueError("Action script has unused decisions")
    return state, events


def objective(instance: Instance, state: State) -> int:
    if not finished(state):
        raise ValueError("Objective requires every request to finish")
    return sum(p.output_times[0] - r.arrival for r, p in zip(instance.requests, state.progress))


@dataclass(frozen=True)
class Limits:
    completion: tuple[Fraction, ...]
    decode_duration: tuple[Fraction, ...]
    token_gap: tuple[Fraction, ...]


def progress_limits(instance: Instance, baseline: State) -> Limits:
    def allowed(value: int) -> Fraction:
        return Fraction(value) + max(Fraction(value * instance.slack_percent, 100),
                                     Fraction(instance.epsilon_ticks))
    completion, duration, gaps = [], [], []
    for request, p in zip(instance.requests, baseline.progress):
        assert p.completed_at is not None
        completion.append(Fraction(request.arrival) + allowed(p.completed_at - request.arrival))
        duration.append(allowed(p.completed_at - p.output_times[0]))
        gaps.append(allowed(max((b - a for a, b in zip(p.output_times, p.output_times[1:])), default=0)))
    return Limits(tuple(completion), tuple(duration), tuple(gaps))


def violations(instance: Instance, state: State, limits: Limits) -> list[str]:
    problems = []
    for i, (request, p) in enumerate(zip(instance.requests, state.progress)):
        completion = state.time if p.completed_at is None else p.completed_at
        if completion > limits.completion[i]:
            problems.append(f"{request.request_id}:session_completion")
        if len(request.output) > 1 and p.output_times:
            if completion - p.output_times[0] > limits.decode_duration[i]:
                problems.append(f"{request.request_id}:decode_duration")
            gaps = [b - a for a, b in zip(p.output_times, p.output_times[1:])]
            if p.completed_at is None:
                gaps.append(state.time - p.output_times[-1])
            if max(gaps, default=0) > limits.token_gap[i]:
                problems.append(f"{request.request_id}:inter_token_gap")
    return problems


@dataclass
class SearchResult:
    baseline: State
    best: State
    actions: tuple[tuple[int, ...], ...]
    limits: Limits
    exact: bool
    visited_states: int
    enumerated_actions: int
    duplicate_states: int
    constraint_pruned: int
    completed_states: int
    termination_reason: str

    def certificate(self, instance: Instance) -> dict:
        incumbent = objective(instance, self.best)
        baseline = objective(instance, self.baseline)
        lower = incumbent if self.exact else 0
        return dict(status="exact_restricted_toy_optimum" if self.exact else "offline_search_benchmark",
                    cost_revision="toy_parallel_max_v1", units="integer toy ticks (not seconds)",
                    objective="sum of first-token latency; divide by request count for mean",
                    baseline_sum_ttft_ticks=baseline, incumbent_sum_ttft_ticks=incumbent,
                    lower_bound_sum_ttft_ticks=lower, optimality_gap_ticks=incumbent - lower,
                    regret_mean_ttft_ticks=float(Fraction(baseline - incumbent, len(instance.requests))),
                    relative_regret=float(Fraction(baseline - incumbent, baseline)),
                    visited_states=self.visited_states, enumerated_actions=self.enumerated_actions,
                    duplicate_states=self.duplicate_states, constraint_pruned=self.constraint_pruned,
                    completed_states=self.completed_states, termination_reason=self.termination_reason,
                    baseline_included_as_feasible_candidate=True,
                    constraint_limits={k: [str(x) for x in v] for k, v in asdict(self.limits).items()},
                    horizon_ticks=str(max(self.limits.completion)),
                    best_waiting_actions=[[instance.requests[i].request_id for i in action] for action in self.actions])


def solve(instance: Instance, max_states: int = 200_000) -> SearchResult:
    """Exhaust all reachable legal orders; a state limit explicitly loses certification."""
    if type(max_states) is not int or max_states <= 0:
        raise ValueError("max_states must be positive")
    baseline, baseline_events = replay(instance)
    lookup = {r.request_id: i for i, r in enumerate(instance.requests)}
    baseline_actions = tuple(tuple(lookup[r] for r in e["action"]) for e in baseline_events)
    limits = progress_limits(instance, baseline)
    assert not violations(instance, baseline, limits)
    result = SearchResult(baseline, baseline, baseline_actions, limits, True, 0, 0, 0, 0, 0,
                          "exhausted_all_reachable_legal_waiting_orders")
    seen: set[State] = set()

    def visit(state: State, actions: tuple[tuple[int, ...], ...]) -> None:
        if not result.exact:
            return
        if violations(instance, state, limits):
            result.constraint_pruned += 1
            return
        if state in seen:
            result.duplicate_states += 1
            return
        if len(seen) >= max_states:
            result.exact = False
            result.termination_reason = "state_limit_reached; no optimality certificate"
            return
        seen.add(state)
        result.visited_states += 1
        if finished(state):
            result.completed_states += 1
            if objective(instance, state) < objective(instance, result.best):
                result.best, result.actions = state, actions
            return
        for action in permutations(eligible(instance, state)):
            result.enumerated_actions += 1
            next_state, _ = transition(instance, state, action)
            visit(next_state, actions + (action,))
            if not result.exact:
                break

    visit(initial_state(instance), ())
    assert objective(instance, result.best) <= objective(instance, baseline)
    assert not violations(instance, result.best, limits)
    return result


def _quantile(values: list[int], fraction: float) -> float:
    """Linear order-statistic interpolation (small-n descriptive values only)."""
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def summarize(instance: Instance, state: State, events: list[dict], limits: Limits) -> dict:
    assert finished(state)
    ttft, all_gaps, rows, no_service_gaps = [], [], [], []
    bypass = [0] * len(instance.requests)
    id_to_index = {r.request_id: i for i, r in enumerate(instance.requests)}
    for event in events:
        waiting = set(id_to_index[r] for r in event["action"])
        for admission in event["admissions"]:
            if not admission["fits"]:
                continue
            index = id_to_index[admission["request_id"]]
            for other in waiting:
                if (instance.requests[other].arrival, other) < (instance.requests[index].arrival, index):
                    bypass[other] += 1
            waiting.remove(index)
    for i, (r, p) in enumerate(zip(instance.requests, state.progress)):
        first, completion = p.output_times[0], p.output_times[-1]
        gap = [b - a for a, b in zip(p.output_times, p.output_times[1:])]
        all_gaps.extend(gap)
        ttft.append(first - r.arrival)
        previous_end = r.arrival
        for event in events:
            if any(s["request_id"] == r.request_id for s in event["services"]):
                no_service_gaps.append(event["start"] - previous_end)
                previous_end = event["end"]
        rows.append(dict(request_id=r.request_id, session_id=r.request_id, arrival=r.arrival,
                         admitted_at=p.admitted_at, first_token=first, completion=completion,
                         ttft_ticks=first - r.arrival, waiting_ticks=p.admitted_at - r.arrival,
                         response_latency_ticks=completion - r.arrival,
                         session_completion_delay_ticks=completion - r.arrival,
                         session_progress_ticks=completion - r.arrival,
                         first_token_from_session_start_ticks=first - r.arrival,
                         decode_duration_ticks=completion - first, output_times=list(p.output_times),
                         inter_token_gaps_ticks=gap, max_inter_token_gap_ticks=max(gap, default=0),
                         time_per_output_token_ticks=(completion - first) / (len(r.output) - 1) if len(r.output) > 1 else None,
                         prompt_tokens=len(r.prompt), output_tokens=len(r.output), cached_tokens=p.cached_tokens,
                         cached_fraction=p.cached_tokens / len(r.prompt), bypass_count=bypass[i]))
    makespan = state.time - min(r.arrival for r in instance.requests)
    output_tokens = sum(len(r.output) for r in instance.requests)
    return dict(units="toy ticks", request_count=len(rows), completed_requests=len(rows), unfinished_requests=0,
                completed_output_tokens=output_tokens, mean_ttft_ticks=sum(ttft) / len(ttft),
                median_ttft_ticks=_quantile(ttft, .5), p95_ttft_ticks=_quantile(ttft, .95), max_ttft_ticks=max(ttft),
                p99_ttft_ticks=None, p99_note="Not reported for four-request instances",
                mean_inter_token_gap_ticks=sum(all_gaps) / len(all_gaps) if all_gaps else None,
                p95_inter_token_gap_ticks=_quantile(all_gaps, .95) if all_gaps else None,
                max_waiting_age_ticks=max(row["waiting_ticks"] for row in rows),
                max_time_without_service_ticks=max(no_service_gaps, default=0),
                makespan_ticks=makespan, drain_after_last_root_ticks=state.time - max(r.arrival for r in instance.requests),
                completed_requests_per_tick=len(rows) / makespan, completed_output_tokens_per_tick=output_tokens / makespan,
                offered_root_rate_per_tick=None, root_arrivals=[r.arrival for r in instance.requests],
                realized_turn_rate_per_tick=len(rows) / makespan,
                rate_note="Finite roots; explicit arrivals replace a fitted offered rate. Throughput includes full drain.",
                cached_fraction=sum(p.cached_tokens for p in state.progress) / sum(len(r.prompt) for r in instance.requests),
                preemptions=0, recomputed_tokens=0, constraint_violations=violations(instance, state, limits),
                allocation_failures=[dict(time=e["start"], **a) for e in events for a in e["admissions"] if not a["fits"]],
                per_request=rows)


def standard_instances() -> tuple[Instance, ...]:
    """Hand-constructed mechanism checks, not sampled production workloads."""
    return (
        Instance("no_sharing_control", tuple(Request(chr(65 + i), (10*i + 1, 10*i + 2), (10*i + 3, 10*i + 4))
                                             for i in range(4)), kv_blocks=16, token_budget=4),
        Instance("sufficient_capacity", (
            Request("A", (1, 2, 3, 4), (5, 6)), Request("B", (11,), (12, 13)),
            Request("C", (21,), (22, 23)), Request("D", (31,), (32, 33))),
            kv_blocks=32, max_sequences=4, token_budget=4),
        Instance("sharing_with_eviction", (
            Request("seed", (1, 2, 3), (4, 5)), Request("unrelated", (10, 11, 12, 13), (14,), 9),
            Request("reuse_a", (1, 2, 3, 4, 5, 6), (7,), 9),
            Request("reuse_b", (1, 2, 3, 4, 5, 8), (9,), 9)),
            kv_blocks=8, token_budget=3),
        Instance("full_prompt_head_failure", (
            Request("running", (1, 2, 3), (4, 5, 6, 7)),
            Request("large_head", (11, 12, 13, 14, 15, 16), (17,)),
            Request("small_a", (21,), (22,)), Request("small_b", (31,), (32,))),
            kv_blocks=10, max_sequences=3, token_budget=3),
    )
