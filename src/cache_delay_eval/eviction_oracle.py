"""Bounded eviction-only search for an explicit full-attention structural model.

FCFS, running-first chunked prefill, dynamic decode allocation, and tail
preemption are fixed across policies. This is not a vLLM equivalence claim.
Inputs are independent roots: child releases and unknown output-prefix sharing
must not be inferred from an anonymized trace. Costs require A30 calibration.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from itertools import permutations
import json
import math
from typing import Any


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival: int
    input_tokens: int
    output_tokens: int
    # Block-content IDs, NOT prefix-chain IDs. Only full blocks can be cached.
    hashes: tuple[int, ...]


@dataclass(frozen=True)
class Instance:
    requests: tuple[Request, ...]
    capacity: int
    block_size: int = 16
    max_sequences: int = 2
    token_budget: int = 32
    prefill_chunk: int = 32
    overhead: int = 100
    prefill_token_cost: int = 100
    decode_token_cost: int = 1000
    slack_percent: int = 5
    epsilon: int = 1
    # Free physical IDs are 0..capacity-1; unlike vLLM the null block is excluded.
    initial_cache: tuple[tuple[int, ...] | None, ...] = ()
    initial_free_order: tuple[int, ...] = ()
    cost_revision: str = 'uncalibrated_affine_v1'
    cost_table: dict[str, int] | None = None

    def validate(self):
        if not 1 <= len(self.requests) <= 8:
            raise ValueError('Only 1..8 independent roots are supported')
        if len({r.request_id for r in self.requests}) != len(self.requests):
            raise ValueError('Duplicate request IDs')
        for k in ('capacity', 'block_size', 'max_sequences', 'token_budget', 'prefill_chunk',
                  'overhead', 'prefill_token_cost', 'decode_token_cost'):
            if type(getattr(self, k)) is not int or getattr(self, k) <= 0:
                raise ValueError(f'{k} must be a positive integer')
        if any(type(x) is not int or x < 0 for x in (self.slack_percent, self.epsilon)):
            raise ValueError('Nonnegative integer progress tolerances required')
        for r in self.requests:
            if any(type(x) is not int or x < 0 for x in (r.arrival, r.input_tokens, r.output_tokens)):
                raise ValueError('Nonnegative integer demands and arrival ticks required')
            if not r.input_tokens or not r.output_tokens:
                raise ValueError('Empty prompt/output not supported')
            if len(r.hashes) != math.ceil(r.input_tokens / self.block_size):
                raise ValueError('Complete source block identities required')
            if math.ceil((r.input_tokens + r.output_tokens - 1) / self.block_size) > self.capacity:
                raise ValueError('Request must fit alone including decode growth')
        if self.initial_cache and len(self.initial_cache) != self.capacity:
            raise ValueError('Initial state must specify every physical block')
        if self.initial_free_order and sorted(self.initial_free_order) != list(range(self.capacity)):
            raise ValueError('Initial state supports quiescent, fully unreferenced cache only')
        if self.cost_table is not None and (not self.cost_table or any(type(v) is not int or v <= 0 for v in self.cost_table.values())):
            raise ValueError('Cost table entries must be positive integer ticks')


@dataclass
class Progress:
    computed: int = 0
    blocks: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    completed: int | None = None
    cached_tokens: int = 0
    preemptions: int = 0


@dataclass
class Block:
    identity: tuple[int, ...] | None = None
    refs: int = 0


class Choice(Exception):
    def __init__(self, candidates, count):
        self.candidates, self.count = tuple(candidates), count


class Infeasible(Exception):
    pass


def batch_key(services: list[dict]) -> str:
    return json.dumps([[s['kind'], s['tokens'], s['context']] for s in services], separators=(',', ':'))


def limits_for(instance: Instance, baseline: dict) -> list[dict]:
    def allowed(n):
        # Integer rounding DOWN makes the integer constraint equivalent to its
        # rational 5% bound, not a looser rounded deadline.
        return n + max(n * instance.slack_percent // 100, instance.epsilon)
    result = []
    for r, p in zip(instance.requests, baseline['progress']):
        result.append(dict(completion=r.arrival + allowed(p['completed'] - r.arrival),
                           duration=allowed(p['completed'] - p['outputs'][0]),
                           gap=allowed(max((b-a for a,b in zip(p['outputs'], p['outputs'][1:])), default=0))))
    return result


def replay(instance: Instance, actions: tuple[tuple[int, ...], ...] | None = None,
           limits: list[dict] | None = None) -> dict:
    """None uses stock queue; a finite action prefix stops at the next choice."""
    instance.validate()
    rs, bs = instance.requests, instance.block_size
    blocks = [Block(h) for h in (instance.initial_cache or (None,) * instance.capacity)]
    free = list(instance.initial_free_order or range(instance.capacity))
    progress = [Progress() for _ in rs]
    running, waiting, released = [], [], set()
    time, action_index, steps = 0, 0, 0
    events, used_actions = [], []
    # Baseline should make finite progress; a bound failure is rejected, never
    # treated as an Oracle result. Search uses baseline-derived deadlines.
    step_limit = 100 * sum(r.input_tokens + r.output_tokens for r in rs)

    def release(i):
        cached, uncached = [], []
        for bid in reversed(progress[i].blocks):
            blocks[bid].refs -= 1
            if blocks[bid].refs == 0:
                (cached if blocks[bid].identity is not None else uncached).append(bid)
        free[:0] = uncached
        free.extend(cached)
        progress[i].blocks = []

    def allocate(i, count):
        nonlocal action_index
        if count == 0:
            return
        if count > len(free):
            raise AssertionError('Allocation without fit check')
        stock = free[:count]
        candidates = [b for b in free if blocks[b].identity is not None]
        victim_count = sum(blocks[b].identity is not None for b in stock)
        choice_count = math.perm(len(candidates), victim_count)
        selected = tuple(b for b in stock if blocks[b].identity is not None)
        if choice_count > 1:
            if actions is not None:
                if action_index == len(actions):
                    raise Choice(candidates, victim_count)
                selected = actions[action_index]
                if len(selected) != victim_count or len(set(selected)) != victim_count or not set(selected) <= set(candidates):
                    raise ValueError('Illegal victim action')
            used_actions.append(selected)
            action_index += 1
        chosen = iter(selected)
        allocated = [next(chosen) if blocks[b].identity is not None else b for b in stock]
        records = [dict(block_id=b, identity=blocks[b].identity, refs=blocks[b].refs) for b in candidates]
        victims = [dict(block_id=b, identity=blocks[b].identity) for b in selected]
        for bid in allocated:
            assert blocks[bid].refs == 0
            free.remove(bid)
            blocks[bid] = Block(None, 1)
            progress[i].blocks.append(bid)
        events.append(dict(type='allocation', time=time, request_id=rs[i].request_id,
                           candidates=records, victim_count=victim_count, victims=victims,
                           allocated=allocated))

    def hits(i):
        found = []
        # Last prompt token must run to produce logits, even on an exact hit.
        for pos in range((rs[i].input_tokens - 1) // bs):
            key = rs[i].hashes[:pos + 1]
            bid = next((b for b, block in enumerate(blocks) if block.identity == key), None)
            if bid is None:
                break
            found.append(bid)
        return found

    def preempt(i):
        release(i)
        progress[i].computed = 0
        progress[i].preemptions += 1
        running.remove(i)
        waiting.insert(0, i)
        events.append(dict(type='preemption', time=time, request_id=rs[i].request_id))

    def validate_state():
        refs = [0] * instance.capacity
        for p in progress:
            for bid in p.blocks:
                refs[bid] += 1
        assert refs == [b.refs for b in blocks]
        assert len(free) == len(set(free))
        assert set(free) == {b for b, block in enumerate(blocks) if block.refs == 0}

    while any(p.completed is None for p in progress):
        steps += 1
        if limits is None and steps > step_limit:
            raise ValueError('Baseline iteration bound reached; no result or certificate')
        for i in sorted(range(len(rs)), key=lambda j: (rs[j].arrival, j)):
            if i not in released and rs[i].arrival <= time:
                released.add(i)
                waiting.append(i)
        budget, services, had_preemption = instance.token_budget, [], False
        # Fixed running-first order, rollback a scheduled tail on preemption.
        for i in list(running):
            if i not in running or budget == 0:
                continue
            p, r = progress[i], rs[i]
            target = r.input_tokens + len(p.outputs)
            n = min(budget, instance.prefill_chunk if p.computed < r.input_tokens else 1, target - p.computed)
            need = math.ceil((p.computed + n)/bs) - len(p.blocks)
            while need > len(free):
                victim = running[-1]
                previous = next((s for s in services if s['index'] == victim), None)
                if previous:
                    services.remove(previous)
                    budget += previous['tokens']
                preempt(victim)
                had_preemption = True
                if victim == i:
                    break
            if i not in running:
                continue
            allocate(i, need)
            services.append(dict(index=i, kind='prefill' if p.computed < r.input_tokens else 'decode', tokens=n, context=p.computed))
            budget -= n
        if not had_preemption:
            while waiting and budget and len(running) < instance.max_sequences:
                i = waiting[0]
                r, p = rs[i], progress[i]
                cached = hits(i)
                target = r.input_tokens + len(p.outputs)
                protected = sum(blocks[b].refs == 0 for b in cached)
                if math.ceil(target/bs) - len(cached) + protected > len(free):
                    break  # full-prompt gate; no bypass or Oracle idling
                for bid in cached:
                    if blocks[bid].refs == 0:
                        free.remove(bid)
                    blocks[bid].refs += 1
                p.blocks = cached
                p.computed = len(cached) * bs
                p.cached_tokens += p.computed
                n = min(budget, instance.prefill_chunk, target - p.computed)
                allocate(i, math.ceil((p.computed+n)/bs) - len(p.blocks))
                waiting.pop(0)
                running.append(i)
                services.append(dict(index=i, kind='prefill', tokens=n, context=p.computed))
                budget -= n
        if not services:
            future = [r.arrival for i,r in enumerate(rs) if i not in released]
            if running or waiting:
                # Tail preemption can consume an iteration but does no GPU work.
                # Retry immediately; repeated no-progress transitions hit bound.
                if had_preemption:
                    raise ValueError('Unsupported zero-service preemption cycle; no certificate')
                raise Infeasible('Fixed scheduler cannot make progress')
            if not future:
                raise Infeasible('No future work')
            time = min(future)
            continue
        signature = batch_key(services)
        if instance.cost_table is None:
            duration = instance.overhead + sum(s['tokens'] * (instance.prefill_token_cost if s['kind'] == 'prefill' else instance.decode_token_cost) for s in services)
        else:
            if signature not in instance.cost_table:
                raise ValueError(f'Missing calibrated batch cost: {signature}')
            duration = instance.cost_table[signature]
        end = time + duration
        events.append(dict(type='iteration', start=time, end=end, signature=signature,
                           services=[dict(s, request_id=rs[s['index']].request_id) for s in services]))
        for s in services:
            i = s['index']
            p, r = progress[i], rs[i]
            p.computed += s['tokens']
            for pos in range(min(p.computed, r.input_tokens)//bs):
                blocks[p.blocks[pos]].identity = r.hashes[:pos+1]
            target = r.input_tokens + len(p.outputs)
            if p.computed >= target:
                p.outputs.append(end)
                if len(p.outputs) == r.output_tokens:
                    p.completed = end
        for i in list(running):
            if progress[i].completed is not None:
                release(i)
                running.remove(i)
        time = end
        validate_state()
        if limits:
            for i, p in enumerate(progress):
                now = p.completed if p.completed is not None else time
                if now > limits[i]['completion']:
                    raise Infeasible('Completion deadline')
                if rs[i].output_tokens > 1 and p.outputs:
                    gaps = [b-a for a,b in zip(p.outputs, p.outputs[1:])]
                    if p.completed is None:
                        gaps.append(time-p.outputs[-1])
                    if now-p.outputs[0] > limits[i]['duration'] or max(gaps, default=0) > limits[i]['gap']:
                        raise Infeasible('Decode progress deadline')
    if actions is not None and action_index != len(actions):
        raise ValueError('Unused victim actions')
    ttft = [p.outputs[0]-r.arrival for p,r in zip(progress,rs)]
    return dict(sum_ttft=sum(ttft), mean_ttft=sum(ttft)/len(rs), ttft=ttft,
                progress=[asdict(p) for p in progress], events=events, actions=used_actions,
                final_cache=[asdict(b) for b in blocks], final_free_order=free, drain_time=time)


def solve(instance: Instance, max_nodes: int = 10000) -> dict[str, Any]:
    if type(max_nodes) is not int or max_nodes <= 0:
        raise ValueError('max_nodes must be positive')
    baseline = replay(instance)
    limits = limits_for(instance, baseline)
    # Ensure the exact baseline action script remains a feasible policy.
    replay(instance, tuple(baseline['actions']), limits)
    best, nodes, pruned, leaves, exact = baseline, 0, 0, 0, True
    def visit(actions):
        nonlocal best, nodes, pruned, leaves, exact
        if nodes >= max_nodes:
            exact = False
            return
        nodes += 1
        try:
            result = replay(instance, actions, limits)
        except Infeasible:
            pruned += 1
        except Choice as c:
            for choice in permutations(c.candidates, c.count):
                visit(actions + (choice,))
                if not exact:
                    break
        else:
            leaves += 1
            if result['sum_ttft'] < best['sum_ttft']:
                best = result
    visit(())
    lower = best['sum_ttft'] if exact else 0
    return dict(status='exact_structural_model_optimum' if exact else 'offline_search_benchmark',
                exact=exact, visited_nodes=nodes, pruned=pruned, completed_leaves=leaves,
                termination='all_legal_ordered_victims_exhausted' if exact else 'node_limit',
                lower_bound_sum_ttft_ticks=lower, incumbent_sum_ttft_ticks=best['sum_ttft'],
                optimality_gap_ticks=best['sum_ttft']-lower,
                mean_ttft_reduction_ticks=(baseline['sum_ttft']-best['sum_ttft'])/len(instance.requests),
                relative_reduction=(baseline['sum_ttft']-best['sum_ttft'])/baseline['sum_ttft'],
                limits=limits, cost_revision=instance.cost_revision, units='integer model ticks',
                baseline_included=True, hardware_validated=False, baseline=baseline, best=best)
