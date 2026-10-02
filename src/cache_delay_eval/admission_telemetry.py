"""CPU-testable passive admission telemetry for one pinned vLLM revision.

The preparation script copies this standard-library-only module into vLLM. Set
CACHE_DELAY_TELEMETRY_DIR to enable it. Files are per process; request mappings
from the API process must be joined with scheduler events after collection.
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

SCHEMA_VERSION = 2
VLLM_VERSION = "0.28.0"
VLLM_COMMIT = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
EVENT_TYPES = {
    "request_id_map", "engine_config", "enqueue", "step_start", "waiting_attempt",
    "allocation_check", "admitted", "preemption", "waiting_remaining", "step_end",
    "pool_initial", "allocation_begin", "allocation_end", "cache_remove",
    "cache_insert", "cache_lookup", "cache_touch", "cache_release", "cache_reset",
    "request_prefixes",
}
_writer: EventWriter | None = None
_state: dict[str, Any] = {}


class EventWriter:
    """Synchronous diagnostic writer. A write error invalidates the run.

    No logging queue can silently drop events. This intentionally has measurable
    overhead; compare enabled/disabled on the A30 before interpreting latency.
    """
    def __init__(self, directory: str | Path):
        self.pid = os.getpid()
        self.counter = 0
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"admission-{self.pid}.jsonl"
        self.file = self.path.open("a", buffering=1, encoding="utf-8")

    def write(self, kind: str, *, engine_request_id: str | None = None,
              request_id: str | None = None, step_id: int | None = None,
              data: dict[str, Any] | None = None) -> dict[str, Any]:
        self.counter += 1
        event = dict(schema_version=SCHEMA_VERSION, event_id=self.counter,
                     process_id=self.pid, monotonic_ns=time.monotonic_ns(),
                     unix_time_ns=time.time_ns(), type=kind, step_id=step_id,
                     engine_request_id=engine_request_id, request_id=request_id,
                     join_status="mapped" if request_id else (
                         "pending_map" if engine_request_id else "not_applicable"),
                     data=data or {})
        validate_event(event)
        self.file.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        return event

    def close(self) -> None:
        self.file.close()


def validate_event(event: dict[str, Any]) -> None:
    """Validate the versioned event envelope without external dependencies."""
    required = {"schema_version", "event_id", "process_id", "monotonic_ns", "unix_time_ns",
                "type", "step_id", "engine_request_id", "request_id", "join_status", "data"}
    if set(event) != required or event["schema_version"] != SCHEMA_VERSION:
        raise ValueError("invalid event envelope or schema version")
    for key in ("event_id", "process_id", "monotonic_ns", "unix_time_ns"):
        if type(event[key]) is not int or event[key] < 0:
            raise ValueError(f"invalid {key}")
    if event["type"] not in EVENT_TYPES or not isinstance(event["data"], dict):
        raise ValueError("invalid event type/data")
    if event["step_id"] is not None and (type(event["step_id"]) is not int or event["step_id"] < 0):
        raise ValueError("invalid step_id")
    for key in ("engine_request_id", "request_id"):
        if event[key] is not None and not isinstance(event[key], str):
            raise ValueError(f"invalid {key}")
    if event["join_status"] not in {"mapped", "pending_map", "not_applicable", "unmatched", "ambiguous"}:
        raise ValueError("invalid join_status")
    if event["type"] == "allocation_check":
        data = event["data"]
        if data.get("gate") not in {"full_sequence", "chunk"} or type(data.get("passed")) is not bool:
            raise ValueError("allocation_check requires a gate and a boolean outcome")
        for key in ("required_blocks", "available_blocks", "proposed_tokens", "local_cached_tokens"):
            if type(data.get(key)) is not int:
                raise ValueError(f"allocation_check missing integer {key}")


def _get_writer() -> EventWriter | None:
    global _writer, _state
    directory = os.environ.get("CACHE_DELAY_TELEMETRY_DIR")
    if not directory:
        return None
    if _writer is None or _writer.pid != os.getpid():
        # Do not import vLLM (which could initialize CUDA). Distribution metadata
        # plus the source checks in the preparation script establish the pin.
        version = importlib.metadata.version("vllm")
        if version != VLLM_VERSION:
            raise RuntimeError(f"telemetry requires vLLM {VLLM_VERSION}, got {version}")
        _writer = EventWriter(directory)
        _state = {}
    return _writer


def emit(kind: str, request: Any = None, **data: Any) -> None:
    writer = _get_writer()
    if writer is not None:
        writer.write(kind, engine_request_id=getattr(request, "request_id", None),
                     step_id=_state.get("step_id"), data=data)


def request_id_map(request: Any) -> None:
    writer = _get_writer()
    if writer is None:
        return
    external = request.external_req_id
    # This is an exact map at the assignment site, never random-suffix guessing.
    client_id = None
    rule = "unsupported_external_id"
    if external.startswith("cmpl-") and external.endswith("-0"):
        client_id = external[len("cmpl-"):-2]
        rule = "strip_one_cmpl_prefix_and_prompt_index_zero_from_explicit_external_req_id"
    elif external.startswith("chatcmpl-"):
        client_id = external[len("chatcmpl-"):]
        rule = "strip_one_chatcmpl_prefix_from_explicit_external_req_id"
    writer.write("request_id_map", engine_request_id=request.request_id,
                 request_id=client_id, data={"external_request_id": external,
                 "normalization_rule": rule,
                 "supported_scope": "completion/chat completion; n=1; one prompt; no beam search"})


def _json_config(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_config({f.name: getattr(value, f.name) for f in dataclasses.fields(value)})
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_config(v) for v in value]
    return {"unavailable_type": type(value).__qualname__}


def pool_snapshot(pool: Any) -> dict[str, Any]:
    """Read block metadata only; never call cache lookup, touch, allocate or free."""
    blocks = [b for b in pool.blocks if not b.is_null]
    cached_aliases = pool.cached_block_hashes_by_block
    return {"total_usable_blocks": len(blocks), "free_blocks": pool.get_num_free_blocks(),
            "active_blocks": sum(b.ref_cnt > 0 for b in blocks),
            "evictable_cached_blocks": sum(b.ref_cnt == 0 and (
                b.block_hash is not None or bool(cached_aliases.get(b.block_id))) for b in blocks),
            "snapshot_scope": "physical block pool; null block excluded"}


def step_start(scheduler: Any) -> None:
    if _get_writer() is None:
        return
    if not _state.get("configured"):
        emit("engine_config", version=VLLM_VERSION, source_commit=VLLM_COMMIT,
             scheduler=_json_config(scheduler.scheduler_config),
             cache=_json_config(scheduler.cache_config),
             parallel=_json_config(scheduler.parallel_config),
             async_scheduling=getattr(scheduler.scheduler_config, "async_scheduling", None),
             unavailable=["counterfactual_tail_feasibility",
                          "server_first_token_timestamp", "GPU_execution_time"])
        _state["configured"] = True
    _state.update(step_id=scheduler.current_step, attempted=set(), stop_gate=None,
                  last_failed_engine_request_id=None, start_ns=time.monotonic_ns())
    emit("step_start", running_ids=[r.request_id for r in scheduler.running],
         waiting_ids=[r.request_id for r in scheduler.waiting],
         skipped_waiting_ids=[r.request_id for r in scheduler.skipped_waiting],
         initial_token_budget=scheduler.max_num_scheduled_tokens,
         initial_input_budget=scheduler.scheduler_config.max_num_batched_tokens,
         max_running_requests=scheduler.max_num_running_reqs,
         pool=pool_snapshot(scheduler.kv_cache_manager.block_pool))


def waiting_attempt(scheduler: Any, request: Any, token_budget: int, input_budget: int) -> None:
    if _get_writer() is None:
        return
    _state["attempted"].add(request.request_id)
    emit("waiting_attempt", request, token_budget=token_budget, input_budget=input_budget,
         running_count=len(scheduler.running), request_status=str(request.status),
         num_computed_tokens=request.num_computed_tokens,
         potential_cached_tokens_at_enqueue=None, counterfactual_tail_feasibility=None)


def gate(reason: str | None) -> None:
    if _get_writer() is not None:
        _state["stop_gate"] = reason


def allocation_check(manager: Any, request: Any, scope: str, required_blocks: int,
                     available_blocks: int, proposed_tokens: int, local_cached_tokens: int,
                     watermark_blocks: int, reserved_blocks: int) -> None:
    if _get_writer() is None:
        return
    passed = required_blocks <= available_blocks
    reason = "passed" if passed else ("full_sequence_kv_fit_failed" if scope == "full_sequence"
                                      else "chunk_kv_allocation_failed")
    if not passed:
        _state["last_failed_engine_request_id"] = request.request_id
        if request.request_id in _state.get("attempted", set()):
            _state["stop_gate"] = reason
    emit("allocation_check", request, gate=scope, passed=passed, reason=reason,
         required_blocks=required_blocks, available_blocks=available_blocks,
         free_blocks=manager.block_pool.get_num_free_blocks(), proposed_tokens=proposed_tokens,
         local_cached_tokens=local_cached_tokens, watermark_blocks=watermark_blocks,
         reserved_blocks=reserved_blocks,
         other_gate_required_blocks=None)


def preemption(scheduler: Any, request: Any) -> None:
    if _get_writer() is None:
        return
    emit("preemption", request, trigger_engine_request_id=_state.get("last_failed_engine_request_id"),
         computed_tokens_before_reset=request.num_computed_tokens,
         free_blocks_before=scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
         actual_recomputed_tokens=None, remaining_output_tokens=None)


def waiting_end(scheduler: Any, token_budget: int, input_budget: int,
                draft_slots: int, had_preemption: bool, paused: bool,
                scheduled_tokens: dict[str, int]) -> None:
    if _get_writer() is None:
        return
    reason = _state.get("stop_gate")
    if had_preemption:
        reason = "preemption_this_step"
    elif paused:
        reason = "scheduler_paused"
    elif token_budget <= 0:
        reason = "token_budget_exhausted"
    elif reason is None and input_budget <= draft_slots:
        reason = "input_budget_exhausted"
    for request in list(scheduler.waiting) + list(scheduler.skipped_waiting):
        evaluated = request.request_id in _state.get("attempted", set())
        emit("waiting_remaining", request, evaluated_in_step=evaluated,
             stopping_gate=reason or "unavailable_other_gate_or_blocked_status",
             request_gate_evaluated=evaluated,
             allocation_feasible=None,
             meaning="examined_this_step" if evaluated else "not_evaluated_this_step")
    emit("step_end", remaining_token_budget=token_budget, remaining_input_budget=input_budget,
         stopping_gate=reason, scheduled_tokens_by_engine_id=dict(scheduled_tokens),
         running_count=len(scheduler.running), waiting_count=len(scheduler.waiting),
         skipped_waiting_count=len(scheduler.skipped_waiting),
         admission_section_elapsed_ns=time.monotonic_ns() - _state["start_ns"],
         pool=pool_snapshot(scheduler.kv_cache_manager.block_pool))


def resolve_events(events: Iterable[dict[str, Any]], client_request_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Exact, order-independent join. Ambiguous and unmatched IDs remain null."""
    events = list(events)
    clients = set(client_request_ids)
    mappings: dict[str, set[str]] = {}
    for event in events:
        validate_event(event)
        if event["type"] == "request_id_map" and event["request_id"] is not None:
            mappings.setdefault(event["engine_request_id"], set()).add(event["request_id"])
    resolved = []
    for event in events:
        copy = dict(event)
        engine_id = copy["engine_request_id"]
        if engine_id is not None:
            candidates = mappings.get(engine_id, set())
            if len(candidates) > 1:
                copy.update(request_id=None, join_status="ambiguous")
            elif len(candidates) == 1 and next(iter(candidates)) in clients:
                copy.update(request_id=next(iter(candidates)), join_status="mapped")
            else:
                copy.update(request_id=None, join_status="unmatched")
        resolved.append(copy)
    return resolved


def request_scope(request: Any = None) -> None:
    if _get_writer() is not None:
        _state['request'] = request
        if request is not None:
            emit('request_prefixes', request, prefix_hashes=[h.hex() for h in request.block_hashes],
                 num_tokens=request.num_tokens)


def block_record(pool: Any, block: Any) -> dict[str, Any]:
    keys = set(pool.cached_block_hashes_by_block.get(block.block_id, ()))
    if block.block_hash is not None:
        keys.add(block.block_hash)
    return dict(block_id=block.block_id, ref_cnt=block.ref_cnt, is_null=block.is_null,
                hashes=[dict(key=h.hex(), prefix_hash=h[:-4].hex(),
                             group_id=int.from_bytes(h[-4:], 'big')) for h in sorted(keys)])


def pool_initial(pool: Any) -> None:
    if _get_writer() is None:
        return
    emit('pool_initial', blocks=[block_record(pool, b) for b in pool.blocks],
         free_order=[b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
         enable_caching=pool.enable_caching, hash_block_size=pool.hash_block_size)


def allocation_begin(pool: Any, count: int) -> None:
    writer = _get_writer()
    if writer is None:
        return
    queue = pool.free_block_queue.get_all_free_blocks()
    records = [block_record(pool, b) for b in queue]
    selected = records[:count]
    decision = f'{writer.pid}:{writer.counter + 1}'
    _state['allocation_id'] = decision
    emit('allocation_begin', _state.get('request'), allocation_id=decision,
         requested_blocks=count, free_order=records,
         candidates=[r for r in records if r['hashes'] and r['ref_cnt'] == 0 and not r['is_null']],
         protected=[block_record(pool, b) for b in pool.blocks if b.ref_cnt > 0 or b.is_null],
         selected_block_ids=[r['block_id'] for r in selected],
         required_victim_count=sum(bool(r['hashes']) for r in selected) if pool.enable_caching else 0,
         rule='stock free queue; branch only on cached victim slots; preserve uncached slots')


def allocation_end(pool: Any, blocks: Any) -> None:
    if _get_writer() is None:
        return
    emit('allocation_end', _state.get('request'), allocation_id=_state.get('allocation_id'),
         blocks=[block_record(pool, b) for b in blocks])
    _state.pop('allocation_id', None)


def cache_remove(pool: Any, block: Any, removed: Any) -> None:
    if _get_writer() is None or not removed:
        return
    # Called after successful hash-map removal; also captures promotion/move.
    emit('cache_remove', _state.get('request'), allocation_id=_state.get('allocation_id'),
         block_id=block.block_id, ref_cnt=block.ref_cnt, is_null=block.is_null,
         removed=[dict(key=h.hex(), prefix_hash=h[:-4].hex(),
                       group_id=int.from_bytes(h[-4:], 'big')) for h in removed],
         cause='allocation' if _state.get('allocation_id') else 'metadata_removal')


def cache_insert(pool: Any, block: Any, key: bytes) -> None:
    if _get_writer() is not None:
        emit('cache_insert', _state.get('request'), block=block_record(pool, block), key=key.hex())


def cache_lookup(pool: Any, key: bytes, block: Any) -> None:
    if _get_writer() is not None:
        emit('cache_lookup', _state.get('request'), key=key.hex(),
             block=block_record(pool, block) if block is not None else None)


def cache_reference(pool: Any, block: Any, kind: str) -> None:
    if _get_writer() is not None:
        emit(kind, _state.get('request'), block=block_record(pool, block),
             phase='before_reference_increment' if kind == 'cache_touch' else 'after_reference_decrement')


def reuse_report(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Follow each removed alias until the recorded process boundary.

    A demand behind a missing prefix is distinct from an attempted lookup, and a
    lookup is distinct from a touch (successful admission). No future observation
    means right-censored, never 'will not be reused'. Duplicate owners may hit.
    """
    events = list(events)
    for e in events:
        validate_event(e)
    if len({(e['process_id'], e['event_id']) for e in events}) != len(events):
        raise ValueError('duplicate process/event IDs')
    result = []
    pending: dict[tuple[int, str], list[dict]] = {}
    for e in sorted(events, key=lambda e: (e['process_id'], e['event_id'])):
        d, pid = e['data'], e['process_id']
        if e['type'] == 'cache_remove' and d['cause'] == 'allocation':
            for h in d['removed']:
                row = dict(process_id=pid, eviction_event_id=e['event_id'],
                           allocation_id=d['allocation_id'], block_id=d['block_id'], **h,
                           first_demand=None, first_lookup=None, first_touch=None,
                           observation='right_censored')
                pending.setdefault((pid, h['prefix_hash']), []).append(row)
                result.append(row)
        elif e['type'] == 'request_prefixes':
            for h in set(d['prefix_hashes']):
                for row in pending.get((pid, h), []):
                    if row['first_demand'] is None:
                        row['first_demand'] = dict(event_id=e['event_id'], engine_request_id=e['engine_request_id'])
                        row['observation'] = 'demand_observed'
        elif e['type'] == 'cache_lookup':
            for row in pending.get((pid, d['key'][:-8]), []):
                if row['key'] == d['key'] and row['first_lookup'] is None:
                    row['first_lookup'] = dict(event_id=e['event_id'], hit=d['block'] is not None,
                                              block_id=d['block']['block_id'] if d['block'] else None)
        elif e['type'] == 'cache_touch':
            for h in d['block']['hashes']:
                for row in pending.get((pid, h['prefix_hash']), []):
                    if row['key'] == h['key'] and row['first_touch'] is None:
                        row['first_touch'] = dict(event_id=e['event_id'], block_id=d['block']['block_id'])
    return result
