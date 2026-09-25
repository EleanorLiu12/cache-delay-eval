"""Open-loop concurrent HTTP replay with per-request timing and cache usage."""
import asyncio
import hashlib
import json
import math
from pathlib import Path

import aiohttp

from .trace import read_trace


async def completion(session, base_url, payload, encoded_payload=None):
    loop = asyncio.get_running_loop()
    start = loop.time()
    first = None
    usage = {}
    finish = None
    done = False
    body = encoded_payload if encoded_payload is not None else json.dumps(payload).encode()
    async with session.post(base_url + "/v1/completions", data=body,
                            headers={"Content-Type": "application/json"}) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:1000]}")
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            for choice in event.get("choices", []):
                # Token IDs avoid delaying TTFT until detokenization emits text.
                if first is None and choice.get("token_ids"):
                    first = loop.time()
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
    if not done or first is None or finish is None:
        raise RuntimeError("incomplete stream or missing token_ids; this server must support return_token_ids")
    if usage.get("prompt_tokens") != len(payload["prompt"]):
        raise RuntimeError("prompt token count missing or changed by server")
    if usage.get("completion_tokens") != payload["max_tokens"]:
        raise RuntimeError("output length missing or differs from fixed workload")
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is not None and (type(cached) is not int or not 0 <= cached <= len(payload["prompt"])):
        raise RuntimeError("invalid cached token count")
    return dict(ttft_ms=(first - start) * 1000,
                e2e_ms=(loop.time() - start) * 1000,
                usage=usage, cached_tokens=cached, finish_reason=finish)


def payload_for(request, model, seed):
    return dict(model=model, request_id=request.request_id,
                prompt=list(request.token_ids), max_tokens=request.output_tokens,
                temperature=0, seed=seed, ignore_eos=True, min_tokens=request.output_tokens,
                stream=True, stream_options={"include_usage": True},
                return_token_ids=True, add_special_tokens=False)


async def replay(trace_path, base_url, model, output, run_meta,
                 arrival_scale=1.0, timeout_s=600, max_dispatch_lag_ms=50):
    if not math.isfinite(arrival_scale) or arrival_scale <= 0:
        raise ValueError("arrival_scale must be finite and positive")
    header, requests = read_trace(trace_path)
    if any(r.token_ids is None for r in requests) or header.model_id != model:
        raise ValueError("need a materialized trace for the selected model")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_digest = hashlib.sha256(Path(trace_path).read_bytes()).hexdigest()
    loop = asyncio.get_running_loop()
    # Serialize payloads before the timed schedule, avoiding CPU work at arrival.
    payloads = [payload_for(r, model, 699 + i) for i, r in enumerate(requests)]
    encoded_payloads = [json.dumps(p).encode() for p in payloads]
    with output.open("x") as stream:
        meta = dict(run_meta, type="run_meta", trace_sha256=trace_digest, model=model,
                    arrival_scale=arrival_scale, max_dispatch_lag_ms=max_dispatch_lag_ms,
                    requests=len(requests), workload="open-loop opaque synthetic prompts",
                    ttft_definition="client time to first streamed generated token ID")
        stream.write(json.dumps(meta) + "\n")
        stream.flush()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s),
                                          connector=aiohttp.TCPConnector(limit=0)) as session:
            epoch = loop.time() + .1

            async def send(index, request):
                scheduled_ms = request.arrival_time_ms * arrival_scale
                await asyncio.sleep(max(0, epoch + scheduled_ms / 1000 - loop.time()))
                dispatched_ms = (loop.time() - epoch) * 1000
                row = dict(type="request", request_id=request.request_id,
                           scheduled_ms=scheduled_ms, dispatched_ms=dispatched_ms,
                           dispatch_lag_ms=dispatched_ms - scheduled_ms,
                           prompt_tokens=request.prompt_tokens,
                           output_tokens=request.output_tokens,
                           session_id=request.session_id, parent_request_id=request.parent_request_id,
                           archetype=request.metadata.get("archetype"),
                           oracle_hit_rate=request.metadata.get("oracle_hit_rate"))
                try:
                    row.update(await completion(session, base_url, payloads[index], encoded_payloads[index]))
                    row["status"] = "ok"
                except Exception as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                row["completed_ms"] = (loop.time() - epoch) * 1000
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                return row

            rows = await asyncio.gather(*(send(i, r) for i, r in enumerate(requests)))
        by_id = {r["request_id"]: r for r in rows}
        summary = dict(type="run_summary", errors=sum(r["status"] != "ok" for r in rows),
                       missing_cache_usage=sum(r.get("cached_tokens") is None for r in rows),
                       late_dispatches=sum(r["dispatch_lag_ms"] > max_dispatch_lag_ms for r in rows),
                       parent_overlap_requests=sum(
                           r["parent_request_id"] is not None and
                           r["dispatched_ms"] < by_id[r["parent_request_id"]]["completed_ms"]
                           for r in rows))
        stream.write(json.dumps(summary) + "\n")
    return summary
