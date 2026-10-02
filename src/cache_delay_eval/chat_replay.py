"""Completion-dependent conversation replay with retained generated responses.

This module never launches a model server. Its CLI only contacts an existing
server with --execute; the CPU validation uses an explicit local mock server.
The historical open-loop replay remains unchanged.
"""
import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time

import aiohttp


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class SessionScript:
    session_id: str
    user_messages: tuple[str, ...]
    root_arrival_s: float = 0.0
    source_transcript_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.session_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", self.session_id):
            raise ValueError("session_id must be a nonempty ASCII identifier")
        if not isinstance(self.user_messages, (tuple, list)) or not self.user_messages or any(
                not isinstance(x, str) or not x.strip() for x in self.user_messages):
            raise ValueError("Preserve complete nonempty user messages; invalid scripts need explicit exclusion")
        if not math.isfinite(self.root_arrival_s) or self.root_arrival_s < 0:
            raise ValueError("root_arrival_s must be finite and nonnegative")


@dataclass(frozen=True)
class ReplayConfig:
    model: str = "Qwen/Qwen3-4B"
    run_id: str = "chat-replay"
    max_tokens: int = 512
    max_model_len: int = 32768
    think_delay_s: float = 5.0
    timeout_s: float = 600.0
    require_cached_tokens: bool = True
    max_dispatch_lag_ms: float = 50.0
    seed: int = 699

    def __post_init__(self):
        if not self.model or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", self.run_id):
            raise ValueError("Need a model and an ASCII run identifier")
        for name in ("max_tokens", "max_model_len"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_tokens >= self.max_model_len:
            raise ValueError("Output allowance must leave room for a prompt")
        for name in ("think_delay_s", "max_dispatch_lag_ms"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if type(self.seed) is not int or type(self.require_cached_tokens) is not bool:
            raise ValueError("Invalid seed/cache validation flag")


class QwenPromptBuilder:
    """Use only the locally retained, hash-checked Qwen3 tokenizer/template.

    Responses are decoded from actual returned token IDs. One terminal EOS is
    omitted from message content; its original ID remains in request evidence.
    Rendering history can change token boundaries or Qwen reasoning scaffolding,
    so prompt/output IDs and parent-prefix lengths are retained rather than
    asserting that re-rendering preserves every previously computed KV block.
    """
    def __init__(self, tokenizer_dir, manifest_path):
        tokenizer_dir, manifest_path = Path(tokenizer_dir), Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        revision = "1cfa9a7208912126459214e8b04321603b3df60c"
        if manifest.get("tokenizer_revision") != revision or manifest.get("tokenizer") != "Qwen/Qwen3-4B":
            raise ValueError("Tokenizer source is not the pinned Qwen3-4B revision")
        files = {}
        for entry in manifest["files"]:
            if entry["file"].startswith("tokenizer/"):
                path = tokenizer_dir / Path(entry["file"]).name
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != entry["sha256"]:
                    raise ValueError(f"Tokenizer file changed: {path.name}")
                files[path.name] = actual
        if not {"tokenizer.json", "tokenizer_config.json"} <= set(files):
            raise ValueError("Missing tokenizer file checksums")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
        self.metadata = dict(model="Qwen/Qwen3-4B", revision=revision, files=files,
                             chat_template_sha256=hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest(),
                             enable_thinking=False, add_generation_prompt=True, invented_system_message=False,
                             assistant_content="decode actual returned IDs, removing one terminal EOS; keep all original IDs in evidence",
                             history_rendering="template re-rendering may change token boundaries and thinking scaffold; no exact KV-preservation assumption")

    def render(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)

    def decode_response(self, token_ids):
        ids = list(token_ids)
        if ids and ids[-1] == self.tokenizer.eos_token_id:
            ids.pop()
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def response_audit(self, token_ids, content):
        expected = list(token_ids)
        omitted = expected.pop() if expected and expected[-1] == self.tokenizer.eos_token_id else None
        encoded = self.tokenizer.encode(content, add_special_tokens=False)
        return dict(terminal_eos_omitted_from_message=omitted,
                    response_text_reencoding_exact=encoded == expected,
                    reencoded_content_token_ids=encoded)


class CompletionError(RuntimeError):
    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


async def sse_data(response):
    """Collect complete SSE data fields across arbitrary HTTP chunks."""
    parts = []
    async for raw in response.content:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if parts:
                yield "\n".join(parts)
                parts = []
        elif line.startswith("data:"):
            parts.append(line[5:].removeprefix(" "))
    if parts:
        yield "\n".join(parts)


async def stream_completion(session, base_url, payload, require_cached_tokens=True):
    """Validate a single n=1 stream, retaining partial outputs on every failure."""
    loop = asyncio.get_running_loop()
    state = dict(output_token_ids=[], streamed_text="", usage={}, finish_reason=None,
                 first_token_at=None, first_text_at=None, response_id=None, done=False,
                 http_started_at=loop.time(), stream_events=[])
    try:
        async with session.post(base_url.rstrip("/") + "/v1/completions", json=payload,
                                headers={"X-Request-Id": payload["request_id"]}) as response:
            state["http_status"] = response.status
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}: {(await response.text())[:2000]}")
            async for data in sse_data(response):
                now = loop.time()
                if data == "[DONE]":
                    state["done"] = True
                    break
                event = json.loads(data)
                if not isinstance(event, dict):
                    raise ValueError("SSE event must be an object")
                if event.get("error"):
                    raise ValueError(f"Server stream error: {event['error']}")
                if event.get("id"):
                    if state["response_id"] not in (None, event["id"]):
                        raise ValueError("Response ID changed within one stream")
                    state["response_id"] = event["id"]
                if isinstance(event.get("usage"), dict):
                    state["usage"] = event["usage"]
                choices = event.get("choices", [])
                if not isinstance(choices, list) or len(choices) > 1:
                    raise ValueError("Expected at most one completion choice per event")
                for choice in choices:
                    if choice.get("index", 0) != 0:
                        raise ValueError("Unexpected completion index")
                    ids = choice.get("token_ids") or []
                    if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids):
                        raise ValueError("Invalid streamed token IDs")
                    text = choice.get("text") or ""
                    if not isinstance(text, str):
                        raise ValueError("Invalid streamed text")
                    if ids and state["finish_reason"] is not None:
                        raise ValueError("Tokens arrived after a finished choice")
                    if ids and state["first_token_at"] is None:
                        state["first_token_at"] = now
                    if text and state["first_text_at"] is None:
                        state["first_text_at"] = now
                    state["output_token_ids"].extend(ids)
                    state["streamed_text"] += text
                    state["stream_events"].append(dict(at=now, token_ids=ids, text=text,
                                                        finish_reason=choice.get("finish_reason")))
                    if choice.get("finish_reason") is not None:
                        state["finish_reason"] = choice["finish_reason"]
                    if len(state["output_token_ids"]) > payload["max_tokens"]:
                        raise ValueError("Server output exceeded the declared cap")
        usage = state["usage"]
        if not state["done"] or state["first_token_at"] is None or state["finish_reason"] not in ("stop", "length"):
            raise ValueError("Incomplete stream or missing token IDs/valid finish reason")
        if type(usage.get("prompt_tokens")) is not int or usage["prompt_tokens"] != len(payload["prompt"]):
            raise ValueError("Prompt usage missing or differs from submitted tokens")
        if type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] != len(state["output_token_ids"]):
            raise ValueError("Completion usage missing or differs from retained output IDs")
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        if cached is None and require_cached_tokens:
            raise ValueError("Actual cached-token count is missing")
        if cached is not None and (type(cached) is not int or not 0 <= cached <= len(payload["prompt"])):
            raise ValueError("Invalid cached-token count")
        state["cached_tokens"] = cached
        state["http_completed_at"] = loop.time()
        return state
    except asyncio.CancelledError:
        # Outer runner retains stream progress even for a cancelled in-flight request.
        state["http_completed_at"] = loop.time()
        exc = asyncio.CancelledError()
        exc.partial = state
        raise exc
    except Exception as exc:
        state["http_completed_at"] = loop.time()
        raise CompletionError(f"{type(exc).__name__}: {exc}", state) from exc


def token_prefix_length(left, right):
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


async def replay_sessions(scripts, builder, base_url, output_dir, config, run_meta=None):
    """Run independent sessions concurrently, with sequential turns per session.

    All planned turns receive a final status, including descendants of failures.
    Clock offsets are local monotonic seconds from the root schedule epoch.
    Release-to-first-token includes client-side preparation/dispatch delay.
    """
    scripts = list(scripts)
    if not scripts or len({s.session_id for s in scripts}) != len(scripts):
        raise ValueError("Need nonempty scripts with unique session IDs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    loop = asyncio.get_running_loop()
    epoch = loop.time() + .05
    created = datetime.now(timezone.utc).isoformat()
    provenance = dict(type="run_meta", created_at_utc=created, config=asdict(config),
                      scripts=[asdict(s) for s in scripts], scripts_sha256=digest([asdict(s) for s in scripts]),
                      tokenizer=builder.metadata, workload="completion-dependent generated-response conversation",
                      timing=dict(origin_monotonic_s=epoch, sampled_monotonic_s=loop.time(), sampled_unix_s=time.time(),
                                  ttft="HTTP POST start to first streamed token ID; release_to_first_token includes client delay",
                                  token_gap="client SSE event/token-ID delivery groups, not server per-token generation times"),
                      metadata=run_meta or {}, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output_dir / "run.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    results = []
    seq = 0
    interrupted = False
    with (output_dir / "events.jsonl").open("x") as events, (output_dir / "turns.jsonl").open("x") as turns:
        def emit(kind, **fields):
            nonlocal seq
            row = dict(event_id=seq, kind=kind, observed_s=loop.time()-epoch, **fields)
            seq += 1
            events.write(json.dumps(row, ensure_ascii=False) + "\n")
            events.flush()

        def finish(row):
            results.append(row)
            turns.write(json.dumps(row, ensure_ascii=False) + "\n")
            turns.flush()
            emit("turn_final", request_id=row["request_id"], status=row["status"])

        def attach_completion(row, state):
            for key in ("output_token_ids", "streamed_text", "usage", "finish_reason", "response_id", "done", "cached_tokens", "http_status"):
                if key in state:
                    row[key] = state[key]
            for source, target in (("first_token_at", "first_token_s"), ("first_text_at", "first_text_s"),
                                   ("http_started_at", "http_started_s"), ("http_completed_at", "http_completed_s")):
                row[target] = state[source]-epoch if state.get(source) is not None else None
            row["stream_events"] = [dict(event, at=event["at"]-epoch) for event in state.get("stream_events", [])]
            first, start = row.get("first_token_s"), row.get("http_started_s")
            row["client_ttft_ms"] = None if first is None else (first-start)*1000
            row["release_to_first_token_ms"] = None if first is None else (first-row["release_s"])*1000
            row["output_token_ids_sha256"] = digest(row.get("output_token_ids", []))

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout_s),
                                          connector=aiohttp.TCPConnector(limit=0)) as session:
            async def run_session(script, session_index):
                history = []
                parent = None
                release = script.root_arrival_s
                failure = None
                for index, user_text in enumerate(script.user_messages, 1):
                    request_id = f"{config.run_id}:{script.session_id}:turn{index}"
                    row = dict(session_id=script.session_id, turn_index=index, request_id=request_id,
                               parent_request_id=parent["request_id"] if parent else None,
                               root_arrival_s=script.root_arrival_s, release_s=release if not failure else None,
                               dispatched_s=None, completed_s=None, prompt_token_ids=[], output_token_ids=[],
                               assistant_content=None, streamed_text="", finish_reason=None, cached_tokens=None)
                    if failure:
                        row.update(status="blocked", blocked_by_request_id=failure, reason="ancestor did not complete successfully")
                        finish(row)
                        parent = row
                        continue
                    try:
                        await asyncio.sleep(max(0, epoch+release-loop.time()))
                        emit("released", request_id=request_id, release_s=release)
                        history.append(dict(role="user", content=user_text))
                        row["history_messages"] = [dict(m) for m in history]
                        tokens = list(await asyncio.to_thread(builder.render, row["history_messages"]))
                        if not tokens or any(type(t) is not int or t < 0 for t in tokens):
                            raise ValueError("Prompt builder returned invalid token IDs")
                        row.update(prompt_token_ids=tokens, prompt_tokens=len(tokens), prompt_token_ids_sha256=digest(tokens),
                                   prompt_prepared_s=loop.time()-epoch,
                                   parent_completed_s=parent["completed_s"] if parent else None,
                                   parent_input_prefix_tokens=token_prefix_length(parent["prompt_token_ids"], tokens) if parent else None)
                        if len(tokens)+config.max_tokens > config.max_model_len:
                            row.update(status="context_overflow", reason="complete history plus output allowance exceeds context limit")
                            row["completed_s"] = loop.time()-epoch
                            failure = request_id
                            finish(row)
                            parent = row
                            continue
                        payload = dict(model=config.model, request_id=request_id, prompt=tokens,
                                       max_tokens=config.max_tokens, temperature=0, seed=config.seed+session_index*10000+index,
                                       n=1, ignore_eos=False, add_special_tokens=False, return_token_ids=True,
                                       skip_special_tokens=False, spaces_between_special_tokens=False,
                                       stream=True, stream_options={"include_usage": True})
                        row["payload_sha256"] = digest(payload)
                        row["dispatched_s"] = loop.time()-epoch
                        row["dispatch_lag_ms"] = (row["dispatched_s"]-release)*1000
                        emit("dispatched", request_id=request_id, parent_request_id=row["parent_request_id"],
                             prompt_tokens=len(tokens), payload_sha256=row["payload_sha256"])
                        response = await stream_completion(session, base_url, payload, config.require_cached_tokens)
                        attach_completion(row, response)
                        content = builder.decode_response(row["output_token_ids"])
                        if not isinstance(content, str):
                            raise ValueError("Response decoder did not return text")
                        row.update(assistant_content=content, assistant_content_sha256=digest(content), status="ok")
                        if hasattr(builder, "response_audit"):
                            row["response_serialization"] = builder.response_audit(row["output_token_ids"], content)
                        # This exact generated content, rather than the source assistant, enters the next history.
                        history.append(dict(role="assistant", content=content))
                        row["completed_s"] = loop.time()-epoch
                        row["end_to_end_ms"] = (row["completed_s"]-row["http_started_s"])*1000
                        release = row["completed_s"]+config.think_delay_s
                    except asyncio.CancelledError as exc:
                        if hasattr(exc, "partial"):
                            attach_completion(row, exc.partial)
                        row.update(status="cancelled", completed_s=loop.time()-epoch, reason="run task cancelled")
                        finish(row)
                        prior = request_id
                        for remaining in range(index+1, len(script.user_messages)+1):
                            child_id = f"{config.run_id}:{script.session_id}:turn{remaining}"
                            finish(dict(session_id=script.session_id, turn_index=remaining, request_id=child_id,
                                        parent_request_id=prior, status="blocked", blocked_by_request_id=request_id,
                                        release_s=None, dispatched_s=None, completed_s=None,
                                        reason="ancestor cancelled", prompt_token_ids=[], output_token_ids=[]))
                            prior = child_id
                        raise
                    except Exception as exc:
                        if isinstance(exc, CompletionError):
                            attach_completion(row, exc.partial)
                        row.update(status="error", error=f"{type(exc).__name__}: {exc}", completed_s=loop.time()-epoch)
                        failure = request_id
                    finish(row)
                    parent = row

            tasks = [asyncio.create_task(run_session(s, i)) for i, s in enumerate(scripts)]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                interrupted = True
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            summary = summarize_replay(results, scripts, config, interrupted)
            emit("run_final", summary=summary)
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    if interrupted:
        raise asyncio.CancelledError("Replay interrupted; partial records and summary retained")
    return summary


def summarize_replay(rows, scripts, config, interrupted=False):
    by_id = {r["request_id"]: r for r in rows}
    successful = [r for r in rows if r["status"] == "ok"]
    submitted = [r for r in rows if r.get("dispatched_s") is not None]
    violations = []
    for row in submitted:
        if row["parent_request_id"] is not None:
            parent = by_id.get(row["parent_request_id"])
            if not parent or parent["status"] != "ok" or row["dispatched_s"]+1e-9 < parent["completed_s"]+config.think_delay_s:
                violations.append(row["request_id"])
    expected = sum(len(s.user_messages) for s in scripts)
    statuses = {status: sum(r["status"] == status for r in rows) for status in ("ok", "error", "blocked", "context_overflow", "cancelled")}
    late = sum(r["dispatch_lag_ms"] > config.max_dispatch_lag_ms for r in submitted)
    sessions = []
    for script in scripts:
        group = [r for r in rows if r["session_id"] == script.session_id]
        done = len(group) == len(script.user_messages) and all(r["status"] == "ok" for r in group)
        end = max((r["completed_s"] for r in group if r.get("completed_s") is not None), default=None)
        sessions.append(dict(session_id=script.session_id, planned_turns=len(script.user_messages),
                             completed=done, completion_s=end if done else None,
                             session_latency_s=end-script.root_arrival_s if done else None))
    return dict(planned_requests=expected, recorded_requests=len(rows), submitted_requests=len(submitted),
                status_counts=statuses, parent_dependency_violations=violations, late_dispatches=late,
                missing_cache_usage=sum(r.get("cached_tokens") is None for r in successful),
                completed_sessions=sum(s["completed"] for s in sessions), sessions=sessions, interrupted=interrupted,
                valid=not interrupted and len(rows)==expected and len(successful)==expected and not violations and not late,
                successful_request_metrics=dict(denominator=len(successful),
                    mean_client_ttft_ms=sum(r["client_ttft_ms"] for r in successful)/len(successful) if successful else None,
                    mean_release_to_first_token_ms=sum(r["release_to_first_token_ms"] for r in successful)/len(successful) if successful else None),
                interpretation="Latency averages exclude failed/blocked turns; status denominators and complete-session outcomes must accompany them. Mock-server values are correctness checks, not model performance.")


def load_scripts(path, schedule_path):
    rows = [json.loads(line) for line in Path(path).open() if line.strip()]
    schedule = json.loads(Path(schedule_path).read_text())
    starts = schedule["root_arrivals_s"]
    if set(starts) != {r["sample_id"] for r in rows}:
        raise ValueError("Schedule must cover exactly the selected sessions")
    expected = schedule.get("source_scripts_sha256")
    if expected and hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError("Script checksum differs from frozen root schedule")
    return [SessionScript(r["sample_id"], tuple(r["user_messages"]), starts[r["sample_id"]], r.get("source_transcript_sha256")) for r in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripts", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--execute", action="store_true", help="Contact an already running server; default only validates a plan")
    args = parser.parse_args(argv)
    config = ReplayConfig(**json.loads(args.config.read_text()))
    scripts = load_scripts(args.scripts, args.schedule)
    builder = QwenPromptBuilder(args.tokenizer_dir, args.tokenizer_manifest)
    if args.execute:
        summary = asyncio.run(replay_sessions(scripts, builder, args.base_url, args.output_dir, config))
        print(json.dumps(summary, indent=2))
        return 0 if summary["valid"] else 2
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = dict(config=asdict(config), scripts=[asdict(s) for s in scripts], tokenizer=builder.metadata,
                first_prompt_tokens={s.session_id:len(builder.render([dict(role="user", content=s.user_messages[0])])) for s in scripts},
                server_contacted=False, note="Later prompts depend on actual generated responses and are checked at runtime.")
    (args.output_dir/"plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False)+"\n")
    print(f"Validated {len(scripts)} sessions; no server contacted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
