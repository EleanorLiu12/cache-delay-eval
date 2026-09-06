"""Calibrate TTFT against exact cached-prefix length and controlled queue depth."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import platform
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .http_client import VLLMClient, metric, parse_prometheus


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = ("base_url", "model", "prompt_tokens", "cached_prefix_tokens", "queue_depths")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing config keys: {', '.join(missing)}")
    return config


def _reservoir(client: VLLMClient, minimum_tokens: int, salt: str) -> list[int]:
    # Each sentence normally yields several tokens. Grow conservatively because
    # sending millions of unnecessary tokens to /tokenize can dominate a run.
    piece_count = max(512, minimum_tokens // 6)
    for _ in range(5):
        pieces = [
            f"calibration {salt} passage {i}: distributed caching queue latency token {i * 7919 % 104729}."
            for i in range(piece_count)
        ]
        tokens = client.tokenize(" ".join(pieces))
        if len(tokens) >= minimum_tokens:
            return tokens
        piece_count *= 2
    raise RuntimeError(f"token reservoir has {len(tokens)} tokens, need {minimum_tokens}")


def _sample_metrics(client: VLLMClient) -> dict[str, float]:
    return parse_prometheus(client.metrics_text())


def _counter_delta(before: dict[str, float], after: dict[str, float], *names: str) -> float | None:
    left = metric(before, *names)
    right = metric(after, *names)
    if left is None or right is None:
        return None
    return max(0.0, right - left)


def _active_requests(metrics: dict[str, float]) -> tuple[float | None, float | None]:
    running = metric(metrics, "vllm:num_requests_running", "vllm:num_requests_running_total")
    waiting = metric(metrics, "vllm:num_requests_waiting", "vllm:num_requests_waiting_total")
    return running, waiting


def _wait_for_blockers(
    client: VLLMClient,
    futures: list[concurrent.futures.Future[Any]],
    target: int,
    timeout_s: float,
) -> dict[str, float]:
    """Wait until target requests are visible as running/waiting, or stop safely."""
    deadline = time.monotonic() + timeout_s
    best: dict[str, float] = _sample_metrics(client)
    best_active = -1.0
    while True:
        current = _sample_metrics(client)
        running, waiting = _active_requests(current)
        active = (running or 0.0) + (waiting or 0.0)
        if active > best_active:
            best, best_active = current, active
        if active >= target or all(future.done() for future in futures):
            return current if active >= target else best
        if time.monotonic() >= deadline:
            return best
        time.sleep(0.01)


def run(config: dict[str, Any], output: Path, trials_override: int | None = None) -> None:
    client = VLLMClient(
        config["base_url"], config.get("api_key", ""), float(config.get("timeout_s", 180))
    )
    client.health()
    server_version = client.version()
    block_size = int(config.get("block_size", 16))
    trials = int(trials_override if trials_override is not None else config.get("trials", 30))
    seed = int(config.get("seed", 699))
    blocker_output_tokens = int(config.get("blocker_output_tokens", 32))
    headstart_ms = float(config.get("blocker_headstart_ms", 25))
    blocker_ready_timeout_s = float(config.get("blocker_ready_timeout_s", 5))
    prompt_lengths = [int(value) for value in config["prompt_tokens"]]
    prefix_lengths = [int(value) for value in config["cached_prefix_tokens"]]
    queue_depths = [int(value) for value in config["queue_depths"]]
    if block_size < 4 or trials < 1 or any(value < 0 for value in queue_depths):
        raise ValueError("block_size must be >= 4, trials positive, and queue depths non-negative")

    conditions = [
        (prompt_len, prefix_len, queue_depth)
        for prompt_len, prefix_len, queue_depth in itertools.product(
            prompt_lengths, prefix_lengths, queue_depths
        )
        if 0 <= prefix_len <= prompt_len - block_size and prefix_len % block_size == 0
    ]
    if not conditions:
        raise ValueError("no valid conditions; prefixes must be block-aligned and leave one suffix block")
    schedule = [(trial, *condition) for trial in range(trials) for condition in conditions]
    random.Random(seed).shuffle(schedule)
    max_prompt = max(prompt_lengths)
    max_queue = max(queue_depths)
    # A unique first block prevents cache bleed across conditions while the
    # remaining bodies can reuse a small token pool. One nonce is needed for
    # each measured condition, each of its possible blocker requests, and the
    # placebo warm request issued when the condition has no cached prefix.
    nonce_count = len(schedule) * (max_queue + 2)
    body_tokens = max_prompt * 4
    run_uuid = uuid.uuid4()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + run_uuid.hex
    body = _reservoir(client, body_tokens, run_id)
    distinct = list(dict.fromkeys(body))
    if len(distinct) < 16:
        raise RuntimeError("token reservoir did not contain 16 distinct token IDs")
    alphabet = distinct[:16]
    run_width = block_size // 2
    digit_width = block_size - run_width
    if nonce_count > len(alphabet) ** digit_width:
        raise RuntimeError("not enough unique nonce blocks for calibration schedule")

    def encode_digits(value: int, width: int) -> list[int]:
        digits = [alphabet[0]] * width
        original = value
        for position in range(width - 1, -1, -1):
            digits[position] = alphabet[value % len(alphabet)]
            value //= len(alphabet)
        if value:
            raise RuntimeError(f"value {original} does not fit in a {width}-token nonce")
        return digits

    # Split the first block into a run code and a request code. This guarantees
    # that even a warm server cannot reuse blocks retained from a prior run.
    run_prefix = encode_digits(run_uuid.int & ((1 << (run_width * 4)) - 1), run_width)
    digit_width = block_size - len(run_prefix)

    def nonce_block(index: int) -> list[int]:
        digits = encode_digits(index, digit_width)
        return run_prefix + digits

    def prompt_with_nonce(nonce: int, length: int, body_offset: int) -> list[int]:
        first = nonce_block(nonce)
        remaining = length - block_size
        return first + body[body_offset : body_offset + remaining]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        for sequence, (trial, prompt_len, prefix_len, queue_depth) in enumerate(schedule):
            nonce_base = sequence * (max_queue + 2)
            if prefix_len:
                shared = prompt_with_nonce(nonce_base, prefix_len, 0)
                warm_suffix = body[max_prompt : max_prompt + prompt_len - prefix_len]
                probe_suffix = body[2 * max_prompt : 2 * max_prompt + prompt_len - prefix_len]
                warm_prompt = shared + warm_suffix
                probe_prompt = shared + probe_suffix
            else:
                shared = []
                probe_prompt = prompt_with_nonce(nonce_base, prompt_len, 2 * max_prompt)
                # Placebo control. The cached conditions issue a warm request before
                # the probe, and that request itself perturbs scheduler state, KV-cache
                # occupancy, allocator state and CPU frequency. Skipping it here would
                # leave the cached and uncached conditions differing in two ways at
                # once, so the warm request's own effect would be scored as a cache
                # effect. This prompt costs the server the same work but starts from a
                # different nonce block, so it shares no reusable prefix with the probe.
                warm_prompt = prompt_with_nonce(
                    nonce_base + max_queue + 1, prompt_len, max_prompt
                )
            result: dict[str, Any] = {
                "type": "calibration_result",
                "schema_version": "1.0",
                "run_id": run_id,
                "sequence": sequence,
                "trial": trial,
                "model": config["model"],
                "backend": config.get("backend", "vllm"),
                "server_version": server_version,
                "hardware": config.get("hardware"),
                "platform": platform.platform(),
                "prompt_tokens_target": prompt_len,
                "cached_prefix_tokens_target": prefix_len,
                "queue_depth_target": queue_depth,
                "warm_request": "shared_prefix" if prefix_len else "placebo",
                "block_size": block_size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            try:
                client.stream_completion(config["model"], warm_prompt, 1, seed + sequence)
                before = _sample_metrics(client)
                blockers = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=queue_depth + 1) as pool:
                    for blocker in range(queue_depth):
                        blocker_prompt = prompt_with_nonce(
                            nonce_base + blocker + 1, prompt_len, 3 * max_prompt
                        )
                        blockers.append(
                            pool.submit(
                                client.stream_completion,
                                config["model"],
                                blocker_prompt,
                                blocker_output_tokens,
                                seed + sequence + blocker + 1,
                            )
                        )
                    if blockers:
                        time.sleep(headstart_ms / 1000)
                        at_dispatch = _wait_for_blockers(
                            client, blockers, queue_depth, blocker_ready_timeout_s
                        )
                    else:
                        at_dispatch = _sample_metrics(client)
                    timing = client.stream_completion(
                        config["model"], probe_prompt, 1, seed + sequence
                    )
                    for future in blockers:
                        future.result()
                after = _sample_metrics(client)
                running_observed, waiting_observed = _active_requests(at_dispatch)
                active_observed = (
                    None
                    if running_observed is None and waiting_observed is None
                    else (running_observed or 0.0) + (waiting_observed or 0.0)
                )
                result.update(
                    {
                        "status": "ok",
                        "ttft_ms": timing.ttft_ms,
                        "e2e_ms": timing.e2e_ms,
                        "prompt_tokens_reported": timing.prompt_tokens,
                        "completion_tokens_reported": timing.completion_tokens,
                        "finish_reason": timing.finish_reason,
                        "queue_depth_observed": active_observed,
                        "requests_waiting_observed": waiting_observed,
                        "requests_running_observed": running_observed,
                        "kv_cache_usage_observed": metric(
                            at_dispatch,
                            "vllm:kv_cache_usage_perc",
                            "vllm:gpu_cache_usage_perc",
                        ),
                        "prefix_cache_hit_tokens_batch": _counter_delta(
                            before,
                            after,
                            "vllm:prefix_cache_hits",
                            "vllm:prefix_cache_hits_total",
                        ),
                        "prefix_cache_query_tokens_batch": _counter_delta(
                            before,
                            after,
                            "vllm:prefix_cache_queries",
                            "vllm:prefix_cache_queries_total",
                        ),
                    }
                )
            except Exception as exc:  # Preserve failures in raw experimental output.
                result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()
            status = result["status"]
            detail = "" if status != "ok" else f" ttft={result['ttft_ms']:.1f}ms"
            print(
                f"[{sequence + 1}/{len(schedule)}] prompt={prompt_len} prefix={prefix_len} "
                f"queue={queue_depth} trial={trial} {status}{detail}",
                flush=True,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trials", type=int, help="override trials for a smoke run")
    args = parser.parse_args(argv)
    try:
        config = _load_config(args.config)
        output = args.output or Path(config.get("output", "results/calibration.jsonl"))
        run(config, output, args.trials)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"calibration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
