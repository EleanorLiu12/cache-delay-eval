"""Plan or execute isolated vLLM caching-on/off runs on one CUDA node."""
import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import aiohttp

from .http_client import VLLMClient
from .live_replay import completion, replay
from .materialize import materialize
from .trace import read_trace, write_trace


def build_plan(config, root):
    if config["repetitions"] < 2:
        raise ValueError("use at least two repetitions to alternate condition order")
    runs = []
    for trace in config["traces"]:
        header, requests = read_trace(root / trace)
        if header.block_size != config["block_size"]:
            raise ValueError("trace/server block sizes differ")
        maximum = max(r.prompt_tokens + r.output_tokens for r in requests)
        if maximum > config["max_model_len"]:
            raise ValueError(f"{trace}: prompt + output exceeds max_model_len")
        for capacity in config["capacities"]:
            if capacity * config["block_size"] < config["max_model_len"]:
                raise ValueError("KV token capacity must cover max_model_len for this pilot")
            for repeat in range(config["repetitions"]):
                for condition in (("off", "on") if repeat % 2 == 0 else ("on", "off")):
                    runs.append(dict(trace=trace, capacity_blocks=capacity,
                                     repeat=repeat, condition=condition))
    return runs


def command(config, run, revision):
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", config["model"], "--revision", revision,
            "--tokenizer-revision", revision, "--host", "127.0.0.1",
            "--port", str(config["port"]), "--dtype", "bfloat16",
            "--kv-cache-dtype", "auto", "--block-size", str(config["block_size"]),
            "--num-gpu-blocks-override", str(run["capacity_blocks"]),
            "--max-model-len", str(config["max_model_len"]),
            "--max-num-seqs", str(config["max_num_seqs"]),
            "--max-num-batched-tokens", str(config["max_num_batched_tokens"]),
            "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
            "--enable-chunked-prefill", "--enable-prompt-tokens-details",
            "--generation-config", "vllm", "--seed", "699",
            "--enable-prefix-caching" if run["condition"] == "on" else "--no-enable-prefix-caching"]


def stop_server(process):
    # Only the process group created by this runner is terminated.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


async def probe(base, model, token, expected_caching):
    payload = dict(model=model, prompt=[token] * 64, max_tokens=4, min_tokens=4,
                   ignore_eos=True, temperature=0, seed=699, return_token_ids=True,
                   add_special_tokens=False, stream=True, stream_options={"include_usage": True})
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        await completion(session, base, payload)
        result = await completion(session, base, payload)
        cached = result["cached_tokens"]
        if cached is None or (expected_caching and cached == 0) or (not expected_caching and cached != 0):
            raise RuntimeError(f"prefix-cache probe failed: expected caching={expected_caching}, cached={cached}")
        async with session.post(base + "/reset_prefix_cache") as response:
            if response.status != 200 or (await response.json()).get("success") is not True:
                raise RuntimeError("server did not confirm a successful cache reset")
        return result


def execute(config, runs, root, destination):
    if importlib.metadata.version("vllm") != config["vllm_version"]:
        raise RuntimeError(f"install the pinned vLLM version {config['vllm_version']} first")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", config["port"]))  # fail if another server owns this port
    revision = config.get("model_revision")
    if not revision:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{config['model']}", timeout=30) as response:
            revision = json.load(response)["sha"]
    config = dict(config, model_revision=revision)
    (destination / "resolved-config.json").write_text(json.dumps(config, indent=2) + "\n")
    with urllib.request.urlopen(
            f"https://huggingface.co/{config['model']}/resolve/{revision}/config.json", timeout=30) as response:
        model_config = json.load(response)
    (destination / "model-config.json").write_text(json.dumps(model_config, indent=2) + "\n")
    if model_config.get("model_type") == "qwen3":
        head_dim = model_config.get("head_dim") or model_config["hidden_size"] // model_config["num_attention_heads"]
        bytes_per_token = 2 * model_config["num_hidden_layers"] * model_config["num_key_value_heads"] * head_dim * 2
        estimate = {"bf16_kv_bytes_per_token": bytes_per_token,
                    "kv_bytes_by_capacity": {str(c): c * config["block_size"] * bytes_per_token for c in config["capacities"]},
                    "note": "KV tensor estimate only; weights, graphs and runtime overhead excluded. Startup must still succeed."}
        (destination / "kv-geometry.json").write_text(json.dumps(estimate, indent=2) + "\n")
    for name, cmd in (("gpu.txt", ["nvidia-smi"]),
                      ("packages.txt", [sys.executable, "-m", "pip", "freeze"])):
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        (destination / name).write_text(result.stdout)
    shutil.copytree(Path(__file__).parent, destination / "code", ignore=shutil.ignore_patterns("__pycache__"))
    base = f"http://127.0.0.1:{config['port']}"
    materialized = {}
    alphabet = None
    for number, run in enumerate(runs):
        folder = destination / f"run-{number:03d}-{run['condition']}"
        folder.mkdir()
        argv = command(config, run, revision)
        (folder / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
        print(f"[{number + 1}/{len(runs)}] {run}", flush=True)
        with (folder / "server.log").open("w") as logfile:
            process = subprocess.Popen(argv, stdout=logfile, stderr=subprocess.STDOUT,
                                       start_new_session=True,
                                       env={**os.environ, "VLLM_SERVER_DEV_MODE": "1"})
            try:
                client = VLLMClient(base, timeout_s=3)
                deadline = time.monotonic() + config["startup_timeout_s"]
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"server exited; inspect {folder / 'server.log'}")
                    try:
                        client.health()
                        break
                    except Exception:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("server startup timed out")
                        time.sleep(1)
                tokens = list(dict.fromkeys(client.tokenize(
                    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
                    "omicron pi rho sigma tau upsilon phi chi psi omega 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ")))
                if len(tokens) < 17:
                    raise RuntimeError("tokenizer did not provide 17 distinct ordinary tokens")
                if alphabet is None:
                    alphabet = tokens[:16]
                elif alphabet != tokens[:16]:
                    raise RuntimeError("tokenizer changed between runs")
                if run["trace"] not in materialized:
                    header, requests = read_trace(root / run["trace"])
                    header, requests = materialize(header, requests, alphabet, config["model"])
                    trace_path = destination / "materialized" / Path(run["trace"]).name
                    write_trace(trace_path, header, requests)
                    materialized[run["trace"]] = trace_path
                probe_result = asyncio.run(probe(base, config["model"], tokens[16], run["condition"] == "on"))
                (folder / "probe.json").write_text(json.dumps(probe_result, indent=2) + "\n")
                (folder / "metrics-before.txt").write_text(client.metrics_text() or "")
                summary = asyncio.run(replay(
                    materialized[run["trace"]], base, config["model"], folder / "requests.jsonl",
                    dict(run, engine_config=config, engine_command=argv),
                    arrival_scale=config["arrival_scale"], timeout_s=config["request_timeout_s"],
                    max_dispatch_lag_ms=config["max_dispatch_lag_ms"]))
                (folder / "metrics-after.txt").write_text(client.metrics_text() or "")
                if summary["errors"] or summary["late_dispatches"] or summary["missing_cache_usage"]:
                    raise RuntimeError(f"run failed validity checks: {summary}; records preserved in {folder}")
            finally:
                stop_server(process)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="start GPU servers; default only writes a plan")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    root = config_path.parent.parent
    runs = build_plan(config, root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "plan.json").write_text(json.dumps(dict(config=config, runs=runs), indent=2) + "\n")
    if args.execute:
        execute(config, runs, root, args.output_dir)
    else:
        print(f"Validated {len(runs)} run plans; no server started. Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
