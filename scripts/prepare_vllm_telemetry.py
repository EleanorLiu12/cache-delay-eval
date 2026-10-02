#!/usr/bin/env python3
"""Fetch/check pinned source and prepare a reviewable passive telemetry patch.

No vLLM import, install, model download, inference, or GPU use is performed.
"""
from __future__ import annotations
import argparse
import ast
import datetime
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
BASE = f"https://raw.githubusercontent.com/vllm-project/vllm/{COMMIT}/"
SCHEDULER = "vllm/v1/core/sched/scheduler.py"
ALLOCATOR = "vllm/v1/core/kv_cache_manager.py"
INPUT = "vllm/v1/engine/input_processor.py"
POOL = "vllm/v1/core/block_pool.py"
HELPER = "vllm/v1/core/cache_delay_telemetry.py"
HASHES = {
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    SCHEDULER: "4b79ab760c3864af6838d555c01335deea3aac97c9be2be415fa0e5dab32f07e",
    ALLOCATOR: "2d20c3d98845cfd8d88a2f66b8fc6402ea1fcfbee16879d2364a4a9e45b8fa47",
    INPUT: "f9a7946a16acc2374ff2bdfc22f212cb43461d9ef4d99c5e19a536339f11212f",
    "vllm/v1/core/block_pool.py": "ddee56dccb2208411b3a035918e917ce8f56a9858471e9ca12b420d5d79bc69c",
    "vllm/v1/core/kv_cache_utils.py": "088f2201bee86fade694e78141b6e99a5cd0cdd23c5c7ab3526dd119f76e4aec",
    "vllm/entrypoints/openai/completion/serving.py": "c4348ccaf254e4a6b729f9c452aa02f899df64a1fb6d3ab7b7dde199f53dff40",
    "vllm/v1/core/sched/request_queue.py": "4b8d938e5fb8152fe61030b8aa991f983fcac9a405c37ed8908045e05e82ae5e",
    "vllm/config/scheduler.py": "3cf5d41a5d662ab0eada6f3128e2538abc0a250d6aef82acab7df7dd84e498bc",
    "vllm/entrypoints/openai/chat_completion/serving.py": "a2440b6b76ad87de86bcf6fc7061ad5d02ab67c1c3c89ca7de3eeb29aa12aa2f",
    "vllm/v1/engine/async_llm.py": "bceed0b3f5f0c834fef79525f2462a092f082390f0070526280abc95945837dd",
    "vllm/v1/engine/parallel_sampling.py": "bf3f0a3c6640aaf706d1b6f7ec3a8e09a970f3dd9e7e96fa6a901779ea84f61d",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def insert(text: str, anchor: str, insertion: str, *, before: bool = False) -> str:
    if text.count(anchor) != 1:
        raise ValueError(f"expected one patch anchor: {anchor[:100]!r}")
    return text.replace(anchor, insertion + anchor if before else anchor + insertion)


def patched_sources(original: dict[str, str]) -> dict[str, str]:
    patched = {}
    for path in (SCHEDULER, ALLOCATOR, INPUT, POOL):
        patched[path] = insert(original[path], "from vllm.logger import init_logger\n",
                              "from vllm.v1.core import cache_delay_telemetry as _cdt\n")
    text = patched[SCHEDULER]
    text = insert(text, "        self.current_step += 1\n", "        _cdt.step_start(self)\n")
    text = insert(text, "        # Next, schedule the WAITING requests.\n", "        _cdt.gate(None)\n")
    text = insert(text, "            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n                if input_budget <= draft_slots:\n",
                  "                    _cdt.gate('input_budget_exhausted')\n")
    text = insert(text, "                if num_running >= self.max_num_running_reqs:\n",
                  "                    _cdt.gate('max_running_requests')\n")
    text = insert(text, "                request = request_queue.peek_request()\n                request_id = request.request_id\n",
                  "                _cdt.waiting_attempt(self, request, token_budget, input_budget)\n")
    text = insert(text, "                    # DP prefill balancing: defer this step's local prefill\n                    # compute to a cadence-aligned step.\n",
                  "                    _cdt.gate('prefill_cadence_deferred')\n")
    text = insert(text, "                        # If chunked_prefill is disabled,\n                        # we can stop the scheduling here.\n",
                  "                        _cdt.gate('unchunked_prefill_exceeds_budget')\n")
    text = insert(text, "                self.running.append(request)\n",
                  "                _cdt.emit('admitted', request, proposed_tokens=num_new_tokens,\n"
                  "                          local_cached_tokens=num_new_local_computed_tokens,\n"
                  "                          previous_status=str(request.status))\n")
    text = insert(text, "        # Check if the scheduling constraints are satisfied.\n",
                  "        _cdt.waiting_end(self, token_budget, input_budget, draft_slots,\n"
                  "                         bool(preempted_reqs), self._pause_state != PauseState.UNPAUSED,\n"
                  "                         num_scheduled_tokens)\n", before=True)
    text = insert(text, "        assert request.status == RequestStatus.RUNNING, (\n            \"Only running requests can be preempted\"\n        )\n",
                  "        _cdt.preemption(self, request)\n")
    text = insert(text, "            self._enqueue_waiting_request(request)\n            self.requests[request.request_id] = request\n",
                  "            _cdt.emit('enqueue', request, arrival_time=request.arrival_time)\n")
    patched[SCHEDULER] = text
    text = patched[ALLOCATOR]
    text = insert(text, "            if required_blocks > self.block_pool.get_num_free_blocks():\n",
                  "            _cdt.allocation_check(self, request, 'full_sequence', required_blocks,\n"
                  "                                  self.block_pool.get_num_free_blocks(), num_new_tokens,\n"
                  "                                  num_new_computed_tokens, watermark_blocks, reserved_blocks)\n", before=True)
    text = insert(text, "        if required_blocks > available_blocks:\n",
                  "        _cdt.allocation_check(self, request, 'chunk', required_blocks, available_blocks,\n"
                  "                              num_new_tokens, num_new_computed_tokens,\n"
                  "                              watermark_blocks, reserved_blocks)\n", before=True)
    patched[ALLOCATOR] = add_request_scopes(text)
    patched[INPUT] = insert(patched[INPUT], '            request.request_id = f"{request.external_req_id}-{random_uuid():.8}"\n',
                            "        _cdt.request_id_map(request)\n")
    text = patched[POOL]
    text = insert(text, "        self.metrics_collector = metrics_collector\n", "        _cdt.pool_initial(self)\n")
    text = insert(text, "        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)\n",
                  "        _cdt.allocation_begin(self, num_blocks)\n", before=True)
    text = insert(text, "        return ret\n", "        _cdt.allocation_end(self, ret)\n", before=True)
    text = insert(text, "        block.reset_hash()\n        return removed_hashes\n",
                  "        _cdt.cache_remove(self, block, removed_hashes)\n", before=True)
    text = insert(text, "        self.cached_block_hash_to_block.insert(block_hash_with_group_id, block)\n",
                  "        _cdt.cache_insert(self, block, block_hash_with_group_id)\n")
    text = insert(text, "            if not block:\n", "            _cdt.cache_lookup(self, block_hash_with_group_id, block)\n", before=True)
    text = insert(text, "            block.ref_cnt += 1\n            if self.metrics_collector:\n",
                  "            _cdt.cache_reference(self, block, 'cache_touch')\n", before=True)
    text = insert(text, "            block.ref_cnt -= 1\n", "            _cdt.cache_reference(self, block, 'cache_release')\n")
    text = insert(text, "        self.cached_block_hash_to_block = BlockHashToBlockMap()\n",
                  "        _cdt.emit('cache_reset')\n", before=True)
    patched[POOL] = text
    patched[HELPER] = (ROOT / "src/cache_delay_eval/admission_telemetry.py").read_text()
    for path, text in patched.items():
        compile(text, path, "exec")
        if path != HELPER:
            assert_observational_ast(original[path], text)
    return patched


def add_request_scopes(text: str) -> str:
    """Insert read-only context calls at method entry and every normal exit."""
    additions = {}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name in {'allocate_slots', 'get_computed_blocks'}:
            first = node.body[1]  # both pinned methods have a docstring
            additions.setdefault(first.lineno - 1, []).append('        _cdt.request_scope(request)\n')
            for ret in ast.walk(node):
                if isinstance(ret, ast.Return):
                    additions.setdefault(ret.lineno - 1, []).append(' ' * ret.col_offset + '_cdt.request_scope()\n')
    lines = text.splitlines(True)
    for index in sorted(additions, reverse=True):
        lines[index:index] = additions[index]
    return ''.join(lines)


class RemoveTelemetry(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == "vllm.v1.core" and any(a.asname == "_cdt" for a in node.names):
            return None
        return node

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "_cdt"):
            return None
        return self.generic_visit(node)


def assert_observational_ast(original: str, patched: str) -> None:
    """Removing only added telemetry calls/import must recover the original AST."""
    stripped = RemoveTelemetry().visit(ast.parse(patched))
    if ast.dump(ast.parse(original), include_attributes=False) != ast.dump(stripped, include_attributes=False):
        raise ValueError("patch changes scheduler/input/allocator semantics beyond telemetry calls")


def prepare(source: Path, output: Path, *, fetch: bool = False, apply: bool = False,
            check_installed_version: bool = False) -> dict:
    if check_installed_version and importlib.metadata.version("vllm") != "0.28.0":
        raise ValueError("installed vLLM must be exactly 0.28.0")
    if fetch and apply:
        raise ValueError("fetch and apply are separate operations")
    if fetch:
        for path, expected in HASHES.items():
            dest = source / path
            if dest.exists():
                if sha(dest.read_bytes()) != expected:
                    raise ValueError(f"existing source hash mismatch: {path}")
                continue
            result = subprocess.run(["curl", "--retry", "3", "--fail", "--silent", "--show-error",
                                     "--location", BASE + path], check=True, capture_output=True)
            if sha(result.stdout) != expected:
                raise ValueError(f"downloaded source hash mismatch: {path}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(result.stdout)
    original = {}
    for path, expected in HASHES.items():
        data = (source / path).read_bytes()
        if sha(data) != expected:
            raise ValueError(f"source hash mismatch: {path}")
        original[path] = data.decode()
    if (source / HELPER).exists():
        raise ValueError("telemetry helper already exists; refusing to overwrite")
    patched = patched_sources(original)
    output.mkdir(parents=True, exist_ok=True)
    diff = []
    for path, text in patched.items():
        dest = output / "patched" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        diff.extend(difflib.unified_diff(original.get(path, "").splitlines(True), text.splitlines(True),
                    fromfile=f"a/{path}" if path in original else "/dev/null", tofile=f"b/{path}"))
    (output / "vllm-v0.28.0-admission-telemetry.patch").write_text("".join(diff))
    (output / "patched/LICENSE").write_text(original["LICENSE"])
    manifest = {"version": "0.28.0", "commit": COMMIT,
                "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source_url": BASE, "source_sha256": HASHES,
                "patched_sha256": {p: sha(s.encode()) for p, s in patched.items()},
                "checks": {"original_source_hashes": True, "compile": True,
                           "original_ast_recovered_after_removing_telemetry": True},
                "applied": apply, "GPU_used": False, "vllm_imported": False}
    if apply:
        # All files were checked and compiled before any engine source mutation.
        for path, text in patched.items():
            (source / path).write_text(text)
    (output / "patch-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Directory containing the pinned vllm/ tree and LICENSE")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--check-installed-version", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.source_root, args.output_dir, fetch=args.fetch, apply=args.apply,
                       check_installed_version=args.check_installed_version)
    print(json.dumps({"commit": manifest["commit"], "checks": manifest["checks"],
                      "applied": manifest["applied"]}, indent=2))


if __name__ == "__main__":
    main()
