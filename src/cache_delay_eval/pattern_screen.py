"""CPU-only structural screening; never estimates TTFT or vLLM scheduling.

Select mixture settings on exploration seeds, then export all held-out seeds.
The finite-cache diagnostic is a serial, prompt-only block LRU with contiguous
prefix lookup. It has no active-request pinning, decoding, or concurrency.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import OrderedDict, defaultdict
from pathlib import Path

from .session_gen import GeneratorConfig, WorkloadKnobs, build_trace, _pearson, _ranks
from .trace import TraceRequest, write_trace


def serial_lru(requests: list[TraceRequest], capacity: int) -> dict:
    if capacity < 1:
        raise ValueError("capacity must be positive")
    cache: OrderedDict = OrderedDict()
    rates, lengths = [], []
    hits_total = blocks_total = evictions = 0
    oversized = 0
    for request in requests:
        blocks = request.block_hashes
        if not blocks:
            raise ValueError("screening requires full prefix block hashes")
        hits = 0
        for block in blocks:
            if block not in cache:
                break
            hits += 1
        rates.append(hits / len(blocks))
        lengths.append(request.prompt_tokens)
        hits_total += hits
        blocks_total += len(blocks)
        oversized += len(blocks) > capacity
        # Serial diagnostic only: completed prompt accesses in prefix order.
        for block in blocks:
            cache[block] = None
            cache.move_to_end(block)
            if len(cache) > capacity:
                cache.popitem(last=False)
                evictions += 1
    return {
        "capacity_blocks": capacity,
        "request_mean_hit_rate": statistics.mean(rates),
        "block_weighted_hit_rate": hits_total / blocks_total,
        "rho_pearson": _pearson(lengths, rates),
        "rho_spearman": _pearson(_ranks(lengths), _ranks(rates)),
        "evicted_blocks": evictions,
        "prompts_larger_than_capacity": oversized,
    }


def select_cells(rows: list[dict]) -> list[dict]:
    """Keep the original three-turn archetype; select means, never lucky seeds."""
    groups = defaultdict(list)
    for row in rows:
        if int(row["shallow_turns"]) == 3:
            groups[(int(row["P"]), float(row["rho_mix"]))].append(row)
    cells = []
    for (prefix, mix), group in sorted(groups.items()):
        pearson = [float(r["rho"]) for r in group]
        spearman = [float(r["rho_spearman"]) for r in group]
        cells.append(dict(P=prefix, rho_mix=mix,
                          exploration_rho_mean=statistics.mean(pearson),
                          exploration_rho_min=min(pearson),
                          exploration_rho_max=max(pearson),
                          exploration_spearman_mean=statistics.mean(spearman)))
    selected = []
    for prefix in sorted({c["P"] for c in cells}):
        candidates = [c for c in cells if c["P"] == prefix]
        for label, key in (
            ("positive", lambda c: -c["exploration_rho_mean"]),
            ("near-zero", lambda c: abs(c["exploration_rho_mean"])),
            ("negative", lambda c: c["exploration_rho_mean"]),
        ):
            selected.append(dict(min(candidates, key=key), label=label))
    if not selected:
        raise ValueError("no three-turn exploration cells found")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--capacities", type=int, nargs="+", default=[2048, 4096])
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists; use a new path to preserve results")
    if any(c < 1 for c in args.capacities):
        parser.error("capacities must be positive")
    with args.profile.open() as stream:
        rows = list(csv.DictReader(stream))
    if set(args.seeds) & {int(r["seed"]) for r in rows}:
        parser.error("validation seeds must not overlap exploration seeds")
    cells = select_cells(rows)
    args.output_dir.mkdir(parents=True)
    results = []
    for cell in cells:
        for seed in args.seeds:
            header, requests, stats = build_trace(
                WorkloadKnobs(P=cell["P"], rho_mix=cell["rho_mix"]),
                GeneratorConfig(seed=seed),
            )
            name = f"P{cell['P']}-{cell['label']}-seed{seed}"
            write_trace(args.output_dir / "traces" / f"{name}.jsonl", header, requests)
            result = dict(cell, seed=seed, trace=f"traces/{name}.jsonl",
                          requests=len(requests), oracle_rho=stats["rho"],
                          oracle_spearman=stats["rho_spearman"],
                          working_set_blocks=stats["working_set_blocks"],
                          max_prompt_tokens=max(r.prompt_tokens for r in requests),
                          serial_lru=[dict(serial_lru(requests, c),
                                           working_set_ratio=stats["working_set_blocks"] / c)
                                      for c in args.capacities])
            results.append(result)
    manifest = {
        "purpose": "structural candidates for GPU measurement of actual cache hit versus TTFT",
        "rho_definition": "prompt length versus unbounded historical prefix reuse",
        "finite_cache_model": "serial prompt-only LRU; not a vLLM scheduler emulator",
        "capacity_note": "Fixed diagnostic capacities override generator W metadata; GPU feasibility unverified.",
        "selection": "three-turn archetype, exploration means; all held-out seeds exported",
        "results": results,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(args.output_dir / "report.md", results, cells)
    print(f"Exported {len(results)} candidates to {args.output_dir}")
    return 0


def write_report(path, results, cells):
    """Keep generated screening notes aligned with the GPU research question."""
    lines = ["# Trace pattern screening", "",
             "Goal: find traces with higher actual cache hit and higher TTFT. This CPU stage only selects workload structures for GPU measurement.", "",
             "The positive / near-zero / negative labels refer to prompt length versus oracle prefix reuse, NOT cache hit versus TTFT. Keep the original filenames for reproducibility.", "",
             "Selection uses the existing 30 exploration seeds and keeps the original three-turn long-session archetype. Validation uses separate seeds; none are discarded.", "",
             "| P | Candidate | Mix | Held-out Pearson range | Held-out Spearman range |",
             "|---|---|---|---|---|"]
    for cell in cells:
        group = [r for r in results if r["P"] == cell["P"] and r["label"] == cell["label"]]
        def span(key):
            return f"{min(r[key] for r in group):.3f} … {max(r[key] for r in group):.3f}"
        lines.append(f"| {cell['P']} | {cell['label']} | {cell['rho_mix']} | {span('oracle_rho')} | {span('oracle_spearman')} |")
    lines += ["", "## Interpretation", "",
              "Negative Pearson with positive Spearman is not a uniformly negative length/reuse relationship. Examine within-archetype and pooled effects separately.", "",
              "Oracle reuse assumes earlier requests have already populated an unlimited cache. It is not actual GPU hit rate. The finite-cache diagnostic counts only a consecutive cached prefix; it excludes decode tokens, in-flight allocations, preemption, scheduling, and request durations. It cannot predict hit/TTFT correlations.", "",
              "W uses distinct blocks over the entire trace, not the concurrent working set. W > 1 alone does not establish harmful thrashing. Fixed capacities in manifest.json are diagnostic inputs, not verified A30 budgets.", "",
              "## GPU measurement", "",
              "Use the [GPU protocol](../../docs/pattern-pilot.md) to replay fixed arrivals and collect per-request cached_tokens / prompt_tokens and TTFT with caching enabled. Analyze each capacity and repetition separately, including within-archetype associations and arrival timelines. Cache-off runs provide a supporting control.", "",
              "The first measured subset is P512 positive/negative, seed 100. See the [GPU report](../cloudlab-pattern-03/report.md). The remaining candidates have no GPU evidence yet.", ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
