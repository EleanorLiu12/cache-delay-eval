# Trace pattern screening

Goal: find traces with higher actual cache hit and higher TTFT. This CPU stage only selects workload structures for GPU measurement.

The positive / near-zero / negative labels refer to prompt length versus oracle prefix reuse, NOT cache hit versus TTFT. Keep the original filenames for reproducibility.

Selection uses the existing 30 exploration seeds and keeps the original three-turn long-session archetype. Validation uses separate seeds; none are discarded.

| P | Candidate | Mix | Held-out Pearson range | Held-out Spearman range |
|---|---|---|---|---|
| 512 | positive | 0.0 | 0.579 … 0.587 | 0.968 … 0.972 |
| 512 | near-zero | 0.3 | -0.147 … 0.025 | 0.872 … 0.913 |
| 512 | negative | 0.8 | -0.354 … -0.345 | 0.524 … 0.706 |
| 8192 | positive | 0.0 | 0.191 … 0.194 | 0.761 … 0.777 |
| 8192 | near-zero | 0.1 | -0.059 … 0.191 | 0.672 … 0.761 |
| 8192 | negative | 0.75 | -0.482 … -0.441 | -0.178 … 0.084 |

## Interpretation

Negative Pearson with positive Spearman is not a uniformly negative length/reuse relationship. Examine within-archetype and pooled effects separately.

Oracle reuse assumes earlier requests have already populated an unlimited cache. It is not actual GPU hit rate. The finite-cache diagnostic counts only a consecutive cached prefix; it excludes decode tokens, in-flight allocations, preemption, scheduling, and request durations. It cannot predict hit/TTFT correlations.

W uses distinct blocks over the entire trace, not the concurrent working set. W > 1 alone does not establish harmful thrashing. Fixed capacities in manifest.json are diagnostic inputs, not verified A30 budgets.

## GPU measurement

Use the [GPU protocol](../../docs/pattern-pilot.md) to replay fixed arrivals and collect per-request cached_tokens / prompt_tokens and TTFT with caching enabled. Analyze each capacity and repetition separately, including within-archetype associations and arrival timelines. Cache-off runs provide a supporting control.

The first measured subset is P512 positive/negative, seed 100. See the [GPU report](../cloudlab-pattern-03/report.md). The remaining candidates have no GPU evidence yet.
