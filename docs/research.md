# Research Scope

Updated: 2026-09-24.

## Core question

**Which trace patterns exhibit higher cache-hit fractions together with higher TTFT?**

First construct and identify this association on a single LLM server. Then vary request structure and arrival timing to test the conditions under which it appears. The current cache-hit metric is each request's actual `cached_tokens / prompt_tokens`. TTFT measures the time until the client receives the first streamed token ID, including transport overhead.

The cache-on/off TTFT difference for the same request is a supporting control. It measures the effect of enabling caching for that request; the primary question concerns the association between hit fraction and TTFT across requests. Both observations can hold in one experiment: requests with higher hit fractions have higher TTFT, while those same requests still have lower TTFT with caching enabled than with caching disabled.

## Experimental approach

1. **Construct request structures.** Mix short-input, many-turn (deep/short) sessions with long-input, few-turn (shallow/long) sessions. Vary shared-prefix length and session mixture.
2. **Screen candidates on CPU.** Use the association between prompt length and ideal cache reuse to select different structures, then generate traces with held-out seeds. This stage contains no measured latency evidence.
3. **Measure on GPU.** Replay fixed arrivals with prefix caching enabled. Record actual cache hit, TTFT, prompt length, session archetype, and arrival time for each request. Replay the same trace with caching disabled as a supporting control.
4. **Identify and interpret patterns.** Compute correlations separately for each trace, KV capacity, and repetition. Use archetype-level statistics and arrival timelines to inspect the roles of session mixture, prompt length, and queueing.

## Evidence to inspect

| Evidence | Purpose |
| --- | --- |
| Pearson correlation between actual hit fraction and TTFT | Identify candidates with an overall positive association |
| Spearman correlation | Inspect rank relationships without assuming that positive Pearson implies a uniformly monotonic trend |
| Correlation after linear prompt-length adjustment | Check whether prompt length explains the association; other confounders can remain |
| Correlations within each session archetype | Separate within-type trends from associations produced by mixing types |
| Arrival, prompt-length, hit, and TTFT timelines | Identify which requests arrive early and which later requests experience long delays |
| Independent seeds, repetitions, and arrival rates | Test whether the pattern persists and whether it depends on the current load |

Repeated runs of one seed are not independent workloads. Requests share session and queue dependencies, so they cannot all be treated as independent samples. A positive correlation does not establish that higher cache hit causes higher TTFT.

## Names and parameters

The existing `positive / near-zero / negative` filenames indicate the **correlation direction between prompt length and oracle reuse**. These are generation-stage labels, distinct from the measured hit–TTFT relationship. The filenames are retained to preserve references to the original experiment; reports describe the measured workloads as the "mixed-session candidate" and "deep/short comparison."

| Parameter | Current meaning |
| --- | --- |
| `P` | Shared-prefix length; the candidate pool includes 512 / 8192 tokens |
| `rho_mix` | Fraction of sessions that are long-input, few-turn sessions; this is not a measured correlation coefficient |
| `W` | Distinct blocks across the entire trace divided by nominal capacity; generator metadata only |
| `capacities` | KV block capacities explicitly configured on the GPU |
| `arrival_scale` | Multiplier for replay arrival offsets; values above 1 spread arrivals farther apart |

The serial LRU diagnostic excludes concurrency, decode, and in-flight KV allocations, so it cannot predict GPU latency. Oracle reuse cannot substitute for the actual cached-token fraction.

## Current stage and next experiment

Candidate construction and the first GPU measurements are complete. The configuration, observations, and limitations are maintained in the [first GPU experiment report](../results/cloudlab-pattern-03/report.md).

The next experiment should test the same candidate pattern:

1. Use the retained seeds 101 and 102 and add repetitions, keeping the model revision and engine settings fixed.
2. Vary `arrival_scale` separately to compare more widely spaced arrivals and test whether the positive association depends on queueing under heavy load.
3. Compare within-archetype relationships and arrival timelines. If the pattern persists, use scheduler records to investigate the queueing mechanism.

These steps remain future work. The current replay is open-loop: later turns arrive at fixed times and may precede completion of their parents. Generated model outputs are not inserted into subsequent prompts. The current evidence therefore describes a synthetic structural workload.
