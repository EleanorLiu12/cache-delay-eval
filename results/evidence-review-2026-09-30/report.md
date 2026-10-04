# Evidence review and corrected research plan

Started September 30; completed October 1, 2026. Repository materials are in English. This review removes misleading presentation material and checks retained measurements without adding a new pattern definition or application classifier.

## What is established

**The target trace class has not yet been counted in real applications.** Existing numbers describe input overlap, conversation structure and source-label composition. The withdrawn chart did not establish the prevalence of the class observed in the GPU experiment.

### What last week's report actually says

The [original report](../cloudlab-pattern-03/report.md) states:

> Long prompts arrived early. Later requests often reused more cached context but still waited long for their first token.

The same report describes a comparison between mixed long/short prompts and short-input, many-turn sessions. The retained mixed-trace generator metadata specifies 42 long-input sessions with three turns each and ten short-input sessions with 30 turns each, totaling 426 requests. The original source is [P512-negative-seed100.jsonl](../pattern-screen/traces/P512-negative-seed100.jsonl); these are synthetic design labels.

The research object is a **class of traces**, while the evidence comes from this generated seed under multiple capacities and repetitions. The report does not define the class as "at least 50% input overlap and another chain's long request in the last 10 seconds." That condition came from a later auxiliary analysis. A classifier for the full class, including the session mixture and arrival sequence, remains to be specified.

The original description supports early long inputs followed by greater cache reuse and long TTFT. It does not prove that every later request is a short-input continuation, or that the long inputs caused its delay.

## Cleanup completed

| Removed material | Reason |
| --- | --- |
| September 30 meeting report, figures, derived membership data and supporting notes | The presentation centered a proxy that did not identify the target trace class. |
| Initial Qwen A scratch outputs | Preliminary derivative data superseded by the retained, independently checked analysis. |
| Older October 1 meeting brief | Redundant presentation with obsolete replay-implementation prerequisites. |

The cleanup removed **21 files, 59,358,150 bytes**, with paths and SHA-256 values recorded in [cleanup.json](cleanup.json). No original dataset was deleted. The live scope, progress and plan documents were consolidated.

The original GPU results, canonical CPU results, source manifests, historical code snapshots and current replay/Oracle evidence remain. The old admission source and Oracle window are retained because tests import them. Preliminary temporal outputs remain dependencies of the historical packaging comparison. Older full reports remain dated numerical evidence; their implementation plans are superseded by the current progress page.

## Datasets actually used

| Dataset | Retained scope | What was checked | What it cannot establish |
| --- | --- | --- | --- |
| Qwen-Bailian A | All 43,058 released requests; 31,744-request text cohort | Source integrity, arrivals, parent links, source types and input-prefix overlap | Actual cache hits, TTFT, shared queue, completion or coding/document purpose |
| Qwen-Bailian B | All 172,800 released requests; API and text analyzed separately | Source integrity, arrivals, types and prefix overlap; all observed chains are singletons | Multi-turn behavior from missing lineage, actual hits/TTFT or task purpose |
| WildChat | Fixed 200-conversation sample from one 59,857-row shard | Content/role consistency, conversation depth and reconstructed prompt lengths | All 606 user turns lack arrival timestamps; no production hit/TTFT |
| BurstGPT | Schema inspection only | Available field names and their meanings | No completed local corpus statistics or target-pattern prevalence |
| ServeGen | Schema inspection only | Bounded sample/schema review | No completed prevalence result; time/input representation unresolved |

Thus, the locally analyzed data are **two Qwen releases and one bounded WildChat sample**, not five completed application-prevalence studies. No new source scan or download was performed.

### Verified source-label composition

These percentages describe the source's request labels. They are **not target-pattern rates** and do not identify application purpose.

| Source | Source label | Requests / all source requests | Share |
| --- | --- | ---: | ---: |
| A | text | 31,744 / 43,058 | 73.72% |
| A | search | 8,187 / 43,058 | 19.01% |
| A | image | 1,617 / 43,058 | 3.76% |
| A | file | 1,510 / 43,058 | 3.51% |
| B | api | 150,936 / 172,800 | 87.35% |
| B | text | 21,864 / 172,800 | 12.65% |

The literal source labels are retained: image does not cover all possible multimodal use, file is not a verified document-reading task, and text/API traffic can include many purposes. These mappings cannot be recovered from anonymized token hashes.

**Coding, document-reading and other task-purpose categories were not counted.** WildChat has content that could support a bounded future review, but no such classification was completed. Its missing arrivals and hit/TTFT measurements would remain missing after classification.

### Valid structural results that remain useful

| Measurement | Verified result | Interpretation |
| --- | --- | --- |
| A text chains with multiple requests | 6,739 / 16,447 = 40.97% | Observed multi-request structure; not complete real-session coverage |
| A text historical input overlap >=50% | 22,720 / 31,744 = 71.57% | Potential prefix reuse across all prior cohort inputs |
| B API historical input overlap >=50% | 103,444 / 150,936 = 68.54% | Cross-component overlap; all observed components are singletons |
| B text historical input overlap >=50% | 8,442 / 21,864 = 38.61% | Separate cohort history and denominator |
| WildChat multi-turn conversations | 110 / 200 = 55.00% | Fixed sample from one shard; not whole-corpus prevalence |
| WildChat missing user arrival times | 606 / 606 | Prevents evaluation of the temporal trace class from original timing |

The old 1/10/60-second temporal comparisons remain valid **for their stated proxy**. They do not establish the target class, and neither a high nor a low proxy rate answers the application-prevalence question. The overall target-class rate is unknown.

## GPU waiting and Oracle evidence

### Historical GPU measurements

All 16 retained pilot runs have per-request cache counts and client TTFT without recorded request errors. For the mixed workload, all four cache-on runs have positive Pearson correlations, ranging from **+0.5224 to +0.5304**. They share one generated seed; repetitions are not independent application samples.

In the representative run, mean server TTFT is **541.4764 s**, including **540.5281 s before first scheduling**. All **374/374 children** were dispatched before parent completion. The preemption counter increased by **14**, without per-request cause attribution.

The raw prefill metric is **0.88475 s** on average. It does not exhaust the complete interval after queueing; the old report's short label should not be treated as an exact decomposition of all server TTFT. Aggregate metrics cannot assign a waiting cause to each slow request.

### Oracle status

The eviction Oracle has future workload knowledge and changes legal cache-victim selection under fixed scheduling, admission and progress rules. It has an implemented restricted CPU solver.

The retained 0%, 5% and 10% progress-budget results are three points on **one** four-request, five-block instance. Stock and Oracle both have mean TTFT **2,525 model ticks**, so regret is **0% at every point**. These support a flat CPU budget-sensitivity plot, not a result across different eviction policies or hardware conditions. Costs are synthetic; hardware validity remains unknown.

Replay and telemetry are implemented and CPU-checked. The existing mock replay completed **220 turns across 80 sessions**, with fabricated replies and zero cache counts. Those are correctness checks, not serving measurements. Other eviction-policy regret comparisons remain unperformed.

## Updated plan

The corrected research sequence is:

1. Match a written counting rule to the actual historical trace class, including session mixture and arrival order. Freeze its counting unit, boundaries and thresholds before real-data prevalence.
2. Keep source labels separate from task purpose. If purpose-specific analysis is needed, review the existing fixed content sample with a declared guide, an unclear category and all category denominators. Report categories regardless of whether the total looks large or small.
3. Use missing-field coverage to determine which quantities each dataset can support. Do not infer actual hits from input overlap or turn synthetic replay rates into production rates.
4. On **October 2**, run **8 smoke sessions → 8 independent calibration sessions → 64 baseline sessions**. Check request IDs, dialogue dependencies, eviction/reuse records and logging overhead. Validate complete-iteration costs and state transitions before hardware regret.
5. For policy curves, declare policies, workloads and the swept variable, then compute the complete set of points. Preserve zero gains, failures and search bounds. CPU computation need not wait for GPU, but hardware claims require validation.

Classification is necessary to assess differences between application purposes when the data support it. It cannot repair the missing target definition or absent timing/cache observations. No category should be selected only because its measured rate is favorable.

## Verification and reproduction

The [independent validator](independent-validation/validation.json) passed: raw A/B structure, complete retained evidence and aggregate/stratum checks, plus 80 exhaustive prefix queries per temporal cohort. The exhaustive prefix check is sampled; it is not claimed for every request.

[Verified statistics](verified-statistics.json) retain counts, denominators, source fields, classification status, GPU measurements and Oracle points. [Source manifest](source-manifest.json) records inputs. The [preservation baseline](preserved-file-hashes.json) covers **420 retained data/result files** after cleanup.

~~~sh
.venv/bin/python results/evidence-review-2026-09-30/review.py
~~~

The independent validation command is retained in its JSON output. Use a fresh output directory for another run. Original multilingual conversation content is preserved; authored repository documents are in English.
