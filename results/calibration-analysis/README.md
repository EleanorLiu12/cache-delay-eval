# Week 1–2 calibration results

## Accepted inputs

- `results/calibration-metal-confirmatory.jsonl`: 810 calibration observations,
  27 conditions, 30 trials per condition. All requests succeeded; reported
  prompt length, cache-hit tokens, and queue depth matched their targets.
- `results/calibration-metal-validation.jsonl`: 135 independent validation
  observations, 27 conditions, 5 trials per condition, seed 1701. All requests
  and manipulation checks passed.

The empirical lookup uses mean TTFT for routing comparisons. Figures also show
condition medians, raw observations, bootstrap 95% confidence intervals, and
p95 TTFT.

## Validation results

| Measure | Condition-level error |
| --- | ---: |
| Mean TTFT MAE | 21.81 ms |
| Mean TTFT MAPE | 1.31% |
| Median TTFT MAE | 25.05 ms |
| Median TTFT MAPE | 1.07% |
| p95 TTFT MAE | 55.48 ms |
| p95 TTFT MAPE | 3.48% |

Representative zero-queue calibration medians were 38.6 ms for a 128-token
uncached prompt, 118.5 ms for a 512-token uncached prompt, and 240.1 ms for a
1024-token uncached prompt. Caching 64, 256, and 512 tokens respectively reduced
those medians to 26.1, 69.9, and 137.6 ms.

Queueing dominated cache savings. For a 1024-token uncached prompt, median TTFT
rose from 240.1 ms at queue depth 0 to 1265.8 ms at depth 2 and 5140.3 ms at
depth 8. Caching 512 tokens at those depths produced medians of 137.6, 1157.1,
and 4988.3 ms.

## Protocol checks and excluded runs

The first validation attempt exposed a prompt-generation defect: repetitive
reservoir offsets allowed warm and probe suffixes to share 32 unintended tokens.
The harness now places distinct nonce blocks immediately after the intended
shared prefix (`suffix-nonce-v2`). The failed validation remains preserved as
`results/calibration-metal-validation-invalid-prefix.jsonl` and is excluded.

The accepted 810-row calibration used the earlier suffix construction, but its
observed cache hits exactly matched every target. A 624-row pre-slowdown v2
sensitivity window differed from its condition medians by 1.86% on average and
3.60% at most. Later v2 attempts encountered a system-wide latency regime shift
while desktop GPU load was high. They remain preserved as files containing
`aborted-system-load` and are excluded from the lookup and figures.

## Outputs

- `calibration-lookup.csv`: empirical TTFT lookup and held-out errors.
- `calibration-validation-report.json`: machine-readable audit and error summary.
- `calibration-response.{png,pdf}`: raw TTFT, medians, and confidence intervals.
- `cache-benefit.{png,pdf}`: median TTFT savings from prefix reuse.
- `calibration-validation.{png,pdf}`: calibration estimates versus validation observations.
