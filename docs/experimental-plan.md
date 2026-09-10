# Experimental Plan

## 1. Objective

A router in a large language model (LLM) serving system places requests using prefix-cache and load information that reaches it through an update path, so its view lags the replicas it routes to. The study quantifies when that lag causes the router to choose a slower replica.

This study targets two results: a response function that maps cache-state and load-state age to routing regret and time to first token (TTFT), and a failure boundary on that function separating negligible from material harm across different workloads and system conditions.

## 2. Research questions

### Main Research Question

How do stale prefix-cache state and replica-load state affect cache reuse, routing regret, and TTFT in LLM routing, and when do these effects become material across update delays, workloads, and system conditions?

### Supporting Question 1: How does router-visible state become inaccurate?

Measure how cache-state age and load-state age change cache-location error,
queue-depth error, router selection, and latency. Cache-state age and load-state
age are the elapsed times since the router-visible cache and load state were
generated at the replica; Section 3 defines them.

### Supporting Question 2: Which workloads are vulnerable?

Vary prefix reuse, request rate, cache capacity, and the fraction of reuse that crosses session boundaries.

Before confirmatory runs, define the minimum TTFT and routing-regret differences
that count as material; do not choose this threshold after inspecting results.

## 3. System model

The main replay uses two simulated inference replicas. Each replica has:

- a **block-level key-value (KV) cache** with finite capacity and an explicit
  eviction policy;
- a **request queue** and service state;
- a ground-truth cache and load view used only by the emulator and oracle; and
- cache and load updates that reach the router after controlled delays.

For request `r` and replica `i`, define:

- `P_r`: prompt length;
- `C_ri`: actual number of reusable prefix tokens;
- `Q_i`: actual number of running plus waiting requests;
- `Ĉ_ri`, `Q̂_i`: the router's estimates; and
- `A^cache_i`, `A^load_i`: the ages of those estimates.

Age is measured from the state or event generation timestamp, not from when the router receives it.

The calibrated latency surface is:

```text
L(P_r, C_ri, Q_i) -> TTFT distribution
```

Routing compares replicas, so it needs one number rather than a distribution.
Write `L̄` for the expected TTFT under `L`. The choice of summary is fixed
before any policy comparison, and every policy and the oracle use the same one.
The full distribution is retained for the tail metrics in Section 7.

Let `i*` be the replica chosen for request `r`. Routing regret is the extra
expected TTFT caused by not choosing the fastest replica:

```text
regret_r = L̄(P_r, C_ri*, Q_i*) - min_i L̄(P_r, C_ri, Q_i)
```

Both terms use ground-truth state, so regret measures decision quality rather
than workload difficulty, and it is zero when the router chooses the fastest
replica. Section 7 gives the reporting form.

## 4. Routing policies

All policies replay the same trace and differ only in replica selection.

| Policy | Decision |
| --- | --- |
| Round-robin | Alternate replicas without reading cache or load state. |
| Session affinity | Hash each session to one replica. |
| Approximate cache-aware | Predict prefix-cache placement from prior routing decisions rather than cache events. |
| Immediate event-driven | Use cache and load updates without injected delay. |
| Delayed event-driven | Use the same routing rule after controlled insertion, eviction, and load-state delays. |
| Oracle | Minimize expected TTFT using current ground-truth cache and queue state. |

The policies use the router-visible state to estimate:

```text
score(i) = L̄(P_r, Ĉ_ri, Q̂_i)
```

The score is predicted latency, so the policies select the replica with the
**lowest score**, and they do so without compensating for state age. This keeps
the routing rule fixed while the experiment varies update delay. The oracle
applies the same `L̄` to `C_ri` and `Q_i`, so routing regret isolates error
caused by the router-visible state.


## 5. Experimental stages

### Stage 1: Validate traces

For every workload, record its origin, license, sampling procedure, prompt form,
arrival-time semantics, session identifiers, tokenizer, and block size. Validate
the converted JSONL.

Before using a trace, report request and session counts; prompt and output length
distributions; arrival rate and burstiness; intra-session and inter-session
prefix reuse; and the fraction of requests with at least one reusable full
block.

### Stage 2: Calibrate live TTFT

Measure TTFT on one vLLM server across prompt length, actual cached-prefix
length, and actual queue depth. Randomize condition order and repeat each valid
condition. Audit target prompt length, observed cache-hit tokens, observed load,
missing metrics, and execution errors before fitting `L`.

Use separate calibration and validation samples. Start with an interpretable
lookup or interpolation method and add model complexity only if held-out error
requires it. Report median absolute error and tail error by condition. Generate
per-request replay latencies before computing replay tail quantiles.

### Stage 3: Validate the emulator

Replay controlled sequences that test insertion, longest-prefix matching,
capacity pressure, eviction, and repeated access. Compare emulator hit tokens
and evictions with vLLM KV events under matched block size, cache capacity, and
policy.

Acceptance target: cache-hit and eviction counts differ by no more than 5% on
the controlled validation trace. If this cannot be met, document the semantic
difference and include an eviction-policy sensitivity analysis.

### Stage 4: Characterize stale state

Run fixed-delay experiments first, with insertion, eviction, and load delays
independently set to `0`, `0.1`, `1`, and `10` seconds. Delayed insertions create
false negatives; delayed evictions create false positives, so combining them too
early would hide the mechanism. Add intermediate log-spaced values around the
observed performance knee rather than expanding the entire grid. After the
controlled sweep, test a variable or batched delay distribution derived from
measured update-path behavior if that measurement is available.

For each request, record cache-state errors, queue-depth error, selected replica,
oracle replica, and latency difference from the oracle. Plot outcomes against
both absolute delay and normalized delay:

```text
cache delay / median cache-entry lifetime
load delay  / characteristic queue-change time
```

### Stage 5: Cross-policy characterization

Run all policies on identical traces, seeds, capacities, arrival schedules, and
delay realizations. Use paired per-request or paired per-seed results to show
which policies are sensitive to stale state and when simpler policies match or
outperform event-driven routing.

### Stage 6: Live validation

If two live replicas are available, select one representative condition before
examining its live result. Treat this as a check on the emulator's conclusion,
not as a second full sweep.

## 6. Workloads and controlled factors

| Workload | Role |
| --- | --- |
| Mooncake | Primary production-derived block-level workload, subject to schema verification before conversion. |
| WildChat-1M | Multi-turn chat and session-affinity comparison. |
| Synthetic | Controlled sweeps over reuse, arrival rate, cache pressure, and inter-session sharing. |
| ShareGPT | One comparability run with prior serving evaluations. |

| Factor | Planned values or treatment |
| --- | --- |
| Insertion/eviction delay | 0, 0.1, 1, 10 s; vary separately, then jointly. |
| Load update delay | Same grid, varied separately before joint-delay runs. |
| Offered load | Low, medium, high relative to calibrated service capacity. |
| Cache pressure | Capacities chosen to produce low, medium, and high eviction rates. |
| Prefix reuse | Controlled shared-prefix length and reuse probability. |
| Inter-session reuse | Controlled fraction of reusable prefixes shared across sessions. |
| Randomness | At least three seeds initially; add seeds where intervals remain wide. |

First locate the delay knee on one reference workload. Then run one-factor sweeps and the selected interactions `delay × load`, `delay × cache pressure`, and `delay × inter-session reuse`. Finally, test whether the resulting boundary appears in the real traces.

## 7. Metrics

### Primary outcomes

- **TTFT:** mean, median, p95, and p99 when sample size supports it.
- **Routing regret:** defined in Section 3. The counterfactual replica is never
  actually run, so regret is defined on the latency model rather than on
  realized draws. Report mean, p95, and the fraction above zero.
- **Service level objective (SLO) violation:** fraction above a threshold
  declared before policy comparison. If no defensible SLO exists, report a
  latency-deadline curve.
- **Reused prompt tokens:** total and fraction of eligible prompt tokens.

### Supporting outcomes

- cache-location precision and recall by cache-state age;
- absolute queue-depth estimation error;
- throughput and replica load imbalance;
- cache-entry lifetime and eviction rate;
- cache- and load-state update count and bytes;
- global-index memory; and
- router CPU time and decision latency.

A route is incorrect only relative to the TTFT oracle. A cache miss alone does
not make a route incorrect when another replica has a shorter queue.

## 8. Statistical and reporting protocol

- Preserve raw per-request results with configuration identifiers, and report
  request counts, runs, errors, and exclusions.
- Use common seeds and identical request sequences across policies, so that
  every policy comparison is paired.
- Report paired differences as bootstrap intervals resampled by replay seed or
  session, not as point estimates; dependent turns from one conversation are
  not independent samples.
- Separate exploratory runs used to choose ranges from confirmatory runs used
  for final claims.
- Distinguish live measurements from emulator-derived results.

## 9. Completion criteria

The characterization is complete when the study identifies where stale state
has negligible and material effects, reproduces the boundary across seeds,
checks it on at least one real workload, and reports primary outcomes and
overhead for all baselines. Claims remain limited to the workloads, model,
backend, replica count, and load regimes actually tested.
