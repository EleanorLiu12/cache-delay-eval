# Evaluating Cache Update Delays in LLM Request Routing

A measurement harness for studying how an LLM request router should trade off
**estimated cache locality, estimated queue depth, and state age** when choosing
a server.

It is written to be run and audited by someone who has never seen the project
before, so the sections below start from the serving background and end with
exact commands.

---

## 1. Background: why this problem exists

Skip to [§2](#2-the-research-question) if you already work on LLM serving.

When an LLM serves a prompt it does two things: **prefill**, where it processes
every prompt token at once to build the attention key/value (KV) tensors, and
**decode**, where it emits output tokens one at a time. Prefill cost grows with
prompt length, and it is what the user waits through before seeing anything.
That wait is **TTFT** — time to first token.

Serving engines such as vLLM keep the KV tensors of recent prompts in GPU
memory as a **prefix cache**, chunked into fixed-size **blocks** (16 tokens by
default). If a new prompt begins with the same tokens as a cached prompt, the
engine reuses those blocks and skips that part of prefill. Chat traffic reuses
prefixes constantly: every turn of a conversation resends the whole history, so
turn *n+1* shares a long exact prefix with turn *n*.

Now put several replicas behind a **router**. Because reuse only helps if the
request lands on the replica that actually holds its prefix, the router wants
to know who caches what. Two known designs bracket the space:

| Design | How it learns cache state | Weakness |
| --- | --- | --- |
| **Approximate** | Infers from its own past routing decisions | Silently wrong after an eviction it never saw |
| **Precise** | Maintains a global index from insert/evict events | Correct, but only as fresh as the event stream |

Both share one property: **the router's picture of the cache is never exactly
current.** Events take time to propagate, and entries are inserted and evicted
the whole time. The router routes on a slightly old map.

## 2. The research question

> **How should an LLM request router quantitatively trade off estimated
> cached-prefix length, estimated queue depth, and state age to minimize TTFT?**

The goal is not merely to show that cache locality, queueing, and stale state
affect latency. It is to derive a measurable **routing criterion**: when is a
longer cached prefix worth sending a request to a more heavily queued server,
and how should that choice change as the router's information gets older?

The terms in the question refer to specific router inputs:

- **Estimated cached-prefix length** comes from the router-visible
  **cache-index state**, which maps cache-block hashes to the servers believed
  to hold them.
- **Estimated queue depth** comes from the router-visible **load state**, such
  as the latest reported number of running and waiting requests.
- **State age** is the time since that cache-index or load state was generated.
  It is not a generic JSON `metadata` field.

The router-visible state can differ from **ground-truth state** at the servers.
For example, the index may still list a block that has already been evicted, or
the reported queue depth may no longer be current. The proposed router should
account for that uncertainty rather than treating every reported value as
equally reliable.

The main question is supported by three narrower questions:

1. **Latency calibration:** What is the TTFT cost of prompt length, actual
   cached-prefix length, and actual queue depth? This produces
   `L(prompt, cache, queue)` and quantifies how much queueing a cached prefix is
   worth.
2. **State accuracy:** How does state age affect the accuracy of the router's
   estimated cache locations and queue depths?
3. **Index scalability:** How do global KV-cache index size and update rate
   affect cache-index update delay? Index memory is recorded as a supporting
   measurement, not treated as a standalone research outcome.

The final evaluation compares the resulting state-age-aware routing criterion
with cache-only, load-only, session-affinity, and oracle routing. Its value is
measured by TTFT and by how closely its server choices approach the oracle.

The plan is trace-driven. A block-level emulator of two vLLM replicas holds
ground-truth cache and load state and emits state updates; the router receives
them immediately or after a controlled delay (100 ms – 10 s). The proposed
state-age-aware criterion is compared with round-robin, session-affinity,
cache-only, load-only, and oracle routing. Live GPUs are used only for
calibration and one validation slice — replaying the full sweep on real
replicas would cost tens of hours per configuration across ~300 configurations.

### What is in this repository *today*

This repo currently contains the **first stage** of that plan:

- ✅ **The trace format** — a versioned JSONL schema plus a validating reader,
  and a synthetic trace generator with controlled prefix reuse.
- ✅ **The TTFT calibration harness** — measures, on a real vLLM server, how
  TTFT responds to prompt length, cached-prefix length, and queue depth.
- ⏳ **Not yet here:** the cache emulator, the router, the delay injection, and
  the policy comparison. Those are later milestones.

The calibration exists because the emulator needs it. The emulator reproduces
*cache state*, not GPU timing — so to report median and p95 TTFT during replay,
it needs an empirically measured function

```text
TTFT ≈ f(prompt_tokens, cached_prefix_tokens, queue_depth)
```

That function is also what makes routing a real trade-off rather than a
one-liner: a replica holding a longer cached prefix is not automatically the
faster choice if it already has a queue. **Measuring where those two effects
cross is the point of stage one.**

---

## 3. How the calibration experiment works

### Three independent variables

| Variable | Meaning |
| --- | --- |
| `prompt_tokens` | Total length of the probe prompt, in tokens |
| `cached_prefix_tokens` | How many leading tokens are already in the KV cache |
| `queue_depth` | How many requests are ahead of the probe when it is dispatched |

**Queue depth here is not "how many requests were sent at once."** It is the
number of requests already `running` or `waiting` on the server at the moment
the probe is dispatched, confirmed from vLLM's Prometheus endpoint:

```text
queue_depth_observed = vllm:num_requests_running + vllm:num_requests_waiting
```

The server is started with `--max-num-seqs 1` so it processes one sequence at a
time and everything else genuinely queues. Without that flag the configured
number is only *offered load*, not a controlled queue depth.

### One measurement

For each `(prompt length, cached prefix length, queue depth)` condition, the
harness ([`calibration.py`](src/cache_delay_eval/calibration.py)) does:

1. Build a prompt of the exact target length via vLLM's `/tokenize`, and send
   it as **token IDs** — so prompt and prefix lengths are exact for the
   server's own tokenizer, not approximated from characters.
2. Send a **warm-up request** to install the intended prefix in the cache.
3. Launch `queue_depth` **blocker requests** to build the queue.
4. **Confirm** the observed depth from `/metrics` before proceeding.
5. Send the **probe** and time it from dispatch to the first non-empty streamed
   token.
6. Record TTFT, end-to-end latency, observed cache-hit tokens, observed queue
   depth, and any error.

Each condition repeats 30 times in the full grid.

### The controls that make it a real experiment

These are the details that separate this from a naive benchmark loop:

- **Placebo warm-up.** The cached conditions send a warm request before the
  probe, and that request perturbs scheduler state, cache occupancy, allocator
  state and CPU frequency. So the *uncached* conditions also send a warm
  request — same cost to the server, but starting from a different nonce block,
  sharing no reusable prefix. Otherwise the warm request's own side effects
  would be scored as a cache effect.
- **Per-run nonce blocks.** Every prompt's first block encodes a run code plus
  a request code, so no condition can accidentally reuse blocks left over from
  another condition — or from a previous run against a still-warm server.
- **Randomized condition order.** The schedule is shuffled with a fixed seed, so
  drift over the run (thermal, memory fragmentation) does not align with any
  one variable. Order *within* a condition stays fixed: warm → blockers →
  probe.
- **Block-aligned prefixes.** vLLM reuses whole 16-token blocks, so cached
  prefix lengths must be multiples of 16 (64, 256, 512…). Conditions that are
  not block-aligned, or that leave no unshared suffix block, are dropped as
  invalid rather than silently measured.
- **Errors are recorded, not dropped.** A failed request is written to the
  output as `status: "error"` so a run cannot quietly become a survivorship
  sample.
- **Independent audit.** [`audit.py`](src/cache_delay_eval/audit.py) re-checks
  every row: did the server report the prompt length we targeted, did it hit
  the number of cached tokens we intended, was the queue as deep as we asked?

---

## 4. Installation

Python 3.10+. The trace tools have **no third-party runtime dependencies**.

```bash
git clone https://github.com/EleanorLiu12/cache-delay-eval.git
cd cache-delay-eval
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

## 5. Quick start — trace tools (no GPU required)

Validate the bundled example and generate a synthetic workload:

```bash
cache-delay-validate-trace traces/example.jsonl
cache-delay-generate-trace --requests 1000 --output traces/synthetic-1k.jsonl
cache-delay-validate-trace traces/synthetic-1k.jsonl
```

`cache-delay-generate-trace` produces Poisson arrivals with controlled prefix
reuse — useful knobs are `--reuse-probability`, `--prefix-groups`,
`--request-rate`, and `--seed`.

## 6. Running the calibration (requires a vLLM server)

### Start a server

**Apple Silicon** — install the community Metal plugin and serve the pilot
model:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source ~/.venv-vllm-metal/bin/activate
vllm serve Qwen/Qwen3-0.6B --max-num-seqs 1 --port 8000
```

**CUDA** — start with automatic prefix caching enabled, and keep
`--max-num-seqs 1`.

Send no other inference traffic to this server while measuring.

### Pilot first

Always exercise the whole pipeline on a short run before committing to the
30-trial grid:

```bash
source .venv/bin/activate
cache-delay-calibrate --config config/calibration.pilot.json
cache-delay-audit results/calibration-pilot.jsonl
cache-delay-summarize results/calibration-pilot.jsonl \
  --output results/calibration-pilot-summary.csv
```

A healthy audit reports zero for every check except `rows`:

```json
{ "rows": 48, "errors": 0, "prompt_length_mismatches": 0,
  "cache_hit_mismatches": 0, "queue_depth_mismatches": 0 }
```

The audit exits nonzero if any check fails, so it drops straight into CI.

### Full runs

```bash
# Apple Silicon pilot stratum — 27 conditions x 30 trials = 810 rows
cache-delay-calibrate --config config/calibration.metal.json
cache-delay-audit      results/calibration-metal.jsonl
cache-delay-summarize  results/calibration-metal.jsonl \
  --output results/calibration-metal-summary.csv

# Final CUDA run — 42 conditions x 30 trials = 1260 rows
cache-delay-calibrate --config config/calibration.cuda.json
cache-delay-audit      results/calibration-cuda.jsonl
cache-delay-summarize  results/calibration-cuda.jsonl \
  --output results/calibration-cuda-summary.csv
```

`--trials 2` overrides the trial count for a quick end-to-end smoke test.

## 7. Reading the results

Raw output is append-safe JSONL, one row per probe. `cache-delay-summarize`
aggregates it to one row per condition:

| Column | What it tells you |
| --- | --- |
| `ttft_median_ms`, `ttft_p95_ms` | The measurement of interest |
| `e2e_median_ms`, `e2e_p95_ms` | End-to-end latency, same trials |
| `queue_depth_observed_median` | **Manipulation check** — did the queue actually form? |
| `prefix_cache_hit_tokens_batch_median` | **Manipulation check** — did the cache actually hit? |
| `trials_ok` / `trials_error` | How many probes survived |

The two manipulation-check columns matter as much as the TTFT columns: a TTFT
number from a condition whose cache did not hit or whose queue did not form is
measuring something other than what it claims.

Expected directions: TTFT rises with prompt length, falls with cached prefix
length, and rises with queue depth — with queueing able to **overwhelm** the
prefix-reuse benefit entirely. That crossover is the finding the router work
depends on.

### Example: the validated pilot

From `results/calibration-pilot-summary.csv` (Apple M3 Pro, vLLM 0.28.0 Metal,
Qwen3-0.6B, 48/48 rows, clean audit) — at zero queue depth:

| Prompt tokens | Cached prefix | Median TTFT |
| --- | --- | --- |
| 128 | 0 | 40.6 ms |
| 512 | 0 | 125.7 ms |
| 512 | 256 | 71.9 ms |
| 1024 | 0 | 247.1 ms |
| 1024 | 256 | 196.1 ms |

Adding just two requests ahead of the probe pushed median TTFT into the
~459–893 ms range — i.e. **queueing dwarfed every cache saving above.**

⚠️ **These are n=3 pipeline-validation numbers, not statistical results.** They
show the harness can distinguish cached from uncached prefill and can impose
queue pressure. They are not evidence about delayed router metadata, which
depends on milestones not yet in this repo.

## 8. The trace format

Workloads are UTF-8 JSONL: a `trace_meta` header, then one record per request.
[`schemas/request-trace-v1.schema.json`](schemas/request-trace-v1.schema.json)
is normative; [`schemas/README.md`](schemas/README.md) explains the design.

Each request carries **exactly one** prompt form, matching what its source
workload can actually publish:

| Form | Source | Why |
| --- | --- | --- |
| `prompt` | ShareGPT, WildChat | Portable text; needs tokenizing to replay at block level |
| `token_ids` | Pre-tokenized corpora | Exact block-level replay |
| `block_hashes` | Mooncake | Production traces cannot publish user prompts — per-block fingerprints are all a prefix-cache study needs |

Arrival times are integer millisecond **offsets** from the trace start, not
wall-clock stamps, so replay is deterministic with no float ambiguity about the
order of two nearby requests.

Optional identity fields (`session_id`, `user_id`, `parent_request_id`) are what
make the reuse breakdown possible — separating intra-session reuse, which plain
session affinity already captures, from **inter-session** reuse, the regime
where a global cache index should actually earn its cost. `prefix_group` is
ground truth for scoring synthetic runs and **must never be read by a routing
policy.**

## 9. Repository layout

```
src/cache_delay_eval/
  calibration.py     # the TTFT experiment: conditions, controls, probes
  http_client.py     # dependency-free vLLM client (streaming, tokenize, metrics)
  trace.py           # trace reader/writer + validation beyond JSON Schema
  generate_trace.py  # synthetic workloads with controlled prefix reuse
  summarize.py       # raw JSONL -> tidy per-condition CSV
  audit.py           # verifies the experimental invariants held
config/              # pilot / Metal / CUDA calibration grids
schemas/             # normative trace schema + format rationale
traces/              # format examples (text form and hash-only form)
results/             # raw measurements, kept in version control
tests/               # unittest suite
```

## 10. Reproducibility rules

- **Do not compare Metal pilot numbers with CUDA numbers.** Different backends;
  the Metal run is a stratum, not a substitute.
- Keep `--max-num-seqs 1` whenever queue depth is an experimental variable.
- Run one benchmark client, and no unrelated inference traffic on the server.
- Record the server command, model revision, vLLM version, hardware, and config
  with every run. The harness stamps most of this into each row automatically.
- **Preserve raw JSONL. Never hand-edit measurements** — regenerate tables from
  the raw file instead.
- Output is opened in append mode. Use a fresh filename per run so two runs
  cannot be silently interleaved.
- An interrupted run has incomplete trial counts; do not treat it as final.

## 11. Limitations

This stage calibrates **one** server. It does not yet demonstrate anything
about cache-update delay — that conclusion requires the multi-replica routing
experiment, which is future work. Nothing here should be cited as a result
about routing.

## License

MIT — see [LICENSE](LICENSE).
