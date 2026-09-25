# Experiment Run Guide

See the [research scope](research.md) for the objective and interpretation criteria, and the [first GPU experiment report](../results/cloudlab-pattern-03/report.md) for the completed configuration and results. This page documents reproduction steps.

## Install and check

Use Python 3.10+ and run the following commands from the `project/` root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[live,plots]'
python -m unittest discover -s tests -v
```

Local tests and run planning do not require a GPU, vLLM, or a model download.

## Generate and screen candidates

```bash
cache-delay-generate-sessions --prefix-tokens 512 --rho-mix 0.8 \
  --seed 100 --output results/scratch-session.jsonl
cache-delay-validate-trace results/scratch-session.jsonl

cache-delay-screen-patterns --profile results/session-gen/rho-profile.csv \
  --output-dir results/scratch-screen
```

Screening selects settings using means across 30 exploration seeds, retains the original three-turn long-session archetype, and exports all held-out seeds 100–102. The `positive/negative` labels describe prompt length versus oracle reuse; the research objective requires GPU measurements of actual hit and TTFT. Outputs include `manifest.json`, `report.md`, and `traces/`.

`scripts/profile_rho.py` reproduces the exploration profile: two prefix lengths, three shallow-session depths, and 30 seeds. It overwrites `results/session-gen/rho-profile.csv`, so run it only when repeating the exploration stage.

## Run the GPU experiment

The single default configuration is `config/pattern-pilot.json`: Qwen3-4B, bf16, vLLM 0.28.0, two P512 seed100 traces, 2048/3072 KV blocks, and two repetitions per condition. Including both caching conditions gives 16 separate server runs. The model revision is pinned to the completed experiment, and the default timeout is the 2400 seconds used in that successful run.

```bash
# Generate a plan only; use a new output directory.
cache-delay-pattern-pilot --config config/pattern-pilot.json \
  --output-dir results/scratch-plan

# Execute on a GPU node with working CUDA and driver support.
python -m pip install 'vllm==0.28.0'
cache-delay-pattern-pilot --config config/pattern-pilot.json \
  --output-dir results/cloudlab-next --execute
```

The node needs the project source, configuration, and `results/pattern-screen/traces/`. Startup checks feasibility on the actual GPU. Do not silently shorten traces, switch models, or change capacity to bypass a failure. Changes to experimental conditions require a separate configuration and result directory.

The previous A30 node lacked `nvcc`. Its runs set `VLLM_USE_FLASHINFER_SAMPLER=0` to bypass compilation of an optional sampler during startup. Set that variable before running when reusing the same environment. Logs from both failed and successful runs are retained in the result directories.

## Measurement protocol

- `materialize.py` encodes synthetic block identities as fixed token sequences, preserving full-block equality, partial blocks, and session prefix extension. Paired runs reuse the same file and SHA-256 digest. Prompts consist of synthetic tokens without natural-language meaning.
- `live_replay.py` sends requests concurrently at fixed offsets and records dispatch lag, TTFT, completion time, actual input/output token counts, and cached tokens. It detects the first output using token IDs, so empty decoded text does not delay the measurement. Output length is fixed and verified.
- `paired_pilot.py` starts a fresh server for every condition, verifies caching behavior during warmup, clears the cache before measurement, and alternates condition order across repetitions. The model revision stays fixed throughout the suite. Source snapshots, commands, package versions, model configuration, and GPU information are retained.
- KV block size=16, max context=32768, max sequences=16, chunked prefill enabled, and max batched tokens=2048. Record any parameter change in the new result directory's `plan.json`.
- `analyze_patterns.py` checks suite completeness, trace identity, paired engine configurations, request failures, cache usage, and dispatch lag before reporting each cache-on run's hit–TTFT relationship.

## Analysis and figures

```bash
cache-delay-analyze-patterns results/cloudlab-next \
  --output results/cloudlab-next/analysis.json
python scripts/plot_gpu_pilot.py \
  --analysis results/cloudlab-next/analysis.json \
  --output-dir results/cloudlab-next/figures
```

Analysis refuses to overwrite existing output. To recompute the retained successful experiment, write to a temporary directory:

```bash
mkdir -p tmp/reanalysis
cache-delay-analyze-patterns results/cloudlab-pattern-03 \
  --output tmp/reanalysis/analysis.json
python scripts/plot_gpu_pilot.py \
  --analysis tmp/reanalysis/analysis.json \
  --output-dir tmp/reanalysis/figures
```

In `analysis.json`, `patterns` summarizes positive associations across repetitions; `summaries` contains per-run Pearson, Spearman, prompt-length-adjusted, and within-archetype statistics; and `requests` retains per-request measurements and arrival times. Positive Pearson is a descriptive candidate flag that must be interpreted alongside the other evidence.

The `cache-hit-ttft` figure separates traces and KV capacities and computes correlations per run. The shorter `cache-hit-ttft-representative` figure shows one run per trace for the weekly report. The `trace-pattern-timeline` figure shows the arrival structure of a representative mixed-session run. Figures are saved as PNG and editable SVG files. Each experiment's interpretation belongs in that directory's `report.md`; the project README provides navigation.
