# Quantifying the Impact of Stale State on Prefix-Cache-Aware Routing for Distributed LLM Serving

This project studies distributed LLM routing when the router's prefix-cache and
load information is delayed. It asks: **How do stale prefix-cache state and
replica-load state affect cache reuse, routing regret, and TTFT in distributed
LLM routing, and when do these effects become material across update delays,
workloads, and system conditions?**

The repository currently contains request-trace tools and a single-server vLLM
calibration pipeline. The two-replica emulator and routing experiments are
planned work.

## Install and test

Python 3.10 or newer is required.

```bash
git clone https://github.com/EleanorLiu12/cache-delay-eval.git
cd cache-delay-eval
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Use the trace tools

```bash
cache-delay-validate-trace traces/example.jsonl
cache-delay-generate-trace --requests 1000 --output traces/synthetic-1k.jsonl
cache-delay-validate-trace traces/synthetic-1k.jsonl
```

The JSONL format is documented in [schemas/README.md](schemas/README.md).

## Run the calibration pilot

Start vLLM-Metal in a separate terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source ~/.venv-vllm-metal/bin/activate
vllm serve Qwen/Qwen3-0.6B --max-num-seqs 1 --port 8000
```

Then run, audit, and summarize the pilot:

```bash
source .venv/bin/activate
cache-delay-calibrate --config config/calibration.pilot.json
cache-delay-audit results/calibration-pilot.jsonl
cache-delay-summarize results/calibration-pilot.jsonl \
  --output results/calibration-pilot-summary.csv
```

If a calibration run is interrupted, restart only its unfinished requests with:

```bash
cache-delay-calibrate --config config/calibration.pilot.json --resume
```

On systems where the display shares the inference GPU, reduce terminal-rendering
load during long runs with `--progress-every 50`.

Resume requires the same configuration and output path. The command validates
the existing rows before contacting the server and refuses configuration or
schedule mismatches. Without `--resume`, calibration refuses to append to a
non-empty output file. Do not send unrelated inference traffic to the server
during calibration.

After collecting separate calibration and validation runs, build the lookup,
validation report, and publication figures with:

```bash
python -m pip install -e '.[plots]'
cache-delay-analyze-calibration \
  --calibration results/calibration-metal-confirmatory.jsonl \
  --validation results/calibration-metal-validation.jsonl \
  --output-dir results/calibration-analysis
```

The analysis retains every successful observation, shows raw measurements
behind the condition medians, and writes both PNG and PDF figures.

## License

MIT; see [LICENSE](LICENSE).
