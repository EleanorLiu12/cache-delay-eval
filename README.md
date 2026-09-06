# Evaluating Cache Update Delays in LLM Request Routing

## Quick start

The trace tools have no third-party runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
cache-delay-validate-trace traces/example.jsonl
cache-delay-generate-trace --requests 1000 --output traces/synthetic-1k.jsonl
cache-delay-validate-trace traces/synthetic-1k.jsonl
```

## Run the TTFT calibration

The harness uses vLLM's OpenAI-compatible `/v1/completions` streaming endpoint,
its `/tokenize` endpoint, and (when available) Prometheus `/metrics`. It sends
token-ID prompts so prompt and shared-prefix lengths are exact for the server's
tokenizer. Client-side TTFT is measured from dispatch until the first non-empty
streamed token.

On Apple Silicon, install the official community-maintained Metal plugin and
serve the pilot model:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source ~/.venv-vllm-metal/bin/activate
vllm serve Qwen/Qwen3-0.6B --max-num-seqs 1 --port 8000
```

Then, in another terminal:

```bash
source .venv/bin/activate
cache-delay-calibrate --config config/calibration.metal.json
cache-delay-summarize results/calibration-metal.jsonl \
  --output results/calibration-metal-summary.csv
cache-delay-audit results/calibration-metal.jsonl
```

Use `config/calibration.pilot.json` first to exercise all three independent
variables in a short run before launching the 30-trial grid.

For the final CUDA run, start the model with automatic prefix caching and use
`config/calibration.cuda.json`. Keep `--max-num-seqs 1` when queue depth is an
experimental variable; otherwise the configured number is only offered load,
not a controlled queue depth.

The full grid is intentionally 30 trials per valid condition. Use `--trials 2`
for a quick end-to-end smoke test. Raw results are append-safe JSONL and include
errors rather than silently dropping failed requests.

## Reproducibility rules

- Do not compare Metal pilot numbers directly with CUDA final numbers.
- Use block-aligned shared-prefix lengths (the supplied configs assume 16-token
  blocks).
- Run one benchmark client and no unrelated inference traffic on the server.
- Record the server command, model revision, vLLM version, hardware, and config.
- Preserve raw JSONL; generate tables from it rather than editing measurements.
