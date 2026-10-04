#!/usr/bin/env bash
# Burst runs and the approximate-index run, after run_a30_main.sh.
# Rates come from calibration/burst-rate.json.
set -euo pipefail

R=results/a30-routing-2026-10-02
TRACE=data/qwen-bailian-thinking/5f7439c51ec248a0c585f7d90a41a6f57773b912/qwen_thinking_blksz_16.jsonl
INSTANCES=http://127.0.0.1:8001=tcp://127.0.0.1:5601,http://127.0.0.1:8002=tcp://127.0.0.1:5602,http://127.0.0.1:8003=tcp://127.0.0.1:5603,http://127.0.0.1:8004=tcp://127.0.0.1:5604
PY=.venv/bin/python

for policy in affinity lmetric load; do
  printf 'Starting burst %s at %s\n' "$policy" "$(date -u +%FT%TZ)"
  "$PY" -m cache_delay_eval.routing --instances "$INSTANCES" --policy "$policy" \
    --workload burst --duration-s 300 --bg-rate 6.5 --hot-rate 6.5 \
    --hot-start-s 60 --hot-duration-s 120 --output "$R/runs/burst-$policy.jsonl"
done
"$PY" scripts/check_routing_runs.py "$R/runs"

printf 'Starting hot x3 lmetric dispatch at %s\n' "$(date -u +%FT%TZ)"
"$PY" -m cache_delay_eval.routing --instances "$INSTANCES" --policy lmetric \
  --index dispatch --workload trace --trace "$TRACE" --start-s 2100 \
  --duration-s 600 --time-scale 3 --output "$R/runs/hot-x3-lmetric-dispatch.jsonl"
"$PY" scripts/check_routing_runs.py "$R/runs"
