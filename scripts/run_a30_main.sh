#!/usr/bin/env bash
# Main A30 routing matrix. Time scales were fixed in calibration/decision.json.
set -euo pipefail

R=results/a30-routing-2026-10-02
TRACE=data/qwen-bailian-thinking/5f7439c51ec248a0c585f7d90a41a6f57773b912/qwen_thinking_blksz_16.jsonl
INSTANCES=http://127.0.0.1:8001=tcp://127.0.0.1:5601,http://127.0.0.1:8002=tcp://127.0.0.1:5602,http://127.0.0.1:8003=tcp://127.0.0.1:5603,http://127.0.0.1:8004=tcp://127.0.0.1:5604
PY=.venv/bin/python

run() {  # run <slice-start> <scale> <policy> <tag>
  printf 'Starting %s x%s %s at %s\n' "$4" "$2" "$3" "$(date -u +%FT%TZ)"
  "$PY" -m cache_delay_eval.routing --instances "$INSTANCES" --policy "$3" \
    --index events --workload trace --trace "$TRACE" --start-s "$1" \
    --duration-s 600 --time-scale "$2" \
    --output "$R/runs/$4-x$2-$3-events.jsonl"
}

check_block() {
  "$PY" scripts/check_routing_runs.py "$R/runs"
}

for scale in 2 3; do
  for policy in lmetric load affinity; do run 2100 "$scale" "$policy" hot; done
  check_block
  for policy in load affinity lmetric; do run 5100 "$scale" "$policy" typical; done
  check_block
done
