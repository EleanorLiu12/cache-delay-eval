#!/bin/bash
# Follow-up sweeps after the main N=16 sweep.
cd "$(dirname "$0")/../.."
R=results/routing-sim-2026-10-03
while pgrep -f "sweep-n16.jsonl" >/dev/null; do sleep 30; done
.venv/bin/python scripts/run_routing_sim_sweep.py --cost $R/cost-calibrated.json --capacity-cache $R/capacity.json \
  --output $R/sweep-n16.jsonl --variants lmetric/inflight >> $R/sweep-n16-inflight.log 2>&1
.venv/bin/python scripts/run_routing_sim_sweep.py --cost $R/cost-calibrated.json --capacity-cache $R/capacity.json \
  --output $R/sweep-n4.jsonl --instances 4 --traces thinking,coder,chatbot --kv-mult 1,4 --loads 0.5,0.7,0.9 \
  --variants load/exact,lmetric/exact,lmetric/dispatch,lmetric/lru,sglang_ca/dispatch,lmetric/inflight >> $R/sweep-n4.log 2>&1
echo done >> $R/followup.done
