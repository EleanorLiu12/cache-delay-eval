"""Placebo for the detector sweep: plain LMetric with the tie-break order shifted by one.

The spread between this run and plain LMetric is the change any small routing
perturbation causes, the floor against which detector effects are judged.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from cache_delay_eval.routing_sim import Cost, simulate, summarize
from run_detector_sweep import config, load


def run(job):
    row, cost = job
    c = Cost(**cost)
    reqs = load(row["trace"], row["start_s"], row["time_scale"])
    simulate(reqs, row["instances"], "lmetric", "exact", c, config(row["kv_mult"]), tie_offset=1)
    keys = ("trace", "start_s", "instances", "kv_mult", "load_frac")
    return dict({k: row[k] for k in keys}, lmetric_tie1=summarize(reqs, c))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cost", required=True)
    p.add_argument("--sweep", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cost = json.load(open(args.cost))["cost"]
    rows = [json.loads(line) for line in open(args.sweep)]
    with Pool() as pool, Path(args.output).open("w") as f:
        for r in pool.imap_unordered(run, [(row, cost) for row in rows]):
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
