"""Grid-calibrate the simulator's decode cost on the A30 events-mode runs."""

import argparse
import itertools
import json
import math
from multiprocessing import Pool
from pathlib import Path

from cache_delay_eval.routing_sim import Cost, simulate, summarize
from validate_routing_sim import measured, workload


def one(job):
    path, fields = job
    cost = Cost(**fields)
    reqs, policy, index = workload(Path(path).name)
    return path, fields, summarize(simulate(reqs, 4, policy, index, cost), cost)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs")
    p.add_argument("--cost", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--decode-ctx", default="3e-4,4e-4,5e-4,6.6e-4")
    p.add_argument("--overlap", default="0,1")
    p.add_argument("--per-token-scale", default="0.8,1")
    args = p.parse_args()
    base = json.load(open(args.cost))["cost"]
    runs = [str(x) for x in sorted(Path(args.runs).glob("*.jsonl"))
            if x.name != "smoke.jsonl" and "dispatch" not in x.name and not x.name.startswith("burst-")
            or x.name.startswith("burst-half")]
    real = {r: measured(r) for r in runs}
    grid = [dict(base, decode_ctx=float(d), overlap=float(o), per_token=base["per_token"] * float(s))
            for d, o, s in itertools.product(args.decode_ctx.split(","), args.overlap.split(","),
                                             args.per_token_scale.split(","))]
    jobs = [(r, g) for g in grid for r in runs]
    with Pool() as pool:
        results = pool.map(one, jobs)
    table = []
    for g in grid:
        errs = []
        for path, fields, sim in results:
            if fields != g:
                continue
            m = real[path]
            errs += [abs(math.log(sim[k] / m[k])) for k in ("ttft_mean", "p90", "tpot_mean")]
        table.append(dict(cost=g, mean_abs_log_error=sum(errs) / len(errs)))
    table.sort(key=lambda t: t["mean_abs_log_error"])
    for t in table:
        c = t["cost"]
        print(f"decode_ctx {c['decode_ctx']:.1e} overlap {c['overlap']:.0f} per_token {c['per_token']:.4f}"
              f"  error {t['mean_abs_log_error']:.3f}")
    Path(args.output).write_text(json.dumps(dict(runs=runs, table=table), indent=1))


if __name__ == "__main__":
    main()
