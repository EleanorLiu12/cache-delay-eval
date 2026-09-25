"""Scan rho(mix) for the report: 2 P levels x 3 shallow-turn variants x 30 seeds."""
import csv, itertools
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from cache_delay_eval.session_gen import (
    SHALLOW_LONG, GeneratorConfig, WorkloadKnobs, generate_requests)

MIXES = [i / 20 for i in range(21)]
SEEDS = list(range(30))
OUT = Path("results/session-gen")


def one(job):
    P, turns, mix, seed = job
    config = GeneratorConfig(seed=seed, shallow=replace(SHALLOW_LONG, turns=turns))
    _, stats = generate_requests(WorkloadKnobs(P=P, rho_mix=mix), config)
    return dict(P=P, shallow_turns=turns, rho_mix=mix, seed=seed,
                rho=stats["rho"], rho_spearman=stats["rho_spearman"],
                requests=stats["requests"], sessions=stats["sessions"],
                working_set_blocks=stats["working_set_blocks"],
                capacity_blocks=stats["capacity_blocks"])


if __name__ == "__main__":
    jobs = list(itertools.product([512, 8192], [3, 2, 1], MIXES, SEEDS))
    with Pool(12) as pool:
        rows = pool.map(one, jobs, chunksize=16)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "rho-profile.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {OUT/'rho-profile.csv'}")
