"""Save exhaustive four-request model checks; costs are uncalibrated toy ticks."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from cache_delay_eval.toy_oracle import replay, solve, standard_instances, summarize


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for source in standard_instances():
        for slack in (0, 5, 10):
            instance = replace(source, slack_percent=slack)
            result = solve(instance)
            baseline, base_events = replay(instance)
            best, best_events = replay(instance, result.actions)
            record = dict(instance=asdict(instance), certificate=result.certificate(instance),
                          baseline=summarize(instance, baseline, base_events, result.limits),
                          oracle=summarize(instance, best, best_events, result.limits),
                          baseline_events=base_events, oracle_events=best_events)
            (output / f'{instance.name}-slack{slack}.json').write_text(json.dumps(record, indent=2)+'\n')
            rows.append(dict(case=instance.name, slack_percent=slack,
                             baseline_mean_ttft=record['baseline']['mean_ttft_ticks'],
                             oracle_mean_ttft=record['oracle']['mean_ttft_ticks'],
                             baseline_makespan=baseline.time, oracle_makespan=best.time,
                             **record['certificate']))
    summary = dict(units='toy ticks; no hardware calibration', GPU_used=False, results=rows)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Four-request exact Oracle: CPU model checks', '',
             'These hand-constructed cases test exhaustive waiting-order search in a restricted model. '
             'They do not estimate vLLM latency or establish a production scheduling benefit.', '',
             '## Fixed model', '',
             '- Single-turn requests with integer arrival times and fixed output token scripts.',
             '- One token per KV block; full-prefix identities, shared reference counts, inactive-block LRU.',
             '- Both policies reserve the full prompt plus the entire scripted output at admission. '
             'This differs from vLLM and removes preemption and decode-growth allocation failures.',
             '- Fixed running order, chunked prefill and token/sequence budgets. Only waiting permutations vary.',
             '- A batch costs one overhead tick plus its largest per-request service cost. '
             'Prefill and decode each cost one tick per token; no cost is hardware calibrated.',
             '- Completion, decode duration and inter-token gap must stay within the FCFS value '
             'plus max(slack × value, 1 tick). The 0% case still allows the one-tick tolerance.', '',
             '## Results', '',
             '| Case | Slack | FCFS mean TTFT | Oracle mean TTFT | Mean regret | States | Exact |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        lines.append(f"| {r['case']} | {r['slack_percent']}% | {r['baseline_mean_ttft']} | "
                     f"{r['oracle_mean_ttft']} | {r['regret_mean_ttft_ticks']} | {r['visited_states']} | "
                     f"{r['status'] == 'exact_restricted_toy_optimum'} |")
    lines += ['', 'All quantities are toy ticks. Each case file retains every baseline and selected '
              'Oracle transition, resource state, fit attempt, constraint limit and search certificate. '
              'Exactness means all reachable legal waiting orders were exhausted under these rules. '
              'It does not establish an optimum for the planned vLLM system.', '',
              'The full-prompt head-failure case demonstrates that waiting order can matter when a '
              'large request cannot fit and a later small request can. The zero-regret controls '
              'show that an improvement is not guaranteed by the objective alone. These constructed '
              'cases provide no prevalence estimate and no new online policy.', '',
              '## Reproduction', '', '```sh',
              '.venv/bin/python scripts/run_toy_oracle.py --output-dir results/toy-oracle-new',
              '.venv/bin/python -m unittest discover -s tests -p test_toy_oracle.py', '```', '',
              'Next: GPU-calibrated service costs, fidelity checks against held-out engine trajectories, '
              'multi-turn dependency modeling and policy comparisons with identical demand.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    paths = [Path('src/cache_delay_eval/toy_oracle.py'), Path(__file__), Path('tests/test_toy_oracle.py')]
    (output/'code-manifest.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                       for p in paths}, indent=2)+'\n')
    print(json.dumps({'runs':len(rows), 'all_exact':all(r['optimality_gap_ticks']==0 for r in rows)}))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args().output_dir)
