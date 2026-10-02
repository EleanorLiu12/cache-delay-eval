"""Extract a reproducible real-trace structural window and search legal evictions."""
from dataclasses import asdict, replace
from decimal import Decimal
import argparse
import hashlib
import json
from pathlib import Path
from cache_delay_eval.eviction_oracle import Instance, Request, replay, solve

REVISION = '5f7439c51ec248a0c585f7d90a41a6f57773b912'
EXPECTED = '68e3f98e2d601d60d0abf4b89bc8a3654372abab7b1cde6373a13d0054379d59'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(f'data/qwen-bailian-trace-b/{REVISION}/qwen_traceB_blksz_16.jsonl'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-nodes', type=int, default=10000)
    parser.add_argument('--slack-percent', type=int, default=5)
    args = parser.parse_args()
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest != EXPECTED:
        raise ValueError('Source does not match the retained publisher LFS checksum')
    frame, total, eligible = [], 0, 0
    for line_number, line in enumerate(args.source.open(), 1):
        r = json.loads(line)
        total += 1
        if r['parent_chat_id'] == -1 and r['type'] == 'api' and 0 < r['output_length'] and r['input_length'] + r['output_length'] <= 64:
            eligible += 1
            frame.append(dict(source_line=line_number, raw=r))
    def has_conflict_then_reuse(window):
        keys = [x['raw']['hash_ids'][0] for x in window[1:]]
        return any(keys[i] == keys[k] != keys[j] for i in range(4) for j in range(i+1,4) for k in range(j+1,4))
    qualifying = [i for i in range(len(frame)-4) if has_conflict_then_reuse(frame[i:i+5])]
    if not qualifying:
        raise ValueError('No window satisfies the declared conflict/reuse rule')
    selected = frame[qualifying[0]:qualifying[0]+5]
    def request(item, origin):
        r = item['raw']
        return Request(str(r['chat_id']), int((Decimal(str(r['timestamp']))-origin)*1000000),
                       r['input_length'], r['output_length'], tuple(r['hash_ids']))
    origin = Decimal(str(selected[1]['raw']['timestamp']))
    warm = request(selected[0], Decimal(str(selected[0]['raw']['timestamp'])))
    base = Instance((warm,), capacity=5, slack_percent=args.slack_percent)
    warm_result = replay(base)
    instance = replace(base, requests=tuple(request(item, origin) for item in selected[1:]),
                       initial_cache=tuple(b['identity'] for b in warm_result['final_cache']),
                       initial_free_order=tuple(warm_result['final_free_order']))
    provenance = dict(dataset='Qwen-Bailian Trace B', revision=REVISION, source=str(args.source.resolve()),
        source_url=f'https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/{REVISION}/qwen_traceB_blksz_16.jsonl',
        source_sha256=digest, total_rows=total, eligible_roots=eligible,
        selection='first consecutive five in filtered API roots input+output<=64 whose four evaluation roots contain A,B,A first-block identities; first root warms',
        candidate_windows=len(frame)-4, conflict_reuse_windows=len(qualifying), selected_filtered_offset=qualifying[0],
        selected=selected, sampled_window_count=1, selection_depends_on_oracle_outcome=False,
        boundary='filtered root-only micro-workload; intervening traffic excluded; quiescent cache after one recorded warmup',
        initial_state='warmup cache contents/free order preserved; clock reset at first evaluation root',
        arrival_transform='source decimal seconds relative to first evaluation root, exact integer microsecond ticks; no rescaling',
        output_assumption='source output lengths preserved; generated token identities unavailable; output-containing blocks remain uncached',
        interpretation='real block-content identities/lengths, structural model only; no original tokens or production engine state',
        hardware_config='5 usable blocks for tractable search; NOT the 2048-block A30 configuration',
        cost_status='uncalibrated synthetic costs; ticks must not be reported as measured microseconds')
    result = solve(instance, args.max_nodes)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in [('instance', asdict(instance)), ('provenance', provenance), ('warmup', warm_result), ('result', result)]:
        (args.output_dir/f'{name}.json').write_text(json.dumps(data, indent=2)+'\n')
    summary = {k:v for k,v in result.items() if k not in ('baseline','best','limits')}
    summary['baseline_mean_ttft_ticks'] = result['baseline']['mean_ttft']
    summary['oracle_mean_ttft_ticks'] = result['best']['mean_ttft']
    summary['baseline_evictions'] = sum(e['victim_count'] for e in result['baseline']['events'] if e['type']=='allocation')
    summary['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).resolve().parents[1]/'src/cache_delay_eval/eviction_oracle.py']}
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
