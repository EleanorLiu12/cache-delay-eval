"""Join client IDs, audit allocation records, and retain censored reuse outcomes."""
import argparse
import json
from pathlib import Path
from cache_delay_eval.admission_telemetry import resolve_events, reuse_report, validate_event


def analyze(events, client_ids):
    streams = {}
    for event in events:
        validate_event(event)
        streams.setdefault(event['process_id'], []).append(event)
    errors, allocations, joined = [], {}, resolve_events(events, client_ids)
    for pid, stream in streams.items():
        stream.sort(key=lambda e:e['event_id'])
        ids = [e['event_id'] for e in stream]
        if ids != list(range(1,len(ids)+1)):
            errors.append(f'process {pid}: event IDs have gaps/duplicates or process log was appended across runs')
        for e in stream:
            d = e['data']
            if e['type']=='allocation_begin':
                key=(pid,d['allocation_id'])
                if key in allocations:
                    errors.append(f'{key}: duplicate allocation')
                allocations[key] = dict(begin=d, end=None, victims=[])
                candidates={b['block_id'] for b in d['candidates']}
                if any(b['ref_cnt'] or b['is_null'] or not b['hashes'] for b in d['candidates']):
                    errors.append(f'{key}: illegal candidate')
                selected=d['selected_block_ids']
                if selected != [b['block_id'] for b in d['free_order'][:d['requested_blocks']]]:
                    errors.append(f'{key}: stock queue selection changed')
                if d['required_victim_count'] and len(candidates & set(selected)) != d['required_victim_count']:
                    errors.append(f'{key}: incorrect victim count')
            elif e['type']=='cache_remove' and d['cause']=='allocation':
                key=(pid,d['allocation_id'])
                if key not in allocations:
                    errors.append(f'{key}: victim without allocation begin')
                else:
                    allocations[key]['victims'].append(d['block_id'])
                if d['ref_cnt'] or d['is_null']:
                    errors.append(f'{key}: protected physical block evicted')
            elif e['type']=='allocation_end':
                key=(pid,d['allocation_id'])
                if key not in allocations:
                    errors.append(f'{key}: allocation end without begin')
                else:
                    allocations[key]['end']=d
    for key, a in allocations.items():
        b=a['begin']
        if a['end'] is None:
            errors.append(f'{key}: incomplete allocation')
        elif [x['block_id'] for x in a['end']['blocks']] != b['selected_block_ids']:
            errors.append(f'{key}: selected/allocated physical IDs disagree')
        if len(a['victims']) != b['required_victim_count']:
            errors.append(f'{key}: expected/observed victim counts disagree')
        candidates={x['block_id'] for x in b['candidates']}
        if not set(a['victims']) <= candidates & set(b['selected_block_ids']):
            errors.append(f'{key}: actual victims not selected eligible blocks')
    unresolved=[e for e in joined if e['engine_request_id'] is not None and e['join_status']!='mapped']
    if unresolved:
        errors.append(f'{len(unresolved)} request events could not be joined uniquely')
    seen_clients={e['request_id'] for e in joined if e['type']=='enqueue' and e['join_status']=='mapped'}
    missing=sorted(set(client_ids)-seen_clients)
    if missing:
        errors.append(f'{len(missing)} client requests missing engine enqueue')
    if not allocations:
        errors.append('no allocations observed; eviction telemetry not exercised')
    scheduler_pids={pid for pid, _ in allocations}
    for pid in scheduler_pids:
        if not any(e['type']=='pool_initial' for e in streams[pid]):
            errors.append(f'process {pid}: missing initial physical cache state')
    return dict(valid=not errors, errors=errors, events=len(events), allocations=len(allocations),
                physical_evictions=sum(len(a['victims']) for a in allocations.values()),
                mapped_client_requests=len(seen_clients), missing_client_requests=missing), joined, reuse_report(events)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--telemetry-dir',type=Path,required=True)
    p.add_argument('--client-turns',type=Path,required=True,help='chat replay turns.jsonl')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    events=[json.loads(l) for f in sorted(a.telemetry_dir.glob('admission-*.jsonl')) for l in f.read_text().splitlines()]
    turns=[json.loads(l) for l in a.client_turns.read_text().splitlines()]
    clients=[r['request_id'] for r in turns if r.get('dispatched_s') is not None]
    report, joined, reuse=analyze(events, clients)
    if any(r.get('status')!='ok' for r in turns):
        report['errors'].append('Client run contains incomplete/failed turns')
        report['valid']=False
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for name, rows in [('joined-events',joined),('eviction-reuse',reuse)]:
        (a.output_dir/f'{name}.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    (a.output_dir/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['valid']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
