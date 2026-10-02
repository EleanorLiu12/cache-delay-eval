"""Fit exact batch-signature costs and validate on disjoint A30 measurements.

Input schema: {device, timing_scope, session_ids, samples:[{signature,
 duration_ns, run_id}]}. timing_scope must include complete iteration wall time,
 not admission-section time or isolated CUDA kernel duration. This tool does
 not collect measurements; retain the original profiler/timing records.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median, mean


def calibrate(training, validation, max_relative_error=.10):
    for data in (training, validation):
        if data.get('device') != 'NVIDIA A30' or data.get('timing_scope') != 'complete_iteration_wall':
            raise ValueError('A30 complete-iteration wall measurements required')
        if not data.get('session_ids') or not data.get('samples') or not data.get('source_measurements'):
            raise ValueError('Retained measurement source, session IDs and samples are required')
        for s in data['samples']:
            signature=json.loads(s['signature'])
            if not isinstance(signature,list) or not signature or any(len(x)!=3 or x[0] not in ('prefill','decode') or type(x[1]) is not int or x[1]<=0 or type(x[2]) is not int or x[2]<0 for x in signature):
                raise ValueError('Invalid batch signature')
            if type(s['duration_ns']) is not int or s['duration_ns'] <= 0 or not s.get('run_id'):
                raise ValueError('Positive measured duration and run provenance required')
    if set(training['session_ids']) & set(validation['session_ids']):
        raise ValueError('Calibration and validation sessions overlap')
    if {s['run_id'] for s in training['samples']} & {s['run_id'] for s in validation['samples']}:
        raise ValueError('Calibration and validation runs overlap')
    grouped={}
    for s in training['samples']:
        grouped.setdefault(s['signature'],[]).append(s['duration_ns'])
    # Positive integer microsecond ticks; ceil(median(ns)/1000).
    table={key:math.ceil(median(values)/1000) for key,values in grouped.items()}
    rows=[]
    for s in validation['samples']:
        predicted=table.get(s['signature'])
        measured=s['duration_ns']/1000
        rows.append(dict(signature=s['signature'], measured_ticks=measured, predicted_ticks=predicted,
                         relative_error=abs(predicted-measured)/measured if predicted is not None else None))
    errors=[r['relative_error'] for r in rows if r['relative_error'] is not None]
    return dict(cost_table=table, cost_revision='A30_exact_signature_median_v1', tick_ns=1000,
                rounding='ceil median measured ns / 1000', validation=rows,
                coverage=len(errors)/len(rows), mean_relative_error=mean(errors) if errors else None,
                max_relative_error=max(errors) if errors else None, threshold=max_relative_error,
                accepted=len(errors)==len(rows) and max(errors)<=max_relative_error,
                trajectory_validated=False,
                limitation='Timing acceptance does not validate allocator/scheduler equivalence; unseen signatures rejected')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-relative-error',type=float,default=.10)
    a=p.parse_args()
    if not 0<a.max_relative_error<1:
        p.error('Error threshold must be between 0 and 1 and frozen before validation')
    result=calibrate(json.loads(a.calibration.read_text()), json.loads(a.validation.read_text()), a.max_relative_error)
    result['input_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.calibration,a.validation)}
    with a.output.open('x') as f:
        f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cost_table','validation')},indent=2))
    if not result['accepted']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
