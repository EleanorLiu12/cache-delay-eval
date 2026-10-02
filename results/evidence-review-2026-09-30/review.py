#!/usr/bin/env python3
"""Audit retained measurements and source labels; do not classify target traces."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SOURCES = {}


def path(name):
    p = ROOT / name
    SOURCES[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def read(name):
    return json.loads(path(name).read_text())


def rows(name):
    with path(name).open() as f:
        return [json.loads(line) for line in f]


def fraction(k, n):
    return dict(numerator=k, denominator=n, fraction=k / n if n else None)


def qwen(label, source, analysis_dir):
    raw = rows(source)
    retained = read(analysis_dir + '/analysis.json')
    assert SOURCES[source] == retained['input_sha256']
    counts = Counter(r['type'] for r in raw)
    assert len(raw) == retained['quality']['valid_records']
    assert dict(counts) == retained['cohorts']['all_types']['types']
    ids = {r['chat_id']: r for r in raw}
    assert len(ids) == len(raw)
    groups = {}
    for r in raw:
        p = r
        while p['parent_chat_id'] != -1:
            p = ids[p['parent_chat_id']]
        groups.setdefault(p['chat_id'], []).append(r)
    by_type = []
    for kind, n in sorted(counts.items()):
        chains = [g for g in groups.values() if all(r['type'] == kind for r in g)]
        assert sum(len(g) for g in chains) == n
        by_type.append(dict(source_label=kind, request_share=fraction(n, len(raw)),
                            observed_components=len(chains),
                            multi_request_components=sum(len(g) > 1 for g in chains)))
    overlap = {}
    for name, old in retained['cohorts'].items():
        evidence = rows(analysis_dir + '/requests-' + name + '.jsonl')
        n = len(evidence)
        k = sum(2 * r['historical_prefix_tokens'] >= r['input_length'] for r in evidence)
        expected = old['prefix_opportunity']['potential_reuse_fraction']['thresholds']['0.5']
        assert fraction(k, n) == expected
        overlap[name] = dict(historical_input_overlap_ge_50pct=fraction(k, n),
                             multi_request_components=fraction(old['multi_request_components'], old['components']))
    return dict(dataset=label, source=source, source_records=len(raw),
                source_fields=sorted(set().union(*(r.keys() for r in raw))), source_type_composition=by_type,
                prefix_statistics=overlap, verified_task_labels=False,
                target_trace_class_prevalence=None, actual_hit_ttft_pairs=0,
                limitation='Source type labels and input overlap do not identify application purpose or measured cache-hit/TTFT behavior.')


def wildchat():
    source = rows('data/wildchat-audit-2026-09-28/sample-200/source-sample-projection.jsonl')
    audit = rows('results/wildchat-readiness-2026-09-28/conversations.jsonl')
    rendered = rows('results/wildchat-readiness-2026-09-28/requests.jsonl')
    by_id = {r['sample_id']: r for r in audit}
    users = missing = multi = 0
    for row in source:
        turns = [m for m in row['source']['conversation'] if m['role'] == 'user']
        absent = sum(m.get('timestamp') is None for m in turns)
        assert len(turns) == by_id[row['sample_id']]['user_turns']
        assert absent == by_id[row['sample_id']]['user_timestamps_missing']
        users += len(turns)
        missing += absent
        multi += len(turns) > 1
    assert len(source) == len(audit) == 200
    assert users == missing == 606 and len(rendered) == 581
    return dict(sample_conversations=200, multi_turn_conversations=fraction(multi, 200),
                user_turns=users, missing_user_arrivals=fraction(missing, users), rendered_prompts=len(rendered),
                sampling_scope='Fixed 200-row sample from one shard; not corpus-wide.',
                source_fields=sorted(source[0]['source']), message_fields=sorted(source[0]['source']['conversation'][0]),
                coding_document_task_classification='not performed in retained analyses',
                actual_hit_ttft_pairs=0, target_trace_class_prevalence=None)


def metric_delta(before, after, name):
    def value(text):
        found = re.findall(r'^vllm:' + re.escape(name) + r'(?:\{[^\n]*\})?\s+([-+\deE.]+)$', text, re.M)
        assert found, name
        return sum(map(float, found))
    return value(after) - value(before)


def gpu():
    base = 'results/cloudlab-pattern-03'
    old = read(base + '/analysis.json')
    trace_name = 'results/pattern-screen/traces/P512-negative-seed100.jsonl'
    with path(trace_name).open() as f:
        design = json.loads(next(f))['metadata']
    assert design['archetype_sessions'] == {'deep_short': 10, 'shallow_long': 42}
    assert design['deep_archetype']['turns'] == 30 and design['shallow_archetype']['turns'] == 3
    summaries = []
    for p in sorted((ROOT / base).glob('run-*/requests.jsonl')):
        data = rows(str(p.relative_to(ROOT)))
        meta = next(r for r in data if r['type'] == 'run_meta')
        requests = [r for r in data if r['type'] == 'request']
        assert len(requests) == meta['requests']
        assert all(r['status'] == 'ok' and r.get('cached_tokens') is not None and r.get('ttft_ms') is not None for r in requests)
        if meta['condition'] == 'on':
            corr = statistics.correlation([r['cached_tokens'] / r['prompt_tokens'] for r in requests], [r['ttft_ms'] for r in requests])
            ref = next(s for s in old['summaries'] if all(s[k] == meta[k] for k in ('trace', 'capacity_blocks', 'repeat')))
            assert abs(corr - ref['pooled_hit_ttft_pearson']) < 1e-12
            summaries.append(dict(run=p.parent.name, trace=meta['trace'], capacity=meta['capacity_blocks'], repeat=meta['repeat'],
                                  requests=len(requests), pearson=corr, spearman=ref['pooled_hit_ttft_spearman']))
    data = rows(base + '/run-009-on/requests.jsonl')
    req = [r for r in data if r['type'] == 'request']
    ids = {r['request_id']: r for r in req}
    children = [r for r in req if r['parent_request_id']]
    before = path(base + '/run-009-on/metrics-before.txt').read_text()
    after = path(base + '/run-009-on/metrics-after.txt').read_text()
    count = metric_delta(before, after, 'request_queue_time_seconds_count')
    assert count == len(req)
    means = {k: metric_delta(before, after, k + '_seconds_sum') / count for k in ('request_queue_time', 'request_prefill_time', 'time_to_first_token')}
    quoted = 'Long prompts arrived early. Later requests often reused more cached context but still waited long for their first token.'
    assert quoted in path(base + '/report.md').read_text()
    return dict(report_quote=quoted, synthetic=True, cache_on_run_results=summaries,
                mixed_trace_design=dict(source=trace_name, sessions=design['archetype_sessions'],
                                       short_input_many_turn=design['deep_archetype'], long_input_few_turn=design['shallow_archetype'],
                                       requests=design['requests'], seed=design['seed']),
                representative=dict(requests=len(req), server_mean_seconds=means,
                                    children=len(children), children_dispatched_before_parent_completion=sum(r['dispatched_ms'] < ids[r['parent_request_id']]['completed_ms'] for r in children),
                                    preemptions=metric_delta(before, after, 'num_preemptions_total')),
                formal_class_membership_rule_in_original_report=False,
                interpretation='Evidence for a candidate class from one generated seed, multiple capacities and repetitions; no real-application prevalence.')


def oracle():
    points = []
    common = None
    for suffix in ('slack0', 'validated', 'slack10'):
        base = 'results/eviction-oracle-cpu-2026-09-29-' + suffix
        inst = read(base + '/instance.json')
        result = read(base + '/result.json')
        budget = inst.pop('slack_percent')
        if common is None:
            common = inst
        assert inst == common and result['exact'] and not result['hardware_validated']
        stock = statistics.mean(result['baseline']['ttft'])
        best = statistics.mean(result['best']['ttft'])
        assert stock == result['baseline']['mean_ttft'] and best == result['best']['mean_ttft']
        points.append(dict(progress_budget_pct=budget, stock_mean_ttft_ticks=stock,
                           oracle_mean_ttft_ticks=best, regret_ticks=stock-best, relative_regret=(stock-best)/stock))
    return dict(available_cpu_points=points, independent_windows=1, hardware_validated=False,
                other_eviction_policies_compared=False,
                limitation='Restricted root-only model with synthetic costs; not hardware regret.')


def main():
    revision = '5f7439c51ec248a0c585f7d90a41a6f57773b912'
    result = dict(
        qwen_a=qwen('Qwen-Bailian Trace A', f'data/qwen-bailian/{revision}/qwen_traceA_blksz_16.jsonl', 'results/qwen-trace-a-2026-09-28'),
        qwen_b=qwen('Qwen-Bailian Trace B', f'data/qwen-bailian-trace-b/{revision}/qwen_traceB_blksz_16.jsonl', 'results/qwen-trace-b-2026-09-28'),
        wildchat=wildchat(), gpu=gpu(), oracle=oracle(),
        other_sources=dict(BurstGPT='Schema investigation only; no retained corpus or prevalence result.', ServeGen='Schema investigation only; input/timestamp semantics unresolved.'),
        target_class_prevalence='Not measured; do not substitute prefix-overlap or long-predecessor proxy rates.')
    (OUT / 'verified-statistics.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'source-manifest.json').write_text(json.dumps({'sha256':SOURCES, 'review_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
    for key in ('qwen_a', 'qwen_b'):
        print(key, result[key]['source_type_composition'])
    print('Verified retained source counts, prefix counts, GPU correlations, waiting metrics and CPU Oracle arithmetic.')


if __name__ == '__main__':
    main()
