import json
from pathlib import Path
import tempfile
import unittest

from cache_delay_eval.analyze_patterns import hit_ttft_summary, main


def rows(hits, ttfts, lengths=None):
    lengths = lengths or [100] * len(hits)
    return [dict(cached_token_fraction=h, ttft_on_ms=t, prompt_tokens=p)
            for h, t, p in zip(hits, ttfts, lengths)]


class PatternAnalysisTests(unittest.TestCase):
    def test_positive_hit_ttft_relation_is_detected(self):
        result = hit_ttft_summary(rows([0, .25, .5, .75], [10, 20, 30, 40]))
        self.assertAlmostEqual(result['pooled_hit_ttft_pearson'], 1)
        self.assertAlmostEqual(result['pooled_hit_ttft_spearman'], 1)
        self.assertAlmostEqual(result['prompt_length_partial_pearson'], 1)

    def test_pooled_sign_can_reverse_within_archetypes(self):
        # Both groups slope down, but the high-hit group has larger TTFT.
        a = rows([.1, .2, .3], [30, 20, 10])
        b = rows([.7, .8, .9], [90, 80, 70])
        self.assertGreater(hit_ttft_summary(a + b)['pooled_hit_ttft_pearson'], 0)
        for group in (a, b):
            self.assertAlmostEqual(hit_ttft_summary(group)['pooled_hit_ttft_pearson'], -1)

    def test_length_adjustment_removes_shared_linear_length_effect(self):
        result = hit_ttft_summary(rows([.2, .2, .4, .8], [3, 3, 9, 9], [1, 2, 3, 4]))
        self.assertGreater(result['pooled_hit_ttft_pearson'], 0)
        self.assertAlmostEqual(result['prompt_length_partial_pearson'], 0)

    def test_constant_or_insufficient_measurements_are_undefined(self):
        for sample in (rows([.5] * 4, [1, 2, 3, 4]), rows([0, 1], [1, 2])):
            result = hit_ttft_summary(sample)
            self.assertIsNone(result['pooled_hit_ttft_pearson'])
            self.assertIsNone(result['pooled_hit_ttft_spearman'])
            self.assertIsNone(result['prompt_length_partial_pearson'])

    def test_candidate_detection_is_independent_of_cache_on_off_effect(self):
        for delta in (-5, 5):
            with self.subTest(control_delta=delta), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plans = []
                for repeat in (0, 1):
                    for condition in ('on', 'off'):
                        run = dict(trace='trace.jsonl', capacity_blocks=2048,
                                   repeat=repeat, condition=condition)
                        plans.append(run)
                        meta = dict(type='run_meta', **run, requests=4,
                                    trace_sha256='fixed-workload', model='test',
                                    arrival_scale=1, engine_config={}, max_dispatch_lag_ms=50)
                        records = [meta]
                        for i in range(4):
                            records.append(dict(
                                type='request', request_id=str(i), scheduled_ms=i,
                                prompt_tokens=100, output_tokens=1, archetype='deep_short',
                                cached_tokens=25 * i if condition == 'on' else 0,
                                ttft_ms=10 * (i + 1) + (delta if condition == 'off' else 0),
                                dispatch_lag_ms=0, status='ok',
                            ))
                        records.append(dict(type='run_summary', errors=0,
                                            late_dispatches=0, missing_cache_usage=0))
                        folder = root / f'run-{repeat}-{condition}'
                        folder.mkdir()
                        (folder / 'requests.jsonl').write_text(
                            '\n'.join(json.dumps(r) for r in records) + '\n')
                (root / 'plan.json').write_text(json.dumps(dict(runs=plans)))
                output = root / 'analysis.json'
                self.assertEqual(main([str(root), '--output', str(output)]), 0)
                result = json.loads(output.read_text())
                self.assertTrue(result['patterns'][0]['positive_pearson_in_all_repetitions'])
                for summary in result['summaries']:
                    self.assertAlmostEqual(summary['pooled_hit_ttft_pearson'], 1)
                    self.assertAlmostEqual(summary['within_archetype']['deep_short']['pooled_hit_ttft_pearson'], 1)
                    self.assertEqual(summary['paired_control']['median_delta_ms'], delta)


if __name__ == '__main__':
    unittest.main()
