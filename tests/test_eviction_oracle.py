from dataclasses import replace
import unittest
from cache_delay_eval.eviction_oracle import Instance, Request, Choice, replay, solve


class OracleTests(unittest.TestCase):
    def fixture(self):
        # A has two cached blocks, B needs two slots, then A returns. With
        # three slots, only A's first block is worth retaining for a prefix hit.
        return Instance((Request('b',0,2,1,(8,9)), Request('a',100,2,1,(1,2))),
                        capacity=3, block_size=1, initial_cache=((1,), (1,2), None),
                        initial_free_order=(2,0,1), overhead=1, prefill_token_cost=1,
                        decode_token_cost=1, prefill_chunk=4, token_budget=4)

    def test_hand_computed_gain_and_legal_victims(self):
        x = self.fixture()
        r = solve(x)
        self.assertTrue(r['exact'])
        self.assertEqual(r['baseline']['ttft'], [3,3])
        self.assertEqual(r['best']['ttft'], [3,2])
        self.assertEqual(r['mean_ttft_reduction_ticks'], .5)
        self.assertEqual(r['optimality_gap_ticks'], 0)
        for e in r['best']['events']:
            if e['type'] == 'allocation':
                self.assertTrue(all(c['refs']==0 for c in e['candidates']))
                self.assertTrue({v['block_id'] for v in e['victims']} <= {c['block_id'] for c in e['candidates']})

    def test_node_limit_never_claims_optimal(self):
        r = solve(self.fixture(), max_nodes=1)
        self.assertFalse(r['exact'])
        self.assertEqual(r['status'], 'offline_search_benchmark')
        self.assertGreater(r['optimality_gap_ticks'], 0)
        self.assertLessEqual(r['best']['sum_ttft'], r['baseline']['sum_ttft'])

    def test_reject_active_or_duplicate_victims(self):
        x = self.fixture()
        with self.assertRaises(ValueError):
            replay(x, ((2,),))  # uncached physical slot cannot be a cached victim
        with self.assertRaises(ValueError):
            replay(x, ((0,0),))

    def test_sufficient_capacity_no_eviction(self):
        x = replace(self.fixture(), capacity=6, initial_cache=(), initial_free_order=())
        r = solve(x)
        self.assertEqual(r['relative_reduction'], 0)
        self.assertTrue(r['exact'])

    def test_dynamic_decode_and_fixed_tail_preemption(self):
        x = Instance((Request('a',0,1,4,(1,)), Request('b',0,1,4,(2,))),
                     capacity=5, block_size=1, token_budget=4, prefill_chunk=4,
                     overhead=1, prefill_token_cost=1, decode_token_cost=1)
        r = solve(x, max_nodes=300)
        self.assertTrue(all(len(p['outputs'])==4 for p in r['baseline']['progress']))
        self.assertGreater(sum(p['preemptions'] for p in r['baseline']['progress']), 0)
        self.assertLessEqual(r['best']['sum_ttft'], r['baseline']['sum_ttft'])
        self.assertEqual(r['baseline']['progress'][0]['preemptions'], 0)

    def test_content_hash_is_not_a_prefix_hash(self):
        x = Instance((Request('x',0,3,1,(2,9,8)),), capacity=4, block_size=1,
                     initial_cache=((1,), (1,9), None, None))
        self.assertEqual(replay(x)['progress'][0]['cached_tokens'], 0)

    def test_real_window_and_cost_table(self):
        import json
        from pathlib import Path
        path=Path(__file__).resolve().parents[1]/'results/eviction-oracle-cpu-2026-09-29-window/instance.json'
        if not path.exists():
            self.skipTest('Generate retained window before integration test')
        d=json.loads(path.read_text())
        d['requests']=tuple(Request(**dict(r,hashes=tuple(r['hashes']))) for r in d['requests'])
        d['initial_cache']=tuple(tuple(h) if h is not None else None for h in d['initial_cache'])
        d['initial_free_order']=tuple(d['initial_free_order'])
        x=Instance(**d)
        r=solve(x)
        self.assertTrue(r['exact'])
        self.assertGreater(r['visited_nodes'], 1)
        self.assertGreater(sum(e['victim_count'] for e in r['baseline']['events'] if e['type']=='allocation'), 0)
        table={e['signature']:e['end']-e['start'] for e in r['baseline']['events'] if e['type']=='iteration'}
        self.assertEqual(replay(replace(x,cost_table=table))['ttft'],r['baseline']['ttft'])
        table.pop(next(iter(table)))
        with self.assertRaises(ValueError):
            replay(replace(x,cost_table=table))


if __name__ == '__main__':
    unittest.main()
