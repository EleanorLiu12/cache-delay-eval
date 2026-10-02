"""Execute the pinned block/queue/pool classes on CPU, without importing vLLM."""
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from cache_delay_eval import admission_telemetry as t

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/admission-telemetry-cpu-2026-09-29/source'


def load_pool(patched=True):
    import runpy
    prep = runpy.run_path(str(ROOT / 'scripts/prepare_vllm_telemetry.py'))
    originals = {p: (SOURCE / p).read_text() for p in prep['HASHES']}
    if patched:
        source = prep['patched_sources'](originals)[prep['POOL']]
    else:
        source = originals[prep['POOL']]
    ns = dict(dataclass=dataclass, _cdt=t, logger=SimpleNamespace(info=lambda *a: None, warning=lambda *a: None),
              BlockHashWithGroupId=bytes, BlockHash=bytes,
              make_block_hash_with_group_id=lambda h, g: h + g.to_bytes(4, 'big'))
    # Future annotations avoid any torch/CUDA/type-only imports. Method bodies
    # and the real intrusive linked-list implementation remain unchanged.
    for text, names in [(originals['vllm/v1/core/kv_cache_utils.py'], {'KVCacheBlock', 'FreeKVCacheBlockQueue'}),
                        (source, {'BlockHashToBlockMap', 'BlockPool'})]:
        nodes = [n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name in names]
        code = 'from __future__ import annotations\n' + '\n'.join(ast.unparse(n) for n in nodes)
        exec(compile(code, '<pinned-cpu-classes>', 'exec'), ns)
    return ns['BlockPool']


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict('os.environ', {'CACHE_DELAY_TELEMETRY_DIR': self.temp.name})
        self.env.start()
        self.version = patch.object(t.importlib.metadata, 'version', return_value='0.28.0')
        self.version.start()
        t._writer = None
        t._state = {}

    def tearDown(self):
        if t._writer:
            t._writer.close()
        t._writer = None
        t._state = {}
        self.version.stop()
        self.env.stop()
        self.temp.cleanup()

    def events(self):
        return [json.loads(l) for p in Path(self.temp.name).glob('*.jsonl') for l in p.read_text().splitlines()]

    def test_id_assignment_and_exact_join(self):
        import runpy
        prep = runpy.run_path(str(ROOT / 'scripts/prepare_vllm_telemetry.py'))
        originals = {p: (SOURCE/p).read_text() for p in prep['HASHES']}
        text = prep['patched_sources'](originals)[prep['INPUT']]
        method = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == 'assign_request_id')
        method.decorator_list = []
        ns = dict(_cdt=t, envs=SimpleNamespace(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=False), random_uuid=lambda: 'abcdefgh1234')
        exec('from __future__ import annotations\n' + ast.unparse(method), ns)
        for external, client in [('cmpl-run:turn0-0', 'run:turn0'), ('chatcmpl-run:turn1', 'run:turn1'),
                                 ('cmpl-cmpl-user-0-0', 'cmpl-user-0')]:
            r = SimpleNamespace(request_id=external, external_req_id=None)
            ns['assign_request_id'](r)
            t.emit('enqueue', r)
            result = t.resolve_events(reversed(self.events()), [client])
            self.assertEqual(next(e for e in result if e['engine_request_id'] == r.request_id)['request_id'], client)
        t.request_id_map(SimpleNamespace(request_id='unsupported', external_req_id='cmpl-batch-1'))
        self.assertEqual(t.resolve_events(self.events(), ['batch'])[-1]['join_status'], 'unmatched')

    def test_victims_aliases_protection_and_reuse(self):
        pool = load_pool()(5, True, 16)
        a, b, c, d = pool.get_new_blocks(4)
        key = b'a' + bytes(4)
        alias = b'alias' + bytes(4)
        pool._insert_block_hash(key, a, 16)
        pool._insert_block_hash(alias, a, 8)
        pool._insert_block_hash(key, b, 16)  # duplicate physical ownership
        pool._insert_block_hash(b'c' + bytes(4), c, 16)
        pool.free_blocks([a, b, c])
        pool.touch([c])  # referenced cached block is protected
        new = pool.get_new_blocks(1)
        self.assertEqual(new[0].block_id, a.block_id)
        request = SimpleNamespace(request_id='engine', block_hashes=[b'a', b'alias'], num_tokens=32)
        t.request_scope(request)
        hit = pool.get_cached_block(b'a', [0])
        self.assertEqual(hit, [b])
        pool.touch(hit)
        self.assertIsNone(pool.get_cached_block(b'alias', [0]))
        t.request_scope()
        events = self.events()
        decision = [e for e in events if e['type'] == 'allocation_begin'][-1]['data']
        self.assertEqual([x['block_id'] for x in decision['candidates']], [a.block_id, b.block_id])
        self.assertEqual(decision['required_victim_count'], 1)
        report = t.reuse_report(events)
        self.assertEqual(len(report), 2)
        rows = {r['prefix_hash']: r for r in report}
        self.assertTrue(rows[b'a'.hex()]['first_lookup']['hit'])
        self.assertFalse(rows[b'alias'.hex()]['first_lookup']['hit'])
        self.assertEqual(rows[b'a'.hex()]['first_touch']['block_id'], b.block_id)
        self.assertTrue(all(r['first_demand'] for r in report))

    def test_original_and_patched_pool_have_identical_state(self):
        stock, instrumented = load_pool(False)(12, True, 16), load_pool()(12, True, 16)
        rng = random.Random(699)
        for step in range(180):
            active = [b.block_id for b in stock.blocks if b.ref_cnt > 0 and not b.is_null]
            if stock.get_num_free_blocks() and (not active or rng.random() < .6):
                n = rng.randint(1, min(3, stock.get_num_free_blocks()))
                for pool in (stock, instrumented):
                    bs = pool.get_new_blocks(n)
                    for j, b in enumerate(bs):
                        pool._insert_block_hash(f'{step}:{j}'.encode() + bytes(4), b, 16)
            else:
                bid = rng.choice(active)
                for pool in (stock, instrumented):
                    pool.free_blocks([pool.blocks[bid]])
            def state(p):
                return ([t.block_record(p, b) for b in p.blocks],
                        [b.block_id for b in p.free_block_queue.get_all_free_blocks()])
            self.assertEqual(state(stock), state(instrumented))

    def test_unexamined_waiters_and_allocation_gate(self):
        pool = load_pool()(4, True, 16)
        a, b = [SimpleNamespace(request_id=x, status='WAITING', num_computed_tokens=0) for x in ['a', 'b']]
        s = SimpleNamespace(current_step=1, scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
                            cache_config={}, parallel_config={}, max_num_scheduled_tokens=8,
                            max_num_running_reqs=2, running=[], waiting=[a,b], skipped_waiting=[],
                            kv_cache_manager=SimpleNamespace(block_pool=pool))
        t.step_start(s)
        t.waiting_attempt(s, a, 8, 8)
        t.allocation_check(s.kv_cache_manager, a, 'full_sequence', 5, 3, 4, 0, 0, 0)
        t.waiting_end(s, 8, 8, 0, False, False, {})
        remaining = [e for e in self.events() if e['type'] == 'waiting_remaining']
        self.assertEqual([e['data']['evaluated_in_step'] for e in remaining], [True, False])
        self.assertTrue(all(e['data']['allocation_feasible'] is None for e in remaining))

    def test_disabled_telemetry_and_censored_reuse(self):
        with patch.dict('os.environ', {'CACHE_DELAY_TELEMETRY_DIR': ''}):
            pool = load_pool()(3, True, 16)
            pool.get_new_blocks(1)
        self.assertEqual(self.events(), [])
        pool = load_pool()(2, True, 16)
        b = pool.get_new_blocks(1)[0]
        pool._insert_block_hash(b'x' + bytes(4), b, 16)
        pool.free_blocks([b])
        pool.get_new_blocks(1)
        self.assertEqual(t.reuse_report(self.events())[0]['observation'], 'right_censored')

    def test_collection_audit_detects_missing_records(self):
        import copy
        import runpy
        analyze = runpy.run_path(str(ROOT/'scripts/analyze_eviction_telemetry.py'))['analyze']
        pool = load_pool()(2, True, 16)
        b = pool.get_new_blocks(1)[0]
        pool._insert_block_hash(b'x' + bytes(4), b, 16)
        pool.free_blocks([b])
        pool.get_new_blocks(1)
        events = self.events()
        report, _, _ = analyze(events, [])
        self.assertTrue(report['valid'], report['errors'])
        self.assertEqual(report['physical_evictions'], 1)
        self.assertFalse(analyze(events[:-1], [])[0]['valid'])
        bad = copy.deepcopy(events)
        next(e for e in bad if e['type']=='cache_remove')['data']['ref_cnt'] = 1
        self.assertFalse(analyze(bad, [])[0]['valid'])


if __name__ == '__main__':
    unittest.main()
