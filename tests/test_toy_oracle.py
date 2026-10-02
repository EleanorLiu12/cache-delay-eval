import unittest
from dataclasses import replace
from itertools import permutations

from cache_delay_eval.toy_oracle import (
    Block, Instance, Request, eligible, finished, initial_state, objective,
    prefix_blocks, progress_limits, replay, solve, standard_instances, transition, violations,
)


class ToyOracleTests(unittest.TestCase):
    def test_hand_calculated_serial_service_and_arrival(self):
        instance = Instance('manual', (Request('a', (1,), (2,)), Request('b', (3,), (4,), 10)),
                            kv_blocks=2, max_sequences=1, token_budget=1)
        state, events = replay(instance)
        self.assertEqual([p.output_times for p in state.progress], [(4,), (14,)])
        self.assertEqual(objective(instance, state), 8)
        self.assertTrue(any(e['forced_idle'] for e in events))

    def test_identity_is_full_prefix_and_requires_computed_blocks(self):
        blocks = (Block(0, (1,), True, (), 0), Block(1, (1, 2), False, (0,), 1),
                  Block(2, (9, 2), True, (), 2))
        self.assertEqual(prefix_blocks(Request('a', (1, 2), (3,)), blocks), (0,))
        self.assertEqual(prefix_blocks(Request('b', (9, 2), (3,)), blocks), ())

    def test_full_prompt_head_failure_has_feasible_tail(self):
        instance = standard_instances()[-1]
        _, events = replay(instance)
        failures = [e['stop'] for e in events if e['stop'] and e['stop']['reason'] == 'full_prompt_kv']
        self.assertTrue(failures)
        self.assertTrue(any(a['fits'] for s in failures for a in s['alternatives']))
        result = solve(instance)
        self.assertTrue(result.exact)
        self.assertLess(objective(instance, result.best), objective(instance, result.baseline))
        restored, _ = replay(instance, result.actions)
        self.assertEqual(restored, result.best)
        self.assertFalse(violations(instance, restored, result.limits))

    def test_search_matches_unpruned_enumeration(self):
        instance = Instance('independent', (Request('a', (1, 2), (3,)),
                                            Request('b', (4,), (5,)), Request('c', (6,), (7,))),
                            kv_blocks=4, max_sequences=2, token_budget=2, slack_percent=100)
        baseline, _ = replay(instance)
        limits = progress_limits(instance, baseline)
        objectives = []
        def enumerate_all(state):
            if finished(state):
                if not violations(instance, state, limits):
                    objectives.append(objective(instance, state))
                return
            for order in permutations(eligible(instance, state)):
                enumerate_all(transition(instance, state, order)[0])
        enumerate_all(initial_state(instance))
        result = solve(instance)
        self.assertTrue(result.exact)
        self.assertEqual(objective(instance, result.best), min(objectives))

    def test_state_limit_never_claims_optimality(self):
        result = solve(standard_instances()[-1], max_states=1)
        self.assertFalse(result.exact)
        self.assertEqual(result.certificate(standard_instances()[-1])['status'], 'offline_search_benchmark')

    def test_all_cases_sensitivities_and_cache_invariants(self):
        for source in standard_instances():
            for slack in (0, 5, 10):
                instance = replace(source, slack_percent=slack)
                with self.subTest(case=source.name, slack=slack):
                    result = solve(instance)
                    self.assertTrue(result.exact)
                    self.assertFalse(violations(instance, result.baseline, result.limits))
                    self.assertFalse(violations(instance, result.best, result.limits))
                    replay(instance, result.actions)  # checks every state's capacity/ownership
        _, events = replay(standard_instances()[2])
        self.assertTrue(any(e['evictions'] for e in events))
        self.assertTrue(any(a['cached_tokens'] for e in events for a in e['admissions']))

    def test_rejects_impossible_requests_and_illegal_actions(self):
        instance = standard_instances()[0]
        with self.assertRaises(ValueError):
            transition(instance, initial_state(instance), (0, 0, 1, 2))
        with self.assertRaises(ValueError):
            initial_state(replace(instance, kv_blocks=1))
