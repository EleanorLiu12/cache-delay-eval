"""Independent exhaustive cases for prefix expiry and arrival comparisons."""
import random
import unittest

from cache_delay_eval.analyze_qwen_temporal import (
    add_context_witnesses, mixed_comparison, prefix_recency, standardized_mixed,
    stratified_mixed,
)
from cache_delay_eval.analyze_qwen_trace import prefix_metrics


def row(i, t, blocks, component=None, length=None):
    return dict(chat_id=i, parent_chat_id=-1, timestamp=t, input_length=length or 16*len(blocks),
                output_length=1, type="api", turn=1, source_line=i+1,
                blocks=tuple(blocks), component_id=i if component is None else component,
                root_observed=True)


def brute(left, right):
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 16
    return count


class TemporalTests(unittest.TestCase):
    def test_recency_matches_exhaustive_all_scopes_and_witnesses(self):
        rng = random.Random(699)
        rows = [row(i, (i//3)*.5, [rng.randrange(3) for _ in range(rng.randrange(0, 7))],
                    component=rng.randrange(8), length=1+rng.randrange(16)) for i in range(150)]
        # Block tuples are already full blocks, independently of parser tests.
        for r in rows:
            r["input_length"] += 16*len(r["blocks"])
        result = prefix_recency(rows, (1, 10, 60, 300, None))
        by_id = {r["chat_id"]: r for r in rows}
        for r in rows:
            for label, scopes in result[r["chat_id"]].items():
                history = [p for p in rows if 0 < r["timestamp"]-p["timestamp"] and
                           (label == "unlimited" or r["timestamp"]-p["timestamp"] <= int(label))]
                for scope, evidence in scopes.items():
                    candidates = [p for p in history if scope == "all" or
                                  ((p["component_id"] == r["component_id"]) == (scope == "within"))]
                    expected = max((brute(r["blocks"], p["blocks"]) for p in candidates), default=0)
                    self.assertEqual(evidence["prefix_tokens"], expected)
                    if expected:
                        witness = by_id[evidence["witness_chat_id"]]
                        self.assertIn(witness, candidates)
                        self.assertEqual(brute(r["blocks"], witness["blocks"]), expected)
                    else:
                        self.assertIsNone(evidence["witness_chat_id"])

    def test_boundary_expiry_ties_and_nonprefix_blocks(self):
        rows = [row(0, 0, [1,2], 0), row(1, 1, [1,2], 0), row(2, 1, [1,2,3], 2),
                row(3, 1.0001, [1,2,3], 3), row(4, 2.0002, [9,2], 4)]
        result = prefix_recency(rows, (1,))
        self.assertEqual(result[1]["1"]["within"]["prefix_tokens"], 32)
        self.assertEqual(result[2]["1"]["all"]["prefix_tokens"], 32)
        self.assertEqual(result[3]["1"]["all"]["prefix_tokens"], 48)
        self.assertEqual(result[4]["1"]["all"]["prefix_tokens"], 0)

    def test_decimal_window_boundary_is_not_lost_to_binary_rounding(self):
        rows=[row(0, 1.003, [1]), row(1, 2.003, [1])]
        result=prefix_recency(rows,(1,))
        self.assertEqual(result[1]["1"]["all"]["prefix_tokens"],16)
        self.assertEqual(result[1]["1"]["all"]["age_seconds"],1)

    def test_precursors_exclude_same_chain_same_time_and_future(self):
        rows = prefix_metrics([row(0, 0, [1]*10, 0), row(1, 1, [1]*10, 0),
                               row(2, 1, [1]*5, 2), row(3, 11, [1]*5, 3)])
        add_context_witnesses(rows, {"p90":160})
        self.assertIsNone(rows[1]["contexts"]["p90"]["any_long"]["age_seconds"])
        self.assertEqual(rows[2]["contexts"]["p90"]["any_long"]["witness_chat_id"], 0)
        self.assertEqual(rows[3]["contexts"]["p90"]["any_long"]["age_seconds"], 10)
        self.assertEqual(rows[3]["contexts"]["p90"]["low_overlap_long"]["age_seconds"], 11)
        compare = mixed_comparison(rows, "p90", 160, "any_long", 10)
        self.assertEqual(compare["high"]["requests"]["numerator"], 2)
        self.assertEqual(compare["high"]["requests"]["denominator"], 3)
        self.assertEqual(compare["low"]["requests"]["denominator"], 1)

    def test_chain_groups_can_overlap_and_empty_rates_are_undefined(self):
        rows = prefix_metrics([row(0,0,[1],0), row(1,1,[1],0)])
        add_context_witnesses(rows, {"p90":16})
        result = mixed_comparison(rows, "p90", 16, "any_long", 1)
        self.assertEqual(result["chains_in_both_groups"], 1)
        result = mixed_comparison(rows[:1], "p90", 16, "any_long", 1)
        self.assertIsNone(result["request_difference_pp"])

    def test_standardization_uses_supported_joint_cells_only(self):
        rows = prefix_metrics([row(i,i,[1],i) for i in range(50)])
        add_context_witnesses(rows, {"p90":16})
        result = standardized_mixed(stratified_mixed(rows,{"p90":16}), len(rows))
        self.assertEqual(result[0]["coverage"]["numerator"], 0)
        self.assertIsNone(result[0]["standardized_difference_pp"])


if __name__ == "__main__":
    unittest.main()
