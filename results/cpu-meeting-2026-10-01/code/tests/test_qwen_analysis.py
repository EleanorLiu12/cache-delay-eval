"""Adversarial checks for scientific prefix and dependency measurements."""
import json
from pathlib import Path
import random
import tempfile
import unittest

from cache_delay_eval.analyze_qwen_trace import (
    add_context_flags, attach_lineage, lcp, prefix_metrics, prevalence, read_records,
)


def record(identifier, timestamp, blocks, component=None, parent=-1, length=None):
    return dict(chat_id=identifier, parent_chat_id=parent, timestamp=timestamp,
                input_length=length or len(blocks) * 16, output_length=1,
                type="text", turn=1, blocks=tuple(blocks),
                component_id=identifier if component is None else component,
                root_observed=True, source_line=identifier + 1)


class QwenAnalysisTests(unittest.TestCase):
    def test_partial_blocks_excluded_and_nonprefix_matches_do_not_count(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            rows = []
            for i, (length, hashes) in enumerate([(17, [1, 2]), (32, [1, 2]), (32, [9, 2])]):
                rows.append(dict(chat_id=i, parent_chat_id=-1, timestamp=i, input_length=length,
                                 output_length=1, type="text", turn=1, hash_ids=hashes))
            path.write_text("\n".join(json.dumps(r) for r in rows))
            loaded, _, _ = read_records(path)
            attach_lineage(loaded)
            output = prefix_metrics(loaded)
            self.assertEqual(output[1]["historical_prefix_tokens"], 16)
            self.assertEqual(output[2]["historical_prefix_tokens"], 0)

    def test_equal_arrivals_are_not_history_and_inputs_not_outputs(self):
        rows = [record(0, 0, [1]), record(1, 0, [1, 2]), record(2, 1, [1, 2, 3])]
        rows[0]["output_length"] = 1000
        output = prefix_metrics(rows)
        self.assertEqual(output[1]["historical_prefix_tokens"], 0)
        self.assertEqual(output[2]["historical_prefix_tokens"], 32)

    def test_neighbors_agree_with_brute_force_including_cross_component(self):
        randomizer = random.Random(699)
        rows = [record(i, i // 3, [randomizer.randrange(4) for _ in range(randomizer.randrange(1, 9))],
                       component=randomizer.randrange(5)) for i in range(180)]
        output = prefix_metrics(rows)
        for row, metrics in zip(rows, output):
            earlier = [r for r in rows if r["timestamp"] < row["timestamp"]]
            for field, candidates in [
                ("historical_prefix_tokens", earlier),
                ("within_component_prefix_tokens", [r for r in earlier if r["component_id"] == row["component_id"]]),
                ("cross_component_prefix_tokens", [r for r in earlier if r["component_id"] != row["component_id"]]),
            ]:
                expected = max((lcp(row["blocks"], r["blocks"]) for r in candidates), default=0) * 16
                self.assertEqual(metrics[field], expected, (row["chat_id"], field))

    def test_lineage_distinguishes_missing_root_and_detects_cycles(self):
        rows = [record(0, 0, [1]), record(1, 1, [1, 2], parent=0),
                record(2, 2, [3], parent=99), record(3, 3, [3, 4], parent=2)]
        _, groups, audit = attach_lineage(rows)
        self.assertEqual(set(groups), {0, 99})
        self.assertFalse(rows[3]["root_observed"])
        self.assertEqual(audit["missing_parent_edges"], 1)
        with self.assertRaisesRegex(ValueError, "Cycle"):
            attach_lineage([record(0, 0, [1], parent=1), record(1, 1, [1], parent=0)])

    def test_cooccurrence_excludes_self_components_simultaneity_and_future(self):
        records = prefix_metrics([record(0, 0, [1] * 10, component=0),
                                  record(1, 1, [1] * 10, component=0),
                                  record(2, 1, [1] * 5, component=2),
                                  record(3, 11, [1] * 5, component=3)])
        add_context_flags(records, 160)
        self.assertIsNone(records[1]["seconds_since_other_component_long"])
        self.assertEqual(records[2]["seconds_since_other_component_long"], 1)
        self.assertEqual(records[3]["seconds_since_other_component_long"], 10)
        result = prevalence(records, .5, 10, 80)
        self.assertEqual(result["short_event_requests"], dict(numerator=2, denominator=4, fraction=.5))
        self.assertEqual(result["event_given_high_reuse"]["denominator"], 3)


if __name__ == "__main__":
    unittest.main()
