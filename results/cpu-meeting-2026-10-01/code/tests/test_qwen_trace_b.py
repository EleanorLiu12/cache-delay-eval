"""Scientific correctness checks for type discovery, topology and acquisition."""
import importlib.util
from pathlib import Path
import sys
import unittest

from cache_delay_eval.analyze_qwen_trace import attach_lineage, prefix_metrics
from cache_delay_eval.analyze_qwen_trace_b import cohort_turn_summary, make_cohorts, topology_audit


def row(identifier, parent=-1, kind="api", turn=1):
    return dict(chat_id=identifier, parent_chat_id=parent, timestamp=identifier,
                type=kind, turn=turn, source_line=identifier + 1, input_length=32,
                output_length=1, blocks=(1, 2))


class TraceBTests(unittest.TestCase):
    def test_type_discovery_excludes_mixed_components_from_uniform_type_cohorts(self):
        rows = [row(0), row(1, kind="text"), row(2, kind="api"), row(3, parent=2, kind="text", turn=2)]
        _, groups, _ = attach_lineage(rows)
        cohorts = make_cohorts(rows, groups)
        self.assertEqual(set(cohorts), {"all_types", "observed_root_api", "observed_root_text"})
        self.assertEqual([r["chat_id"] for r in cohorts["observed_root_api"]], [0])
        self.assertEqual([r["chat_id"] for r in cohorts["observed_root_text"]], [1])

    def test_branch_depth_is_distinct_from_component_size_and_declared_turn(self):
        rows = [row(0), row(1, parent=0, turn=2), row(2, parent=0, turn=2), row(3, parent=1, turn=7)]
        by_id, groups, lineage = attach_lineage(rows)
        audit = topology_audit(rows, by_id, groups, lineage)
        self.assertEqual(audit["branching_parents"], 1)
        self.assertEqual(audit["branching_components"], 1)
        self.assertEqual(audit["maximum_children"], 2)
        self.assertEqual(audit["component_maximum_path_depth"]["maximum"], 3)
        self.assertEqual(audit["nonconsecutive_turn_edges"], 1)
        self.assertEqual([r["observed_path_depth"] for r in rows], [1, 2, 2, 3])

    def test_singleton_initial_cohort_does_not_claim_empty_continuation_rate(self):
        rows = [row(0), row(1)]
        attach_lineage(rows)
        summary = cohort_turn_summary(prefix_metrics(rows))
        self.assertEqual(summary["initial"]["historical_overlap_at_least_50pct"]["numerator"], 1)
        self.assertEqual(summary["initial"]["within_component_overlap_at_least_50pct"]["numerator"], 0)
        self.assertEqual(summary["continuation"]["requests"], 0)
        self.assertIsNone(summary["continuation"]["historical_overlap_at_least_50pct"]["fraction"])

    def test_lfs_pointer_is_parsed_exactly(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location("fetch_qwen_trace_b", scripts / "fetch_qwen_trace_b.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 42\n"
            self.assertEqual(module.parse_lfs_pointer(pointer), dict(sha256="a" * 64, bytes=42))
            for malformed in (pointer + "size 77\n", pointer.replace("sha256:", "sha1:"), pointer.replace("size 42", "size 0")):
                with self.assertRaises(ValueError):
                    module.parse_lfs_pointer(malformed)
        finally:
            sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()
