import json
import tempfile
import unittest
from pathlib import Path

from cache_delay_eval.audit import audit


class AuditTests(unittest.TestCase):
    def test_detects_invariant_mismatch(self):
        row = {
            "status": "ok",
            "prompt_tokens_target": 128,
            "prompt_tokens_reported": 127,
            "cached_prefix_tokens_target": 64,
            "prefix_cache_hit_tokens_batch": 64,
            "queue_depth_target": 2,
            "queue_depth_observed": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            checks = audit(path)
        self.assertEqual(checks["rows"], 1)
        self.assertEqual(checks["prompt_length_mismatches"], 1)
        self.assertEqual(checks["cache_hit_mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
