import json
import tempfile
import unittest
from pathlib import Path

from cache_delay_eval.calibration import (
    _conditions,
    _resume_state,
    _schedule,
)
from cache_delay_eval.analyze_calibration import _bootstrap_median_ci, _condition_stats


class CalibrationResumeTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "base_url": "http://127.0.0.1:8000",
            "model": "test-model",
            "prompt_tokens": [128],
            "cached_prefix_tokens": [0, 64],
            "queue_depths": [0, 2],
            "block_size": 16,
            "trials": 2,
            "seed": 699,
        }
        self.schedule = _schedule(_conditions(self.config, 16), 2, 699)

    def _row(self, sequence):
        trial, prompt, prefix, queue = self.schedule[sequence]
        return {
            "type": "calibration_result",
            "run_id": "20260903T154541-289eb1748c454b5bbb7033e1bd7a69b8",
            "sequence": sequence,
            "trial": trial,
            "model": "test-model",
            "block_size": 16,
            "prompt_tokens_target": prompt,
            "cached_prefix_tokens_target": prefix,
            "queue_depth_target": queue,
            "status": "ok",
        }

    def test_resume_accepts_valid_noncontiguous_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "partial.jsonl"
            rows = [self._row(0), self._row(3)]
            output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            run_id, completed, prompt_scheme = _resume_state(
                output, self.schedule, self.config, 2
            )
        self.assertEqual(run_id, rows[0]["run_id"])
        self.assertEqual(completed, {0, 3})
        self.assertEqual(prompt_scheme, "offset-v1")

    def test_resume_rejects_a_changed_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "partial.jsonl"
            row = self._row(0)
            row["queue_depth_target"] = 99
            output.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configured schedule"):
                _resume_state(output, self.schedule, self.config, 2)

    def test_resume_rejects_duplicate_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "partial.jsonl"
            row = self._row(0)
            output.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate sequence"):
                _resume_state(output, self.schedule, self.config, 2)


class CalibrationAnalysisTests(unittest.TestCase):
    def test_bootstrap_interval_is_exact_for_constant_sample(self):
        self.assertEqual(_bootstrap_median_ci([5.0, 5.0, 5.0], 699), (5.0, 5.0))

    def test_condition_stats_preserve_mean_median_and_p95(self):
        rows = {
            (128, 0, 0): [
                {"ttft_ms": 10.0},
                {"ttft_ms": 20.0},
                {"ttft_ms": 30.0},
            ]
        }
        stats = _condition_stats(rows, 699)[(128, 0, 0)]
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["mean"], 20.0)
        self.assertEqual(stats["median"], 20.0)
        self.assertEqual(stats["p95"], 29.0)


if __name__ == "__main__":
    unittest.main()
