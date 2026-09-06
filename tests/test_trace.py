import json
import tempfile
import unittest
from pathlib import Path

from cache_delay_eval.trace import (
    TraceHeader,
    TraceRequest,
    TraceValidationError,
    read_trace,
    write_trace,
)


class TraceTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            header = TraceHeader("trace", "2026-09-03T00:00:00Z", "unit-test")
            requests = [
                TraceRequest("a", 0, 1, prompt="hello", session_id="s"),
                TraceRequest("b", 2, 2, token_ids=(1, 2), session_id="s", parent_request_id="a"),
            ]
            write_trace(path, header, requests)
            actual_header, actual_requests = read_trace(path)
            self.assertEqual(actual_header.trace_id, "trace")
            self.assertEqual(actual_requests, requests)

    def test_rejects_both_prompt_forms(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            rows = [
                {
                    "type": "trace_meta",
                    "schema_version": "1.1",
                    "trace_id": "t",
                    "created_at": "2026-09-03T00:00:00Z",
                    "time_unit": "ms",
                    "source": "test",
                },
                {
                    "type": "request",
                    "request_id": "r",
                    "arrival_time_ms": 0,
                    "output_tokens": 1,
                    "prompt": "x",
                    "token_ids": [1],
                },
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaises(TraceValidationError):
                read_trace(path)

    def test_rejects_out_of_order_arrivals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            header = TraceHeader("trace", "2026-09-03T00:00:00Z", "unit-test")
            write_trace(
                path,
                header,
                [TraceRequest("a", 2, 1, prompt="a"), TraceRequest("b", 1, 1, prompt="b")],
            )
            with self.assertRaisesRegex(TraceValidationError, "earlier"):
                read_trace(path)


    def test_block_hash_round_trip(self):
        """A Mooncake-style record carries neither prompt nor token_ids."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            header = TraceHeader(
                "mooncake", "2026-09-03T00:00:00Z", "mooncake-trace", block_size=512
            )
            requests = [
                TraceRequest(
                    "a", 0, 500, block_hashes=(46, 47, 48), prompt_tokens=1536,
                    session_id="s", user_id="u",
                ),
                TraceRequest(
                    "b", 27482, 500, block_hashes=(46, 47, 99), prompt_tokens=1600,
                    session_id="s", user_id="u", parent_request_id="a",
                ),
            ]
            write_trace(path, header, requests)
            actual_header, actual_requests = read_trace(path)
            self.assertEqual(actual_header.block_size, 512)
            self.assertEqual(actual_requests, requests)
            self.assertEqual(actual_requests[1].block_hashes[:2], (46, 47))

    def test_rejects_block_hashes_without_prompt_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            self._write_rows(path, [{
                "type": "request", "request_id": "r", "arrival_time_ms": 0,
                "output_tokens": 1, "block_hashes": [1, 2],
            }])
            with self.assertRaisesRegex(TraceValidationError, "prompt_tokens is required"):
                read_trace(path)

    def test_rejects_all_three_prompt_forms(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            self._write_rows(path, [{
                "type": "request", "request_id": "r", "arrival_time_ms": 0,
                "output_tokens": 1, "prompt": "x", "token_ids": [1],
                "block_hashes": [1], "prompt_tokens": 4,
            }])
            with self.assertRaisesRegex(TraceValidationError, "exactly one"):
                read_trace(path)

    def test_rejects_no_prompt_form(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            self._write_rows(path, [{
                "type": "request", "request_id": "r", "arrival_time_ms": 0, "output_tokens": 1,
            }])
            with self.assertRaisesRegex(TraceValidationError, "exactly one"):
                read_trace(path)

    def test_rejects_session_claimed_by_two_users(self):
        """Guards against conversion bugs when deriving user_id from a hashed client id."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            header = TraceHeader("trace", "2026-09-03T00:00:00Z", "unit-test")
            write_trace(path, header, [
                TraceRequest("a", 0, 1, prompt="x", session_id="s", user_id="u1"),
                TraceRequest("b", 1, 1, prompt="y", session_id="s", user_id="u2"),
            ])
            with self.assertRaisesRegex(TraceValidationError, "already attributed"):
                read_trace(path)

    def test_reads_schema_1_0_traces(self):
        """v1.0 traces written before the block-hash form stay readable."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.jsonl"
            self._write_rows(path, [{
                "type": "request", "request_id": "r", "arrival_time_ms": 0,
                "output_tokens": 1, "prompt": "x",
            }], schema_version="1.0")
            header, requests = read_trace(path)
            self.assertEqual(len(requests), 1)
            self.assertIsNone(requests[0].block_hashes)

    def test_rejects_unknown_schema_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            self._write_rows(path, [{
                "type": "request", "request_id": "r", "arrival_time_ms": 0,
                "output_tokens": 1, "prompt": "x",
            }], schema_version="9.9")
            with self.assertRaisesRegex(TraceValidationError, "unsupported schema_version"):
                read_trace(path)

    @staticmethod
    def _write_rows(path, requests, schema_version="1.1"):
        rows = [{
            "type": "trace_meta", "schema_version": schema_version, "trace_id": "t",
            "created_at": "2026-09-03T00:00:00Z", "time_unit": "ms", "source": "test",
        }] + list(requests)
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

