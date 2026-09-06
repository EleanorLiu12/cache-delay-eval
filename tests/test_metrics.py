import unittest

from cache_delay_eval.http_client import metric, parse_prometheus
from cache_delay_eval.summarize import percentile


class MetricTests(unittest.TestCase):
    def test_prometheus_samples_sum_across_labels(self):
        parsed = parse_prometheus(
            '# HELP vllm:test example\n'
            'vllm:test{model="a"} 2\n'
            'vllm:test{model="b"} 3\n'
        )
        self.assertEqual(metric(parsed, "vllm:test"), 5)

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([0, 10], 0.95), 9.5)


if __name__ == "__main__":
    unittest.main()

