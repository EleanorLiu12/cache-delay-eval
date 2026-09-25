import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cache_delay_eval.session_gen import (
    DEEP_SHORT,
    SHALLOW_LONG,
    GeneratorConfig,
    WorkloadKnobs,
    build_trace,
    generate_requests,
    nearest_mix,
    rho_profile,
)
from cache_delay_eval.trace import read_trace, write_trace


SMALL = GeneratorConfig(requests=120, prefix_groups=2, seed=0)


class KnobGuardTests(unittest.TestCase):
    def test_rejects_out_of_range_implemented_knobs(self):
        for knobs in (WorkloadKnobs(P=0), WorkloadKnobs(W=0.0), WorkloadKnobs(rho_mix=1.5)):
            with self.subTest(knobs=knobs):
                with self.assertRaises(ValueError):
                    generate_requests(knobs, SMALL)


class TraceShapeTests(unittest.TestCase):
    def test_round_trips_through_the_v1_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.jsonl"
            header, requests, stats = build_trace(WorkloadKnobs(rho_mix=0.5), SMALL)
            write_trace(path, header, requests)
            actual_header, actual_requests = read_trace(path)
            self.assertEqual(actual_requests, requests)
            self.assertEqual(actual_header.block_size, SMALL.block_size)
            self.assertEqual(actual_header.metadata["rho"], stats["rho"])

    def test_same_seed_reproduces_the_trace(self):
        first, _ = generate_requests(WorkloadKnobs(rho_mix=0.4), SMALL)
        second, _ = generate_requests(WorkloadKnobs(rho_mix=0.4), SMALL)
        self.assertEqual(first, second)
        third, _ = generate_requests(WorkloadKnobs(rho_mix=0.4), replace(SMALL, seed=1))
        self.assertNotEqual(first, third)

    def test_request_count_holds_steady_across_the_mixture(self):
        """Cells must differ in rho, not in trace size."""
        for mix in (0.0, 0.5, 1.0):
            with self.subTest(mix=mix):
                requests, _ = generate_requests(WorkloadKnobs(rho_mix=mix), SMALL)
                self.assertGreaterEqual(len(requests), SMALL.requests)
                self.assertLess(len(requests), SMALL.requests + DEEP_SHORT.turns)


class BlockHashTests(unittest.TestCase):
    """The properties a block-level replay depends on."""

    def _by_session(self, requests):
        sessions: dict[str, list] = {}
        for request in requests:
            sessions.setdefault(request.session_id, []).append(request)
        for turns in sessions.values():
            turns.sort(key=lambda r: r.metadata["turn_index"])
        return sessions

    def test_each_turn_extends_the_previous_turns_blocks(self):
        requests, _ = generate_requests(WorkloadKnobs(rho_mix=0.5), SMALL)
        for session_id, turns in self._by_session(requests).items():
            for earlier, later in zip(turns, turns[1:]):
                with self.subTest(session=session_id):
                    self.assertGreaterEqual(len(later.block_hashes), len(earlier.block_hashes))
                    self.assertEqual(
                        later.block_hashes[: len(earlier.block_hashes)], earlier.block_hashes
                    )

    def test_same_prefix_group_shares_exactly_the_whole_blocks_of_the_prefix(self):
        knobs = WorkloadKnobs(P=512)
        requests, _ = generate_requests(knobs, SMALL)
        expected_shared = knobs.P // SMALL.block_size
        first_turns = [r for r in requests if r.metadata["turn_index"] == 0]
        by_group: dict[str, list] = {}
        for request in first_turns:
            by_group.setdefault(request.prefix_group, []).append(request)
        pairs_checked = 0
        for group, members in by_group.items():
            for a, b in zip(members, members[1:]):
                shared = self._common_prefix_length(a.block_hashes, b.block_hashes)
                self.assertEqual(shared, expected_shared, f"within {group}")
                pairs_checked += 1
        self.assertGreater(pairs_checked, 0, "test needs at least one same-group pair")

    def test_different_prefix_groups_share_nothing(self):
        requests, _ = generate_requests(WorkloadKnobs(P=512), SMALL)
        first_turns = [r for r in requests if r.metadata["turn_index"] == 0]
        cross = [
            (a, b)
            for a in first_turns
            for b in first_turns
            if a.prefix_group != b.prefix_group
        ]
        self.assertTrue(cross, "test needs at least two prefix groups present")
        for a, b in cross[:20]:
            self.assertEqual(self._common_prefix_length(a.block_hashes, b.block_hashes), 0)

    def test_a_block_straddling_the_prefix_boundary_is_not_shared(self):
        """P = 520 with block_size 16 leaves 8 tokens of prefix in a mixed block."""
        config = replace(SMALL, block_size=16)
        requests, _ = generate_requests(WorkloadKnobs(P=520), config)
        first_turns = [r for r in requests if r.metadata["turn_index"] == 0]
        by_group: dict[str, list] = {}
        for request in first_turns:
            by_group.setdefault(request.prefix_group, []).append(request)
        for members in by_group.values():
            for a, b in zip(members, members[1:]):
                # floor(520 / 16) == 32 whole shared blocks, not 33.
                self.assertEqual(self._common_prefix_length(a.block_hashes, b.block_hashes), 32)

    def test_longer_shared_prefix_raises_the_shared_block_count(self):
        short, _ = generate_requests(WorkloadKnobs(P=512), SMALL)
        long, _ = generate_requests(WorkloadKnobs(P=8192), SMALL)
        self.assertLess(self._first_turn_reuse(short), self._first_turn_reuse(long))

    @staticmethod
    def _common_prefix_length(left, right):
        length = 0
        for a, b in zip(left, right):
            if a != b:
                break
            length += 1
        return length

    @staticmethod
    def _first_turn_reuse(requests):
        first = [r for r in requests if r.metadata["turn_index"] == 0]
        return sum(r.metadata["oracle_hit_rate"] for r in first) / len(first)


class WorkingSetTests(unittest.TestCase):
    def test_capacity_realises_the_requested_ratio(self):
        for W in (0.3, 1.0, 3.0):
            with self.subTest(W=W):
                _, stats = generate_requests(WorkloadKnobs(W=W), SMALL)
                self.assertEqual(
                    stats["capacity_blocks"], max(1, round(stats["working_set_blocks"] / W))
                )

    def test_only_W_above_one_can_evict(self):
        _, loose = generate_requests(WorkloadKnobs(W=0.3), SMALL)
        _, tight = generate_requests(WorkloadKnobs(W=3.0), SMALL)
        self.assertGreater(loose["capacity_blocks"], loose["working_set_blocks"])
        self.assertLess(tight["capacity_blocks"], tight["working_set_blocks"])


class RhoTests(unittest.TestCase):
    """The knob the whole design hinges on: rho must reach negative values."""

    def test_pure_deep_short_gives_positive_coupling(self):
        _, stats = generate_requests(WorkloadKnobs(rho_mix=0.0), SMALL)
        self.assertGreater(stats["rho"], 0.3)

    def test_the_mixture_reaches_negative_rho(self):
        profile = rho_profile(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], WorkloadKnobs(P=8192), SMALL, seeds=(0, 1, 2)
        )
        self.assertLess(min(row["rho_mean"] for row in profile), 0.0)
        self.assertGreater(max(row["rho_mean"] for row in profile), 0.0)

    def test_rho_is_not_monotone_in_the_mixture(self):
        """Both pure archetypes couple positively; the negative values are the
        between-archetype contrast, so callers must scan rather than bisect."""
        profile = rho_profile([0.0, 0.5, 1.0], WorkloadKnobs(P=512), SMALL, seeds=(0, 1, 2))
        ends = (profile[0]["rho_mean"], profile[-1]["rho_mean"])
        self.assertGreater(min(ends), profile[1]["rho_mean"])

    def test_single_turn_shallow_archetype_reaches_strongly_negative_rho(self):
        """The 3-turn shallow archetype caps |rho|; its later turns are long
        *and* well cached, which cancels the negative contrast."""
        config = replace(SMALL, shallow=replace(SHALLOW_LONG, turns=1))
        profile = rho_profile([0.9], WorkloadKnobs(P=8192), config, seeds=(0, 1))
        self.assertLess(profile[0]["rho_mean"], -0.8)

    def test_nearest_mix_picks_the_closest_grid_point(self):
        profile = [
            {"rho_mix": 0.0, "rho_mean": 0.6},
            {"rho_mix": 0.5, "rho_mean": -0.1},
            {"rho_mix": 0.8, "rho_mean": -0.4},
        ]
        self.assertEqual(nearest_mix(profile, -0.35), 0.8)
        self.assertEqual(nearest_mix(profile, 0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
