"""Unit tests for ``workflows.mutation_distance_analysis``."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from collections import Counter
from typing import Dict, List
from unittest import mock

import matplotlib

matplotlib.use("Agg")  # headless backend for tests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workflows.mutation_distance_analysis import (
    mutation_distance_analysis as mda,
)
from workflows.mutation_distance_analysis.mutation_distance_analysis import (
    _sample_positions_blocking,
    _to_proportions,
    compute_random_distances,
    compute_real_distances,
    counter_to_array,
    plot_difference,
    run_distance_analysis,
)
from workflows.mutation_distribution_analysis.mutation_pool import (
    GENE_STATS_COLUMNS,
    MUTATION_COLUMNS,
    MutationPool,
)


def _mut_row(
    gene_id: str, position: int, source_base: str, new_base: str
) -> Dict[str, object]:
    return {
        "gene_id": gene_id,
        "position": position,
        "source_base": source_base,
        "new_base": new_base,
    }


def _stats_row(
    gene_id: str,
    n_mutations: int,
    initial_fitness: float = 0.0,
    final_fitness: float = 1.0,
) -> Dict[str, object]:
    return {
        "gene_id": gene_id,
        "n_mutations": n_mutations,
        "initial_fitness": initial_fitness,
        "final_fitness": final_fitness,
    }


def _make_pool(
    mutations_records: List[Dict[str, object]],
    refs: Dict[str, str],
    gene_stats_records: List[Dict[str, object]],
) -> MutationPool:
    mutations_df = pd.DataFrame(mutations_records, columns=MUTATION_COLUMNS)
    gene_stats_df = pd.DataFrame(
        gene_stats_records, columns=GENE_STATS_COLUMNS
    )
    return MutationPool(
        mutations=mutations_df,
        gene_stats=gene_stats_df,
        references=dict(refs),
    )


# ---------------------------------------------------------------------------
# _sample_positions_blocking
# ---------------------------------------------------------------------------


class TestSamplePositionsBlocking(unittest.TestCase):
    """Tests for _sample_positions_blocking."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _uniform_weights(n: int) -> np.ndarray:
        """Return a uniform probability vector of length *n*."""
        return np.full(n, 1.0 / n)

    # ------------------------------------------------------------------
    # basic contract
    # ------------------------------------------------------------------

    def test_returns_correct_number_of_positions(self) -> None:
        """Result has exactly *k* elements."""
        positions = np.arange(10)
        weights = self._uniform_weights(10)

        result = _sample_positions_blocking(
            positions, weights, k=4, rng=np.random.default_rng(0)
        )

        self.assertEqual(result.shape, (4,))
        self.assertTrue(all(isinstance(pos, np.integer) for pos in result))
        self.assertTrue(all(0 <= pos < 10 for pos in result))

    def test_all_returned_positions_are_from_input(self) -> None:
        """Every drawn position belongs to *unique_positions*."""
        positions = np.array([3, 7, 15, 22, 40])
        weights = self._uniform_weights(5)

        result = _sample_positions_blocking(
            positions, weights, k=3, rng=np.random.default_rng(1)
        )

        for pos in result:
            self.assertIn(pos, positions)

    def test_all_returned_positions_are_unique(self) -> None:
        """Position-blocking: no duplicates in a single draw."""
        positions = np.arange(20)
        weights = self._uniform_weights(20)

        result = _sample_positions_blocking(
            positions, weights, k=10, rng=np.random.default_rng(2)
        )

        self.assertEqual(len(result), len(set(result.tolist())))

    # ------------------------------------------------------------------
    # boundary / edge cases
    # ------------------------------------------------------------------

    def test_draw_k_equals_pool_size(self) -> None:
        """Drawing all positions returns every position exactly once."""
        positions = np.array([1, 5, 9])
        weights = self._uniform_weights(3)

        result = _sample_positions_blocking(
            positions, weights, k=3, rng=np.random.default_rng(3)
        )

        self.assertEqual(sorted(result.tolist()), [1, 5, 9])

    def test_draw_k_equals_one(self) -> None:
        """Requesting a single position returns a length-1 array."""
        positions = np.array([10, 20, 30])
        weights = self._uniform_weights(3)

        result = _sample_positions_blocking(
            positions, weights, k=1, rng=np.random.default_rng(4)
        )

        self.assertEqual(result.shape, (1,))
        self.assertIn(result[0], positions)

    # ------------------------------------------------------------------
    # error handling
    # ------------------------------------------------------------------

    def test_raises_value_error_when_k_exceeds_pool(self) -> None:
        """ValueError is raised when *k* > number of unique positions."""
        positions = np.array([1, 2, 3])
        weights = self._uniform_weights(3)

        with self.assertRaises(ValueError):
            _sample_positions_blocking(
                positions, weights, k=4, rng=np.random.default_rng(0)
            )

    # ------------------------------------------------------------------
    # determinism
    # ------------------------------------------------------------------

    def test_deterministic_under_fixed_seed(self) -> None:
        """Two calls with the same seed return identical arrays."""
        positions = np.arange(50)
        weights = self._uniform_weights(50)

        result_a = _sample_positions_blocking(
            positions, weights, k=10, rng=np.random.default_rng(99)
        )
        result_b = _sample_positions_blocking(
            positions, weights, k=10, rng=np.random.default_rng(99)
        )

        np.testing.assert_array_equal(result_a, result_b)

    def test_different_seeds_produce_different_draws(self) -> None:
        """Two calls with different seeds almost surely differ for a large pool."""
        positions = np.arange(100)
        weights = self._uniform_weights(100)

        result_a = _sample_positions_blocking(
            positions, weights, k=20, rng=np.random.default_rng(7)
        )
        result_b = _sample_positions_blocking(
            positions, weights, k=20, rng=np.random.default_rng(8)
        )

        # Not guaranteed in general but virtually certain for k=20 out of 100.
        self.assertFalse(
            np.array_equal(np.sort(result_a), np.sort(result_b)),
            "Expected distinct draws from different seeds.",
        )


    # ------------------------------------------------------------------
    # weight bias
    # ------------------------------------------------------------------

    def test_weights_bias_the_draw(self) -> None:
        """A heavily skewed weight vector concentrates draws on one position.

        With k=1, position 0 has weight 0.9 and position 1 has 0.1.
        Over many independent draws the fraction selecting position 0
        should be close to 0.9.
        """
        positions = np.array([0, 1])
        weights = np.array([0.9, 0.1])
        n_trials = 2_000
        rng = np.random.default_rng(42)

        hits = sum(
            _sample_positions_blocking(
                positions, weights, k=1, rng=rng
            )[0] == 0
            for _ in range(n_trials)
        )

        # Allow generous tolerance: expect fraction in [0.85, 0.95].
        fraction = hits / n_trials
        self.assertGreater(fraction, 0.85)
        self.assertLess(fraction, 0.95)


# ---------------------------------------------------------------------------
# compute_real_distances
# ---------------------------------------------------------------------------


class TestComputeRealDistances(unittest.TestCase):
    """Tests for compute_real_distances."""

    def test_two_gene_pool_known_positions(self) -> None:
        """Two genes with known positions produce the expected Counter."""
        # g_a positions [10, 13, 20] -> diffs [3, 7]
        # g_b positions [5, 8, 9, 100] -> diffs [3, 1, 91]
        mutations_records = [
            _mut_row("g_a", 10, "A", "T"),
            _mut_row("g_a", 13, "A", "G"),
            _mut_row("g_a", 20, "A", "C"),
            _mut_row("g_b", 5, "C", "A"),
            _mut_row("g_b", 8, "C", "G"),
            _mut_row("g_b", 9, "C", "T"),
            _mut_row("g_b", 100, "C", "A"),
        ]
        pool = _make_pool(
            mutations_records,
            refs={"g_a": "X", "g_b": "X"},
            gene_stats_records=[
                _stats_row("g_a", 3),
                _stats_row("g_b", 4),
            ],
        )

        result = compute_real_distances(pool)

        expected = Counter({3: 2, 7: 1, 1: 1, 91: 1})
        self.assertEqual(result, expected)

    def test_ordering_insensitivity(self) -> None:
        """Shuffling rows within a gene does not change the result."""
        mutations_ordered = [
            _mut_row("g_a", 10, "A", "T"),
            _mut_row("g_a", 13, "A", "G"),
            _mut_row("g_a", 20, "A", "C"),
        ]
        mutations_shuffled = [
            _mut_row("g_a", 20, "A", "C"),
            _mut_row("g_a", 10, "A", "T"),
            _mut_row("g_a", 13, "A", "G"),
        ]
        pool_ordered = _make_pool(
            mutations_ordered,
            refs={"g_a": "X"},
            gene_stats_records=[_stats_row("g_a", 3)],
        )
        pool_shuffled = _make_pool(
            mutations_shuffled,
            refs={"g_a": "X"},
            gene_stats_records=[_stats_row("g_a", 3)],
        )

        result_ordered = compute_real_distances(pool_ordered)
        result_shuffled = compute_real_distances(pool_shuffled)

        self.assertEqual(result_ordered, result_shuffled)

    def test_gene_with_single_mutation_contributes_nothing(self) -> None:
        """A gene with <2 mutations adds nothing to the global counter."""
        mutations_records = [
            _mut_row("g_a", 10, "A", "T"),
            _mut_row("g_b", 5, "C", "A"),
            _mut_row("g_b", 9, "C", "G"),
        ]
        pool = _make_pool(
            mutations_records,
            refs={"g_a": "X", "g_b": "X"},
            gene_stats_records=[
                _stats_row("g_a", 1),
                _stats_row("g_b", 2),
            ],
        )

        result = compute_real_distances(pool)

        self.assertEqual(result, Counter({4: 1}))


# ---------------------------------------------------------------------------
# compute_random_distances
# ---------------------------------------------------------------------------


def _two_gene_pool_for_random() -> MutationPool:
    """A two-gene pool with plenty of unique positions for sampling."""
    mutations_records: List[Dict[str, object]] = []
    for position in range(50):
        mutations_records.append(
            _mut_row("pool", position, "A", "T")
        )
    return _make_pool(
        mutations_records,
        refs={"g_a": "X", "g_b": "X"},
        gene_stats_records=[
            _stats_row("g_a", n_mutations=4),
            _stats_row("g_b", n_mutations=6),
        ],
    )


class TestComputeRandomDistances(unittest.TestCase):
    """Tests for compute_random_distances."""

    def test_deterministic_under_seed(self) -> None:
        """Two runs with the same seed produce identical counters."""
        pool = _two_gene_pool_for_random()

        result_a = compute_random_distances(
            pool, n_per_gene=3, rng=np.random.default_rng(42)
        )
        result_b = compute_random_distances(
            pool, n_per_gene=3, rng=np.random.default_rng(42)
        )

        self.assertEqual(result_a, result_b)
        # make sure the numer of events in the counter is as expected
        self.assertEqual(sum(result_a.values()), sum(result_b.values()))

    def test_total_mass_matches_formula(self) -> None:
        """Total count equals sum_g (k_g - 1) * n_per_gene."""
        pool = _two_gene_pool_for_random()
        n_per_gene = 3
        # g_a: k=4 -> 3 diffs/replicate; g_b: k=6 -> 5 diffs/replicate.
        expected_total = (4 - 1) * n_per_gene + (6 - 1) * n_per_gene

        result = compute_random_distances(
            pool, n_per_gene=n_per_gene, rng=np.random.default_rng(0)
        )

        self.assertEqual(sum(result.values()), expected_total)

    def test_all_distances_at_least_one(self) -> None:
        """Position-blocking guarantees every distance is >= 1."""
        pool = _two_gene_pool_for_random()

        result = compute_random_distances(
            pool, n_per_gene=5, rng=np.random.default_rng(123)
        )

        self.assertTrue(all(d >= 1 for d in result.keys()))

    def test_skips_genes_with_k_below_two(self) -> None:
        """Genes whose n_mutations is <2 contribute nothing."""
        mutations_records = [
            _mut_row("pool", p, "A", "T") for p in range(20)
        ]
        pool = _make_pool(
            mutations_records,
            refs={"g_solo": "X", "g_pair": "X"},
            gene_stats_records=[
                _stats_row("g_solo", n_mutations=1),
                _stats_row("g_pair", n_mutations=2),
            ],
        )
        n_per_gene = 4

        result = compute_random_distances(
            pool, n_per_gene=n_per_gene, rng=np.random.default_rng(0)
        )

        # g_solo contributes 0; g_pair contributes (2-1) * n_per_gene = 4.
        self.assertEqual(sum(result.values()), n_per_gene)


# ---------------------------------------------------------------------------
# counter_to_array
# ---------------------------------------------------------------------------


class TestCounterToArray(unittest.TestCase):
    """Tests for counter_to_array."""

    def test_expands_counts_to_repeated_values(self) -> None:
        """Counter({2: 3, 5: 1}) expands to a sorted array [2, 2, 2, 5]."""
        counter = Counter({2: 3, 5: 1})

        result = counter_to_array(counter)

        np.testing.assert_array_equal(
            np.sort(result), np.array([2, 2, 2, 5])
        )

    def test_empty_counter_yields_empty_array(self) -> None:
        """An empty counter yields an empty array."""
        result = counter_to_array(Counter())

        self.assertEqual(result.size, 0)


# ---------------------------------------------------------------------------
# plot_difference
# ---------------------------------------------------------------------------


class TestToProportions(unittest.TestCase):
    """Tests for _to_proportions."""

    def test_values_sum_to_one(self) -> None:
        """Proportions of a non-empty counter sum to 1."""
        counter = Counter({1: 10, 2: 5, 3: 5})
        result = _to_proportions(counter)
        self.assertAlmostEqual(sum(result.values()), 1.0)
        self.assertAlmostEqual(result[1], 0.5)
        self.assertAlmostEqual(result[2], 0.25)
        self.assertAlmostEqual(result[3], 0.25)

    def test_empty_counter_returns_empty_dict(self) -> None:
        """An empty counter yields an empty dict."""
        self.assertEqual(_to_proportions(Counter()), {})

    def test_proportions_match_expected_values(self) -> None:
        """Known counter produces the expected proportions."""
        counter = Counter({1: 3, 2: 1})
        result = _to_proportions(counter)
        self.assertAlmostEqual(result[1], 0.75)
        self.assertAlmostEqual(result[2], 0.25)


class TestPlotDifference(unittest.TestCase):
    """Tests for plot_difference."""

    def test_creates_output_file(self) -> None:
        """plot_difference writes a file with the expected name."""
        real = _to_proportions(Counter({1: 10, 2: 5, 3: 2}))
        random_dist = _to_proportions(Counter({1: 4, 2: 8, 3: 6}))

        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_difference(real, random_dist, "test", tmp_dir, "png")
            out_path = os.path.join(
                tmp_dir, "mutation_distances_test_difference.png"
            )
            self.assertTrue(os.path.isfile(out_path))

    def test_creates_smaller_output_file_when_max_distance_given(self) -> None:
        """When max_distance is set the _smaller suffix is used."""
        real = _to_proportions(Counter({1: 10, 2: 5, 300: 1}))
        random_dist = _to_proportions(Counter({1: 4, 2: 8, 300: 2}))

        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_difference(real, random_dist, "test", tmp_dir, "png", max_distance=200)
            out_path = os.path.join(
                tmp_dir, "mutation_distances_test_difference_smaller.png"
            )
            self.assertTrue(os.path.isfile(out_path))

    def test_empty_dicts_do_not_raise(self) -> None:
        """Empty proportion dicts produce no file and no exception."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_difference({}, {}, "test", tmp_dir, "png")
            self.assertEqual(os.listdir(tmp_dir), [])


# ---------------------------------------------------------------------------
# run_distance_analysis (end-to-end driver)
# ---------------------------------------------------------------------------


class TestRunDistanceAnalysis(unittest.TestCase):
    """Tests for run_distance_analysis."""

    def test_writes_expected_output_files(self) -> None:
        """End-to-end writes four plots and a stats file with 3 numeric lines."""
        mutations_records: List[Dict[str, object]] = []
        for position in range(0, 100, 5):
            mutations_records.append(
                _mut_row("g_a", position, "A", "T")
            )
            mutations_records.append(
                _mut_row("g_b", position + 1, "C", "G")
            )
        pool = _make_pool(
            mutations_records,
            refs={"g_a": "X", "g_b": "X"},
            gene_stats_records=[
                _stats_row("g_a", n_mutations=5),
                _stats_row("g_b", n_mutations=5),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch.object(
            mda.MutationPool, "load", return_value=pool
        ) as mock_load:
            run_distance_analysis(
                pool_path="ignored.json",
                output_dir=tmp_dir,
                name="testname",
                n_per_gene=2,
                seed=0,
                output_format="png",
            )
            expected_files = [
                "mutation_distances_testname_overlay.png",
                "mutation_distances_testname_overlay_smaller.png",
                "mutation_distances_testname_difference.png",
                "mutation_distances_testname_difference_smaller.png",
                "mutation_distances_testname_stats.txt",
            ]
            for filename in expected_files:
                self.assertTrue(
                    os.path.isfile(os.path.join(tmp_dir, filename)),
                    msg=f"Expected output file missing: {filename}",
                )

            stats_path = os.path.join(
                tmp_dir, expected_files[-1]
            )
            with open(stats_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(lines), 3)
            for line in lines:
                _, value_str = line.split("=", 1)
                value = float(value_str)
                self.assertTrue(np.isfinite(value))

        mock_load.assert_called_once_with("ignored.json")


# ---------------------------------------------------------------------------
# parse_args / main (CLI)
# ---------------------------------------------------------------------------


class TestCli(unittest.TestCase):
    """Tests for the CLI entry point."""

    def test_main_invokes_run_distance_analysis(self) -> None:
        """main() forwards parsed args to run_distance_analysis."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pool_path = os.path.join(tmp_dir, "pool.json")
            with open(pool_path, "w") as f:
                f.write("{}")
            output_dir = os.path.join(tmp_dir, "out")

            argv = [
                "mutation_distance_analysis",
                "--pool",
                pool_path,
                "--output-dir",
                output_dir,
                "--name",
                "abc",
                "--n-per-gene",
                "7",
                "--seed",
                "13",
                "--format",
                "pdf",
            ]

            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                mda, "run_distance_analysis"
            ) as mock_run:
                mda.main()

        mock_run.assert_called_once_with(
            pool_path=pool_path,
            output_dir=output_dir,
            name="abc",
            n_per_gene=7,
            seed=13,
            output_format="pdf",
        )


class TestDistancePlotAxInjection(unittest.TestCase):
    """Ax-injection behaviour of the distance plotting functions."""

    @mock.patch("matplotlib.pyplot.savefig")
    def test_plot_overlay_with_ax(self, mock_savefig):
        real = {1: 0.5, 2: 0.3, 3: 0.2}
        random_dist = {1: 0.4, 2: 0.4, 3: 0.2}
        fig, ax = plt.subplots()
        try:
            mda.plot_overlay(real, random_dist, "name", "unused_dir", ax=ax)
            self.assertGreater(len(ax.patches) + len(ax.lines), 0)
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)

    @mock.patch("matplotlib.pyplot.savefig")
    def test_plot_difference_with_ax(self, mock_savefig):
        real = {1: 0.5, 2: 0.3, 3: 0.2}
        random_dist = {1: 0.4, 2: 0.4, 3: 0.2}
        fig, ax = plt.subplots()
        try:
            mda.plot_difference(real, random_dist, "name", "unused_dir", ax=ax)
            self.assertGreater(len(ax.patches), 0)  # difference bars drawn
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
