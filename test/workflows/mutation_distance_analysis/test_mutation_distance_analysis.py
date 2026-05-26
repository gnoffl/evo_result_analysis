"""Unit tests for ``workflows.mutation_distance_analysis``."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from collections import Counter
from typing import Dict, List
from unittest import mock

import numpy as np
import pandas as pd

from workflows.mutation_distance_analysis import (
    mutation_distance_analysis as mda,
)
from workflows.mutation_distance_analysis.mutation_distance_analysis import (
    compute_random_distances,
    compute_real_distances,
    counter_to_array,
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
        self.assertEqual(len(result), 4)

    def test_empty_counter_yields_empty_array(self) -> None:
        """An empty counter yields an empty array."""
        result = counter_to_array(Counter())

        self.assertEqual(result.size, 0)


# ---------------------------------------------------------------------------
# run_distance_analysis (end-to-end driver)
# ---------------------------------------------------------------------------


class TestRunDistanceAnalysis(unittest.TestCase):
    """Tests for run_distance_analysis."""

    def test_writes_expected_three_files(self) -> None:
        """End-to-end writes two plots and a stats file with 3 numeric lines."""
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
            full_plot = os.path.join(
                tmp_dir, "mutation_distances_testname_overlay.png"
            )
            small_plot = os.path.join(
                tmp_dir,
                "mutation_distances_testname_overlay_smaller.png",
            )
            stats_path = os.path.join(
                tmp_dir, "mutation_distances_testname_stats.txt"
            )

            self.assertTrue(os.path.isfile(full_plot))
            self.assertTrue(os.path.isfile(small_plot))
            self.assertTrue(os.path.isfile(stats_path))

            with open(stats_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(lines), 3)
            for line in lines:
                _, value_str = line.split("=", 1)
                # Each line must parse to a finite float.
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


if __name__ == "__main__":
    unittest.main()
