"""Unit tests for the TF-comparison calculation toolbox."""

import json
import math
import os
import tempfile
import unittest
from typing import List, Tuple

import numpy as np
import pandas as pd

from workflows.evo_alg_pooled_plots.tf_comparison.tf_comparison_calc import (
    build_matrix,
    count_genes,
    load_per_gene_diffs,
    order_tfs_by_group_contrast,
    order_tfs_by_mean,
    paired_tf_significance,
    single_run_tf_significance,
    top_bottom_tfs,
)


def _write_run(run_dir: str, n_genes: int, summary_df: pd.DataFrame) -> None:
    """Create a run dir with a stats_*.json and a deepcis_scan peak summary."""
    os.makedirs(run_dir, exist_ok=True)
    stats = {f"gene_{i}": {"final_fitness": 0.5} for i in range(n_genes)}
    with open(os.path.join(run_dir, "stats_run.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle)
    scan_dir = os.path.join(run_dir, "deepcis_scan")
    os.makedirs(scan_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(scan_dir, "run_peak_summary.csv"), index=False)


def _write_annotated_peaks(run_dir: str, peaks: List[Tuple[str, str, str, int]]) -> None:
    """Write a synthetic ``*_annotated_peaks_*.csv`` from (gene, tf, signal, count) rows.

    Each tuple is expanded to ``count`` peak rows (one annotated peak per row).
    """
    scan_dir = os.path.join(run_dir, "deepcis_scan")
    os.makedirs(scan_dir, exist_ok=True)
    records = [
        {"gene": gene, "tf": tf, "signal_type": signal, "peak_area": 1.0}
        for gene, tf, signal, count in peaks
        for _ in range(count)
    ]
    frame = pd.DataFrame(records, columns=["gene", "tf", "signal_type", "peak_area"])
    frame.to_csv(os.path.join(scan_dir, "run_annotated_peaks_x.csv"), index=False)


class CountGenesTest(unittest.TestCase):
    """Tests for count_genes."""

    def test_returns_stats_dict_length(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(tmp, 7, pd.DataFrame({"tf": ["WRKY"], "diff_calc": [1]}))

            # Act
            result = count_genes(tmp)

        # Assert
        self.assertEqual(result, 7)


class BuildMatrixTest(unittest.TestCase):
    """Tests for matrix assembly across runs."""

    def test_union_rows_normalized_and_columns_in_given_order(self) -> None:
        # Arrange: two runs with partially overlapping TF sets.
        with tempfile.TemporaryDirectory() as tmp:
            first_dir = os.path.join(tmp, "first_run")
            second_dir = os.path.join(tmp, "second_run")
            _write_run(first_dir, 10, pd.DataFrame({"tf": ["WRKY", "MYB"], "diff_calc": [-30, 5]}))
            _write_run(second_dir, 10, pd.DataFrame({"tf": ["WRKY", "bHLH"], "diff_calc": [10, -20]}))
            # Columns must follow the given order, not be reshuffled.
            runs = [(first_dir, "FIRST"), (second_dir, "SECOND")]

            # Act
            matrix = build_matrix(runs)

        # Assert: column order preserved, union rows, NaN for absent TFs, normalized values.
        self.assertEqual(list(matrix.columns), ["FIRST", "SECOND"])
        self.assertEqual(set(matrix.index), {"WRKY", "bHLH", "MYB"})
        self.assertAlmostEqual(matrix.loc["WRKY", "SECOND"], 1.0)
        self.assertTrue(np.isnan(matrix.loc["bHLH", "FIRST"]))
        self.assertTrue(np.isnan(matrix.loc["MYB", "SECOND"]))


class OrderTfsByGroupContrastTest(unittest.TestCase):
    """Tests for the left-minus-right group contrast ordering."""

    def test_high_in_left_low_in_right_ranks_first(self) -> None:
        # Arrange: introduced is high in left / low in right; removed is its mirror.
        matrix = pd.DataFrame(
            {
                "MIN": {"introduced": 5.0, "removed": -5.0},
                "MAX": {"introduced": -5.0, "removed": 5.0},
            }
        )

        # Act
        ordered = order_tfs_by_group_contrast(matrix, ["MAX"], ["MIN"])

        # Assert
        self.assertEqual(list(ordered.index), ["removed", "introduced"])

    def test_missing_group_value_counts_as_zero(self) -> None:
        # Arrange: only_min appears only in the right group with -2 -> score 0-(-2)=+2,
        # above only_max whose score is +1.
        matrix = pd.DataFrame(
            {
                "MIN": {"only_max": 1.0, "only_min": np.nan},
                "MAX": {"only_max": np.nan, "only_min": -2.0},
            }
        )

        # Act
        ordered = order_tfs_by_group_contrast(matrix, ["MAX"], ["MIN"])

        # Assert
        self.assertEqual(list(ordered.index), ["only_max", "only_min"])


class OrderTfsByMeanTest(unittest.TestCase):
    """Tests for the flat overall-mean ordering."""

    def test_rows_sorted_by_descending_row_mean(self) -> None:
        # Arrange: high mean on top, low mean on bottom; NaN ignored in the mean.
        matrix = pd.DataFrame(
            {
                "A": {"low": -3.0, "mid": 1.0, "top": 4.0},
                "B": {"low": -6.0, "mid": np.nan, "top": 4.0},
            }
        )

        # Act
        ordered = order_tfs_by_mean(matrix)

        # Assert: means are 5.0, 1.0, -4.0.
        self.assertEqual(list(ordered.index), ["top", "mid", "low"])


class LoadPerGeneDiffsTest(unittest.TestCase):
    """Tests for per-gene, per-TF diff loading from annotated peaks."""

    def test_counts_diff_per_core_gene_and_ignores_difference_signal(self) -> None:
        # Arrange: one gene, WRKY gains a peak (mutated 3 - reference 2 = +1);
        # a "difference" row must be ignored; the gene id collapses to "1_ATX".
        with tempfile.TemporaryDirectory() as tmp:
            _write_annotated_peaks(
                tmp,
                [
                    ("1_ATX_a_111", "WRKY", "reference", 2),
                    ("1_ATX_a_111", "WRKY", "max_mutated", 3),
                    ("1_ATX_a_111", "WRKY", "difference", 5),
                    ("1_ATX_a_111", "MYB", "reference", 4),
                ],
            )

            # Act
            diffs = load_per_gene_diffs(tmp)

        # Assert
        self.assertEqual(diffs.loc[("1_ATX", "WRKY"), "diff"], 1)
        self.assertEqual(diffs.loc[("1_ATX", "MYB"), "diff"], -4)

    def test_raises_without_exactly_one_annotated_peaks_file(self) -> None:
        # Arrange: empty run dir (no deepcis_scan).
        with tempfile.TemporaryDirectory() as tmp:
            # Act / Assert
            with self.assertRaises(FileNotFoundError):
                load_per_gene_diffs(tmp)


class PairedTfSignificanceTest(unittest.TestCase):
    """Tests for the paired Wilcoxon + BH significance table."""

    def _make_runs(self, tmp: str) -> Tuple[str, str]:
        """Build an A and a B run sharing 8 genes with a SIG and a FLAT TF."""
        run_a_dir = os.path.join(tmp, "run_a")
        run_b_dir = os.path.join(tmp, "run_b")
        a_peaks: List[Tuple[str, str, str, int]] = []
        b_peaks: List[Tuple[str, str, str, int]] = []
        for i in range(8):
            gene = f"{i}_GENE{i}_loc_{i}"
            # SIG: introduced in run A (diff +2), removed in run B (diff -1).
            a_peaks += [(gene, "SIG", "reference", 1), (gene, "SIG", "max_mutated", 3)]
            b_peaks += [(gene, "SIG", "reference", 1)]  # no max_mutated peaks -> count 0
            # FLAT: identical in both runs (diff 0 everywhere).
            a_peaks += [(gene, "FLAT", "reference", 2), (gene, "FLAT", "max_mutated", 2)]
            b_peaks += [(gene, "FLAT", "reference", 2), (gene, "FLAT", "max_mutated", 2)]
        _write_annotated_peaks(run_a_dir, a_peaks)
        _write_annotated_peaks(run_b_dir, b_peaks)
        return run_a_dir, run_b_dir

    def test_significant_and_flat_tfs(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            run_a_dir, run_b_dir = self._make_runs(tmp)

            # Act
            result = paired_tf_significance(run_a_dir, run_b_dir)

        # Assert: SIG moves consistently (D = 2 - (-1) = 3) -> significant.
        sig = result.set_index("tf").loc["SIG"]
        self.assertEqual(sig["n_genes"], 8)
        self.assertEqual(sig["median_D"], 3)
        self.assertEqual(sig["n_nonzero_D"], 8)
        self.assertLess(sig["p_contrast"], 0.05)
        self.assertLessEqual(sig["q_contrast"], 1.0)
        self.assertGreaterEqual(sig["q_contrast"], 0.0)

        # FLAT never changes -> all-zero differences -> undefined test (NaN).
        flat = result.set_index("tf").loc["FLAT"]
        self.assertEqual(flat["median_D"], 0)
        self.assertTrue(math.isnan(flat["p_contrast"]))
        self.assertTrue(math.isnan(flat["q_contrast"]))

        # Sorted by q_contrast ascending -> the significant TF comes first.
        self.assertEqual(result.iloc[0]["tf"], "SIG")

    def test_neutral_per_run_columns_present(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            run_a_dir, run_b_dir = self._make_runs(tmp)

            # Act
            result = paired_tf_significance(run_a_dir, run_b_dir)

        # Assert: per-run columns use neutral a/b names, contrast columns keep names.
        for col in (
            "n_genes", "median_D", "n_nonzero_D", "p_contrast", "q_contrast",
            "median_diff_a", "n_nonzero_a", "p_a", "q_a",
            "median_diff_b", "n_nonzero_b", "p_b", "q_b",
        ):
            self.assertIn(col, result.columns)


class SingleRunTfSignificanceTest(unittest.TestCase):
    """Tests for single_run_tf_significance."""

    def test_significant_tf_detected_and_flat_tf_is_nan(self) -> None:
        # Arrange: SIG gains a peak in every gene (diff +2 uniformly); FLAT never changes.
        with tempfile.TemporaryDirectory() as tmp:
            peaks: List[Tuple[str, str, str, int]] = []
            for i in range(8):
                gene = f"{i}_GENE{i}_loc_{i}"
                peaks += [(gene, "SIG", "reference", 1), (gene, "SIG", "max_mutated", 3)]
                peaks += [(gene, "FLAT", "reference", 2), (gene, "FLAT", "max_mutated", 2)]
            _write_annotated_peaks(tmp, peaks)

            # Act
            result = single_run_tf_significance(tmp)

        # Assert
        indexed = result.set_index("tf")
        self.assertLess(indexed.loc["SIG", "p_intra"], 0.05)
        self.assertLessEqual(indexed.loc["SIG", "q_intra"], 1.0)
        self.assertTrue(math.isnan(indexed.loc["FLAT", "p_intra"]))
        self.assertTrue(math.isnan(indexed.loc["FLAT", "q_intra"]))
        # Sorted ascending: significant TF should come first.
        self.assertEqual(result.iloc[0]["tf"], "SIG")

    def test_columns_present(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            _write_annotated_peaks(
                tmp, [("1_G_a_0", "WRKY", "reference", 2), ("1_G_a_0", "WRKY", "max_mutated", 3)]
            )

            # Act
            result = single_run_tf_significance(tmp)

        # Assert
        for col in ("tf", "n_genes", "median_diff", "n_nonzero", "p_intra", "q_intra"):
            self.assertIn(col, result.columns)


class TopBottomTfsTest(unittest.TestCase):
    """Tests for top_bottom_tfs row slicing of an ordered matrix."""

    def _make_ordered_matrix(self) -> pd.DataFrame:
        """Return a 6-TF ordered matrix (rows already sorted best → worst)."""
        return pd.DataFrame(
            {"MAX": [5.0, 4.0, 3.0, -3.0, -4.0, -5.0]},
            index=["tf1", "tf2", "tf3", "tf4", "tf5", "tf6"],
        )

    def test_top_and_bottom_n_rows_are_kept(self) -> None:
        # Arrange
        matrix = self._make_ordered_matrix()

        # Act
        sliced = top_bottom_tfs(matrix, 2)

        # Assert: first 2 and last 2 rows are kept, middle rows are dropped.
        self.assertEqual(list(sliced.index), ["tf1", "tf2", "tf5", "tf6"])

    def test_values_are_unchanged_after_slicing(self) -> None:
        # Arrange
        matrix = self._make_ordered_matrix()

        # Act
        sliced = top_bottom_tfs(matrix, 2)

        # Assert
        self.assertAlmostEqual(sliced.loc["tf1", "MAX"], 5.0)
        self.assertAlmostEqual(sliced.loc["tf6", "MAX"], -5.0)

    def test_no_duplicate_rows_when_n_overlaps(self) -> None:
        # Arrange: n=4 means top 4 and bottom 4 of a 6-row matrix overlap in tf3/tf4.
        matrix = self._make_ordered_matrix()

        # Act
        sliced = top_bottom_tfs(matrix, 4)

        # Assert: all 6 TFs present with no duplicates.
        self.assertEqual(len(sliced), 6)
        self.assertEqual(len(set(sliced.index)), 6)

    def test_n_equals_total_rows_returns_all(self) -> None:
        # Arrange
        matrix = self._make_ordered_matrix()

        # Act
        sliced = top_bottom_tfs(matrix, len(matrix))

        # Assert
        self.assertEqual(len(sliced), len(matrix))

    def test_order_is_preserved_top_before_bottom(self) -> None:
        # Arrange
        matrix = self._make_ordered_matrix()

        # Act
        sliced = top_bottom_tfs(matrix, 2)

        # Assert: top rows appear before bottom rows, each group retains its order.
        self.assertEqual(list(sliced.index), ["tf1", "tf2", "tf5", "tf6"])


if __name__ == "__main__":
    unittest.main()
