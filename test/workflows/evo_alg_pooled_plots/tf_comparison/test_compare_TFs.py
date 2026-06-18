"""Unit tests for the cross-run TF comparison script."""

import json
import math
import os
import tempfile
import unittest
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from workflows.evo_alg_pooled_plots.tf_comparison.compare_TFs import (
    build_matrix,
    count_genes,
    load_per_gene_diffs,
    order_tfs,
    paired_tf_significance,
    plot_heatmap,
    q_to_stars,
)


def _write_run(run_dir: str, n_genes: int, summary_df: pd.DataFrame) -> None:
    """Create a run dir with a stats_*.json and a deepcis_scan peak summary."""
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

    def test_union_rows_normalized_and_max_before_min(self) -> None:
        # Arrange: two runs with partially overlapping TF sets.
        with tempfile.TemporaryDirectory() as tmp:
            max_dir = os.path.join(tmp, "max_run")
            min_dir = os.path.join(tmp, "min_run")
            os.makedirs(max_dir)
            os.makedirs(min_dir)
            _write_run(max_dir, 10, pd.DataFrame({"tf": ["WRKY", "bHLH"], "diff_calc": [10, -20]}))
            _write_run(min_dir, 10, pd.DataFrame({"tf": ["WRKY", "MYB"], "diff_calc": [-30, 5]}))
            # Pass min before max to confirm reordering.
            runs = [(min_dir, "min", "MIN"), (max_dir, "max", "MAX")]

            # Act
            matrix = build_matrix(runs)

        # Assert: union rows, max column first, NaN for absent TFs, normalized values.
        self.assertEqual(list(matrix.columns), ["MAX", "MIN"])
        self.assertEqual(set(matrix.index), {"WRKY", "bHLH", "MYB"})
        self.assertAlmostEqual(matrix.loc["WRKY", "MAX"], 1.0)
        self.assertTrue(np.isnan(matrix.loc["bHLH", "MIN"]))
        self.assertTrue(np.isnan(matrix.loc["MYB", "MAX"]))


class OrderTfsTest(unittest.TestCase):
    """Tests for the max-minus-min ordering score."""

    def test_high_in_max_low_in_min_ranks_first(self) -> None:
        # Arrange: introduced is high in max / low in min; removed is its mirror.
        matrix = pd.DataFrame(
            {
                "MAX": {"introduced": 5.0, "removed": -5.0},
                "MIN": {"introduced": -5.0, "removed": 5.0},
            }
        )
        directions = {"MAX": "max", "MIN": "min"}

        # Act
        ordered = order_tfs(matrix, directions)

        # Assert
        self.assertEqual(list(ordered.index), ["introduced", "removed"])

    def test_missing_direction_counts_as_zero(self) -> None:
        # Arrange: only_min appears only in min with -2 -> score 0-(-2)=+2,
        # above only_max whose score is +1.
        matrix = pd.DataFrame(
            {
                "MAX": {"only_max": 1.0, "only_min": np.nan},
                "MIN": {"only_max": np.nan, "only_min": -2.0},
            }
        )
        directions = {"MAX": "max", "MIN": "min"}

        # Act
        ordered = order_tfs(matrix, directions)

        # Assert
        self.assertEqual(list(ordered.index), ["only_min", "only_max"])


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
        """Build a max and a min run sharing 8 genes with a SIG and a FLAT TF."""
        max_dir = os.path.join(tmp, "max_run")
        min_dir = os.path.join(tmp, "min_run")
        max_peaks: List[Tuple[str, str, str, int]] = []
        min_peaks: List[Tuple[str, str, str, int]] = []
        for i in range(8):
            gene = f"{i}_GENE{i}_loc_{i}"
            # SIG: introduced when maximizing (diff +2), removed when minimizing (diff -1).
            max_peaks += [(gene, "SIG", "reference", 1), (gene, "SIG", "max_mutated", 3)]
            min_peaks += [(gene, "SIG", "reference", 1)]  # no max_mutated peaks -> count 0
            # FLAT: identical in both runs (diff 0 everywhere).
            max_peaks += [(gene, "FLAT", "reference", 2), (gene, "FLAT", "max_mutated", 2)]
            min_peaks += [(gene, "FLAT", "reference", 2), (gene, "FLAT", "max_mutated", 2)]
        _write_annotated_peaks(max_dir, max_peaks)
        _write_annotated_peaks(min_dir, min_peaks)
        return max_dir, min_dir

    def test_significant_and_flat_tfs(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            max_dir, min_dir = self._make_runs(tmp)

            # Act
            result = paired_tf_significance(max_dir, min_dir)

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


class QToStarsTest(unittest.TestCase):
    """Tests for the q-value to star mapping."""

    def test_thresholds_and_nan(self) -> None:
        # Arrange / Act / Assert
        self.assertEqual(q_to_stars(0.0005), "***")
        self.assertEqual(q_to_stars(0.005), "** ")
        self.assertEqual(q_to_stars(0.02), "*  ")
        self.assertEqual(q_to_stars(0.2), "")
        self.assertEqual(q_to_stars(float("nan")), "")


class PlotHeatmapTest(unittest.TestCase):
    """Smoke test for figure creation."""

    def test_returns_figure(self) -> None:
        # Arrange
        matrix = pd.DataFrame(
            {"MAX": {"WRKY": 1.0, "bHLH": -2.0}, "MIN": {"WRKY": -1.0, "bHLH": 2.0}}
        )

        # Act
        fig = plot_heatmap(matrix, n_max_runs=1, annotate=True)

        # Assert
        self.assertIsInstance(fig, Figure)

    def test_row_stars_annotate_tf_labels(self) -> None:
        # Arrange
        matrix = pd.DataFrame(
            {"MAX": {"WRKY": 1.0, "bHLH": -2.0}, "MIN": {"WRKY": -1.0, "bHLH": 2.0}}
        )

        # Act
        fig = plot_heatmap(matrix, n_max_runs=1, annotate=True, row_stars={"WRKY": "** "})

        # Assert: the WRKY row label carries its stars, bHLH stays plain.
        labels = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
        self.assertIn("WRKY ** ", labels)
        self.assertIn("bHLH    ", labels)


if __name__ == "__main__":
    unittest.main()
