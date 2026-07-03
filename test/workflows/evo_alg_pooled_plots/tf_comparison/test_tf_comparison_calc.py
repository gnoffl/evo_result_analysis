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
    _pair_contrast_matrix,
    build_matrix,
    count_genes,
    interaction_model_tf_significance,
    load_per_gene_diffs,
    load_per_gene_tf_counts,
    order_tfs_by_group_contrast,
    order_tfs_by_mean,
    paired_tf_significance,
    per_gene_tf_binding_summary,
    pooled_model_tf_significance,
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

    def test_n_core_fields_controls_replicate_grouping(self) -> None:
        # Arrange: two random-start sequences whose first two underscore fields are
        # identical ("random_sequence") but whose first three fields differ.
        with tempfile.TemporaryDirectory() as tmp:
            _write_annotated_peaks(
                tmp,
                [
                    ("random_sequence_000_ts_a", "WRKY", "reference", 1),
                    ("random_sequence_000_ts_a", "WRKY", "max_mutated", 3),
                    ("random_sequence_001_ts_b", "WRKY", "reference", 2),
                    ("random_sequence_001_ts_b", "WRKY", "max_mutated", 2),
                ],
            )

            # Act
            collapsed = load_per_gene_diffs(tmp)  # default 2 fields
            distinct = load_per_gene_diffs(tmp, n_core_fields=3)

        # Assert: 2 fields collapse both sequences into one replicate; 3 fields keep
        # them separate (the random-start naming convention).
        self.assertEqual(
            set(collapsed.index.get_level_values("core_gene")), {"random_sequence"}
        )
        self.assertEqual(
            set(distinct.index.get_level_values("core_gene")),
            {"random_sequence_000", "random_sequence_001"},
        )


class LoadPerGeneTfCountsTest(unittest.TestCase):
    """Tests for the shared reference/max_mutated peak-count loader."""

    def test_reference_mutated_and_diff_columns(self) -> None:
        # Arrange: WRKY has 2 reference and 3 max_mutated peaks (diff +1); MYB has
        # 4 reference peaks and no max_mutated (diff -4); a "difference" row is noise.
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
            counts = load_per_gene_tf_counts(tmp)

        # Assert
        self.assertEqual(list(counts.columns), ["reference", "max_mutated", "diff"])
        self.assertEqual(counts.loc[("1_ATX", "WRKY"), "reference"], 2)
        self.assertEqual(counts.loc[("1_ATX", "WRKY"), "max_mutated"], 3)
        self.assertEqual(counts.loc[("1_ATX", "WRKY"), "diff"], 1)
        # MYB has no max_mutated peaks -> count 0, diff -4.
        self.assertEqual(counts.loc[("1_ATX", "MYB"), "max_mutated"], 0)
        self.assertEqual(counts.loc[("1_ATX", "MYB"), "diff"], -4)

    def test_diff_matches_load_per_gene_diffs(self) -> None:
        # Arrange: the refactored load_per_gene_diffs must return exactly this loader's
        # diff column, so their values agree cell for cell.
        with tempfile.TemporaryDirectory() as tmp:
            _write_annotated_peaks(
                tmp,
                [
                    ("1_ATX_a_111", "WRKY", "reference", 2),
                    ("1_ATX_a_111", "WRKY", "max_mutated", 3),
                    ("2_ATY_b_222", "MYB", "reference", 4),
                ],
            )

            # Act
            counts = load_per_gene_tf_counts(tmp)
            diffs = load_per_gene_diffs(tmp)

        # Assert
        pd.testing.assert_series_equal(
            counts["diff"], diffs["diff"], check_names=False
        )


class PerGeneTfBindingSummaryTest(unittest.TestCase):
    """Tests for per-TF mean/std of reference, max_mutated, and diff over genes."""

    def test_means_divide_by_all_genes_with_zero_fill(self) -> None:
        # Arrange: 2 genes. WRKY appears in both (max_mutated 2 and 4 -> mean 3);
        # RARE appears only in gene 1 (max_mutated 5), so gene 2 contributes 0 and
        # the mean over both genes is 2.5.
        with tempfile.TemporaryDirectory() as tmp:
            _write_annotated_peaks(
                tmp,
                [
                    ("1_G1_a_0", "WRKY", "reference", 1),
                    ("1_G1_a_0", "WRKY", "max_mutated", 2),
                    ("2_G2_b_0", "WRKY", "reference", 1),
                    ("2_G2_b_0", "WRKY", "max_mutated", 4),
                    ("1_G1_a_0", "RARE", "reference", 0),
                    ("1_G1_a_0", "RARE", "max_mutated", 5),
                ],
            )

            # Act
            summary = per_gene_tf_binding_summary(tmp).set_index("tf")

        # Assert
        self.assertTrue((summary["n_genes"] == 2).all())
        self.assertAlmostEqual(summary.loc["WRKY", "mean_max_mutated"], 3.0)
        # RARE: (5 + 0) / 2 genes = 2.5 thanks to zero-fill over the absent gene.
        self.assertAlmostEqual(summary.loc["RARE", "mean_max_mutated"], 2.5)
        self.assertAlmostEqual(summary.loc["RARE", "mean_reference"], 0.0)
        self.assertAlmostEqual(summary.loc["RARE", "mean_diff"], 2.5)

    def test_std_is_sample_std_and_zero_for_constant_tf(self) -> None:
        # Arrange: CONST has max_mutated 3 in both genes (std 0); VARIED has 2 and 4
        # (sample std of [2, 4] is sqrt(2) ~= 1.4142).
        with tempfile.TemporaryDirectory() as tmp:
            _write_annotated_peaks(
                tmp,
                [
                    ("1_G1_a_0", "CONST", "max_mutated", 3),
                    ("2_G2_b_0", "CONST", "max_mutated", 3),
                    ("1_G1_a_0", "VARIED", "max_mutated", 2),
                    ("2_G2_b_0", "VARIED", "max_mutated", 4),
                ],
            )

            # Act
            summary = per_gene_tf_binding_summary(tmp).set_index("tf")

        # Assert
        self.assertAlmostEqual(summary.loc["CONST", "std_max_mutated"], 0.0)
        self.assertAlmostEqual(summary.loc["VARIED", "std_max_mutated"], math.sqrt(2.0))

    def test_columns_and_sorted_by_mean_max_mutated_descending(self) -> None:
        # Arrange: HIGH binds more than LOW in the optimized sequence.
        with tempfile.TemporaryDirectory() as tmp:
            _write_annotated_peaks(
                tmp,
                [
                    ("1_G1_a_0", "HIGH", "max_mutated", 5),
                    ("1_G1_a_0", "LOW", "max_mutated", 1),
                ],
            )

            # Act
            summary = per_gene_tf_binding_summary(tmp)

        # Assert
        self.assertEqual(
            list(summary.columns),
            [
                "tf", "n_genes", "mean_reference", "std_reference",
                "mean_max_mutated", "std_max_mutated", "mean_diff", "std_diff",
            ],
        )
        self.assertEqual(list(summary["tf"]), ["HIGH", "LOW"])

    def test_single_gene_gives_nan_std(self) -> None:
        # Arrange: one gene -> sample std (ddof=1) is undefined.
        with tempfile.TemporaryDirectory() as tmp:
            _write_annotated_peaks(
                tmp, [("1_G1_a_0", "WRKY", "max_mutated", 3)]
            )

            # Act
            summary = per_gene_tf_binding_summary(tmp).set_index("tf")

        # Assert
        self.assertEqual(summary.loc["WRKY", "n_genes"], 1)
        self.assertTrue(math.isnan(summary.loc["WRKY", "std_max_mutated"]))


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

    def test_custom_labels_rename_per_run_columns(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            run_a_dir, run_b_dir = self._make_runs(tmp)

            # Act
            result = paired_tf_significance(
                run_a_dir, run_b_dir, label_a="ara_model", label_b="zea_model"
            )

        # Assert: the per-run columns carry the meaningful suffixes; the neutral
        # a/b names are gone; the contrast columns are unchanged.
        for col in (
            "median_diff_ara_model", "n_nonzero_ara_model", "p_ara_model", "q_ara_model",
            "median_diff_zea_model", "n_nonzero_zea_model", "p_zea_model", "q_zea_model",
            "median_D", "q_contrast",
        ):
            self.assertIn(col, result.columns)
        for col in ("p_a", "q_a", "p_b", "q_b"):
            self.assertNotIn(col, result.columns)


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

    def test_n_core_fields_recovers_per_sequence_replicates(self) -> None:
        # Arrange: 8 random-start sequences (first two fields all "random_sequence"),
        # SIG gains a peak in each (diff +2 per sequence).
        with tempfile.TemporaryDirectory() as tmp:
            peaks: List[Tuple[str, str, str, int]] = []
            for i in range(8):
                sequence = f"random_sequence_{i:03d}_ts_{i}"
                peaks += [(sequence, "SIG", "reference", 1), (sequence, "SIG", "max_mutated", 3)]
            _write_annotated_peaks(tmp, peaks)

            # Act
            collapsed = single_run_tf_significance(tmp)  # default 2 fields
            recovered = single_run_tf_significance(tmp, n_core_fields=3)

        # Assert: the default field count collapses all sequences into one replicate
        # (n_genes 1 -> Wilcoxon on a single value is never significant); 3 fields
        # recover the 8 per-sequence replicates and detect SIG.
        self.assertEqual(collapsed.set_index("tf").loc["SIG", "n_genes"], 1)
        recovered_sig = recovered.set_index("tf").loc["SIG"]
        self.assertEqual(recovered_sig["n_genes"], 8)
        self.assertLess(recovered_sig["p_intra"], 0.05)


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


_PAIR_REF_COUNT = 3


def _write_model_pair(
    model_a_dir: str,
    model_b_dir: str,
    gene_indices: List[int],
    per_tf_diff: "dict[str, List[int]]",
) -> None:
    """Write a shared-gene run pair with a known per-gene contrast D per TF.

    For each gene and TF, ``D = diff_a - diff_b`` is split into non-negative
    per-run diffs (``diff_a = max(D, 0)``, ``diff_b = max(-D, 0)``) and realized
    as reference/max_mutated peak counts around a fixed reference count, so the
    two runs share ``gene_indices`` and the model-A vs model-B contrast equals the
    requested D.

    Args:
        model_a_dir: Directory for the model-A run.
        model_b_dir: Directory for the model-B run.
        gene_indices: Gene ids (as integers) shared by both runs.
        per_tf_diff: ``{tf: [D per gene]}`` (length must match ``gene_indices``).
    """
    a_peaks: List[Tuple[str, str, str, int]] = []
    b_peaks: List[Tuple[str, str, str, int]] = []
    for position, gene_index in enumerate(gene_indices):
        gene = f"{gene_index}_GENE{gene_index}_loc_{gene_index}"
        for tf, contrast_values in per_tf_diff.items():
            contrast = contrast_values[position]
            diff_a = max(contrast, 0)
            diff_b = max(-contrast, 0)
            a_peaks += [
                (gene, tf, "reference", _PAIR_REF_COUNT),
                (gene, tf, "max_mutated", _PAIR_REF_COUNT + diff_a),
            ]
            b_peaks += [
                (gene, tf, "reference", _PAIR_REF_COUNT),
                (gene, tf, "max_mutated", _PAIR_REF_COUNT + diff_b),
            ]
    _write_annotated_peaks(model_a_dir, a_peaks)
    _write_annotated_peaks(model_b_dir, b_peaks)


class PairContrastMatrixTest(unittest.TestCase):
    """Tests for the shared _pair_contrast_matrix helper."""

    def test_shared_genes_tf_union_and_fill_zero(self) -> None:
        # Arrange: run A has genes 0,1,2 with TF X; run B has genes 1,2,3 with X and Y.
        with tempfile.TemporaryDirectory() as tmp:
            run_a_dir = os.path.join(tmp, "a")
            run_b_dir = os.path.join(tmp, "b")
            a_peaks = []
            for i in (0, 1, 2):
                gene = f"{i}_G{i}_loc_{i}"
                a_peaks += [(gene, "X", "reference", 1), (gene, "X", "max_mutated", 3)]
            b_peaks = []
            for i in (1, 2, 3):
                gene = f"{i}_G{i}_loc_{i}"
                b_peaks += [(gene, "X", "reference", 1), (gene, "X", "max_mutated", 2)]
                b_peaks += [(gene, "Y", "reference", 2), (gene, "Y", "max_mutated", 4)]
            _write_annotated_peaks(run_a_dir, a_peaks)
            _write_annotated_peaks(run_b_dir, b_peaks)

            # Act
            a_mat, b_mat, contrast_mat, shared_genes, all_tfs = _pair_contrast_matrix(
                run_a_dir, run_b_dir
            )

        # Assert: only genes 1,2 are shared; TFs are the union; Y absent in A -> 0.
        self.assertEqual(shared_genes, ["1_G1", "2_G2"])
        self.assertEqual(all_tfs, ["X", "Y"])
        self.assertTrue((a_mat["Y"] == 0.0).all())
        # X: diff_a = 3-1 = 2, diff_b = 2-1 = 1 -> contrast 1; Y contrast = 0 - 2 = -2.
        self.assertTrue((contrast_mat["X"] == 1.0).all())
        self.assertTrue((contrast_mat["Y"] == -2.0).all())


class PooledModelTfSignificanceTest(unittest.TestCase):
    """Tests for the pooled per-TF model main effect across gene sources."""

    def _make_pairs(self, tmp: str) -> List[Tuple[str, str]]:
        """Two disjoint-gene pairs with known per-gene contrast D per TF.

        CONSISTENT has D=+2 in every gene of both sources; FLIP has D=+2 for the
        first source and D=-2 for the second (sign flip); ONLYARA has D=+3 in the
        first source and is absent from the second (fill-0 case).
        """
        ara_a = os.path.join(tmp, "ara_a")
        ara_b = os.path.join(tmp, "ara_b")
        zea_a = os.path.join(tmp, "zea_a")
        zea_b = os.path.join(tmp, "zea_b")
        _write_model_pair(
            ara_a,
            ara_b,
            list(range(8)),
            {"CONSISTENT": [2] * 8, "FLIP": [2] * 8, "ONLYARA": [3] * 8},
        )
        _write_model_pair(
            zea_a,
            zea_b,
            list(range(8, 16)),
            {"CONSISTENT": [2] * 8, "FLIP": [-2] * 8},
        )
        return [(ara_a, ara_b), (zea_a, zea_b)]

    def test_pooled_statistics_columns_and_sort(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            pairs = self._make_pairs(tmp)

            # Act
            result = pooled_model_tf_significance(pairs)

        # Assert: expected columns.
        self.assertEqual(
            list(result.columns),
            ["tf", "n_genes", "median_D", "n_nonzero_D", "p_model", "q_model"],
        )
        indexed = result.set_index("tf")
        # Pooled gene count is the total across both sources.
        self.assertTrue((result["n_genes"] == 16).all())
        # CONSISTENT: D=+2 everywhere -> median 2, all 16 nonzero, significant.
        self.assertEqual(indexed.loc["CONSISTENT", "median_D"], 2)
        self.assertEqual(indexed.loc["CONSISTENT", "n_nonzero_D"], 16)
        self.assertLess(indexed.loc["CONSISTENT", "p_model"], 0.05)
        # ONLYARA absent from the second source -> 8 nonzero, median 1.5 (fill-0).
        self.assertEqual(indexed.loc["ONLYARA", "n_nonzero_D"], 8)
        self.assertAlmostEqual(indexed.loc["ONLYARA", "median_D"], 1.5)
        # Sorted by q_model ascending -> the strongest effect (CONSISTENT) is first.
        self.assertEqual(result.iloc[0]["tf"], "CONSISTENT")

    def test_sign_flip_effect_attenuates(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            pairs = self._make_pairs(tmp)

            # Act
            result = pooled_model_tf_significance(pairs).set_index("tf")

        # Assert: the sign-flip TF washes out (median 0, far less significant than
        # the consistent TF) -> documents the pooled main-effect caveat.
        self.assertEqual(result.loc["FLIP", "median_D"], 0)
        self.assertGreater(result.loc["FLIP", "p_model"], result.loc["CONSISTENT", "p_model"])

    def test_q_values_monotone_in_p(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            pairs = self._make_pairs(tmp)

            # Act
            result = pooled_model_tf_significance(pairs)

        # Assert: BH q-values are non-decreasing when ordered by p-value.
        finite = result.dropna(subset=["p_model"]).sort_values("p_model")
        self.assertTrue((finite["q_model"].diff().dropna() >= -1e-12).all())

    def test_empty_pairs_raises(self) -> None:
        # Act / Assert
        with self.assertRaises(ValueError):
            pooled_model_tf_significance([])


class InteractionModelTfSignificanceTest(unittest.TestCase):
    """Tests for the unpaired per-TF interaction (model effect differs by group)."""

    def _make_groups(self, tmp: str) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """Two disjoint-gene groups with a differing, a matched, and a degenerate TF.

        DIFFERS has D=+3 in group A and D=-3 in group B (clearly different); MATCHED
        has the same alternating D distribution in both groups; DEGENERATE has D=0
        everywhere (Mann-Whitney degenerate).
        """
        group_a_model_a = os.path.join(tmp, "ga_ma")
        group_a_model_b = os.path.join(tmp, "ga_mb")
        group_b_model_a = os.path.join(tmp, "gb_ma")
        group_b_model_b = os.path.join(tmp, "gb_mb")
        matched = [0, 2, 0, 2, 0, 2, 0, 2]
        _write_model_pair(
            group_a_model_a,
            group_a_model_b,
            list(range(8)),
            {"DIFFERS": [3] * 8, "MATCHED": matched, "DEGENERATE": [0] * 8},
        )
        _write_model_pair(
            group_b_model_a,
            group_b_model_b,
            list(range(8, 16)),
            {"DIFFERS": [-3] * 8, "MATCHED": matched, "DEGENERATE": [0] * 8},
        )
        return (group_a_model_a, group_a_model_b), (group_b_model_a, group_b_model_b)

    def test_differing_tf_is_most_significant(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            group_a_pair, group_b_pair = self._make_groups(tmp)

            # Act
            result = interaction_model_tf_significance(group_a_pair, group_b_pair)

        # Assert: expected columns.
        self.assertEqual(
            list(result.columns),
            [
                "tf", "n_a", "n_b", "median_D_a", "median_D_b",
                "u_stat", "p_interaction", "q_interaction",
            ],
        )
        indexed = result.set_index("tf")
        # DIFFERS separates the two groups -> smallest p; MATCHED does not.
        self.assertLess(
            indexed.loc["DIFFERS", "p_interaction"], indexed.loc["MATCHED", "p_interaction"]
        )
        self.assertEqual(indexed.loc["DIFFERS", "n_a"], 8)
        self.assertEqual(indexed.loc["DIFFERS", "n_b"], 8)
        self.assertAlmostEqual(indexed.loc["DIFFERS", "median_D_a"], 3.0)
        self.assertAlmostEqual(indexed.loc["DIFFERS", "median_D_b"], -3.0)
        # Sorted by q_interaction ascending -> the differing TF comes first.
        self.assertEqual(result.iloc[0]["tf"], "DIFFERS")

    def test_custom_labels_rename_group_columns(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            group_a_pair, group_b_pair = self._make_groups(tmp)

            # Act
            result = interaction_model_tf_significance(
                group_a_pair, group_b_pair, label_a="ara_genes", label_b="zea_genes"
            )

        # Assert: the group columns carry the meaningful suffixes and the neutral
        # a/b names are gone; the shared stat columns are unchanged.
        for col in (
            "n_ara_genes", "n_zea_genes", "median_D_ara_genes", "median_D_zea_genes",
            "u_stat", "p_interaction", "q_interaction",
        ):
            self.assertIn(col, result.columns)
        for col in ("n_a", "n_b", "median_D_a", "median_D_b"):
            self.assertNotIn(col, result.columns)

    def test_degenerate_tf_is_not_significant(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            group_a_pair, group_b_pair = self._make_groups(tmp)

            # Act
            result = interaction_model_tf_significance(group_a_pair, group_b_pair).set_index("tf")

        # Assert: an all-equal (D=0 in both groups) TF cannot be significant.
        # scipy 1.10 returns p=1.0 here rather than raising; the function's
        # ValueError guard still protects the empty-input path.
        degenerate_p = result.loc["DEGENERATE", "p_interaction"]
        self.assertTrue(math.isnan(degenerate_p) or degenerate_p >= 0.99)


if __name__ == "__main__":
    unittest.main()
