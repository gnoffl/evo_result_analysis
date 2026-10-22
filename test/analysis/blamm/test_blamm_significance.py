"""Unit tests for analysis.blamm.blamm_significance."""

import math
import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from analysis.blamm.blamm_significance import (
    PER_GENE_COUNTS_FILE,
    SELECTED_ENTRIES_FILE,
    SIGNIFICANCE_COLUMNS,
    SIGNIFICANCE_FILE,
    analyse_output_folder,
    build_diff_matrix,
    motif_significance,
    parse_arguments,
    summarise_significance,
)

GENES = [f"gene_{index}" for index in range(8)]


def make_per_gene_counts(diffs_per_motif: dict) -> pd.DataFrame:
    """Build a per-gene count table from a ``{motif_id: {gene: diff}}`` mapping.

    Reference counts are fixed at 1 so ``count_mutated`` follows the requested
    difference; only ``gene``, ``motif_id`` and ``diff`` matter to the tests.

    Args:
        diffs_per_motif: Difference per motif and gene. Genes absent from a
            motif's mapping are left out of the table entirely, mimicking the
            "no occurrence in either sequence" case.

    Returns:
        DataFrame in the schema of
        :func:`analysis.blamm.blamm_tfbs_counts.count_motif_hits`.
    """
    rows = []
    for motif_id, per_gene in diffs_per_motif.items():
        for gene, diff in per_gene.items():
            rows.append(
                {
                    "gene": gene,
                    "motif_id": motif_id,
                    "motif_name": f"name_{motif_id}",
                    "source_db": "test_db.meme",
                    "count_reference": 1,
                    "count_mutated": 1 + diff,
                    "diff": diff,
                }
            )
    return pd.DataFrame(rows)


class TestBuildDiffMatrix(unittest.TestCase):
    """Tests for the genes x motifs difference matrix."""

    def test_absent_gene_motif_pairs_become_zero(self):
        # Arrange
        counts = make_per_gene_counts({"M1": {GENES[0]: 2}})

        # Act
        matrix = build_diff_matrix(counts, GENES)

        # Assert
        self.assertEqual(list(matrix.index), GENES)
        self.assertEqual(matrix.at[GENES[0], "M1"], 2)
        self.assertEqual(matrix.at[GENES[1], "M1"], 0.0)

    def test_gene_order_follows_gene_list(self):
        # Arrange
        counts = make_per_gene_counts({"M1": {gene: 1 for gene in GENES}})
        reversed_genes = list(reversed(GENES))

        # Act
        matrix = build_diff_matrix(counts, reversed_genes)

        # Assert
        self.assertEqual(list(matrix.index), reversed_genes)

    def test_empty_gene_list_raises(self):
        # Arrange
        counts = make_per_gene_counts({"M1": {GENES[0]: 1}})

        # Act / Assert
        with self.assertRaises(ValueError):
            build_diff_matrix(counts, [])

    def test_duplicate_genes_raise(self):
        # Arrange
        counts = make_per_gene_counts({"M1": {GENES[0]: 1}})

        # Act / Assert
        with self.assertRaises(ValueError):
            build_diff_matrix(counts, [GENES[0], GENES[0]])

    def test_gene_missing_from_gene_list_raises(self):
        # Arrange
        counts = make_per_gene_counts({"M1": {"unlisted_gene": 1}})

        # Act / Assert
        with self.assertRaises(ValueError):
            build_diff_matrix(counts, GENES)

    def test_duplicated_gene_motif_pair_raises(self):
        # Arrange
        counts = make_per_gene_counts({"M1": {GENES[0]: 1}})
        counts = pd.concat([counts, counts], ignore_index=True)

        # Act / Assert
        with self.assertRaises(ValueError):
            build_diff_matrix(counts, GENES)


class TestMotifSignificance(unittest.TestCase):
    """Tests for the per-motif within-run significance test."""

    def setUp(self):
        self.counts = make_per_gene_counts(
            {
                "gained": {gene: 1 for gene in GENES},
                "lost": {gene: -2 for gene in GENES},
                "unchanged": {gene: 0 for gene in GENES},
                "mixed": {GENES[0]: 3, GENES[1]: -3},
            }
        )
        self.significance = motif_significance(self.counts, GENES).set_index("motif_id")

    def test_columns_and_row_count(self):
        # Assert
        self.assertEqual(
            list(self.significance.reset_index().columns), SIGNIFICANCE_COLUMNS
        )
        self.assertEqual(len(self.significance), 4)

    def test_consistent_gain_is_significant_and_positive(self):
        # Assert
        self.assertLess(self.significance.at["gained", "q_intra"], 0.05)
        self.assertEqual(self.significance.at["gained", "significance_stars"], "*")
        self.assertEqual(self.significance.at["gained", "median_diff"], 1)
        self.assertEqual(self.significance.at["gained", "n_genes_gained"], len(GENES))
        self.assertEqual(self.significance.at["gained", "n_genes_lost"], 0)

    def test_consistent_loss_is_significant_and_negative(self):
        # Assert
        self.assertLess(self.significance.at["lost", "q_intra"], 0.05)
        self.assertEqual(self.significance.at["lost", "median_diff"], -2)
        self.assertEqual(self.significance.at["lost", "n_genes_lost"], len(GENES))

    def test_unchanged_motif_has_no_pvalue(self):
        # Assert
        self.assertTrue(math.isnan(self.significance.at["unchanged", "p_intra"]))
        self.assertTrue(math.isnan(self.significance.at["unchanged", "q_intra"]))
        self.assertEqual(self.significance.at["unchanged", "significance_stars"], "")
        self.assertEqual(
            self.significance.at["unchanged", "n_genes_unchanged"], len(GENES)
        )

    def test_gene_count_covers_all_genes_including_those_without_hits(self):
        # Assert
        self.assertEqual(self.significance.at["mixed", "n_genes"], len(GENES))
        self.assertEqual(self.significance.at["mixed", "n_genes_unchanged"], 6)
        self.assertEqual(self.significance.at["mixed", "median_diff"], 0.0)

    def test_mean_diff_uses_all_genes_as_denominator(self):
        # Assert: -2 in every one of the eight genes
        self.assertAlmostEqual(self.significance.at["lost", "mean_diff"], -2.0)

    def test_direction_follows_net_change_not_median(self):
        # Arrange: gains in 3 of 8 genes leave the median at 0
        counts = make_per_gene_counts(
            {"sparse_gain": {GENES[0]: 1, GENES[1]: 1, GENES[2]: 1}}
        )

        # Act
        significance = motif_significance(counts, GENES)

        # Assert
        self.assertEqual(significance.at[0, "median_diff"], 0.0)
        self.assertEqual(significance.at[0, "direction"], "enriched")

    def test_direction_is_empty_when_gains_and_losses_cancel(self):
        # Arrange
        counts = make_per_gene_counts({"balanced": {GENES[0]: 2, GENES[1]: -2}})

        # Act
        significance = motif_significance(counts, GENES)

        # Assert
        self.assertEqual(significance.at[0, "direction"], "")

    def test_directions_of_consistent_changes(self):
        # Assert
        self.assertEqual(self.significance.at["gained", "direction"], "enriched")
        self.assertEqual(self.significance.at["lost", "direction"], "removed")
        self.assertEqual(self.significance.at["unchanged", "direction"], "")

    def test_zero_filled_genes_do_not_change_the_pvalue(self):
        # Arrange: the same non-zero diffs, but observed in every gene
        sparse = make_per_gene_counts({"M1": {GENES[0]: 3, GENES[1]: -3}})
        dense_diffs = {gene: 0 for gene in GENES}
        dense_diffs[GENES[0]] = 3
        dense_diffs[GENES[1]] = -3
        dense = make_per_gene_counts({"M1": dense_diffs})

        # Act
        sparse_p = motif_significance(sparse, GENES).at[0, "p_intra"]
        dense_p = motif_significance(dense, GENES).at[0, "p_intra"]

        # Assert
        self.assertAlmostEqual(sparse_p, dense_p)

    def test_sorted_by_qvalue_ascending(self):
        # Arrange
        qvalues = motif_significance(self.counts, GENES)["q_intra"].dropna()

        # Assert
        self.assertEqual(list(qvalues), sorted(qvalues))

    def test_empty_input_returns_empty_table_with_header(self):
        # Act
        significance = motif_significance(
            pd.DataFrame(columns=pd.Index(["gene", "motif_id", "diff"])), GENES
        )

        # Assert
        self.assertTrue(significance.empty)
        self.assertEqual(list(significance.columns), SIGNIFICANCE_COLUMNS)


class TestSummariseSignificance(unittest.TestCase):
    """Tests for the console summary line."""

    def test_counts_directions(self):
        # Arrange
        significance = pd.DataFrame(
            {
                "direction": ["enriched", "removed", "", "enriched"],
                "significance_stars": ["***", "*", "*", ""],
            }
        )

        # Act
        summary = summarise_significance(significance)

        # Assert
        self.assertIn("3/4 motifs significant", summary)
        self.assertIn("1 enriched", summary)
        self.assertIn("1 removed", summary)


class TestAnalyseOutputFolder(unittest.TestCase):
    """Integration tests over a temporary output folder."""

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        make_per_gene_counts({"gained": {gene: 1 for gene in GENES}}).to_csv(
            os.path.join(self.folder, PER_GENE_COUNTS_FILE), index=False
        )
        pd.DataFrame({"gene": GENES}).to_csv(
            os.path.join(self.folder, SELECTED_ENTRIES_FILE), index=False
        )

    def test_writes_significance_file(self):
        # Act
        significance = analyse_output_folder(self.folder)
        written = pd.read_csv(os.path.join(self.folder, SIGNIFICANCE_FILE))

        # Assert
        self.assertEqual(len(significance), 1)
        self.assertEqual(list(written["motif_id"]), ["gained"])
        self.assertLess(written.at[0, "q_intra"], 0.05)

    def test_missing_per_gene_file_raises(self):
        # Arrange
        os.remove(os.path.join(self.folder, PER_GENE_COUNTS_FILE))

        # Act / Assert
        with self.assertRaises(FileNotFoundError):
            analyse_output_folder(self.folder)

    def test_missing_selected_entries_file_raises(self):
        # Arrange
        os.remove(os.path.join(self.folder, SELECTED_ENTRIES_FILE))

        # Act / Assert
        with self.assertRaises(FileNotFoundError):
            analyse_output_folder(self.folder)


class TestParseArguments(unittest.TestCase):
    """Tests for command line argument validation."""

    def test_accepts_several_existing_folders(self):
        # Arrange
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            # Act
            parsed = parse_arguments(["--output-folder", first, second])

            # Assert
            self.assertEqual(parsed.output_folder, [first, second])

    def test_rejects_missing_folder(self):
        # Act / Assert
        with self.assertRaises(SystemExit):
            parse_arguments(["--output-folder", "/nonexistent/folder"])


if __name__ == "__main__":
    unittest.main()
