"""Unit tests for the calculation functions of ``binding_vs_nonbinding``.

Only the pure calculation/aggregation functions are tested; plotting and the
deepCRE model are not exercised. Fixtures are small in-memory DataFrames.
"""
import unittest

import pandas as pd

from workflows.overlap_analysis.binding_vs_nonbinding import binding_vs_nonbinding as bvn


class TestBuildVectorComparisonDf(unittest.TestCase):
    """Tests for ``build_vector_comparison_df``."""

    def test_averages_barcodes_and_parses_id(self) -> None:
        # Arrange: one binding and one non-binding sequence, two barcodes each.
        predictions_df = pd.DataFrame(
            {
                "id": [
                    "WRKY_1:100-200_binding_5",
                    "WRKY_1:100-200_binding_5",
                    "bHLH_2:300-400_non_binding_9",
                    "bHLH_2:300-400_non_binding_9",
                ],
                "barcode": [1, 2, 1, 2],
                "prediction": [0.8, 0.6, 0.2, 0.4],
            }
        )

        # Act
        result = bvn.build_vector_comparison_df(predictions_df)

        # Assert
        self.assertEqual(
            sorted(result.columns), ["binding_category", "prediction", "tf_family"]
        )
        self.assertEqual(len(result), 2)
        wrky_row = result[result["tf_family"] == "WRKY"].iloc[0]
        self.assertEqual(wrky_row["binding_category"], "binding")
        self.assertAlmostEqual(wrky_row["prediction"], 0.7)
        bhlh_row = result[result["tf_family"] == "bHLH"].iloc[0]
        self.assertEqual(bhlh_row["binding_category"], "non_binding")
        self.assertAlmostEqual(bhlh_row["prediction"], 0.3)


class TestBuildGenomicComparisonDf(unittest.TestCase):
    """Tests for ``build_genomic_comparison_df``."""

    def test_tags_tf_drops_reference_and_collapses_conditions(self) -> None:
        # Arrange: s1 appears twice (Light/Dark duplicate, same gene) and should
        # collapse to one row; s2 maps to two genes (g1, g2) and both are kept;
        # s3 is a reference baseline and should be dropped.
        wrky_df = pd.DataFrame(
            {
                "starr_full_name": ["s1", "s1", "s2", "s2", "s3"],
                "gene": ["g1", "g1", "g1", "g2", "g1"],
                "starr_binding_status": [
                    "binding",
                    "binding",
                    "binding",
                    "binding",
                    "reference",
                ],
                "prediction_mutated": [0.9, 0.9, 0.4, 0.6, 0.7],
            }
        )
        bhlh_df = pd.DataFrame(
            {
                "starr_full_name": ["b1", "b2"],
                "gene": ["g3", "g3"],
                "starr_binding_status": ["binding", "non_binding"],
                "prediction_mutated": [0.8, 0.3],
            }
        )

        # Act
        result = bvn.build_genomic_comparison_df(wrky_df, bhlh_df)

        # Assert
        self.assertEqual(
            sorted(result.columns), ["binding_category", "prediction", "tf_family"]
        )
        self.assertNotIn("reference", result["binding_category"].tolist())
        self.assertEqual(set(result["tf_family"]), {"WRKY", "bHLH"})
        # WRKY: s1 collapsed to 1 + s2 across two genes = 3; bHLH: 2 => total 5.
        self.assertEqual(len(result), 5)
        wrky = result[result["tf_family"] == "WRKY"]
        self.assertEqual(len(wrky), 3)
        # s1 Light/Dark duplicate collapsed to a single 0.9 prediction.
        self.assertEqual((wrky["prediction"] == 0.9).sum(), 1)
        # Both distinct gene-window predictions for s2 are retained.
        self.assertIn(0.4, wrky["prediction"].tolist())
        self.assertIn(0.6, wrky["prediction"].tolist())


class TestComputeBindingSignificance(unittest.TestCase):
    """Tests for ``compute_binding_significance``."""

    def test_counts_medians_and_significant_separation(self) -> None:
        # Arrange: WRKY binding clearly above non_binding; bHLH similarly.
        comparison_df = pd.DataFrame(
            {
                "tf_family": ["WRKY"] * 10 + ["bHLH"] * 10,
                "binding_category": (["binding"] * 5 + ["non_binding"] * 5) * 2,
                "prediction": (
                    [0.90, 0.92, 0.95, 0.93, 0.91]
                    + [0.20, 0.22, 0.25, 0.21, 0.19]
                )
                * 2,
            }
        )

        # Act
        stats = bvn.compute_binding_significance(comparison_df)

        # Assert
        self.assertEqual(set(stats["tf_family"]), {"WRKY", "bHLH"})
        wrky = stats[stats["tf_family"] == "WRKY"].iloc[0]
        self.assertEqual(wrky["n_binding"], 5)
        self.assertEqual(wrky["n_non_binding"], 5)
        self.assertGreater(wrky["median_binding"], wrky["median_non_binding"])
        self.assertLess(wrky["p_value"], 0.05)

    def test_missing_category_yields_nan_pvalue(self) -> None:
        # Arrange: only binding rows present for the single TF.
        comparison_df = pd.DataFrame(
            {
                "tf_family": ["WRKY", "WRKY"],
                "binding_category": ["binding", "binding"],
                "prediction": [0.8, 0.9],
            }
        )

        # Act
        stats = bvn.compute_binding_significance(comparison_df)

        # Assert
        self.assertEqual(stats.iloc[0]["n_non_binding"], 0)
        self.assertTrue(pd.isna(stats.iloc[0]["p_value"]))


class TestBuildVectorDeltaDf(unittest.TestCase):
    """Tests for ``build_vector_delta_df``."""

    def test_delta_vs_locus_reference_excludes_references(self) -> None:
        # Arrange: one locus with a reference, a binding and a non_binding
        # variant, each scored on two barcodes.
        predictions_df = pd.DataFrame(
            {
                "id": [
                    "WRKY_1:100-200_reference_binding_1",
                    "WRKY_1:100-200_reference_binding_1",
                    "WRKY_1:100-200_binding_2",
                    "WRKY_1:100-200_binding_2",
                    "WRKY_1:100-200_non_binding_3",
                    "WRKY_1:100-200_non_binding_3",
                ],
                "barcode": [1, 2, 1, 2, 1, 2],
                "prediction": [0.80, 0.80, 0.90, 0.70, 0.50, 0.30],
            }
        )

        # Act
        result = bvn.build_vector_delta_df(predictions_df)

        # Assert
        self.assertEqual(
            sorted(result.columns), ["binding_category", "delta", "tf_family"]
        )
        self.assertNotIn("reference", result["binding_category"].tolist())
        self.assertEqual(len(result), 2)
        binding_delta = result[result["binding_category"] == "binding"]["delta"].iloc[0]
        non_binding_delta = result[result["binding_category"] == "non_binding"][
            "delta"
        ].iloc[0]
        # ref mean = 0.80; binding mean = 0.80 -> 0.0; non_binding mean = 0.40 -> -0.40.
        self.assertAlmostEqual(binding_delta, 0.0)
        self.assertAlmostEqual(non_binding_delta, -0.40)

    def test_variants_without_locus_reference_are_dropped(self) -> None:
        # Arrange: a variant whose locus has no reference entry.
        predictions_df = pd.DataFrame(
            {
                "id": ["WRKY_1:300-400_binding_9", "WRKY_1:300-400_binding_9"],
                "barcode": [1, 2],
                "prediction": [0.6, 0.6],
            }
        )

        # Act
        result = bvn.build_vector_delta_df(predictions_df)

        # Assert
        self.assertEqual(len(result), 0)


class TestBuildGenomicDeltaDf(unittest.TestCase):
    """Tests for ``build_genomic_delta_df``."""

    def test_uses_delta_excludes_reference_and_collapses_conditions(self) -> None:
        # Arrange: r1 is a reference (excluded); v1 appears twice (Light/Dark)
        # and collapses; v2 is non_binding.
        wrky_df = pd.DataFrame(
            {
                "starr_full_name": ["r1", "r1", "v1", "v1", "v2"],
                "gene": ["g1", "g1", "g1", "g1", "g1"],
                "starr_reference": [True, True, False, False, False],
                "starr_binding_status": [
                    "binding",
                    "binding",
                    "binding",
                    "binding",
                    "non_binding",
                ],
                "delta_prediction": [0.0, 0.0, 0.1, 0.1, -0.2],
            }
        )
        bhlh_df = pd.DataFrame(
            {
                "starr_full_name": ["b1", "b2"],
                "gene": ["g2", "g2"],
                "starr_reference": [False, False],
                "starr_binding_status": ["binding", "non_binding"],
                "delta_prediction": [0.05, -0.1],
            }
        )

        # Act
        result = bvn.build_genomic_delta_df(wrky_df, bhlh_df)

        # Assert
        self.assertEqual(
            sorted(result.columns), ["binding_category", "delta", "tf_family"]
        )
        # WRKY: v1 (collapsed) + v2 = 2; bHLH: 2 => total 4 (references dropped).
        self.assertEqual(len(result), 4)
        wrky = result[result["tf_family"] == "WRKY"]
        self.assertEqual(len(wrky), 2)
        self.assertAlmostEqual(
            wrky[wrky["binding_category"] == "binding"]["delta"].iloc[0], 0.1
        )


class TestRankEffectSize(unittest.TestCase):
    """Tests for ``_rank_effect_size``."""

    def test_auc_and_rank_biserial_from_u(self) -> None:
        # Arrange: U = 75 with 10 x 10 pairs -> AUC = 0.75.
        # Act
        auc, rank_biserial_r = bvn._rank_effect_size(75.0, 10, 10)

        # Assert
        self.assertAlmostEqual(auc, 0.75)
        self.assertAlmostEqual(rank_biserial_r, 0.5)

    def test_no_effect_maps_to_half_and_zero(self) -> None:
        # Arrange / Act: U at exactly half the pairs.
        auc, rank_biserial_r = bvn._rank_effect_size(50.0, 10, 10)

        # Assert
        self.assertAlmostEqual(auc, 0.5)
        self.assertAlmostEqual(rank_biserial_r, 0.0)

    def test_empty_group_returns_nan(self) -> None:
        # Arrange / Act
        auc, rank_biserial_r = bvn._rank_effect_size(float("nan"), 0, 5)

        # Assert
        self.assertTrue(pd.isna(auc))
        self.assertTrue(pd.isna(rank_biserial_r))


class TestComputeDeltaSignificance(unittest.TestCase):
    """Tests for ``compute_delta_significance``."""

    def test_two_group_and_vs_zero_tests(self) -> None:
        # Arrange: binding deltas near zero, non_binding deltas clearly negative.
        # Eight per group so Wilcoxon can reach significance (min two-sided p at
        # n=5 is 0.0625).
        delta_df = pd.DataFrame(
            {
                "tf_family": ["WRKY"] * 16,
                "binding_category": ["binding"] * 8 + ["non_binding"] * 8,
                "delta": [0.01, 0.02, -0.01, 0.0, 0.015, -0.005, 0.008, 0.012]
                + [-0.30, -0.25, -0.28, -0.31, -0.27, -0.29, -0.26, -0.32],
            }
        )

        # Act
        stats = bvn.compute_delta_significance(delta_df)

        # Assert
        row = stats[stats["tf_family"] == "WRKY"].iloc[0]
        self.assertEqual(row["n_binding"], 8)
        self.assertEqual(row["n_non_binding"], 8)
        self.assertGreater(row["median_binding"], row["median_non_binding"])
        self.assertLess(row["p_value"], 0.05)
        # Non-binding deltas are all negative -> clearly different from zero.
        self.assertLess(row["p_non_binding_vs_zero"], 0.05)
        self.assertFalse(pd.isna(row["p_binding_vs_zero"]))


if __name__ == "__main__":
    unittest.main()
