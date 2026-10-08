"""Unit tests for the four-group construct-background comparison.

Model inference is mocked out by patching the per-model scoring function; the
tests cover this module's own logic: the group median, the group summary, and
that the plotting functions write a file.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import pandas as pd

from workflows.overlap_analysis.model_species_comparison import model_group_comparison
from workflows.overlap_analysis.model_species_comparison._model_groups import ModelGroup
from workflows.overlap_analysis.model_species_comparison._models import SpeciesModel

MSR_MODEL = SpeciesModel("MSR", "A_thaliana", "/models/msr_1.h5")
MSR_MODEL_2 = SpeciesModel("MSR", "S_lycopersicum", "/models/msr_2.h5")
MSR_MODEL_3 = SpeciesModel("MSR", "Z_mays", "/models/msr_3.h5")
NTAB_MODEL = SpeciesModel("Ntab", "NtabPC1", "/models/ntab_1.h5")


def _correlation_frame(predictions):
    """Build a minimal merged correlation frame with three inserts."""
    return pd.DataFrame(
        {
            "id": ["insert_a", "insert_b", "insert_c"],
            "condition": ["Dark", "Light", "Dark"],
            "enrichment": [1.0, 2.0, 3.0],
            "tf_family": ["WRKY", "bHLH", "WRKY"],
            "binding_category": ["binding", "non_binding", "binding"],
            "prediction": predictions,
        }
    )


class TestBuildGroupMedianDf(unittest.TestCase):
    def test_uses_the_per_insert_median_over_the_group(self) -> None:
        # Arrange
        model_dataframes = [
            (MSR_MODEL, _correlation_frame([0.0, 1.0, 2.0])),
            (MSR_MODEL_2, _correlation_frame([1.0, 2.0, 3.0])),
            (MSR_MODEL_3, _correlation_frame([10.0, 2.5, 3.5])),
        ]

        # Act
        median_df = model_group_comparison.build_group_median_df(model_dataframes)

        # Assert
        self.assertEqual(list(median_df["prediction"]), [1.0, 2.0, 3.0])

    def test_keeps_the_other_columns_untouched(self) -> None:
        # Arrange
        model_dataframes = [(MSR_MODEL, _correlation_frame([0.1, 0.2, 0.3]))]

        # Act
        median_df = model_group_comparison.build_group_median_df(model_dataframes)

        # Assert
        self.assertEqual(list(median_df["enrichment"]), [1.0, 2.0, 3.0])
        self.assertEqual(list(median_df["id"]), ["insert_a", "insert_b", "insert_c"])

    def test_does_not_modify_the_input_dataframes(self) -> None:
        # Arrange
        first_df = _correlation_frame([0.0, 1.0, 2.0])
        model_dataframes = [
            (MSR_MODEL, first_df),
            (MSR_MODEL_2, _correlation_frame([2.0, 3.0, 4.0])),
        ]

        # Act
        model_group_comparison.build_group_median_df(model_dataframes)

        # Assert
        self.assertEqual(list(first_df["prediction"]), [0.0, 1.0, 2.0])

    def test_raises_without_models(self) -> None:
        # Act / Assert
        with self.assertRaises(ValueError):
            model_group_comparison.build_group_median_df([])

    def test_raises_when_the_pairs_span_two_groups(self) -> None:
        # Arrange
        model_dataframes = [
            (MSR_MODEL, _correlation_frame([0.0, 1.0, 2.0])),
            (NTAB_MODEL, _correlation_frame([0.0, 1.0, 2.0])),
        ]

        # Act / Assert
        with self.assertRaises(ValueError):
            model_group_comparison.build_group_median_df(model_dataframes)


class TestSummarizeGroupMedians(unittest.TestCase):
    def test_reports_one_row_per_group_best_first(self) -> None:
        # Arrange: MSR predictions are monotone in enrichment, Ntab anti-monotone.
        group_dataframes = [
            ("MSR", _correlation_frame([0.1, 0.2, 0.3])),
            ("Ntab", _correlation_frame([0.3, 0.2, 0.1])),
        ]

        # Act
        summary_df = model_group_comparison.summarize_group_medians(group_dataframes)

        # Assert
        self.assertEqual(list(summary_df["group"]), ["MSR", "Ntab"])
        self.assertAlmostEqual(float(summary_df.loc[0, "spearman_r"]), 1.0)
        self.assertAlmostEqual(float(summary_df.loc[1, "spearman_r"]), -1.0)
        self.assertEqual(list(summary_df["count_points"]), [3, 3])

    def test_ignores_rows_with_a_missing_prediction(self) -> None:
        # Arrange
        frame = _correlation_frame([0.1, None, 0.3])
        group_dataframes = [("MSR", frame)]

        # Act
        summary_df = model_group_comparison.summarize_group_medians(group_dataframes)

        # Assert
        self.assertEqual(int(summary_df.loc[0, "count_points"]), 2)


class TestPlotting(unittest.TestCase):
    def test_panel_figure_is_written(self) -> None:
        # Arrange
        labelled_dataframes = [
            (f"group_{index}", _correlation_frame([0.1, 0.2, 0.3]))
            for index in range(4)
        ]

        # Act
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "panels", "figure.png")
            model_group_comparison.plot_group_panels(labelled_dataframes, output_path)

            # Assert
            self.assertTrue(os.path.exists(output_path))

    def test_panel_figure_handles_an_odd_number_of_groups(self) -> None:
        # Arrange
        labelled_dataframes = [
            (f"group_{index}", _correlation_frame([0.1, 0.2, 0.3]))
            for index in range(3)
        ]

        # Act
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "figure.png")
            model_group_comparison.plot_group_panels(labelled_dataframes, output_path)

            # Assert
            self.assertTrue(os.path.exists(output_path))

    def test_per_model_spread_figure_is_written(self) -> None:
        # Arrange
        summary_df = pd.DataFrame(
            {
                "species": ["MSR", "MSR", "Ntab", "Ntab"],
                "held_out_chromosome": ["A_thaliana", "Z_mays", "NtabPC1", "NtabPC2"],
                "count_points": [3, 3, 3, 3],
                "spearman_r": [0.4, 0.5, 0.2, 0.1],
                "spearman_p": [0.01, 0.01, 0.2, 0.3],
            }
        )

        # Act
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "spread.png")
            model_group_comparison.plot_per_model_spread(summary_df, output_path)

            # Assert
            self.assertTrue(os.path.exists(output_path))


class TestCorrelationDataForGroup(unittest.TestCase):
    def test_scores_every_model_of_the_group_and_aligns_the_rows(self) -> None:
        # Arrange
        group = ModelGroup("MSR", "/models/msr", r"(x)")
        starrseq_df = pd.DataFrame({"id": ["insert_a"]})
        unsorted_frame = _correlation_frame([0.3, 0.2, 0.1]).iloc[::-1]

        # Act
        with patch.object(
            model_group_comparison, "list_group_models",
            return_value=[MSR_MODEL, MSR_MODEL_2],
        ), patch.object(
            model_group_comparison, "correlation_data_for_model",
            side_effect=[unsorted_frame, _correlation_frame([0.9, 0.8, 0.7])],
        ) as scoring_mock:
            aligned = model_group_comparison.correlation_data_for_group(
                group, starrseq_df
            )

        # Assert
        self.assertEqual(scoring_mock.call_count, 2)
        self.assertEqual([model for model, _ in aligned], [MSR_MODEL, MSR_MODEL_2])
        for _, dataframe in aligned:
            self.assertEqual(
                list(dataframe["id"]), ["insert_a", "insert_b", "insert_c"]
            )


if __name__ == "__main__":
    unittest.main()
