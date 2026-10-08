"""Unit tests for the shared species-comparison helpers.

Covers the row alignment that guards cross-model averaging, the per-model
Spearman summary, the between-species comparison, and the CSV writer's comment
header. Plotting is exercised only far enough to confirm it writes a file.
"""
import os
import tempfile
import unittest
from collections import namedtuple

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import pandas as pd

from workflows.overlap_analysis.model_species_comparison import _summary

FakeModel = namedtuple("FakeModel", ["species", "held_out_chromosome"])


def _fake_frame(ids, predictions, enrichments=None):
    """Build a minimal per-model frame with an ``id`` row key."""
    return pd.DataFrame(
        {
            "id": ids,
            "prediction": predictions,
            "enrichment": enrichments if enrichments is not None else [0.0] * len(ids),
        }
    )


class TestAlignByRowKey(unittest.TestCase):
    def test_sorts_every_frame_into_one_shared_order(self) -> None:
        # Arrange: the two models emit the same rows in opposite order.
        model_a = FakeModel("Ara", "NC_1")
        model_b = FakeModel("Slyc", "NC_2")
        frame_a = _fake_frame(["b", "a"], [1.0, 2.0])
        frame_b = _fake_frame(["a", "b"], [3.0, 4.0])

        # Act
        aligned = _summary.align_by_row_key(
            [(model_a, frame_a), (model_b, frame_b)], ["id"]
        )

        # Assert: both frames now start with row "a", carrying that row's value.
        self.assertEqual(list(aligned[0][1]["id"]), ["a", "b"])
        self.assertEqual(list(aligned[1][1]["id"]), ["a", "b"])
        self.assertEqual(list(aligned[0][1]["prediction"]), [2.0, 1.0])
        self.assertEqual(list(aligned[1][1]["prediction"]), [3.0, 4.0])

    def test_narrows_key_to_columns_present(self) -> None:
        # Arrange: "condition" is absent, so only "id" may be used as the key.
        model = FakeModel("Ara", "NC_1")
        frame = _fake_frame(["b", "a"], [1.0, 2.0])

        # Act
        aligned = _summary.align_by_row_key([(model, frame)], ["id", "condition"])

        # Assert
        self.assertEqual(list(aligned[0][1]["id"]), ["a", "b"])

    def test_raises_when_no_key_column_present(self) -> None:
        # Arrange
        model = FakeModel("Ara", "NC_1")
        frame = _fake_frame(["a"], [1.0])

        # Act / Assert
        with self.assertRaises(ValueError):
            _summary.align_by_row_key([(model, frame)], ["missing_column"])

    def test_raises_on_duplicated_row_key(self) -> None:
        # Arrange: the same id twice means averaging could pair wrong rows.
        model = FakeModel("Ara", "NC_1")
        frame = _fake_frame(["a", "a"], [1.0, 2.0])

        # Act / Assert
        with self.assertRaises(ValueError) as context:
            _summary.align_by_row_key([(model, frame)], ["id"])
        self.assertIn("not unique", str(context.exception))

    def test_raises_when_models_scored_different_rows(self) -> None:
        # Arrange
        model_a = FakeModel("Ara", "NC_1")
        model_b = FakeModel("Slyc", "NC_2")
        frame_a = _fake_frame(["a", "b"], [1.0, 2.0])
        frame_b = _fake_frame(["a", "c"], [3.0, 4.0])

        # Act / Assert
        with self.assertRaises(ValueError) as context:
            _summary.align_by_row_key([(model_a, frame_a), (model_b, frame_b)], ["id"])
        self.assertIn("different", str(context.exception))


class TestSummarizePerModel(unittest.TestCase):
    def test_one_row_per_model_with_spearman(self) -> None:
        # Arrange: Ara predictions rank with enrichment, Slyc against it.
        long_predictions = pd.DataFrame(
            {
                "species": ["Ara"] * 4 + ["Slyc"] * 4,
                "held_out_chromosome": ["NC_1"] * 4 + ["NC_2"] * 4,
                "prediction": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
                "enrichment": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
            }
        )

        # Act
        summary_df = _summary.summarize_per_model(long_predictions)

        # Assert
        self.assertEqual(len(summary_df), 2)
        ara_row = summary_df[summary_df["species"] == "Ara"].iloc[0]
        slyc_row = summary_df[summary_df["species"] == "Slyc"].iloc[0]
        self.assertAlmostEqual(ara_row["spearman_r"], 1.0)
        self.assertAlmostEqual(slyc_row["spearman_r"], -1.0)
        self.assertEqual(ara_row["count_points"], 4)

    def test_drops_incomplete_pairs_from_the_count(self) -> None:
        # Arrange: one row has no measurement.
        long_predictions = pd.DataFrame(
            {
                "species": ["Ara"] * 4,
                "held_out_chromosome": ["NC_1"] * 4,
                "prediction": [1.0, 2.0, 3.0, 4.0],
                "enrichment": [1.0, 2.0, 3.0, None],
            }
        )

        # Act
        summary_df = _summary.summarize_per_model(long_predictions)

        # Assert
        self.assertEqual(summary_df.iloc[0]["count_points"], 3)

    def test_raises_on_missing_column(self) -> None:
        # Arrange
        long_predictions = pd.DataFrame({"species": ["Ara"], "prediction": [1.0]})

        # Act / Assert
        with self.assertRaises(ValueError):
            _summary.summarize_per_model(long_predictions)


class TestCompareSpecies(unittest.TestCase):
    def test_reports_medians_ranges_and_test(self) -> None:
        # Arrange: Ara clearly above Slyc, two models each.
        summary_df = pd.DataFrame(
            {
                "species": ["Ara", "Ara", "Slyc", "Slyc"],
                "held_out_chromosome": ["NC_1", "NC_2", "NC_3", "NC_4"],
                "count_points": [10, 10, 10, 10],
                "spearman_r": [0.40, 0.50, 0.10, 0.20],
                "spearman_p": [0.01] * 4,
            }
        )

        # Act
        comparison_df = _summary.compare_species(summary_df)

        # Assert
        row = comparison_df.iloc[0]
        self.assertEqual(row["count_models_Ara"], 2)
        self.assertEqual(row["count_models_Slyc"], 2)
        self.assertAlmostEqual(row["median_spearman_r_Ara"], 0.45)
        self.assertAlmostEqual(row["min_spearman_r_Slyc"], 0.10)
        self.assertAlmostEqual(row["max_spearman_r_Slyc"], 0.20)
        self.assertFalse(pd.isna(row["mannwhitney_p"]))

    def test_test_is_nan_with_a_single_model_per_species(self) -> None:
        # Arrange
        summary_df = pd.DataFrame(
            {
                "species": ["Ara", "Slyc"],
                "held_out_chromosome": ["NC_1", "NC_2"],
                "count_points": [10, 10],
                "spearman_r": [0.4, 0.1],
                "spearman_p": [0.01, 0.01],
            }
        )

        # Act
        comparison_df = _summary.compare_species(summary_df)

        # Assert
        self.assertTrue(pd.isna(comparison_df.iloc[0]["mannwhitney_p"]))


class TestOutputWriters(unittest.TestCase):
    def setUp(self) -> None:
        self.summary_df = pd.DataFrame(
            {
                "species": ["Ara", "Slyc"],
                "held_out_chromosome": ["NC_1", "NC_2"],
                "count_points": [10, 10],
                "spearman_r": [0.4, 0.1],
                "spearman_p": [0.01, 0.20],
            }
        )

    def test_csv_carries_caveat_and_reads_back_unchanged(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "nested", "per_model_correlations.csv")

            # Act
            _summary.write_summary_csv(self.summary_df, output_path)
            with open(output_path) as output_file:
                first_line = output_file.readline()
            round_tripped = pd.read_csv(output_path, comment="#")

            # Assert
            self.assertTrue(first_line.startswith("#"))
            pd.testing.assert_frame_equal(round_tripped, self.summary_df)

    def test_plot_writes_an_image(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "nested", "per_model_correlations.png")

            # Act
            _summary.plot_per_model_correlations(self.summary_df, output_path, "title")

            # Assert
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)


if __name__ == "__main__":
    unittest.main()
