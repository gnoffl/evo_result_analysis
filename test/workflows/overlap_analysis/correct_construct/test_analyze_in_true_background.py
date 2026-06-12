import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from workflows.overlap_analysis.correct_construct.analyze_in_true_background import (
    _parse_binding_category,
    load_starrseq_data,
    average_barcode_predictions,
    prepare_barcode_comparison_data,
    prepare_correlation_data,
)


class TestParseBindingCategory(unittest.TestCase):
    def test_non_binding(self) -> None:
        self.assertEqual(_parse_binding_category("WRKY_1:100-200_non_binding_537"), "non_binding")

    def test_binding(self) -> None:
        self.assertEqual(_parse_binding_category("bHLH_1:100-200_binding_1080"), "binding")

    def test_reference_binding(self) -> None:
        self.assertEqual(
            _parse_binding_category("WRKY_1:100-200_reference_binding_2114"), "binding"
        )

    def test_binding_takes_precedence_check(self) -> None:
        # non_binding must be detected before binding substring
        result = _parse_binding_category("bHLH_1:100-200_non_binding_999")
        self.assertEqual(result, "non_binding")


class TestLoadStarrseqData(unittest.TestCase):
    def _make_raw_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "condition": ["Dark", "Light", "Dark", "Dark"],
                "id": [
                    "WRKY_1:100-200_binding_1",
                    "bHLH_1:300-400_non_binding_2",
                    "other_gene_control",
                    "WRKY_1:500-600_reference_binding_3",
                ],
                "enrichment": [0.5, -0.3, 1.0, 0.8],
                "GC": [0.4, 0.5, 0.4, 0.4],
                "length": [170, 170, 170, 170],
                "sequence": ["ACGT", "ACGT", "ACGT", "ACGT"],
                "n_experiments": [2, 2, 2, 2],
                "min_bc": [1, 1, 1, 1],
                "min_ci": [1, 1, 1, 1],
                "min_co": [1, 1, 1, 1],
            }
        )

    @patch("workflows.overlap_analysis.correct_construct.analyze_in_true_background.pd.read_csv")
    def test_filters_to_bhlh_and_wrky(self, mock_read_csv: MagicMock) -> None:
        mock_read_csv.return_value = self._make_raw_df()
        result = load_starrseq_data("dummy_path.csv")
        self.assertFalse(result["id"].str.startswith("other").any())
        self.assertEqual(len(result), 3)

    @patch("workflows.overlap_analysis.correct_construct.analyze_in_true_background.pd.read_csv")
    def test_adds_tf_family_column(self, mock_read_csv: MagicMock) -> None:
        mock_read_csv.return_value = self._make_raw_df()
        result = load_starrseq_data("dummy_path.csv")
        self.assertIn("tf_family", result.columns)
        self.assertSetEqual(set(result["tf_family"].unique()), {"WRKY", "bHLH"})

    @patch("workflows.overlap_analysis.correct_construct.analyze_in_true_background.pd.read_csv")
    def test_adds_binding_category_column(self, mock_read_csv: MagicMock) -> None:
        mock_read_csv.return_value = self._make_raw_df()
        result = load_starrseq_data("dummy_path.csv")
        self.assertIn("binding_category", result.columns)
        categories = set(result["binding_category"].unique())
        self.assertTrue(categories.issubset({"binding", "non_binding"}))

    @patch("workflows.overlap_analysis.correct_construct.analyze_in_true_background.pd.read_csv")
    def test_output_columns(self, mock_read_csv: MagicMock) -> None:
        mock_read_csv.return_value = self._make_raw_df()
        result = load_starrseq_data("dummy_path.csv")
        expected = {"id", "condition", "enrichment", "tf_family", "binding_category"}
        self.assertSetEqual(set(result.columns), expected)


class TestAverageBarcodePrediictions(unittest.TestCase):
    def _make_predictions_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": ["seq_A", "seq_A", "seq_B", "seq_B"],
                "barcode": [1, 2, 1, 2],
                "prediction": [0.4, 0.6, 0.8, 0.2],
            }
        )

    def test_averages_barcodes(self) -> None:
        df = self._make_predictions_df()
        result = average_barcode_predictions(df)
        self.assertEqual(len(result), 2)
        row_a = result[result["id"] == "seq_A"].iloc[0]
        self.assertAlmostEqual(row_a["prediction"], 0.5)

    def test_output_columns(self) -> None:
        df = self._make_predictions_df()
        result = average_barcode_predictions(df)
        self.assertSetEqual(set(result.columns), {"id", "prediction"})

    def test_single_barcode_unchanged(self) -> None:
        df = pd.DataFrame({"id": ["seq_A"], "barcode": [1], "prediction": [0.7]})
        result = average_barcode_predictions(df)
        self.assertAlmostEqual(result.iloc[0]["prediction"], 0.7)


class TestPrepBarrodeComparisonData(unittest.TestCase):
    def _make_predictions_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": ["WRKY_seq_A", "WRKY_seq_A", "bHLH_seq_B", "bHLH_seq_B"],
                "barcode": [1, 2, 1, 2],
                "prediction": [0.4, 0.6, 0.8, 0.2],
            }
        )

    def test_pivots_to_two_barcode_columns(self) -> None:
        df = self._make_predictions_df()
        result = prepare_barcode_comparison_data(df)
        self.assertIn("barcode_1", result.columns)
        self.assertIn("barcode_2", result.columns)
        self.assertEqual(len(result), 2)

    def test_adds_tf_family(self) -> None:
        df = self._make_predictions_df()
        result = prepare_barcode_comparison_data(df)
        self.assertIn("tf_family", result.columns)
        self.assertSetEqual(set(result["tf_family"].unique()), {"WRKY", "bHLH"})

    def test_barcode_values_correct(self) -> None:
        df = self._make_predictions_df()
        result = prepare_barcode_comparison_data(df)
        row = result[result["id"] == "WRKY_seq_A"].iloc[0]
        self.assertAlmostEqual(row["barcode_1"], 0.4)
        self.assertAlmostEqual(row["barcode_2"], 0.6)


class TestPrepareCorrelationData(unittest.TestCase):
    def _make_inputs(self) -> tuple:
        predictions_df = pd.DataFrame(
            {
                "id": ["seq_A", "seq_A", "seq_B", "seq_B"],
                "barcode": [1, 2, 1, 2],
                "prediction": [0.4, 0.6, 0.8, 0.2],
            }
        )
        starrseq_df = pd.DataFrame(
            {
                "id": ["seq_A", "seq_A", "seq_B"],
                "condition": ["Light", "Dark", "Light"],
                "enrichment": [1.0, 0.5, -0.3],
                "tf_family": ["WRKY", "WRKY", "bHLH"],
                "binding_category": ["binding", "binding", "non_binding"],
            }
        )
        return predictions_df, starrseq_df

    def test_merges_on_id(self) -> None:
        preds, starrseq = self._make_inputs()
        result = prepare_correlation_data(preds, starrseq)
        self.assertIn("prediction", result.columns)
        self.assertIn("enrichment", result.columns)

    def test_uses_averaged_predictions(self) -> None:
        preds, starrseq = self._make_inputs()
        result = prepare_correlation_data(preds, starrseq)
        seq_a_rows = result[result["id"] == "seq_A"]
        self.assertTrue((seq_a_rows["prediction"] == 0.5).all())

    def test_unmatched_starrseq_ids_dropped(self) -> None:
        preds, starrseq = self._make_inputs()
        starrseq_extra = pd.concat(
            [starrseq, pd.DataFrame({"id": ["seq_C"], "condition": ["Light"],
                                     "enrichment": [0.0], "tf_family": ["WRKY"],
                                     "binding_category": ["binding"]})],
            ignore_index=True,
        )
        result = prepare_correlation_data(preds, starrseq_extra)
        self.assertFalse((result["id"] == "seq_C").any())


if __name__ == "__main__":
    unittest.main()
