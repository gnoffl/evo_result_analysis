"""Tests for the one-off mutation-assessment script.

Only the sequence-manipulation logic is checked (no model is loaded).
"""

import io
import unittest
from unittest.mock import mock_open, patch

import pandas as pd

from workflows.hoffie.mutation_assessment.assess_mutations import (
    insert_poly_c,
    load_prediction_results,
)


class TestInsertPolyC(unittest.TestCase):
    def test_grows_run_by_one_and_keeps_1500_bp(self):
        # Arrange: 1500 bp promoter with a CGTCCT-flanked 14x C run.
        promoter = ("A" * 100 + "CGTCCT" + "C" * 14 + "T" * 1380)[:1500]
        self.assertEqual(len(promoter), 1500)

        # Act
        result = insert_poly_c(promoter, "CGTCCT")

        # Assert
        self.assertEqual(len(result), 1500)  # stays 1500 bp
        self.assertIn("CGTCCT" + "C" * 15, result)  # run grew 14 -> 15
        self.assertEqual(result[-1], promoter[-2])  # last promoter base dropped


class TestLoadPredictionResults(unittest.TestCase):
    _CSV_CONTENT = (
        "condition,variant,length,prediction\n"
        "zma_min,original,3020,0.049\n"
        "zma_min,lab_variant,3020,0.052\n"
        "zma_max,original,3020,0.9997\n"
    )

    def test_returns_dataframe_with_expected_columns(self):
        with patch("builtins.open", mock_open(read_data=self._CSV_CONTENT)):
            with patch("pandas.read_csv", return_value=pd.read_csv(io.StringIO(self._CSV_CONTENT))):
                result = load_prediction_results("dummy.csv")
        self.assertListEqual(list(result.columns), ["condition", "variant", "length", "prediction"])

    def test_returns_all_rows(self):
        with patch("pandas.read_csv", return_value=pd.read_csv(io.StringIO(self._CSV_CONTENT))):
            result = load_prediction_results("dummy.csv")
        self.assertEqual(len(result), 3)

    def test_prediction_values_are_floats(self):
        with patch("pandas.read_csv", return_value=pd.read_csv(io.StringIO(self._CSV_CONTENT))):
            result = load_prediction_results("dummy.csv")
        self.assertTrue(result["prediction"].dtype == float)

    def test_missing_lab_variant_condition_has_only_original(self):
        with patch("pandas.read_csv", return_value=pd.read_csv(io.StringIO(self._CSV_CONTENT))):
            result = load_prediction_results("dummy.csv")
        zma_max_variants = set(result[result["condition"] == "zma_max"]["variant"])
        self.assertEqual(zma_max_variants, {"original"})


if __name__ == "__main__":
    unittest.main()
