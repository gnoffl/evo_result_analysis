"""Tests for the deepCRE-TPM correlation analysis and plotting helpers."""

import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless backend for tests
import matplotlib.pyplot as plt
import pandas as pd

from workflows.deepCRE_TPM_correlation.correlate import (
    analyze_model,
    create_plots,
    merge_data,
)


class TestAnalyzeModel(unittest.TestCase):
    def test_returns_expected_keys(self) -> None:
        merged = pd.DataFrame(
            {"model_a": [0.1, 0.2, 0.3, 0.4], "logMaxTPM": [1.0, 2.0, 2.9, 4.1]}
        )
        result = analyze_model(merged, "model_a")
        for key in (
            "data",
            "slope",
            "intercept",
            "r_value",
            "p_value",
            "corr_coef",
            "corr_pval",
        ):
            self.assertIn(key, result)


class TestMergeData(unittest.TestCase):
    def test_inner_join_on_gene_id(self) -> None:
        tpm = pd.DataFrame(
            {"logMaxTPM": [1.0, 2.0]}, index=pd.Index(["g1", "g2"], name="gene_id")
        )
        pred = pd.DataFrame(
            {"model_a": [0.5, 0.6]}, index=pd.Index(["g1", "g3"], name="gene_id")
        )
        merged = merge_data(tpm, pred)
        self.assertEqual(list(merged.index), ["g1"])
        self.assertIn("model_a", merged.columns)


class TestCreatePlotsAxInjection(unittest.TestCase):
    @patch("matplotlib.pyplot.savefig")
    def test_create_plots_with_ax(self, mock_savefig) -> None:
        # Arrange
        data = pd.DataFrame(
            {"model_a": [0.1, 0.2, 0.3, 0.4], "logMaxTPM": [1.0, 2.0, 2.9, 4.1]}
        )
        results = {"model_a": {"data": data, "slope": 1.0, "intercept": 0.0}}
        fig, ax = plt.subplots()
        try:
            # Act
            create_plots(results, ["model_a"], ax=ax)

            # Assert
            self.assertGreater(len(ax.collections), 0)  # scatter drawn
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
