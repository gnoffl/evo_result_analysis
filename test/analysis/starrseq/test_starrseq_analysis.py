import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import pandas as pd
from analysis.starrseq.starrseq_analysis import (
    visualize_simple,
    visualize_simple_2,
    visualize_differences,
    binding_boxplots,
)


def make_binding_boxplot_data() -> dict:
    """Build a minimal ``{category: DataFrame}`` mapping for binding_boxplots."""
    return {
        "catA": pd.DataFrame({"binding": [True, False, True], "enrichment": [2.0, 0.5, 1.8]}),
        "catB": pd.DataFrame({"binding": [False, True, False], "enrichment": [0.4, 2.2, 0.6]}),
    }


def make_mock_dataframe() -> pd.DataFrame:
    """Build a tiny StarrSeq-like DataFrame covering both transcription factors.

    The frame carries every column the refactored plotting functions read,
    including the extra metadata columns that ``visualize_differences`` drops.

    Returns:
        pd.DataFrame: Minimal StarrSeq data for WRKY and bHLH with one reference,
        one binding and one non-binding variant per transcription factor.
    """
    rows = []
    for transcription_factor in ["WRKY", "bHLH"]:
        rows.extend(
            [
                {"id": f"{transcription_factor}_a_reference", "binding": False,
                 "reference": True, "enrichment": 1.0},
                {"id": f"{transcription_factor}_a_bind", "binding": True,
                 "reference": False, "enrichment": 2.0},
                {"id": f"{transcription_factor}_a_nonbind", "binding": False,
                 "reference": False, "enrichment": 0.5},
            ]
        )
    dataframe = pd.DataFrame(rows)
    # Metadata columns dropped by visualize_differences; present to mirror load_df.
    for extra_column in ["GC", "length", "min_bc", "min_ci", "min_co", "n_experiments"]:
        dataframe[extra_column] = 0
    return dataframe


class TestStarrseqVisualizationsWithAx(unittest.TestCase):
    """Ax-injection tests for the StarrSeq single-figure plotting functions."""

    def setUp(self) -> None:
        """Patch load_df so no data file is read during the tests."""
        self.load_df_patcher = patch(
            'analysis.starrseq.starrseq_analysis.load_df',
            side_effect=make_mock_dataframe,
        )
        self.mock_load_df = self.load_df_patcher.start()

    def tearDown(self) -> None:
        """Stop patches and close any open figures."""
        self.load_df_patcher.stop()
        plt.close('all')

    @patch('matplotlib.pyplot.savefig')
    def test_visualize_simple_with_ax(self, mock_savefig):
        """Passing an ax draws onto it and does not save a figure."""
        fig, ax = plt.subplots()
        try:
            visualize_simple(ax=ax)
            self.assertGreater(len(ax.texts), 0)  # n= annotations drawn on ax
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)

    @patch('matplotlib.pyplot.savefig')
    def test_visualize_simple_2_with_ax(self, mock_savefig):
        """Passing an ax draws onto it and does not save a figure."""
        fig, ax = plt.subplots()
        try:
            visualize_simple_2(ax=ax)
            self.assertGreater(len(ax.texts), 0)  # n= annotations drawn on ax
            self.assertIsNotNone(ax.get_legend())  # legend drawn on ax
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)

    @patch('matplotlib.pyplot.savefig')
    def test_visualize_differences_with_ax(self, mock_savefig):
        """Passing an ax draws onto it and does not save a figure."""
        fig, ax = plt.subplots()
        try:
            visualize_differences(ax=ax)
            self.assertGreater(len(ax.patches), 0)  # histogram bars drawn on ax
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)

    @patch('matplotlib.pyplot.savefig')
    def test_binding_boxplots_with_ax(self, mock_savefig):
        """Passing an ax draws onto it and does not save a figure."""
        fig, ax = plt.subplots()
        try:
            binding_boxplots(make_binding_boxplot_data(), "test", ax=ax)
            self.assertGreater(len(ax.patches) + len(ax.lines), 0)  # boxes drawn on ax
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)


class TestStarrseqVisualizationsStandalone(unittest.TestCase):
    """Standalone-mode tests: figures are created and saved when ax is None."""

    def setUp(self) -> None:
        """Patch load_df so no data file is read during the tests."""
        self.load_df_patcher = patch(
            'analysis.starrseq.starrseq_analysis.load_df',
            side_effect=make_mock_dataframe,
        )
        self.mock_load_df = self.load_df_patcher.start()

    def tearDown(self) -> None:
        """Stop patches and close any open figures."""
        self.load_df_patcher.stop()
        plt.close('all')

    @patch('matplotlib.pyplot.savefig')
    def test_visualize_simple_standalone_saves(self, mock_savefig):
        """Without an ax, a figure is saved (once per transcription factor)."""
        visualize_simple()
        self.assertEqual(mock_savefig.call_count, 2)

    @patch('matplotlib.pyplot.savefig')
    def test_visualize_simple_2_standalone_saves(self, mock_savefig):
        """Without an ax, exactly one figure is saved."""
        visualize_simple_2()
        mock_savefig.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    def test_visualize_differences_standalone_saves(self, mock_savefig):
        """Without an ax, a figure is saved (once per transcription factor)."""
        visualize_differences()
        self.assertEqual(mock_savefig.call_count, 2)

    @patch('matplotlib.pyplot.savefig')
    def test_binding_boxplots_standalone_saves(self, mock_savefig):
        """Without an ax, exactly one figure is saved."""
        binding_boxplots(make_binding_boxplot_data(), "test")
        mock_savefig.assert_called_once()


if __name__ == '__main__':
    unittest.main()
