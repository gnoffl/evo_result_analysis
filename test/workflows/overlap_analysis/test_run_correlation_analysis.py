"""Unit tests for the shared ``_common.run_correlation_analysis`` stage.

Every plotting routine and the statistics writer are mocked so the test only
verifies the orchestration: which subsets are produced and that each plotting
routine and the bucket-statistics writer are invoked the expected number of
times. No figures are rendered and nothing is written to disk.
"""
import unittest
from unittest.mock import patch

import pandas as pd

from workflows.overlap_analysis import _common as overlap_analysis


class TestRunCorrelationAnalysis(unittest.TestCase):
    @staticmethod
    def _sample_df() -> pd.DataFrame:
        """Two-condition df with both reference/synthetic and binding states."""
        return pd.DataFrame(
            [
                {"condition": "light", "starr_reference": True, "starr_binding_status": "binding"},
                {"condition": "dark", "starr_reference": False, "starr_binding_status": "non_binding"},
            ]
        )

    def test_emits_all_subsets_when_both_conditions_present(self):
        # Arrange: plot routines return empty stat-row lists so ``extend`` works.
        df = self._sample_df()
        patches = {
            name: patch.object(overlap_analysis, name, return_value=[])
            for name in (
                "plot_deepcre_starrseq_correlation",
                "plot_mutation_starrseq_correlation",
                "plot_mutation_deepcre_correlation",
            )
        }
        with patches["plot_deepcre_starrseq_correlation"] as mock_deepcre, \
                patches["plot_mutation_starrseq_correlation"] as mock_mut_starr, \
                patches["plot_mutation_deepcre_correlation"] as mock_mut_deepcre, \
                patch.object(overlap_analysis, "plot_individual_bucket_views") as mock_bucket_views, \
                patch.object(overlap_analysis, "plot_overlay_highlight_correlation") as mock_overlay, \
                patch.object(overlap_analysis, "plot_correlation_over_positions_fixed_window") as mock_pos_window, \
                patch.object(overlap_analysis, "plot_correlation_over_positions_fixed_number_elements") as mock_pos_elems, \
                patch.object(overlap_analysis, "save_bucket_statistics") as mock_save:
            # Act
            overlap_analysis.run_correlation_analysis(df)

        # Assert: position-series plots run once over the condition-split series.
        mock_pos_window.assert_called_once()
        mock_pos_elems.assert_called_once()
        # The overlay plot runs for all + light + dark, each in a single-color
        # and a binding-status-colored variant: 3 subsets x 2 variants.
        self.assertEqual(mock_overlay.call_count, 6)
        overlay_calls = {
            (call.kwargs.get("subset_label"), call.kwargs.get("color_by_binding", False))
            for call in mock_overlay.call_args_list
        }
        self.assertEqual(
            overlay_calls,
            {
                ("", False),
                ("", True),
                ("light", False),
                ("light", True),
                ("dark", False),
                ("dark", True),
            },
        )
        self.assertEqual(len(mock_pos_window.call_args[0][0]), 3)  # all, light, dark

        # Seven subsets: "", reference, synthetic, binding, non_binding, light, dark.
        expected_subsets = 7
        self.assertEqual(mock_deepcre.call_count, expected_subsets)
        self.assertEqual(mock_mut_starr.call_count, expected_subsets)
        self.assertEqual(mock_mut_deepcre.call_count, expected_subsets)
        self.assertEqual(mock_save.call_count, expected_subsets)
        # One individual-bucket view per subset per configured bucket label.
        self.assertEqual(
            mock_bucket_views.call_count,
            expected_subsets * len(overlap_analysis.INDIVIDUAL_BUCKET_LABELS_TO_PLOT),
        )

    def test_single_condition_skips_light_dark_subsets(self):
        # Arrange: only one condition value present.
        df = pd.DataFrame(
            [
                {"condition": "light", "starr_reference": True, "starr_binding_status": "binding"},
                {"condition": "light", "starr_reference": False, "starr_binding_status": "non_binding"},
            ]
        )
        with patch.object(overlap_analysis, "plot_deepcre_starrseq_correlation", return_value=[]) as mock_deepcre, \
                patch.object(overlap_analysis, "plot_mutation_starrseq_correlation", return_value=[]), \
                patch.object(overlap_analysis, "plot_mutation_deepcre_correlation", return_value=[]), \
                patch.object(overlap_analysis, "plot_individual_bucket_views"), \
                patch.object(overlap_analysis, "plot_overlay_highlight_correlation") as mock_overlay, \
                patch.object(overlap_analysis, "plot_correlation_over_positions_fixed_window") as mock_pos_window, \
                patch.object(overlap_analysis, "plot_correlation_over_positions_fixed_number_elements"), \
                patch.object(overlap_analysis, "save_bucket_statistics"):
            # Act
            overlap_analysis.run_correlation_analysis(df)

        # Assert: no light/dark subsets, and the position series is unsplit.
        self.assertEqual(mock_deepcre.call_count, 5)  # "", reference, synthetic, binding, non_binding
        self.assertEqual(len(mock_pos_window.call_args[0][0]), 1)  # all only
        # Overlay only for the full dataset: single-color + binding-status variant.
        self.assertEqual(mock_overlay.call_count, 2)
        overlay_subset_labels = {
            call.kwargs.get("subset_label") for call in mock_overlay.call_args_list
        }
        self.assertEqual(overlay_subset_labels, {""})


if __name__ == "__main__":
    unittest.main()
