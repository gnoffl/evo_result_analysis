"""Tests for the figure-composition helpers in ``compose_plots_mpl``.

Only the lightweight, data-driven helpers are exercised here; the ``_populate_*``
and ``figN`` functions run the full deepCRE prediction pipelines (TensorFlow
model loads, reference-window scans) and are not unit-tested.
"""

import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from workflows.overlap_analysis._common import BINDING_STATUS_COLORS
from workflows.paper_plots.compose_plots_mpl import (
    GOF_UNCONSTRAINED_RUN_DIR,
    GOF_VCF_DIR,
    LOF_UNCONSTRAINED_RUN_DIR,
    LOF_VCF_DIR,
    _actual_mutation_allowance_dataframe,
    _draw_binding_status_boxplot,
    _move_legend_below,
)


class DrawBindingStatusBoxplotTest(unittest.TestCase):
    """Behaviour of the panel-D binding-status enrichment boxplot helper."""

    def setUp(self) -> None:
        self.figure, self.ax = plt.subplots()

    def tearDown(self) -> None:
        plt.close(self.figure)

    def _highlight_points(self) -> pd.DataFrame:
        """Return a small highlight-window frame with both binding groups."""
        return pd.DataFrame(
            {
                "enrichment": [0.5, 1.2, 0.9, -0.3, -0.1, 0.2],
                "starr_binding_status": [
                    "binding",
                    "binding",
                    "binding",
                    "non_binding",
                    "non_binding",
                    "non_binding",
                ],
            }
        )

    def test_draws_two_boxes_with_pretty_labels(self) -> None:
        # Arrange
        points = self._highlight_points()

        # Act
        _draw_binding_status_boxplot(self.ax, points)

        # Assert: a box was drawn per binding group and the axis is labelled.
        self.assertGreater(len(self.ax.patches), 0)
        self.assertEqual(self.ax.get_ylabel(), "STARR-seq Enrichment")
        self.assertEqual(self.ax.get_xlabel(), "")
        self.assertEqual(
            [label.get_text() for label in self.ax.get_xticklabels()],
            ["Binding", "Non-binding"],
        )

    def test_annotates_sample_sizes_per_group(self) -> None:
        # Arrange
        points = self._highlight_points()

        # Act
        _draw_binding_status_boxplot(self.ax, points)

        # Assert: one "n=<count>" annotation per group with the right counts.
        annotations = {
            text.get_text()
            for text in self.ax.texts
            if text.get_text().startswith("n=")
        }
        self.assertEqual(annotations, {"n=3"})

    def test_uses_shared_binding_status_palette(self) -> None:
        # Arrange
        points = self._highlight_points()

        # Act
        _draw_binding_status_boxplot(self.ax, points)

        # Assert: the box facecolors match the shared BINDING_STATUS_COLORS.
        expected = {
            matplotlib.colors.to_hex(BINDING_STATUS_COLORS["binding"]),
            matplotlib.colors.to_hex(BINDING_STATUS_COLORS["non_binding"]),
        }
        drawn = {
            matplotlib.colors.to_hex(patch.get_facecolor())
            for patch in self.ax.patches
        }
        self.assertTrue(expected.issubset(drawn))

    def test_omits_absent_binding_group(self) -> None:
        # Arrange: only binding points present.
        points = self._highlight_points()
        points = points[points["starr_binding_status"] == "binding"].copy()

        # Act
        _draw_binding_status_boxplot(self.ax, points)

        # Assert: a single box labelled "Binding".
        self.assertEqual(
            [label.get_text() for label in self.ax.get_xticklabels()],
            ["Binding"],
        )


class ActualMutationAllowanceDataframeTest(unittest.TestCase):
    """Behaviour of the figure-4 panel-J data assembly helper."""

    def test_counts_both_gene_sets_from_their_own_run_and_vcf_dirs(self) -> None:
        # Arrange: stub the scanning helpers so no run directory is touched.
        gof_counts = pd.DataFrame({"gene": ["g1"], "region": ["promoter"]})
        lof_counts = pd.DataFrame({"gene": ["g2"], "region": ["terminator"]})
        combined = pd.DataFrame({"gene": ["g1", "g2"], "percent_allowed": [10.0, 20.0]})
        module = "workflows.paper_plots.compose_plots_mpl"

        with patch(f"{module}.load_vcf_positions_by_gene") as load_positions, patch(
            f"{module}.collect_actual_mutation_region_counts"
        ) as collect_counts, patch(f"{module}.build_group_dataframe") as build_frame:
            load_positions.side_effect = lambda vcf_dir: {"positions": vcf_dir}
            collect_counts.side_effect = [gof_counts, lof_counts]
            build_frame.return_value = combined

            # Act
            result = _actual_mutation_allowance_dataframe()

        # Assert: GOF then LOF, each with its own unconstrained run and VCF dir.
        self.assertIs(result, combined)
        self.assertEqual(
            [call.args[0] for call in collect_counts.call_args_list],
            [GOF_UNCONSTRAINED_RUN_DIR, LOF_UNCONSTRAINED_RUN_DIR],
        )
        self.assertEqual(
            [call.args[0] for call in load_positions.call_args_list],
            [GOF_VCF_DIR, LOF_VCF_DIR],
        )
        build_frame.assert_called_once_with(gof_counts, lof_counts)


class MoveLegendBelowTest(unittest.TestCase):
    """Behaviour of the bottom-row legend repositioning helper."""

    def setUp(self) -> None:
        self.figure, self.ax = plt.subplots()

    def tearDown(self) -> None:
        plt.close(self.figure)

    def test_redraws_legend_below_axes_in_one_row(self) -> None:
        # Arrange: two labelled artists and a titled inside-the-axes legend.
        self.ax.plot([0, 1], [0, 1], label="first")
        self.ax.plot([0, 1], [1, 0], label="second")
        self.ax.legend(loc="upper right", title="Condition")

        # Act
        _move_legend_below(self.ax)

        # Assert: one frameless untitled legend below the axes, all entries kept.
        legend = self.ax.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(legend.get_title().get_text(), "")
        self.assertEqual(
            [text.get_text() for text in legend.get_texts()], ["first", "second"]
        )
        self.assertFalse(legend.get_frame_on())
        self.assertLess(legend.get_bbox_to_anchor().y1, 0)

    def test_no_legend_is_left_alone(self) -> None:
        # Act / Assert: no legend to move, and none created.
        _move_legend_below(self.ax)
        self.assertIsNone(self.ax.get_legend())


if __name__ == "__main__":
    unittest.main()
