"""Tests for the figure-composition helpers in ``compose_plots_mpl``.

Only the lightweight, data-driven helpers are exercised here; the ``_populate_*``
and ``figN`` functions run the full deepCRE prediction pipelines (TensorFlow
model loads, reference-window scans) and are not unit-tested.
"""

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from workflows.overlap_analysis._common import BINDING_STATUS_COLORS
from workflows.paper_plots.compose_plots_mpl import _draw_binding_status_boxplot


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


if __name__ == "__main__":
    unittest.main()
