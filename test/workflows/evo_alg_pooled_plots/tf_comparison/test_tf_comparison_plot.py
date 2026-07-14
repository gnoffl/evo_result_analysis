"""Unit tests for the TF-comparison plotting toolbox."""

import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from workflows.evo_alg_pooled_plots.tf_comparison.tf_comparison_plot import (  # noqa: E402
    plot_heatmap,
    q_to_stars,
)


class QToStarsTest(unittest.TestCase):
    """Tests for the q-value to star mapping."""

    def test_thresholds_and_nan(self) -> None:
        # Arrange / Act / Assert
        self.assertEqual(q_to_stars(0.0005), "***")
        self.assertEqual(q_to_stars(0.005), "**")
        self.assertEqual(q_to_stars(0.02), "*")
        self.assertEqual(q_to_stars(0.2), "")
        self.assertEqual(q_to_stars(float("nan")), "")


class PlotHeatmapTest(unittest.TestCase):
    """Smoke test for figure creation."""

    def _matrix(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"MAX": {"WRKY": 1.0, "bHLH": -2.0}, "MIN": {"WRKY": -1.0, "bHLH": 2.0}}
        )

    def test_returns_figure(self) -> None:
        # Arrange
        matrix = self._matrix()

        # Act
        fig = plot_heatmap(matrix, annotate=True, separator_after_column=1)

        # Assert
        self.assertIsInstance(fig, Figure)

    def test_separator_drawn_when_within_range(self) -> None:
        # Arrange
        matrix = self._matrix()

        # Act
        fig = plot_heatmap(matrix, annotate=False, separator_after_column=1)

        # Assert: one vertical separator line was added at x=1.
        separators = [
            line for line in fig.axes[0].lines if list(line.get_xdata()) == [1, 1]
        ]
        self.assertEqual(len(separators), 1)

    def test_no_separator_when_none(self) -> None:
        # Arrange
        matrix = self._matrix()

        # Act
        fig = plot_heatmap(matrix, annotate=False, separator_after_column=None)

        # Assert: no full-height vertical divider line at an integer column.
        separators = [
            line
            for line in fig.axes[0].lines
            if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]
        ]
        self.assertEqual(separators, [])

    def test_row_stars_annotate_tf_labels(self) -> None:
        # Arrange
        matrix = self._matrix()

        # Act
        fig = plot_heatmap(
            matrix, annotate=True, row_stars={"WRKY": "** "}, separator_after_column=1
        )

        # Assert: the WRKY row label carries its stars, bHLH stays plain.
        labels = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
        self.assertIn("WRKY ** ", labels)
        self.assertIn("bHLH    ", labels)

    def test_cell_stars_embedded_in_annotation_text(self) -> None:
        # Arrange: WRKY in MAX is significant; bHLH in MIN is significant.
        matrix = self._matrix()
        cell_stars = {"MAX": {"WRKY": "*  "}, "MIN": {"bHLH": "** "}}

        # Act
        fig = plot_heatmap(
            matrix, annotate=True, cell_stars=cell_stars, separator_after_column=1
        )

        # Assert: cell texts contain both the numeric value and the star string.
        texts = [t.get_text() for t in fig.axes[0].texts]
        self.assertTrue(any("1.00*  " in t for t in texts), f"Expected '1.00*  ' in {texts}")
        self.assertTrue(any("2.00** " in t for t in texts), f"Expected '2.00** ' in {texts}")
        # Cells with no stars have just the number.
        self.assertTrue(any("-2.00" in t and "**" not in t for t in texts))

    def test_cell_stars_only_when_annotate_values_false(self) -> None:
        # Arrange: WRKY in MAX is significant; bHLH in MIN is significant.
        matrix = self._matrix()
        cell_stars = {"MAX": {"WRKY": "*"}, "MIN": {"bHLH": "**"}}

        # Act
        fig = plot_heatmap(
            matrix,
            annotate=True,
            cell_stars=cell_stars,
            separator_after_column=1,
            annotate_values=False,
        )

        # Assert: annotations are the bare stars, with no numeric value.
        texts = [t.get_text() for t in fig.axes[0].texts]
        self.assertIn("*", texts)
        self.assertIn("**", texts)
        self.assertFalse(
            any(any(char.isdigit() for char in t) for t in texts),
            f"Expected no numeric annotations, got {texts}",
        )

    def test_no_colorbar_when_add_colorbar_false(self) -> None:
        # Arrange
        matrix = self._matrix()

        # Act: draw onto a known single-axes figure with the colorbar suppressed.
        fig, ax = plt.subplots()
        try:
            plot_heatmap(
                matrix, annotate=False, separator_after_column=1, add_colorbar=False, ax=ax
            )

            # Assert: no extra colorbar axes was added, and the mappable is
            # reachable for a caller-managed colorbar.
            self.assertEqual(len(fig.axes), 1)
            self.assertGreater(len(ax.collections), 0)
        finally:
            plt.close(fig)

    @patch("matplotlib.pyplot.savefig")
    def test_plot_heatmap_with_ax(self, mock_savefig) -> None:
        # Arrange
        matrix = self._matrix()
        fig, ax = plt.subplots()

        # Act
        try:
            returned = plot_heatmap(matrix, annotate=True, separator_after_column=1, ax=ax)

            # Assert: drawn onto the provided ax, its figure returned, nothing saved.
            self.assertIs(returned, ax.get_figure())
            self.assertGreater(len(ax.collections), 0)  # heatmap mesh drawn
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
