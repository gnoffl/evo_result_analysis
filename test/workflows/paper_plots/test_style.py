"""Unit tests for the publication styling primitives in ``paper_plots.style``."""

import unittest
from unittest.mock import MagicMock, call

import matplotlib

matplotlib.use("Agg")  # headless backend for tests
import matplotlib.pyplot as plt

from workflows.paper_plots.style import (
    PUBLICATION_RC,
    broken_y_axes,
    figure_size_inches,
    mm_to_inch,
    panel_label,
    publication_style,
    save_publication_figure,
    sync_axis_limits,
)


class TestUnitConversion(unittest.TestCase):
    """Tests for millimetre-to-inch conversion helpers."""

    def test_mm_to_inch_known_value(self) -> None:
        # Arrange / Act
        result = mm_to_inch(25.4)

        # Assert
        self.assertAlmostEqual(result, 1.0)

    def test_figure_size_inches_returns_tuple(self) -> None:
        # Arrange / Act
        width_inch, height_inch = figure_size_inches(180.0, 90.0)

        # Assert
        self.assertAlmostEqual(width_inch, 180.0 / 25.4)
        self.assertAlmostEqual(height_inch, 90.0 / 25.4)


class TestPublicationStyle(unittest.TestCase):
    """Tests for the ``publication_style`` context manager."""

    def test_applies_rcparams_inside_context(self) -> None:
        # Arrange
        expected_font_size = PUBLICATION_RC["font.size"]

        # Act / Assert
        with publication_style():
            self.assertEqual(plt.rcParams["font.size"], expected_font_size)
            self.assertFalse(plt.rcParams["axes.spines.top"])

    def test_restores_rcparams_after_context(self) -> None:
        # Arrange
        original_font_size = plt.rcParams["font.size"]

        # Act
        with publication_style():
            pass

        # Assert
        self.assertEqual(plt.rcParams["font.size"], original_font_size)

    def test_extra_rc_overrides_defaults(self) -> None:
        # Arrange / Act / Assert
        with publication_style(extra_rc={"font.size": 20.0}):
            self.assertEqual(plt.rcParams["font.size"], 20.0)


class TestPanelLabel(unittest.TestCase):
    """Tests for the ``panel_label`` helper."""

    def test_adds_text_with_label(self) -> None:
        # Arrange
        fig, ax = plt.subplots()
        try:
            # Act
            text_artist = panel_label(ax, "A")

            # Assert
            self.assertEqual(text_artist.get_text(), "A")
            self.assertEqual(text_artist.get_fontweight(), "bold")
            self.assertIn(text_artist, ax.texts)
        finally:
            plt.close(fig)


class TestSavePublicationFigure(unittest.TestCase):
    """Tests for ``save_publication_figure`` (savefig is mocked)."""

    def test_forwards_expected_savefig_arguments(self) -> None:
        # Arrange
        mock_figure = MagicMock()

        # Act
        save_publication_figure(mock_figure, "out.svg")

        # Assert
        mock_figure.savefig.assert_has_calls(
            [
                call("out.svg", dpi=600, bbox_inches="tight", transparent=False),
                call("out.png", dpi=600, bbox_inches="tight", transparent=False),
            ]
        )
        self.assertEqual(mock_figure.savefig.call_count, 2)


class TestSyncAxisLimits(unittest.TestCase):
    """Tests for the ``sync_axis_limits`` helper."""

    def test_syncs_x_and_y_to_union(self) -> None:
        # Arrange
        fig, (ax_a, ax_b) = plt.subplots(1, 2)
        try:
            ax_a.set_xlim(0.0, 5.0)
            ax_a.set_ylim(0.0, 2.0)
            ax_b.set_xlim(1.0, 8.0)
            ax_b.set_ylim(-1.0, 1.0)

            # Act
            sync_axis_limits([ax_a, ax_b])

            # Assert: union of both ranges on both axes
            for ax in (ax_a, ax_b):
                self.assertEqual(ax.get_xlim(), (0.0, 8.0))
                self.assertEqual(ax.get_ylim(), (-1.0, 2.0))
        finally:
            plt.close(fig)

    def test_sync_x_only_leaves_y_untouched(self) -> None:
        # Arrange
        fig, (ax_a, ax_b) = plt.subplots(1, 2)
        try:
            ax_a.set_xlim(0.0, 5.0)
            ax_a.set_ylim(0.0, 2.0)
            ax_b.set_xlim(1.0, 8.0)
            ax_b.set_ylim(-1.0, 1.0)

            # Act
            sync_axis_limits([ax_a, ax_b], sync_y=False)

            # Assert
            self.assertEqual(ax_a.get_xlim(), (0.0, 8.0))
            self.assertEqual(ax_b.get_xlim(), (0.0, 8.0))
            self.assertEqual(ax_a.get_ylim(), (0.0, 2.0))
            self.assertEqual(ax_b.get_ylim(), (-1.0, 1.0))
        finally:
            plt.close(fig)

    def test_single_axis_is_noop(self) -> None:
        # Arrange
        fig, ax = plt.subplots()
        try:
            ax.set_xlim(2.0, 3.0)

            # Act
            sync_axis_limits([ax])

            # Assert
            self.assertEqual(ax.get_xlim(), (2.0, 3.0))
        finally:
            plt.close(fig)


class TestBrokenYAxes(unittest.TestCase):
    """Tests for splitting a grid cell into a broken-y axes pair."""

    def setUp(self) -> None:
        self.fig = plt.figure()
        self.grid = self.fig.add_gridspec(nrows=1, ncols=1)

    def tearDown(self) -> None:
        plt.close(self.fig)

    def test_returns_pair_with_requested_limits(self) -> None:
        # Act
        upper_ax, lower_ax = broken_y_axes(
            self.fig, self.grid[0, 0], (0.0, 20.0), (50.0, 60.0)
        )

        # Assert
        self.assertEqual(lower_ax.get_ylim(), (0.0, 20.0))
        self.assertEqual(upper_ax.get_ylim(), (50.0, 60.0))
        self.assertGreater(
            upper_ax.get_position().y0, lower_ax.get_position().y0
        )

    def test_limits_survive_later_plotting(self) -> None:
        # Arrange
        upper_ax, lower_ax = broken_y_axes(
            self.fig, self.grid[0, 0], (0.0, 20.0), (50.0, 60.0)
        )

        # Act — a bar taller than the lower range must not rescale the axes.
        lower_ax.bar([1.0], [60.0])
        upper_ax.bar([1.0], [60.0])

        # Assert
        self.assertEqual(lower_ax.get_ylim(), (0.0, 20.0))
        self.assertEqual(upper_ax.get_ylim(), (50.0, 60.0))

    def test_hides_facing_spines_and_upper_tick_labels(self) -> None:
        # Act
        upper_ax, lower_ax = broken_y_axes(
            self.fig, self.grid[0, 0], (0.0, 20.0), (50.0, 60.0)
        )

        # Assert
        self.assertFalse(upper_ax.spines["bottom"].get_visible())
        self.assertFalse(lower_ax.spines["top"].get_visible())
        self.assertEqual(
            [label.get_text() for label in upper_ax.get_xticklabels()
             if label.get_visible()],
            [],
        )

    def test_break_marks_follow_visible_spines(self) -> None:
        # Arrange — hiding the right spine must drop the right-hand marks.
        upper_ax, lower_ax = broken_y_axes(
            self.fig, self.grid[0, 0], (0.0, 20.0), (50.0, 60.0)
        )
        marks_with_right_spine = len(lower_ax.get_lines()[0].get_xdata())
        plt.close(self.fig)
        self.fig = plt.figure()
        self.grid = self.fig.add_gridspec(nrows=1, ncols=1)

        # Act
        with plt.rc_context({"axes.spines.right": False}):
            _, lower_ax_without_right = broken_y_axes(
                self.fig, self.grid[0, 0], (0.0, 20.0), (50.0, 60.0)
            )

        # Assert
        self.assertEqual(marks_with_right_spine, 2)
        self.assertEqual(
            len(lower_ax_without_right.get_lines()[0].get_xdata()), 1
        )

    def test_break_marks_excluded_from_layout(self) -> None:
        # Arrange / Act
        upper_ax, lower_ax = broken_y_axes(
            self.fig, self.grid[0, 0], (0.0, 20.0), (50.0, 60.0)
        )

        # Assert — marks straddling the cut must not reserve layout space, which
        # would push the two halves apart again.
        self.assertFalse(upper_ax.get_lines()[0].get_in_layout())
        self.assertFalse(lower_ax.get_lines()[0].get_in_layout())

    def test_overlapping_ranges_raise(self) -> None:
        # Act / Assert
        with self.assertRaises(ValueError):
            broken_y_axes(self.fig, self.grid[0, 0], (0.0, 20.0), (10.0, 60.0))


if __name__ == "__main__":
    unittest.main()
