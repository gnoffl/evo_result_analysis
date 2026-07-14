"""Unit tests for the publication styling primitives in ``paper_plots.style``."""

import unittest
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")  # headless backend for tests
import matplotlib.pyplot as plt

from workflows.paper_plots.style import (
    PUBLICATION_RC,
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
        mock_figure.savefig.assert_called_once_with(
            "out.svg", dpi=600, bbox_inches="tight", transparent=False
        )


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


if __name__ == "__main__":
    unittest.main()
