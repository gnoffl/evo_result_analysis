"""Ax-injection tests for the plotting leaf drawers / wrappers in ``_common``.

Each test checks that passing an ``ax`` draws onto that axes and never saves a
figure (``Figure.savefig`` is patched and asserted not called).
"""

import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless backend for tests
import matplotlib.pyplot as plt
import pandas as pd

from workflows.overlap_analysis._common import (
    _plot_bucketed_correlation,
    _plot_individual_bucket_correlation,
    _plot_overlay_correlation,
    plot_overlay_highlight_correlation,
    simply_plot_multi,
)


class TestCommonAxInjection(unittest.TestCase):
    """Passing ``ax`` draws onto it and never calls ``Figure.savefig``."""

    def _correlation_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "prediction_mutated": [0.1, 0.2, 0.3, 0.4, 0.5],
                "enrichment": [1.0, 1.4, 1.6, 2.1, 2.3],
                "group_overlap_start": [900, 950, 1000, 1050, 1100],
            }
        )

    @patch("matplotlib.figure.Figure.savefig")
    def test_simply_plot_multi_with_ax(self, mock_savefig):
        fig, ax = plt.subplots()
        try:
            series = [([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4], "series-a", "tab:blue")]
            simply_plot_multi(
                series, "x", "y", "title", "out", "subfolder", ax=ax
            )
            self.assertGreater(len(ax.lines) + len(ax.collections), 0)
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)

    @patch("matplotlib.figure.Figure.savefig")
    def test_plot_bucketed_correlation_with_ax(self, mock_savefig):
        fig, ax = plt.subplots()
        try:
            _plot_bucketed_correlation(
                self._correlation_df(),
                "prediction_mutated",
                "enrichment",
                "out.png",
                "title",
                "x label",
                "y label",
                "analysis",
                include_differences=False,
                use_buckets=False,
                ax=ax,
            )
            self.assertGreater(len(ax.collections), 0)  # scatter drawn
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)

    @patch("matplotlib.figure.Figure.savefig")
    def test_plot_individual_bucket_correlation_with_ax(self, mock_savefig):
        fig, ax = plt.subplots()
        try:
            data = pd.DataFrame(
                {
                    "overlap_bucket_label": ["b", "b", "b", "b"],
                    "prediction_mutated": [0.1, 0.2, 0.3, 0.4],
                    "enrichment": [1.0, 1.5, 1.7, 2.2],
                }
            )
            _plot_individual_bucket_correlation(
                data,
                "analysis",
                "prediction_mutated",
                "enrichment",
                "out.png",
                "title",
                "x label",
                "y label",
                "b",
                include_differences=False,
                ax=ax,
            )
            self.assertGreater(len(ax.collections), 0)
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)

    @patch("matplotlib.figure.Figure.savefig")
    def test_plot_overlay_correlation_with_ax(self, mock_savefig):
        fig, ax = plt.subplots()
        try:
            _plot_overlay_correlation(
                self._correlation_df(),
                "prediction_mutated",
                "enrichment",
                1000,
                "out.png",
                "title",
                "x label",
                "y label",
                "analysis",
                ax=ax,
            )
            self.assertGreater(len(ax.collections), 0)
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)

    @patch("matplotlib.figure.Figure.savefig")
    def test_plot_overlay_highlight_correlation_with_ax(self, mock_savefig):
        fig, ax = plt.subplots()
        try:
            plot_overlay_highlight_correlation(
                self._correlation_df(), window_center=1000, ax=ax
            )
            self.assertGreater(len(ax.collections), 0)
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
