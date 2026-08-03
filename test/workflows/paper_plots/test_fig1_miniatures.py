"""Tests for the figure-1 inset miniatures in ``fig1_miniatures``.

The three miniature functions each wrap an expensive data source (a Pareto
front on disk, a large deepCIS scan CSV, or the full deepCRE/STARR-seq
prediction pipeline), so those sources are mocked/stubbed here; only the
plotting behaviour (what is/is not drawn) is under test.
"""

import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps

from workflows.paper_plots.fig1_miniatures import (
    _DEEPCIS_GENE,
    _DEEPCIS_TF,
    _FIG6C_MINI_CMAP,
    _FIG6C_MINI_COLOR_SHADE,
    deepcis_scan_mini,
    fig3g_mini,
    fig6c_mini,
    pareto_front_mini,
)

_MODULE = "workflows.paper_plots.fig1_miniatures"


class ParetoFrontMiniTest(unittest.TestCase):
    """Behaviour of the final-Pareto-front miniature."""

    def tearDown(self) -> None:
        plt.close("all")

    def test_scatters_final_front_with_labels_and_no_legend(self) -> None:
        # Arrange
        mutation_counts = [0.0, 2.0, 5.0]
        predictions = [0.2, 0.5, 0.9]

        # Act
        with patch(
            f"{_MODULE}.load_final_front",
            return_value=(mutation_counts, predictions),
        ), patch(f"{_MODULE}._save_mini_figure") as save_mini_figure:
            pareto_front_mini()

        # Assert
        save_mini_figure.assert_called_once()
        fig = save_mini_figure.call_args.args[0]
        ax = fig.axes[0]
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.collections[0].get_offsets()), 3)
        self.assertEqual(ax.get_xlabel(), "Mutation count")
        self.assertEqual(ax.get_ylabel(), "deepCRE prediction")
        self.assertEqual(ax.get_title(), "")
        self.assertIsNone(ax.get_legend())
        self.assertEqual(save_mini_figure.call_args.args[1], "pareto_front_mini.svg")


class DeepcisScanMiniTest(unittest.TestCase):
    """Behaviour of the deepCIS scan miniature."""

    def tearDown(self) -> None:
        plt.close("all")

    def _scan_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gene": [_DEEPCIS_GENE] * 4,
                "sequence_type": [
                    "reference",
                    "reference",
                    "optimized",
                    "optimized",
                ],
                "window_start": [0, 50, 0, 50],
                "window_end": [250, 300, 250, 300],
                "contains_padding": [False] * 4,
                _DEEPCIS_TF: [0.1, 0.5, 0.2, 0.6],
            }
        )

    def test_draws_curves_with_legend_but_no_markers(self) -> None:
        # Act
        with patch(f"{_MODULE}.pd.read_csv", return_value=self._scan_frame()), patch(
            f"{_MODULE}._save_mini_figure"
        ) as save_mini_figure:
            deepcis_scan_mini()

        # Assert: both curves drawn with a matching legend, TSS/TTS/mutation
        # markers absent.
        save_mini_figure.assert_called_once()
        fig = save_mini_figure.call_args.args[0]
        ax = fig.axes[0]
        labels = [
            line.get_label()
            for line in ax.get_lines()
            if not str(line.get_label()).startswith("_")
        ]
        self.assertEqual(labels, ["Reference", "Optimized"])
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(
            [text.get_text() for text in legend.get_texts()],
            ["Reference", "Optimized"],
        )
        self.assertEqual(legend._loc, 2)  # matplotlib's "upper left" location code
        self.assertFalse(legend.get_frame_on())
        self.assertFalse(ax.get_xgridlines()[0].get_visible())
        self.assertEqual(ax.get_ylabel(), "TF binding score")
        self.assertEqual(ax.get_ylim()[1], 1.3)
        self.assertEqual(
            save_mini_figure.call_args.args[1], "deepcis_scan_mini.svg"
        )

    def test_raises_when_gene_missing_from_scan(self) -> None:
        # Arrange
        scan_frame = pd.DataFrame({"gene": ["some_other_gene"]})

        # Act / Assert
        with patch(f"{_MODULE}.pd.read_csv", return_value=scan_frame):
            with self.assertRaises(ValueError):
                deepcis_scan_mini()


class Fig6cMiniTest(unittest.TestCase):
    """Behaviour of the figure-6-panel-C miniature."""

    def tearDown(self) -> None:
        plt.close("all")

    def test_scatters_only_highlight_points_without_legend(self) -> None:
        # Arrange: two disjoint point groups; only the second is "in the window".
        background_points = pd.DataFrame(
            {"prediction_mutated": [0.1, 0.2], "enrichment": [0.3, 0.4]}
        )
        highlight_points = pd.DataFrame(
            {"prediction_mutated": [0.5, 0.6, 0.7], "enrichment": [1.0, 1.2, 1.4]}
        )
        highlight_fit = (1.0, 0.5, 0.9, 0.01)

        # Act
        with patch(
            f"{_MODULE}.prepare_wrky_enrichment_df", return_value=pd.DataFrame()
        ), patch(
            f"{_MODULE}.compute_overlay_correlation_data",
            return_value=(background_points, highlight_points, None, highlight_fit),
        ), patch(f"{_MODULE}._save_mini_figure") as save_mini_figure:
            fig6c_mini()

        # Assert: only the 3 highlight points are scattered, and no legend.
        save_mini_figure.assert_called_once()
        fig = save_mini_figure.call_args.args[0]
        ax = fig.axes[0]
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.collections[0].get_offsets()), 3)
        self.assertEqual(ax.get_xlabel(), "deepCRE prediction")
        self.assertEqual(ax.get_ylabel(), "STARR-seq enrichment")
        self.assertIsNone(ax.get_legend())
        self.assertEqual(save_mini_figure.call_args.args[1], "fig6c_mini.svg")

    def test_scatter_and_fit_use_magma_shade(self) -> None:
        # Arrange
        highlight_points = pd.DataFrame(
            {"prediction_mutated": [0.5, 0.6], "enrichment": [1.0, 1.2]}
        )
        highlight_fit = (1.0, 0.5, 0.9, 0.01)
        expected_color = colormaps[_FIG6C_MINI_CMAP](_FIG6C_MINI_COLOR_SHADE)

        # Act
        with patch(
            f"{_MODULE}.prepare_wrky_enrichment_df", return_value=pd.DataFrame()
        ), patch(
            f"{_MODULE}.compute_overlay_correlation_data",
            return_value=(pd.DataFrame(), highlight_points, None, highlight_fit),
        ), patch(f"{_MODULE}._save_mini_figure") as save_mini_figure:
            fig6c_mini()

        # Assert: both the scatter facecolor and the fit line colour are the
        # same magma shade.
        fig = save_mini_figure.call_args.args[0]
        ax = fig.axes[0]
        scatter_color = tuple(ax.collections[0].get_facecolor()[0])
        self.assertEqual(scatter_color[:3], expected_color[:3])
        self.assertEqual(ax.get_lines()[0].get_color(), expected_color)


class Fig3gMiniTest(unittest.TestCase):
    """Behaviour of the figure-3-panel-G heatmap miniature."""

    def tearDown(self) -> None:
        plt.close("all")

    def test_keeps_only_top_bottom_extremes_without_stars_or_tnt_suffix(self) -> None:
        # Arrange: 6 TFs (3 favouring "ara max", 3 favouring "ara min"), all
        # reaching the three-star tier, so the ordering/filtering is trivial
        # and only the miniature's top/bottom-3 truncation is under test.
        tf_names = [f"TF{index}_tnt" for index in range(6)]
        significance = pd.DataFrame(
            {"tf": tf_names, "q_contrast": [0.0001] * 6}
        )
        matrix = pd.DataFrame(
            {
                "ara max": [5.0, 4.0, 3.0, -3.0, -4.0, -5.0],
                "GOF": [5.0, 4.0, 3.0, -3.0, -4.0, -5.0],
                "ara min": [-5.0, -4.0, -3.0, 3.0, 4.0, 5.0],
                "LOF": [-5.0, -4.0, -3.0, 3.0, 4.0, 5.0],
            },
            index=tf_names,
        )

        # Act
        with patch(
            f"{_MODULE}.paired_tf_significance", return_value=significance
        ), patch(f"{_MODULE}.build_matrix", return_value=matrix), patch(
            f"{_MODULE}._save_mini_figure"
        ) as save_mini_figure:
            fig3g_mini()

        # Assert: only 6 TFs kept (all of them, since there are exactly 3 on
        # each side already), the "_tnt" suffix stripped, no star
        # annotations drawn, only the two ara columns shown (relabelled
        # "max"/"min"), and no "run" x-axis label.
        save_mini_figure.assert_called_once()
        fig = save_mini_figure.call_args.args[0]
        ax = fig.axes[0]
        y_labels = [label.get_text() for label in ax.get_yticklabels()]
        self.assertEqual(y_labels, [f"TF{index}" for index in range(6)])
        x_labels = [label.get_text() for label in ax.get_xticklabels()]
        self.assertEqual(x_labels, ["max", "min"])
        self.assertEqual(ax.get_xlabel(), "")
        self.assertEqual(len(ax.texts), 0)
        self.assertEqual(fig.axes[-1].get_ylabel(), "introduction frequency")
        self.assertEqual(save_mini_figure.call_args.args[1], "fig3g_mini.svg")


if __name__ == "__main__":
    unittest.main()
