"""Unit tests for the runtime-benchmark plotting module."""

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_hex, to_rgba  # noqa: E402

from workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark_calc import (  # noqa: E402
    fit_log_log_scaling,
)
from workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark_plot import (  # noqa: E402
    DEVICE_PALETTE,
    REGIME_PALETTE,
    color_box_outlines,
    plot_runtime_sweep,
    plot_scaling_fit,
)


def make_sweep_frame(
    swept_parameter: str = "population_size", hue_column: str = "regime"
) -> pd.DataFrame:
    """Build a small tidy runtime frame covering two hue levels and three cells.

    Args:
        swept_parameter: Value to put into the ``swept_parameter`` column.
        hue_column: Column to fill with two levels, ``"regime"`` or ``"device"``.

    Returns:
        Tidy frame with three sequences per cell.
    """
    levels = (
        ["unconstrained", "natural"] if hue_column == "regime" else ["CPU", "GPU"]
    )
    rows = []
    for level_index, level in enumerate(levels):
        for swept_value in (25, 50, 100):
            for sequence_index in range(3):
                rows.append(
                    {
                        "algorithm": "ga",
                        "device": level if hue_column == "device" else "CPU",
                        "regime": level if hue_column == "regime" else "unconstrained",
                        "swept_parameter": swept_parameter,
                        "swept_value": swept_value,
                        "n_evaluations": swept_value * 1001,
                        "sequence_name": f"benchmark_{sequence_index:02d}",
                        "algorithm_seconds": (level_index + 1)
                        * swept_value
                        * (1.0 + 0.05 * sequence_index),
                    }
                )
    return pd.DataFrame(rows)


class PlotRuntimeSweepTest(unittest.TestCase):
    """Tests for the sweep boxplot."""

    def test_returns_axes_with_log_y_and_expected_labels(self) -> None:
        # Arrange
        frame = make_sweep_frame()
        # Act
        ax = plot_runtime_sweep(frame)
        # Assert
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(ax.get_xlabel(), "Population size")
        self.assertEqual([label.get_text() for label in ax.get_xticklabels()],
                         ["25", "50", "100"])
        plt.close(ax.get_figure())

    def test_palette_argument_overrides_the_module_colours(self) -> None:
        # Arrange
        frame = make_sweep_frame()
        fits = {
            regime: fit_log_log_scaling(
                group["n_evaluations"], group["algorithm_seconds"]
            )
            for regime, group in frame.groupby("regime")
        }
        palette = {"unconstrained": "#641a80", "natural": "#f9795d"}
        # Act
        ax = plot_scaling_fit(frame, fits=fits, palette=palette)
        # Assert
        self.assertEqual(
            {to_hex(line.get_color()) for line in ax.lines}, set(palette.values())
        )
        plt.close(ax.get_figure())

    def test_saves_file_when_output_dir_given(self) -> None:
        # Arrange
        frame = make_sweep_frame()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            # Act
            plot_runtime_sweep(
                frame, output_dir=output_dir, file_stem="sweep_test", fmt="png"
            )
            # Assert
            self.assertTrue((output_dir / "sweep_test.png").is_file())

    def test_injected_axes_are_not_saved_or_closed(self) -> None:
        # Arrange
        frame = make_sweep_frame()
        figure, axes = plt.subplots()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            # Act
            returned = plot_runtime_sweep(frame, output_dir=output_dir, ax=axes)
            # Assert
            self.assertIs(returned, axes)
            self.assertEqual(list(output_dir.iterdir()), [])
            self.assertTrue(plt.fignum_exists(figure.number))
        plt.close(figure)

    def test_hue_by_device(self) -> None:
        # Arrange
        frame = make_sweep_frame(hue_column="device")
        # Act
        ax = plot_runtime_sweep(frame, hue_column="device")
        # Assert
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(
            [text.get_text() for text in legend.get_texts()], ["CPU", "GPU"]
        )
        plt.close(ax.get_figure())

    def test_legend_suppressed_for_panel_use(self) -> None:
        # Arrange
        frame = make_sweep_frame()
        figure, axes = plt.subplots()
        # Act
        plot_runtime_sweep(frame, show_legend=False, show_title=False, ax=axes)
        # Assert
        self.assertIsNone(axes.get_legend())
        self.assertEqual(axes.get_title(), "")
        plt.close(figure)

    def test_empty_frame_raises(self) -> None:
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            plot_runtime_sweep(make_sweep_frame().iloc[0:0])

    def test_ambiguous_swept_parameter_raises(self) -> None:
        # Arrange
        frame = pd.concat(
            [make_sweep_frame("population_size"), make_sweep_frame("number_of_generations")]
        )
        # Act / Assert
        with self.assertRaises(ValueError):
            plot_runtime_sweep(frame)

    def test_palettes_cover_both_hue_columns(self) -> None:
        # Arrange / Act / Assert
        self.assertEqual(set(REGIME_PALETTE), {"unconstrained", "natural"})
        self.assertEqual(set(DEVICE_PALETTE), {"CPU", "GPU"})


class ColorBoxOutlinesTest(unittest.TestCase):
    """Tests for painting each box's line artists in its own hue."""

    def test_every_box_line_takes_its_box_colour(self) -> None:
        # Arrange: a collapsed cell would be unreadable in seaborn's default grey,
        # so the outline must end up in the box's own colour
        frame = make_sweep_frame()
        ax = plot_runtime_sweep(frame, color_outlines=False, show_points=False)
        box_colours = {
            tuple(round(channel, 4) for channel in patch.get_facecolor()[:3])
            for patch in ax.patches
        }
        # Act
        color_box_outlines(ax)
        # Assert: seaborn leaves an empty Line2D per box for the suppressed
        # fliers, which carries no position and is left alone
        drawn_lines = [line for line in ax.lines if len(line.get_xdata()) > 0]
        line_colours = {
            tuple(round(channel, 4) for channel in to_rgba(line.get_color())[:3])
            for line in drawn_lines
        }
        self.assertTrue(line_colours)
        self.assertTrue(line_colours.issubset(box_colours))
        # six boxes (three levels x two regimes), each with a median, two
        # whiskers and two caps
        self.assertEqual(len(drawn_lines), 30)
        plt.close(ax.get_figure())

    def test_fill_is_lightened_and_edge_is_opaque(self) -> None:
        # Arrange
        frame = make_sweep_frame()
        ax = plot_runtime_sweep(frame, color_outlines=False, show_points=False)
        # Act
        color_box_outlines(ax)
        # Assert
        for patch in ax.patches:
            self.assertLess(patch.get_facecolor()[3], 1.0)
            self.assertEqual(patch.get_edgecolor()[3], 1.0)
            self.assertEqual(
                tuple(round(channel, 6) for channel in patch.get_facecolor()[:3]),
                tuple(round(channel, 6) for channel in patch.get_edgecolor()[:3]),
            )
        plt.close(ax.get_figure())

    def test_axes_without_boxes_is_left_alone(self) -> None:
        # Arrange
        figure, axes = plt.subplots()
        axes.plot([1, 2, 3], [1, 2, 3], color="black")
        # Act
        color_box_outlines(axes)
        # Assert
        self.assertEqual(to_rgba(axes.lines[0].get_color()), to_rgba("black"))
        plt.close(figure)

    def test_replicate_dots_are_not_recoloured(self) -> None:
        # Arrange: the dots are drawn as collections, not lines, and must stay black
        frame = make_sweep_frame()
        # Act
        ax = plot_runtime_sweep(frame, color_outlines=True, show_points=True)
        # Assert
        self.assertTrue(ax.collections)
        for collection in ax.collections:
            colours = collection.get_facecolor()
            if len(colours) == 0:
                continue
            self.assertEqual(tuple(round(channel, 4) for channel in colours[0][:3]),
                             (0.0, 0.0, 0.0))
        plt.close(ax.get_figure())


class PlotScalingFitTest(unittest.TestCase):
    """Tests for the log-log scaling plot."""

    def test_draws_one_line_per_fitted_level(self) -> None:
        # Arrange
        frame = make_sweep_frame()
        fits = {
            regime: fit_log_log_scaling(group["n_evaluations"], group["algorithm_seconds"])
            for regime, group in frame.groupby("regime")
        }
        # Act
        ax = plot_scaling_fit(frame, fits=fits)
        # Assert
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(len(ax.lines), 2)
        legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
        self.assertTrue(any("exponent" in text for text in legend_texts))
        plt.close(ax.get_figure())

    def test_level_without_fit_gets_points_but_no_line(self) -> None:
        # Arrange
        frame = make_sweep_frame()
        fits = {
            "unconstrained": fit_log_log_scaling(
                frame.loc[frame["regime"] == "unconstrained", "n_evaluations"],
                frame.loc[frame["regime"] == "unconstrained", "algorithm_seconds"],
            )
        }
        # Act
        ax = plot_scaling_fit(frame, fits=fits)
        # Assert
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(len(ax.collections), 2)
        plt.close(ax.get_figure())

    def test_saves_file_when_output_dir_given(self) -> None:
        # Arrange
        frame = make_sweep_frame()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            # Act
            plot_scaling_fit(
                frame, fits={}, output_dir=output_dir, file_stem="scaling_test"
            )
            # Assert
            self.assertTrue((output_dir / "scaling_test.png").is_file())

    def test_non_positive_values_raise(self) -> None:
        # Arrange
        frame = make_sweep_frame()
        frame["algorithm_seconds"] = 0.0
        # Act / Assert
        with self.assertRaises(ValueError):
            plot_scaling_fit(frame, fits={})


if __name__ == "__main__":
    unittest.main()
