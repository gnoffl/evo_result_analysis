"""Tests for the supplementary figure composition in ``supplementary_plots_mpl``.

The heavy inputs (per-run deepCIS scans and the significance statistics) are
mocked, so the tests exercise only the composition logic: TF-name shortening,
significance filtering, and the drawing of the single-panel heatmap onto a
caller-provided axes.
"""

import unittest
from unittest.mock import patch

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from workflows.evo_alg_pooled_plots.tf_comparison.tf_comparison_calc import (  # noqa: E402
    TF_COLUMN,
)
from workflows.paper_plots.supplementary_plots_mpl import (  # noqa: E402
    _ADVERSARIAL_COLUMN_TITLES,
    _ADVERSARIAL_RUNS,
    _REGIME_COLORS,
    _cpu_benchmark_runtimes,
    _draw_mut5_tf_heatmap,
    _draw_runtime_scaling_panel,
    _mut5_cell_stars,
    _populate_fig_s2,
    _populate_fig_s3,
    _regime_scaling_fits,
    _set_sweep_ticks,
    _short_tf_name,
    _significant_mut5_matrix,
)

_MODULE = "workflows.paper_plots.supplementary_plots_mpl"

# Four TF families: two significant and max-favoured, one significant and
# min-favoured, one not significant (must be dropped).
_SIGNIFICANCE = pd.DataFrame(
    {
        TF_COLUMN: ["BBRBPC_tnt", "ND_tnt", "bZIP_tnt", "MYB_tnt"],
        "q_contrast": [0.0001, 0.02, 0.03, 0.4],
        "q_a": [0.0005, 0.2, 0.6, 0.7],
        "q_b": [0.0005, 0.008, 0.6, 0.7],
    }
)

_MATRIX = pd.DataFrame(
    {
        "ara max": [0.05, 0.03, -0.02, 0.01],
        "GOF": [0.06, 0.04, -0.03, 0.01],
        "ara min": [-0.04, -0.03, 0.02, 0.00],
        "LOF": [-0.05, -0.02, 0.03, 0.00],
    },
    index=["BBRBPC_tnt", "ND_tnt", "bZIP_tnt", "MYB_tnt"],
)


class ShortTfNameTest(unittest.TestCase):
    """The ``_tnt`` suffix is stripped; other motif-set suffixes are kept."""

    def test_strips_tnt_suffix(self) -> None:
        self.assertEqual(_short_tf_name("BBRBPC_tnt"), "BBRBPC")

    def test_keeps_other_suffixes(self) -> None:
        self.assertEqual(_short_tf_name("Homeobox_ecoli"), "Homeobox_ecoli")

    def test_keeps_unsuffixed_name(self) -> None:
        self.assertEqual(_short_tf_name("bHLH"), "bHLH")


class Mut5CellStarsTest(unittest.TestCase):
    """The intra-run stars combine the paired columns with the single runs."""

    def test_builds_stars_for_all_four_runs(self) -> None:
        intra = pd.DataFrame(
            {TF_COLUMN: ["BBRBPC_tnt", "ND_tnt"], "q_intra": [0.0001, 0.6]}
        )
        with patch(f"{_MODULE}.single_run_tf_significance", return_value=intra):
            cell_stars = _mut5_cell_stars(_SIGNIFICANCE)

        self.assertEqual(
            set(cell_stars), {"ara max", "ara min", "GOF", "LOF"}
        )
        self.assertEqual(cell_stars["ara max"]["BBRBPC_tnt"], "***")
        self.assertEqual(cell_stars["ara min"]["ND_tnt"], "**")
        self.assertEqual(cell_stars["ara max"]["bZIP_tnt"], "")
        self.assertEqual(cell_stars["GOF"]["BBRBPC_tnt"], "***")
        self.assertEqual(cell_stars["LOF"]["ND_tnt"], "")


class SignificantMut5MatrixTest(unittest.TestCase):
    """Only TFs with a significant paired contrast reach the figure."""

    def test_drops_non_significant_tfs(self) -> None:
        with patch(f"{_MODULE}.build_matrix", return_value=_MATRIX), patch(
            f"{_MODULE}.order_tfs_by_group_contrast", side_effect=lambda m, *_: m
        ):
            matrix, row_stars = _significant_mut5_matrix(_SIGNIFICANCE)

        self.assertEqual(list(matrix.index), ["BBRBPC_tnt", "ND_tnt", "bZIP_tnt"])
        self.assertEqual(row_stars["BBRBPC_tnt"], "***")
        self.assertEqual(row_stars["bZIP_tnt"], "*")
        self.assertEqual(row_stars["MYB_tnt"], "")


class DrawMut5TfHeatmapTest(unittest.TestCase):
    """The panel draws onto the given axes and saves nothing."""

    def setUp(self) -> None:
        self.figure = plt.figure()
        grid = self.figure.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 0.02])
        self.heatmap_ax = self.figure.add_subplot(grid[0, 0])
        self.colorbar_ax = self.figure.add_subplot(grid[0, 1])

    def tearDown(self) -> None:
        plt.close(self.figure)

    def _draw(self) -> None:
        """Draw the panel with every data source mocked out."""
        intra = pd.DataFrame(
            {TF_COLUMN: ["BBRBPC_tnt", "ND_tnt"], "q_intra": [0.0001, 0.6]}
        )
        with patch(
            f"{_MODULE}.paired_tf_significance", return_value=_SIGNIFICANCE
        ), patch(f"{_MODULE}.single_run_tf_significance", return_value=intra), patch(
            f"{_MODULE}.build_matrix", return_value=_MATRIX
        ), patch(
            f"{_MODULE}.order_tfs_by_group_contrast", side_effect=lambda m, *_: m
        ), patch.object(
            plt.Figure, "savefig"
        ) as self.savefig_mock:
            _draw_mut5_tf_heatmap(self.heatmap_ax, self.colorbar_ax)

    def test_draws_transposed_with_short_starred_labels(self) -> None:
        self._draw()

        # Transposed: TF families on the x axis, runs on the y axis.
        self.assertEqual(
            [label.get_text() for label in self.heatmap_ax.get_xticklabels()],
            ["BBRBPC ***", "ND *", "bZIP *"],
        )
        self.assertEqual(
            [label.get_text() for label in self.heatmap_ax.get_yticklabels()],
            ["ara max", "GOF", "ara min", "LOF"],
        )

    def test_annotations_are_stars_only(self) -> None:
        self._draw()

        annotations = {text.get_text() for text in self.heatmap_ax.texts}
        self.assertTrue(annotations.issubset({"", "*", "**", "***"}))
        self.assertIn("***", annotations)

    def test_saves_nothing_and_fills_the_colorbar_axes(self) -> None:
        self._draw()

        self.savefig_mock.assert_not_called()
        self.assertTrue(self.colorbar_ax.has_data())

    def test_draws_both_group_dividers(self) -> None:
        self._draw()

        horizontal = [
            line
            for line in self.heatmap_ax.lines
            if line.get_ydata()[0] == line.get_ydata()[1]
        ]
        vertical = [
            line
            for line in self.heatmap_ax.lines
            if line.get_xdata()[0] == line.get_xdata()[1]
        ]
        # One divider between the max and min run groups (after 2 rows) and one
        # between the max-favoured and min-favoured TF blocks (after 2 columns).
        self.assertEqual([line.get_ydata()[0] for line in horizontal], [2.0])
        self.assertEqual([line.get_xdata()[0] for line in vertical], [2.0])


def _benchmark_frame(
    algorithm: str,
    swept_parameter: str,
    levels: "list[int]",
    device: str = "CPU",
    runtimes_per_level: "list[float] | None" = None,
) -> pd.DataFrame:
    """Build a synthetic runtime frame for one algorithm and one sweep.

    By default the runtime is ``scale * x``, a clean power law with exponent 1, so
    a fit of it is quotable. Passing ``runtimes_per_level`` replaces that with an
    arbitrary curve, which is how a curve no power law describes is built.

    Args:
        algorithm: Value for the ``algorithm`` column.
        swept_parameter: Value for the ``swept_parameter`` column.
        levels: Swept values; each gets three replicate rows.
        device: Value for the ``device`` column.
        runtimes_per_level: Runtime of each level before the per-regime scale.
            When None the runtime is the level itself.

    Returns:
        Tidy frame with the columns the figure code reads.
    """
    if runtimes_per_level is None:
        runtimes_per_level = [float(level) for level in levels]
    rows = []
    for regime, scale in (("unconstrained", 1.0), ("natural", 4.0)):
        for level, runtime in zip(levels, runtimes_per_level):
            for replicate in range(3):
                rows.append(
                    {
                        "device": device,
                        "algorithm": algorithm,
                        "regime": regime,
                        "swept_parameter": swept_parameter,
                        "swept_value": level,
                        "algorithm_seconds": scale * runtime * (1.0 + 0.01 * replicate),
                    }
                )
    return pd.DataFrame(rows)


class CpuBenchmarkRuntimesTest(unittest.TestCase):
    """Only the CPU rows of the benchmark reach the figure."""

    def test_drops_gpu_rows(self) -> None:
        loaded = pd.concat(
            [
                _benchmark_frame("ga", "population_size", [25, 50, 100]),
                _benchmark_frame(
                    "ga", "population_size", [25, 50, 100], device="GPU"
                ),
            ],
            ignore_index=True,
        )
        with patch(f"{_MODULE}.load_benchmark_runtimes", return_value=loaded):
            runtimes = _cpu_benchmark_runtimes()

        self.assertEqual(set(runtimes["device"]), {"CPU"})
        self.assertEqual(len(runtimes), len(loaded) // 2)


class RegimeScalingFitsTest(unittest.TestCase):
    """Every regime is fitted; a curve that is no power law shows it in its R²."""

    def test_fits_a_clean_power_law_in_both_regimes(self) -> None:
        frame = _benchmark_frame("ga", "population_size", [25, 50, 100, 200])

        fits = _regime_scaling_fits(frame)

        self.assertEqual(list(fits), ["unconstrained", "natural"])
        for fit in fits.values():
            self.assertAlmostEqual(fit.exponent, 1.0, places=2)
            self.assertGreater(fit.r_squared, 0.99)

    def test_fits_a_curve_that_is_not_a_power_law_with_a_low_r_squared(self) -> None:
        # A curve that falls before it rises, as the measured CPU/unconstrained
        # population sweep does: monotone in neither direction, so no power law.
        frame = _benchmark_frame(
            "ga",
            "population_size",
            [25, 50, 100, 200],
            runtimes_per_level=[140.0, 98.0, 198.0, 305.0],
        )

        fits = _regime_scaling_fits(frame)

        self.assertEqual(list(fits), ["unconstrained", "natural"])
        for fit in fits.values():
            self.assertLess(fit.r_squared, 0.95)


class SetSweepTicksTest(unittest.TestCase):
    """The x axis carries one integer-labelled tick per swept level."""

    def setUp(self) -> None:
        self.figure, self.ax = plt.subplots()
        self.ax.set_xscale("log")

    def tearDown(self) -> None:
        plt.close(self.figure)

    def test_ticks_are_the_swept_levels(self) -> None:
        _set_sweep_ticks(self.ax, np.array([100, 25, 50, 25], dtype=float))

        self.assertEqual(list(self.ax.get_xticks()), [25.0, 50.0, 100.0])
        self.assertEqual(
            [label.get_text() for label in self.ax.get_xticklabels()],
            ["25", "50", "100"],
        )


class DrawRuntimeScalingPanelTest(unittest.TestCase):
    """The panel draws onto the given axes, without a legend and without saving."""

    def setUp(self) -> None:
        self.figure, self.ax = plt.subplots()
        self.runtimes = _benchmark_frame("ga", "population_size", [25, 50, 100, 200])

    def tearDown(self) -> None:
        plt.close(self.figure)

    def test_draws_points_and_fit_lines_per_regime(self) -> None:
        _draw_runtime_scaling_panel(
            self.ax, self.runtimes, "ga", "population_size", "Genetic algorithm"
        )

        self.assertEqual(len(self.ax.collections), 2)
        self.assertEqual(len(self.ax.lines), 2)
        self.assertEqual(self.ax.get_title(), "Genetic algorithm")
        self.assertEqual(self.ax.get_xlabel(), "Population size")
        self.assertEqual(self.ax.get_xscale(), "log")
        self.assertEqual(self.ax.get_yscale(), "log")
        self.assertIsNone(self.ax.get_legend())

    def test_labels_every_fit_line_with_its_r_squared(self) -> None:
        _draw_runtime_scaling_panel(
            self.ax, self.runtimes, "ga", "population_size", "Genetic algorithm"
        )

        labels = [text.get_text() for text in self.ax.texts]
        self.assertEqual(len(labels), 2)
        for label in labels:
            self.assertTrue(label.startswith("R²"))

    def test_lines_and_labels_carry_the_magma_regime_colours(self) -> None:
        _draw_runtime_scaling_panel(
            self.ax, self.runtimes, "ga", "population_size", "Genetic algorithm"
        )

        expected = {to_hex(_REGIME_COLORS[regime]) for regime in _REGIME_COLORS}
        self.assertEqual({to_hex(line.get_color()) for line in self.ax.lines}, expected)
        self.assertEqual(
            {to_hex(text.get_color()) for text in self.ax.texts}, expected
        )

    def test_widens_the_x_axis_to_hold_the_labels(self) -> None:
        _draw_runtime_scaling_panel(
            self.ax, self.runtimes, "ga", "population_size", "Genetic algorithm"
        )

        _, right = self.ax.get_xlim()
        self.assertGreater(right, self.runtimes["swept_value"].max())

    def test_raises_on_an_empty_selection(self) -> None:
        with self.assertRaises(ValueError):
            _draw_runtime_scaling_panel(
                self.ax, self.runtimes, "greedy", "max_number_mutations", "Greedy"
            )


class PopulateFigS2Test(unittest.TestCase):
    """The three panels are drawn from one loaded frame, with a shared legend."""

    def setUp(self) -> None:
        self.figure = plt.figure(layout="constrained")
        self.runtimes = pd.concat(
            [
                _benchmark_frame("ga", "population_size", [25, 50, 100, 200]),
                _benchmark_frame("ga", "number_of_generations", [50, 200, 2000]),
                _benchmark_frame(
                    "greedy", "max_number_mutations", [5, 10, 20, 40, 100]
                ),
            ],
            ignore_index=True,
        )

    def tearDown(self) -> None:
        plt.close(self.figure)

    def test_draws_three_labelled_panels_and_one_legend(self) -> None:
        with patch(
            f"{_MODULE}._cpu_benchmark_runtimes", return_value=self.runtimes
        ):
            _populate_fig_s2(self.figure)

        self.assertEqual(len(self.figure.axes), 3)
        self.assertEqual(len(self.figure.legends), 1)
        panel_labels = [
            text.get_text()
            for ax in self.figure.axes
            for text in ax.texts
            if text.get_text() in {"A", "B", "C"}
        ]
        self.assertEqual(panel_labels, ["A", "B", "C"])

    def test_only_the_leftmost_panel_carries_the_y_label(self) -> None:
        with patch(
            f"{_MODULE}._cpu_benchmark_runtimes", return_value=self.runtimes
        ):
            _populate_fig_s2(self.figure)

        labels = [ax.get_ylabel() for ax in self.figure.axes]
        self.assertNotEqual(labels[0], "")
        self.assertEqual(labels[1:], ["", ""])


def _prepared_adversarial_frame(number_of_genes: int) -> pd.DataFrame:
    """Build a prepared adversarial table for ``number_of_genes`` genes.

    Mirrors the output of ``workflows.adversarial.plot.prepare_predictions``: one
    row per gene and mutation count, with the raw predictions of one other model
    and their normalized counterparts.

    Args:
        number_of_genes: How many genes the table holds.

    Returns:
        A table accepted by the axes level drawing functions of ``adversarial.plot``.
    """
    mutation_counts = [0, 1, 2]
    genes = []
    for gene_index in range(number_of_genes):
        fitness = np.array([0.0, 0.5, 1.0]) + 0.01 * gene_index
        other = fitness - 0.1
        genes.append(
            pd.DataFrame(
                {
                    "gene_id": [f"AT1G0000{gene_index}"] * len(mutation_counts),
                    "mutation_count": mutation_counts,
                    "optimization_model": ["model_opt"] * len(mutation_counts),
                    "original_fitness": fitness,
                    "prediction_model_opt": fitness,
                    "prediction_model_other": other,
                    "normalized_original_fitness": [0.0, 0.5, 1.0],
                    "normalized_prediction_model_opt": [0.0, 0.5, 1.0],
                    "normalized_prediction_model_other": [-0.1, 0.4, 0.9],
                }
            )
        )
    return pd.concat(genes, ignore_index=True)


class TestPopulateFigS3(unittest.TestCase):
    """Tests for the two-run adversarial re-evaluation composition."""

    def setUp(self) -> None:
        self.figure = plt.figure(figsize=(10.0, 6.0), layout="constrained")
        self.prepared = _prepared_adversarial_frame(4)

    def tearDown(self) -> None:
        plt.close(self.figure)

    def _populate(self) -> None:
        """Draw figure S3 with the run tables replaced by a small fake one."""
        with patch(f"{_MODULE}._load_adversarial_run", return_value=self.prepared):
            _populate_fig_s3(self.figure)

    def test_draws_eight_labelled_panels_and_one_legend(self) -> None:
        self._populate()

        self.assertEqual(len(self.figure.axes), 8)
        self.assertEqual(len(self.figure.legends), 1)
        panel_labels = [
            text.get_text()
            for ax in self.figure.axes
            for text in ax.texts
            if text.get_text() in set("ABCDEFGH")
        ]
        self.assertEqual(panel_labels, list("ABCDEFGH"))

    def test_only_the_bottom_row_carries_the_x_label(self) -> None:
        self._populate()

        x_labels = [ax.get_xlabel() for ax in self.figure.axes]
        self.assertEqual(x_labels[:4], [""] * 4)
        self.assertTrue(all(label == "number of mutations" for label in x_labels[4:]))

    def test_rows_are_identified_by_the_run_objective(self) -> None:
        self._populate()

        row_labels = [
            text.get_text()
            for ax in self.figure.axes
            for text in ax.texts
            if text.get_text() in {label for _, label in _ADVERSARIAL_RUNS}
        ]
        self.assertEqual(row_labels, [label for _, label in _ADVERSARIAL_RUNS])

    def test_legend_comes_from_a_panel_that_draws_every_element(self) -> None:
        self._populate()

        labels = [text.get_text() for text in self.figure.legends[0].get_texts()]
        self.assertIn("optimization model", labels)
        self.assertNotIn("optimization model (median over genes)", labels)
        # The pooled panels leave the min-max band out, the example panels do not.
        self.assertIn("other models, min-max", labels)

    def test_only_the_top_row_carries_the_column_titles(self) -> None:
        self._populate()

        titles = [ax.get_title() for ax in self.figure.axes]
        self.assertEqual(titles[:4], _ADVERSARIAL_COLUMN_TITLES)
        self.assertEqual(titles[4:], [""] * 4)

    def test_pooled_panels_omit_the_min_max_band(self) -> None:
        self._populate()

        pooled_axes = [self.figure.axes[0], self.figure.axes[4]]
        example_axes = [self.figure.axes[1], self.figure.axes[5]]
        self.assertTrue(all(len(ax.collections) == 1 for ax in pooled_axes))
        self.assertTrue(all(len(ax.collections) == 2 for ax in example_axes))


if __name__ == "__main__":
    unittest.main()
