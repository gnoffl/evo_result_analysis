"""Tests for workflows.adversarial.plot."""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

# plot imports reevaluate, which imports the evolution package only inside functions.
sys.modules.setdefault("evolution", MagicMock())
sys.modules.setdefault("evolution.load_models", MagicMock())
sys.modules.setdefault("evolution.sequences", MagicMock())

from analysis.overview.simple_result_stats import expand_pareto_front  # noqa: E402
from workflows.adversarial import plot  # noqa: E402

plt = plot.plt

MAX_NUMBER_MUTATIONS = 4


def make_gene_table(gene_id, mutation_counts, fitnesses, other_predictions):
    """Build a re-evaluation table for a single gene.

    Args:
        gene_id: Identifier of the gene.
        mutation_counts: Mutation counts present in the front.
        fitnesses: Optimization model fitness per mutation count.
        other_predictions: Prediction of the other model per mutation count.

    Returns:
        A DataFrame with the columns written by ``reevaluate.py``.
    """
    return pd.DataFrame(
        {
            "gene_id": [gene_id] * len(mutation_counts),
            "sequence": ["ACGT"] * len(mutation_counts),
            "mutation_count": mutation_counts,
            "max_number_mutations": [MAX_NUMBER_MUTATIONS] * len(mutation_counts),
            "optimization_model": ["model_opt"] * len(mutation_counts),
            "original_fitness": fitnesses,
            "prediction_model_opt": fitnesses,
            "prediction_model_other": other_predictions,
        }
    )


class TestGetOtherModelColumns(unittest.TestCase):
    """Tests for excluding the optimization model."""

    def test_optimization_model_is_excluded(self):
        table = make_gene_table("g", [0, 1], [0.1, 0.9], [0.2, 0.4])

        self.assertEqual(["prediction_model_other"], plot.get_other_model_columns(table))

    def test_multiple_optimization_models_are_excluded(self):
        table = make_gene_table("g", [0, 1], [0.1, 0.9], [0.2, 0.4])
        table["optimization_model"] = "model_opt;model_other"

        with self.assertRaises(ValueError):
            plot.get_other_model_columns(table)


class TestExpandGeneFront(unittest.TestCase):
    """Tests for filling the gaps of a Pareto front."""

    def test_gaps_are_filled_by_carrying_rows_forward(self):
        table = make_gene_table("g", [0, 1, 3], [0.1, 0.2, 0.4], [0.5, 0.6, 0.7])

        expanded = plot.expand_gene_front(table, MAX_NUMBER_MUTATIONS)

        self.assertEqual([0, 1, 2, 3, 4], list(expanded["mutation_count"]))
        np.testing.assert_allclose([0.1, 0.2, 0.2, 0.4, 0.4], expanded["original_fitness"])
        np.testing.assert_allclose([0.5, 0.6, 0.6, 0.7, 0.7], expanded["prediction_model_other"])
        self.assertEqual([False, False, True, False, True], list(expanded[plot.EXPANDED_COLUMN]))

    def test_expansion_matches_expand_pareto_front(self):
        mutation_counts = [0, 1, 3]
        fitnesses = [0.1, 0.2, 0.4]
        table = make_gene_table("g", mutation_counts, fitnesses, [0.5, 0.6, 0.7])
        pareto_front = [
            ["SEQ", fitness, mutation_count]
            for fitness, mutation_count in zip(fitnesses, mutation_counts)
        ]

        expanded = plot.expand_gene_front(table, MAX_NUMBER_MUTATIONS)
        reference = expand_pareto_front(pareto_front, MAX_NUMBER_MUTATIONS)

        np.testing.assert_allclose(
            [item[1] for item in reference], expanded["original_fitness"].to_numpy()
        )
        self.assertEqual([item[2] for item in reference], list(expanded["mutation_count"]))

    def test_missing_reference_point_raises(self):
        table = make_gene_table("g", [1, 2], [0.2, 0.4], [0.6, 0.7])

        with self.assertRaises(ValueError):
            plot.expand_gene_front(table, MAX_NUMBER_MUTATIONS)


class TestNormalizeGeneFront(unittest.TestCase):
    """Tests for the per gene normalization."""

    def test_optimization_model_spans_zero_to_one(self):
        table = make_gene_table("g", [0, 1, 2], [0.2, 0.6, 1.0], [0.0, 0.4, 0.6])

        normalized = plot.normalize_gene_front(table)

        np.testing.assert_allclose([0.0, 0.5, 1.0], normalized[plot.NORMALIZED_FITNESS_COLUMN])

    def test_other_models_use_the_same_transformation_and_are_unbounded(self):
        table = make_gene_table("g", [0, 1, 2], [0.2, 0.6, 1.0], [0.0, 0.4, 1.4])

        normalized = plot.normalize_gene_front(table)

        # (p - 0.2) / (1.0 - 0.2)
        np.testing.assert_allclose(
            [-0.25, 0.25, 1.5], normalized["normalized_prediction_model_other"]
        )

    def test_constant_fitness_is_dropped_with_a_warning(self):
        table = make_gene_table("g", [0, 1], [0.5, 0.5], [0.2, 0.4])

        with self.assertWarns(UserWarning):
            self.assertIsNone(plot.normalize_gene_front(table))


class TestSummarizeSpread(unittest.TestCase):
    """Tests for the pooled spread summary."""

    def test_quartiles_and_extremes_are_pooled_over_columns_and_rows(self):
        prepared = pd.DataFrame(
            {
                "mutation_count": [0, 0, 1, 1],
                "prediction_a": [0.0, 2.0, 10.0, 10.0],
                "prediction_b": [1.0, 3.0, 20.0, 30.0],
            }
        )

        spread = plot.summarize_spread(prepared, ["prediction_a", "prediction_b"])

        self.assertAlmostEqual(1.5, spread.loc[0, "median"])
        self.assertAlmostEqual(0.0, spread.loc[0, "minimum"])
        self.assertAlmostEqual(3.0, spread.loc[0, "maximum"])
        self.assertAlmostEqual(0.75, spread.loc[0, "lower_quartile"])
        self.assertAlmostEqual(2.25, spread.loc[0, "upper_quartile"])
        self.assertAlmostEqual(15.0, spread.loc[1, "median"])


class TestPreparePredictions(unittest.TestCase):
    """Tests for expanding and normalizing all genes at once."""

    def test_every_gene_covers_the_full_grid(self):
        predictions = pd.concat(
            [
                make_gene_table("gene_a", [0, 1, 4], [0.1, 0.5, 0.9], [0.1, 0.2, 0.3]),
                make_gene_table("gene_b", [0, 4], [0.2, 0.8], [0.2, 0.9]),
            ],
            ignore_index=True,
        )

        prepared = plot.prepare_predictions(predictions)

        self.assertEqual({"gene_a", "gene_b"}, set(prepared["gene_id"]))
        for _, gene_predictions in prepared.groupby("gene_id"):
            self.assertEqual(
                list(range(MAX_NUMBER_MUTATIONS + 1)), list(gene_predictions["mutation_count"])
            )

    def test_all_genes_unnormalizable_raises(self):
        predictions = make_gene_table("gene_a", [0, 4], [0.5, 0.5], [0.1, 0.2])

        with self.assertWarns(UserWarning):
            with self.assertRaises(ValueError):
                plot.prepare_predictions(predictions)


class TestMinimumFitnessRange(unittest.TestCase):
    """Tests for dropping genes with a degenerate normalization denominator."""

    def setUp(self):
        self.predictions = pd.concat(
            [
                make_gene_table("wide", [0, 4], [0.0, 1.0], [0.0, 0.9]),
                make_gene_table("narrow", [0, 4], [0.990, 0.995], [0.5, 0.6]),
            ],
            ignore_index=True,
        )

    def test_narrow_gene_is_dropped(self):
        prepared = plot.prepare_predictions(self.predictions, minimum_fitness_range=0.1)

        self.assertEqual({"wide"}, set(prepared["gene_id"]))

    def test_no_filter_keeps_both_genes(self):
        prepared = plot.prepare_predictions(self.predictions)

        self.assertEqual({"wide", "narrow"}, set(prepared["gene_id"]))

    def test_dropping_every_gene_raises(self):
        with self.assertRaises(ValueError):
            plot.prepare_predictions(self.predictions, minimum_fitness_range=2.0)


class TestSetRobustYlimits(unittest.TestCase):
    """Tests for the y axis limits of the pooled figure."""

    def setUp(self):
        self.spread = pd.DataFrame(
            {
                "median": [0.0, 0.5],
                "lower_quartile": [0.0, 0.4],
                "upper_quartile": [0.1, 0.6],
                "minimum": [-20.0, 0.2],
                "maximum": [0.2, 30.0],
            },
            index=pd.Index([0, 1], name="mutation_count"),
        )

    def test_limits_follow_the_interquartile_band_not_the_extremes(self):
        figure, axes = plt.subplots()
        try:
            plot.set_robust_ylimits(axes, self.spread, np.array([0.0, 1.0]))
            lower, upper = axes.get_ylim()
        finally:
            plt.close(figure)

        self.assertGreater(lower, -1.0)
        self.assertLess(upper, 2.0)

    def test_truncated_band_is_annotated(self):
        figure, axes = plt.subplots()
        try:
            plot.set_robust_ylimits(axes, self.spread, np.array([0.0, 1.0]))
            annotations = [text.get_text() for text in axes.texts]
        finally:
            plt.close(figure)

        self.assertEqual(1, len(annotations))
        self.assertIn("down to -20", annotations[0])
        self.assertIn("up to 30", annotations[0])

    def test_band_within_limits_is_not_annotated(self):
        spread = self.spread.copy()
        spread["minimum"] = [0.0, 0.3]
        spread["maximum"] = [0.1, 0.7]
        figure, axes = plt.subplots()
        try:
            plot.set_robust_ylimits(axes, spread, np.array([0.0, 1.0]))
            annotations = list(axes.texts)
        finally:
            plt.close(figure)

        self.assertEqual([], annotations)

    def test_short_note_drops_the_explanatory_sentence(self):
        figure, axes = plt.subplots()
        try:
            plot.set_robust_ylimits(
                axes, self.spread, np.array([0.0, 1.0]), short_note=True
            )
            annotations = [text.get_text() for text in axes.texts]
        finally:
            plt.close(figure)

        self.assertEqual(1, len(annotations))
        self.assertNotIn("extends beyond", annotations[0])
        self.assertIn("down to -20", annotations[0])


class TestPanelDrawers(unittest.TestCase):
    """Tests for the axes level drawing functions reused by composed figures."""

    def setUp(self):
        self.prepared = plot.prepare_predictions(
            pd.concat(
                [
                    make_gene_table(
                        "gene_a", [0, 2, 4], [0.1, 0.5, 0.9], [0.2, 0.4, 0.8]
                    ),
                    make_gene_table(
                        "gene_b", [0, 2, 4], [0.2, 0.6, 1.0], [0.1, 0.3, 0.7]
                    ),
                ],
                ignore_index=True,
            )
        )

    def test_pooled_panel_draws_onto_the_given_axes(self):
        figure, axes = plt.subplots()
        try:
            handles = plot.draw_pooled_panel(axes, self.prepared)
            number_of_lines = len(axes.lines)
            y_label = axes.get_ylabel()
        finally:
            plt.close(figure)

        # The other models' median and the optimization model's front.
        self.assertEqual(2, number_of_lines)
        self.assertEqual("normalized prediction", y_label)
        self.assertEqual(4, len(handles))

    def test_pooled_panel_creates_no_figure_of_its_own(self):
        figure, axes = plt.subplots()
        try:
            figures_before = set(plt.get_fignums())
            plot.draw_pooled_panel(axes, self.prepared)
            figures_after = set(plt.get_fignums())
        finally:
            plt.close(figure)

        self.assertEqual(figures_before, figures_after)

    def test_pooled_panel_can_drop_the_min_max_band_and_its_note(self):
        figure, axes = plt.subplots()
        try:
            handles = plot.draw_pooled_panel(axes, self.prepared, show_min_max=False)
            number_of_bands = len(axes.collections)
            annotations = list(axes.texts)
        finally:
            plt.close(figure)

        # Only the interquartile band is left.
        self.assertEqual(1, number_of_bands)
        self.assertEqual([], annotations)
        self.assertEqual(3, len(handles))
        self.assertNotIn(
            "min-max", " ".join(handle.get_label() for handle in handles)
        )

    def test_panel_colors_are_configurable(self):
        figure, axes = plt.subplots()
        try:
            plot.draw_example_panel(
                axes,
                self.prepared,
                "gene_a",
                reference_color="#641a80",
                others_color="#f9795d",
            )
            line_colors = [line.get_color() for line in axes.lines]
        finally:
            plt.close(figure)

        self.assertEqual(["#f9795d", "#641a80"], line_colors)

    def test_example_panel_draws_the_raw_values_of_one_gene(self):
        figure, axes = plt.subplots()
        try:
            plot.draw_example_panel(axes, self.prepared, "gene_a")
            reference_values = axes.lines[-1].get_ydata()
        finally:
            plt.close(figure)

        # Raw, unnormalized fitness of gene_a, expanded onto the full grid.
        np.testing.assert_allclose(
            np.array([0.1, 0.1, 0.5, 0.5, 0.9]), np.asarray(reference_values)
        )


class TestReportExpansion(unittest.TestCase):
    """Tests for the summary of carried forward rows."""

    def test_fraction_and_onset_are_reported(self):
        # Both genes have entries at 0 and 1 only, so counts 2 to 4 are carried forward.
        prepared = plot.prepare_predictions(
            pd.concat(
                [
                    make_gene_table("gene_a", [0, 1], [0.1, 0.9], [0.1, 0.5]),
                    make_gene_table("gene_b", [0, 1], [0.2, 0.8], [0.2, 0.3]),
                ],
                ignore_index=True,
            )
        )

        summary = plot.report_expansion(prepared)

        self.assertIn("60.0%", summary)
        self.assertIn("mutation count 2 on", summary)

    def test_complete_front_reports_no_onset(self):
        prepared = plot.prepare_predictions(
            make_gene_table("gene_a", [0, 1, 2, 3, 4], [0.0, 0.2, 0.4, 0.6, 1.0], [0.0] * 5)
        )

        summary = plot.report_expansion(prepared)

        self.assertIn("0.0%", summary)
        self.assertNotIn("mutation count", summary)


class TestGeneSelection(unittest.TestCase):
    """Tests for the mean normalized gap and picking example genes."""

    def setUp(self):
        # All three genes have the front at mutation counts 0 and 4, so after expansion
        # counts 1 to 3 repeat the value at count 0.
        # gene_a: the other model reproduces the front exactly -> gap 0 everywhere.
        # gene_b: the other model never moves -> gap 0 at counts 0 to 3, 1 at count 4.
        # gene_c: the other model moves half way -> gap 0.5 at count 4 only.
        self.prepared = plot.prepare_predictions(
            pd.concat(
                [
                    make_gene_table("gene_a", [0, 4], [0.0, 1.0], [0.0, 1.0]),
                    make_gene_table("gene_b", [0, 4], [0.0, 1.0], [0.0, 0.0]),
                    make_gene_table("gene_c", [0, 4], [0.0, 1.0], [0.0, 0.5]),
                ],
                ignore_index=True,
            )
        )

    def test_gap_is_the_mean_over_the_whole_front(self):
        gaps = plot.compute_mean_normalized_gap(self.prepared)

        self.assertAlmostEqual(0.0, gaps["gene_a"])
        self.assertAlmostEqual(0.5 / 5, gaps["gene_c"])
        self.assertAlmostEqual(1.0 / 5, gaps["gene_b"])

    def test_gap_distinguishes_a_lagging_model_the_endpoint_would_miss(self):
        # Both genes end at the same value, so an endpoint based measure would be 0 for
        # both; the lagging gene must still get the larger gap.
        prepared = plot.prepare_predictions(
            pd.concat(
                [
                    make_gene_table(
                        "follows", [0, 1, 2, 3, 4], [0.0, 1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0]
                    ),
                    make_gene_table(
                        "lags", [0, 1, 2, 3, 4], [0.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.5, 1.0]
                    ),
                ],
                ignore_index=True,
            )
        )

        gaps = plot.compute_mean_normalized_gap(prepared)

        self.assertAlmostEqual(0.0, gaps["follows"])
        self.assertAlmostEqual(2.5 / 5, gaps["lags"])

    def test_selection_returns_all_genes_when_fewer_than_requested(self):
        selected = plot.select_example_genes(self.prepared, 5)

        self.assertEqual(3, len(selected))
        self.assertEqual({"gene_a", "gene_b", "gene_c"}, set(selected))

    def test_selection_is_best_median_worst(self):
        selected = plot.select_example_genes(self.prepared, 3)

        self.assertEqual(["gene_a", "gene_c", "gene_b"], selected)

    def test_selection_spans_the_extremes_of_a_larger_set(self):
        # Gene i has the other model reach i/9 of the front, so the score rises with i.
        prepared = plot.prepare_predictions(
            pd.concat(
                [
                    make_gene_table(f"gene_{index}", [0, 4], [0.0, 1.0], [0.0, index / 9])
                    for index in range(10)
                ],
                ignore_index=True,
            )
        )

        selected = plot.select_example_genes(prepared, 3)

        # Ranks 0, 4 and 9 of 10 genes: the middle rank of an even count is the lower
        # of the two central genes.
        self.assertEqual(["gene_9", "gene_5", "gene_0"], selected)


class TestPlotCsv(unittest.TestCase):
    """End to end test writing both figures."""

    def setUp(self):
        self.temporary_folder = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temporary_folder.name, "adversarial_predictions.csv")
        predictions = pd.concat(
            [
                make_gene_table("gene_a", [0, 2, 4], [0.1, 0.5, 0.9], [0.1, 0.4, 0.7]),
                make_gene_table("gene_b", [0, 1, 4], [0.2, 0.4, 0.8], [0.2, 0.2, 0.3]),
            ],
            ignore_index=True,
        )
        predictions.to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.temporary_folder.cleanup()

    def test_both_figures_and_the_gap_table_are_written(self):
        written = plot.plot_csv(self.csv_path, "png")

        self.assertEqual(
            {plot.POOLED_FILE_NAME, plot.EXAMPLES_FILE_NAME, plot.GAP_FILE_NAME},
            set(written),
        )
        for path in written.values():
            self.assertTrue(os.path.isfile(path), path)

    def test_long_sequence_names_in_the_csv_are_cleaned(self):
        # Tables written before gene id cleaning was introduced carry the full sequence
        # name, as produced by the evolution runs.
        predictions = pd.concat(
            [
                make_gene_table(
                    "1_AT1G00001_gene:267992-269819", [0, 2, 4], [0.1, 0.5, 0.9], [0.1, 0.4, 0.7]
                ),
                make_gene_table(
                    "5_AT5G00002_gene:6982163-6979156", [0, 1, 4], [0.2, 0.4, 0.8], [0.2, 0.2, 0.3]
                ),
            ],
            ignore_index=True,
        )
        predictions.to_csv(self.csv_path, index=False)

        written = plot.plot_csv(self.csv_path, "png")
        gaps = pd.read_csv(written[plot.GAP_FILE_NAME])

        self.assertEqual({"AT1G00001", "AT5G00002"}, set(gaps["gene_id"]))

    def test_unknown_example_gene_raises(self):
        with self.assertRaises(ValueError):
            plot.plot_csv(self.csv_path, "png", example_genes=["gene_missing"])


if __name__ == "__main__":
    unittest.main()
