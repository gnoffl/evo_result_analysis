"""Unit tests for the example-gene Pareto scatter calculation functions."""

import json
import unittest
from unittest.mock import mock_open, patch

from workflows.evo_alg_pooled_plots.example_gene_pareto.example_gene_pareto import (
    _frame_slug,
    compute_shared_limits,
    extract_zero_mutation_points,
    load_mutation_prediction,
    order_front_files,
    plot_final_front_mini,
    sample_gradient_colors,
)


class TestSampleGradientColors(unittest.TestCase):
    """Tests for :func:`sample_gradient_colors`."""

    def test_returns_one_rgba_per_front(self) -> None:
        # Arrange / Act
        colors = sample_gradient_colors(4)

        # Assert
        self.assertEqual(len(colors), 4)
        self.assertTrue(all(len(color) == 4 for color in colors))

    def test_empty_for_non_positive_count(self) -> None:
        # Arrange / Act / Assert
        self.assertEqual(sample_gradient_colors(0), [])


class TestFrameSlug(unittest.TestCase):
    """Tests for :func:`_frame_slug`."""

    def test_zero_pads_index_and_sanitizes_label(self) -> None:
        # Arrange / Act / Assert
        self.assertEqual(_frame_slug(0, "start"), "step_00_start")
        self.assertEqual(_frame_slug(3, "Gen 100"), "step_03_gen_100")


class TestOrderFrontFiles(unittest.TestCase):
    """Tests for :func:`order_front_files`."""

    def test_sorts_by_generation_with_final_last(self) -> None:
        # Arrange: deliberately shuffled, final front not last.
        file_names = [
            "pareto_front.json",
            "pareto_front_gen_00100.json",
            "pareto_front_gen_00005.json",
            "pareto_front_gen_00001.json",
        ]

        # Act
        ordered = order_front_files(file_names)

        # Assert
        self.assertEqual(
            ordered,
            [
                ("pareto_front_gen_00001.json", "Gen 1"),
                ("pareto_front_gen_00005.json", "Gen 5"),
                ("pareto_front_gen_00100.json", "Gen 100"),
                ("pareto_front.json", "Final"),
            ],
        )

    def test_ignores_unrelated_files(self) -> None:
        # Arrange
        file_names = ["pareto_front_gen_00002.json", "population_gen_00001_before.json"]

        # Act
        ordered = order_front_files(file_names)

        # Assert
        self.assertEqual(ordered, [("pareto_front_gen_00002.json", "Gen 2")])


class TestLoadMutationPrediction(unittest.TestCase):
    """Tests for :func:`load_mutation_prediction`."""

    def test_parses_mutation_counts_and_predictions(self) -> None:
        # Arrange
        entries = [
            ["ACGT", 0.5, 3.0],
            ["ACGA", 0.8, 5.0],
        ]
        fake_file = mock_open(read_data=json.dumps(entries))

        # Act
        with patch("builtins.open", fake_file):
            mutation_counts, predictions = load_mutation_prediction("ignored.json")

        # Assert
        self.assertEqual(mutation_counts, [3.0, 5.0])
        self.assertEqual(predictions, [0.5, 0.8])

    def test_empty_file_returns_empty_lists(self) -> None:
        # Arrange
        fake_file = mock_open(read_data=json.dumps([]))

        # Act
        with patch("builtins.open", fake_file):
            mutation_counts, predictions = load_mutation_prediction("ignored.json")

        # Assert
        self.assertEqual(mutation_counts, [])
        self.assertEqual(predictions, [])


class TestComputeSharedLimits(unittest.TestCase):
    """Tests for :func:`compute_shared_limits`."""

    def test_applies_fractional_margin(self) -> None:
        # Arrange
        mutation_counts = [0.0, 10.0]
        predictions = [0.2, 0.7]

        # Act
        xlim, ylim = compute_shared_limits(mutation_counts, predictions, margin=0.1)

        # Assert: 10% of the range (10 and 0.5) padded on each side.
        self.assertAlmostEqual(xlim[0], -1.0)
        self.assertAlmostEqual(xlim[1], 11.0)
        self.assertAlmostEqual(ylim[0], 0.15)
        self.assertAlmostEqual(ylim[1], 0.75)

    def test_zero_span_uses_fallback_padding(self) -> None:
        # Arrange: all identical values -> span is zero.
        mutation_counts = [5.0, 5.0]
        predictions = [0.4, 0.4]

        # Act
        xlim, ylim = compute_shared_limits(mutation_counts, predictions, margin=0.1)

        # Assert: fallback pad = abs(high) * margin (0.5 for x, 0.04 for y).
        self.assertAlmostEqual(xlim[0], 4.5)
        self.assertAlmostEqual(xlim[1], 5.5)
        self.assertAlmostEqual(ylim[0], 0.36)
        self.assertAlmostEqual(ylim[1], 0.44)

    def test_empty_input_raises_value_error(self) -> None:
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            compute_shared_limits([], [])


class TestExtractZeroMutationPoints(unittest.TestCase):
    """Tests for :func:`extract_zero_mutation_points`."""

    def test_keeps_only_zero_mutation_entries(self) -> None:
        # Arrange
        mutation_counts = [0.0, 2.0, 0.0, 3.0]
        predictions = [0.19, 0.5, 0.20, 0.8]

        # Act
        zero_counts, zero_predictions = extract_zero_mutation_points(
            mutation_counts, predictions
        )

        # Assert
        self.assertEqual(zero_counts, [0.0, 0.0])
        self.assertEqual(zero_predictions, [0.19, 0.20])

    def test_no_zero_entries_returns_empty(self) -> None:
        # Arrange
        mutation_counts = [1.0, 2.0]
        predictions = [0.3, 0.4]

        # Act
        zero_counts, zero_predictions = extract_zero_mutation_points(
            mutation_counts, predictions
        )

        # Assert
        self.assertEqual(zero_counts, [])
        self.assertEqual(zero_predictions, [])


class TestPlotFinalFrontMini(unittest.TestCase):
    """Tests for :func:`plot_final_front_mini`."""

    def test_draws_scatter_with_axis_labels_but_no_title_or_legend(self) -> None:
        # Arrange
        mutation_counts = [0.0, 2.0, 5.0]
        predictions = [0.2, 0.5, 0.9]

        # Act
        fig = plot_final_front_mini(mutation_counts, predictions)

        # Assert
        ax = fig.axes[0]
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.collections[0].get_offsets()), 3)
        self.assertNotEqual(ax.get_xlabel(), "")
        self.assertNotEqual(ax.get_ylabel(), "")
        self.assertEqual(ax.get_title(), "")
        self.assertIsNone(ax.get_legend())


if __name__ == "__main__":
    unittest.main()
