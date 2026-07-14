"""Unit tests for the natural vs unconstrained final-fitness comparison."""

import json
import unittest
from unittest.mock import mock_open, patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.workflows.evo_alg_pooled_plots.natural_unconstrained_comparison.natural_unconstrained_comparison import (  # noqa: E402
    core_gene_id,
    load_final_fitness,
    p_value_to_stars,
    pair_final_fitness,
    plot_paired_fitness,
    rank_biserial_effect_size,
    wilcoxon_paired_test,
)


class TestPValueToStars(unittest.TestCase):
    def test_thresholds_map_to_expected_stars(self) -> None:
        # Arrange / Act / Assert
        self.assertEqual(p_value_to_stars(0.0005), "***")
        self.assertEqual(p_value_to_stars(0.005), "**")
        self.assertEqual(p_value_to_stars(0.03), "*")
        self.assertEqual(p_value_to_stars(0.2), "ns")


class TestCoreGeneId(unittest.TestCase):
    def test_strips_suffix_to_first_two_fields(self) -> None:
        # Arrange
        gene_id = "5_AT5G13640_gene:4392936-4397541_251010_182339_304179"
        # Act
        result = core_gene_id(gene_id)
        # Assert
        self.assertEqual(result, "5_AT5G13640")


class TestLoadFinalFitness(unittest.TestCase):
    def test_maps_core_gene_to_final_fitness(self) -> None:
        # Arrange
        stats = {
            "5_AT5G13640_gene_a_111": {"final_fitness": 0.9, "start_fitness": 0.5},
            "1_AT1G01720_gene_b_222": {"final_fitness": 0.7, "start_fitness": 0.4},
        }
        # Act
        with patch(
            "glob.glob", return_value=["/run/stats_x.json"]
        ), patch("builtins.open", mock_open(read_data=json.dumps(stats))):
            result = load_final_fitness("/run")
        # Assert
        self.assertEqual(result, {"5_AT5G13640": 0.9, "1_AT1G01720": 0.7})

    def test_raises_when_not_exactly_one_stats_file(self) -> None:
        # Arrange / Act / Assert
        with patch("glob.glob", return_value=[]):
            with self.assertRaises(FileNotFoundError):
                load_final_fitness("/run")


class TestPairFinalFitness(unittest.TestCase):
    def test_inner_join_drops_unmatched_and_sorts(self) -> None:
        # Arrange
        unconstrained = {"1_A": 0.9, "2_B": 0.8, "3_C": 0.7}
        natural = {"2_B": 0.6, "1_A": 0.5}  # 3_C missing, order scrambled
        # Act
        with patch(
            "src.workflows.evo_alg_pooled_plots.natural_unconstrained_comparison."
            "natural_unconstrained_comparison.load_final_fitness",
            side_effect=[unconstrained, natural],
        ):
            paired = pair_final_fitness("/unconstrained", "/natural")
        # Assert
        self.assertEqual(list(paired.index), ["1_A", "2_B"])
        self.assertEqual(list(paired["unconstrained"]), [0.9, 0.8])
        self.assertEqual(list(paired["natural"]), [0.5, 0.6])


class TestRankBiserialEffectSize(unittest.TestCase):
    def test_all_positive_differences_gives_plus_one(self) -> None:
        # Arrange
        differences = np.array([0.1, 0.2, 0.3])
        # Act
        result = rank_biserial_effect_size(differences)
        # Assert
        self.assertAlmostEqual(result, 1.0)

    def test_all_negative_differences_gives_minus_one(self) -> None:
        # Arrange
        differences = np.array([-0.1, -0.2, -0.3])
        # Act
        result = rank_biserial_effect_size(differences)
        # Assert
        self.assertAlmostEqual(result, -1.0)

    def test_zero_differences_are_dropped(self) -> None:
        # Arrange: only nonzero diffs count; here all positive -> +1
        differences = np.array([0.0, 0.5, 0.5])
        # Act
        result = rank_biserial_effect_size(differences)
        # Assert
        self.assertAlmostEqual(result, 1.0)

    def test_all_zero_differences_returns_nan(self) -> None:
        # Arrange
        differences = np.array([0.0, 0.0])
        # Act
        result = rank_biserial_effect_size(differences)
        # Assert
        self.assertTrue(np.isnan(result))


class TestWilcoxonPairedTest(unittest.TestCase):
    def test_reports_expected_summary_fields(self) -> None:
        # Arrange: natural consistently below unconstrained (maximization weaker)
        paired = pd.DataFrame(
            {"unconstrained": [0.9, 0.8, 0.95, 0.7], "natural": [0.6, 0.5, 0.7, 0.4]}
        )
        # Act
        result = wilcoxon_paired_test(paired)
        # Assert
        self.assertEqual(result["n_pairs"], 4)
        self.assertAlmostEqual(result["median_unconstrained"], 0.85)
        self.assertAlmostEqual(result["median_natural"], 0.55)
        self.assertLess(result["median_difference"], 0.0)
        self.assertAlmostEqual(result["rank_biserial"], -1.0)
        # n=4 all same sign -> minimum two-sided Wilcoxon p is 2 / 2**4 = 0.125
        self.assertAlmostEqual(result["p_value"], 0.125)

    def test_one_sided_halves_the_p_value(self) -> None:
        # Arrange: all differences negative, so the "less" tail is the correct side
        paired = pd.DataFrame(
            {"unconstrained": [0.9, 0.8, 0.95, 0.7], "natural": [0.6, 0.5, 0.7, 0.4]}
        )
        # Act
        result = wilcoxon_paired_test(paired, alternative="less")
        # Assert: one-sided p is half the two-sided minimum (0.125 / 2)
        self.assertAlmostEqual(result["p_value"], 0.0625)


class TestPlotPairedFitness(unittest.TestCase):
    @patch("matplotlib.pyplot.savefig")
    def test_draws_onto_provided_ax_without_saving(self, mock_savefig) -> None:
        # Arrange
        paired = pd.DataFrame(
            {"unconstrained": [0.9, 0.8, 0.95, 0.7], "natural": [0.6, 0.5, 0.7, 0.4]}
        )
        fig, ax = plt.subplots()

        # Act
        try:
            returned = plot_paired_fitness(paired, "maximization", 0.01, ax=ax)

            # Assert: drawn onto the provided ax, its figure returned, nothing saved.
            self.assertIs(returned, ax.get_figure())
            self.assertGreater(len(ax.patches), 0)  # boxes drawn
            self.assertGreater(len(ax.lines), 0)  # connecting lines / bracket drawn
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
