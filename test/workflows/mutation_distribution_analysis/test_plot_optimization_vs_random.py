"""Tests for plot_optimization_vs_random calculation functions."""

import textwrap
import unittest
from unittest.mock import mock_open, patch

from workflows.mutation_distribution_analysis.plot_optimization_vs_random import (
    build_plot_dataframe,
    load_evolution_summary,
    load_random_mutation_averages,
)


RANDOM_MUTATIONS_CONTENT = textwrap.dedent("""\
    Average predictions after random mutations:
    Arabidopsis: 0.48
    Maize: 0.41
""")

ARA_SUMMARY_CONTENT = textwrap.dedent("""\
    Summary for ara_msr_max_single:
      final_fitness_mean: 0.9999
      final_fitness_std: 0.001
      start_fitness_mean: 0.5177
      start_fitness_std: 0.37
    Summary for ara_msr_max_single_main_chromosome:
      final_fitness_mean: 0.9998
      final_fitness_std: 0.001
      start_fitness_mean: 0.5214
      start_fitness_std: 0.37
""")

ZEA_SUMMARY_CONTENT = textwrap.dedent("""\
    Summary for zea_msr_max_single:
      final_fitness_mean: 0.9959
      final_fitness_std: 0.015
      start_fitness_mean: 0.3987
      start_fitness_std: 0.32
""")


class TestLoadRandomMutationAverages(unittest.TestCase):
    def test_parses_arabidopsis_and_maize(self):
        with patch("builtins.open", mock_open(read_data=RANDOM_MUTATIONS_CONTENT)):
            result = load_random_mutation_averages("dummy.txt")

        self.assertAlmostEqual(result["Arabidopsis"], 0.48)
        self.assertAlmostEqual(result["Maize"], 0.41)

    def test_returns_only_two_keys(self):
        with patch("builtins.open", mock_open(read_data=RANDOM_MUTATIONS_CONTENT)):
            result = load_random_mutation_averages("dummy.txt")

        self.assertEqual(set(result.keys()), {"Arabidopsis", "Maize"})


class TestLoadEvolutionSummary(unittest.TestCase):
    def test_parses_correct_block(self):
        with patch("builtins.open", mock_open(read_data=ARA_SUMMARY_CONTENT)):
            result = load_evolution_summary("dummy.txt", "ara_msr_max_single")

        self.assertAlmostEqual(result["start_fitness_mean"], 0.5177)
        self.assertAlmostEqual(result["final_fitness_mean"], 0.9999)

    def test_does_not_bleed_into_next_block(self):
        with patch("builtins.open", mock_open(read_data=ARA_SUMMARY_CONTENT)):
            result = load_evolution_summary("dummy.txt", "ara_msr_max_single")

        # main_chromosome block has start 0.5214; should not appear
        self.assertAlmostEqual(result["start_fitness_mean"], 0.5177)

    def test_parses_zea_block(self):
        with patch("builtins.open", mock_open(read_data=ZEA_SUMMARY_CONTENT)):
            result = load_evolution_summary("dummy.txt", "zea_msr_max_single")

        self.assertAlmostEqual(result["start_fitness_mean"], 0.3987)
        self.assertAlmostEqual(result["final_fitness_mean"], 0.9959)


class TestBuildPlotDataframe(unittest.TestCase):
    def _make_opens(self):
        """Return a side_effect list for three sequential open() calls.

        Call order in build_plot_dataframe: random mutations, ara summary, zea summary.
        """
        return [
            mock_open(read_data=RANDOM_MUTATIONS_CONTENT).return_value,
            mock_open(read_data=ARA_SUMMARY_CONTENT).return_value,
            mock_open(read_data=ZEA_SUMMARY_CONTENT).return_value,
        ]

    def test_dataframe_shape_and_columns(self):
        with patch("builtins.open") as mock_file:
            mock_file.side_effect = self._make_opens()
            df = build_plot_dataframe("ara.txt", "zea.txt", "random.txt")

        self.assertEqual(df.shape, (6, 3))
        self.assertListEqual(list(df.columns), ["species", "condition", "score"])

    def test_correct_species_values(self):
        with patch("builtins.open") as mock_file:
            mock_file.side_effect = self._make_opens()
            df = build_plot_dataframe("ara.txt", "zea.txt", "random.txt")

        self.assertEqual(set(df["species"].unique()), {"Arabidopsis", "Maize"})

    def test_correct_conditions(self):
        with patch("builtins.open") as mock_file:
            mock_file.side_effect = self._make_opens()
            df = build_plot_dataframe("ara.txt", "zea.txt", "random.txt")

        expected_conditions = {
            "Before optimization",
            "After random mutations",
            "After optimization",
        }
        self.assertEqual(set(df["condition"].unique()), expected_conditions)

    def test_arabidopsis_scores(self):
        with patch("builtins.open") as mock_file:
            mock_file.side_effect = self._make_opens()
            df = build_plot_dataframe("ara.txt", "zea.txt", "random.txt")

        def _get_score(species: str, condition: str) -> float:
            mask = (df["species"] == species) & (df["condition"] == condition)
            return float(df.loc[mask, "score"].values[0])

        self.assertAlmostEqual(_get_score("Arabidopsis", "Before optimization"), 0.5177)
        self.assertAlmostEqual(_get_score("Arabidopsis", "After optimization"), 0.9999)
        self.assertAlmostEqual(_get_score("Arabidopsis", "After random mutations"), 0.48)


if __name__ == "__main__":
    unittest.main()
