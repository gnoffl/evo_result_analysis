"""Tests for natural_unconstrained_mutation_vis.py."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from workflows.evo_alg_pooled_plots.natural_unconstrained_comparison.natural_unconstrained_mutation_vis import (
    _add_significance_brackets,
    _barplot_figure,
    _pvalue_label,
    build_comparison_dataframe,
    collect_unconstrained_stats,
    count_unconstrained_positions,
    run_statistical_tests,
)


class TestCountUnconstrainedPositions(unittest.TestCase):
    """Tests for count_unconstrained_positions."""

    def test_counts_non_n_bases(self):
        """Sequence with some Ns returns count of non-N bases."""
        mock_fasta = MagicMock()
        mock_fasta.__getitem__.return_value = "ACGTNNN"

        with patch(
            "workflows.evo_alg_pooled_plots"
            ".natural_unconstrained_comparison"
            ".natural_unconstrained_mutation_vis.Fasta",
            return_value=mock_fasta,
        ):
            result = count_unconstrained_positions(Path("dummy.fa"))

        self.assertEqual(result, 4)

    def test_all_n_returns_zero(self):
        """Sequence of all Ns returns zero."""
        mock_fasta = MagicMock()
        mock_fasta.__getitem__.return_value = "NNNNN"

        with patch(
            "workflows.evo_alg_pooled_plots"
            ".natural_unconstrained_comparison"
            ".natural_unconstrained_mutation_vis.Fasta",
            return_value=mock_fasta,
        ):
            result = count_unconstrained_positions(Path("dummy.fa"))

        self.assertEqual(result, 0)

    def test_no_n_returns_full_length(self):
        """Sequence without any Ns returns full sequence length."""
        mock_fasta = MagicMock()
        mock_fasta.__getitem__.return_value = "ACGTACGT"

        with patch(
            "workflows.evo_alg_pooled_plots"
            ".natural_unconstrained_comparison"
            ".natural_unconstrained_mutation_vis.Fasta",
            return_value=mock_fasta,
        ):
            result = count_unconstrained_positions(Path("dummy.fa"))

        self.assertEqual(result, 8)

    def test_lowercase_n_counted_as_masked(self):
        """Lowercase n is also treated as a masked position."""
        mock_fasta = MagicMock()
        mock_fasta.__getitem__.return_value = "ACGTnnn"

        with patch(
            "workflows.evo_alg_pooled_plots"
            ".natural_unconstrained_comparison"
            ".natural_unconstrained_mutation_vis.Fasta",
            return_value=mock_fasta,
        ):
            result = count_unconstrained_positions(Path("dummy.fa"))

        self.assertEqual(result, 4)


class TestCollectUnconstrainedStats(unittest.TestCase):
    """Tests for collect_unconstrained_stats."""

    def test_skips_non_digit_directories(self):
        """Non-digit entries (JSON files, __pycache__) are ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "problem_configs.json").touch()
            (run_dir / "1_GENE_A").mkdir()
            (run_dir / "1_GENE_A" / "reference_sequence.fa").touch()
            (run_dir / "2_GENE_B").mkdir()
            (run_dir / "2_GENE_B" / "reference_sequence.fa").touch()

            with patch(
                "workflows.evo_alg_pooled_plots"
                ".natural_unconstrained_comparison"
                ".natural_unconstrained_mutation_vis.count_unconstrained_positions",
                side_effect=[100, 200],
            ):
                positions, mutations = collect_unconstrained_stats(run_dir)

        self.assertEqual(positions, [100, 200])
        self.assertEqual(mutations, [300, 600])

    def test_mutations_are_three_times_positions(self):
        """Total mutations equals mutable positions * 3."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "1_GENE_A").mkdir()
            (run_dir / "1_GENE_A" / "reference_sequence.fa").touch()

            with patch(
                "workflows.evo_alg_pooled_plots"
                ".natural_unconstrained_comparison"
                ".natural_unconstrained_mutation_vis.count_unconstrained_positions",
                return_value=50,
            ):
                positions, mutations = collect_unconstrained_stats(run_dir)

        self.assertEqual(positions, [50])
        self.assertEqual(mutations, [150])


class TestBuildComparisonDataframe(unittest.TestCase):
    """Tests for build_comparison_dataframe."""

    def setUp(self):
        self.data = build_comparison_dataframe(
            gof_constrained_locs=[10, 20],
            gof_constrained_muts=[30, 40],
            lof_constrained_locs=[15],
            lof_constrained_muts=[45],
            gof_unconstrained_pos=[1000, 2000],
            gof_unconstrained_muts=[3000, 6000],
            lof_unconstrained_pos=[1500],
            lof_unconstrained_muts=[4500],
        )

    def test_row_count(self):
        """DataFrame has one row per gene per condition."""
        self.assertEqual(len(self.data), 6)

    def test_columns_present(self):
        """Required columns are present."""
        for col in ("group", "condition", "positions", "mutations"):
            self.assertIn(col, self.data.columns)

    def test_group_labels(self):
        """Only GOF and LOF group labels are present."""
        self.assertEqual(set(self.data["group"].unique()), {"GOF", "LOF"})

    def test_condition_labels(self):
        """Only Constrained and Unconstrained condition labels are present."""
        self.assertEqual(
            set(self.data["condition"].unique()), {"Constrained", "Unconstrained"}
        )

    def test_constrained_gof_positions(self):
        """GOF constrained positions match input."""
        subset = self.data[
            (self.data["group"] == "GOF") & (self.data["condition"] == "Constrained")
        ]
        self.assertListEqual(sorted(subset["positions"].tolist()), [10, 20])

    def test_unconstrained_lof_mutations(self):
        """LOF unconstrained mutations match input."""
        subset = self.data[
            (self.data["group"] == "LOF") & (self.data["condition"] == "Unconstrained")
        ]
        self.assertListEqual(subset["mutations"].tolist(), [4500])


class TestPvalueLabel(unittest.TestCase):
    """Tests for _pvalue_label."""

    def test_three_stars_below_0001(self):
        self.assertEqual(_pvalue_label(0.0001), "***")

    def test_two_stars_below_001(self):
        self.assertEqual(_pvalue_label(0.005), "**")

    def test_one_star_below_005(self):
        self.assertEqual(_pvalue_label(0.03), "*")

    def test_ns_at_005(self):
        self.assertEqual(_pvalue_label(0.05), "ns")

    def test_ns_above_005(self):
        self.assertEqual(_pvalue_label(0.99), "ns")

    def test_boundary_0001_is_three_stars(self):
        self.assertEqual(_pvalue_label(0.001 - 1e-10), "***")

    def test_boundary_001_is_two_stars(self):
        self.assertEqual(_pvalue_label(0.001), "**")


class TestAddSignificanceBrackets(unittest.TestCase):
    """Smoke tests for _add_significance_brackets."""

    def _make_figure_and_data(self):
        gof_c = list(range(10, 20))
        lof_c = list(range(15, 25))
        gof_u = [v * 100 for v in gof_c]
        lof_u = [v * 100 for v in lof_c]
        data = build_comparison_dataframe(gof_c, [v * 3 for v in gof_c], lof_c, [v * 3 for v in lof_c],
                                          gof_u, [v * 3 for v in gof_u], lof_u, [v * 3 for v in lof_u])
        test_results = run_statistical_tests(gof_c, lof_c, gof_u, lof_u, "positions")
        palette = {"Constrained": "#4C72B0", "Unconstrained": "#DD8452"}
        fig, ax = _barplot_figure(data, "positions", "y", "title", palette,
                                  ["GOF", "LOF"], ["Constrained", "Unconstrained"])
        return fig, ax, data, test_results

    def test_runs_without_error(self):
        """_add_significance_brackets completes without raising."""
        fig, ax, data, test_results = self._make_figure_and_data()
        _add_significance_brackets(ax, data, "positions", test_results)
        plt.close(fig)

    def test_ylim_expanded(self):
        """y-axis upper limit is expanded after annotation."""
        fig, ax, data, test_results = self._make_figure_and_data()
        ylim_before = ax.get_ylim()[1]
        _add_significance_brackets(ax, data, "positions", test_results)
        self.assertGreater(ax.get_ylim()[1], ylim_before)
        plt.close(fig)


class TestBarplotFigureWithAx(unittest.TestCase):
    """Ax-injection behaviour for _barplot_figure."""

    def _data(self):
        return build_comparison_dataframe(
            gof_constrained_locs=[10, 20],
            gof_constrained_muts=[30, 60],
            lof_constrained_locs=[15, 25],
            lof_constrained_muts=[45, 75],
            gof_unconstrained_pos=[1000, 2000],
            gof_unconstrained_muts=[3000, 6000],
            lof_unconstrained_pos=[1500, 2500],
            lof_unconstrained_muts=[4500, 7500],
        )

    @patch("matplotlib.pyplot.savefig")
    def test_draws_onto_provided_ax_without_saving(self, mock_savefig):
        """Passing an ax draws onto it, returns its figure, and saves nothing."""
        palette = {"Constrained": "#4C72B0", "Unconstrained": "#DD8452"}
        fig, ax = plt.subplots()
        try:
            returned_fig, returned_ax = _barplot_figure(
                self._data(), "positions", "y", "title", palette,
                ["GOF", "LOF"], ["Constrained", "Unconstrained"], ax=ax,
            )
            self.assertIs(returned_ax, ax)
            self.assertIs(returned_fig, ax.get_figure())
            self.assertGreater(len(ax.patches), 0)  # bars drawn
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)


class TestRunStatisticalTests(unittest.TestCase):
    """Tests for run_statistical_tests."""

    def _make_result(self, gof_constrained, lof_constrained, gof_unconstrained, lof_unconstrained, metric="positions"):
        return run_statistical_tests(
            gof_constrained,
            lof_constrained,
            gof_unconstrained,
            lof_unconstrained,
            metric,
        )

    def test_returns_four_rows(self):
        """Result has exactly four rows: two Wilcoxon and two Mann-Whitney."""
        result = self._make_result(
            [10, 20, 30], [12, 22, 32], [1000, 2000, 3000], [1200, 2200, 3200]
        )
        self.assertEqual(len(result), 4)

    def test_expected_columns(self):
        """DataFrame contains required columns."""
        result = self._make_result(
            [10, 20], [12, 22], [1000, 2000], [1200, 2200]
        )
        for col in ("test", "group_a", "group_b", "statistic", "p_value", "metric"):
            self.assertIn(col, result.columns)

    def test_metric_label_propagated(self):
        """The metric column carries the supplied label in all rows."""
        result = self._make_result(
            [10, 20], [12, 22], [1000, 2000], [1200, 2200], metric="mutations"
        )
        self.assertTrue((result["metric"] == "mutations").all())

    def test_two_wilcoxon_and_two_mannwhitney_rows(self):
        """Result contains exactly two rows per test type."""
        result = self._make_result(
            [10, 20, 30], [12, 22, 32], [1000, 2000, 3000], [1200, 2200, 3200]
        )
        self.assertEqual((result["test"] == "Wilcoxon signed-rank").sum(), 2)
        self.assertEqual((result["test"] == "Mann-Whitney U").sum(), 2)

    def test_wilcoxon_significant_for_clearly_different_data(self):
        """Wilcoxon p-value is small when constrained values are consistently smaller."""
        constrained = list(range(10, 20))
        unconstrained = [v * 100 for v in constrained]
        result = self._make_result(
            constrained, constrained, unconstrained, unconstrained
        )
        wilcoxon_rows = result[result["test"] == "Wilcoxon signed-rank"]
        for p_value in wilcoxon_rows["p_value"]:
            self.assertLess(p_value, 0.05)

    def test_mannwhitney_not_significant_for_equal_groups(self):
        """Mann-Whitney p-value is not significant when GOF and LOF distributions match."""
        # constrained and unconstrained differ (so Wilcoxon is valid), but
        # GOF and LOF constrained are drawn from the same distribution.
        gof_constrained = list(range(10, 25))
        lof_constrained = list(range(10, 25))
        gof_unconstrained = [v * 10 for v in gof_constrained]
        lof_unconstrained = [v * 10 for v in lof_constrained]
        result = self._make_result(
            gof_constrained, lof_constrained, gof_unconstrained, lof_unconstrained
        )
        mwu_rows = result[result["test"] == "Mann-Whitney U"]
        for p_value in mwu_rows["p_value"]:
            self.assertGreater(p_value, 0.05)

    def test_p_values_are_between_zero_and_one(self):
        """All p-values are valid probabilities."""
        result = self._make_result(
            [10, 20, 30], [15, 25, 35], [1000, 2000, 3000], [1200, 2200, 3200]
        )
        for p_value in result["p_value"]:
            self.assertGreaterEqual(p_value, 0.0)
            self.assertLessEqual(p_value, 1.0)


if __name__ == "__main__":
    unittest.main()
