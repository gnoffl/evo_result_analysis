"""Tests for actual_mutation_region_breakdown.py."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from workflows.evo_alg_pooled_plots.natural_unconstrained_comparison.actual_mutation_region_breakdown import (
    build_group_dataframe,
    build_summary_dataframe,
    collect_actual_mutation_region_counts,
    load_most_mutated_individual,
    load_vcf_positions_by_gene,
    plot_actual_mutation_allowance,
)
from workflows.evo_alg_pooled_plots.natural_unconstrained_comparison.region_mutation_breakdown import (
    REGION_ORDER,
)

_MODULE = (
    "workflows.evo_alg_pooled_plots"
    ".natural_unconstrained_comparison"
    ".actual_mutation_region_breakdown"
)


def _write_vcf(directory: Path, name: str, positions: list) -> None:
    """Write a minimal VCF with a header and one record per position."""
    lines = ["##fileformat=VCFv4.2", "#CHROM\tPOS\tID\tREF\tALT"]
    for position in positions:
        lines.append(f"chr1\t{position}\t.\tA\tG")
    (directory / name).write_text("\n".join(lines) + "\n")


class TestLoadVcfPositionsByGene(unittest.TestCase):
    """Tests for load_vcf_positions_by_gene."""

    def test_maps_core_gene_id_to_positions(self):
        """VCF positions are keyed by the core gene id, not the full filename."""
        with tempfile.TemporaryDirectory() as tmp:
            vcf_dir = Path(tmp)
            _write_vcf(vcf_dir, "1_AT1G01720_gene:100-200.vcf", [10, 500])
            result = load_vcf_positions_by_gene(vcf_dir)

        self.assertEqual(set(result.keys()), {"1_AT1G01720"})
        self.assertEqual(result["1_AT1G01720"], {10, 500})

    def test_multiple_genes(self):
        """Each VCF file contributes its own entry."""
        with tempfile.TemporaryDirectory() as tmp:
            vcf_dir = Path(tmp)
            _write_vcf(vcf_dir, "1_GENEA_gene:1-2.vcf", [1])
            _write_vcf(vcf_dir, "2_GENEB_gene:3-4.vcf", [2, 3])
            result = load_vcf_positions_by_gene(vcf_dir)

        self.assertEqual(result["1_GENEA"], {1})
        self.assertEqual(result["2_GENEB"], {2, 3})


class TestLoadMostMutatedIndividual(unittest.TestCase):
    """Tests for load_most_mutated_individual."""

    def test_reads_front_zero_and_diffs_against_reference(self):
        """front[0] (most mutations) is read directly and diffed correctly."""
        reference = "AAAA"
        mock_fasta = MagicMock()
        mock_fasta.__getitem__.return_value.__getitem__.return_value.seq = reference

        with tempfile.TemporaryDirectory() as tmp:
            gene_dir = Path(tmp)
            (gene_dir / "saved_populations").mkdir()
            pareto_front = [["ATAA", 0.5, 1.0], ["AAAA", 0.1, 0.0]]
            with open(gene_dir / "saved_populations" / "pareto_front.json", "w") as handle:
                json.dump(pareto_front, handle)

            with patch(f"{_MODULE}.Fasta", return_value=mock_fasta):
                individual = load_most_mutated_individual(gene_dir)

        self.assertEqual(individual.mutations, [(1, "A", "T")])
        self.assertEqual(individual.fitness, 0.5)


class TestCollectActualMutationRegionCounts(unittest.TestCase):
    """Tests for collect_actual_mutation_region_counts."""

    def _make_gene_dir(
        self, run_dir: Path, gene_name: str, reference: str, mutated: str
    ) -> None:
        gene_dir = run_dir / gene_name
        (gene_dir / "saved_populations").mkdir(parents=True)
        pareto_front = [[mutated, 0.9, 1.0], [reference, 0.1, 0.0]]
        with open(gene_dir / "saved_populations" / "pareto_front.json", "w") as handle:
            json.dump(pareto_front, handle)

    def test_bins_mutations_and_flags_allowed_positions(self):
        """Mutations are assigned to regions and checked against VCF positions."""
        reference = "N" * 3020
        mutated = list(reference)
        mutated[0] = "A"  # 1-based position 1 -> promoter, allowed (in VCF)
        mutated[1] = "C"  # 1-based position 2 -> promoter, not allowed
        mutated[1000] = "T"  # 1-based position 1001 -> 5'-UTR, allowed
        mutated = "".join(mutated)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._make_gene_dir(run_dir, "1_GENE_suffix", reference, mutated)
            mock_fasta = MagicMock()
            mock_fasta.__getitem__.return_value.__getitem__.return_value.seq = reference

            with patch(f"{_MODULE}.Fasta", return_value=mock_fasta):
                result = collect_actual_mutation_region_counts(
                    run_dir, {"1_GENE": {1, 1001}}
                )

        by_region = result.set_index("region")
        self.assertEqual(by_region.loc["promoter", "total_mutations"], 2)
        self.assertEqual(by_region.loc["promoter", "allowed_mutations"], 1)
        self.assertEqual(by_region.loc["5'-UTR", "total_mutations"], 1)
        self.assertEqual(by_region.loc["5'-UTR", "allowed_mutations"], 1)
        self.assertEqual(by_region.loc["3'-UTR", "total_mutations"], 0)
        self.assertEqual(by_region.loc["terminator", "total_mutations"], 0)

    def test_skips_gene_without_matching_vcf(self):
        """Genes with no VCF entry are dropped, not counted as zero."""
        reference = "N" * 3020
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._make_gene_dir(run_dir, "1_GENE_suffix", reference, reference)
            mock_fasta = MagicMock()
            mock_fasta.__getitem__.return_value.__getitem__.return_value.seq = reference

            with patch(f"{_MODULE}.Fasta", return_value=mock_fasta):
                result = collect_actual_mutation_region_counts(run_dir, {})

        self.assertTrue(result.empty)

    def test_skips_non_digit_directories(self):
        """Directories not starting with a digit are ignored."""
        reference = "N" * 3020
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "problem_configs.json").touch()
            self._make_gene_dir(run_dir, "1_GENE_suffix", reference, reference)
            self._make_gene_dir(run_dir, "Pt_GENE_suffix", reference, reference)
            mock_fasta = MagicMock()
            mock_fasta.__getitem__.return_value.__getitem__.return_value.seq = reference

            with patch(f"{_MODULE}.Fasta", return_value=mock_fasta):
                result = collect_actual_mutation_region_counts(
                    run_dir, {"1_GENE": set(), "Pt_GENE": set()}
                )

        self.assertEqual(set(result["gene"].unique()), {"1_GENE_suffix"})


class TestBuildGroupDataframe(unittest.TestCase):
    """Tests for build_group_dataframe."""

    def _frame(self, total: int, allowed: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "gene": "g",
                    "region": region,
                    "total_mutations": total,
                    "allowed_mutations": allowed,
                }
                for region in REGION_ORDER
            ]
        )

    def test_percent_allowed_computed(self):
        data = build_group_dataframe(self._frame(4, 1), self._frame(2, 2))
        gof_percent = data[data["group"] == "GOF"]["percent_allowed"].unique()
        lof_percent = data[data["group"] == "LOF"]["percent_allowed"].unique()
        self.assertEqual(list(gof_percent), [25.0])
        self.assertEqual(list(lof_percent), [100.0])

    def test_zero_total_is_nan(self):
        data = build_group_dataframe(self._frame(0, 0), self._frame(0, 0))
        self.assertTrue(data["percent_allowed"].isna().all())

    def test_group_labels(self):
        data = build_group_dataframe(self._frame(1, 1), self._frame(1, 0))
        self.assertEqual(set(data["group"].unique()), {"GOF", "LOF"})


class TestBuildSummaryDataframe(unittest.TestCase):
    """Tests for build_summary_dataframe."""

    def test_pooled_percent(self):
        rows = []
        # Gene 1: 4 mutations, 1 allowed (25%); Gene 2: 2 mutations, 2 allowed (100%)
        for gene, total, allowed in [("g1", 4, 1), ("g2", 2, 2)]:
            rows.append(
                {
                    "gene": gene,
                    "group": "GOF",
                    "region": "promoter",
                    "total_mutations": total,
                    "allowed_mutations": allowed,
                    "percent_allowed": allowed / total * 100,
                }
            )
        data = pd.DataFrame(rows)
        summary = build_summary_dataframe(data)
        row = summary.loc[("GOF", "promoter")]

        self.assertEqual(row["n_genes"], 2)
        # Pooled: (1 + 2) allowed out of (4 + 2) total = 50%
        self.assertAlmostEqual(row["pooled_percent"], 50.0)
        self.assertEqual(list(summary.columns), ["n_genes", "pooled_percent"])

    def test_zero_mutation_genes_excluded_from_gene_count(self):
        rows = [
            {
                "gene": "g1",
                "group": "GOF",
                "region": "promoter",
                "total_mutations": 2,
                "allowed_mutations": 1,
                "percent_allowed": 50.0,
            },
            {
                "gene": "g2",
                "group": "GOF",
                "region": "promoter",
                "total_mutations": 0,
                "allowed_mutations": 0,
                "percent_allowed": float("nan"),
            },
        ]
        data = pd.DataFrame(rows)
        summary = build_summary_dataframe(data)
        row = summary.loc[("GOF", "promoter")]

        self.assertEqual(row["n_genes"], 1)
        self.assertAlmostEqual(row["pooled_percent"], 50.0)


class TestPlotActualMutationAllowance(unittest.TestCase):
    """Smoke tests for plot_actual_mutation_allowance."""

    def _data(self) -> pd.DataFrame:
        rows = []
        for group, percent in (("GOF", 25.0), ("LOF", 75.0)):
            for gene in ("g1", "g2"):
                for region in REGION_ORDER:
                    rows.append(
                        {
                            "gene": gene,
                            "group": group,
                            "region": region,
                            "total_mutations": 4,
                            "allowed_mutations": 1,
                            "percent_allowed": percent,
                        }
                    )
        return pd.DataFrame(rows)

    def test_saves_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            plot_actual_mutation_allowance(self._data(), output_dir, fmt="png")
            self.assertTrue((output_dir / "actual_mutation_allowance.png").exists())
        plt.close("all")

    def _axes_after_plot(self, **kwargs):
        with patch(f"{_MODULE}.plt.close"), tempfile.TemporaryDirectory() as tmp:
            plot_actual_mutation_allowance(self._data(), Path(tmp), **kwargs)
        return plt.gcf().axes[0]

    def test_legend_hidden_when_disabled(self):
        ax = self._axes_after_plot(show_legend=False)
        self.assertIsNone(ax.get_legend())
        plt.close("all")

    def test_title_hidden_when_disabled(self):
        ax = self._axes_after_plot(show_title=False)
        self.assertEqual(ax.get_title(), "")
        plt.close("all")

    def test_rows_with_nan_percent_are_dropped(self):
        """Genes with zero mutations in a region (NaN percent) are not plotted."""
        data = self._data()
        data.loc[0, "percent_allowed"] = float("nan")
        with patch(f"{_MODULE}.plt.close"), tempfile.TemporaryDirectory() as tmp:
            plot_actual_mutation_allowance(data, Path(tmp))
        plt.close("all")


class TestPlotActualMutationAllowanceWithAx(unittest.TestCase):
    """Ax-injection behaviour for plot_actual_mutation_allowance."""

    def _data(self) -> pd.DataFrame:
        rows = []
        for group in ("GOF", "LOF"):
            for gene in ("g1", "g2"):
                for region in REGION_ORDER:
                    rows.append(
                        {
                            "gene": gene,
                            "group": group,
                            "region": region,
                            "total_mutations": 4,
                            "allowed_mutations": 1,
                            "percent_allowed": 25.0,
                        }
                    )
        return pd.DataFrame(rows)

    @patch("matplotlib.pyplot.savefig")
    def test_draws_onto_provided_ax_without_saving(self, mock_savefig):
        fig, ax = plt.subplots()
        try:
            returned_ax = plot_actual_mutation_allowance(self._data(), ax=ax)
            self.assertIs(returned_ax, ax)
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)

    def test_provided_ax_not_closed(self):
        fig, ax = plt.subplots()
        try:
            plot_actual_mutation_allowance(self._data(), ax=ax)
            self.assertTrue(plt.fignum_exists(fig.number))
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
