"""Tests for region_mutation_breakdown.py."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import PathPatch

from workflows.evo_alg_pooled_plots.natural_unconstrained_comparison.region_mutation_breakdown import (
    REGION_ORDER,
    assign_region,
    build_region_dataframe,
    collect_unconstrained_region_counts,
    collect_vcf_region_counts,
    plot_region_breakdown,
    plot_region_breakdown_boxes,
)

_MODULE = (
    "workflows.evo_alg_pooled_plots"
    ".natural_unconstrained_comparison"
    ".region_mutation_breakdown"
)


class TestAssignRegion(unittest.TestCase):
    """Boundary tests for assign_region (1-based coordinates)."""

    def test_promoter_lower_bound(self):
        self.assertEqual(assign_region(1), "promoter")

    def test_promoter_upper_bound(self):
        self.assertEqual(assign_region(1000), "promoter")

    def test_five_utr_lower_bound(self):
        self.assertEqual(assign_region(1001), "5'-UTR")

    def test_five_utr_upper_bound(self):
        self.assertEqual(assign_region(1500), "5'-UTR")

    def test_gap_lower_edge_is_none(self):
        self.assertIsNone(assign_region(1501))

    def test_gap_upper_edge_is_none(self):
        self.assertIsNone(assign_region(1520))

    def test_three_utr_lower_bound(self):
        self.assertEqual(assign_region(1521), "3'-UTR")

    def test_three_utr_upper_bound(self):
        self.assertEqual(assign_region(2020), "3'-UTR")

    def test_terminator_lower_bound(self):
        self.assertEqual(assign_region(2021), "terminator")

    def test_terminator_upper_bound(self):
        self.assertEqual(assign_region(3020), "terminator")

    def test_below_range_is_none(self):
        self.assertIsNone(assign_region(0))

    def test_above_range_is_none(self):
        self.assertIsNone(assign_region(3021))


class TestCollectVcfRegionCounts(unittest.TestCase):
    """Tests for collect_vcf_region_counts."""

    def _write_vcf(self, directory: Path, name: str, positions: list) -> None:
        """Write a minimal VCF with a header and one record per position."""
        lines = ["##fileformat=VCFv4.2", "#CHROM\tPOS\tID\tREF\tALT"]
        for position in positions:
            lines.append(f"chr1\t{position}\t.\tA\tG")
        (directory / name).write_text("\n".join(lines) + "\n")

    def test_bins_positions_across_regions(self):
        """Records are counted into the region their position maps to."""
        with tempfile.TemporaryDirectory() as tmp:
            vcf_dir = Path(tmp)
            # promoter x2, 5'-UTR x1, gap (dropped) x1, terminator x1
            self._write_vcf(vcf_dir, "1_GENE.vcf", [10, 500, 1200, 1510, 3000])
            result = collect_vcf_region_counts(vcf_dir)

        counts = dict(zip(result["region"], result["count"]))
        self.assertEqual(counts["promoter"], 2)
        self.assertEqual(counts["5'-UTR"], 1)
        self.assertEqual(counts["3'-UTR"], 0)
        self.assertEqual(counts["terminator"], 1)

    def test_gap_positions_are_dropped(self):
        """Positions in the 1501-1520 gap are not counted anywhere."""
        with tempfile.TemporaryDirectory() as tmp:
            vcf_dir = Path(tmp)
            self._write_vcf(vcf_dir, "1_GENE.vcf", [1501, 1510, 1520])
            result = collect_vcf_region_counts(vcf_dir)

        self.assertEqual(result["count"].sum(), 0)

    def test_every_region_present_per_gene(self):
        """Each gene contributes one row per region even with no records."""
        with tempfile.TemporaryDirectory() as tmp:
            vcf_dir = Path(tmp)
            self._write_vcf(vcf_dir, "1_GENE.vcf", [10])
            result = collect_vcf_region_counts(vcf_dir)

        self.assertListEqual(sorted(result["region"].tolist()), sorted(REGION_ORDER))


class TestCollectUnconstrainedRegionCounts(unittest.TestCase):
    """Tests for collect_unconstrained_region_counts."""

    def test_counts_possible_mutations_by_region(self):
        """Non-N positions are binned by region and multiplied by 3 SNPs."""
        # Build a sequence: promoter positions 1-2 non-N (x3 = 6 mutations),
        # one 5'-UTR and one terminator position; the rest N.
        sequence = list("N" * 3020)
        sequence[0] = "A"  # 1-based position 1 -> promoter
        sequence[1] = "C"  # position 2 -> promoter
        sequence[1000] = "T"  # position 1001 -> 5'-UTR
        sequence[3019] = "G"  # position 3020 -> terminator
        mock_fasta = MagicMock()
        mock_fasta.__getitem__.return_value = "".join(sequence)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "1_GENE").mkdir()
            (run_dir / "1_GENE" / "reference_sequence.fa").touch()
            with patch(f"{_MODULE}.Fasta", return_value=mock_fasta):
                result = collect_unconstrained_region_counts(run_dir)

        counts = dict(zip(result["region"], result["count"]))
        self.assertEqual(counts["promoter"], 6)  # 2 positions x 3 SNPs
        self.assertEqual(counts["5'-UTR"], 3)
        self.assertEqual(counts["3'-UTR"], 0)
        self.assertEqual(counts["terminator"], 3)

    def test_skips_non_digit_directories(self):
        """Directories not starting with a digit are ignored."""
        mock_fasta = MagicMock()
        mock_fasta.__getitem__.return_value = "A" + "N" * 3019

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "problem_configs.json").touch()
            (run_dir / "1_GENE").mkdir()
            (run_dir / "1_GENE" / "reference_sequence.fa").touch()
            with patch(f"{_MODULE}.Fasta", return_value=mock_fasta):
                result = collect_unconstrained_region_counts(run_dir)

        self.assertEqual(set(result["gene"].unique()), {"1_GENE"})


class TestBuildRegionDataframe(unittest.TestCase):
    """Tests for build_region_dataframe."""

    def _frame(self, count: int) -> pd.DataFrame:
        return pd.DataFrame(
            [{"gene": "g", "region": region, "count": count} for region in REGION_ORDER]
        )

    def setUp(self):
        self.data = build_region_dataframe(
            gof_vcf=self._frame(1),
            lof_vcf=self._frame(2),
            gof_unconstrained=self._frame(3),
            lof_unconstrained=self._frame(4),
        )

    def test_columns_present(self):
        for col in ("gene", "group", "condition", "region", "count"):
            self.assertIn(col, self.data.columns)

    def test_group_and_condition_labels(self):
        self.assertEqual(set(self.data["group"].unique()), {"GOF", "LOF"})
        self.assertEqual(
            set(self.data["condition"].unique()), {"Constrained", "Unconstrained"}
        )

    def test_row_count(self):
        """Four source frames x four regions each."""
        self.assertEqual(len(self.data), 4 * len(REGION_ORDER))

    def test_condition_count_mapping(self):
        """Each source frame's counts land under the right group/condition."""
        subset = self.data[
            (self.data["group"] == "LOF")
            & (self.data["condition"] == "Unconstrained")
        ]
        self.assertTrue((subset["count"] == 4).all())


class TestPlotRegionBreakdown(unittest.TestCase):
    """Smoke tests for plot_region_breakdown."""

    def _data(self) -> pd.DataFrame:
        rows = []
        for group, condition, count in (
            ("GOF", "Constrained", 5),
            ("GOF", "Unconstrained", 50),
        ):
            for gene in ("g1", "g2"):
                for region in REGION_ORDER:
                    rows.append(
                        {
                            "gene": gene,
                            "group": group,
                            "condition": condition,
                            "region": region,
                            "count": count,
                        }
                    )
        return pd.DataFrame(rows)

    def test_saves_one_figure_per_group(self):
        """Plotting a group writes a figure file and draws bars."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            plot_region_breakdown(self._data(), "GOF", output_dir, fmt="png")
            self.assertTrue((output_dir / "region_breakdown_GOF.png").exists())
        plt.close("all")

    def _axes_after_plot(self, **kwargs):
        """Run plot_region_breakdown but keep the figure alive for inspection.

        Patches the module's plt.close to a no-op so the figure built inside the
        function survives; returns its first axes.
        """
        with patch(f"{_MODULE}.plt.close"), tempfile.TemporaryDirectory() as tmp:
            plot_region_breakdown(self._data(), "GOF", Path(tmp), **kwargs)
        return plt.gcf().axes[0]

    def test_legend_hidden_when_disabled(self):
        """show_legend=False leaves no legend on the axes."""
        ax = self._axes_after_plot(show_legend=False)
        self.assertIsNone(ax.get_legend())
        plt.close("all")

    def test_legend_shown_by_default(self):
        """A legend is present when show_legend is left at its default."""
        ax = self._axes_after_plot()
        self.assertIsNotNone(ax.get_legend())
        plt.close("all")

    def test_title_hidden_when_disabled(self):
        """show_title=False leaves an empty axis title."""
        ax = self._axes_after_plot(show_title=False)
        self.assertEqual(ax.get_title(), "")
        plt.close("all")

    def test_custom_palette_applied(self):
        """A custom condition palette colours the bars accordingly."""
        custom = {"Constrained": "#000000", "Unconstrained": "#ffffff"}
        ax = self._axes_after_plot(palette=custom)
        face_colors = {
            tuple(round(channel, 3) for channel in patch_obj.get_facecolor())
            for patch_obj in ax.patches
        }
        # Black and white (alpha 1) must both appear among the bar faces.
        self.assertIn((0.0, 0.0, 0.0, 1.0), face_colors)
        self.assertIn((1.0, 1.0, 1.0, 1.0), face_colors)
        plt.close("all")


class TestPlotRegionBreakdownWithAx(unittest.TestCase):
    """Ax-injection behaviour for plot_region_breakdown."""

    def _data(self) -> pd.DataFrame:
        rows = []
        for condition, count in (("Constrained", 5), ("Unconstrained", 50)):
            for gene in ("g1", "g2"):
                for region in REGION_ORDER:
                    rows.append(
                        {
                            "gene": gene,
                            "group": "GOF",
                            "condition": condition,
                            "region": region,
                            "count": count,
                        }
                    )
        return pd.DataFrame(rows)

    @patch("matplotlib.pyplot.savefig")
    def test_draws_onto_provided_ax_without_saving(self, mock_savefig):
        """Passing an ax draws onto it, returns it, and saves nothing."""
        fig, ax = plt.subplots()
        try:
            returned_ax = plot_region_breakdown(self._data(), "GOF", ax=ax)
            self.assertIs(returned_ax, ax)
            self.assertGreater(len(ax.patches), 0)  # bars drawn
            mock_savefig.assert_not_called()
        finally:
            plt.close(fig)

    def test_provided_ax_not_closed(self):
        """The caller's figure survives the call (not closed internally)."""
        fig, ax = plt.subplots()
        try:
            plot_region_breakdown(self._data(), "GOF", ax=ax)
            self.assertTrue(plt.fignum_exists(fig.number))
        finally:
            plt.close(fig)


class TestPlotRegionBreakdownBoxes(unittest.TestCase):
    """Behaviour of the boxplot variant of the region breakdown."""

    def _data(self) -> pd.DataFrame:
        """Two genes per condition with distinct counts, so boxes have spread."""
        rows = []
        for condition, counts in (
            ("Constrained", (4, 6)),
            ("Unconstrained", (40, 60)),
        ):
            for gene, count in zip(("g1", "g2"), counts):
                for region in REGION_ORDER:
                    rows.append(
                        {
                            "gene": gene,
                            "group": "GOF",
                            "condition": condition,
                            "region": region,
                            "count": count,
                        }
                    )
        return pd.DataFrame(rows)

    def test_saves_figure_under_boxes_name(self):
        """Standalone mode writes its own filename, distinct from the bar plot."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            plot_region_breakdown_boxes(self._data(), "GOF", output_dir, fmt="png")
            self.assertTrue(
                (output_dir / "region_breakdown_boxes_GOF.png").exists()
            )
        plt.close("all")

    @staticmethod
    def _boxes(ax: plt.Axes) -> list:
        """Return the box artists, excluding the legend swatch rectangles."""
        return [
            patch_obj
            for patch_obj in ax.patches
            if isinstance(patch_obj, PathPatch)
        ]

    def test_box_outlines_take_the_palette_colour(self):
        """Every cell gets a box outlined in its hue colour over a light fill."""
        palette = {"Constrained": "#000000", "Unconstrained": "#ff8800"}
        fig, ax = plt.subplots()
        try:
            plot_region_breakdown_boxes(self._data(), "GOF", ax=ax, palette=palette)
            boxes = self._boxes(ax)
            self.assertEqual(len(boxes), 2 * len(REGION_ORDER))
            edge_colors = Counter(
                tuple(round(channel, 3) for channel in box.get_edgecolor())
                for box in boxes
            )
            expected_edges = Counter(
                {
                    tuple(round(channel, 3) for channel in to_rgba(colour)): len(
                        REGION_ORDER
                    )
                    for colour in palette.values()
                }
            )
            self.assertEqual(edge_colors, expected_edges)
        finally:
            plt.close(fig)

    def test_box_fill_is_lightened(self):
        """The fill keeps the hue but at reduced opacity, so lines read on it."""
        fig, ax = plt.subplots()
        try:
            plot_region_breakdown_boxes(self._data(), "GOF", ax=ax)
            for box in self._boxes(ax):
                fill_alpha = box.get_facecolor()[3]
                edge_color = box.get_edgecolor()
                self.assertLess(fill_alpha, 1.0)
                self.assertGreater(fill_alpha, 0.0)
                self.assertEqual(box.get_facecolor()[:3], edge_color[:3])
        finally:
            plt.close(fig)

    def test_legend_swatches_match_the_boxes(self):
        """The legend shows the box style, not a solid block of hue.

        Seaborn's hue swatches are plain ``Rectangle`` patches, so they are the
        patches that are *not* boxes.
        """
        fig, ax = plt.subplots()
        try:
            plot_region_breakdown_boxes(self._data(), "GOF", ax=ax)
            swatches = [
                patch_obj
                for patch_obj in ax.patches
                if not isinstance(patch_obj, PathPatch)
            ]
            box_styles = {
                (
                    tuple(round(channel, 3) for channel in box.get_facecolor()),
                    tuple(round(channel, 3) for channel in box.get_edgecolor()),
                )
                for box in self._boxes(ax)
            }
            self.assertEqual(len(swatches), 2)
            for swatch in swatches:
                style = (
                    tuple(round(channel, 3) for channel in swatch.get_facecolor()),
                    tuple(round(channel, 3) for channel in swatch.get_edgecolor()),
                )
                self.assertIn(style, box_styles)
        finally:
            plt.close(fig)

    def test_whiskers_are_black_and_medians_take_the_hue(self):
        """Whiskers and caps stay black; the median carries its box's colour."""
        palette = {"Constrained": "#123456", "Unconstrained": "#654321"}
        fig, ax = plt.subplots()
        try:
            plot_region_breakdown_boxes(self._data(), "GOF", ax=ax, palette=palette)
            non_black = [
                to_rgba(line.get_color())
                for line in ax.lines
                if to_rgba(line.get_color()) != to_rgba("black")
            ]
            # One median per region x condition cell, none of them black.
            self.assertEqual(len(non_black), 2 * len(REGION_ORDER))
            self.assertEqual(
                set(non_black), {to_rgba(colour) for colour in palette.values()}
            )
        finally:
            plt.close(fig)

    def test_gene_points_not_drawn_by_default(self):
        """The default panel is boxes only, without a dot per gene."""
        fig, ax = plt.subplots()
        try:
            plot_region_breakdown_boxes(self._data(), "GOF", ax=ax)
            self.assertEqual(len(ax.collections), 0)
        finally:
            plt.close(fig)

    def test_gene_points_can_be_enabled(self):
        """show_gene_points=True scatters every gene on top of the boxes."""
        fig, ax = plt.subplots()
        try:
            plot_region_breakdown_boxes(
                self._data(), "GOF", ax=ax, show_gene_points=True
            )
            scattered_genes = sum(
                len(collection.get_offsets()) for collection in ax.collections
            )
            self.assertEqual(scattered_genes, len(self._data()))
        finally:
            plt.close(fig)

    def test_y_axis_is_symlog_with_room_below_zero(self):
        """Counts span orders of magnitude, but zero must remain plottable.

        A plain log axis maps a gene with no possible mutation in a region to
        -inf and draws its whisker off the panel, so the axis is symlog with a
        linear band at the bottom. The limit sits slightly below 0 so a whisker
        ending at 0 is not hidden by the bottom spine, but stays inside the
        linear band so no tick appears below 0.
        """
        fig, ax = plt.subplots()
        try:
            plot_region_breakdown_boxes(self._data(), "GOF", ax=ax)
            lower_limit = ax.get_ylim()[0]
            self.assertEqual(ax.get_yscale(), "symlog")
            self.assertLess(lower_limit, 0.0)
            self.assertGreater(lower_limit, -1.0)
        finally:
            plt.close(fig)

    def test_zero_count_whisker_stays_inside_the_axes(self):
        """A gene with zero possible mutations does not push a whisker off-axis."""
        data = self._data()
        data.loc[data["gene"] == "g1", "count"] = 0
        fig, ax = plt.subplots()
        try:
            plot_region_breakdown_boxes(data, "GOF", ax=ax)
            lower_limit = ax.get_ylim()[0]
            for line in ax.lines:
                y_data = np.asarray(line.get_ydata(), dtype=float)
                finite = y_data[np.isfinite(y_data)]
                self.assertEqual(len(finite), len(y_data))
                if len(finite):
                    # Strictly inside: an endpoint on the spine reads as running
                    # off the panel.
                    self.assertGreater(finite.min(), lower_limit)
        finally:
            plt.close(fig)

    def test_region_ticks_in_fixed_order(self):
        """x tick labels follow REGION_ORDER."""
        fig, ax = plt.subplots()
        try:
            plot_region_breakdown_boxes(self._data(), "GOF", ax=ax)
            self.assertEqual(
                [label.get_text() for label in ax.get_xticklabels()], REGION_ORDER
            )
        finally:
            plt.close(fig)

    def test_other_group_rows_are_ignored(self):
        """Only the requested group's rows are drawn."""
        data = self._data()
        other_group = data.copy()
        other_group["group"] = "LOF"
        fig, ax = plt.subplots()
        try:
            plot_region_breakdown_boxes(
                pd.concat([data, other_group], ignore_index=True),
                "GOF",
                ax=ax,
                show_gene_points=True,
            )
            scattered_genes = sum(
                len(collection.get_offsets()) for collection in ax.collections
            )
            self.assertEqual(scattered_genes, len(data))
        finally:
            plt.close(fig)

    @patch("matplotlib.pyplot.savefig")
    def test_provided_ax_is_returned_and_not_saved(self, mock_savefig):
        """Ax injection draws in place, returns the axes, and saves nothing."""
        fig, ax = plt.subplots()
        try:
            returned_ax = plot_region_breakdown_boxes(self._data(), "GOF", ax=ax)
            self.assertIs(returned_ax, ax)
            mock_savefig.assert_not_called()
            self.assertTrue(plt.fignum_exists(fig.number))
        finally:
            plt.close(fig)

    def test_legend_and_title_hidden_when_disabled(self):
        """show_legend/show_title False leave no legend and an empty title."""
        fig, ax = plt.subplots()
        try:
            plot_region_breakdown_boxes(
                self._data(), "GOF", ax=ax, show_legend=False, show_title=False
            )
            self.assertIsNone(ax.get_legend())
            self.assertEqual(ax.get_title(), "")
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
