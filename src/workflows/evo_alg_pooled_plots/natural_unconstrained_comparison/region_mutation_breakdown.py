"""Break down the possible-mutation solution space by genomic region.

For each gene, count how many mutable positions / mutations fall into each
genomic region (promoter, 5'-UTR, 3'-UTR, terminator), separately for the
constrained (natural VCF) and unconstrained (any non-N reference position)
conditions, and for the GOF and LOF gene sets.

Coordinates are 1-based. VCF records are already 1-based; the reference
sequence is iterated 0-based and converted to 1-based (index + 1) before region
assignment, so both sources share a single coordinate frame. Positions in the
gap 1501-1520 are always N and carry no VCF record, so they self-exclude
(``assign_region`` returns None and they are dropped).

Region lengths differ (promoter/terminator 1000 bp, UTRs 500 bp); counts are raw
and this size difference is not normalized.
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.patches import PathPatch
from pyfaidx import Fasta

from workflows.evo_alg_pooled_plots.natural_unconstrained_comparison.natural_unconstrained_mutation_vis import (
    GOF_RUN_DIR,
    GOF_VCF_DIR,
    LOF_RUN_DIR,
    LOF_VCF_DIR,
)

# Region borders, 1-based inclusive. The gap 1501-1520 is intentionally
# unassigned (always N in the reference, never present in VCFs).
REGIONS: Tuple[Tuple[str, int, int], ...] = (
    ("promoter", 1, 1000),
    ("5'-UTR", 1001, 1500),
    ("3'-UTR", 1521, 2020),
    ("terminator", 2021, 3020),
)

REGION_ORDER = [name for name, _, _ in REGIONS]
OUTPUT_DIR = Path(__file__).parent / "region_mutation_breakdown"
FMT = "png"


def assign_region(position: int) -> Optional[str]:
    """Assign a 1-based sequence position to a genomic region.

    Args:
        position: 1-based sequence position.

    Returns:
        The region name, or None if the position falls in the unassigned gap
        (1501-1520) or outside the 1-3020 sequence range.
    """
    for name, start, end in REGIONS:
        if start <= position <= end:
            return name
    return None


def _empty_region_counts() -> dict:
    """Return a fresh region-count dict initialized to zero for every region.

    Returns:
        Dict mapping each region name to 0.
    """
    return {name: 0 for name in REGION_ORDER}


def collect_vcf_region_counts(vcf_dir: Path) -> pd.DataFrame:
    """Count VCF records per region for every gene in a directory.

    Each VCF record (a possible natural mutation) is assigned to a region by its
    1-based position. Records in the gap are dropped. Every gene contributes a
    count for every region (zero if none), so downstream means are unbiased.

    Args:
        vcf_dir: Directory containing one ``*.vcf`` file per gene.

    Returns:
        Tidy DataFrame with columns ``gene``, ``region``, ``count``.
    """
    rows = []
    for vcf_file in sorted(vcf_dir.glob("*.vcf")):
        counts = _empty_region_counts()
        for line in vcf_file.read_text().splitlines():
            if line.startswith("#"):
                continue
            position = int(line.split("\t")[1])
            region = assign_region(position)
            if region is not None:
                counts[region] += 1
        for region in REGION_ORDER:
            rows.append(
                {"gene": vcf_file.stem, "region": region, "count": counts[region]}
            )
    return pd.DataFrame(rows)


def collect_unconstrained_region_counts(run_dir: Path) -> pd.DataFrame:
    """Count possible unconstrained mutations per region for every gene.

    Iterates gene subdirectories (those starting with a digit) and reads their
    ``reference_sequence_full`` entry. Each non-N position (0-based index,
    converted to 1-based) is assigned to a region. To match the constrained
    condition (where each VCF record is one possible mutation), the per-region
    non-N position count is multiplied by 3, since each position admits three
    alternative SNPs. Every gene contributes a count for every region (zero if
    none).

    Args:
        run_dir: Top-level run directory containing one subdir per gene.

    Returns:
        Tidy DataFrame with columns ``gene``, ``region``, ``count`` (possible
        mutations = non-N positions x 3).
    """
    rows = []
    for gene_dir in sorted(run_dir.iterdir()):
        if not gene_dir.is_dir() or not gene_dir.name[0].isdigit():
            continue
        fasta = Fasta(str(gene_dir / "reference_sequence.fa"))
        sequence = str(fasta["reference_sequence_full"])
        counts = _empty_region_counts()
        for index, base in enumerate(sequence):
            if base.upper() == "N":
                continue
            region = assign_region(index + 1)
            if region is not None:
                counts[region] += 1
        for region in REGION_ORDER:
            rows.append(
                {"gene": gene_dir.name, "region": region, "count": counts[region] * 3}
            )
    return pd.DataFrame(rows)


def build_region_dataframe(
    gof_vcf: pd.DataFrame,
    lof_vcf: pd.DataFrame,
    gof_unconstrained: pd.DataFrame,
    lof_unconstrained: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble a tidy DataFrame combining all groups and conditions.

    Args:
        gof_vcf: Per-gene region counts for GOF constrained (VCF).
        lof_vcf: Per-gene region counts for LOF constrained (VCF).
        gof_unconstrained: Per-gene region counts for GOF unconstrained.
        lof_unconstrained: Per-gene region counts for LOF unconstrained.

    Returns:
        DataFrame with columns ``gene``, ``group``, ``condition``, ``region``,
        ``count``.
    """
    labeled = []
    for frame, group, condition in (
        (gof_vcf, "GOF", "Constrained"),
        (lof_vcf, "LOF", "Constrained"),
        (gof_unconstrained, "GOF", "Unconstrained"),
        (lof_unconstrained, "LOF", "Unconstrained"),
    ):
        annotated = frame.copy()
        annotated["group"] = group
        annotated["condition"] = condition
        labeled.append(annotated)
    combined = pd.concat(labeled, ignore_index=True)
    return pd.DataFrame(combined[["gene", "group", "condition", "region", "count"]])


DEFAULT_CONDITION_PALETTE = {"Constrained": "#4C72B0", "Unconstrained": "#DD8452"}


def plot_region_breakdown(
    data: pd.DataFrame,
    group: str,
    output_dir: Optional[Path] = None,
    fmt: str = "png",
    show_legend: bool = True,
    show_title: bool = True,
    palette: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Draw a grouped bar plot of region counts for one gene group.

    x-axis is region (fixed order), hue is condition (constrained vs
    unconstrained), bars show mean per-gene count with standard-deviation error
    bars.

    When ``ax`` is None (standalone) a new figure is created, styled with the
    module theme, and — if ``output_dir`` is given — saved to
    ``region_breakdown_<group>.<fmt>`` and closed. When ``ax`` is provided the
    bars are drawn onto it for panel composition: no theme is set, the figure is
    not laid out, saved, or closed, and nothing is written to disk.

    Args:
        data: Tidy DataFrame from ``build_region_dataframe``.
        group: Gene group to plot (``"GOF"`` or ``"LOF"``).
        output_dir: Directory to save the figure into. Ignored when ``ax`` is
            given; when None in standalone mode the figure is not saved.
        fmt: File format (e.g. ``"png"``, ``"svg"``, ``"pdf"``).
        show_legend: Whether to draw the condition legend. Set False when the
            figure is a panel whose legend is provided elsewhere.
        show_title: Whether to draw the axis title. Set False for panel use.
        palette: Mapping of condition name to colour. When None, uses
            ``DEFAULT_CONDITION_PALETTE``.
        ax: Axes to draw onto. When None a standalone figure is created.

    Returns:
        The Axes the bars were drawn onto.
    """
    group_data = pd.DataFrame(data[data["group"] == group])
    if palette is None:
        palette = DEFAULT_CONDITION_PALETTE

    own_figure = ax is None
    if ax is None:
        sns.set_theme(style="whitegrid", font_scale=1.1)
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    sns.barplot(
        data=group_data,
        x="region",
        y="count",
        hue="condition",
        order=REGION_ORDER,
        hue_order=["Constrained", "Unconstrained"],
        palette=palette,
        errorbar="sd",
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Genomic region")
    ax.set_ylabel("Possible mutations per gene (log scale)")
    if show_title:
        ax.set_title(f"{group}: possible mutations by region")
    if show_legend:
        ax.legend(title="Condition", loc="upper right", fontsize=8, title_fontsize=8)
    elif ax.get_legend() is not None:
        ax.get_legend().remove()  # type: ignore[union-attr]

    if own_figure:
        fig.tight_layout()
        if output_dir is not None:
            output_path = output_dir / f"region_breakdown_{group}.{fmt}"
            fig.savefig(output_path, bbox_inches="tight", dpi=150)  # type: ignore[union-attr]
            plt.close(fig)
            print(f"Saved: {output_path}")
    return ax


# Overlay geometry for the boxplot variant: the individual genes are drawn as
# small jittered dots on top of the boxes, so the reader sees the gene-level
# structure the box summarizes.
_GENE_POINT_SIZE = 1.8
_GENE_POINT_ALPHA = 0.55
_GENE_POINT_JITTER = 0.08

# Box styling. Whiskers, caps and fliers are black; the box outline and median
# carry the condition colour over a lightened fill of the same colour. The
# unconstrained counts are nearly constant across genes, so that box collapses
# onto its own median: a black median there would cover the only mark the
# condition has, which is why the median keeps the hue and the fill is lightened
# far enough for a coloured median to read inside it.
_BOX_LINE_WIDTH = 1.1
_BOX_FILL_ALPHA = 0.35
_FLIER_MARKER_SIZE = 2.5

# Symlog y-axis: counts are integers, so a linear band up to 1 carries only the
# value 0 (the genes with no possible mutation in a region) and everything else
# stays on the log part of the axis. The linear band is given less height than a
# full decade would get, since it holds a single value.
_SYMLOG_LINEAR_THRESHOLD = 1.0
_SYMLOG_LINEAR_SCALE = 0.4
# Headroom below zero, as a fraction of the linear threshold. A whisker reaching
# exactly 0 would otherwise terminate on the bottom spine, where the spine hides
# its end and the whisker reads as running off the panel. Kept below the
# threshold so the axis gains no tick below 0.
_SYMLOG_BOTTOM_MARGIN = 0.35


def _restyle_box(patch: PathPatch) -> Tuple[float, float, float]:
    """Give one box a full-opacity outline over a lightened fill of its colour.

    Args:
        patch: Box artist whose current face colour defines the hue.

    Returns:
        The box's ``(red, green, blue)`` hue, for reuse on its median line.
    """
    red, green, blue = patch.get_facecolor()[:3]
    patch.set_edgecolor((red, green, blue, 1.0))
    patch.set_facecolor((red, green, blue, _BOX_FILL_ALPHA))
    patch.set_linewidth(_BOX_LINE_WIDTH)
    return red, green, blue


def _recolor_box_outlines(ax: plt.Axes) -> None:
    """Give every box a hue-coloured outline and median over a lightened fill.

    Seaborn fills each box with its hue colour and draws outline and median in a
    single dark grey. This repaints, per box, the outline and the median line in
    that box's own fill colour and drops the fill to ``_BOX_FILL_ALPHA``, leaving
    the black whiskers, caps and fliers untouched.

    Medians are identified as the only non-black lines on the axes (whiskers,
    caps and fliers are set black before this runs) and matched to their box by
    horizontal centre, so a panel with any number of boxes and hue levels is
    handled without knowing seaborn's drawing order.

    The legend swatches seaborn adds for the hue levels are plain ``Rectangle``
    patches rather than boxes, and are restyled the same way so the legend shows
    the boxes as they are actually drawn rather than a solid block of hue.

    Args:
        ax: Axes holding a drawn seaborn boxplot.
    """
    black = to_rgba("black")
    color_by_center = {}
    for patch in ax.patches:
        red, green, blue = _restyle_box(patch)
        if not isinstance(patch, PathPatch):
            # A legend swatch: no box on the axes, so no median to match to it.
            continue
        vertices = patch.get_path().vertices
        center = (vertices[:, 0].min() + vertices[:, 0].max()) / 2.0
        color_by_center[round(center, 4)] = (red, green, blue, 1.0)

    for line in ax.lines:
        if to_rgba(line.get_color()) == black:
            continue
        x_data = np.asarray(line.get_xdata(), dtype=float)
        if len(x_data) == 0:
            continue
        center = round((x_data.min() + x_data.max()) / 2.0, 4)
        if center in color_by_center:
            line.set_color(color_by_center[center])


def plot_region_breakdown_boxes(
    data: pd.DataFrame,
    group: str,
    output_dir: Optional[Path] = None,
    fmt: str = "png",
    show_legend: bool = True,
    show_title: bool = True,
    palette: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
    show_gene_points: bool = False,
) -> plt.Axes:
    """Draw region counts for one gene group as per-condition boxes.

    Same data and grouping as :func:`plot_region_breakdown` (x-axis region in
    fixed order, hue condition), but each cell is shown as a box over the
    per-gene counts instead of a mean bar with a standard-deviation error bar. A
    bar with an error bar implies a symmetric spread around a mean, which these
    counts do not have (they scale with each gene's non-N sequence content per
    region); the box reports the median and quartiles instead.

    Whiskers, caps and fliers are black; the box outline and median carry the
    condition colour over a lightened fill of it (see
    :func:`_recolor_box_outlines`). Genes outside the whiskers show up as individual
    fliers; ``show_gene_points`` additionally overlays *every* gene as a small
    jittered dot, in which case the fliers are suppressed to avoid drawing those
    genes twice.

    When ``ax`` is None (standalone) a new figure is created, styled with the
    module theme, and — if ``output_dir`` is given — saved to
    ``region_breakdown_boxes_<group>.<fmt>`` and closed. When ``ax`` is provided
    the boxes are drawn onto it for panel composition: no theme is set, the
    figure is not laid out, saved, or closed, and nothing is written to disk.

    When ``ax`` is None (standalone) a new figure is created, styled with the
    module theme, and — if ``output_dir`` is given — saved to
    ``region_breakdown_boxes_<group>.<fmt>`` and closed. When ``ax`` is provided
    the boxes are drawn onto it for panel composition: no theme is set, the
    figure is not laid out, saved, or closed, and nothing is written to disk.

    Args:
        data: Tidy DataFrame from :func:`build_region_dataframe`.
        group: Gene group to plot (``"GOF"`` or ``"LOF"``).
        output_dir: Directory to save the figure into. Ignored when ``ax`` is
            given; when None in standalone mode the figure is not saved.
        fmt: File format (e.g. ``"png"``, ``"svg"``, ``"pdf"``).
        show_legend: Whether to draw the condition legend. Set False when the
            figure is a panel whose legend is provided elsewhere.
        show_title: Whether to draw the axis title. Set False for panel use.
        palette: Mapping of condition name to colour. When None, uses
            ``DEFAULT_CONDITION_PALETTE``.
        ax: Axes to draw onto. When None a standalone figure is created.
        show_gene_points: Whether to overlay the individual genes as jittered
            dots on top of the boxes.

    Returns:
        The Axes the boxes were drawn onto.
    """
    group_data = pd.DataFrame(data[data["group"] == group])
    if palette is None:
        palette = DEFAULT_CONDITION_PALETTE

    own_figure = ax is None
    if ax is None:
        sns.set_theme(style="whitegrid", font_scale=1.1)
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    shared_arguments = {
        "data": group_data,
        "x": "region",
        "y": "count",
        "hue": "condition",
        "order": REGION_ORDER,
        "hue_order": ["Constrained", "Unconstrained"],
        "palette": palette,
        "ax": ax,
    }
    # Whiskers, caps and fliers are black; the box keeps the condition colour.
    # ``saturation=1.0`` switches off seaborn's default 0.75 desaturation, so the
    # boxes carry the palette's exact colours and stay comparable to panels that
    # use the same palette elsewhere in a composed figure.
    sns.boxplot(
        showfliers=not show_gene_points,
        linewidth=_BOX_LINE_WIDTH,
        saturation=1.0,
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        flierprops={
            "markersize": _FLIER_MARKER_SIZE,
            "markeredgewidth": _BOX_LINE_WIDTH,
            "markeredgecolor": "black",
        },
        **shared_arguments,
    )
    _recolor_box_outlines(ax)
    if show_gene_points:
        # The dots are black rather than palette-coloured, so they stay
        # distinguishable from the box outline they sit inside. The hue is still
        # passed (as an all-black palette, which seaborn needs to keep the dots
        # dodged onto their box) but kept out of the legend.
        point_arguments = dict(shared_arguments)
        point_arguments["palette"] = {
            condition: "black" for condition in shared_arguments["hue_order"]
        }
        sns.stripplot(
            dodge=True,
            jitter=_GENE_POINT_JITTER,
            size=_GENE_POINT_SIZE,
            alpha=_GENE_POINT_ALPHA,
            linewidth=0.0,
            legend=False,
            **point_arguments,
        )
    # Symlog, not log: a few genes have no possible mutation at all in a region
    # (their VCF holds no record there), and log(0) is -inf, so a plain log axis
    # draws that whisker off the bottom of the panel. Below
    # ``_SYMLOG_LINEAR_THRESHOLD`` the axis is linear, which puts those genes at
    # exactly 0 while every count above the threshold keeps its log position --
    # unlike plotting count + 1, no value is shifted.
    ax.set_yscale(
        "symlog", linthresh=_SYMLOG_LINEAR_THRESHOLD, linscale=_SYMLOG_LINEAR_SCALE
    )
    ax.set_ylim(bottom=-_SYMLOG_BOTTOM_MARGIN * _SYMLOG_LINEAR_THRESHOLD)
    ax.set_xlabel("Genomic region")
    ax.set_ylabel("Possible mutations per gene (log scale, 0 included)")
    if show_title:
        ax.set_title(f"{group}: possible mutations by region")
    if show_legend:
        ax.legend(title="Condition", loc="upper right", fontsize=8, title_fontsize=8)
    elif ax.get_legend() is not None:
        ax.get_legend().remove()  # type: ignore[union-attr]

    if own_figure:
        fig.tight_layout()
        if output_dir is not None:
            output_path = output_dir / f"region_breakdown_boxes_{group}.{fmt}"
            fig.savefig(output_path, bbox_inches="tight", dpi=150)  # type: ignore[union-attr]
            plt.close(fig)
            print(f"Saved: {output_path}")
    return ax


def main() -> None:
    """Run the region breakdown for both conditions and gene sets."""
    gof_vcf = collect_vcf_region_counts(GOF_VCF_DIR)
    lof_vcf = collect_vcf_region_counts(LOF_VCF_DIR)
    gof_unconstrained = collect_unconstrained_region_counts(GOF_RUN_DIR)
    lof_unconstrained = collect_unconstrained_region_counts(LOF_RUN_DIR)

    data = build_region_dataframe(gof_vcf, lof_vcf, gof_unconstrained, lof_unconstrained)

    OUTPUT_DIR.mkdir(exist_ok=True)

    per_gene_path = OUTPUT_DIR / "region_breakdown_per_gene.csv"
    data.to_csv(per_gene_path, index=False)
    print(f"Saved: {per_gene_path}")

    summary = (
        data.groupby(["group", "condition", "region"])["count"]
        .agg(n="count", mean="mean", std="std")
        .reindex(REGION_ORDER, level="region")
        .round(1)
    )
    summary_path = OUTPUT_DIR / "region_breakdown_summary.csv"
    summary.to_csv(summary_path)
    print(f"Saved: {summary_path}")

    for group in ("GOF", "LOF"):
        plot_region_breakdown(data, group, OUTPUT_DIR, fmt=FMT)


if __name__ == "__main__":
    main()
