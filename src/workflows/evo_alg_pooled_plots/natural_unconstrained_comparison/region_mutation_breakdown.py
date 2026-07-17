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
import pandas as pd
import seaborn as sns
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
