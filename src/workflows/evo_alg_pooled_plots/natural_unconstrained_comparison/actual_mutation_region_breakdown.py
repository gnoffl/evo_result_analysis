"""Break down actual mutations introduced by unconstrained runs by region.

For each gene, the most-mutated individual from the final pareto front of its
unconstrained optimization run (``front[0]``, the optimized endpoint; see the
pareto-front ordering convention in ``natural_unconstrained_comparison.py``) is
diffed against the reference to get the mutations actually introduced. Each
mutation's 1-based position is checked against that gene's natural-variation
VCF: a position present in the VCF means a constrained (natural-only) run
would have been allowed to introduce a mutation there too ("allowed"),
regardless of which base it mutated to.

Mutations are binned by genomic region (promoter, UTRs, terminator; see
``region_mutation_breakdown.py``) and the percentage that would have been
allowed is reported per region, for the GOF and LOF gene sets separately.

Two summaries are produced. The per-gene percentage (boxplot) shows the spread
across genes but is strongly right-skewed, since many genes carry only a handful
of mutations per region. The **pooled** percentage (bar plot;
``plot_pooled_allowance_bars``) sums allowed and total mutations across genes and
brackets the ratio with a **gene-level cluster bootstrap** interval — mutations
are clustered within genes, so a binomial interval on the pooled mutation count
would be too narrow, and resampling whole genes keeps every endpoint inside
[0, 100]. The pooled version is what figure 4 panel J shows; see
``paper_plots/DESIGN.md`` §3c for the full rationale.

Run sources: these are the UNCONSTRAINED runs (``*_single_mutation_*``, whose
``parameters.json`` has ``allowed_mutations: null`` and no ``vcf_path``), NOT
the ``*_single_natural_*`` runs used elsewhere in this package. The natural
runs are constrained to VCF positions by construction, so reading their
introduced mutations would trivially yield 100% allowed. Only the VCF
directories are shared with the other scripts here.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyfaidx import Fasta

from analysis.mutations.summarize_mutations import MutatedSequence
from workflows.evo_alg_pooled_plots.natural_unconstrained_comparison.natural_unconstrained_comparison import (
    core_gene_id,
)
from workflows.evo_alg_pooled_plots.natural_unconstrained_comparison.natural_unconstrained_mutation_vis import (
    GOF_VCF_DIR,
    LOF_VCF_DIR,
)
from workflows.evo_alg_pooled_plots.natural_unconstrained_comparison.region_mutation_breakdown import (
    REGION_ORDER,
    assign_region,
)

# Unconstrained optimization runs (allowed_mutations = null). Distinct from the
# constrained *_single_natural_* runs referenced by the other scripts.
GOF_UNCONSTRAINED_RUN_DIR = Path(
    "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset"
    "/GOF_LOF/GOF/GOF_single_mutation_251009_121226_109368"
)
LOF_UNCONSTRAINED_RUN_DIR = Path(
    "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset"
    "/GOF_LOF/LOF/LOF_single_mutation_251020_180028_564570"
)

OUTPUT_DIR = Path(__file__).parent / "actual_mutation_region_breakdown"
FMT = "png"


def load_vcf_positions_by_gene(vcf_dir: Path) -> Dict[str, Set[int]]:
    """Load the set of natural-variation VCF positions for each gene.

    Args:
        vcf_dir: Directory containing one ``*.vcf`` file per gene.

    Returns:
        Mapping from core gene id (see ``core_gene_id``) to the set of
        1-based VCF positions recorded for that gene.
    """
    positions_by_gene = {}
    for vcf_file in sorted(vcf_dir.glob("*.vcf")):
        positions = set()
        for line in vcf_file.read_text().splitlines():
            if line.startswith("#"):
                continue
            positions.add(int(line.split("\t")[1]))
        positions_by_gene[core_gene_id(vcf_file.stem)] = positions
    return positions_by_gene


def load_most_mutated_individual(gene_dir: Path) -> MutatedSequence:
    """Load the most-mutated individual from a gene's final pareto front.

    Follows the established ``pareto_front.json`` ordering convention:
    ``front[0]`` is the optimized endpoint with the most mutations,
    ``front[-1]`` the 0-mutation reference. The first entry is read directly
    rather than searched for, since this ordering holds for both maximization
    and minimization runs.

    Args:
        gene_dir: Gene run directory containing ``reference_sequence.fa`` and
            ``saved_populations/pareto_front.json``.

    Returns:
        A ``MutatedSequence`` with mutations diffed against the reference.
    """
    fasta = Fasta(str(gene_dir / "reference_sequence.fa"))
    reference_sequence = str(fasta[1][:].seq)
    with open(gene_dir / "saved_populations" / "pareto_front.json") as handle:
        pareto_front = json.load(handle)
    sequence, fitness, _mutation_count = pareto_front[0]
    return MutatedSequence(
        reference_sequence=reference_sequence,
        mutated_sequence=sequence,
        fitness=fitness,
    )


def collect_actual_mutation_region_counts(
    run_dir: Path, vcf_positions_by_gene: Dict[str, Set[int]]
) -> pd.DataFrame:
    """Count actual mutations and how many were VCF-allowed, per gene/region.

    Genes with no matching VCF file are skipped and reported (not silently
    dropped).

    Args:
        run_dir: Top-level run directory containing one subdir per gene.
        vcf_positions_by_gene: Mapping from core gene id to VCF positions, as
            returned by ``load_vcf_positions_by_gene``.

    Returns:
        Tidy DataFrame with columns ``gene``, ``region``, ``total_mutations``,
        ``allowed_mutations``. Every matched gene contributes one row per
        region (zero counts if it had no mutations there).
    """
    rows = []
    skipped_genes = []
    for gene_dir in sorted(run_dir.iterdir()):
        if not gene_dir.is_dir() or not gene_dir.name[0].isdigit():
            continue
        gene_id = core_gene_id(gene_dir.name)
        if gene_id not in vcf_positions_by_gene:
            skipped_genes.append(gene_dir.name)
            continue
        vcf_positions = vcf_positions_by_gene[gene_id]
        individual = load_most_mutated_individual(gene_dir)

        totals = {name: 0 for name in REGION_ORDER}
        allowed = {name: 0 for name in REGION_ORDER}
        for position, _ref_base, _mut_base in individual.mutations:
            one_based_position = position + 1
            region = assign_region(one_based_position)
            if region is None:
                continue
            totals[region] += 1
            if one_based_position in vcf_positions:
                allowed[region] += 1

        for region in REGION_ORDER:
            rows.append(
                {
                    "gene": gene_dir.name,
                    "region": region,
                    "total_mutations": totals[region],
                    "allowed_mutations": allowed[region],
                }
            )

    if skipped_genes:
        print(
            f"Skipped {len(skipped_genes)} gene(s) with no matching VCF file: "
            f"{skipped_genes}"
        )
    return pd.DataFrame(rows)


def build_group_dataframe(gof_data: pd.DataFrame, lof_data: pd.DataFrame) -> pd.DataFrame:
    """Combine GOF and LOF per-gene region counts and add a percent-allowed column.

    Args:
        gof_data: Per-gene region counts for GOF, from
            ``collect_actual_mutation_region_counts``.
        lof_data: Per-gene region counts for LOF, from
            ``collect_actual_mutation_region_counts``.

    Returns:
        Combined DataFrame with an added ``group`` column and a
        ``percent_allowed`` column (``allowed_mutations / total_mutations *
        100``, ``NaN`` where a gene had zero mutations in that region).
    """
    labeled = []
    for frame, group in ((gof_data, "GOF"), (lof_data, "LOF")):
        annotated = frame.copy()
        annotated["group"] = group
        labeled.append(annotated)
    combined = pd.concat(labeled, ignore_index=True)
    combined["percent_allowed"] = np.where(
        combined["total_mutations"] > 0,
        combined["allowed_mutations"] / combined["total_mutations"] * 100,
        np.nan,
    )
    return pd.DataFrame(
        combined[
            ["gene", "group", "region", "total_mutations", "allowed_mutations", "percent_allowed"]
        ]
    )


def build_summary_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize the pooled percent-allowed per group and region.

    Reports the pooled percentage (total allowed / total mutations summed
    across all genes), which is more robust than averaging per-gene
    percentages for regions where individual genes have few mutations.

    Args:
        data: Tidy DataFrame from ``build_group_dataframe``.

    Returns:
        DataFrame indexed by ``group``, ``region`` with columns ``n_genes``
        (genes with at least one mutation in the region) and
        ``pooled_percent`` (``NaN`` where no gene had any mutation there).
    """
    gene_counts = (
        data.groupby(["group", "region"])["percent_allowed"]
        .agg(n_genes="count")
        .reindex(REGION_ORDER, level="region")
    )
    pooled_totals = data.groupby(["group", "region"])[
        ["total_mutations", "allowed_mutations"]
    ].sum()
    pooled_totals["pooled_percent"] = (
        pooled_totals["allowed_mutations"] / pooled_totals["total_mutations"] * 100
    )
    summary = gene_counts.join(pooled_totals[["pooled_percent"]])
    return pd.DataFrame(summary)


DEFAULT_GROUP_PALETTE = {"GOF": "#4C72B0", "LOF": "#DD8452"}


def plot_actual_mutation_allowance(
    data: pd.DataFrame,
    output_dir: Optional[Path] = None,
    fmt: str = "png",
    show_legend: bool = True,
    show_title: bool = True,
    palette: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Draw a boxplot of per-gene percent-allowed mutations by region.

    x-axis is region (fixed order), hue is gene group (GOF vs LOF). Each box
    summarizes the per-gene ``percent_allowed`` distribution for genes that
    had at least one mutation in that region (genes with zero mutations there
    are dropped, since their percentage is undefined).

    When ``ax`` is None (standalone) a new figure is created, styled with the
    module theme, and — if ``output_dir`` is given — saved to
    ``actual_mutation_allowance.<fmt>`` and closed. When ``ax`` is provided
    the boxes are drawn onto it for panel composition: no theme is set, the
    figure is not laid out, saved, or closed, and nothing is written to disk.

    Args:
        data: Tidy DataFrame from ``build_group_dataframe``.
        output_dir: Directory to save the figure into. Ignored when ``ax`` is
            given; when None in standalone mode the figure is not saved.
        fmt: File format (e.g. ``"png"``, ``"svg"``, ``"pdf"``).
        show_legend: Whether to draw the group legend. Set False when the
            figure is a panel whose legend is provided elsewhere.
        show_title: Whether to draw the axis title. Set False for panel use.
        palette: Mapping of group name to colour. When None, uses
            ``DEFAULT_GROUP_PALETTE``.
        ax: Axes to draw onto. When None a standalone figure is created.

    Returns:
        The Axes the boxes were drawn onto.
    """
    plottable = pd.DataFrame(data.dropna(subset=["percent_allowed"]))
    if palette is None:
        palette = DEFAULT_GROUP_PALETTE

    own_figure = ax is None
    if own_figure:
        sns.set_theme(style="whitegrid", font_scale=1.1)
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    sns.boxplot(
        data=plottable,
        x="region",
        y="percent_allowed",
        hue="group",
        order=REGION_ORDER,
        hue_order=["GOF", "LOF"],
        palette=palette,
        ax=ax,
    )
    ax.set_xlabel("Genomic region")
    ax.set_ylabel("Actual mutations at natural-VCF-allowed positions (%)")
    ax.set_ylim(-5, 105)
    if show_title:
        ax.set_title("Actual mutations that would have been allowed naturally")
    if show_legend:
        ax.legend(title="Gene set", loc="lower right", fontsize=8, title_fontsize=8)
    elif ax.get_legend() is not None:
        ax.get_legend().remove()  # type: ignore[union-attr]

    if own_figure:
        fig.tight_layout()
        if output_dir is not None:
            output_path = output_dir / f"actual_mutation_allowance.{fmt}"
            fig.savefig(output_path, bbox_inches="tight", dpi=150)  # type: ignore[union-attr]
            plt.close(fig)
            print(f"Saved: {output_path}")
    return ax


# Bootstrap settings for the pooled-percentage confidence intervals. 10000
# resamples keep the Monte-Carlo noise on a percentile endpoint well below 0.1
# percentage points at this sample size; the fixed seed makes the published
# figure reproducible.
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20240101
BOOTSTRAP_CONFIDENCE = 95.0

# Bar geometry for the pooled panel: two dodged bars (GOF, LOF) per region.
_GROUP_ORDER = ["GOF", "LOF"]
_BAR_WIDTH = 0.38


def pooled_percent_with_bootstrap_ci(
    allowed_per_gene: np.ndarray,
    total_per_gene: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> Tuple[float, float, float]:
    """Pooled percent-allowed with a gene-level cluster bootstrap interval.

    The point estimate is the pooled percentage ``sum(allowed) / sum(total) *
    100`` — a ratio of sums, weighting each gene by how many mutations it
    received. The interval is a percentile bootstrap over *genes*: whole
    ``(allowed, total)`` pairs are resampled with replacement, so mutations
    clustered inside one gene are never treated as independent observations (a
    plain binomial interval on the pooled mutation count would be too narrow).
    Because every resample is itself a valid percentage, the interval can never
    leave [0, 100].

    Genes with zero mutations are kept: under the pooled estimator they
    contribute nothing to either sum, but they are part of the gene sample being
    resampled. Resamples that draw only such genes have a zero denominator and
    are discarded before the percentiles are taken.

    Args:
        allowed_per_gene: Per-gene count of mutations at VCF-allowed positions.
        total_per_gene: Per-gene count of mutations, same length and order.
        n_resamples: Number of bootstrap resamples.
        seed: Seed for the random generator, for reproducible intervals.
        confidence: Two-sided confidence level in percent (e.g. 95.0).

    Returns:
        Tuple of ``(pooled_percent, ci_low, ci_high)``. All three are ``NaN``
        when no gene had any mutation; the interval endpoints are ``NaN`` when
        every resample degenerated to a zero denominator.

    Raises:
        ValueError: If the two arrays differ in length, or ``confidence`` is not
            strictly between 0 and 100.
    """
    if len(allowed_per_gene) != len(total_per_gene):
        raise ValueError(
            "allowed_per_gene and total_per_gene must have the same length, got "
            f"{len(allowed_per_gene)} and {len(total_per_gene)}"
        )
    if not 0.0 < confidence < 100.0:
        raise ValueError(f"confidence must be in (0, 100), got {confidence}")

    allowed = np.asarray(allowed_per_gene, dtype=float)
    total = np.asarray(total_per_gene, dtype=float)
    total_mutations = total.sum()
    if len(total) == 0 or total_mutations == 0:
        return float("nan"), float("nan"), float("nan")

    pooled_percent = allowed.sum() / total_mutations * 100.0

    generator = np.random.default_rng(seed)
    gene_indices = generator.integers(0, len(total), size=(n_resamples, len(total)))
    resampled_totals = total[gene_indices].sum(axis=1)
    usable = resampled_totals > 0
    if not usable.any():
        return pooled_percent, float("nan"), float("nan")
    resampled_percent = (
        allowed[gene_indices].sum(axis=1)[usable] / resampled_totals[usable] * 100.0
    )

    tail = (100.0 - confidence) / 2.0
    ci_low, ci_high = np.percentile(resampled_percent, [tail, 100.0 - tail])
    return pooled_percent, float(ci_low), float(ci_high)


def build_pooled_ci_dataframe(
    data: pd.DataFrame,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> pd.DataFrame:
    """Pool each group/region cell and bootstrap its confidence interval.

    Each cell is summarized by :func:`pooled_percent_with_bootstrap_ci` over its
    genes. Every cell gets its own seed offset so the resamples are not
    identical across cells while the whole table stays reproducible.

    Args:
        data: Tidy DataFrame from ``build_group_dataframe``.
        n_resamples: Number of bootstrap resamples per cell.
        seed: Base seed; cell ``i`` uses ``seed + i``.
        confidence: Two-sided confidence level in percent.

    Returns:
        One row per group/region (groups then ``REGION_ORDER``) with columns
        ``group``, ``region``, ``n_genes`` (genes with at least one mutation
        there), ``n_genes_total`` (genes resampled, including zero-mutation
        ones), ``total_mutations``, ``allowed_mutations``, ``pooled_percent``,
        ``ci_low`` and ``ci_high``.
    """
    rows = []
    for cell_index, (group, region) in enumerate(
        (group, region) for group in _GROUP_ORDER for region in REGION_ORDER
    ):
        cell = data[(data["group"] == group) & (data["region"] == region)]
        pooled_percent, ci_low, ci_high = pooled_percent_with_bootstrap_ci(
            cell["allowed_mutations"].to_numpy(),
            cell["total_mutations"].to_numpy(),
            n_resamples=n_resamples,
            seed=seed + cell_index,
            confidence=confidence,
        )
        rows.append(
            {
                "group": group,
                "region": region,
                "n_genes": int((cell["total_mutations"] > 0).sum()),
                "n_genes_total": len(cell),
                "total_mutations": int(cell["total_mutations"].sum()),
                "allowed_mutations": int(cell["allowed_mutations"].sum()),
                "pooled_percent": pooled_percent,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def plot_pooled_allowance_bars(
    data: pd.DataFrame,
    output_dir: Optional[Path] = None,
    fmt: str = "png",
    show_legend: bool = True,
    show_title: bool = True,
    palette: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> plt.Axes:
    """Draw pooled percent-allowed bars with bootstrap CIs, by region and group.

    Each bar is the pooled percentage of that group/region cell (total allowed
    mutations over total mutations, summed across genes) and each error bar the
    gene-level cluster bootstrap interval from
    :func:`pooled_percent_with_bootstrap_ci`. Unlike the per-gene mean, the
    pooled estimate is not inflated by genes with one or two mutations, and the
    interval cannot reach below zero, so the y-axis stays tight and every drawn
    value is a possible percentage.

    Note that pooling weights genes by their mutation count, so heavily mutated
    genes dominate a cell and the per-gene spread is not visible; the returned
    axes' data (see :func:`build_pooled_ci_dataframe`) carries the per-cell gene
    counts for the caption.

    When ``ax`` is None (standalone) a new figure is created, styled with the
    module theme, and — if ``output_dir`` is given — saved to
    ``pooled_allowance_bars.<fmt>`` and closed. When ``ax`` is provided the bars
    are drawn onto it for panel composition: no theme is set, the figure is not
    laid out, saved, or closed, and nothing is written to disk.

    Args:
        data: Tidy DataFrame from ``build_group_dataframe``.
        output_dir: Directory to save the figure into. Ignored when ``ax`` is
            given; when None in standalone mode the figure is not saved.
        fmt: File format (e.g. ``"png"``, ``"svg"``, ``"pdf"``).
        show_legend: Whether to draw the group legend. Set False when the
            figure is a panel whose legend is provided elsewhere.
        show_title: Whether to draw the axis title. Set False for panel use.
        palette: Mapping of group name to colour. When None, uses
            ``DEFAULT_GROUP_PALETTE``.
        ax: Axes to draw onto. When None a standalone figure is created.
        n_resamples: Number of bootstrap resamples per cell.
        seed: Base seed for the bootstrap, for reproducible intervals.

    Returns:
        The Axes the bars were drawn onto.
    """
    pooled = build_pooled_ci_dataframe(data, n_resamples=n_resamples, seed=seed)
    if palette is None:
        palette = DEFAULT_GROUP_PALETTE

    own_figure = ax is None
    if own_figure:
        sns.set_theme(style="whitegrid", font_scale=1.1)
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    region_positions = np.arange(len(REGION_ORDER), dtype=float)
    for group_index, group in enumerate(_GROUP_ORDER):
        group_rows = pooled[pooled["group"] == group].set_index("region")
        heights = [
            group_rows.loc[region, "pooled_percent"] if region in group_rows.index
            else float("nan")
            for region in REGION_ORDER
        ]
        # Asymmetric error bars: the bootstrap interval is not centred on the bar.
        lower_errors = [
            height - group_rows.loc[region, "ci_low"]
            for region, height in zip(REGION_ORDER, heights)
        ]
        upper_errors = [
            group_rows.loc[region, "ci_high"] - height
            for region, height in zip(REGION_ORDER, heights)
        ]
        offset = (group_index - (len(_GROUP_ORDER) - 1) / 2) * _BAR_WIDTH
        ax.bar(
            region_positions + offset,
            heights,
            width=_BAR_WIDTH,
            color=palette[group],
            label=group,
            yerr=np.abs(np.vstack([lower_errors, upper_errors])),
            capsize=2.5,
            error_kw={"linewidth": 1.0, "ecolor": "0.25"},
        )

    ax.set_xticks(region_positions)
    ax.set_xticklabels(REGION_ORDER)
    ax.set_xlabel("Genomic region")
    ax.set_ylabel("Actual mutations at natural-VCF-allowed positions (%)")
    ax.set_ylim(bottom=0.0)
    if show_title:
        ax.set_title("Actual mutations that would have been allowed naturally")
    if show_legend:
        ax.legend(title="Gene set", loc="upper right", fontsize=8, title_fontsize=8)
    elif ax.get_legend() is not None:
        ax.get_legend().remove()  # type: ignore[union-attr]

    if own_figure:
        fig.tight_layout()
        if output_dir is not None:
            output_path = output_dir / f"pooled_allowance_bars.{fmt}"
            fig.savefig(output_path, bbox_inches="tight", dpi=150)  # type: ignore[union-attr]
            plt.close(fig)
            print(f"Saved: {output_path}")
    return ax


def main() -> None:
    """Run the actual-mutation allowance breakdown for GOF and LOF."""
    gof_vcf_positions = load_vcf_positions_by_gene(GOF_VCF_DIR)
    lof_vcf_positions = load_vcf_positions_by_gene(LOF_VCF_DIR)

    gof_data = collect_actual_mutation_region_counts(
        GOF_UNCONSTRAINED_RUN_DIR, gof_vcf_positions
    )
    lof_data = collect_actual_mutation_region_counts(
        LOF_UNCONSTRAINED_RUN_DIR, lof_vcf_positions
    )

    data = build_group_dataframe(gof_data, lof_data)

    OUTPUT_DIR.mkdir(exist_ok=True)

    per_gene_path = OUTPUT_DIR / "actual_mutation_allowance_per_gene.csv"
    data.to_csv(per_gene_path, index=False)
    print(f"Saved: {per_gene_path}")

    summary = build_summary_dataframe(data)
    summary_path = OUTPUT_DIR / "actual_mutation_allowance_summary.csv"
    summary.to_csv(summary_path)
    print(f"Saved: {summary_path}")
    print(summary.round(1).to_string())

    pooled_ci = build_pooled_ci_dataframe(data)
    pooled_ci_path = OUTPUT_DIR / "actual_mutation_allowance_pooled_ci.csv"
    pooled_ci.to_csv(pooled_ci_path, index=False)
    print(f"Saved: {pooled_ci_path}")
    print(pooled_ci.round(2).to_string(index=False))

    plot_actual_mutation_allowance(data, OUTPUT_DIR, fmt=FMT)
    plot_pooled_allowance_bars(data, OUTPUT_DIR, fmt=FMT)


if __name__ == "__main__":
    main()
