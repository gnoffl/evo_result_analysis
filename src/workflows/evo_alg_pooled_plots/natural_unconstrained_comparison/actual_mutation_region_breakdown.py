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

Run sources: these are the UNCONSTRAINED runs (``*_single_mutation_*``, whose
``parameters.json`` has ``allowed_mutations: null`` and no ``vcf_path``), NOT
the ``*_single_natural_*`` runs used elsewhere in this package. The natural
runs are constrained to VCF positions by construction, so reading their
introduced mutations would trivially yield 100% allowed. Only the VCF
directories are shared with the other scripts here.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Set

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

    plot_actual_mutation_allowance(data, OUTPUT_DIR, fmt=FMT)


if __name__ == "__main__":
    main()
