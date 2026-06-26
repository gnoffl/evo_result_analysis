"""Compare constrained vs unconstrained solution space sizes.

Constrained runs can only insert mutations present in natural-variation VCFs.
Unconstrained runs can insert any SNP at any non-N position of the reference.
Produces boxplots showing how severely the constraint limits the solution space.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyfaidx import Fasta
from scipy import stats

from workflows.evo_alg_pooled_plots.natural_unconstrained_comparison.vcf_mutation_stats import (
    collect_stats,
)

GOF_RUN_DIR = Path(
    "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset"
    "/GOF_LOF/GOF/GOF_single_natural_260311_122228_299093"
)
LOF_RUN_DIR = Path(
    "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset"
    "/GOF_LOF/LOF/LOF_single_natural_260311_122613_033117"
)
GOF_VCF_DIR = Path(
    "/home/gernot/Code/PhD_Code/Evolution/data/Arabidopsis_GOF_reextracted_vcfs"
)
LOF_VCF_DIR = Path(
    "/home/gernot/Code/PhD_Code/Evolution/data/Arabidopsis_LOF_reextracted_vcfs"
)
OUTPUT_DIR = Path(__file__).parent / "natural_unconstrained_mutation_vis"
FMT = "png"


def count_unconstrained_positions(fa_path: Path) -> int:
    """Count non-N positions in reference_sequence_full (mutable positions).

    Args:
        fa_path: Path to reference_sequence.fa.

    Returns:
        Number of non-N bases in the reference_sequence_full entry.
    """
    fasta = Fasta(str(fa_path))
    sequence = str(fasta["reference_sequence_full"])
    return sum(1 for base in sequence.upper() if base != "N")


def collect_unconstrained_stats(run_dir: Path) -> Tuple[List[int], List[int]]:
    """Collect unconstrained mutable positions and mutations per gene.

    Iterates over gene subdirectories (those starting with a digit) and reads
    their reference_sequence_full to compute per-gene solution space.

    Args:
        run_dir: Top-level run directory containing one subdir per gene.

    Returns:
        Tuple of (mutable_positions_per_gene, total_mutations_per_gene).
        total_mutations = mutable_positions * 3 (three possible SNPs per site).
    """
    mutable_positions: List[int] = []
    total_mutations: List[int] = []
    for gene_dir in sorted(run_dir.iterdir()):
        if not gene_dir.is_dir() or not gene_dir.name[0].isdigit():
            continue
        positions = count_unconstrained_positions(gene_dir / "reference_sequence.fa")
        mutable_positions.append(positions)
        total_mutations.append(positions * 3)
    return mutable_positions, total_mutations


def build_comparison_dataframe(
    gof_constrained_locs: List[int],
    gof_constrained_muts: List[int],
    lof_constrained_locs: List[int],
    lof_constrained_muts: List[int],
    gof_unconstrained_pos: List[int],
    gof_unconstrained_muts: List[int],
    lof_unconstrained_pos: List[int],
    lof_unconstrained_muts: List[int],
) -> pd.DataFrame:
    """Assemble a tidy DataFrame for plotting.

    Args:
        gof_constrained_locs: Per-gene unique VCF locations for GOF constrained.
        gof_constrained_muts: Per-gene total VCF mutations for GOF constrained.
        lof_constrained_locs: Per-gene unique VCF locations for LOF constrained.
        lof_constrained_muts: Per-gene total VCF mutations for LOF constrained.
        gof_unconstrained_pos: Per-gene mutable positions for GOF unconstrained.
        gof_unconstrained_muts: Per-gene total SNPs for GOF unconstrained.
        lof_unconstrained_pos: Per-gene mutable positions for LOF unconstrained.
        lof_unconstrained_muts: Per-gene total SNPs for LOF unconstrained.

    Returns:
        DataFrame with columns: group, condition, positions, mutations.
    """
    rows = []
    for positions, mutations in zip(gof_constrained_locs, gof_constrained_muts):
        rows.append({"group": "GOF", "condition": "Constrained", "positions": positions, "mutations": mutations})
    for positions, mutations in zip(lof_constrained_locs, lof_constrained_muts):
        rows.append({"group": "LOF", "condition": "Constrained", "positions": positions, "mutations": mutations})
    for positions, mutations in zip(gof_unconstrained_pos, gof_unconstrained_muts):
        rows.append({"group": "GOF", "condition": "Unconstrained", "positions": positions, "mutations": mutations})
    for positions, mutations in zip(lof_unconstrained_pos, lof_unconstrained_muts):
        rows.append({"group": "LOF", "condition": "Unconstrained", "positions": positions, "mutations": mutations})
    return pd.DataFrame(rows)


def _pvalue_label(p_value: float) -> str:
    """Return a significance label for a p-value.

    Args:
        p_value: The p-value to label.

    Returns:
        '***', '**', '*', or 'ns'.
    """
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _add_significance_brackets(
    ax: plt.Axes,
    data: pd.DataFrame,
    y_col: str,
    test_results: pd.DataFrame,
) -> None:
    """Add significance lines with stars between compared bar pairs.

    Draws a horizontal line connecting each pair of compared bars and places
    the significance label above the midpoint. Within-group (Wilcoxon) brackets
    appear at tier 1; between-group (Mann-Whitney) brackets at tier 2.

    Bar x-positions are read from ax.patches, assuming seaborn ordering:
    all bars of each hue in sequence across groups.

    Args:
        ax: Axes containing the grouped bar plot.
        data: Tidy DataFrame with 'group', 'condition', and y_col columns.
        y_col: Column being plotted (used to determine bracket heights).
        test_results: DataFrame from run_statistical_tests for this metric.
    """
    patches = [p for p in ax.patches if hasattr(p, "get_x")]
    # Seaborn fills patches per hue level across groups:
    # [GOF-Constrained, LOF-Constrained, GOF-Unconstrained, LOF-Unconstrained]
    x_gof_constrained = patches[0].get_x() + patches[0].get_width() / 2
    x_lof_constrained = patches[1].get_x() + patches[1].get_width() / 2
    x_gof_unconstrained = patches[2].get_x() + patches[2].get_width() / 2
    x_lof_unconstrained = patches[3].get_x() + patches[3].get_width() / 2

    # Base tiers on mean+std of the tallest bar to avoid over-inflating on a log scale
    grouped = data.groupby(["group", "condition"])[y_col]
    bar_top = (grouped.mean() + grouped.std()).max()
    tier1 = bar_top * 1.2
    tier2 = bar_top * 1.6

    def get_label(group_a: str, group_b: str) -> str:
        row = test_results[
            (test_results["group_a"] == group_a) & (test_results["group_b"] == group_b)
        ]
        return _pvalue_label(float(row["p_value"].iloc[0]))

    for x1, x2, y, label in [
        (x_gof_constrained, x_gof_unconstrained, tier1,
         get_label("GOF Constrained", "GOF Unconstrained")),
        (x_lof_constrained, x_lof_unconstrained, tier1,
         get_label("LOF Constrained", "LOF Unconstrained")),
        (x_gof_constrained, x_lof_constrained, tier2,
         get_label("GOF Constrained", "LOF Constrained")),
        (x_gof_unconstrained, x_lof_unconstrained, tier2 * 1.3,
         get_label("GOF Unconstrained", "LOF Unconstrained")),
    ]:
        ax.plot([x1, x2], [y, y], lw=0.8, color="black")
        ax.text((x1 + x2) / 2, y, label, ha="center", va="bottom", fontsize=8)

    ax.set_ylim(ax.get_ylim()[0], tier2 * 2)


def _barplot_figure(
    data: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    palette: dict,
    group_order: list,
    hue_order: list,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a single bar plot figure for one metric.

    Args:
        data: Tidy DataFrame with 'group', 'condition', and y_col columns.
        y_col: Column to plot on the y-axis.
        y_label: Y-axis label string.
        title: Figure title.
        palette: Colour mapping for condition hue.
        group_order: Order of x-axis groups.
        hue_order: Order of hue levels.

    Returns:
        Tuple of (Figure, Axes).
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(
        data=data,
        x="group",
        y=y_col,
        hue="condition",
        order=group_order,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
        errorbar="sd",
        width=0.9,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Gene group")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(title="Condition", loc="lower right", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    return fig, ax


def run_statistical_tests(
    gof_constrained: List[int],
    lof_constrained: List[int],
    gof_unconstrained: List[int],
    lof_unconstrained: List[int],
    metric_name: str,
) -> pd.DataFrame:
    """Run paired and unpaired statistical tests on solution-space metrics.

    Paired Wilcoxon signed-rank tests compare constrained vs unconstrained
    within each gene group (GOF and LOF). Mann-Whitney U tests compare GOF vs
    LOF within each condition (Constrained and Unconstrained).

    Args:
        gof_constrained: Per-gene metric values for GOF constrained runs.
        lof_constrained: Per-gene metric values for LOF constrained runs.
        gof_unconstrained: Per-gene metric values for GOF unconstrained runs.
        lof_unconstrained: Per-gene metric values for LOF unconstrained runs.
        metric_name: Label for the metric (e.g. 'positions' or 'mutations').

    Returns:
        DataFrame with columns: test, group_a, group_b, statistic, p_value, metric.
    """
    wilcoxon_gof = stats.wilcoxon(gof_constrained, gof_unconstrained)
    wilcoxon_lof = stats.wilcoxon(lof_constrained, lof_unconstrained)
    mwu_constrained = stats.mannwhitneyu(
        gof_constrained, lof_constrained, alternative="two-sided"
    )
    mwu_unconstrained = stats.mannwhitneyu(
        gof_unconstrained, lof_unconstrained, alternative="two-sided"
    )

    rows = [
        {
            "test": "Wilcoxon signed-rank",
            "group_a": "GOF Constrained",
            "group_b": "GOF Unconstrained",
            "statistic": wilcoxon_gof.statistic,
            "p_value": wilcoxon_gof.pvalue,
            "metric": metric_name,
        },
        {
            "test": "Wilcoxon signed-rank",
            "group_a": "LOF Constrained",
            "group_b": "LOF Unconstrained",
            "statistic": wilcoxon_lof.statistic,
            "p_value": wilcoxon_lof.pvalue,
            "metric": metric_name,
        },
        {
            "test": "Mann-Whitney U",
            "group_a": "GOF Constrained",
            "group_b": "LOF Constrained",
            "statistic": mwu_constrained.statistic,
            "p_value": mwu_constrained.pvalue,
            "metric": metric_name,
        },
        {
            "test": "Mann-Whitney U",
            "group_a": "GOF Unconstrained",
            "group_b": "LOF Unconstrained",
            "statistic": mwu_unconstrained.statistic,
            "p_value": mwu_unconstrained.pvalue,
            "metric": metric_name,
        },
    ]
    return pd.DataFrame(rows)


def plot_solution_space_comparison(
    data: pd.DataFrame,
    output_dir: str,
    fmt: str = "png",
    test_results: Optional[pd.DataFrame] = None,
) -> None:
    """Save two bar plot figures comparing constrained vs unconstrained solution space.

    Figures show mean ± std per group and condition on a log-scaled y-axis.
    If test_results is provided, significance lines and stars are added.

    Args:
        data: Tidy DataFrame from build_comparison_dataframe.
        output_dir: Directory to save the figures.
        fmt: File format (e.g. 'png', 'svg', 'pdf').
        test_results: Optional DataFrame from run_statistical_tests (both metrics).
    """
    palette = {"Constrained": "#4C72B0", "Unconstrained": "#DD8452"}
    group_order = ["GOF", "LOF"]
    hue_order = ["Constrained", "Unconstrained"]

    sns.set_theme(style="whitegrid", font_scale=1.1)

    fig_positions, ax_positions = _barplot_figure(
        data, "positions", "Mutable positions per gene (log scale)",
        "Unique mutable positions", palette, group_order, hue_order,
    )
    if test_results is not None:
        _add_significance_brackets(
            ax_positions, data, "positions",
            pd.DataFrame(test_results[test_results["metric"] == "positions"]),
        )
    path_positions = Path(output_dir) / f"solution_space_positions.{fmt}"
    fig_positions.savefig(path_positions, bbox_inches="tight", dpi=150)
    plt.close(fig_positions)
    print(f"Saved: {path_positions}")

    fig_mutations, ax_mutations = _barplot_figure(
        data, "mutations", "Possible mutations per gene (log scale)",
        "Total possible mutations", palette, group_order, hue_order,
    )
    if test_results is not None:
        _add_significance_brackets(
            ax_mutations, data, "mutations",
            pd.DataFrame(test_results[test_results["metric"] == "mutations"]),
        )
    path_mutations = Path(output_dir) / f"solution_space_mutations.{fmt}"
    fig_mutations.savefig(path_mutations, bbox_inches="tight", dpi=150)
    plt.close(fig_mutations)
    print(f"Saved: {path_mutations}")


if __name__ == "__main__":
    gof_constrained_locs, gof_constrained_muts = collect_stats(GOF_VCF_DIR)
    lof_constrained_locs, lof_constrained_muts = collect_stats(LOF_VCF_DIR)

    gof_unconstrained_pos, gof_unconstrained_muts = collect_unconstrained_stats(GOF_RUN_DIR)
    lof_unconstrained_pos, lof_unconstrained_muts = collect_unconstrained_stats(LOF_RUN_DIR)

    data = build_comparison_dataframe(
        gof_constrained_locs,
        gof_constrained_muts,
        lof_constrained_locs,
        lof_constrained_muts,
        gof_unconstrained_pos,
        gof_unconstrained_muts,
        lof_unconstrained_pos,
        lof_unconstrained_muts,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    stats_positions = run_statistical_tests(
        gof_constrained_locs,
        lof_constrained_locs,
        gof_unconstrained_pos,
        lof_unconstrained_pos,
        "positions",
    )
    stats_mutations = run_statistical_tests(
        gof_constrained_muts,
        lof_constrained_muts,
        gof_unconstrained_muts,
        lof_unconstrained_muts,
        "mutations",
    )
    statistical_tests = pd.concat([stats_positions, stats_mutations], ignore_index=True)
    print(statistical_tests.to_string(index=False))
    stats_path = OUTPUT_DIR / "statistical_tests.csv"
    statistical_tests.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")

    plot_solution_space_comparison(data, str(OUTPUT_DIR), fmt=FMT, test_results=statistical_tests)

    summary_path = OUTPUT_DIR / "solution_space_summary.csv"
    data.groupby(["group", "condition"]).agg(
        n=("positions", "count"),
        positions_mean=("positions", "mean"),
        positions_std=("positions", "std"),
        mutations_mean=("mutations", "mean"),
        mutations_std=("mutations", "std"),
    ).round(1).to_csv(summary_path)
    print(f"Saved: {summary_path}")
