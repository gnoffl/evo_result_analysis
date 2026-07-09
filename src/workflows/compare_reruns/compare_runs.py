"""Compare optimization results between two evolutionary-algorithm run folders.

The analysis answers: how consistent are optimization results between two runs of
the (possibly slightly modified) evolutionary algorithm on the same genes?

For every gene present in *both* run folders it compares:

- the deepCRE prediction before optimization (the 0-mutation reference member) and
  after optimization (the maximally mutated endpoint of the Pareto front),
- the maximum mutation count reached,
- which individual mutations (position + introduced base) the two runs share and
  which differ.

Data conventions (see project memory ``pareto-front-ordering``):

- ``saved_populations/pareto_front.json`` is a list of
  ``[sequence_str, fitness_float, mutation_count_float]``.
- The list is ordered: ``front[0]`` is the optimized endpoint ("after"),
  ``front[-1]`` is the 0-mutation reference ("before"). Indexing works for both
  maximization and minimization runs.
- All Pareto sequences share the reference length; mutations are substitutions, so
  positional comparison is valid.

The module keeps calculation functions (pure, unit-tested) strictly separate from
plotting functions (rendering only, not unit-tested).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy.stats import wilcoxon

# A single Pareto front member: [sequence, fitness, mutation_count].
ParetoMember = list
# A full Pareto front: ordered list of members.
ParetoFront = list

PER_GENE_COLUMNS = [
    "gene",
    "before",
    "after_A",
    "after_B",
    "n_mutations_A",
    "n_mutations_B",
    "delta_after",
    "delta_n_mutations",
    "shared_mutations",
    "a_only_mutations",
    "b_only_mutations",
]


# --------------------------------------------------------------------------- #
# Calculation functions (pure, unit-tested)
# --------------------------------------------------------------------------- #
def extract_gene_key(folder_name: str) -> str:
    """Extract the gene identity key from a run's gene-folder name.

    Gene folders are named ``{chr}_{AGI}_gene:{coords}_{timestamp}`` (e.g.
    ``1_AT1G75760_gene:28448735-28446631_260224_163252_182620``). The identity key
    is the chromosome prefix plus the AGI code, matching the convention used
    elsewhere in the codebase.

    Args:
        folder_name: Base name of a gene folder.

    Returns:
        The gene key, e.g. ``"1_AT1G75760"``.
    """
    return "_".join(folder_name.split("_")[:2])


def discover_gene_fronts(run_dir: str) -> dict[str, str]:
    """Map each gene key in a run folder to its Pareto front file path.

    A gene folder qualifies only if it contains
    ``saved_populations/pareto_front.json``.

    Args:
        run_dir: Path to a run folder directly containing gene subfolders.

    Returns:
        Mapping of gene key to the absolute Pareto front file path.

    Raises:
        FileNotFoundError: If ``run_dir`` does not exist.
        ValueError: If two gene folders map to the same gene key.
    """
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run folder does not exist: {run_dir}")

    gene_fronts: dict[str, str] = {}
    for entry in sorted(os.listdir(run_dir)):
        folder_path = os.path.join(run_dir, entry)
        pareto_path = os.path.join(folder_path, "saved_populations", "pareto_front.json")
        if not os.path.isfile(pareto_path):
            continue
        gene_key = extract_gene_key(entry)
        if gene_key in gene_fronts:
            raise ValueError(
                f"Duplicate gene key '{gene_key}' in run folder {run_dir}: "
                f"'{os.path.basename(gene_fronts[gene_key])}' and '{entry}'"
            )
        gene_fronts[gene_key] = pareto_path
    return gene_fronts


def load_pareto_front(pareto_path: str) -> ParetoFront:
    """Load a Pareto front JSON file.

    Args:
        pareto_path: Path to a ``pareto_front.json`` file.

    Returns:
        The Pareto front as a list of ``[sequence, fitness, mutation_count]``.
    """
    with open(pareto_path) as pareto_file:
        return json.load(pareto_file)


def mutation_set(reference_sequence: str, optimized_sequence: str) -> set[tuple[int, str]]:
    """Return the set of substitutions in an optimized sequence relative to reference.

    Each substitution is identified by its 0-based position and the introduced base
    (position + base), so two runs "share" a mutation only when both the site and the
    resulting base agree.

    Args:
        reference_sequence: The 0-mutation reference sequence.
        optimized_sequence: The optimized sequence (same length as reference).

    Returns:
        Set of ``(position, introduced_base)`` tuples.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(reference_sequence) != len(optimized_sequence):
        raise ValueError(
            "Reference and optimized sequences differ in length: "
            f"{len(reference_sequence)} vs {len(optimized_sequence)}"
        )
    return {
        (position, optimized_base)
        for position, (reference_base, optimized_base) in enumerate(
            zip(reference_sequence, optimized_sequence)
        )
        if reference_base != optimized_base
    }


def compare_single_gene(
    gene_key: str,
    front_a: ParetoFront,
    front_b: ParetoFront,
    fitness_tolerance: float = 1e-6,
) -> dict:
    """Compute the comparison record for a single gene present in both runs.

    The before-optimization baseline must be identical across runs (same gene, same
    reference). Both the reference sequence and its stored fitness are validated; a
    mismatch aborts the analysis because it signals the runs are not comparable.

    Args:
        gene_key: Gene identity key.
        front_a: Pareto front of run A (baseline).
        front_b: Pareto front of run B (comparison).
        fitness_tolerance: Absolute tolerance for the before-fitness equality check.

    Returns:
        A record dict with the keys listed in :data:`PER_GENE_COLUMNS`.

    Raises:
        ValueError: If reference sequences differ, or before-fitness values differ
            by more than ``fitness_tolerance``.
    """
    before_a_sequence, before_a_fitness, _ = front_a[-1]
    before_b_sequence, before_b_fitness, _ = front_b[-1]

    if before_a_sequence != before_b_sequence:
        raise ValueError(
            f"Reference sequence mismatch for gene {gene_key}: the two runs do not "
            "share the same 0-mutation reference."
        )
    if not math.isclose(before_a_fitness, before_b_fitness, abs_tol=fitness_tolerance):
        raise ValueError(
            f"Before-fitness mismatch for gene {gene_key}: "
            f"{before_a_fitness} vs {before_b_fitness}. The runs likely used a "
            "different model/objective, so 'before' is not a shared baseline."
        )

    after_a_sequence, after_a_fitness, n_mutations_a = front_a[0]
    after_b_sequence, after_b_fitness, n_mutations_b = front_b[0]

    reference_sequence = before_a_sequence
    mutations_a = mutation_set(reference_sequence, after_a_sequence)
    mutations_b = mutation_set(reference_sequence, after_b_sequence)

    return {
        "gene": gene_key,
        "before": before_a_fitness,
        "after_A": after_a_fitness,
        "after_B": after_b_fitness,
        "n_mutations_A": n_mutations_a,
        "n_mutations_B": n_mutations_b,
        "delta_after": after_b_fitness - after_a_fitness,
        "delta_n_mutations": n_mutations_b - n_mutations_a,
        "shared_mutations": len(mutations_a & mutations_b),
        "a_only_mutations": len(mutations_a - mutations_b),
        "b_only_mutations": len(mutations_b - mutations_a),
    }


def build_per_gene_table(
    fronts_a: dict[str, ParetoFront],
    fronts_b: dict[str, ParetoFront],
    fitness_tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the per-gene comparison table for genes present in both runs.

    Args:
        fronts_a: Mapping of gene key to loaded Pareto front for run A.
        fronts_b: Mapping of gene key to loaded Pareto front for run B.
        fitness_tolerance: Absolute tolerance for the before-fitness check.

    Returns:
        A tuple ``(per_gene, overlap_counts)`` where ``per_gene`` is a DataFrame with
        the columns in :data:`PER_GENE_COLUMNS` (one row per overlapping gene, sorted
        by gene key) and ``overlap_counts`` reports how many genes are in both runs,
        only run A, and only run B.

    Raises:
        ValueError: If no genes overlap, or a per-gene validation fails.
    """
    keys_a = set(fronts_a)
    keys_b = set(fronts_b)
    shared_keys = sorted(keys_a & keys_b)

    overlap_counts = {
        "both": len(shared_keys),
        "a_only": len(keys_a - keys_b),
        "b_only": len(keys_b - keys_a),
    }
    if not shared_keys:
        raise ValueError("No overlapping genes between the two run folders.")

    records = [
        compare_single_gene(gene_key, fronts_a[gene_key], fronts_b[gene_key], fitness_tolerance)
        for gene_key in shared_keys
    ]
    # Each record dict is built with keys in PER_GENE_COLUMNS order, so the
    # DataFrame columns follow that order without an explicit ``columns`` argument.
    per_gene = pd.DataFrame(records)
    return per_gene, overlap_counts


def _paired_wilcoxon_pvalue(values_a: np.ndarray, values_b: np.ndarray) -> float | None:
    """Return the paired Wilcoxon signed-rank p-value, or None when undefined.

    The test is undefined when all paired differences are zero (e.g. fitness that
    saturates at 1.0 in both runs); scipy raises in that case.

    Args:
        values_a: Paired values from run A.
        values_b: Paired values from run B.

    Returns:
        The two-sided p-value, or ``None`` if the test cannot be computed.
    """
    try:
        _, p_value = wilcoxon(values_a, values_b)
    except ValueError:
        return None
    return float(p_value)


def compute_summary(
    per_gene: pd.DataFrame,
    overlap_counts: dict[str, int],
    label_a: str,
    label_b: str,
) -> dict:
    """Aggregate the per-gene table into summary statistics.

    Args:
        per_gene: Per-gene comparison table from :func:`build_per_gene_table`.
        overlap_counts: Gene-overlap counts from :func:`build_per_gene_table`.
        label_a: Human-readable label for run A.
        label_b: Human-readable label for run B.

    Returns:
        A dict of summary statistics: run labels, overlap counts, mean/median of the
        before/after/mutation-count columns and their per-gene differences, paired
        Wilcoxon p-values for after-fitness and mutation count, pooled mutation-overlap
        totals, and the mean per-gene shared-mutation fraction.
    """
    total_mutations_per_gene = (
        per_gene["shared_mutations"]
        + per_gene["a_only_mutations"]
        + per_gene["b_only_mutations"]
    )
    has_mutations = total_mutations_per_gene > 0
    shared_fraction = (
        per_gene.loc[has_mutations, "shared_mutations"]
        / total_mutations_per_gene[has_mutations]
    )

    return {
        "label_a": label_a,
        "label_b": label_b,
        "overlap_counts": overlap_counts,
        "n_genes_compared": len(per_gene),
        "mean_before": float(per_gene["before"].mean()),
        "mean_after_a": float(per_gene["after_A"].mean()),
        "mean_after_b": float(per_gene["after_B"].mean()),
        "median_after_a": float(per_gene["after_A"].median()),
        "median_after_b": float(per_gene["after_B"].median()),
        "mean_n_mutations_a": float(per_gene["n_mutations_A"].mean()),
        "mean_n_mutations_b": float(per_gene["n_mutations_B"].mean()),
        "mean_delta_after": float(per_gene["delta_after"].mean()),
        "mean_abs_delta_after": float(per_gene["delta_after"].abs().mean()),
        "mean_delta_n_mutations": float(per_gene["delta_n_mutations"].mean()),
        "mean_abs_delta_n_mutations": float(per_gene["delta_n_mutations"].abs().mean()),
        "wilcoxon_p_after": _paired_wilcoxon_pvalue(
            per_gene["after_A"].to_numpy(), per_gene["after_B"].to_numpy()
        ),
        "wilcoxon_p_n_mutations": _paired_wilcoxon_pvalue(
            per_gene["n_mutations_A"].to_numpy(), per_gene["n_mutations_B"].to_numpy()
        ),
        "pooled_shared_mutations": int(per_gene["shared_mutations"].sum()),
        "pooled_a_only_mutations": int(per_gene["a_only_mutations"].sum()),
        "pooled_b_only_mutations": int(per_gene["b_only_mutations"].sum()),
        "mean_shared_fraction": (
            float(shared_fraction.mean()) if not shared_fraction.empty else float("nan")
        ),
    }


def format_summary(summary: dict) -> str:
    """Render a summary dict as a human-readable text report.

    Args:
        summary: Summary dict from :func:`compute_summary`.

    Returns:
        A multi-line report string.
    """
    label_a = summary["label_a"]
    label_b = summary["label_b"]
    overlap = summary["overlap_counts"]

    def format_pvalue(p_value: float | None) -> str:
        return "undefined (all paired differences zero)" if p_value is None else f"{p_value:.3g}"

    lines = [
        "Comparison of two evolutionary-algorithm runs",
        "=" * 46,
        f"Run A (baseline):   {label_a}",
        f"Run B (comparison): {label_b}",
        "",
        "Gene overlap",
        "-" * 46,
        f"  in both runs:     {overlap['both']}",
        f"  only in run A:    {overlap['a_only']}",
        f"  only in run B:    {overlap['b_only']}",
        f"  genes compared:   {summary['n_genes_compared']}",
        "",
        "deepCRE prediction (fitness)",
        "-" * 46,
        f"  mean before (shared baseline): {summary['mean_before']:.4f}",
        f"  mean after  A / B:  {summary['mean_after_a']:.4f} / {summary['mean_after_b']:.4f}",
        f"  median after A / B: {summary['median_after_a']:.4f} / {summary['median_after_b']:.4f}",
        f"  mean per-gene delta (B - A):   {summary['mean_delta_after']:+.4f}",
        f"  mean per-gene |delta|:         {summary['mean_abs_delta_after']:.4f}",
        f"  paired Wilcoxon p (after A vs B): {format_pvalue(summary['wilcoxon_p_after'])}",
        "",
        "Mutation count (max mutations reached)",
        "-" * 46,
        f"  mean A / B:         {summary['mean_n_mutations_a']:.2f} / {summary['mean_n_mutations_b']:.2f}",
        f"  mean per-gene delta (B - A):   {summary['mean_delta_n_mutations']:+.2f}",
        f"  mean per-gene |delta|:         {summary['mean_abs_delta_n_mutations']:.2f}",
        f"  paired Wilcoxon p (n_mut A vs B): {format_pvalue(summary['wilcoxon_p_n_mutations'])}",
        "",
        "Introduced mutations (position + base, exact match)",
        "-" * 46,
        f"  pooled shared:      {summary['pooled_shared_mutations']}",
        f"  pooled A-only:      {summary['pooled_a_only_mutations']}",
        f"  pooled B-only:      {summary['pooled_b_only_mutations']}",
        f"  mean per-gene shared fraction: {summary['mean_shared_fraction']:.3f}",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotting functions (rendering only, not unit-tested)
# --------------------------------------------------------------------------- #
def save_figure(fig: Figure, filename: str, output_dir: str, fmt: str = "png") -> None:
    """Save a figure to ``output_dir/filename.fmt``.

    Args:
        fig: The matplotlib figure to save.
        filename: File name without extension.
        output_dir: Directory to write the figure into.
        fmt: Output format (e.g. ``"png"``, ``"svg"``, ``"pdf"``).
    """
    path = Path(output_dir) / f"{filename}.{fmt}"
    fig.savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_paired_scatter(
    per_gene: pd.DataFrame,
    x_column: str,
    y_column: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Figure:
    """Scatter one per-gene value of run A against run B with a y = x reference line.

    Args:
        per_gene: Per-gene comparison table.
        x_column: Column plotted on the x-axis (run A).
        y_column: Column plotted on the y-axis (run B).
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.

    Returns:
        The matplotlib figure.
    """
    fig, axis = plt.subplots(figsize=(5.5, 5.5))
    sns.scatterplot(data=per_gene, x=x_column, y=y_column, ax=axis, alpha=0.6, s=30)

    lower = min(per_gene[x_column].min(), per_gene[y_column].min())
    upper = max(per_gene[x_column].max(), per_gene[y_column].max())
    axis.plot([lower, upper], [lower, upper], color="grey", linestyle="--", label="y = x")

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    return fig


def plot_mutation_overlap(per_gene: pd.DataFrame) -> Figure:
    """Plot the distribution of per-gene shared-mutation fractions.

    The shared fraction is ``shared / (shared + a_only + b_only)``; genes without any
    mutations are excluded.

    Args:
        per_gene: Per-gene comparison table.

    Returns:
        The matplotlib figure.
    """
    total_mutations = (
        per_gene["shared_mutations"]
        + per_gene["a_only_mutations"]
        + per_gene["b_only_mutations"]
    )
    shared_fraction = per_gene.loc[total_mutations > 0, "shared_mutations"] / total_mutations[total_mutations > 0]

    fig, axis = plt.subplots(figsize=(6, 4.5))
    sns.histplot(shared_fraction, bins=20, ax=axis)
    axis.set_xlabel("Per-gene shared-mutation fraction (position + base)")
    axis.set_ylabel("Number of genes")
    axis.set_title("Agreement of introduced mutations between runs")
    return fig


def render_all_figures(
    per_gene: pd.DataFrame,
    label_a: str,
    label_b: str,
    output_dir: str,
    name: str,
    fmt: str = "png",
) -> None:
    """Render and save all comparison figures.

    Args:
        per_gene: Per-gene comparison table.
        label_a: Human-readable label for run A.
        label_b: Human-readable label for run B.
        output_dir: Directory to write figures into.
        name: File-name prefix for all outputs.
        fmt: Figure format.
    """
    after_scatter = plot_paired_scatter(
        per_gene,
        "after_A",
        "after_B",
        xlabel=f"after-optimization fitness ({label_a})",
        ylabel=f"after-optimization fitness ({label_b})",
        title="Optimized deepCRE prediction per gene",
    )
    save_figure(after_scatter, f"{name}_scatter_after", output_dir, fmt)

    n_mutations_scatter = plot_paired_scatter(
        per_gene,
        "n_mutations_A",
        "n_mutations_B",
        xlabel=f"max mutation count ({label_a})",
        ylabel=f"max mutation count ({label_b})",
        title="Maximum mutation count per gene",
    )
    save_figure(n_mutations_scatter, f"{name}_scatter_n_mutations", output_dir, fmt)

    overlap_figure = plot_mutation_overlap(per_gene)
    save_figure(overlap_figure, f"{name}_mutation_overlap", output_dir, fmt)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_comparison(
    run_a: str,
    run_b: str,
    output_dir: str,
    name: str,
    fmt: str = "png",
    fitness_tolerance: float = 1e-6,
) -> None:
    """Run the full comparison and write CSV, summary, and figures.

    Args:
        run_a: Path to run folder A (baseline).
        run_b: Path to run folder B (comparison).
        output_dir: Directory for all outputs (created if missing).
        name: File-name prefix for all outputs.
        fmt: Figure format.
        fitness_tolerance: Absolute tolerance for the before-fitness check.
    """
    label_a = os.path.basename(os.path.normpath(run_a))
    label_b = os.path.basename(os.path.normpath(run_b))

    fronts_a = {gene: load_pareto_front(path) for gene, path in discover_gene_fronts(run_a).items()}
    fronts_b = {gene: load_pareto_front(path) for gene, path in discover_gene_fronts(run_b).items()}

    per_gene, overlap_counts = build_per_gene_table(fronts_a, fronts_b, fitness_tolerance)
    summary = compute_summary(per_gene, overlap_counts, label_a, label_b)
    summary_text = format_summary(summary)

    os.makedirs(output_dir, exist_ok=True)
    per_gene.to_csv(os.path.join(output_dir, f"{name}_per_gene.csv"), index=False)
    with open(os.path.join(output_dir, f"{name}_summary.txt"), "w") as summary_file:
        summary_file.write(summary_text)
    render_all_figures(per_gene, label_a, label_b, output_dir, name, fmt)

    print(summary_text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare optimization results between two evolutionary-algorithm runs."
    )
    parser.add_argument("--run-a", required=True, help="Run folder A (baseline).")
    parser.add_argument("--run-b", required=True, help="Run folder B (comparison).")
    parser.add_argument("--output-dir", required=True, help="Output directory (created if missing).")
    parser.add_argument("--name", default="compare_reruns", help="File-name prefix for outputs.")
    parser.add_argument("--format", default="png", help="Figure format (png, svg, pdf).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    run_comparison(args.run_a, args.run_b, args.output_dir, args.name, args.format)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
