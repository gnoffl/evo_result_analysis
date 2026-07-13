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
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy.stats import wilcoxon

from analysis.overview.simple_result_stats import calculate_half_max_mutations
from workflows.candidate_selection import get_data_at_mutation_count

# A single Pareto front member: [sequence, fitness, mutation_count].
ParetoMember = list
# A full Pareto front: ordered list of members.
ParetoFront = list

# Supported comparison points (which front member to report / overlap at).
COMPARISON_POINTS = ("endpoint", "half_max")

PER_GENE_COLUMNS = [
    "gene",
    "before",
    "after_A",
    "after_B",
    "n_mutations_A",
    "n_mutations_B",
    "delta_after",
    "delta_n_mutations",
    "overlap_mutation_count",
    "shared_mutations",
    "a_only_mutations",
    "b_only_mutations",
]


class GeneComparisonSkipped(Exception):
    """Raised when a single gene cannot be compared but the run should continue.

    Distinct from :class:`ValueError`, which signals the two runs are fundamentally
    incomparable (reference/before-fitness mismatch) and must abort the whole
    analysis. A skipped gene carries a human-readable reason and is recorded rather
    than aborting.
    """


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


def mutation_set(
    reference_sequence: str,
    optimized_sequence: str,
    position_only: bool = False,
) -> set[tuple[int, str]] | set[int]:
    """Return the set of substitutions in an optimized sequence relative to reference.

    By default each substitution is identified by its 0-based position and the
    introduced base (position + base), so two runs "share" a mutation only when both
    the site and the resulting base agree. With ``position_only`` the introduced base
    is ignored and each substitution is identified by its position alone, so two runs
    share a mutation whenever they mutate the same site (regardless of the substituted
    base). The downstream ``&``/``-`` set algebra is identical for either element type.

    Args:
        reference_sequence: The 0-mutation reference sequence.
        optimized_sequence: The optimized sequence (same length as reference).
        position_only: If True, return positions only; otherwise (position, base).

    Returns:
        Set of ``(position, introduced_base)`` tuples, or set of ``position`` ints
        when ``position_only`` is True.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(reference_sequence) != len(optimized_sequence):
        raise ValueError(
            "Reference and optimized sequences differ in length: "
            f"{len(reference_sequence)} vs {len(optimized_sequence)}"
        )
    if position_only:
        return {
            position
            for position, (reference_base, optimized_base) in enumerate(
                zip(reference_sequence, optimized_sequence)
            )
            if reference_base != optimized_base
        }
    return {
        (position, optimized_base)
        for position, (reference_base, optimized_base) in enumerate(
            zip(reference_sequence, optimized_sequence)
        )
        if reference_base != optimized_base
    }


def resolve_comparison_members(
    front_a: ParetoFront,
    front_b: ParetoFront,
    comparison_point: str,
) -> tuple[Sequence, Sequence, Sequence, Sequence, int]:
    """Resolve which front members to report and which to compute the overlap from.

    The mutation-set overlap is always computed at a *common* mutation count so two
    runs with different mutation counts can in principle reach 100% overlap. The
    reported fitness / mutation count depends on the comparison point:

    - ``endpoint``: the reported members are each run's true optimized endpoint
      (``front[0]``); the common overlap count is the **largest mutation count present
      on both fronts** (their rounded mutation-count sets always share 0, so this is
      well defined and never misses). Pareto fronts drop dominated points, so
      intermediate counts have gaps and the two endpoint counts rarely coincide;
      taking the largest shared count keeps every gene comparable without skipping.
    - ``half_max``: the common overlap count is ``min`` of the two runs' half-max
      mutation counts (:func:`calculate_half_max_mutations`); both the reported and
      the overlap members are each front sampled at that common count (so the
      reported mutation counts are equal and ``delta_n_mutations`` is zero). Half-max
      counts sit in the dense part of the front, so a missing count is rare; when it
      does happen the gene is skipped.

    Args:
        front_a: Pareto front of run A.
        front_b: Pareto front of run B.
        comparison_point: One of :data:`COMPARISON_POINTS`.

    Returns:
        ``(reported_a, reported_b, overlap_member_a, overlap_member_b, overlap_count)``.

    Raises:
        ValueError: If ``comparison_point`` is not a supported value.
        GeneComparisonSkipped: If the common count is absent on either front
            (``half_max`` only; ``endpoint`` cannot miss by construction).
    """
    if comparison_point == "endpoint":
        counts_a = {round(member[2]) for member in front_a}
        counts_b = {round(member[2]) for member in front_b}
        # Both fronts include the 0-mutation reference, so the intersection is never
        # empty and ``max`` is always defined.
        overlap_count = max(counts_a & counts_b)
    elif comparison_point == "half_max":
        overlap_count = min(
            calculate_half_max_mutations(front_a),
            calculate_half_max_mutations(front_b),
        )
    else:
        raise ValueError(
            f"Unknown comparison_point {comparison_point!r}; "
            f"expected one of {COMPARISON_POINTS}."
        )

    try:
        overlap_member_a = get_data_at_mutation_count(front_a, overlap_count)
        overlap_member_b = get_data_at_mutation_count(front_b, overlap_count)
    except ValueError as error:
        raise GeneComparisonSkipped(
            f"common count {overlap_count} absent on a front"
        ) from error

    if comparison_point == "endpoint":
        reported_a, reported_b = front_a[0], front_b[0]
    else:
        reported_a, reported_b = overlap_member_a, overlap_member_b

    return reported_a, reported_b, overlap_member_a, overlap_member_b, overlap_count


def compare_single_gene(
    gene_key: str,
    front_a: ParetoFront,
    front_b: ParetoFront,
    fitness_tolerance: float = 1e-6,
    comparison_point: str = "endpoint",
    position_only: bool = False,
) -> dict:
    """Compute the comparison record for a single gene present in both runs.

    The before-optimization baseline must be identical across runs (same gene, same
    reference). Both the reference sequence and its stored fitness are validated; a
    mismatch aborts the analysis because it signals the runs are not comparable.

    The reported fitness / mutation count and the mutation-set overlap are resolved by
    :func:`resolve_comparison_members` according to ``comparison_point``. The overlap
    is always computed at a common mutation count (recorded in
    ``overlap_mutation_count``).

    Args:
        gene_key: Gene identity key.
        front_a: Pareto front of run A (baseline).
        front_b: Pareto front of run B (comparison).
        fitness_tolerance: Absolute tolerance for the before-fitness equality check.
        comparison_point: One of :data:`COMPARISON_POINTS`.
        position_only: If True, match mutations by position only (ignore the base).

    Returns:
        A record dict with the keys listed in :data:`PER_GENE_COLUMNS`.

    Raises:
        ValueError: If reference sequences differ, or before-fitness values differ
            by more than ``fitness_tolerance``.
        GeneComparisonSkipped: If the common overlap count is absent on either front.
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

    (
        reported_a,
        reported_b,
        overlap_member_a,
        overlap_member_b,
        overlap_count,
    ) = resolve_comparison_members(front_a, front_b, comparison_point)

    _, after_a_fitness, n_mutations_a = reported_a
    _, after_b_fitness, n_mutations_b = reported_b

    reference_sequence = before_a_sequence
    mutations_a = mutation_set(reference_sequence, overlap_member_a[0], position_only)
    mutations_b = mutation_set(reference_sequence, overlap_member_b[0], position_only)

    return {
        "gene": gene_key,
        "before": before_a_fitness,
        "after_A": after_a_fitness,
        "after_B": after_b_fitness,
        "n_mutations_A": n_mutations_a,
        "n_mutations_B": n_mutations_b,
        "delta_after": after_b_fitness - after_a_fitness,
        "delta_n_mutations": n_mutations_b - n_mutations_a,
        "overlap_mutation_count": overlap_count,
        "shared_mutations": len(mutations_a & mutations_b),
        "a_only_mutations": len(mutations_a - mutations_b),
        "b_only_mutations": len(mutations_b - mutations_a),
    }


def build_per_gene_table(
    fronts_a: dict[str, ParetoFront],
    fronts_b: dict[str, ParetoFront],
    fitness_tolerance: float = 1e-6,
    comparison_point: str = "endpoint",
    position_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, int], list[tuple[str, str]]]:
    """Build the per-gene comparison table for genes present in both runs.

    Genes that cannot be compared at the chosen comparison point (the common overlap
    count is absent on a front) are skipped and recorded rather than aborting the run;
    a reference/before-fitness mismatch still aborts (raises ``ValueError``), because
    it means the runs are not comparable at all.

    Args:
        fronts_a: Mapping of gene key to loaded Pareto front for run A.
        fronts_b: Mapping of gene key to loaded Pareto front for run B.
        fitness_tolerance: Absolute tolerance for the before-fitness check.
        comparison_point: One of :data:`COMPARISON_POINTS`.
        position_only: If True, match mutations by position only (ignore the base).

    Returns:
        A tuple ``(per_gene, overlap_counts, excluded_genes)`` where ``per_gene`` is a
        DataFrame with the columns in :data:`PER_GENE_COLUMNS` (one row per compared
        gene, sorted by gene key), ``overlap_counts`` reports how many genes are in
        both runs / only run A / only run B, and ``excluded_genes`` is a list of
        ``(gene_key, reason)`` for genes skipped at this comparison point.

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

    records = []
    excluded_genes: list[tuple[str, str]] = []
    for gene_key in shared_keys:
        try:
            records.append(
                compare_single_gene(
                    gene_key,
                    fronts_a[gene_key],
                    fronts_b[gene_key],
                    fitness_tolerance,
                    comparison_point,
                    position_only,
                )
            )
        except GeneComparisonSkipped as skip:
            excluded_genes.append((gene_key, str(skip)))

    # Each record dict is built with keys in PER_GENE_COLUMNS order, so the
    # DataFrame columns follow that order. Guard the empty case (every gene skipped)
    # so the expected columns are still present.
    if records:
        per_gene = pd.DataFrame(records)
    else:
        per_gene = pd.DataFrame({column: [] for column in PER_GENE_COLUMNS})
    return per_gene, overlap_counts, excluded_genes


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
    comparison_point: str = "endpoint",
    position_only: bool = False,
    n_excluded: int = 0,
) -> dict:
    """Aggregate the per-gene table into summary statistics.

    Args:
        per_gene: Per-gene comparison table from :func:`build_per_gene_table`.
        overlap_counts: Gene-overlap counts from :func:`build_per_gene_table`.
        label_a: Human-readable label for run A.
        label_b: Human-readable label for run B.
        comparison_point: The comparison point used (one of :data:`COMPARISON_POINTS`).
        position_only: Whether mutations were matched by position only.
        n_excluded: Number of genes skipped at this comparison point.

    Returns:
        A dict of summary statistics: run labels, comparison point, match mode, overlap
        counts, mean/median of the before/after/mutation-count columns and their
        per-gene differences, the mean common overlap count used, paired Wilcoxon
        p-values for after-fitness and mutation count, pooled mutation-overlap totals,
        the mean per-gene shared-mutation fraction, and the skipped-gene count.
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
        "comparison_point": comparison_point,
        "position_only": position_only,
        "n_excluded": n_excluded,
        "mean_overlap_mutation_count": (
            float(per_gene["overlap_mutation_count"].mean())
            if not per_gene.empty
            else float("nan")
        ),
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
    comparison_point = summary["comparison_point"]
    match_mode = "position only" if summary["position_only"] else "position + base"
    point_labels = {"endpoint": "optimized endpoint", "half_max": "half-max point"}
    point_label = point_labels.get(comparison_point, comparison_point)

    def format_pvalue(p_value: float | None) -> str:
        return "undefined (all paired differences zero)" if p_value is None else f"{p_value:.3g}"

    # The reported after-fitness / mutation count meaning depends on the point.
    if comparison_point == "half_max":
        mutation_count_header = "Mutation count (common half-max count; delta is 0 by design)"
    else:
        mutation_count_header = "Mutation count (max mutations reached)"

    lines = [
        "Comparison of two evolutionary-algorithm runs",
        "=" * 46,
        f"Run A (baseline):   {label_a}",
        f"Run B (comparison): {label_b}",
        f"Comparison point:   {point_label} ({comparison_point})",
        f"Mutation match:     {match_mode}",
        "",
        "Gene overlap",
        "-" * 46,
        f"  in both runs:     {overlap['both']}",
        f"  only in run A:    {overlap['a_only']}",
        f"  only in run B:    {overlap['b_only']}",
        f"  genes compared:   {summary['n_genes_compared']}",
        f"  genes skipped:    {summary['n_excluded']}",
        f"  mean overlap count used: {summary['mean_overlap_mutation_count']:.2f}",
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
        mutation_count_header,
        "-" * 46,
        f"  mean A / B:         {summary['mean_n_mutations_a']:.2f} / {summary['mean_n_mutations_b']:.2f}",
        f"  mean per-gene delta (B - A):   {summary['mean_delta_n_mutations']:+.2f}",
        f"  mean per-gene |delta|:         {summary['mean_abs_delta_n_mutations']:.2f}",
        f"  paired Wilcoxon p (n_mut A vs B): {format_pvalue(summary['wilcoxon_p_n_mutations'])}",
        "",
        f"Introduced mutations ({match_mode}, at common overlap count)",
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


def plot_mutation_overlap(per_gene: pd.DataFrame, position_only: bool = False) -> Figure:
    """Plot the distribution of per-gene shared-mutation fractions.

    The shared fraction is ``shared / (shared + a_only + b_only)``; genes without any
    mutations are excluded.

    Args:
        per_gene: Per-gene comparison table.
        position_only: Whether mutations were matched by position only (affects the
            axis wording).

    Returns:
        The matplotlib figure.
    """
    total_mutations = (
        per_gene["shared_mutations"]
        + per_gene["a_only_mutations"]
        + per_gene["b_only_mutations"]
    )
    shared_fraction = per_gene.loc[total_mutations > 0, "shared_mutations"] / total_mutations[total_mutations > 0]

    match_mode = "position only" if position_only else "position + base"
    fig, axis = plt.subplots(figsize=(6, 4.5))
    sns.histplot(shared_fraction, bins=20, ax=axis)
    axis.set_xlabel(f"Per-gene shared-mutation fraction ({match_mode})")
    axis.set_ylabel("Number of genes")
    axis.set_title("Agreement of introduced mutations between runs")
    return fig


def render_all_figures(
    per_gene: pd.DataFrame,
    label_a: str,
    label_b: str,
    output_dir: str,
    fmt: str = "png",
    comparison_point: str = "endpoint",
    position_only: bool = False,
) -> None:
    """Render and save all comparison figures.

    Figures are written with fixed base names (``scatter_after``,
    ``scatter_n_mutations``, ``mutation_overlap``) into ``output_dir``, which is
    expected to be a per-configuration subfolder.

    In ``half_max`` mode the reported mutation counts are equal by design, so the
    mutation-count scatter collapses onto the ``y = x`` line (expected, harmless).

    Args:
        per_gene: Per-gene comparison table.
        label_a: Human-readable label for run A.
        label_b: Human-readable label for run B.
        output_dir: Directory to write figures into.
        fmt: Figure format.
        comparison_point: The comparison point used (affects the axis/title wording).
        position_only: Whether mutations were matched by position only.
    """
    point_labels = {"endpoint": "optimized endpoint", "half_max": "half-max point"}
    point_label = point_labels.get(comparison_point, comparison_point)
    if comparison_point == "half_max":
        fitness_word = "fitness at half-max point"
        count_word = "common overlap mutation count"
    else:
        fitness_word = "after-optimization fitness"
        count_word = "max mutation count"

    after_scatter = plot_paired_scatter(
        per_gene,
        "after_A",
        "after_B",
        xlabel=f"{fitness_word} ({label_a})",
        ylabel=f"{fitness_word} ({label_b})",
        title=f"deepCRE prediction per gene ({point_label})",
    )
    save_figure(after_scatter, "scatter_after", output_dir, fmt)

    n_mutations_scatter = plot_paired_scatter(
        per_gene,
        "n_mutations_A",
        "n_mutations_B",
        xlabel=f"{count_word} ({label_a})",
        ylabel=f"{count_word} ({label_b})",
        title=f"Mutation count per gene ({point_label})",
    )
    save_figure(n_mutations_scatter, "scatter_n_mutations", output_dir, fmt)

    overlap_figure = plot_mutation_overlap(per_gene, position_only)
    save_figure(overlap_figure, "mutation_overlap", output_dir, fmt)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_comparison(
    run_a: str,
    run_b: str,
    output_dir: str,
    fmt: str = "png",
    fitness_tolerance: float = 1e-6,
    comparison_point: str = "endpoint",
    position_only: bool = False,
) -> None:
    """Run the full comparison and write CSV, summary, and figures.

    All outputs for a given configuration go into a per-configuration subfolder of
    ``output_dir`` named for the comparison point and the match mode — e.g.
    ``endpoint/``, ``half_max/``, ``endpoint_position_only/`` — so runs with different
    settings never overwrite each other. Within a configuration subfolder the files
    have fixed names (``per_gene.csv``, ``summary.txt``, ``excluded_genes.txt`` and the
    figures), so ``output_dir`` itself identifies the comparison (e.g. one
    ``output_dir`` per run pair).

    Args:
        run_a: Path to run folder A (baseline).
        run_b: Path to run folder B (comparison).
        output_dir: Root output directory (created if missing); a per-configuration
            subfolder is created inside it.
        fmt: Figure format.
        fitness_tolerance: Absolute tolerance for the before-fitness check.
        comparison_point: One of :data:`COMPARISON_POINTS`.
        position_only: If True, match mutations by position only (ignore the base).
    """
    label_a = os.path.basename(os.path.normpath(run_a))
    label_b = os.path.basename(os.path.normpath(run_b))

    config_folder = comparison_point
    if position_only:
        config_folder += "_position_only"
    config_dir = os.path.join(output_dir, config_folder)

    fronts_a = {gene: load_pareto_front(path) for gene, path in discover_gene_fronts(run_a).items()}
    fronts_b = {gene: load_pareto_front(path) for gene, path in discover_gene_fronts(run_b).items()}

    per_gene, overlap_counts, excluded_genes = build_per_gene_table(
        fronts_a, fronts_b, fitness_tolerance, comparison_point, position_only
    )
    summary = compute_summary(
        per_gene,
        overlap_counts,
        label_a,
        label_b,
        comparison_point,
        position_only,
        len(excluded_genes),
    )
    summary_text = format_summary(summary)

    os.makedirs(config_dir, exist_ok=True)
    per_gene.to_csv(os.path.join(config_dir, "per_gene.csv"), index=False)
    with open(os.path.join(config_dir, "summary.txt"), "w") as summary_file:
        summary_file.write(summary_text)
    render_all_figures(
        per_gene, label_a, label_b, config_dir, fmt, comparison_point, position_only
    )

    if excluded_genes:
        excluded_path = os.path.join(config_dir, "excluded_genes.txt")
        with open(excluded_path, "w") as excluded_file:
            for gene_key, reason in excluded_genes:
                excluded_file.write(f"{gene_key}\t{reason}\n")
        print(f"Skipped {len(excluded_genes)} gene(s); written to {excluded_path}")

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
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Output directory for this comparison (created if missing). A "
            "per-configuration subfolder (e.g. 'endpoint/', 'half_max_position_only/') "
            "is created inside it with fixed-name outputs."
        ),
    )
    parser.add_argument("--format", default="png", help="Figure format (png, svg, pdf).")
    parser.add_argument(
        "--comparison-point",
        choices=COMPARISON_POINTS,
        default="endpoint",
        help=(
            "Point at which to compare solutions: 'endpoint' (each run's maximally "
            "mutated front[0]) or 'half_max' (each run's half-max mutation point). "
            "The mutation overlap is always computed at a common mutation count."
        ),
    )
    parser.add_argument(
        "--position-only",
        action="store_true",
        help=(
            "Match mutations by position only, ignoring the substituted base "
            "(default: match by position + base)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    run_comparison(
        args.run_a,
        args.run_b,
        args.output_dir,
        args.format,
        comparison_point=args.comparison_point,
        position_only=args.position_only,
    )


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
