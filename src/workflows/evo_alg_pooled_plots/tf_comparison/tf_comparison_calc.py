"""Loading and statistics for cross-run transcription-factor (TF) comparisons.

This module is the calculation half of the TF-comparison toolbox: it loads the
per-TF binding diffs produced by an evolutionary run's deepCIS peak scan, builds
a TF × run matrix, orders the TFs, and computes per-TF significance tests. It
contains **no plotting and no orchestration** — concrete analyses live in thin
scripts (e.g. ``minmax_comparison.py``) that import from here and from
``tf_comparison_plot.py``.

A run optimizes promoter sequences and, per gene, reports a per-TF ``diff_calc``:
the net change in predicted binding peaks vs the reference (positive = TF binding
introduced, negative = removed). Genes are the natural replicates, so per-gene
diffs (recovered from each run's ``*_annotated_peaks_*.csv``) drive the
significance tests; the aggregated ``diff_calc`` drives the descriptive matrix.
"""

import glob
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

# Data-schema column names shared by the loaders and significance functions.
TF_COLUMN = "tf"
DIFF_COLUMN = "diff_calc"
REF_COLUMN = "reference"
MUTATED_COLUMN = "max_mutated"
ANNOTATED_PEAKS_GLOB = "*_annotated_peaks_*.csv"
SIGNAL_TYPE_COLUMN = "signal_type"
GENE_COLUMN = "gene"


def count_genes(run_dir: str) -> int:
    """Return the number of genes in a run (length of its stats_*.json)."""
    matches = glob.glob(os.path.join(run_dir, "stats_*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one stats_*.json in {run_dir}, found {matches}")
    with open(matches[0], "r", encoding="utf-8") as handle:
        return len(json.load(handle))


def load_run_diffs(run_dir: str, label: str, normalization: str = "per_gene") -> pd.Series:
    """Load per-TF ``diff_calc`` for one run, normalized by gene count.

    Args:
        run_dir: Run directory containing the stats JSON and peak summary CSV.
        label: Column name to give the returned Series.
        normalization: Normalization method; "per_gene" or "fold_change", defaults to "per_gene".

    Returns:
        Series indexed by TF name with ``diff_calc / n_genes`` values.
    """
    matches = glob.glob(os.path.join(run_dir, "deepcis_scan", "*_peak_summary.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one *_peak_summary.csv in {run_dir}/deepcis_scan, found {matches}"
        )
    summary = pd.read_csv(matches[0])
    diffs = summary.set_index(TF_COLUMN)
    if normalization == "per_gene":
        diffs = diffs[DIFF_COLUMN] / count_genes(run_dir)
    elif normalization == "fold_change":
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = diffs[MUTATED_COLUMN] / diffs[REF_COLUMN]
        diffs = pd.Series(
            np.where(np.isfinite(ratio), np.log2(np.where(ratio > 0, ratio, np.nan)), np.nan),
            index=diffs.index,
        )
    diffs.name = label
    return diffs


def build_matrix(
    runs: List[Tuple[str, str]], normalization: str = "per_gene"
) -> pd.DataFrame:
    """Build a TF × run matrix of normalized diff values (NaN for absent TFs).

    Columns are built in the order given — grouping/reordering of runs is the
    caller's responsibility.

    Args:
        runs: List of ``(run_directory, display_label)`` tuples, in display order.
        normalization: Passed to :func:`load_run_diffs` ("per_gene" or "fold_change").

    Returns:
        DataFrame with one column per run (in the given order) and one row per TF
        (union across runs), values from :func:`load_run_diffs`.
    """
    columns = {label: load_run_diffs(run_dir, label, normalization) for run_dir, label in runs}
    return pd.DataFrame(columns)


def order_tfs_by_group_contrast(
    matrix: pd.DataFrame, left_columns: List[str], right_columns: List[str]
) -> pd.DataFrame:
    """Reorder rows by ``mean(left columns) - mean(right columns)`` descending.

    Missing values are ignored when averaging; a group with no values for a TF
    contributes zero. TFs high in the left group and low in the right group end
    up on top (e.g. introduced when maximizing, removed when minimizing).

    Args:
        matrix: TF × run DataFrame.
        left_columns: Column labels forming the positive (left) group.
        right_columns: Column labels forming the negative (right) group.

    Returns:
        ``matrix`` with rows reindexed by descending contrast score.
    """
    left_mean = matrix[left_columns].mean(axis=1)
    right_mean = matrix[right_columns].mean(axis=1)
    score = (left_mean.fillna(0.0) - right_mean.fillna(0.0)).sort_values(ascending=False)
    return matrix.reindex(score.index)


def order_tfs_by_mean(matrix: pd.DataFrame) -> pd.DataFrame:
    """Reorder rows by overall row mean descending (the flat / no-grouping case).

    Missing values are ignored when averaging.

    Args:
        matrix: TF × run DataFrame.

    Returns:
        ``matrix`` with rows reindexed by descending row mean.
    """
    score = matrix.mean(axis=1).sort_values(ascending=False)
    return matrix.reindex(score.index)


def top_bottom_tfs(matrix: pd.DataFrame, n: int) -> pd.DataFrame:
    """Keep only the top-N and bottom-N rows of an already-ordered matrix.

    Order is preserved (top rows before bottom rows) and overlapping rows are
    de-duplicated, so an ``n`` large enough to overlap returns every row once.

    Args:
        matrix: TF × run DataFrame whose rows are already sorted best → worst.
        n: Number of rows to keep from each end.

    Returns:
        Row-sliced copy of ``matrix``.
    """
    keep = list(dict.fromkeys(list(matrix.index[:n]) + list(matrix.index[-n:])))
    return matrix.loc[keep]


def load_per_gene_tf_counts(run_dir: str, n_core_fields: int = 2) -> pd.DataFrame:
    """Load per-replicate, per-TF reference and max_mutated peak counts for one run.

    Reads the run's ``*_annotated_peaks_*.csv`` (one peak per row) and counts peaks
    per replicate/TF for the reference and optimized (``max_mutated``) sequences.
    Each replicate is reduced to its core id — the first ``n_core_fields``
    underscore-separated fields of the sequence id — so the per-run trailing
    timestamp is dropped and replicates match across runs (see
    :func:`load_per_gene_diffs` for the field-count convention and examples).

    Args:
        run_dir: Run directory containing the ``deepcis_scan`` subfolder.
        n_core_fields: Number of leading underscore-separated fields that uniquely
            identify a replicate.

    Returns:
        DataFrame indexed by (core_gene, tf) with columns ``reference``,
        ``max_mutated`` and ``diff`` (``max_mutated`` minus ``reference``). Only
        (gene, TF) combinations with at least one reference or max_mutated peak
        appear; filling absent combinations with 0 is the caller's responsibility.

    Raises:
        FileNotFoundError: If there is not exactly one annotated-peaks CSV.
    """
    matches = glob.glob(os.path.join(run_dir, "deepcis_scan", ANNOTATED_PEAKS_GLOB))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {ANNOTATED_PEAKS_GLOB} in {run_dir}/deepcis_scan, found {matches}"
        )
    peaks = pd.read_csv(matches[0])
    peaks = peaks[peaks[SIGNAL_TYPE_COLUMN].isin([REF_COLUMN, MUTATED_COLUMN])].copy()
    peaks["core_gene"] = peaks[GENE_COLUMN].str.split("_").str[:n_core_fields].str.join("_")
    counts = (
        peaks.groupby(["core_gene", TF_COLUMN, SIGNAL_TYPE_COLUMN])
        .size()
        .unstack(SIGNAL_TYPE_COLUMN, fill_value=0)
        .reindex(columns=[REF_COLUMN, MUTATED_COLUMN], fill_value=0)
    )
    counts["diff"] = counts[MUTATED_COLUMN] - counts[REF_COLUMN]
    return counts


def load_per_gene_diffs(run_dir: str, n_core_fields: int = 2) -> pd.DataFrame:
    """Load per-replicate, per-TF binding diffs for one run.

    Reads the run's ``*_annotated_peaks_*.csv`` (one peak per row), counts peaks
    per replicate/TF for the reference and optimized (``max_mutated``) sequences,
    and returns their difference. Each replicate is reduced to its core id — the
    first ``n_core_fields`` underscore-separated fields of the sequence id — so the
    per-run trailing timestamp is dropped and replicates match across runs.

    The default of 2 fields suits natural-gene ids like
    ``1_AT1G01150_gene:..._<timestamp>`` (core ``1_AT1G01150``). Random-start
    sequences are named ``random_sequence_<index>_<timestamp>``, whose first two
    fields are always ``random_sequence``; they need ``n_core_fields=3`` (core
    ``random_sequence_000``) to keep each sequence a distinct replicate instead of
    collapsing all of them into one.

    Args:
        run_dir: Run directory containing the ``deepcis_scan`` subfolder.
        n_core_fields: Number of leading underscore-separated fields that uniquely
            identify a replicate.

    Returns:
        DataFrame indexed by (core_gene, tf) with a single ``diff`` column
        (``max_mutated`` peak count minus ``reference`` peak count).

    Raises:
        FileNotFoundError: If there is not exactly one annotated-peaks CSV.
    """
    counts = load_per_gene_tf_counts(run_dir, n_core_fields)
    return counts[["diff"]].copy()


def _wilcoxon_pvalue(values: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value of ``values`` vs 0 (NaN if undefined)."""
    try:
        return float(wilcoxon(values)[1])
    except ValueError:
        return float("nan")


def _bh_qvalues(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values, leaving NaN p-values as NaN."""
    qvalues = pd.Series(np.nan, index=pvalues.index)
    finite = pvalues.notna()
    if finite.any():
        qvalues[finite] = multipletests(pvalues[finite].to_numpy(), method="fdr_bh")[1]
    return qvalues


def _diff_series_to_matrix(
    diff: pd.Series,
    genes: List[str],
    tfs: List[str],
) -> pd.DataFrame:
    """Unstack a (core_gene, tf)-indexed diff Series into a genes × TFs matrix.

    Missing (gene, TF) combinations are filled with 0.
    """
    return diff.unstack(TF_COLUMN).reindex(index=genes, columns=tfs).fillna(0.0)


def _tf_stats(arr: np.ndarray) -> Tuple[float, int, float]:
    """Return (median, n_nonzero, wilcoxon_p) for a per-gene diff vector."""
    return float(np.median(arr)), int(np.count_nonzero(arr)), _wilcoxon_pvalue(arr)


def _pair_contrast_matrix(
    run_a_directory: str, run_b_directory: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Return ``(a_mat, b_mat, contrast_mat, shared_genes, all_tfs)`` for a run pair.

    ``a_mat``/``b_mat`` are genes × TFs matrices (absent (gene, TF) = 0) over the
    genes shared by both runs and the union of their TFs; ``contrast_mat =
    a_mat - b_mat``. Extracted so the paired/pooled/interaction significance
    functions share one copy of the intersect/unstack/fill-0 logic.

    Args:
        run_a_directory: First run directory (the "A" side of the contrast).
        run_b_directory: Second run directory (the "B" side of the contrast).

    Returns:
        Tuple of the A matrix, B matrix, their difference, the sorted shared-gene
        list, and the sorted union of TFs.
    """
    diff_a = load_per_gene_diffs(run_a_directory)["diff"]
    diff_b = load_per_gene_diffs(run_b_directory)["diff"]

    shared_genes = sorted(
        set(diff_a.index.get_level_values("core_gene"))
        & set(diff_b.index.get_level_values("core_gene"))
    )
    all_tfs = sorted(
        set(diff_a.index.get_level_values(TF_COLUMN))
        | set(diff_b.index.get_level_values(TF_COLUMN))
    )
    a_mat = _diff_series_to_matrix(diff_a, shared_genes, all_tfs)
    b_mat = _diff_series_to_matrix(diff_b, shared_genes, all_tfs)
    contrast_mat = a_mat - b_mat
    return a_mat, b_mat, contrast_mat, shared_genes, all_tfs


def paired_tf_significance(
    run_a_directory: str,
    run_b_directory: str,
    label_a: str = "a",
    label_b: str = "b",
) -> pd.DataFrame:
    """Per-TF paired significance between two runs over their shared genes.

    For every TF, builds per-gene diff vectors over the genes shared by both runs
    (absent gene/TF combinations count as 0) and runs three two-sided Wilcoxon
    signed-rank tests: the A-vs-B contrast (``diff_a - diff_b``), A-only
    (``diff_a`` vs 0) and B-only (``diff_b`` vs 0). The three p-value columns are
    each BH-corrected to q-values across TFs.

    Args:
        run_a_directory: First run directory (the "left"/A side of the contrast).
        run_b_directory: Second run directory (the "right"/B side of the contrast).
        label_a: Suffix naming the A-side columns; defaults to ``"a"``. Pass a
            meaningful name (e.g. ``"ara_model"``) to make the CSV self-describing.
        label_b: Suffix naming the B-side columns; defaults to ``"b"``.

    Returns:
        One row per TF with the contrast columns (``median_D``, ``n_nonzero_D``,
        ``p_contrast``, ``q_contrast``, ``n_genes``) and the per-run columns
        ``median_diff_<label_a>``, ``n_nonzero_<label_a>``, ``p_<label_a>``,
        ``q_<label_a>`` and their ``<label_b>`` counterparts, sorted by
        ``q_contrast`` ascending. With the default labels the per-run columns are
        the neutral ``*_a`` / ``*_b``.
    """
    a_mat, b_mat, contrast_mat, shared_genes, all_tfs = _pair_contrast_matrix(
        run_a_directory, run_b_directory
    )

    rows = []
    for tf in all_tfs:
        median_contrast, n_nonzero_contrast, p_contrast = _tf_stats(contrast_mat[tf].to_numpy())
        median_a, n_nonzero_a, p_a = _tf_stats(a_mat[tf].to_numpy())
        median_b, n_nonzero_b, p_b = _tf_stats(b_mat[tf].to_numpy())
        rows.append(
            {
                TF_COLUMN: tf,
                "n_genes": len(shared_genes),
                "median_D": median_contrast,
                "n_nonzero_D": n_nonzero_contrast,
                "p_contrast": p_contrast,
                f"median_diff_{label_a}": median_a,
                f"n_nonzero_{label_a}": n_nonzero_a,
                f"p_{label_a}": p_a,
                f"median_diff_{label_b}": median_b,
                f"n_nonzero_{label_b}": n_nonzero_b,
                f"p_{label_b}": p_b,
            }
        )

    result = pd.DataFrame(rows)
    for p_col, q_col in [
        ("p_contrast", "q_contrast"),
        (f"p_{label_a}", f"q_{label_a}"),
        (f"p_{label_b}", f"q_{label_b}"),
    ]:
        result[q_col] = _bh_qvalues(result[p_col])
    return result.sort_values("q_contrast").reset_index(drop=True)


def single_run_tf_significance(run_dir: str, n_core_fields: int = 2) -> pd.DataFrame:
    """Per-TF significance of diff (max_mutated − reference) vs 0 within one run.

    Loads per-replicate diffs from the run's annotated-peaks CSV, tests each TF's
    distribution against zero with a two-sided Wilcoxon signed-rank test, and
    BH-corrects the p-values across TFs.

    Args:
        run_dir: Run directory containing the deepcis_scan subfolder.
        n_core_fields: Number of leading underscore-separated fields identifying a
            replicate; passed to :func:`load_per_gene_diffs`. Use 3 for
            random-start runs (see that function's note).

    Returns:
        DataFrame with columns tf, n_genes, median_diff, n_nonzero, p_intra,
        q_intra, sorted by q_intra ascending.

    Raises:
        FileNotFoundError: If there is not exactly one annotated-peaks CSV.
    """
    diff = load_per_gene_diffs(run_dir, n_core_fields)["diff"]
    all_genes = sorted(set(diff.index.get_level_values("core_gene")))
    all_tfs = sorted(set(diff.index.get_level_values(TF_COLUMN)))
    mat = _diff_series_to_matrix(diff, all_genes, all_tfs)

    rows = []
    for tf in all_tfs:
        median, n_nonzero, p = _tf_stats(mat[tf].to_numpy())
        rows.append(
            {
                TF_COLUMN: tf,
                "n_genes": len(all_genes),
                "median_diff": median,
                "n_nonzero": n_nonzero,
                "p_intra": p,
            }
        )

    result = pd.DataFrame(rows)
    result["q_intra"] = _bh_qvalues(result["p_intra"])
    return result.sort_values("q_intra").reset_index(drop=True)


def per_gene_tf_binding_summary(run_dir: str, n_core_fields: int = 2) -> pd.DataFrame:
    """Mean and standard deviation of per-gene TF binding within one run.

    For every TF, aggregates over all genes in the run and reports the mean and
    sample standard deviation (ddof=1) of three per-(gene, TF) quantities: the
    ``reference`` peak count (binding in the starting sequence), the
    ``max_mutated`` peak count (binding in the optimized sequence), and their
    ``diff`` (net change introduced by the optimizer).

    Genes are the replicates. A gene with no peak for a given TF counts as 0, so
    the means divide by the full gene count — matching the 0-fill convention of
    :func:`single_run_tf_significance`. With a single gene the standard deviations
    are NaN (undefined for ddof=1).

    Args:
        run_dir: Run directory containing the ``deepcis_scan`` subfolder.
        n_core_fields: Number of leading underscore-separated fields identifying a
            replicate; passed to :func:`load_per_gene_tf_counts`. Use 3 for
            random-start runs (see :func:`load_per_gene_diffs`).

    Returns:
        DataFrame with one row per TF and columns ``tf``, ``n_genes``,
        ``mean_reference``, ``std_reference``, ``mean_max_mutated``,
        ``std_max_mutated``, ``mean_diff`` and ``std_diff``, sorted by
        ``mean_max_mutated`` descending.

    Raises:
        FileNotFoundError: If there is not exactly one annotated-peaks CSV.
    """
    counts = load_per_gene_tf_counts(run_dir, n_core_fields)
    all_genes = sorted(set(counts.index.get_level_values("core_gene")))
    all_tfs = sorted(set(counts.index.get_level_values(TF_COLUMN)))

    reference = _diff_series_to_matrix(counts[REF_COLUMN], all_genes, all_tfs)
    mutated = _diff_series_to_matrix(counts[MUTATED_COLUMN], all_genes, all_tfs)
    diff = _diff_series_to_matrix(counts["diff"], all_genes, all_tfs)

    summary = pd.DataFrame(
        {
            TF_COLUMN: all_tfs,
            "n_genes": len(all_genes),
            "mean_reference": reference.mean(axis=0).to_numpy(),
            "std_reference": reference.std(axis=0).to_numpy(),
            "mean_max_mutated": mutated.mean(axis=0).to_numpy(),
            "std_max_mutated": mutated.std(axis=0).to_numpy(),
            "mean_diff": diff.mean(axis=0).to_numpy(),
            "std_diff": diff.std(axis=0).to_numpy(),
        }
    )
    return summary.sort_values("mean_max_mutated", ascending=False).reset_index(drop=True)


def pooled_model_tf_significance(
    model_pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """Pooled per-TF model main effect across several paired gene sources.

    Each element of ``model_pairs`` is ``(model_a_run_dir, model_b_run_dir)`` for
    one gene source (the two runs share that source's genes). For every TF, the
    per-gene contrast ``D = diff_a - diff_b`` is computed within each pair and the
    D vectors from all pairs are concatenated (gene sets are disjoint across
    pairs), then tested against 0 with a two-sided Wilcoxon signed-rank test.
    p-values are BH-corrected across TFs.

    Because each ``D`` is a within-gene difference, per-gene wild-type baselines
    cancel and genes (nested in gene source) act as blocks, so pooling tests the
    model main effect with maximum power. Caveat: a TF whose model effect flips
    sign between gene sources washes out here; that interaction is surfaced by
    :func:`interaction_model_tf_significance`.

    Args:
        model_pairs: ``(model_a_run_dir, model_b_run_dir)`` tuples, one per gene
            source, sharing an A = model-A / B = model-B convention.

    Returns:
        One row per TF with columns ``tf, n_genes, median_D, n_nonzero_D,
        p_model, q_model``, sorted by ``q_model`` ascending. ``n_genes`` is the
        total pooled gene count.

    Raises:
        ValueError: If ``model_pairs`` is empty.
    """
    if not model_pairs:
        raise ValueError("model_pairs must contain at least one (run_a, run_b) pair")

    contrast_matrices = []
    tf_union: set = set()
    for run_a_directory, run_b_directory in model_pairs:
        _, _, contrast_mat, _, pair_tfs = _pair_contrast_matrix(
            run_a_directory, run_b_directory
        )
        contrast_matrices.append(contrast_mat)
        tf_union.update(pair_tfs)

    all_tfs = sorted(tf_union)
    # Reindex every pair's columns to the global TF set (fill 0) so pooled columns
    # align, and give each pair a distinct gene-index prefix to avoid collisions.
    aligned = [
        contrast_mat.reindex(columns=all_tfs, fill_value=0.0).set_axis(
            [f"pair{pair_index}_{gene}" for gene in contrast_mat.index], axis=0
        )
        for pair_index, contrast_mat in enumerate(contrast_matrices)
    ]
    pooled = pd.concat(aligned, axis=0)

    rows = []
    for tf in all_tfs:
        median_contrast, n_nonzero_contrast, p_model = _tf_stats(pooled[tf].to_numpy())
        rows.append(
            {
                TF_COLUMN: tf,
                "n_genes": len(pooled),
                "median_D": median_contrast,
                "n_nonzero_D": n_nonzero_contrast,
                "p_model": p_model,
            }
        )

    result = pd.DataFrame(rows)
    result["q_model"] = _bh_qvalues(result["p_model"])
    return result.sort_values("q_model").reset_index(drop=True)


def interaction_model_tf_significance(
    group_a_pair: Tuple[str, str],
    group_b_pair: Tuple[str, str],
    label_a: str = "a",
    label_b: str = "b",
) -> pd.DataFrame:
    """Per-TF test of whether the model effect DIFFERS between two gene groups.

    Each ``*_pair`` is ``(model_a_run_dir, model_b_run_dir)`` for one gene group.
    For every TF, the per-gene contrast ``D = diff_a - diff_b`` is formed within
    each group and the two D distributions (group A genes vs group B genes) are
    compared with a two-sided Mann-Whitney U test (unpaired: the groups have
    different genes). p-values are BH-corrected across TFs.

    A significant TF means the model swap does *different* things to that TF
    depending on the gene group (group-dependent model behavior). This is the
    unpaired interaction axis, distinct from the "is there a model effect at all"
    main effect tested by :func:`pooled_model_tf_significance`.

    Args:
        group_a_pair: ``(model_a_run_dir, model_b_run_dir)`` for the first gene group.
        group_b_pair: ``(model_a_run_dir, model_b_run_dir)`` for the second gene group.
        label_a: Suffix naming the group-A columns; defaults to ``"a"``. Pass a
            meaningful name (e.g. ``"ara_genes"``) to make the CSV self-describing.
        label_b: Suffix naming the group-B columns; defaults to ``"b"``.

    Returns:
        One row per TF with columns ``tf, n_<label_a>, n_<label_b>,
        median_D_<label_a>, median_D_<label_b>, u_stat, p_interaction,
        q_interaction``, sorted by ``q_interaction`` ascending. Degenerate TFs
        (Mann-Whitney undefined) get NaN for ``u_stat`` and ``p_interaction``.
    """
    _, _, contrast_mat_a, _, tfs_a = _pair_contrast_matrix(*group_a_pair)
    _, _, contrast_mat_b, _, tfs_b = _pair_contrast_matrix(*group_b_pair)

    all_tfs = sorted(set(tfs_a) | set(tfs_b))
    # A TF absent in a group contributes that group's all-zero D vector.
    contrast_mat_a = contrast_mat_a.reindex(columns=all_tfs, fill_value=0.0)
    contrast_mat_b = contrast_mat_b.reindex(columns=all_tfs, fill_value=0.0)

    rows = []
    for tf in all_tfs:
        d_a = contrast_mat_a[tf].to_numpy()
        d_b = contrast_mat_b[tf].to_numpy()
        try:
            u_stat, p_interaction = mannwhitneyu(d_a, d_b, alternative="two-sided")
            u_stat, p_interaction = float(u_stat), float(p_interaction)
        except ValueError:
            u_stat, p_interaction = float("nan"), float("nan")
        rows.append(
            {
                TF_COLUMN: tf,
                f"n_{label_a}": int(len(d_a)),
                f"n_{label_b}": int(len(d_b)),
                f"median_D_{label_a}": float(np.median(d_a)) if len(d_a) else float("nan"),
                f"median_D_{label_b}": float(np.median(d_b)) if len(d_b) else float("nan"),
                "u_stat": u_stat,
                "p_interaction": p_interaction,
            }
        )

    result = pd.DataFrame(rows)
    result["q_interaction"] = _bh_qvalues(result["p_interaction"])
    return result.sort_values("q_interaction").reset_index(drop=True)
