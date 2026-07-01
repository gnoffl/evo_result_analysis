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
from scipy.stats import wilcoxon
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


def load_per_gene_diffs(run_dir: str) -> pd.DataFrame:
    """Load per-gene, per-TF binding diffs for one run.

    Reads the run's ``*_annotated_peaks_*.csv`` (one peak per row), counts peaks
    per gene/TF for the reference and optimized (``max_mutated``) sequences, and
    returns their difference. Genes are reduced to their core id (the first two
    underscore fields, e.g. ``1_AT1G01150``) so they match across runs.

    Args:
        run_dir: Run directory containing the ``deepcis_scan`` subfolder.

    Returns:
        DataFrame indexed by (core_gene, tf) with a single ``diff`` column
        (``max_mutated`` peak count minus ``reference`` peak count).

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
    peaks["core_gene"] = peaks[GENE_COLUMN].str.split("_").str[:2].str.join("_")
    counts = (
        peaks.groupby(["core_gene", TF_COLUMN, SIGNAL_TYPE_COLUMN])
        .size()
        .unstack(SIGNAL_TYPE_COLUMN, fill_value=0)
        .reindex(columns=[REF_COLUMN, MUTATED_COLUMN], fill_value=0)
    )
    diff = counts[MUTATED_COLUMN] - counts[REF_COLUMN]
    return diff.to_frame("diff")


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


def paired_tf_significance(run_a_directory: str, run_b_directory: str) -> pd.DataFrame:
    """Per-TF paired significance between two runs over their shared genes.

    For every TF, builds per-gene diff vectors over the genes shared by both runs
    (absent gene/TF combinations count as 0) and runs three two-sided Wilcoxon
    signed-rank tests: the A-vs-B contrast (``diff_a - diff_b``), A-only
    (``diff_a`` vs 0) and B-only (``diff_b`` vs 0). The three p-value columns are
    each BH-corrected to q-values across TFs.

    Args:
        run_a_directory: First run directory (the "left"/A side of the contrast).
        run_b_directory: Second run directory (the "right"/B side of the contrast).

    Returns:
        One row per TF with the contrast columns (``median_D``, ``n_nonzero_D``,
        ``p_contrast``, ``q_contrast``, ``n_genes``) and the neutral per-run
        columns (``median_diff_a``, ``n_nonzero_a``, ``p_a``, ``q_a`` and their
        ``_b`` counterparts), sorted by ``q_contrast`` ascending.
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
                "median_diff_a": median_a,
                "n_nonzero_a": n_nonzero_a,
                "p_a": p_a,
                "median_diff_b": median_b,
                "n_nonzero_b": n_nonzero_b,
                "p_b": p_b,
            }
        )

    result = pd.DataFrame(rows)
    for p_col, q_col in [("p_contrast", "q_contrast"), ("p_a", "q_a"), ("p_b", "q_b")]:
        result[q_col] = _bh_qvalues(result[p_col])
    return result.sort_values("q_contrast").reset_index(drop=True)


def single_run_tf_significance(run_dir: str) -> pd.DataFrame:
    """Per-TF significance of diff (max_mutated − reference) vs 0 within one run.

    Loads per-gene diffs from the run's annotated-peaks CSV, tests each TF's
    distribution against zero with a two-sided Wilcoxon signed-rank test, and
    BH-corrects the p-values across TFs.

    Args:
        run_dir: Run directory containing the deepcis_scan subfolder.

    Returns:
        DataFrame with columns tf, n_genes, median_diff, n_nonzero, p_intra,
        q_intra, sorted by q_intra ascending.

    Raises:
        FileNotFoundError: If there is not exactly one annotated-peaks CSV.
    """
    diff = load_per_gene_diffs(run_dir)["diff"]
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
