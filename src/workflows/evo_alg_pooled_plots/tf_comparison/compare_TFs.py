"""Compare introduced/removed transcription factors across a few evolutionary runs.

Each run optimizes promoter sequences to maximize ("max") or minimize ("min")
predicted gene fitness. Its deepCIS peak summary reports a per-TF ``diff_calc``:
the net change in predicted binding peaks vs the reference (positive = TF binding
introduced, negative = removed).

This script loads the runs listed in ``RUNS``, normalizes each ``diff_calc`` by
the number of genes in the run (``len`` of the run's ``stats_*.json``), orders the
TFs by a max-minus-min score, and draws a heatmap (columns = runs, rows = TFs).
With that ordering, max and min runs should read as mirror images.

Two layers of significance are computed:

1. **Inter-run (paired contrast):** For the two Arabidopsis runs (``ARA_MAX_RUN``
   / ``ARA_MIN_RUN``), which share the same genes, three two-sided Wilcoxon
   signed-rank tests per TF (max-vs-min contrast, max-only and min-only change vs
   reference) are BH-corrected to q-values and written to
   ``compare_TFs_significance.csv``. The contrast q-value stars are appended to
   the TF row labels in the heatmap.

2. **Intra-run:** For every run independently, a per-TF Wilcoxon test of
   ``diff = max_mutated − reference`` vs 0 over all genes in that run is
   BH-corrected and written to ``compare_TFs_significance_<label>.csv``.
   Significant intra-run q-values are shown as stars appended to each cell's
   numeric annotation in the heatmap.

It is a one-off analysis script: edit ``RUNS`` and run it directly.
"""

import glob
import json
import os
from typing import Dict, List, Optional, Tuple, Union, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

# (run directory, direction "max"/"min", display label)
RUNS: List[Tuple[str, str, str]] = [
    ("/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/ara_msr_max_single", "max", "ara max"),
    ("/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/GOF_LOF/GOF/GOF_single", "max", "GOF"),
    ("/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/ara_msr_min_single", "min", "ara min"),
    ("/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/GOF_LOF/LOF/LOF_single", "min", "LOF"),
]

OUTPUT_BASENAME = "compare_TFs"
FIGURE_FORMAT = "png"

TF_COLUMN = "tf"
DIFF_COLUMN = "diff_calc"
REF_COLUMN = "reference"
MUTATED_COLUMN = "max_mutated"
COLORMAP = "RdBu_r"

# The two Arabidopsis runs used for significance testing. They share the same
# genes, so the per-gene diffs can be paired by gene.
ARA_MAX_RUN = RUNS[0][0]
ARA_MIN_RUN = RUNS[2][0]

ANNOTATED_PEAKS_GLOB = "*_annotated_peaks_*.csv"
SIGNAL_TYPE_COLUMN = "signal_type"
GENE_COLUMN = "gene"
SIGNIFICANCE_BASENAME = "compare_TFs_significance"
# q-value thresholds (most stringent first) mapped to the star annotation.
STAR_THRESHOLDS: List[Tuple[float, str]] = [(0.001, "***"), (0.01, "**"), (0.05, "*")]


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


def build_matrix(runs: List[Tuple[str, str, str]], normalization: str = "per_gene") -> pd.DataFrame:
    """Build a TF x run matrix of normalized diff_calc (NaN for absent TFs).

    Columns are ordered with all "max" runs first, then all "min" runs,
    preserving the order within each group.
    """
    ordered = [r for r in runs if r[1] == "max"] + [r for r in runs if r[1] == "min"]
    columns = {label: load_run_diffs(run_dir, label, normalization) for run_dir, _, label in ordered}
    return pd.DataFrame(columns)


def order_tfs(matrix: pd.DataFrame, directions: Dict[str, str]) -> pd.DataFrame:
    """Reorder rows by ``mean(max runs) - mean(min runs)`` descending.

    Missing values are ignored when averaging; a direction with no values for a
    TF contributes zero. TFs introduced in max and removed in min end up on top.
    """
    max_cols = [c for c in matrix.columns if directions[c] == "max"]
    min_cols = [c for c in matrix.columns if directions[c] == "min"]
    max_mean = cast(pd.Series, matrix[max_cols].mean(axis=1))
    min_mean = cast(pd.Series, matrix[min_cols].mean(axis=1))
    score = (max_mean.fillna(0.0) - min_mean.fillna(0.0)).sort_values(ascending=False)
    return matrix.reindex(score.index)


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


def paired_tf_significance(max_run_dir: str, min_run_dir: str) -> pd.DataFrame:
    """Per-TF paired significance between a max and a min run over shared genes.

    For every TF, builds per-gene diff vectors over the genes shared by both runs
    (absent gene/TF combinations count as 0) and runs three two-sided Wilcoxon
    signed-rank tests: the max-vs-min contrast (``diff_max - diff_min``), max-only
    (``diff_max`` vs 0) and min-only (``diff_min`` vs 0). The three p-value
    columns are each BH-corrected to q-values across TFs.

    Args:
        max_run_dir: Maximization run directory.
        min_run_dir: Minimization run directory.

    Returns:
        One row per TF with median per-gene diffs (effect size), non-zero gene
        counts, raw p-values and BH q-values, sorted by ``q_contrast`` ascending.
    """
    diff_max = load_per_gene_diffs(max_run_dir)["diff"]
    diff_min = load_per_gene_diffs(min_run_dir)["diff"]

    shared_genes = sorted(
        set(diff_max.index.get_level_values("core_gene"))
        & set(diff_min.index.get_level_values("core_gene"))
    )
    all_tfs = sorted(
        set(diff_max.index.get_level_values(TF_COLUMN))
        | set(diff_min.index.get_level_values(TF_COLUMN))
    )
    max_mat = _diff_series_to_matrix(diff_max, shared_genes, all_tfs)
    min_mat = _diff_series_to_matrix(diff_min, shared_genes, all_tfs)
    contrast_mat = max_mat - min_mat

    rows = []
    for tf in all_tfs:
        median_contrast, n_nonzero_contrast, p_contrast = _tf_stats(contrast_mat[tf].to_numpy())
        median_max, n_nonzero_max, p_max = _tf_stats(max_mat[tf].to_numpy())
        median_min, n_nonzero_min, p_min = _tf_stats(min_mat[tf].to_numpy())
        rows.append(
            {
                TF_COLUMN: tf,
                "n_genes": len(shared_genes),
                "median_D": median_contrast,
                "n_nonzero_D": n_nonzero_contrast,
                "p_contrast": p_contrast,
                "median_diff_max": median_max,
                "n_nonzero_max": n_nonzero_max,
                "p_max": p_max,
                "median_diff_min": median_min,
                "n_nonzero_min": n_nonzero_min,
                "p_min": p_min,
            }
        )

    result = pd.DataFrame(rows)
    for p_col, q_col in [("p_contrast", "q_contrast"), ("p_max", "q_max"), ("p_min", "q_min")]:
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


def q_to_stars(qvalue: float) -> str:
    """Return the significance star string for a q-value (empty if not significant)."""
    if pd.isna(qvalue):
        return ""
    for threshold, stars in STAR_THRESHOLDS:
        if qvalue < threshold:
            return stars
    return ""


def plot_heatmap(
    matrix: pd.DataFrame,
    n_max_runs: int,
    annotate: bool,
    cbar_label: str = "diff_calc / gene",
    row_stars: Optional[Dict[str, str]] = None,
    cell_stars: Optional[Dict[str, Dict[str, str]]] = None,
) -> Figure:
    """Draw the cross-run TF heatmap on a symmetric diverging scale.

    When ``row_stars`` maps a TF to a non-empty star string, that string is
    appended to the TF's row label to mark inter-run significance.

    When ``cell_stars`` maps a run label to a ``{tf: stars}`` dict, the stars
    are appended to each cell's numeric annotation to mark intra-run significance
    (i.e. whether diff vs reference is significant within that run).

    Args:
        matrix: TF × run DataFrame of normalized diff values.
        n_max_runs: Number of maximization runs (for the separator line).
        annotate: Whether to annotate cells with numeric values (and stars).
        cbar_label: Colour bar axis label.
        row_stars: Optional inter-run significance stars keyed by TF name.
        cell_stars: Optional intra-run significance stars keyed by run label then
            TF name; only used when ``annotate`` is True.

    Returns:
        Matplotlib Figure with a single heatmap axes.
    """
    limit = float(np.nanmax(np.abs(matrix.to_numpy()))) or 1.0
    n_tfs, n_runs = matrix.shape
    fig, ax = plt.subplots(figsize=(max(4.0, 0.9 * n_runs + 2.5), max(4.0, 0.3 * n_tfs + 1.0)))

    annot_arg: Union[bool, pd.DataFrame] = annotate
    fmt_arg = ".2f"
    if annotate and cell_stars:
        annot_data = pd.DataFrame("", index=matrix.index, columns=matrix.columns)
        for col in matrix.columns:
            col_stars = cell_stars.get(str(col), {})
            for tf in matrix.index:
                val = matrix.loc[tf, col]
                if pd.isna(val):
                    annot_data.loc[tf, col] = ""
                else:
                    stars = col_stars.get(str(tf), "")
                    annot_data.loc[tf, col] = f"{val:.2f}{stars}"
        annot_arg = annot_data
        fmt_arg = ""

    sns.heatmap(
        matrix,
        cmap=COLORMAP,
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=annot_arg,
        fmt=fmt_arg,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    if 0 < n_max_runs < n_runs:
        ax.axvline(n_max_runs, color="black", linewidth=2.0)
    ax.set_xlabel("run")
    ax.set_ylabel("TF family")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    if row_stars:
        labels = [
            f"{tf} {row_stars[tf]: <3}" if row_stars.get(tf) else str(tf) + "    " for tf in matrix.index
        ]
        ax.set_yticklabels(labels, rotation=0)
    fig.tight_layout()
    return fig


def main() -> None:
    """Build the matrix, compute significance tests, and save annotated heatmaps."""
    directions = {label: direction for _, direction, label in RUNS}
    n_max_runs = sum(1 for _, direction, _ in RUNS if direction == "max")
    output_folder = os.path.dirname(__file__)

    # Inter-run paired significance for the two ara runs (they share the same genes).
    significance = paired_tf_significance(ARA_MAX_RUN, ARA_MIN_RUN)
    significance_path = os.path.join(output_folder, f"{SIGNIFICANCE_BASENAME}.csv")
    significance.to_csv(significance_path, index=False)
    print(f"Saved {significance_path}")
    row_stars = dict(zip(significance[TF_COLUMN], significance["q_contrast"].map(q_to_stars)))

    # Intra-run significance (diff vs reference within each run). The ara runs
    # reuse the q_max / q_min columns already computed by paired_tf_significance;
    # GOF/LOF are computed independently via single_run_tf_significance.
    ara_max_label = next(label for run_dir, _, label in RUNS if run_dir == ARA_MAX_RUN)
    ara_min_label = next(label for run_dir, _, label in RUNS if run_dir == ARA_MIN_RUN)

    cell_stars: Dict[str, Dict[str, str]] = {
        ara_max_label: dict(
            zip(significance[TF_COLUMN], significance["q_max"].map(q_to_stars))
        ),
        ara_min_label: dict(
            zip(significance[TF_COLUMN], significance["q_min"].map(q_to_stars))
        ),
    }

    ara_run_dirs = {ARA_MAX_RUN, ARA_MIN_RUN}
    for run_dir, _, label in RUNS:
        if run_dir in ara_run_dirs:
            continue
        intra_sig = single_run_tf_significance(run_dir)
        safe_label = label.replace(" ", "_")
        intra_path = os.path.join(
            output_folder, f"{SIGNIFICANCE_BASENAME}_{safe_label}.csv"
        )
        intra_sig.to_csv(intra_path, index=False)
        print(f"Saved {intra_path}")
        cell_stars[label] = dict(
            zip(intra_sig[TF_COLUMN], intra_sig["q_intra"].map(q_to_stars))
        )

    matrix_per_gene = order_tfs(build_matrix(RUNS, normalization="per_gene"), directions)
    matrix_fold_change = order_tfs(build_matrix(RUNS, normalization="fold_change"), directions)

    norm_labels = {"_per_gene": "diff_calc / gene", "_log_fold_change": "log2 fold change"}
    for matrix, suffix in [
        (matrix_per_gene, "_per_gene"),
        (matrix_fold_change, "_log_fold_change"),
    ]:
        fig = plot_heatmap(
            matrix,
            n_max_runs,
            True,
            cbar_label=norm_labels[suffix],
            row_stars=row_stars,
            cell_stars=cell_stars,
        )
        output_path = os.path.join(output_folder, f"{OUTPUT_BASENAME}{suffix}.{FIGURE_FORMAT}")
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
