"""Compare deepCRE predictions for binding vs non-binding sequences, per TF.

Tests whether deepCRE assigns significantly different prediction scores to
STAR-seq construct sequences labelled ``binding`` (contain a predicted TF site)
versus ``non_binding`` (controls), separately for each TF family (WRKY, bHLH).

Two independent backgrounds are analysed and kept strictly separate:

1. **Vector (construct) background** -- inserts embedded in the experimental
   plasmid with two barcodes, scored by deepCRE. Cached predictions come from
   :func:`predict_sequences` (``predictions_cache.csv``).
2. **Genomic (native) background** -- inserts mapped onto their native deepCRE
   reference window and scored there. Built by the per-TF correlation pipelines
   (``prediction_mutated`` column).

Run from the repository root inside the ``deepCREshap`` conda environment::

    conda run -n deepCREshap python \\
        src/workflows/overlap_analysis/binding_vs_nonbinding/binding_vs_nonbinding.py
"""
import os
from typing import List, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon

from workflows.overlap_analysis.correct_construct.analyze_in_true_background import (
    _BINDING_PALETTE,
    _format_p_value,
    _parse_binding_category,
    average_barcode_predictions,
    predict_sequences,
    save_figure,
)
from workflows.overlap_analysis.starrseq_deepcre_correlation_WRKY import (
    prepare_wrky_enrichment_df,
)
from workflows.overlap_analysis.starrseq_deepcre_correlation_bHLH import (
    prepare_bhlh_enrichment_df,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
FMT = "png"

_CATEGORY_ORDER = ["binding", "non_binding"]


def build_vector_comparison_df(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-sequence comparison table for the vector background.

    Barcode predictions are averaged into one value per sequence, then the
    binding category and TF family are parsed from the sequence ID.

    Args:
        predictions_df: DataFrame with columns ``id``, ``barcode``,
            ``prediction`` (as returned by :func:`predict_sequences`).

    Returns:
        DataFrame with columns ``tf_family``, ``binding_category``,
        ``prediction`` (one row per sequence).
    """
    averaged = average_barcode_predictions(predictions_df)
    averaged["tf_family"] = averaged["id"].str.split("_").str[0]
    averaged["binding_category"] = averaged["id"].apply(_parse_binding_category)
    return cast(
        pd.DataFrame, averaged[["tf_family", "binding_category", "prediction"]].copy()
    )


def build_genomic_comparison_df(
    wrky_df: pd.DataFrame, bhlh_df: pd.DataFrame
) -> pd.DataFrame:
    """Build the comparison table for the genomic (native) background.

    Each input is single-TF, so the TF family is tagged per source. The
    ``prediction_mutated`` score becomes ``prediction`` and only
    ``binding`` / ``non_binding`` rows are kept (``reference`` rows, which are
    baselines rather than experimental categories, are dropped).

    The correlation pipeline emits one row per (sequence, gene window, STAR-seq
    condition). Since deepCRE scores a sequence purely from its reference window
    -- the Light/Dark condition has no effect on ``prediction_mutated`` -- the
    Light/Dark rows are exact duplicates here and are collapsed by keeping one
    row per ``(starr_full_name, gene)``. Distinct gene windows are retained:
    the same fragment in a different reference context is a different prediction.

    Args:
        wrky_df: Analysis-ready WRKY DataFrame from
            :func:`prepare_wrky_enrichment_df`.
        bhlh_df: Analysis-ready bHLH DataFrame from
            :func:`prepare_bhlh_enrichment_df`.

    Returns:
        DataFrame with columns ``tf_family``, ``binding_category``,
        ``prediction``.
    """
    frames = []
    for tf_family, source_df in (("WRKY", wrky_df), ("bHLH", bhlh_df)):
        deduped = source_df.drop_duplicates(subset=["starr_full_name", "gene"])
        frames.append(
            pd.DataFrame(
                {
                    "tf_family": tf_family,
                    "binding_category": deduped["starr_binding_status"].to_numpy(),
                    "prediction": deduped["prediction_mutated"].to_numpy(),
                }
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["binding_category"].isin(_CATEGORY_ORDER)]
    return cast(pd.DataFrame, combined.reset_index(drop=True))


def _rank_effect_size(
    u_statistic: float, n_binding: int, n_non_binding: int
) -> tuple:
    """Convert a Mann-Whitney U into interpretable effect sizes.

    Args:
        u_statistic: U for the binding group (first sample in ``mannwhitneyu``).
        n_binding: Number of binding observations.
        n_non_binding: Number of non_binding observations.

    Returns:
        Tuple ``(prob_binding_gt_non_binding, rank_biserial_r)``. The first is
        the common-language effect size P(binding > non_binding) (0.5 = no
        effect); the second is rank-biserial correlation in [-1, 1]. Both NaN
        if either group is empty.
    """
    if n_binding == 0 or n_non_binding == 0:
        return float("nan"), float("nan")
    auc = u_statistic / (n_binding * n_non_binding)
    return auc, 2.0 * auc - 1.0


def compute_binding_significance(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Run a Mann-Whitney U test of binding vs non-binding per TF family.

    Args:
        comparison_df: DataFrame with columns ``tf_family``,
            ``binding_category``, ``prediction``.

    Returns:
        DataFrame with one row per TF family and columns ``tf_family``,
        ``n_binding``, ``n_non_binding``, ``median_binding``,
        ``median_non_binding``, ``u_statistic``, ``p_value``,
        ``prob_binding_gt_non_binding`` (common-language effect size, 0.5 = no
        effect) and ``rank_biserial_r`` ([-1, 1]). The p-value is NaN for any TF
        family lacking both categories.
    """
    rows: List[dict] = []
    for tf_family in sorted(comparison_df["tf_family"].unique()):
        tf_data = comparison_df[comparison_df["tf_family"] == tf_family]
        binding = tf_data[tf_data["binding_category"] == "binding"]["prediction"]
        non_binding = tf_data[tf_data["binding_category"] == "non_binding"]["prediction"]
        if len(binding) > 0 and len(non_binding) > 0:
            u_statistic, p_value = mannwhitneyu(
                binding, non_binding, alternative="two-sided"
            )
        else:
            u_statistic, p_value = float("nan"), float("nan")
        prob_superior, rank_biserial_r = _rank_effect_size(
            u_statistic, len(binding), len(non_binding)
        )
        rows.append(
            {
                "tf_family": tf_family,
                "n_binding": len(binding),
                "n_non_binding": len(non_binding),
                "median_binding": binding.median(),
                "median_non_binding": non_binding.median(),
                "u_statistic": u_statistic,
                "p_value": p_value,
                "prob_binding_gt_non_binding": prob_superior,
                "rank_biserial_r": rank_biserial_r,
            }
        )
    return pd.DataFrame(rows)


def _annotate_significance(
    ax: plt.Axes,
    predictions: pd.Series,
    p_value: float,
    rank_biserial_r: float = float("nan"),
) -> None:
    """Draw a significance bar above the two boxes annotated with p and effect size.

    Args:
        ax: Axes holding a two-box (binding vs non_binding) boxplot.
        predictions: All prediction values in the panel, used to place the bar.
        p_value: P-value to display; skipped when NaN.
        rank_biserial_r: Rank-biserial effect size; appended as ``r=…`` when not
            NaN, so significance is always read alongside magnitude.
    """
    if pd.isna(p_value):
        return
    top = predictions.max()
    bottom = predictions.min()
    span = top - bottom if top > bottom else 1.0
    bar_y = top + 0.05 * span
    tick = 0.02 * span
    label = _format_p_value(p_value)
    if not pd.isna(rank_biserial_r):
        label = f"{label}, r={rank_biserial_r:.2f}"
    ax.plot([0, 0, 1, 1], [bar_y, bar_y + tick, bar_y + tick, bar_y], color="black", linewidth=1.2)
    ax.text(0.5, bar_y + tick, label, ha="center", va="bottom", fontsize=10)
    ax.set_ylim(bottom - 0.05 * span, bar_y + 5 * tick)


def plot_binding_comparison(
    comparison_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    background_name: str,
    output_dir: str,
    fmt: str = "png",
) -> None:
    """Boxplot of deepCRE predictions split by binding category, one panel per TF.

    Args:
        comparison_df: DataFrame with columns ``tf_family``,
            ``binding_category``, ``prediction``.
        stats_df: Output of :func:`compute_binding_significance`, providing the
            per-TF p-value for the significance bar.
        background_name: Label for the title/filename, e.g. ``vector_background``.
        output_dir: Directory to save the figure into.
        fmt: Output format (e.g. ``png``, ``pdf``, ``svg``).
    """
    sns.set_theme(style="whitegrid")
    tf_families = sorted(comparison_df["tf_family"].unique())
    fig, axes_array = plt.subplots(
        1, len(tf_families), figsize=(5 * len(tf_families), 5), squeeze=False
    )
    axes_list = axes_array[0].tolist()

    stats_by_tf = stats_df.set_index("tf_family")
    for ax, tf_family in zip(axes_list, tf_families):
        panel = cast(pd.DataFrame, comparison_df[comparison_df["tf_family"] == tf_family])
        sns.boxplot(
            data=panel,
            x="binding_category",
            y="prediction",
            hue="binding_category",
            order=_CATEGORY_ORDER,
            palette=_BINDING_PALETTE,
            legend=False,
            ax=ax,
        )
        row = stats_by_tf.loc[tf_family]
        _annotate_significance(
            ax, panel["prediction"], row["p_value"], row["rank_biserial_r"]
        )
        ax.set_xlabel("Binding status")
        ax.set_ylabel("deepCRE prediction")
        ax.set_title(tf_family)

    pretty_background = background_name.replace("_", " ")
    fig.suptitle(
        f"deepCRE prediction: binding vs non-binding ({pretty_background})", y=1.02
    )
    save_figure(fig, f"{background_name}_boxplot", output_dir, fmt)


def _classify_vector_status(id_str: str) -> str:
    """Classify a vector sequence ID into reference / binding / non_binding.

    Unlike :func:`_parse_binding_category`, references are kept as their own
    category so they can be used as the per-locus baseline rather than folded
    into ``binding``.

    Args:
        id_str: Sequence identifier, e.g. ``WRKY_1:100-200_reference_binding_7``.

    Returns:
        ``reference``, ``binding`` or ``non_binding``.
    """
    if "reference" in id_str:
        return "reference"
    return _parse_binding_category(id_str)


def build_vector_delta_df(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-variant deltas vs the per-locus reference (vector background).

    Barcodes are averaged per sequence. Each locus (the ``TF_chrom:start-end``
    prefix shared by a reference and its variants) has exactly one reference, so
    its prediction is taken directly. Each binding / non_binding variant's delta
    is ``prediction - reference_prediction``. References themselves are excluded
    -- they are the baseline, not observations.

    Args:
        predictions_df: DataFrame with columns ``id``, ``barcode``,
            ``prediction`` (as returned by :func:`predict_sequences`).

    Returns:
        DataFrame with columns ``tf_family``, ``binding_category``, ``delta``.
    """
    averaged = average_barcode_predictions(predictions_df)
    averaged["tf_family"] = averaged["id"].str.split("_").str[0]
    averaged["locus"] = averaged["id"].str.split("_").str[:2].str.join("_")
    averaged["status"] = averaged["id"].apply(_classify_vector_status)

    reference_prediction = averaged[averaged["status"] == "reference"][
        ["locus", "prediction"]
    ].rename(columns={"prediction": "reference_prediction"})
    variants = averaged[averaged["status"].isin(_CATEGORY_ORDER)].merge(
        reference_prediction, on="locus", how="inner"
    )
    variants["delta"] = variants["prediction"] - variants["reference_prediction"]
    variants = variants.rename(columns={"status": "binding_category"})
    return cast(
        pd.DataFrame,
        variants[["tf_family", "binding_category", "delta"]].reset_index(drop=True),
    )


def build_genomic_delta_df(
    wrky_df: pd.DataFrame, bhlh_df: pd.DataFrame
) -> pd.DataFrame:
    """Build per-variant deltas vs the reference window (genomic background).

    Uses the precomputed ``delta_prediction`` (``prediction_mutated -
    deepcre_ref_fitness``). Reference rows (``starr_reference == True``) are
    excluded, and Light/Dark duplicates are collapsed to one row per
    ``(starr_full_name, gene)`` (the delta is condition-independent).

    Args:
        wrky_df: Analysis-ready WRKY DataFrame from
            :func:`prepare_wrky_enrichment_df`.
        bhlh_df: Analysis-ready bHLH DataFrame from
            :func:`prepare_bhlh_enrichment_df`.

    Returns:
        DataFrame with columns ``tf_family``, ``binding_category``, ``delta``.
    """
    frames = []
    for tf_family, source_df in (("WRKY", wrky_df), ("bHLH", bhlh_df)):
        variants = cast(
            pd.DataFrame, source_df[~source_df["starr_reference"].astype(bool)]
        )
        variants = variants.drop_duplicates(subset=["starr_full_name", "gene"])
        frames.append(
            pd.DataFrame(
                {
                    "tf_family": tf_family,
                    "binding_category": variants["starr_binding_status"].to_numpy(),
                    "delta": variants["delta_prediction"].to_numpy(),
                }
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["binding_category"].isin(_CATEGORY_ORDER)]
    return cast(pd.DataFrame, combined.reset_index(drop=True))


def _wilcoxon_vs_zero(values: pd.Series) -> float:
    """Wilcoxon signed-rank p-value testing whether deltas differ from zero.

    Zero deltas are dropped (they carry no signed rank). Returns NaN when too
    few non-zero values remain for the test to run.

    Args:
        values: Delta values for one group.

    Returns:
        Two-sided p-value, or NaN if the test cannot be computed.
    """
    nonzero = values[values != 0]
    if len(nonzero) < 1:
        return float("nan")
    try:
        _, p_value = wilcoxon(nonzero)
    except ValueError:
        return float("nan")
    return float(p_value)


def compute_delta_significance(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Significance tests for the delta analysis, per TF family.

    Runs a Mann-Whitney U test of binding vs non_binding deltas, plus a Wilcoxon
    signed-rank test of each group's deltas against zero.

    Args:
        delta_df: DataFrame with columns ``tf_family``, ``binding_category``,
            ``delta``.

    Returns:
        DataFrame with one row per TF family and columns ``tf_family``,
        ``n_binding``, ``n_non_binding``, ``median_binding``,
        ``median_non_binding``, ``u_statistic``, ``p_value``,
        ``prob_binding_gt_non_binding`` (common-language effect size, 0.5 = no
        effect), ``rank_biserial_r`` ([-1, 1]), ``p_binding_vs_zero`` and
        ``p_non_binding_vs_zero``.
    """
    rows: List[dict] = []
    for tf_family in sorted(delta_df["tf_family"].unique()):
        tf_data = delta_df[delta_df["tf_family"] == tf_family]
        binding = tf_data[tf_data["binding_category"] == "binding"]["delta"]
        non_binding = tf_data[tf_data["binding_category"] == "non_binding"]["delta"]
        if len(binding) > 0 and len(non_binding) > 0:
            u_statistic, p_value = mannwhitneyu(
                binding, non_binding, alternative="two-sided"
            )
        else:
            u_statistic, p_value = float("nan"), float("nan")
        prob_superior, rank_biserial_r = _rank_effect_size(
            u_statistic, len(binding), len(non_binding)
        )
        rows.append(
            {
                "tf_family": tf_family,
                "n_binding": len(binding),
                "n_non_binding": len(non_binding),
                "median_binding": binding.median(),
                "median_non_binding": non_binding.median(),
                "u_statistic": u_statistic,
                "p_value": p_value,
                "prob_binding_gt_non_binding": prob_superior,
                "rank_biserial_r": rank_biserial_r,
                "p_binding_vs_zero": _wilcoxon_vs_zero(binding),
                "p_non_binding_vs_zero": _wilcoxon_vs_zero(non_binding),
            }
        )
    return pd.DataFrame(rows)


def plot_delta_comparison(
    delta_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    background_name: str,
    output_dir: str,
    fmt: str = "png",
) -> None:
    """Boxplot of per-variant deltas split by binding category, one panel per TF.

    A dashed ``y = 0`` line marks the reference baseline. The binding vs
    non_binding p-value is shown as a significance bar; each group's Wilcoxon
    vs-zero p-value is annotated beneath its box.

    Args:
        delta_df: DataFrame with columns ``tf_family``, ``binding_category``,
            ``delta``.
        stats_df: Output of :func:`compute_delta_significance`.
        background_name: Label for the title/filename, e.g. ``vector_background``.
        output_dir: Directory to save the figure into.
        fmt: Output format (e.g. ``png``, ``pdf``, ``svg``).
    """
    sns.set_theme(style="whitegrid")
    tf_families = sorted(delta_df["tf_family"].unique())
    fig, axes_array = plt.subplots(
        1, len(tf_families), figsize=(5 * len(tf_families), 5), squeeze=False
    )
    axes_list = axes_array[0].tolist()
    stats_by_tf = stats_df.set_index("tf_family")

    for ax, tf_family in zip(axes_list, tf_families):
        panel = cast(pd.DataFrame, delta_df[delta_df["tf_family"] == tf_family])
        sns.boxplot(
            data=panel,
            x="binding_category",
            y="delta",
            hue="binding_category",
            order=_CATEGORY_ORDER,
            palette=_BINDING_PALETTE,
            legend=False,
            ax=ax,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        row = stats_by_tf.loc[tf_family]
        _annotate_significance(
            ax, panel["delta"], row["p_value"], row["rank_biserial_r"]
        )
        vs_zero = [row["p_binding_vs_zero"], row["p_non_binding_vs_zero"]]
        for category_index, p_vs_zero in enumerate(vs_zero):
            if not pd.isna(p_vs_zero):
                ax.text(
                    category_index,
                    ax.get_ylim()[0],
                    f"vs 0: {_format_p_value(p_vs_zero)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xlabel("Binding status")
        ax.set_ylabel("deepCRE prediction delta (variant - reference)")
        ax.set_title(tf_family)

    pretty_background = background_name.replace("_", " ")
    fig.suptitle(
        f"deepCRE prediction delta vs reference: binding vs non-binding "
        f"({pretty_background})",
        y=1.02,
    )
    save_figure(fig, f"{background_name}_delta_boxplot", output_dir, fmt)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    predictions = predict_sequences()
    wrky_df = prepare_wrky_enrichment_df()
    bhlh_df = prepare_bhlh_enrichment_df()

    # Absolute analysis: predictions, references kept in the binding group.
    for comparison, name in (
        (build_vector_comparison_df(predictions), "vector_background"),
        (build_genomic_comparison_df(wrky_df, bhlh_df), "genomic_background"),
    ):
        stats = compute_binding_significance(comparison)
        stats.to_csv(os.path.join(OUTPUT_DIR, f"{name}_stats.csv"), index=False)
        plot_binding_comparison(comparison, stats, name, OUTPUT_DIR, FMT)
        print(f"{name} (absolute):")
        print(stats.to_string(index=False))

    # Delta analysis: variant - reference, references excluded as the baseline.
    for delta_data, name in (
        (build_vector_delta_df(predictions), "vector_background"),
        (build_genomic_delta_df(wrky_df, bhlh_df), "genomic_background"),
    ):
        delta_stats = compute_delta_significance(delta_data)
        delta_stats.to_csv(
            os.path.join(OUTPUT_DIR, f"{name}_delta_stats.csv"), index=False
        )
        plot_delta_comparison(delta_data, delta_stats, name, OUTPUT_DIR, FMT)
        print(f"{name} (delta):")
        print(delta_stats.to_string(index=False))
