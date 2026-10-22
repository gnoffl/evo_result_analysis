"""Which deepCRE model group agrees best with the STARR-seq measurement?

Repeats the construct-background analysis of :mod:`construct_by_species` for the
four model groups of :mod:`_model_groups` (MSR, Arabidopsis SSR, tomato SSR,
tobacco SSR). Within a group every model scores every construct insert and the
per-insert *median* over the group's models is used as that group's deepCRE
value, which is then correlated against the measured enrichment.

Outputs, all under ``model_groups/``:

* ``prediction_vs_enrichment_by_group.png`` - the four-panel figure, one panel
  per group, each the group-median counterpart of
  ``construct/<species>/prediction_vs_enrichment_overall.png``.
* ``group_summary.csv`` - the Spearman correlation between each group's median
  prediction and the measured enrichment, best group first. The fitted line,
  its slope and its Pearson r are shown in the panel itself.
* ``per_model_correlations.csv`` / ``.png`` - the per-model Spearman values the
  medians are built from, which show the within-group spread.

The group medians are not a like-for-like comparison: the groups differ in
training recipe, training species *and* size (12/5/12/5 models), and a median
over more models is less noisy for reasons unrelated to what the models learned.
The per-model outputs are there to judge how much of a group difference survives
that.

Predictions are cached per model under ``cache/``; the two SSR groups reuse the
cache files :mod:`construct_by_species` already wrote. Like the modules it
reuses, this script resolves its inputs through repo-relative paths and must
therefore be run from the repository root.
"""
import os
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from analysis.utils.io import print_section_header, print_status, print_subsection
from workflows.overlap_analysis.correct_construct.analyze_in_true_background import (
    STARRSEQ_PATH,
    load_starrseq_data,
    plot_overall_correlation,
)
from workflows.overlap_analysis.model_species_comparison import _summary
from workflows.overlap_analysis.model_species_comparison._model_groups import (
    MODEL_GROUPS,
    ModelGroup,
    list_group_models,
)
from workflows.overlap_analysis.model_species_comparison._models import SpeciesModel
from workflows.overlap_analysis.model_species_comparison.construct_by_species import (
    ROW_KEY_COLUMNS,
    build_long_predictions,
    correlation_data_for_model,
)

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "model_groups")
PANEL_PLOT_PATH = os.path.join(OUTPUT_DIR, "prediction_vs_enrichment_by_group.png")
GROUP_SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, "group_summary.csv")
PER_MODEL_CSV_PATH = os.path.join(OUTPUT_DIR, "per_model_correlations.csv")
PER_MODEL_PLOT_PATH = os.path.join(OUTPUT_DIR, "per_model_correlations.png")


def build_group_median_df(
    model_dataframes: List[Tuple[SpeciesModel, pd.DataFrame]],
) -> pd.DataFrame:
    """Reduce one group's models to a single dataframe of median predictions.

    Args:
        model_dataframes: Row-aligned ``(model, correlation dataframe)`` pairs of
            exactly one group, as returned by
            :func:`_summary.align_by_row_key`.

    Returns:
        The first model's dataframe with ``prediction`` replaced by the
        per-insert median prediction across the group's models.

    Raises:
        ValueError: If the pairs are empty or span more than one group.
    """
    if not model_dataframes:
        raise ValueError("No models given for the group median")
    group_names = {model.species for model, _ in model_dataframes}
    if len(group_names) > 1:
        raise ValueError(f"Expected one model group, got {sorted(group_names)}")

    median_df = model_dataframes[0][1].copy()
    stacked = pd.concat(
        [correlation_df["prediction"] for _, correlation_df in model_dataframes], axis=1
    )
    median_df["prediction"] = stacked.median(axis=1)
    return median_df


def summarize_group_medians(
    group_dataframes: List[Tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Correlate each group's median prediction with the measured enrichment.

    Args:
        group_dataframes: One ``(group name, median dataframe)`` pair per group.

    Returns:
        One row per group with columns ``group``,
        ``count_points``, ``spearman_r`` and ``spearman_p``, sorted by
        descending ``spearman_r`` so the best-agreeing group comes first.
    """
    summary_rows = []
    for group_name, median_df in group_dataframes:
        paired_df = median_df.dropna(subset=["prediction", "enrichment"])
        correlation, p_value = spearmanr(
            paired_df["prediction"], paired_df["enrichment"]
        )
        summary_rows.append(
            {
                "group": group_name,
                "count_points": len(paired_df),
                "spearman_r": float(correlation),
                "spearman_p": float(p_value),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    return summary_df.sort_values("spearman_r", ascending=False).reset_index(drop=True)


def plot_group_panels(
    labelled_dataframes: List[Tuple[str, pd.DataFrame]], output_path: str
) -> None:
    """Draw one prediction-vs-enrichment panel per group into a single figure.

    Args:
        labelled_dataframes: One ``(panel title, median dataframe)`` pair per
            group.
        output_path: Image path to write; parent directories are created.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid")
    column_count = 2
    row_count = -(-len(labelled_dataframes) // column_count)
    figure, axes_array = plt.subplots(
        row_count,
        column_count,
        figsize=(6 * column_count, 5 * row_count),
        squeeze=False,
    )
    axes_list = [ax for row in axes_array for ax in row]

    for ax, (panel_title, median_df) in zip(axes_list, labelled_dataframes):
        plot_overall_correlation(median_df, ax=ax)
        ax.set_title(panel_title)
    for unused_ax in axes_list[len(labelled_dataframes) :]:
        unused_ax.set_visible(False)

    figure.suptitle(
        "Group-median deepCRE prediction vs. STARR-seq enrichment", y=1.0
    )
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(figure)


def plot_per_model_spread(summary_df: pd.DataFrame, output_path: str) -> None:
    """Plot the per-model Spearman values of every group as box plus points.

    Args:
        summary_df: Output of :func:`_summary.summarize_per_model`, whose
            ``species`` column holds the group name.
        output_path: Image path to write; parent directories are created.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(figsize=(7, 5))
    group_order = sorted(summary_df["species"].unique())
    sns.boxplot(
        data=summary_df, x="species", y="spearman_r", order=group_order,
        showfliers=False, width=0.5, ax=axes,
    )
    sns.stripplot(
        data=summary_df, x="species", y="spearman_r", order=group_order,
        color="#222222", size=6, jitter=0.15, ax=axes,
    )
    axes.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
    axes.set_xlabel("deepCRE model group")
    axes.set_ylabel("Spearman r (prediction vs. STARR-seq enrichment)")
    axes.set_title("Per-model agreement with STARR-seq, by model group")
    figure.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(figure)


def correlation_data_for_group(
    group: ModelGroup, starrseq_df: pd.DataFrame
) -> List[Tuple[SpeciesModel, pd.DataFrame]]:
    """Score the constructs with every model of one group.

    Args:
        group: The model group to score with.
        starrseq_df: Output of :func:`load_starrseq_data`.

    Returns:
        Row-aligned ``(model, correlation dataframe)`` pairs for the group.
    """
    models = list_group_models(group)
    print_subsection(f"{group.name}: {len(models)} models")
    model_dataframes: List[Tuple[SpeciesModel, pd.DataFrame]] = []
    for _, model in enumerate(models, start=1):
        model_dataframes.append((model, correlation_data_for_model(model, starrseq_df)))
    return _summary.align_by_row_key(model_dataframes, ROW_KEY_COLUMNS)


def main() -> None:
    """Compare the four model groups by their median construct predictions."""
    starrseq_df = load_starrseq_data(STARRSEQ_PATH)

    all_model_dataframes: List[Tuple[SpeciesModel, pd.DataFrame]] = []
    group_dataframes: List[Tuple[str, pd.DataFrame]] = []
    for group in MODEL_GROUPS:
        aligned = correlation_data_for_group(group, starrseq_df)
        all_model_dataframes.extend(aligned)
        group_dataframes.append((group.name, build_group_median_df(aligned)))

    print_subsection("Summaries")
    group_summary_df = summarize_group_medians(group_dataframes)
    model_counts = {model.species: 0 for model, _ in all_model_dataframes}
    for model, _ in all_model_dataframes:
        model_counts[model.species] += 1
    group_summary_df.insert(
        1, "count_models", group_summary_df["group"].map(model_counts)
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    group_summary_df.to_csv(GROUP_SUMMARY_CSV_PATH, index=False)

    per_model_summary_df = _summary.summarize_per_model(
        build_long_predictions(all_model_dataframes)
    )
    _summary.write_summary_csv(per_model_summary_df, PER_MODEL_CSV_PATH)

    panel_labels = [
        (f"{group_name} (median of {model_counts[group_name]} models)", median_df)
        for group_name, median_df in group_dataframes
    ]
    plot_group_panels(panel_labels, PANEL_PLOT_PATH)
    plot_per_model_spread(per_model_summary_df, PER_MODEL_PLOT_PATH)

    print(group_summary_df.to_string(index=False))
    print(per_model_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
