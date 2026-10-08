"""Shared helpers for the two species-comparison passes.

Both passes end with the same question: for each cross-validation model, how
well do its predictions agree with the STARR-seq measurement, and does that
agreement differ between the two training species? This module holds the row
alignment needed before averaging models, and turns a long prediction table
into that summary, writes it, and plots it.

The per-model correlations - not the species ensembles - are the fair
comparison. Averaging the predictions of ``K`` cross-validation models suppresses
model-specific noise by a factor of ``K``, so the 12-model tomato ensemble is
denoised more than the 5-model Arabidopsis ensemble, which would inflate the
tomato correlation for reasons unrelated to training species.
"""
import os
from typing import Any, List, Sequence, Tuple
from workflows.overlap_analysis.model_species_comparison._models import SpeciesModel

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr

CAVEAT_LINES = [
    "Per-model Spearman correlation between deepCRE prediction and STARR-seq",
    "enrichment.",
    "One row per cross-validation model. These per-model values are the fair",
    "species comparison: the per-species ensemble figures average 12 tomato models",
    "against only 5 Arabidopsis models, and averaging K models suppresses",
    "model-specific noise by a factor of K, which inflates the tomato correlation",
    "independently of training species.",
]


def align_by_row_key(
    model_dataframes: Sequence[Tuple[SpeciesModel, pd.DataFrame]],
    candidate_key_columns: Sequence[str],
) -> List[Tuple[SpeciesModel, pd.DataFrame]]:
    """Sort every model's dataframe into one shared row order.

    Averaging predictions across models requires that row ``i`` describes the
    same scored sequence in every dataframe. Rather than relying on the pipeline
    producing a stable row order, each dataframe is sorted by the row key and
    the resulting key sequences are checked for equality.

    Args:
        model_dataframes: One ``(model, dataframe)`` pair per model. The model
            objects are only used for error messages and must expose ``species``
            and ``held_out_chromosome``.
        candidate_key_columns: Columns that jointly identify one scored
            sequence. Narrowed to those actually present, since e.g.
            ``condition`` only exists for the light+dark STARR-seq file.

    Returns:
        The same pairs with each dataframe sorted by the row key and reindexed.

    Raises:
        ValueError: If none of the candidate key columns is present, if the row
            key is not unique within a dataframe, or if two models do not
            describe the same set of sequences.
    """
    aligned: List[Tuple[Any, pd.DataFrame]] = []
    reference_keys = None
    reference_model = None
    for model, dataframe in model_dataframes:
        key_columns = [c for c in candidate_key_columns if c in dataframe.columns]
        if not key_columns:
            raise ValueError(
                f"None of {list(candidate_key_columns)} present in the dataframe for "
                f"{model.species}/{model.held_out_chromosome}"
            )
        sorted_df = dataframe.sort_values(key_columns).reset_index(drop=True)
        keys = sorted_df[key_columns].astype(str).agg("|".join, axis=1)
        if keys.duplicated().any():
            raise ValueError(
                f"Row key {key_columns} is not unique for model "
                f"{model.species}/{model.held_out_chromosome}: "
                f"{int(keys.duplicated().sum())} duplicated rows"
            )
        if reference_keys is None:
            reference_keys, reference_model = keys, model
        elif not keys.equals(reference_keys):
            raise ValueError(
                f"Model {model.species}/{model.held_out_chromosome} scored a different "
                f"set of sequences than {reference_model.species}/"
                f"{reference_model.held_out_chromosome}"
            )
        aligned.append((model, sorted_df))
    return aligned


def summarize_per_model(long_predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute the Spearman correlation of each model against the measurement.

    Args:
        long_predictions: One row per (model, scored sequence), with columns
            ``species``, ``held_out_chromosome``, ``prediction`` and
            ``enrichment``.

    Returns:
        One row per model with columns ``species``, ``held_out_chromosome``,
        ``count_points``, ``spearman_r`` and ``spearman_p``, sorted by species
        then accession.

    Raises:
        ValueError: If a required column is missing.
    """
    required_columns = {"species", "held_out_chromosome", "prediction", "enrichment"}
    missing_columns = required_columns - set(long_predictions.columns)
    if missing_columns:
        raise ValueError(
            f"long_predictions is missing columns: {sorted(missing_columns)}"
        )

    summary_rows: List[dict] = []
    grouped = long_predictions.groupby(["species", "held_out_chromosome"], sort=True)
    for (species, held_out_chromosome), model_df in grouped:
        paired_df = model_df.dropna(subset=["prediction", "enrichment"])
        correlation, p_value = spearmanr(
            paired_df["prediction"], paired_df["enrichment"]
        )
        summary_rows.append(
            {
                "species": species,
                "held_out_chromosome": held_out_chromosome,
                "count_points": len(paired_df),
                "spearman_r": float(correlation),
                "spearman_p": float(p_value),
            }
        )
    return pd.DataFrame(summary_rows)


def compare_species(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compare the two species' per-model correlation distributions.

    Uses a two-sided Mann-Whitney U test on the per-model Spearman values. With
    5 against 12 models this has little power, so it is reported as a descriptive
    summary alongside the medians and ranges rather than as a decisive test.

    Args:
        summary_df: Output of :func:`summarize_per_model`.

    Returns:
        A single-row dataframe with each species' model count, median and
        min/max correlation, plus the Mann-Whitney U statistic and p-value.
        ``mannwhitney_u`` and ``mannwhitney_p`` are ``NaN`` when either species
        has fewer than two models.
    """
    species_names = sorted(summary_df["species"].unique())
    row = {}
    correlation_groups = []
    for species in species_names:
        species_correlations = summary_df.loc[
            summary_df["species"] == species, "spearman_r"
        ]
        correlation_groups.append(species_correlations)
        row[f"count_models_{species}"] = len(species_correlations)
        row[f"median_spearman_r_{species}"] = float(species_correlations.median())
        row[f"min_spearman_r_{species}"] = float(species_correlations.min())
        row[f"max_spearman_r_{species}"] = float(species_correlations.max())

    has_two_comparable_groups = len(correlation_groups) == 2 and all(
        len(group) >= 2 for group in correlation_groups
    )
    if has_two_comparable_groups:
        statistic, p_value = mannwhitneyu(
            correlation_groups[0], correlation_groups[1], alternative="two-sided"
        )
        row["mannwhitney_u"] = float(statistic)
        row["mannwhitney_p"] = float(p_value)
    else:
        row["mannwhitney_u"] = float("nan")
        row["mannwhitney_p"] = float("nan")
    return pd.DataFrame([row])


def write_summary_csv(summary_df: pd.DataFrame, output_path: str) -> None:
    """Write the per-model summary, prefixed with the interpretation caveat.

    The caveat is written as ``#``-prefixed comment lines so that
    ``pd.read_csv(path, comment="#")`` reads the table back unchanged.

    Args:
        summary_df: Output of :func:`summarize_per_model`.
        output_path: CSV path to write; parent directories are created.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as output_file:
        for line in CAVEAT_LINES:
            output_file.write(f"# {line}\n")
        summary_df.to_csv(output_file, index=False)


def plot_per_model_correlations(
    summary_df: pd.DataFrame, output_path: str, title: str
) -> None:
    """Plot the per-model correlation distribution of each species.

    Draws one box per species with the individual models overlaid as points, so
    that both the within-species cross-validation spread and the between-species
    difference are visible.

    Args:
        summary_df: Output of :func:`summarize_per_model`.
        output_path: Image path to write; parent directories are created.
        title: Figure title.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(figsize=(6, 5))
    species_order = sorted(summary_df["species"].unique())
    sns.boxplot(
        data=summary_df, x="species", y="spearman_r", order=species_order,
        showfliers=False, width=0.5, ax=axes,
    )
    sns.stripplot(
        data=summary_df, x="species", y="spearman_r", order=species_order,
        color="#222222", size=7, jitter=0.15, ax=axes,
    )
    for _, model_row in summary_df.iterrows():
        axes.annotate(
            str(model_row["held_out_chromosome"]),
            (species_order.index(str(model_row["species"])), model_row["spearman_r"]),
            textcoords="offset points", xytext=(10, -3), fontsize=6, color="#555555",
        )
    axes.axhline(0.0, color="#999999", linestyle="--", linewidth=1)
    axes.set_xlabel("Training species of the deepCRE model")
    axes.set_ylabel("Spearman r (prediction vs. STARR-seq enrichment)")
    axes.set_title(title)
    figure.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(figure)
