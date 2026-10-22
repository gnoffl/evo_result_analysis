"""Construct-background analysis, repeated for every model of both species.

The sibling of :mod:`correlation_by_species` for the true-construct analysis:
every STARR-seq insert is scored inside the actual plasmid background (see
``correct_construct/analyze_in_true_background``) by each cross-validation model
of both training species, to test whether the training species influences
agreement with the measurement.

Outputs mirror the other pass: ``construct/per_model_correlations.csv`` / ``.png``
as the primary result, and ``construct/<species>/`` holding that species'
ensemble-mean plots.

Each model's raw predictions are cached under ``cache/`` on first computation,
so re-running only redoes the models whose cache is missing.

Like the module it reuses, this script resolves its inputs through repo-relative
paths and must therefore be run from the repository root.
"""
import os
import time
from typing import Any, Dict, List, Tuple

import pandas as pd

from analysis.utils.io import print_section_header, print_status, print_subsection
from workflows.overlap_analysis.correct_construct.analyze_in_true_background import (
    STARRSEQ_PATH,
    load_starrseq_data,
    plot_overall_correlation,
    plot_prediction_vs_enrichment,
    prepare_correlation_data,
    predict_sequences,
)
from workflows.overlap_analysis.model_species_comparison import _summary
from workflows.overlap_analysis.model_species_comparison._models import (
    SpeciesModel,
    list_species_models,
)

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "construct")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, "per_model_correlations.csv")
SUMMARY_PLOT_PATH = os.path.join(OUTPUT_DIR, "per_model_correlations.png")
SPECIES_COMPARISON_CSV_PATH = os.path.join(OUTPUT_DIR, "species_comparison.csv")
FMT = "png"

# One scored construct insert is identified by its STARR-seq id and the
# measurement condition; barcodes are already averaged out by
# ``prepare_correlation_data``.
ROW_KEY_COLUMNS = ["id", "condition"]


def cache_path_for(model: SpeciesModel) -> str:
    """Return the prediction cache path for one model.

    Args:
        model: The cross-validation model.

    Returns:
        Absolute path to the per-model cache CSV.
    """
    return os.path.join(
        CACHE_DIR, f"construct_{model.species}_{model.held_out_chromosome}.csv"
    )


def correlation_data_for_model(
    model: SpeciesModel, starrseq_df: pd.DataFrame
) -> pd.DataFrame:
    """Score every construct insert with one model and merge in the measurement.

    Prediction is cached per model by :func:`predict_sequences`; only the cheap
    merge is redone on a cache hit.

    Args:
        model: The cross-validation model to score with.
        starrseq_df: Output of :func:`load_starrseq_data`.

    Returns:
        Dataframe with columns ``id``, ``condition``, ``enrichment``,
        ``tf_family``, ``binding_category`` and ``prediction``.
    """
    path = cache_path_for(model)
    os.makedirs(CACHE_DIR, exist_ok=True)
    predictions_df = predict_sequences(cache_path=path, model_path=model.path)
    return prepare_correlation_data(predictions_df, starrseq_df)


def build_long_predictions(
    model_dataframes: List[Tuple[SpeciesModel, pd.DataFrame]],
) -> pd.DataFrame:
    """Stack every model's predictions into one long table for summarising.

    Args:
        model_dataframes: Aligned ``(model, correlation dataframe)`` pairs.

    Returns:
        One row per (model, insert) with columns ``species``,
        ``held_out_chromosome``, ``prediction`` and ``enrichment``.
    """
    long_frames = [
        pd.DataFrame(
            {
                "species": model.species,
                "held_out_chromosome": model.held_out_chromosome,
                "prediction": correlation_df["prediction"],
                "enrichment": correlation_df["enrichment"],
            }
        )
        for model, correlation_df in model_dataframes
    ]
    return pd.concat(long_frames, ignore_index=True)


def build_species_ensemble_df(
    model_dataframes: List[Tuple[SpeciesModel, pd.DataFrame]],
) -> pd.DataFrame:
    """Average one species' models into a single correlation dataframe.

    Args:
        model_dataframes: Aligned pairs for exactly one species.

    Returns:
        The first model's dataframe with ``prediction`` replaced by the mean
        prediction across the species' models.

    Raises:
        ValueError: If the pairs are empty or span more than one species.
    """
    if not model_dataframes:
        raise ValueError("No models given for the species ensemble")
    species_names = {model.species for model, _ in model_dataframes}
    if len(species_names) > 1:
        raise ValueError(f"Expected one species, got {sorted(species_names)}")

    ensemble_df = model_dataframes[0][1].copy()
    stacked = pd.concat(
        [correlation_df["prediction"] for _, correlation_df in model_dataframes], axis=1
    )
    ensemble_df["prediction"] = stacked.mean(axis=1)
    return ensemble_df


def group_by_species(
    model_dataframes: List[Tuple[Any, pd.DataFrame]],
) -> Dict[str, List[Tuple[Any, pd.DataFrame]]]:
    """Group aligned model pairs by training species, preserving order.

    Args:
        model_dataframes: Aligned ``(model, dataframe)`` pairs.

    Returns:
        Mapping of species name to its pairs.
    """
    grouped: Dict[str, List[Tuple[Any, pd.DataFrame]]] = {}
    for model, dataframe in model_dataframes:
        grouped.setdefault(model.species, []).append((model, dataframe))
    return grouped


def main() -> None:
    """Score the constructs with every model and summarise by species."""
    models = list_species_models()
    starrseq_df = load_starrseq_data(STARRSEQ_PATH)

    model_dataframes: List[Tuple[SpeciesModel, pd.DataFrame]] = []
    for model_index, model in enumerate(models, start=1):
        print_subsection(
            f"[{model_index}/{len(models)}] {model.species} / "
            f"{model.held_out_chromosome}"
        )
        model_dataframes.append((model, correlation_data_for_model(model, starrseq_df)))

    print_subsection("Per-model summary")
    aligned = _summary.align_by_row_key(model_dataframes, ROW_KEY_COLUMNS)
    summary_df = _summary.summarize_per_model(build_long_predictions(aligned))
    _summary.write_summary_csv(summary_df, SUMMARY_CSV_PATH)
    _summary.plot_per_model_correlations(
        summary_df,
        SUMMARY_PLOT_PATH,
        "Per-model agreement with STARR-seq (true construct background)",
    )
    comparison_df = _summary.compare_species(summary_df)
    comparison_df.to_csv(SPECIES_COMPARISON_CSV_PATH, index=False)
    print(summary_df.to_string(index=False))
    print(comparison_df.to_string(index=False))

    for species, species_pairs in group_by_species(aligned).items():
        print_subsection(f"Ensemble plots: {species} ({len(species_pairs)} models)")
        species_output_dir = os.path.join(OUTPUT_DIR, species)
        ensemble_df = build_species_ensemble_df(species_pairs)
        plot_overall_correlation(ensemble_df, species_output_dir, FMT)
        for condition in [None, "Light", "Dark"]:
            plot_prediction_vs_enrichment(
                ensemble_df, condition, species_output_dir, FMT
            )


if __name__ == "__main__":
    main()
