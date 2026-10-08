"""STARR-seq x deepCRE correlation, repeated for every model of both species.

Tests whether the training species of the deepCRE model influences how well its
predictions agree with the STARR-seq measurement. Reuses the pooled WRKY + bHLH
pipeline unchanged (see ``starrseq_deepcre_correlation_combined``) and only
swaps the scoring model, so any difference in the resulting correlation comes
from the model rather than from the data preparation.

Two kinds of output are produced:

- ``per_model_correlations.csv`` / ``.png`` - the primary result: one Spearman
  correlation per cross-validation model, so the within-species cross-validation
  spread serves as the error bar on the between-species comparison.
- ``<species>/`` - the full standard plot suite, run once per species on that
  species' ensemble-mean prediction, for viewing. See
  :mod:`_summary` for why the two ensemble correlations are not directly
  comparable to each other.

Every model's full enrichment dataframe is cached under ``cache/`` on first
computation, so re-running only redoes the models whose cache is missing.
"""
import os
import time
from typing import Dict, List, Tuple

import pandas as pd

from analysis.utils.io import print_section_header, print_status, print_subsection
from workflows.overlap_analysis import _common
from workflows.overlap_analysis.model_species_comparison import _summary
from workflows.overlap_analysis.model_species_comparison._models import (
    SpeciesModel,
    list_species_models,
)
from workflows.overlap_analysis.starrseq_deepcre_correlation_bHLH import (
    prepare_bhlh_enrichment_df,
)
from workflows.overlap_analysis.starrseq_deepcre_correlation_WRKY import (
    prepare_wrky_enrichment_df,
)

BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, "cache")
SUMMARY_CSV_PATH = os.path.join(BASE_DIR, "per_model_correlations.csv")
SUMMARY_PLOT_PATH = os.path.join(BASE_DIR, "per_model_correlations.png")
SPECIES_COMPARISON_CSV_PATH = os.path.join(BASE_DIR, "species_comparison.csv")

# Columns that jointly identify one scored STARR-seq fragment. Used to bring
# every model's dataframe into the same row order before averaging, so the
# combination never depends on directory listing order.
ROW_KEY_COLUMNS = ["starr_full_name", "gene", "condition"]


def cache_path_for(model: SpeciesModel) -> str:
    """Return the cache file path for one model's enrichment dataframe.

    Args:
        model: The cross-validation model.

    Returns:
        Absolute path to the per-model cache CSV.
    """
    return os.path.join(
        CACHE_DIR, f"enrichment_{model.species}_{model.held_out_chromosome}.csv"
    )


def enrichment_df_for_model(model: SpeciesModel) -> pd.DataFrame:
    """Load or compute the pooled WRKY + bHLH enrichment dataframe for one model.

    Reads the cache when present; otherwise runs both preparation steps with
    ``model`` as the scoring model and writes the result to the cache.

    Args:
        model: The cross-validation model to score with.

    Returns:
        The pooled enrichment dataframe, as
        :func:`_common.run_correlation_analysis` expects it.
    """
    path = cache_path_for(model)
    if os.path.exists(path):
        return pd.read_csv(path)

    wrky_df = prepare_wrky_enrichment_df(model_path=model.path)
    bhlh_df = prepare_bhlh_enrichment_df(model_path=model.path)
    pooled_df = pd.concat([wrky_df, bhlh_df], ignore_index=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    pooled_df.to_csv(path, index=False)
    return pooled_df


def build_long_predictions(
    model_dataframes: List[Tuple[SpeciesModel, pd.DataFrame]],
) -> pd.DataFrame:
    """Stack every model's predictions into one long table for summarising.

    Args:
        model_dataframes: Aligned ``(model, enrichment dataframe)`` pairs.

    Returns:
        One row per (model, fragment) with columns ``species``,
        ``held_out_chromosome``, ``prediction`` and ``enrichment``.
    """
    long_frames = [
        pd.DataFrame(
            {
                "species": model.species,
                "held_out_chromosome": model.held_out_chromosome,
                "prediction": enrichment_df["prediction_mutated"],
                "enrichment": enrichment_df["enrichment"],
            }
        )
        for model, enrichment_df in model_dataframes
    ]
    return pd.concat(long_frames, ignore_index=True)


def build_species_ensemble_df(
    model_dataframes: List[Tuple[SpeciesModel, pd.DataFrame]],
) -> pd.DataFrame:
    """Average one species' models into a single enrichment dataframe.

    Takes the first model's dataframe as the template - all metadata columns are
    identical across models of the same species - and replaces the two
    model-dependent columns with their mean across the species' models.
    ``delta_prediction`` is recomputed from the averaged columns so the
    dataframe stays internally consistent.

    Args:
        model_dataframes: Aligned pairs for exactly one species.

    Returns:
        One enrichment dataframe carrying the species' ensemble-mean prediction.

    Raises:
        ValueError: If the pairs are empty or span more than one species.
    """
    if not model_dataframes:
        raise ValueError("No models given for the species ensemble")
    species_names = {model.species for model, _ in model_dataframes}
    if len(species_names) > 1:
        raise ValueError(f"Expected one species, got {sorted(species_names)}")

    ensemble_df = model_dataframes[0][1].copy()
    for column in ["prediction_mutated", "deepcre_ref_fitness"]:
        if column not in ensemble_df.columns:
            continue
        stacked = pd.concat(
            [enrichment_df[column] for _, enrichment_df in model_dataframes], axis=1
        )
        ensemble_df[column] = stacked.mean(axis=1)
    if {"prediction_mutated", "deepcre_ref_fitness"} <= set(ensemble_df.columns):
        ensemble_df["delta_prediction"] = (
            ensemble_df["prediction_mutated"] - ensemble_df["deepcre_ref_fitness"]
        )
    return ensemble_df


def group_by_species(
    model_dataframes: List[Tuple[SpeciesModel, pd.DataFrame]],
) -> Dict[str, List[Tuple[SpeciesModel, pd.DataFrame]]]:
    """Group aligned model pairs by training species, preserving order.

    Args:
        model_dataframes: Aligned ``(model, enrichment dataframe)`` pairs.

    Returns:
        Mapping of species name to its pairs.
    """
    grouped: Dict[str, List[Tuple[SpeciesModel, pd.DataFrame]]] = {}
    for model, enrichment_df in model_dataframes:
        grouped.setdefault(model.species, []).append((model, enrichment_df))
    return grouped


def main() -> None:
    """Run the correlation analysis for every model and summarise by species."""
    _common.configure_matplotlib()
    models = list_species_models()
    print_section_header("STARR-SEQ CORRELATION BY MODEL TRAINING SPECIES")
    print_status(f"{len(models)} models: " + ", ".join(
        f"{m.species}/{m.held_out_chromosome}" for m in models
    ))

    model_dataframes: List[Tuple[SpeciesModel, pd.DataFrame]] = []
    for _, model in enumerate(models, start=1):
        model_dataframes.append((model, enrichment_df_for_model(model)))

    print_subsection("Per-model summary")
    aligned = _summary.align_by_row_key(model_dataframes, ROW_KEY_COLUMNS)
    summary_df = _summary.summarize_per_model(build_long_predictions(aligned))
    _summary.write_summary_csv(summary_df, SUMMARY_CSV_PATH)
    _summary.plot_per_model_correlations(
        summary_df,
        SUMMARY_PLOT_PATH,
        "Per-model agreement with STARR-seq (WRKY + bHLH pooled)",
    )
    comparison_df = _summary.compare_species(summary_df)
    comparison_df.to_csv(SPECIES_COMPARISON_CSV_PATH, index=False)
    print(summary_df.to_string(index=False))
    print(comparison_df.to_string(index=False))

    for species, species_pairs in group_by_species(aligned).items():
        _common.set_output_root(os.path.join(BASE_DIR, species))
        _common.run_correlation_analysis(build_species_ensemble_df(species_pairs))


if __name__ == "__main__":
    main()
