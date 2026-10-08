"""Unit tests for the per-species STARR-seq correlation pass.

Model inference and the two TF preparation steps are mocked, so the tests cover
only this module's own logic: per-model caching, the long prediction table, and
the ensemble averaging.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import pandas as pd

from workflows.overlap_analysis.model_species_comparison import correlation_by_species
from workflows.overlap_analysis.model_species_comparison._models import SpeciesModel

ARA_MODEL = SpeciesModel("Ara", "NC_003070.9", "/models/ara_1.h5")
ARA_MODEL_2 = SpeciesModel("Ara", "NC_003071.7", "/models/ara_2.h5")
SLYC_MODEL = SpeciesModel("Slyc", "NC_015438.3", "/models/slyc_1.h5")


def _enrichment_frame(predictions, ref_fitness=None):
    """Build a minimal enrichment frame with two scored fragments."""
    if ref_fitness is None:
        ref_fitness = [0.1, 0.2]
    return pd.DataFrame(
        {
            "starr_full_name": ["frag_a", "frag_b"],
            "gene": ["AT1G00001", "AT1G00002"],
            "condition": ["Dark", "Dark"],
            "prediction_mutated": predictions,
            "deepcre_ref_fitness": ref_fitness,
            "enrichment": [1.0, 2.0],
            "delta_prediction": [0.0, 0.0],
        }
    )


class TestCachePath(unittest.TestCase):
    def test_path_identifies_species_and_chromosome(self) -> None:
        # Act
        path = correlation_by_species.cache_path_for(ARA_MODEL)

        # Assert
        self.assertEqual(os.path.basename(path), "enrichment_Ara_NC_003070.9.csv")


class TestEnrichmentDfForModel(unittest.TestCase):
    def test_computes_and_writes_cache_on_miss(self) -> None:
        # Arrange
        wrky_df = _enrichment_frame([0.5, 0.6])
        bhlh_df = _enrichment_frame([0.7, 0.8])

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(correlation_by_species, "CACHE_DIR", temp_dir), \
                    patch.object(correlation_by_species, "prepare_wrky_enrichment_df",
                                 return_value=wrky_df) as mock_wrky, \
                    patch.object(correlation_by_species, "prepare_bhlh_enrichment_df",
                                 return_value=bhlh_df) as mock_bhlh:
                # Act
                pooled_df = correlation_by_species.enrichment_df_for_model(ARA_MODEL)

                # Assert: both TF steps got this model's path, rows are pooled,
                # and the cache file now exists.
                mock_wrky.assert_called_once_with(model_path=ARA_MODEL.path)
                mock_bhlh.assert_called_once_with(model_path=ARA_MODEL.path)
                self.assertEqual(len(pooled_df), 4)
                self.assertTrue(
                    os.path.exists(correlation_by_species.cache_path_for(ARA_MODEL))
                )

    def test_reads_cache_without_running_the_model(self) -> None:
        # Arrange: a cache file is already present.
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(correlation_by_species, "CACHE_DIR", temp_dir):
                cache_path = correlation_by_species.cache_path_for(ARA_MODEL)
                _enrichment_frame([0.5, 0.6]).to_csv(cache_path, index=False)

                with patch.object(
                    correlation_by_species, "prepare_wrky_enrichment_df"
                ) as mock_wrky, patch.object(
                    correlation_by_species, "prepare_bhlh_enrichment_df"
                ) as mock_bhlh:
                    # Act
                    cached_df = correlation_by_species.enrichment_df_for_model(
                        ARA_MODEL
                    )

                    # Assert
                    mock_wrky.assert_not_called()
                    mock_bhlh.assert_not_called()
                    self.assertEqual(list(cached_df["prediction_mutated"]), [0.5, 0.6])


class TestBuildLongPredictions(unittest.TestCase):
    def test_stacks_models_with_their_labels(self) -> None:
        # Arrange
        pairs = [
            (ARA_MODEL, _enrichment_frame([0.5, 0.6])),
            (SLYC_MODEL, _enrichment_frame([0.7, 0.8])),
        ]

        # Act
        long_df = correlation_by_species.build_long_predictions(pairs)

        # Assert
        self.assertEqual(len(long_df), 4)
        self.assertEqual(list(long_df["species"]), ["Ara", "Ara", "Slyc", "Slyc"])
        self.assertEqual(list(long_df["prediction"]), [0.5, 0.6, 0.7, 0.8])
        self.assertEqual(list(long_df["enrichment"]), [1.0, 2.0, 1.0, 2.0])


class TestBuildSpeciesEnsembleDf(unittest.TestCase):
    def test_averages_prediction_and_reference_and_recomputes_delta(self) -> None:
        # Arrange: two Ara models with predictions 0.4/0.6 and 0.6/1.0, and
        # reference fitness 0.1/0.2 and 0.3/0.4.
        pairs = [
            (ARA_MODEL, _enrichment_frame([0.4, 0.6], ref_fitness=[0.1, 0.2])),
            (ARA_MODEL_2, _enrichment_frame([0.6, 1.0], ref_fitness=[0.3, 0.4])),
        ]

        # Act
        ensemble_df = correlation_by_species.build_species_ensemble_df(pairs)

        # Assert: means, and delta rebuilt from the averaged columns.
        for column, expected_values in [
            ("prediction_mutated", [0.5, 0.8]),
            ("deepcre_ref_fitness", [0.2, 0.3]),
            ("delta_prediction", [0.3, 0.5]),
        ]:
            for actual, expected in zip(ensemble_df[column], expected_values):
                self.assertAlmostEqual(actual, expected)

    def test_leaves_the_template_frame_untouched(self) -> None:
        # Arrange
        first_frame = _enrichment_frame([0.4, 0.6])
        pairs = [(ARA_MODEL, first_frame), (ARA_MODEL_2, _enrichment_frame([0.6, 1.0]))]

        # Act
        correlation_by_species.build_species_ensemble_df(pairs)

        # Assert: averaging must not mutate the cached per-model frame.
        self.assertEqual(list(first_frame["prediction_mutated"]), [0.4, 0.6])

    def test_raises_on_empty_input(self) -> None:
        # Act / Assert
        with self.assertRaises(ValueError):
            correlation_by_species.build_species_ensemble_df([])

    def test_raises_when_species_are_mixed(self) -> None:
        # Arrange
        pairs = [
            (ARA_MODEL, _enrichment_frame([0.4, 0.6])),
            (SLYC_MODEL, _enrichment_frame([0.6, 1.0])),
        ]

        # Act / Assert
        with self.assertRaises(ValueError):
            correlation_by_species.build_species_ensemble_df(pairs)


class TestGroupBySpecies(unittest.TestCase):
    def test_groups_preserving_order(self) -> None:
        # Arrange
        pairs = [
            (ARA_MODEL, _enrichment_frame([0.4, 0.6])),
            (SLYC_MODEL, _enrichment_frame([0.5, 0.7])),
            (ARA_MODEL_2, _enrichment_frame([0.6, 1.0])),
        ]

        # Act
        grouped = correlation_by_species.group_by_species(pairs)

        # Assert
        self.assertEqual(sorted(grouped), ["Ara", "Slyc"])
        self.assertEqual(
            [model.held_out_chromosome for model, _ in grouped["Ara"]],
            ["NC_003070.9", "NC_003071.7"],
        )
        self.assertEqual(len(grouped["Slyc"]), 1)


if __name__ == "__main__":
    unittest.main()
