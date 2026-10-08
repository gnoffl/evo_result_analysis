"""Unit tests for the per-species construct-background pass.

Model inference is mocked; the tests cover this module's own logic: per-model
cache paths and reuse, the long prediction table, and the ensemble averaging.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import pandas as pd

from workflows.overlap_analysis.model_species_comparison import construct_by_species
from workflows.overlap_analysis.model_species_comparison._models import SpeciesModel

ARA_MODEL = SpeciesModel("Ara", "NC_003070.9", "/models/ara_1.h5")
ARA_MODEL_2 = SpeciesModel("Ara", "NC_003071.7", "/models/ara_2.h5")
SLYC_MODEL = SpeciesModel("Slyc", "NC_015438.3", "/models/slyc_1.h5")


def _correlation_frame(predictions):
    """Build a minimal merged correlation frame with two inserts."""
    return pd.DataFrame(
        {
            "id": ["insert_a", "insert_b"],
            "condition": ["Dark", "Light"],
            "enrichment": [1.0, 2.0],
            "tf_family": ["WRKY", "bHLH"],
            "binding_category": ["binding", "non_binding"],
            "prediction": predictions,
        }
    )


class TestCachePath(unittest.TestCase):
    def test_path_identifies_species_and_chromosome(self) -> None:
        # Act
        path = construct_by_species.cache_path_for(SLYC_MODEL)

        # Assert
        self.assertEqual(os.path.basename(path), "construct_Slyc_NC_015438.3.csv")


class TestCorrelationDataForModel(unittest.TestCase):
    def test_passes_model_and_cache_path_to_the_predictor(self) -> None:
        # Arrange
        predictions_df = pd.DataFrame(
            {
                "id": ["insert_a", "insert_a"],
                "barcode": [1, 2],
                "prediction": [0.4, 0.6],
            }
        )
        starrseq_df = pd.DataFrame(
            {
                "id": ["insert_a"],
                "condition": ["Dark"],
                "enrichment": [1.0],
                "tf_family": ["WRKY"],
                "binding_category": ["binding"],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(construct_by_species, "CACHE_DIR", temp_dir), \
                    patch.object(construct_by_species, "predict_sequences",
                                 return_value=predictions_df) as mock_predict:
                # Act
                correlation_df = construct_by_species.correlation_data_for_model(
                    ARA_MODEL, starrseq_df
                )

                # Assert: the predictor is pointed at this model and its own
                # cache file, and the two barcodes are averaged into one row.
                mock_predict.assert_called_once_with(
                    cache_path=construct_by_species.cache_path_for(ARA_MODEL),
                    model_path=ARA_MODEL.path,
                )
                self.assertEqual(len(correlation_df), 1)
                self.assertAlmostEqual(correlation_df.iloc[0]["prediction"], 0.5)


class TestBuildLongPredictions(unittest.TestCase):
    def test_stacks_models_with_their_labels(self) -> None:
        # Arrange
        pairs = [
            (ARA_MODEL, _correlation_frame([0.5, 0.6])),
            (SLYC_MODEL, _correlation_frame([0.7, 0.8])),
        ]

        # Act
        long_df = construct_by_species.build_long_predictions(pairs)

        # Assert
        self.assertEqual(len(long_df), 4)
        self.assertEqual(list(long_df["species"]), ["Ara", "Ara", "Slyc", "Slyc"])
        self.assertEqual(list(long_df["prediction"]), [0.5, 0.6, 0.7, 0.8])


class TestBuildSpeciesEnsembleDf(unittest.TestCase):
    def test_averages_predictions_across_models(self) -> None:
        # Arrange
        pairs = [
            (ARA_MODEL, _correlation_frame([0.4, 0.6])),
            (ARA_MODEL_2, _correlation_frame([0.6, 1.0])),
        ]

        # Act
        ensemble_df = construct_by_species.build_species_ensemble_df(pairs)

        # Assert
        self.assertEqual(list(ensemble_df["prediction"]), [0.5, 0.8])
        self.assertEqual(list(ensemble_df["enrichment"]), [1.0, 2.0])

    def test_leaves_the_template_frame_untouched(self) -> None:
        # Arrange
        first_frame = _correlation_frame([0.4, 0.6])
        pairs = [
            (ARA_MODEL, first_frame),
            (ARA_MODEL_2, _correlation_frame([0.6, 1.0])),
        ]

        # Act
        construct_by_species.build_species_ensemble_df(pairs)

        # Assert
        self.assertEqual(list(first_frame["prediction"]), [0.4, 0.6])

    def test_raises_on_empty_input(self) -> None:
        # Act / Assert
        with self.assertRaises(ValueError):
            construct_by_species.build_species_ensemble_df([])

    def test_raises_when_species_are_mixed(self) -> None:
        # Arrange
        pairs = [
            (ARA_MODEL, _correlation_frame([0.4, 0.6])),
            (SLYC_MODEL, _correlation_frame([0.6, 1.0])),
        ]

        # Act / Assert
        with self.assertRaises(ValueError):
            construct_by_species.build_species_ensemble_df(pairs)


class TestGroupBySpecies(unittest.TestCase):
    def test_groups_preserving_order(self) -> None:
        # Arrange
        pairs = [
            (ARA_MODEL, _correlation_frame([0.4, 0.6])),
            (SLYC_MODEL, _correlation_frame([0.5, 0.7])),
            (ARA_MODEL_2, _correlation_frame([0.6, 1.0])),
        ]

        # Act
        grouped = construct_by_species.group_by_species(pairs)

        # Assert
        self.assertEqual(sorted(grouped), ["Ara", "Slyc"])
        self.assertEqual(len(grouped["Ara"]), 2)
        self.assertEqual(len(grouped["Slyc"]), 1)


if __name__ == "__main__":
    unittest.main()
