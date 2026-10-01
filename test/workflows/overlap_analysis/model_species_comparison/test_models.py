"""Unit tests for the species model registry.

The filesystem is mocked so the tests do not depend on the model directories
being populated.
"""
import unittest
from unittest.mock import patch

from workflows.overlap_analysis.model_species_comparison import _models


class TestParseHeldOutChromosome(unittest.TestCase):
    def test_parses_arabidopsis_accession(self) -> None:
        # Arrange
        file_name = "Atha_S0X0.75dC7K25f_NC_003070.9_ssr_train_models_250617_232757.h5"

        # Act
        accession = _models._parse_held_out_chromosome(file_name)

        # Assert
        self.assertEqual(accession, "NC_003070.9")

    def test_parses_tomato_accession(self) -> None:
        # Arrange
        file_name = "Slyc_S0X0.75dC7K25f_NC_015449.3_ssr_train_models_250617_232757.h5"

        # Act
        accession = _models._parse_held_out_chromosome(file_name)

        # Assert
        self.assertEqual(accession, "NC_015449.3")

    def test_raises_without_accession(self) -> None:
        # Arrange
        file_name = "some_model_without_an_accession.h5"

        # Act / Assert
        with self.assertRaises(ValueError):
            _models._parse_held_out_chromosome(file_name)


class TestListSpeciesModels(unittest.TestCase):
    def test_lists_both_species_sorted_and_labelled(self) -> None:
        # Arrange: two Arabidopsis models given out of order, one tomato model.
        directory_contents = {
            _models.SPECIES_DIRS["Ara"]: [
                "Atha_S0X0.75dC7K25f_NC_003071.7_ssr_train_models_1.h5",
                "Atha_S0X0.75dC7K25f_NC_003070.9_ssr_train_models_1.h5",
                "notes.txt",
            ],
            _models.SPECIES_DIRS["Slyc"]: [
                "Slyc_S0X0.75dC7K25f_NC_015438.3_ssr_train_models_1.h5",
            ],
        }

        # Act
        def fake_listdir(directory):
            return directory_contents[directory]

        with patch.object(_models.os.path, "isdir", return_value=True), \
                patch.object(_models.os, "listdir", side_effect=fake_listdir):
            models = _models.list_species_models()

        # Assert: non-.h5 files ignored, Arabidopsis first, accessions sorted.
        self.assertEqual(
            [(m.species, m.held_out_chromosome) for m in models],
            [
                ("Ara", "NC_003070.9"),
                ("Ara", "NC_003071.7"),
                ("Slyc", "NC_015438.3"),
            ],
        )
        self.assertTrue(models[0].path.endswith("NC_003070.9_ssr_train_models_1.h5"))

    def test_raises_when_directory_missing(self) -> None:
        # Act / Assert
        with patch.object(_models.os.path, "isdir", return_value=False):
            with self.assertRaises(FileNotFoundError):
                _models.list_species_models()

    def test_raises_when_directory_has_no_models(self) -> None:
        # Act / Assert
        with patch.object(_models.os.path, "isdir", return_value=True), \
                patch.object(_models.os, "listdir", return_value=["readme.md"]):
            with self.assertRaises(FileNotFoundError):
                _models.list_species_models()


if __name__ == "__main__":
    unittest.main()
