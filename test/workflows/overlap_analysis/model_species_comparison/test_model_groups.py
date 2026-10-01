"""Unit tests for the four-group model listing.

The model directories are replaced by temporary directories holding empty files
named like the real ``.h5`` models, so the tests cover the name parsing and the
listing without depending on the models being present.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

from workflows.overlap_analysis.model_species_comparison import _model_groups
from workflows.overlap_analysis.model_species_comparison._model_groups import (
    MODEL_GROUPS,
    ModelGroup,
    list_all_group_models,
    list_group_models,
    parse_member_name,
)

MSR_FILE = (
    "MSR_atha_bvul_bole_csat_cqui_ccum_dcar_gmax_osat_slyc_sbic_zmay_"
    "M0X0.75dP7K25f_A_thaliana_msr_train_models_250705_070641.h5"
)
ARA_FILE = "Atha_S0X0.75dC7K25f_NC_003070.9_ssr_train_models_250617_232757.h5"
SLYC_FILE = "Slyc_S0X0.75dC7K25f_NC_015438.3_ssr_train_models_250617_232757.h5"
NTAB_FILE = "nicotiana_tabacum_NtabPC1_ssr_train_models_260915_180819.h5"

GROUPS_BY_NAME = {group.name: group for group in MODEL_GROUPS}


def _write_files(directory: str, file_names: list) -> None:
    """Create empty files with the given names inside a directory."""
    for file_name in file_names:
        open(os.path.join(directory, file_name), "w").close()


class TestParseMemberName(unittest.TestCase):
    def test_parses_the_held_out_species_of_an_msr_model(self) -> None:
        # Act
        name = parse_member_name(MSR_FILE, GROUPS_BY_NAME["MSR"].member_pattern)

        # Assert
        self.assertEqual(name, "A_thaliana")

    def test_parses_the_held_out_chromosome_of_an_ssr_model(self) -> None:
        # Act
        ara_name = parse_member_name(ARA_FILE, GROUPS_BY_NAME["Ara"].member_pattern)
        slyc_name = parse_member_name(SLYC_FILE, GROUPS_BY_NAME["Slyc"].member_pattern)
        ntab_name = parse_member_name(NTAB_FILE, GROUPS_BY_NAME["Ntab"].member_pattern)

        # Assert
        self.assertEqual(ara_name, "NC_003070.9")
        self.assertEqual(slyc_name, "NC_015438.3")
        self.assertEqual(ntab_name, "NtabPC1")

    def test_ssr_member_names_match_the_species_comparison_cache_keys(self) -> None:
        # Arrange: the SSR caches are keyed by the accession _models parses, so
        # both parsers must agree for the cache to be reusable.
        from workflows.overlap_analysis.model_species_comparison._models import (
            _parse_held_out_chromosome,
        )

        # Act
        group_name = parse_member_name(ARA_FILE, GROUPS_BY_NAME["Ara"].member_pattern)

        # Assert
        self.assertEqual(group_name, _parse_held_out_chromosome(ARA_FILE))

    def test_raises_when_the_pattern_does_not_match(self) -> None:
        # Act / Assert
        with self.assertRaises(ValueError):
            parse_member_name("unexpected_name.h5", r"_(NC_[0-9.]+)_ssr_train")


class TestListGroupModels(unittest.TestCase):
    def test_lists_every_model_with_the_group_name_as_species(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as directory:
            _write_files(directory, [NTAB_FILE, NTAB_FILE.replace("PC1", "PC2")])
            group = ModelGroup(
                "Ntab", directory, GROUPS_BY_NAME["Ntab"].member_pattern
            )

            # Act
            models = list_group_models(group)

        # Assert
        self.assertEqual([model.species for model in models], ["Ntab", "Ntab"])
        self.assertEqual(
            [model.held_out_chromosome for model in models], ["NtabPC1", "NtabPC2"]
        )
        self.assertEqual([model.path for model in models], [os.path.join(directory, NTAB_FILE), os.path.join(directory, NTAB_FILE.replace("PC1", "PC2"))])

    def test_sorts_by_member_name_and_ignores_non_model_files(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as directory:
            _write_files(
                directory,
                [SLYC_FILE.replace("015438", "015440"), SLYC_FILE, "notes.txt"],
            )
            group = ModelGroup(
                "Slyc", directory, GROUPS_BY_NAME["Slyc"].member_pattern
            )

            # Act
            models = list_group_models(group)

        # Assert
        self.assertEqual(
            [model.held_out_chromosome for model in models],
            ["NC_015438.3", "NC_015440.3"],
        )

    def test_raises_when_the_directory_is_missing(self) -> None:
        # Arrange
        group = ModelGroup("Ntab", "/does/not/exist", r"(x)")

        # Act / Assert
        with self.assertRaises(FileNotFoundError):
            list_group_models(group)

    def test_raises_when_the_directory_holds_no_models(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as directory:
            group = ModelGroup("Ntab", directory, r"(x)")

            # Act / Assert
            with self.assertRaises(FileNotFoundError):
                list_group_models(group)


class TestListAllGroupModels(unittest.TestCase):
    def test_concatenates_the_groups_in_declaration_order(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as root:
            group_files = {
                "MSR": [MSR_FILE],
                "Ara": [ARA_FILE],
                "Slyc": [SLYC_FILE],
                "Ntab": [NTAB_FILE],
            }
            groups = []
            for name, file_names in group_files.items():
                directory = os.path.join(root, name)
                os.makedirs(directory)
                _write_files(directory, file_names)
                groups.append(
                    ModelGroup(name, directory, GROUPS_BY_NAME[name].member_pattern)
                )

            # Act
            with patch.object(_model_groups, "MODEL_GROUPS", groups):
                models = list_all_group_models()

        # Assert
        self.assertEqual(
            [model.species for model in models], ["MSR", "Ara", "Slyc", "Ntab"]
        )
        self.assertEqual(
            [model.held_out_chromosome for model in models],
            ["A_thaliana", "NC_003070.9", "NC_015438.3", "NtabPC1"],
        )


if __name__ == "__main__":
    unittest.main()
