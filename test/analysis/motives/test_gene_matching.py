"""Unit tests for :mod:`analysis.motives.gene_matching`."""

import unittest

from analysis.motives.gene_matching import resolve_gene_ids


class TestResolveGeneIds(unittest.TestCase):
    """Tests for the ``resolve_gene_ids`` substring matcher."""

    def test_matches_id_as_substring_of_full_name(self):
        # Arrange
        available_names = ["AT1G12345_control_run3", "AT2G00010_control_run3"]
        # Act
        matched_names, unmatched_ids = resolve_gene_ids(
            ["AT1G12345"], available_names
        )
        # Assert
        self.assertEqual(matched_names, ["AT1G12345_control_run3"])
        self.assertEqual(unmatched_ids, [])

    def test_unmatched_id_is_reported(self):
        # Arrange
        available_names = ["AT1G12345_run3"]
        # Act
        matched_names, unmatched_ids = resolve_gene_ids(
            ["AT9G99999"], available_names
        )
        # Assert
        self.assertEqual(matched_names, [])
        self.assertEqual(unmatched_ids, ["AT9G99999"])

    def test_single_id_matching_multiple_names_returns_all(self):
        # Arrange
        available_names = ["AT1G12345_rep1", "AT1G12345_rep2", "AT2G00010_rep1"]
        # Act
        matched_names, unmatched_ids = resolve_gene_ids(
            ["AT1G12345"], available_names
        )
        # Assert
        self.assertEqual(matched_names, ["AT1G12345_rep1", "AT1G12345_rep2"])
        self.assertEqual(unmatched_ids, [])

    def test_mixed_matched_and_unmatched_ids(self):
        # Arrange
        available_names = ["AT1G12345_run3", "AT2G00010_run3"]
        # Act
        matched_names, unmatched_ids = resolve_gene_ids(
            ["AT1G12345", "AT9G99999"], available_names
        )
        # Assert
        self.assertEqual(matched_names, ["AT1G12345_run3"])
        self.assertEqual(unmatched_ids, ["AT9G99999"])

    def test_exact_full_name_still_matches(self):
        # Arrange: passing the full name is a trivial substring of itself.
        available_names = ["AT1G12345_run3"]
        # Act
        matched_names, unmatched_ids = resolve_gene_ids(
            ["AT1G12345_run3"], available_names
        )
        # Assert
        self.assertEqual(matched_names, ["AT1G12345_run3"])
        self.assertEqual(unmatched_ids, [])

    def test_results_are_deduplicated_and_sorted(self):
        # Arrange: two IDs that both hit the same name must not duplicate it,
        # and two unmatched IDs must be returned uniquely and sorted.
        available_names = ["AT1G12345_run3"]
        # Act
        matched_names, unmatched_ids = resolve_gene_ids(
            ["AT1G", "12345", "zzz", "aaa"], available_names
        )
        # Assert
        self.assertEqual(matched_names, ["AT1G12345_run3"])
        self.assertEqual(unmatched_ids, ["aaa", "zzz"])

    def test_empty_requested_ids_matches_nothing(self):
        # Arrange / Act
        matched_names, unmatched_ids = resolve_gene_ids([], ["AT1G12345_run3"])
        # Assert
        self.assertEqual(matched_names, [])
        self.assertEqual(unmatched_ids, [])


if __name__ == "__main__":
    unittest.main()
