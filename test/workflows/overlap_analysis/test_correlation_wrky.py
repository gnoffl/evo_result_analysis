"""Unit tests for ``starrseq_deepcre_correlation_WRKY``.

The deepCRE run-folder layout is faked: ``os.listdir``/``os.path.isdir`` are
patched to present gene folders, ``pyfaidx.Fasta`` yields the reference
sequence, and ``json.load`` returns the Pareto front so that
``load_relevant_deepcre_window_candidates`` can be exercised without disk I/O.
"""
import unittest
from unittest.mock import mock_open, patch

from workflows.overlap_analysis import starrseq_deepcre_correlation_WRKY as wrky


class _FakeRecord:
    """Minimal stand-in for a ``pyfaidx`` record (``str()`` returns the seq)."""

    def __init__(self, name: str, sequence: str) -> None:
        self.name = name
        self._sequence = sequence

    def __str__(self) -> str:
        return self._sequence


class _FakeFasta:
    """Indexable stand-in for a ``pyfaidx.Fasta`` holding one sequence."""

    def __init__(self, sequence: str) -> None:
        self._sequence = sequence

    def __getitem__(self, name):
        return _FakeRecord(name, self._sequence)


class TestLoadRelevantDeepcreWindowCandidates(unittest.TestCase):
    def test_parses_folders_and_orders_coordinates(self):
        # Arrange: one plus-strand folder (start<end) and one minus-strand
        # folder (start>end) whose coordinates must be swapped to start<=end.
        folders = [
            "1_AT1G19040_gene:6575393-6578695",
            "1_AT1G23140_gene:8204500-8203000",
        ]
        fake_fastas = [_FakeFasta("ACGTACGT"), _FakeFasta("TTTTGGGG")]
        pareto_fronts = [
            [["seq0", 0.0], ["seq1", 0.42]],
            [["only", 0.99]],
        ]

        # Act
        with patch("os.listdir", return_value=folders), \
                patch("os.path.isdir", return_value=True), \
                patch.object(wrky, "Fasta", side_effect=fake_fastas), \
                patch("builtins.open", mock_open()), \
                patch("json.load", side_effect=pareto_fronts):
            gene_data = wrky.load_relevant_deepcre_window_candidates("/run")

        # Assert
        self.assertEqual(len(gene_data), 2)

        plus_gene = gene_data[0]
        self.assertEqual(plus_gene["gene"], "AT1G19040")
        self.assertEqual(plus_gene["start"], 6575393)
        self.assertEqual(plus_gene["end"], 6578695)
        self.assertEqual(plus_gene["ref_seq"], "ACGTACGT")
        self.assertAlmostEqual(plus_gene["ref_fitness"], 0.42)
        self.assertEqual(
            plus_gene["folder_name"], "1_AT1G19040_gene:6575393-6578695"
        )

        minus_gene = gene_data[1]
        self.assertEqual(minus_gene["gene"], "AT1G23140")
        self.assertEqual(minus_gene["start"], 8203000)
        self.assertEqual(minus_gene["end"], 8204500)
        self.assertLessEqual(minus_gene["start"], minus_gene["end"])
        self.assertAlmostEqual(minus_gene["ref_fitness"], 0.99)


if __name__ == "__main__":
    unittest.main()
