"""Unit tests for analysis.blamm.meme_to_jaspar."""

import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from analysis.blamm.meme_to_jaspar import (
    DEFAULT_NSITES,
    collect_motifs,
    convert_meme_databases,
    parse_meme_file,
    write_jaspar,
    write_metadata,
)

MEME_HEADER = """MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

"""

MOTIF_BLOCK_A = """MOTIF MA0001.1 AlphaTF
letter-probability matrix: alength= 4 w= 3 nsites= 10 E= 0
 1.000000  0.000000  0.000000  0.000000
 0.500000  0.500000  0.000000  0.000000
 0.000000  0.000000  0.200000  0.800000
URL https://example.org/MA0001.1

"""

MOTIF_BLOCK_B = """MOTIF MA0002.1 BetaTF
letter-probability matrix: alength= 4 w= 2 nsites= 4 E= 0
 0.250000  0.250000  0.250000  0.250000
 0.000000  1.000000  0.000000  0.000000

"""


class MemeToJasparTest(unittest.TestCase):
    """Tests for MEME parsing and JASPAR conversion."""

    def setUp(self) -> None:
        """Create a temporary directory holding synthetic MEME files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir)

    def _write_meme(self, filename: str, body: str) -> str:
        """Write a MEME file with the standard header and return its path."""
        path = os.path.join(self.temp_dir, filename)
        with open(path, "w") as handle:
            handle.write(MEME_HEADER + body)
        return path

    def test_parse_meme_file_converts_probabilities_to_counts(self) -> None:
        """Probabilities are multiplied by nsites and rounded to integers."""
        # Arrange
        meme_path = self._write_meme("a.meme", MOTIF_BLOCK_A)

        # Act
        motifs = parse_meme_file(meme_path)

        # Assert
        self.assertEqual(len(motifs), 1)
        motif = motifs[0]
        self.assertEqual(motif.motif_id, "MA0001.1")
        self.assertEqual(motif.motif_name, "AlphaTF")
        self.assertEqual(motif.source_db, "a.meme")
        self.assertEqual(motif.nsites, 10)
        self.assertEqual(motif.width, 3)
        self.assertEqual(motif.counts["A"], [10, 5, 0])
        self.assertEqual(motif.counts["C"], [0, 5, 0])
        self.assertEqual(motif.counts["G"], [0, 0, 2])
        self.assertEqual(motif.counts["T"], [0, 0, 8])

    def test_parse_meme_file_ignores_log_odds_matrices(self) -> None:
        """Only letter-probability matrices are converted."""
        # Arrange
        body = (
            "MOTIF MYB52 MYB52\n"
            "log-odds matrix: alength= 4 w= 2 E= 0\n"
            " 1.0 -1.0 0.0 0.0\n"
            " 0.0  0.0 1.0 -1.0\n"
            "letter-probability matrix: alength= 4 w= 2 nsites= 8 E= 0\n"
            " 1.000000 0.000000 0.000000 0.000000\n"
            " 0.000000 0.000000 0.500000 0.500000\n"
        )
        meme_path = self._write_meme("pbm.meme", body)

        # Act
        motifs = parse_meme_file(meme_path)

        # Assert
        self.assertEqual(len(motifs), 1)
        self.assertEqual(motifs[0].counts["A"], [8, 0])
        self.assertEqual(motifs[0].counts["T"], [0, 4])

    def test_parse_meme_file_defaults_nsites_when_missing(self) -> None:
        """A motif without nsites falls back to the documented default."""
        # Arrange
        body = (
            "MOTIF NO_SITES\n"
            "letter-probability matrix: alength= 4 w= 1\n"
            " 1.000000 0.000000 0.000000 0.000000\n"
        )
        meme_path = self._write_meme("nosites.meme", body)

        # Act
        motifs = parse_meme_file(meme_path)

        # Assert
        self.assertEqual(motifs[0].nsites, DEFAULT_NSITES)
        self.assertEqual(motifs[0].counts["A"], [DEFAULT_NSITES])
        self.assertEqual(motifs[0].motif_name, "NO_SITES")

    def test_parse_meme_file_rejects_non_dna_alphabet(self) -> None:
        """An alphabet length other than four is an error."""
        # Arrange
        body = (
            "MOTIF PROT1\n"
            "letter-probability matrix: alength= 20 w= 1 nsites= 5\n"
            " 1.0 0.0 0.0 0.0\n"
        )
        meme_path = self._write_meme("prot.meme", body)

        # Act / Assert
        with self.assertRaises(ValueError):
            parse_meme_file(meme_path)

    def test_parse_meme_file_rejects_truncated_matrix(self) -> None:
        """A matrix with fewer rows than declared is an error."""
        # Arrange
        body = (
            "MOTIF SHORT\n"
            "letter-probability matrix: alength= 4 w= 3 nsites= 5\n"
            " 1.0 0.0 0.0 0.0\n"
        )
        meme_path = self._write_meme("short.meme", body)

        # Act / Assert
        with self.assertRaises(ValueError):
            parse_meme_file(meme_path)

    def test_collect_motifs_rejects_duplicate_identifiers(self) -> None:
        """The same motif id in two databases is rejected."""
        # Arrange
        first = self._write_meme("first.meme", MOTIF_BLOCK_A)
        second = self._write_meme("second.meme", MOTIF_BLOCK_A)

        # Act / Assert
        with self.assertRaises(ValueError):
            collect_motifs([first, second])

    def test_collect_motifs_concatenates_in_order(self) -> None:
        """Motifs of several databases keep their file order."""
        # Arrange
        first = self._write_meme("first.meme", MOTIF_BLOCK_A)
        second = self._write_meme("second.meme", MOTIF_BLOCK_B)

        # Act
        motifs = collect_motifs([first, second])

        # Assert
        self.assertEqual(
            [motif.motif_id for motif in motifs], ["MA0001.1", "MA0002.1"]
        )
        self.assertEqual(
            [motif.source_db for motif in motifs], ["first.meme", "second.meme"]
        )

    def test_write_jaspar_produces_blamm_readable_layout(self) -> None:
        """Each motif is one header line followed by four count rows."""
        # Arrange
        meme_path = self._write_meme("a.meme", MOTIF_BLOCK_A)
        motifs = parse_meme_file(meme_path)
        jaspar_path = os.path.join(self.temp_dir, "motifs.jaspar")

        # Act
        write_jaspar(motifs, jaspar_path)

        # Assert
        with open(jaspar_path) as handle:
            lines = handle.read().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[0], ">MA0001.1\tAlphaTF")
        self.assertTrue(lines[1].startswith("A  ["))
        self.assertTrue(lines[1].rstrip().endswith("]"))
        self.assertEqual(
            [int(value) for value in lines[1].split("[")[1].strip(" ]").split()],
            [10, 5, 0],
        )
        self.assertEqual([line[0] for line in lines[1:]], list("ACGT"))

    def test_write_metadata_records_source_database(self) -> None:
        """The metadata table carries id, name, source, width and nsites."""
        # Arrange
        meme_path = self._write_meme("a.meme", MOTIF_BLOCK_A)
        motifs = parse_meme_file(meme_path)
        metadata_path = os.path.join(self.temp_dir, "metadata.csv")

        # Act
        metadata = write_metadata(motifs, metadata_path)

        # Assert
        expected_columns = [
            "motif_id",
            "motif_name",
            "source_db",
            "width",
            "nsites",
        ]
        self.assertEqual(list(metadata.columns), expected_columns)
        reloaded = pd.read_csv(metadata_path)
        self.assertEqual(reloaded.loc[0, "motif_id"], "MA0001.1")
        self.assertEqual(reloaded.loc[0, "source_db"], "a.meme")
        self.assertEqual(reloaded.loc[0, "width"], 3)

    def test_convert_meme_databases_writes_both_outputs(self) -> None:
        """The convenience wrapper writes the jaspar file and the metadata."""
        # Arrange
        first = self._write_meme("first.meme", MOTIF_BLOCK_A)
        second = self._write_meme("second.meme", MOTIF_BLOCK_B)
        jaspar_path = os.path.join(self.temp_dir, "out", "motifs.jaspar")
        metadata_path = os.path.join(self.temp_dir, "out", "metadata.csv")

        # Act
        metadata = convert_meme_databases(
            [first, second], jaspar_path, metadata_path
        )

        # Assert
        self.assertTrue(os.path.isfile(jaspar_path))
        self.assertTrue(os.path.isfile(metadata_path))
        self.assertEqual(len(metadata), 2)


if __name__ == "__main__":
    unittest.main()
