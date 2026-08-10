"""Unit tests for the natural_variation_in_window one-off summary script."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import pandas as pd

from workflows.mutation_distribution_analysis import natural_variation_in_window as nviw


VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##source=extract_sequences.py\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
)


def _write_vcf(directory: str, gene_id: str, positions: list[int]) -> str:
    """Write a minimal re-extracted VCF for one gene.

    Args:
        directory: Directory to write into.
        gene_id: Value placed in the CHROM column.
        positions: 1-based positions in the 3020 bp construct.

    Returns:
        Path of the written file.
    """
    path = os.path.join(directory, f"{gene_id}.vcf")
    with open(path, "w") as handle:
        handle.write(VCF_HEADER)
        for position in positions:
            region = "promoter" if position <= 1500 else "terminator"
            handle.write(
                f"{gene_id}\t{position}\t.\tG\tT\t.\t.\tREGION={region}\n"
            )
    return path


class TestWindowConstants(unittest.TestCase):
    """The window must match the 170 bp test sequence at -220..-51."""

    def test_window_length_is_170(self) -> None:
        self.assertEqual(nviw.WINDOW_LENGTH, 170)

    def test_window_lies_inside_the_promoter_half(self) -> None:
        self.assertGreaterEqual(nviw.WINDOW_START, nviw.PROMOTER_START)
        self.assertLessEqual(nviw.WINDOW_END, nviw.PROMOTER_END)

    def test_window_matches_tss_relative_coordinates(self) -> None:
        # 1-based POS p sits at relative coordinate p - 1001 (index 1000 = +1).
        self.assertEqual(nviw.WINDOW_START - 1001, -220)
        self.assertEqual(nviw.WINDOW_END - 1001, -51)


class TestReadVcfPositions(unittest.TestCase):
    """Header lines are skipped and only CHROM/POS are kept."""

    def test_parses_positions_and_skips_headers(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as directory:
            path = _write_vcf(directory, "geneA", [10, 800, 2000])

            # Act
            table = nviw.read_vcf_positions(path)

        # Assert
        self.assertEqual(list(table.columns), ["gene_id", "position"])
        self.assertEqual(list(table["position"]), [10, 800, 2000])
        self.assertEqual(set(table["gene_id"]), {"geneA"})


class TestLoadVariation(unittest.TestCase):
    """All VCFs in a directory are concatenated; an empty directory is an error."""

    def test_concatenates_all_files(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as directory:
            _write_vcf(directory, "geneA", [800, 900])
            _write_vcf(directory, "geneB", [100])

            # Act
            variation = nviw.load_variation(directory)

        # Assert
        self.assertEqual(len(variation), 3)
        self.assertEqual(set(variation["gene_id"]), {"geneA", "geneB"})

    def test_raises_on_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                nviw.load_variation(directory)


class TestSummarize(unittest.TestCase):
    """Counting is inclusive at both window edges and genes with zero hits appear."""

    def _summarize(self, variation: pd.DataFrame) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            nviw.summarize("TEST", variation)
        return buffer.getvalue()

    def test_window_edges_are_inclusive(self) -> None:
        # Arrange: one SNP on each boundary, one just outside on each side.
        variation = pd.DataFrame(
            {
                "gene_id": ["geneA"] * 4,
                "position": [
                    nviw.WINDOW_START - 1,
                    nviw.WINDOW_START,
                    nviw.WINDOW_END,
                    nviw.WINDOW_END + 1,
                ],
            }
        )

        # Act
        output = self._summarize(variation)

        # Assert
        self.assertIn(
            f"SNPs in window ({nviw.WINDOW_START}-{nviw.WINDOW_END}):  2", output
        )

    def test_genes_without_window_snps_are_counted(self) -> None:
        # Arrange: geneB has only a terminator SNP.
        variation = pd.DataFrame(
            {
                "gene_id": ["geneA", "geneB"],
                "position": [nviw.WINDOW_START, 2000],
            }
        )

        # Act
        output = self._summarize(variation)

        # Assert
        self.assertIn("genes:                       2", output)
        self.assertIn("genes with 0 in window:      1", output)

    def test_promoter_count_excludes_terminator(self) -> None:
        # Arrange
        variation = pd.DataFrame(
            {
                "gene_id": ["geneA"] * 3,
                "position": [1, nviw.PROMOTER_END, nviw.PROMOTER_END + 21],
            }
        )

        # Act
        output = self._summarize(variation)

        # Assert
        self.assertIn("SNPs total (3020 bp):        3", output)
        self.assertIn("SNPs in promoter (1-1500):   2", output)


if __name__ == "__main__":
    unittest.main()
