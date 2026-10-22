"""Tests for natural_gof_lof_top_deltas.py."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workflows.evo_alg_pooled_plots.natural_gof_lof_top_deltas import (
    build_gene_row,
    build_table,
    find_front_entry,
    format_mutations,
    load_genomic_positions,
    parse_gene_dir_name,
)

VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
)


def _vcf_line(sequence_position: int, reference: str, alternative: str, genomic_position: int) -> str:
    """Build one reextracted-VCF record line (1-based sequence position)."""
    return (
        f"1_ATTEST_gene:100-200\t{sequence_position}\t.\t{reference}\t{alternative}"
        f"\t.\t.\tREGION=promoter;GENOMIC_POS={genomic_position};SOURCE=x.vcf\n"
    )


def _fake_genome(chromosome: str, bases: dict) -> dict:
    """Build a chromosome mapping carrying the given 1-based positions.

    Args:
        chromosome: Chromosome name to key the mapping by.
        bases: Mapping from 1-based genomic position to base; all other
            positions become ``N``.

    Returns:
        A ``{chromosome: sequence}`` dict, indexable like a pyfaidx Fasta.
    """
    sequence = "".join(bases.get(position, "N") for position in range(1, max(bases) + 1))
    return {chromosome: sequence}


class TestParseGeneDirName(unittest.TestCase):
    """Tests for parse_gene_dir_name."""

    def test_plus_strand_gene(self):
        # Arrange
        dir_name = "1_AT1G01720_gene:267992-269819_260311_122242_380559"

        # Act
        gene_id, chromosome, start, end, strand = parse_gene_dir_name(dir_name)

        # Assert
        self.assertEqual((gene_id, chromosome, start, end, strand), ("AT1G01720", "1", 267992, 269819, "+"))

    def test_minus_strand_gene(self):
        # Arrange
        dir_name = "1_AT1G07867_gene:19807113-19806916_260311_122242_377475"

        # Act
        _gene_id, _chromosome, start, end, strand = parse_gene_dir_name(dir_name)

        # Assert
        self.assertEqual((start, end, strand), (19807113, 19806916, "-"))


class TestLoadGenomicPositions(unittest.TestCase):
    """Tests for load_genomic_positions."""

    def test_converts_sequence_position_to_zero_based_key(self):
        # Arrange
        with tempfile.TemporaryDirectory() as directory:
            vcf_path = Path(directory) / "gene.vcf"
            vcf_path.write_text(
                VCF_HEADER + _vcf_line(30, "G", "T", 267022) + _vcf_line(31, "A", "C", 267023)
            )

            # Act
            positions = load_genomic_positions(vcf_path)

        # Assert
        self.assertEqual(positions, {29: 267022, 30: 267023})


class TestFindFrontEntry(unittest.TestCase):
    """Tests for find_front_entry."""

    def test_returns_matching_entry(self):
        # Arrange
        front = [["AAA", 0.9, 2.0], ["AAC", 0.8, 1.0], ["ACC", 0.7, 0.0]]

        # Act
        sequence, fitness = find_front_entry(front, 1)

        # Assert
        self.assertEqual((sequence, fitness), ("AAC", 0.8))

    def test_raises_when_budget_absent(self):
        # Arrange
        front = [["AAA", 0.9, 2.0]]

        # Act / Assert
        with self.assertRaises(ValueError):
            find_front_entry(front, 5)


class TestFormatMutations(unittest.TestCase):
    """Tests for format_mutations."""

    def test_plus_strand_uses_bases_as_is(self):
        # Arrange
        reference = "ACGT"
        mutated = "ATGT"
        genome = _fake_genome("2", {500: "C"})

        # Act
        mutations = format_mutations(reference, mutated, "2", "+", {1: 500}, genome)

        # Assert
        self.assertEqual(mutations, ["2:500 C->T"])

    def test_minus_strand_complements_both_bases(self):
        # Arrange
        reference = "ACGT"
        mutated = "ATGT"
        genome = _fake_genome("2", {500: "G"})

        # Act
        mutations = format_mutations(reference, mutated, "2", "-", {1: 500}, genome)

        # Assert
        self.assertEqual(mutations, ["2:500 G->A"])

    def test_orders_by_sequence_position(self):
        # Arrange
        reference = "AAAA"
        mutated = "CAAC"
        genome = _fake_genome("2", {10: "A", 40: "A"})

        # Act
        mutations = format_mutations(reference, mutated, "2", "+", {0: 10, 3: 40}, genome)

        # Assert
        self.assertEqual(mutations, ["2:10 A->C", "2:40 A->C"])

    def test_raises_on_position_without_vcf_record(self):
        # Arrange
        genome = _fake_genome("1", {10: "A"})

        # Act / Assert
        with self.assertRaises(ValueError):
            format_mutations("AAAA", "ACAA", "1", "+", {0: 10}, genome)

    def test_raises_when_genome_base_disagrees(self):
        # Arrange
        genome = _fake_genome("1", {500: "T"})

        # Act / Assert
        with self.assertRaises(ValueError):
            format_mutations("ACGT", "ATGT", "1", "+", {1: 500}, genome)

    def test_raises_on_length_mismatch(self):
        # Arrange
        genome = _fake_genome("1", {1: "A"})

        # Act / Assert
        with self.assertRaises(ValueError):
            format_mutations("AAAA", "AAA", "1", "+", {}, genome)


def _write_gene_run(root: Path, dir_name: str, reference: str, front: list) -> Path:
    """Create a minimal gene run directory with a pareto front on disk."""
    gene_dir = root / dir_name
    (gene_dir / "saved_populations").mkdir(parents=True)
    with open(gene_dir / "saved_populations" / "pareto_front.json", "w") as handle:
        json.dump(front, handle)
    return gene_dir


class TestBuildGeneRow(unittest.TestCase):
    """Tests for build_gene_row."""

    def test_builds_row_with_one_column_per_mutation(self):
        # Arrange
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gene_dir = _write_gene_run(
                root,
                "1_ATTEST_gene:100-200_260311_122242_380559",
                "ACGT",
                [["ATGA", 0.9, 2.0], ["ATGT", 0.8, 1.0], ["ACGT", 0.5, 0.0]],
            )
            vcf_dir = root / "vcfs"
            vcf_dir.mkdir()
            (vcf_dir / "1_ATTEST_gene:100-200.vcf").write_text(
                VCF_HEADER + _vcf_line(2, "C", "T", 500) + _vcf_line(4, "T", "A", 502)
            )
            genome = _fake_genome("1", {500: "C", 502: "T"})

            # Act
            with patch(
                "workflows.evo_alg_pooled_plots.natural_gof_lof_top_deltas."
                "load_reference_sequence",
                return_value="ACGT",
            ):
                row = build_gene_row(gene_dir, "GOF", vcf_dir, 2, genome)

        # Assert
        self.assertEqual(row["gene_id"], "ATTEST")
        self.assertEqual(row["group"], "GOF")
        self.assertEqual(row["strand"], "+")
        self.assertAlmostEqual(row["delta_mut"], 0.4)
        self.assertEqual(row["mutation1"], "1:500 C->T")
        self.assertEqual(row["mutation2"], "1:502 T->A")

    def test_raises_when_diff_count_disagrees_with_budget(self):
        # Arrange
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gene_dir = _write_gene_run(
                root,
                "1_ATTEST_gene:100-200_260311_122242_380559",
                "ACGT",
                [["ATGT", 0.8, 2.0], ["ACGT", 0.5, 0.0]],
            )
            vcf_dir = root / "vcfs"
            vcf_dir.mkdir()
            (vcf_dir / "1_ATTEST_gene:100-200.vcf").write_text(
                VCF_HEADER + _vcf_line(2, "C", "T", 500)
            )
            genome = _fake_genome("1", {500: "C"})

            # Act / Assert
            with patch(
                "workflows.evo_alg_pooled_plots.natural_gof_lof_top_deltas."
                "load_reference_sequence",
                return_value="ACGT",
            ):
                with self.assertRaises(ValueError):
                    build_gene_row(gene_dir, "GOF", vcf_dir, 2, genome)


class TestBuildTable(unittest.TestCase):
    """Tests for build_table."""

    def test_pools_runs_and_sorts_by_delta_descending(self):
        # Arrange
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gof_dir, lof_dir = root / "gof", root / "lof"
            for run_dir in (gof_dir, lof_dir):
                (run_dir / "1_ATTEST_gene:100-200_x").mkdir(parents=True)
            rows = iter(
                [
                    {"gene_id": "A", "group": "GOF", "delta_mut": 0.4},
                    {"gene_id": "B", "group": "LOF", "delta_mut": -0.6},
                ]
            )

            # Act
            with patch(
                "workflows.evo_alg_pooled_plots.natural_gof_lof_top_deltas.build_gene_row",
                side_effect=lambda *args, **kwargs: next(rows),
            ):
                table = build_table(
                    [(gof_dir, root, "GOF"), (lof_dir, root, "LOF")],
                    5,
                    _fake_genome("1", {1: "A"}),
                )

        # Assert
        self.assertEqual(list(table["gene_id"]), ["A", "B"])
        self.assertEqual(list(table["delta_mut"]), [0.4, -0.6])


if __name__ == "__main__":
    unittest.main()
