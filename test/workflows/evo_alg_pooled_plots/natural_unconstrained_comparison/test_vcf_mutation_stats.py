"""Unit tests for vcf_mutation_stats."""

import unittest
from pathlib import Path
from unittest.mock import patch, mock_open

from workflows.evo_alg_pooled_plots.natural_unconstrained_comparison.vcf_mutation_stats import (
    build_summary_row,
    collect_stats,
    count_mutations_in_vcf,
)

VCF_HEADER = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"


def _vcf_with_positions(*positions: int) -> str:
    rows = "".join(
        f"CHR\t{pos}\t.\tA\tT\t.\t.\tREGION=promoter\n" for pos in positions
    )
    return VCF_HEADER + rows


class TestCountMutationsInVcf(unittest.TestCase):

    def test_all_unique_positions(self):
        content = _vcf_with_positions(10, 20, 30)
        with patch.object(Path, "read_text", return_value=content):
            unique_locs, total_muts = count_mutations_in_vcf(Path("fake.vcf"))
        self.assertEqual(unique_locs, 3)
        self.assertEqual(total_muts, 3)

    def test_duplicate_positions_counted_separately(self):
        content = _vcf_with_positions(10, 10, 30)
        with patch.object(Path, "read_text", return_value=content):
            unique_locs, total_muts = count_mutations_in_vcf(Path("fake.vcf"))
        self.assertEqual(unique_locs, 2)
        self.assertEqual(total_muts, 3)

    def test_header_lines_skipped(self):
        content = VCF_HEADER  # no data rows
        with patch.object(Path, "read_text", return_value=content):
            unique_locs, total_muts = count_mutations_in_vcf(Path("fake.vcf"))
        self.assertEqual(unique_locs, 0)
        self.assertEqual(total_muts, 0)


class TestCollectStats(unittest.TestCase):

    def test_returns_counts_for_each_vcf(self):
        files = [Path("gene1.vcf"), Path("gene2.vcf")]
        vcf_contents = {
            files[0]: _vcf_with_positions(1, 2, 3),
            files[1]: _vcf_with_positions(1, 1, 5, 6),
        }

        def fake_read_text(self):
            return vcf_contents[self]

        with patch.object(Path, "read_text", fake_read_text), \
             patch.object(Path, "glob", return_value=iter(files)):
            unique_locs, total_muts = collect_stats(Path("fake_dir"))

        self.assertEqual(unique_locs, [3, 3])
        self.assertEqual(total_muts, [3, 4])

    def test_empty_directory(self):
        with patch.object(Path, "glob", return_value=iter([])):
            unique_locs, total_muts = collect_stats(Path("fake_dir"))
        self.assertEqual(unique_locs, [])
        self.assertEqual(total_muts, [])


class TestBuildSummaryRow(unittest.TestCase):

    def test_correct_statistics(self):
        row = build_summary_row("GOF", [10, 20, 30], [11, 21, 31])
        self.assertEqual(row["group"], "GOF")
        self.assertEqual(row["n_genes"], 3)
        self.assertAlmostEqual(row["unique_locations_mean"], 20.0)
        self.assertAlmostEqual(row["unique_locations_min"], 10)
        self.assertAlmostEqual(row["unique_locations_max"], 30)
        self.assertAlmostEqual(row["total_mutations_mean"], 21.0)
        self.assertAlmostEqual(row["total_mutations_min"], 11)
        self.assertAlmostEqual(row["total_mutations_max"], 31)

    def test_single_gene(self):
        row = build_summary_row("LOF", [5], [7])
        self.assertEqual(row["n_genes"], 1)
        self.assertEqual(row["unique_locations_mean"], 5.0)
        self.assertEqual(row["unique_locations_std"], 0.0)
        self.assertEqual(row["total_mutations_mean"], 7.0)


if __name__ == "__main__":
    unittest.main()
