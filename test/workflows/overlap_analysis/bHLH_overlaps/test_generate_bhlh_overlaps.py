"""Unit tests for ``generate_bhlh_overlaps``.

The tests exercise the real ``find_start_end`` / ``genomic_to_relative_position``
from the ``evolution`` package against hand-computed expectations so that any
silent change to the windowing math will fail loudly.
"""
import os
import tempfile
import unittest
from typing import Dict, List
from unittest.mock import patch

import pandas as pd

from workflows.overlap_analysis.bHLH_overlaps import generate_bhlh_overlaps as gbo
from workflows.overlap_analysis.bHLH_overlaps.generate_bhlh_overlaps import (
    OUTPUT_COLUMNS,
    Site,
    build_mapping,
    overlaps_for_site,
    parse_fasta_sites,
)


def _make_genes_df(rows: List[Dict]) -> pd.DataFrame:
    """Build a genes DataFrame in the shape returned by ``find_genes``."""
    return pd.DataFrame(rows, columns=["chromosome", "start", "end", "strand", "gene_id"])


def _write_fasta(tmpdir: str, headers_and_seqs: List[tuple]) -> str:
    """Write a tiny FASTA file and return its path."""
    fasta_path = os.path.join(tmpdir, "sites.fasta")
    with open(fasta_path, "w") as handle:
        for header, sequence in headers_and_seqs:
            handle.write(f">{header}\n{sequence}\n")
    return fasta_path


class TestParseFastaSites(unittest.TestCase):
    def test_collapses_variants_at_the_same_genomic_site(self):
        # Arrange — three variants on one site, one variant on another.
        records = [
            ("bHLH_1:100-349_binding_0", "ACGT"),
            ("bHLH_1:100-349_binding_1", "ACGT"),
            ("bHLH_1:100-349_non_binding_2", "ACGT"),
            ("bHLH_1:500-749_binding_3", "ACGT"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = _write_fasta(tmpdir, records)

            # Act
            sites = parse_fasta_sites(fasta_path)

        # Assert
        self.assertEqual(len(sites), 2)
        self.assertEqual(
            sites[0],
            Site(site_id="bHLH_1:100-349", tf="bHLH", chrom="1", start=100, end=349),
        )
        self.assertEqual(
            sites[1],
            Site(site_id="bHLH_1:500-749", tf="bHLH", chrom="1", start=500, end=749),
        )

    def test_rejects_malformed_header(self):
        # Arrange
        records = [("bHLH-bad-header", "ACGT")]
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = _write_fasta(tmpdir, records)

            # Act & Assert
            with self.assertRaises(ValueError):
                parse_fasta_sites(fasta_path)


class TestOverlapsForSitePlusStrandPromoter(unittest.TestCase):
    def test_matches_worked_example_from_plan(self):
        # Arrange — the WRKY worked example: site 1:8049090-8049339, gene
        # AT1G22740 at TSS 8049089 on + strand. Promoter window is
        # [8048089, 8049589) and the site sits fully inside it.
        site = Site(
            site_id="WRKY_1:8049090-8049339",
            tf="WRKY",
            chrom="1",
            start=8049090,
            end=8049339,
        )
        # Gene body must be at least 2 * intragenic = 1000 bp long, otherwise
        # find_start_end triggers the short-gene branch and additional_padding
        # > 0; here we want the normal-gene path.
        genes_df = _make_genes_df(
            [
                {
                    "chromosome": "1",
                    "start": 8049089,
                    "end": 8060000,
                    "strand": "+",
                    "gene_id": "AT1G22740_gene",
                }
            ]
        )

        # Act
        rows = overlaps_for_site(site, genes_df)

        # Assert
        self.assertEqual(
            rows,
            [
                {
                    "site_id": "WRKY_1:8049090-8049339",
                    "gene_id": "AT1G22740",
                    "strand": "+",
                    "region": "promoter",
                    "start": 1001,
                    "end": 1251,
                    "additional_padding": 0,
                }
            ],
        )


class TestOverlapsForSiteMinusStrand(unittest.TestCase):
    def test_minus_strand_promoter_overlap_reverses_position(self):
        # Arrange — gene [2000, 3000) on - strand: promoter window
        # = [2500, 4000), terminator window = [1000, 2500). Site genomic
        # [2550, 2649] sits fully inside the promoter window. On the minus
        # strand the extracted position is mirrored:
        #   rel(2550) = prom_end - 1 - 2550 = 3999 - 2550 = 1449
        #   rel(2649) = 3999 - 2649 = 1350
        # so the half-open output interval is [1350, 1450) (length 100).
        site = Site(
            site_id="bHLH_1:2550-2649",
            tf="bHLH",
            chrom="1",
            start=2550,
            end=2649,
        )
        genes_df = _make_genes_df(
            [
                {
                    "chromosome": "1",
                    "start": 2000,
                    "end": 3000,
                    "strand": "-",
                    "gene_id": "AT1GTEST1_gene",
                }
            ]
        )

        # Act
        rows = overlaps_for_site(site, genes_df)

        # Assert
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["strand"], "-")
        self.assertEqual(rows[0]["region"], "promoter")
        self.assertEqual(rows[0]["start"], 1350)
        self.assertEqual(rows[0]["end"], 1450)
        self.assertEqual(rows[0]["end"] - rows[0]["start"], 100)
        self.assertEqual(rows[0]["additional_padding"], 0)
        self.assertEqual(rows[0]["gene_id"], "AT1GTEST1")


class TestOverlapsForSiteClippedAtWindowEdge(unittest.TestCase):
    def test_partial_overlap_is_clipped_to_window(self):
        # Arrange — site genomic [8500, 9499] is 1000 bp long, but the gene's
        # promoter window [9000, 10500) only contains the right half. The
        # emitted row should reflect just the in-window portion (length 500).
        site = Site(
            site_id="bHLH_1:8500-9499",
            tf="bHLH",
            chrom="1",
            start=8500,
            end=9499,
        )
        genes_df = _make_genes_df(
            [
                {
                    "chromosome": "1",
                    "start": 10000,
                    "end": 11000,
                    "strand": "+",
                    "gene_id": "AT1GCLIP_gene",
                }
            ]
        )

        # Act
        rows = overlaps_for_site(site, genes_df)

        # Assert
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["region"], "promoter")
        self.assertEqual(rows[0]["start"], 0)
        self.assertEqual(rows[0]["end"], 500)
        self.assertLess(rows[0]["end"] - rows[0]["start"], site.end - site.start + 1)
        self.assertEqual(rows[0]["additional_padding"], 0)


class TestOverlapsForSiteShortGene(unittest.TestCase):
    def test_short_gene_emits_two_rows_with_padded_terminator(self):
        # Arrange — gene [1000, 1100) on + strand has length 100, triggering
        # the short-gene branch: longer=shorter=50, additional_padding=900.
        # Promoter window = [0, 1050), terminator window = [1050, 2100).
        # A 401 bp site spanning [900, 1300] inclusive hits both windows.
        site = Site(
            site_id="bHLH_1:900-1300",
            tf="bHLH",
            chrom="1",
            start=900,
            end=1300,
        )
        genes_df = _make_genes_df(
            [
                {
                    "chromosome": "1",
                    "start": 1000,
                    "end": 1100,
                    "strand": "+",
                    "gene_id": "AT1GSHORT_gene",
                }
            ]
        )

        # Act
        rows = overlaps_for_site(site, genes_df)

        # Assert
        self.assertEqual(len(rows), 2)
        by_region = {row["region"]: row for row in rows}
        self.assertIn("promoter", by_region)
        self.assertIn("terminator", by_region)
        # Both rows share the same gene-level provenance.
        for row in rows:
            self.assertEqual(row["gene_id"], "AT1GSHORT")
            self.assertEqual(row["strand"], "+")
            self.assertEqual(row["additional_padding"], 900)

        # Promoter row: rel = genomic_pos - 0 = genomic_pos for + strand.
        prom_row = by_region["promoter"]
        self.assertEqual(prom_row["start"], 900)
        self.assertEqual(prom_row["end"], 1050)

        # Terminator row: rel = 1050 + 20 + 900 + (genomic_pos - 1050) =
        # 920 + genomic_pos. The 1970 start sits well past the normal
        # 1520 terminator offset, confirming the padding is applied.
        term_row = by_region["terminator"]
        self.assertEqual(term_row["start"], 1970)
        self.assertEqual(term_row["end"], 2221)
        self.assertGreater(term_row["start"], 1520)


class TestOverlapsForSiteMultipleGenes(unittest.TestCase):
    def test_one_site_emits_one_row_per_overlapping_gene(self):
        # Arrange — site [5000, 5249] sits inside the promoter window of two
        # neighbouring + strand genes whose TSSes are 100 bp apart.
        site = Site(
            site_id="bHLH_1:5000-5249",
            tf="bHLH",
            chrom="1",
            start=5000,
            end=5249,
        )
        genes_df = _make_genes_df(
            [
                {
                    "chromosome": "1",
                    "start": 5100,
                    "end": 6100,
                    "strand": "+",
                    "gene_id": "AT1GMULTI_A_gene",
                },
                {
                    "chromosome": "1",
                    "start": 5200,
                    "end": 6200,
                    "strand": "+",
                    "gene_id": "AT1GMULTI_B_gene",
                },
            ]
        )

        # Act
        rows = overlaps_for_site(site, genes_df)

        # Assert
        self.assertEqual(len(rows), 2)
        by_gene = {row["gene_id"]: row for row in rows}
        self.assertEqual(by_gene["AT1GMULTI_A"]["start"], 900)
        self.assertEqual(by_gene["AT1GMULTI_A"]["end"], 1150)
        self.assertEqual(by_gene["AT1GMULTI_B"]["start"], 800)
        self.assertEqual(by_gene["AT1GMULTI_B"]["end"], 1050)
        for row in rows:
            self.assertEqual(row["region"], "promoter")
            self.assertEqual(row["additional_padding"], 0)

    def test_genes_on_other_chromosomes_are_ignored(self):
        # Arrange
        site = Site(
            site_id="bHLH_1:5000-5249",
            tf="bHLH",
            chrom="1",
            start=5000,
            end=5249,
        )
        genes_df = _make_genes_df(
            [
                {
                    "chromosome": "2",
                    "start": 5100,
                    "end": 6100,
                    "strand": "+",
                    "gene_id": "AT2GOFF_gene",
                }
            ]
        )

        # Act
        rows = overlaps_for_site(site, genes_df)

        # Assert
        self.assertEqual(rows, [])


class TestBuildMappingEndToEnd(unittest.TestCase):
    def test_writes_seven_column_csv_via_patched_find_genes(self):
        # Arrange — two unique sites in FASTA (one with a duplicate variant
        # entry that must be deduplicated), three genes in the patched gene
        # table. We expect the worked-example WRKY row plus the
        # minus-strand row from the dedicated tests above.
        records = [
            ("WRKY_1:8049090-8049339_binding_0", "ACGT"),
            ("WRKY_1:8049090-8049339_binding_1", "ACGT"),  # duplicate site
            ("bHLH_1:2550-2649_binding_2", "ACGT"),
        ]
        patched_genes_df = _make_genes_df(
            [
                {
                    "chromosome": "1",
                    "start": 8049089,
                    "end": 8060000,
                    "strand": "+",
                    "gene_id": "AT1G22740_gene",
                },
                {
                    "chromosome": "1",
                    "start": 2000,
                    "end": 3000,
                    "strand": "-",
                    "gene_id": "AT1GTEST1_gene",
                },
                {
                    "chromosome": "2",
                    "start": 5000,
                    "end": 6000,
                    "strand": "+",
                    "gene_id": "AT2GFAR_gene",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = _write_fasta(tmpdir, records)
            output_path = os.path.join(tmpdir, "out.csv")

            # Act
            with patch.object(gbo, "find_genes", return_value=patched_genes_df) as mocked:
                df = build_mapping(fasta_path, gtf_path="ignored-by-mock")
                df.to_csv(output_path, index=False)

            # Assert — find_genes was called once with the keyword arguments
            # the plan specifies.
            mocked.assert_called_once_with(
                annotation_path="ignored-by-mock",
                gene_name_attribute="gene_id",
                feature_type_filter=["gene"],
                genes_of_interest=[],
            )

            self.assertEqual(list(df.columns), OUTPUT_COLUMNS)
            self.assertEqual(len(df), 2)  # one row per unique site x overlap

            # Each emitted row's region matches the window that was hit and
            # the strand matches the patched gene table.
            wrky_row = df[df["site_id"] == "WRKY_1:8049090-8049339"].iloc[0]
            self.assertEqual(wrky_row["gene_id"], "AT1G22740")
            self.assertEqual(wrky_row["strand"], "+")
            self.assertEqual(wrky_row["region"], "promoter")
            self.assertEqual(int(wrky_row["start"]), 1001)
            self.assertEqual(int(wrky_row["end"]), 1251)
            self.assertEqual(int(wrky_row["additional_padding"]), 0)

            bhlh_row = df[df["site_id"] == "bHLH_1:2550-2649"].iloc[0]
            self.assertEqual(bhlh_row["gene_id"], "AT1GTEST1")
            self.assertEqual(bhlh_row["strand"], "-")
            self.assertEqual(bhlh_row["region"], "promoter")
            self.assertEqual(int(bhlh_row["start"]), 1350)
            self.assertEqual(int(bhlh_row["end"]), 1450)
            self.assertEqual(int(bhlh_row["additional_padding"]), 0)

            # Round-trip through CSV preserves the same shape.
            reloaded = pd.read_csv(output_path)
            self.assertEqual(list(reloaded.columns), OUTPUT_COLUMNS)
            self.assertEqual(len(reloaded), 2)


if __name__ == "__main__":
    unittest.main()
