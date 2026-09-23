"""Tests for the Arabidopsis -> N. tabacum flowering-gene ortholog search."""

import unittest

from src.workflows.starrseq_v2.flowering_data.find_nicotiana_orthologs import (
    best_hits,
    longest_per_gene,
)


class TestLongestPerGene(unittest.TestCase):
    def test_keeps_longest_isoform_of_wanted_genes(self):
        proteins = {"AT1G01010.1": "MMM", "AT1G01010.2": "MMMMM", "AT2G00000.1": "MM"}

        result = longest_per_gene(proteins, ["AT1G01010"])

        self.assertEqual(result, {"AT1G01010": "MMMMM"})

    def test_keeps_every_gene_when_no_list_is_given(self):
        proteins = {"Ntab_1.1": "MMM", "Ntab_2.1": "MM"}

        result = longest_per_gene(proteins)

        self.assertEqual(result, {"Ntab_1": "MMM", "Ntab_2": "MM"})


class TestBestHits(unittest.TestCase):
    def test_keeps_only_the_first_hit_per_query(self):
        tsv = (
            "AT1G01010\tNtab_1\t70.0\t1e-50\t300\n"
            "AT1G01010\tNtab_2\t65.0\t1e-40\t250\n"
            "AT2G00000\tNtab_3\t40.0\t1e-20\t120\n"
        )

        result = best_hits(tsv)

        self.assertEqual(result, {"AT1G01010": "Ntab_1", "AT2G00000": "Ntab_3"})

    def test_drops_hits_below_identity_or_above_evalue(self):
        tsv = (
            "AT1G01010\tNtab_1\t29.9\t1e-50\t300\n"
            "AT2G00000\tNtab_2\t80.0\t1e-5\t250\n"
        )

        result = best_hits(tsv)

        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
