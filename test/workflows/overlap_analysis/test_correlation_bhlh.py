"""Unit tests for ``starrseq_deepcre_correlation_bHLH``.

The filesystem (``pyfaidx.Fasta``), the deepCRE model (``load_model``),
the sequence encoder (``one_hot_encode``), and the extraction subprocess are
all mocked so the bHLH-specific glue can be exercised without heavy I/O.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from workflows.overlap_analysis import starrseq_deepcre_correlation_bHLH as bhlh


class _FakeRecord:
    """Minimal stand-in for a ``pyfaidx`` record (``.name`` + ``str()``)."""

    def __init__(self, name: str, sequence: str) -> None:
        self.name = name
        self._sequence = sequence

    def __str__(self) -> str:
        return self._sequence


class TestEnsureRefsFasta(unittest.TestCase):
    def test_short_circuits_when_refs_already_exist(self):
        # Arrange: an existing refs FASTA.
        with tempfile.TemporaryDirectory() as temp_dir:
            refs_path = os.path.join(temp_dir, "refs.fa")
            with open(refs_path, "w") as fh:
                fh.write(">already_here\nACGT\n")
            genes_json_path = os.path.join(temp_dir, "genes.json")

            # Act
            with patch.object(bhlh, "subprocess") as mock_subprocess:
                bhlh.ensure_refs_fasta(
                    refs_path, genes_json_path, "unused.csv", "genome.fa", "ann.gtf"
                )

            # Assert: no extraction, no genes JSON written.
            mock_subprocess.run.assert_not_called()
            self.assertFalse(os.path.exists(genes_json_path))

    def test_runs_extraction_and_writes_genes_json_when_absent(self):
        # Arrange: a mapping CSV with a duplicate gene id; no refs/genes yet.
        with tempfile.TemporaryDirectory() as temp_dir:
            refs_path = os.path.join(temp_dir, "refs.fa")
            genes_json_path = os.path.join(temp_dir, "genes.json")
            mapping_csv_path = os.path.join(temp_dir, "mapping.csv")
            pd.DataFrame(
                {
                    "site_id": ["bHLH_1:1-2", "bHLH_1:1-2", "bHLH_1:3-4"],
                    "gene_id": ["AT1G00002", "AT1G00001", "AT1G00001"],
                }
            ).to_csv(mapping_csv_path, index=False)

            # Act
            with patch.object(bhlh, "subprocess") as mock_subprocess:
                bhlh.ensure_refs_fasta(
                    refs_path, genes_json_path, mapping_csv_path, "genome.fa", "ann.gtf"
                )

            # Assert: subprocess invoked with the extract_sequences CLI.
            mock_subprocess.run.assert_called_once()
            command = mock_subprocess.run.call_args[0][0]
            self.assertIn("-m", command)
            self.assertIn("evolution.extract_sequences", command)
            self.assertIn(refs_path, command)
            self.assertIn("genome.fa", command)
            self.assertIn("ann.gtf", command)
            self.assertIn(genes_json_path, command)
            self.assertTrue(mock_subprocess.run.call_args[1]["check"])

            # Assert: genes JSON holds the sorted, de-duplicated gene ids.
            with open(genes_json_path) as fh:
                written_genes = json.load(fh)
            self.assertEqual(written_genes, ["AT1G00001", "AT1G00002"])


class TestLoadBhlhWindowCandidates(unittest.TestCase):
    def test_parses_both_strands_and_fills_ref_fitness(self):
        # Arrange: one + strand (start<end) and one - strand (start>end) header.
        records = [
            _FakeRecord("1_AT1G19040_gene:6575393-6578695", "ACGTACGT"),
            _FakeRecord("1_AT1G23140_gene:8204500-8203000", "TTTTGGGG"),
        ]
        fake_model = Mock()
        fake_model.predict.return_value = np.array([[0.11], [0.22]])

        # Act
        with patch.object(bhlh, "Fasta", return_value=records), \
                patch.object(bhlh, "load_model", return_value=fake_model), \
                patch.object(bhlh, "one_hot_encode", lambda seq: np.zeros(4)):
            gene_data = bhlh.load_bhlh_window_candidates("refs.fa", "model.h5", additional_padding_map=pd.DataFrame({"additional_padding": [5, 10]}, index=["AT1G19040", "AT1G23140"]))

        # Assert: gene ids, coordinates (minus strand swapped to start<=end),
        # ref sequences, and batched ref_fitness values.
        self.assertEqual(len(gene_data), 2)

        plus_gene = gene_data[0]
        self.assertEqual(plus_gene["gene"], "AT1G19040")
        self.assertEqual(plus_gene["start"], 6575393)
        self.assertEqual(plus_gene["end"], 6578695)
        self.assertEqual(plus_gene["ref_seq"], "ACGTACGT")
        self.assertEqual(plus_gene["ref_fitness"], 0.11)
        self.assertEqual(plus_gene["additional_padding"], 5)

        minus_gene = gene_data[1]
        self.assertEqual(minus_gene["gene"], "AT1G23140")
        self.assertEqual(minus_gene["start"], 8203000)
        self.assertEqual(minus_gene["end"], 8204500)
        self.assertLessEqual(minus_gene["start"], minus_gene["end"])
        self.assertEqual(minus_gene["ref_fitness"], 0.22)
        self.assertEqual(minus_gene["additional_padding"], 10)


class TestBuildSiteToGeneIds(unittest.TestCase):
    def test_dedups_duplicate_site_gene_rows(self):
        # Arrange: a short gene yields a duplicated (site_id, gene_id) row.
        mapping_candidates = pd.DataFrame(
            [
                {"site_id": "bHLH_1:100-200", "gene_id": "AT1G00001"},
                {"site_id": "bHLH_1:100-200", "gene_id": "AT1G00001"},
                {"site_id": "bHLH_1:100-200", "gene_id": "AT1G00002"},
                {"site_id": "bHLH_2:300-400", "gene_id": "AT2G00001"},
            ]
        )

        # Act
        site_to_gene_ids = bhlh.build_site_to_gene_ids(mapping_candidates)

        # Assert
        self.assertEqual(site_to_gene_ids["bHLH_1:100-200"], ["AT1G00001", "AT1G00002"])
        self.assertEqual(site_to_gene_ids["bHLH_2:300-400"], ["AT2G00001"])


class TestBhlhMappingResults(unittest.TestCase):
    def test_passes_padding_map_and_site_genes_to_the_aligner(self):
        # Arrange: one site with two candidate genes, each with its own padding.
        mapping_candidates = pd.DataFrame(
            [
                {"site_id": "bHLH_1:100-200", "gene_id": "AT1G00001",
                 "additional_padding": 0},
                {"site_id": "bHLH_1:100-200", "gene_id": "AT1G00002",
                 "additional_padding": 12},
            ]
        )
        gene_data = [{"gene": "AT1G00001"}]
        alignments = [{"starr_full_name": "frag_1", "gene": "AT1G00001"}]

        # Act
        with patch.object(bhlh, "ensure_refs_fasta"), \
                patch.object(bhlh.pd, "read_csv", return_value=mapping_candidates), \
                patch.object(
                    bhlh, "load_bhlh_window_candidates", return_value=gene_data,
                ) as loader, \
                patch.object(
                    bhlh._common, "load_starrseq_data", return_value=["entry"],
                ), \
                patch.object(
                    bhlh._common, "map_starrseq_to_deepcre", return_value=alignments,
                ) as mapper:
            results = bhlh.bhlh_mapping_results("model.h5")

        # Assert
        self.assertEqual(results, alignments)
        padding_map = loader.call_args[0][2]
        self.assertEqual(padding_map.loc["AT1G00002", "additional_padding"], 12)
        passed_starrseq, passed_genes, passed_sites = mapper.call_args[0]
        self.assertEqual(passed_starrseq, ["entry"])
        self.assertEqual(passed_genes, gene_data)
        self.assertEqual(
            passed_sites, {"bHLH_1:100-200": ["AT1G00001", "AT1G00002"]}
        )


if __name__ == "__main__":
    unittest.main()
