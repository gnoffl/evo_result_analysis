"""Unit tests for analysis.blamm.blamm_tfbs_counts."""

import argparse
import math
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from analysis.blamm.blamm_tfbs_counts import (
    BLAMM_GROUP_ID,
    MAX_MUTATIONS,
    SEQUENCE_ID_SEPARATOR,
    VARIANT_MUTATED,
    VARIANT_REFERENCE,
    _parse_mutation_count,
    aggregate_over_genes,
    SELECTION_COLUMNS,
    SKIPPED_COLUMNS,
    build_sequence_records,
    count_motif_hits,
    parse_occurrences,
    run_blamm,
    select_pareto_entry,
    write_fasta,
    write_manifest,
)
from analysis.motives.deepcis_scanner import (
    _get_entry_by_mutation_count,
    _get_max_mutation_entry,
)

MOTIF_METADATA = pd.DataFrame(
    {
        "motif_id": ["MA0001.1", "MA0002.1"],
        "motif_name": ["AlphaTF", "BetaTF"],
        "source_db": ["jaspar.meme", "dap.meme"],
    }
)


GENE_ID = "1_AT1G01010_gene:1-2_251009"


def _occurrence_line(gene: str, variant: str, motif_id: str) -> str:
    """Build one tab-separated line of a blamm occurrences file."""
    sequence_id = f"{gene}{SEQUENCE_ID_SEPARATOR}{variant}"
    return f"{sequence_id}\tblamm\t{motif_id}\t10\t18\t12.5\t+\t.\t."


class SelectParetoEntryTest(unittest.TestCase):
    """Tests for pareto entry selection by mutation count."""

    def setUp(self) -> None:
        """Build a front where the highest mutation count is duplicated."""
        self.pareto_front = [
            ("AAA", 1.0, 7),
            ("CCC", 1.0, 7),
            ("GGG", 0.9, 5),
            ("TTT", 0.8, 0),
        ]

    def test_max_selects_first_of_tied_entries(self) -> None:
        """Ties at the maximum mutation count resolve to the first entry."""
        # Act
        index, entry = select_pareto_entry(self.pareto_front, MAX_MUTATIONS)

        # Assert
        self.assertEqual(index, 0)
        self.assertEqual(entry, ("AAA", 1.0, 7))

    def test_max_agrees_with_deepcis_scanner_helper(self) -> None:
        """Selection matches the established helper in deepcis_scanner."""
        # Act
        _, entry = select_pareto_entry(self.pareto_front, MAX_MUTATIONS)

        # Assert
        self.assertEqual(entry, _get_max_mutation_entry(self.pareto_front))

    def test_exact_count_agrees_with_deepcis_scanner_helper(self) -> None:
        """Exact-count selection matches the established helper."""
        # Act
        index, entry = select_pareto_entry(self.pareto_front, 5)

        # Assert
        self.assertEqual(index, 2)
        self.assertEqual(entry, _get_entry_by_mutation_count(self.pareto_front, 5))

    def test_missing_count_raises(self) -> None:
        """A mutation count absent from the front is an error."""
        # Act / Assert
        with self.assertRaises(ValueError):
            select_pareto_entry(self.pareto_front, 3)

    def test_empty_front_raises(self) -> None:
        """An empty pareto front is an error."""
        # Act / Assert
        with self.assertRaises(ValueError):
            select_pareto_entry([], MAX_MUTATIONS)

    def test_unknown_sentinel_raises(self) -> None:
        """A string other than the MAX sentinel is an error."""
        # Act / Assert
        with self.assertRaises(ValueError):
            select_pareto_entry(self.pareto_front, "SOME_STRING")


class BuildSequenceRecordsTest(unittest.TestCase):
    """Tests for assembling reference/optimized sequence pairs."""

    def _gene_data(self, gene_name: str, pareto_front: list) -> MagicMock:
        """Build a stand-in for a loaded GeneRunData object."""
        gene_data = MagicMock()
        gene_data.gene_name = gene_name
        gene_data.pareto_front = pareto_front
        gene_data.reference_sequence_full = "TTTT"
        gene_data.mutation_start = 1
        gene_data.mutation_end = 3
        return gene_data

    def test_skips_gene_without_requested_mutation_count(self) -> None:
        """Genes lacking the exact mutation count are skipped, not aborted."""
        # Arrange
        good = self._gene_data("gene_ok", [("AA", 0.9, 5), ("TT", 0.5, 0)])
        bad = self._gene_data("gene_missing", [("CC", 0.9, 4), ("TT", 0.5, 0)])

        # Act
        with patch(
            "analysis.blamm.blamm_tfbs_counts.GeneRunData.load_gene_run_data",
            side_effect=[good, bad],
        ):
            records, selections, skipped = build_sequence_records(
                ["/runs/gene_ok", "/runs/gene_missing"], 5
            )

        # Assert
        self.assertEqual(len(records), 2)
        self.assertEqual(list(selections["gene"]), ["gene_ok"])
        self.assertEqual(list(skipped["gene"]), ["gene_missing"])
        self.assertIn("mutation count 5", skipped.loc[0, "reason"])

    def test_splices_mutable_region_into_reference(self) -> None:
        """The optimized sequence is the reference with the region replaced."""
        # Arrange
        gene_data = self._gene_data("gene_ok", [("AA", 0.9, 5)])

        # Act
        with patch(
            "analysis.blamm.blamm_tfbs_counts.GeneRunData.load_gene_run_data",
            return_value=gene_data,
        ):
            records, selections, _ = build_sequence_records(["/runs/gene_ok"], 5)

        # Assert
        identifiers = [record[0] for record in records]
        self.assertEqual(
            identifiers,
            [
                f"gene_ok{SEQUENCE_ID_SEPARATOR}{VARIANT_REFERENCE}",
                f"gene_ok{SEQUENCE_ID_SEPARATOR}{VARIANT_MUTATED}",
            ],
        )
        self.assertEqual(records[0][1], "TTTT")
        self.assertEqual(records[1][1], "TAAT")
        self.assertEqual(selections.loc[0, "selected_mutation_count"], 5)
        self.assertEqual(selections.loc[0, "pareto_index"], 0)
        self.assertEqual(selections.loc[0, "fitness"], 0.9)


    def test_tables_carry_headers_when_nothing_was_skipped(self) -> None:
        """Empty result tables still declare their columns."""
        # Arrange
        gene_data = self._gene_data("gene_ok", [("AA", 0.9, 5)])

        # Act
        with patch(
            "analysis.blamm.blamm_tfbs_counts.GeneRunData.load_gene_run_data",
            return_value=gene_data,
        ):
            _, selections, skipped = build_sequence_records(["/runs/gene_ok"], 5)

        # Assert
        self.assertTrue(skipped.empty)
        self.assertEqual(list(skipped.columns), SKIPPED_COLUMNS)
        self.assertEqual(list(selections.columns), SELECTION_COLUMNS)


class FastaAndManifestTest(unittest.TestCase):
    """Tests for FASTA and manifest writing."""

    def setUp(self) -> None:
        """Create a temporary working directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Remove the temporary working directory."""
        shutil.rmtree(self.temp_dir)

    def test_write_fasta_writes_all_records(self) -> None:
        """Each record becomes a header line and a sequence line."""
        # Arrange
        fasta_path = os.path.join(self.temp_dir, "sequences.fa")

        # Act
        write_fasta(
            [("gene_a__reference", "ACGT"), ("gene_a__mutated", "AGGT")],
            fasta_path,
        )

        # Assert
        with open(fasta_path) as handle:
            content = handle.read()
        self.assertEqual(
            content, ">gene_a__reference\nACGT\n>gene_a__mutated\nAGGT\n"
        )

    def test_write_fasta_rejects_duplicate_identifiers(self) -> None:
        """Duplicate identifiers would make blamm output ambiguous."""
        # Act / Assert
        with self.assertRaises(ValueError):
            write_fasta(
                [("gene_a__reference", "ACGT"), ("gene_a__reference", "AGGT")],
                os.path.join(self.temp_dir, "sequences.fa"),
            )

    def test_write_fasta_rejects_whitespace_in_identifier(self) -> None:
        """blamm truncates descriptors at whitespace, so it is rejected."""
        # Act / Assert
        with self.assertRaises(ValueError):
            write_fasta(
                [("gene a__reference", "ACGT")],
                os.path.join(self.temp_dir, "sequences.fa"),
            )

    def test_write_manifest_uses_single_group(self) -> None:
        """One group means one shared background model."""
        # Arrange
        fasta_path = os.path.join(self.temp_dir, "sequences.fa")
        manifest_path = os.path.join(self.temp_dir, "sequences.mf")

        # Act
        write_manifest(fasta_path, manifest_path)

        # Assert
        with open(manifest_path) as handle:
            content = handle.read()
        self.assertEqual(content, f"{BLAMM_GROUP_ID}\t{fasta_path}\n")


class RunBlammTest(unittest.TestCase):
    """Tests for the blamm command construction."""

    def test_runs_dict_hist_and_scan_in_order(self) -> None:
        """All three blamm steps run in the working directory, in order."""
        # Arrange
        with patch(
            "analysis.blamm.blamm_tfbs_counts.subprocess.run"
        ) as mocked_run:
            # Act
            occurrences_path = run_blamm(
                blamm_executable="/usr/local/bin/blamm",
                motifs_path="/data/motifs.jaspar",
                manifest_path="/work/sequences.mf",
                work_dir="/work",
                p_value=0.0001,
                empirical_histograms=False,
            )

        # Assert
        self.assertEqual(occurrences_path, os.path.join("/work", "occurrences.txt"))
        commands = [call.args[0] for call in mocked_run.call_args_list]
        self.assertEqual([command[1] for command in commands], ["dict", "hist", "scan"])
        self.assertNotIn("-e", commands[1])
        self.assertIn("-rc", commands[2])
        self.assertIn("-pt", commands[2])
        for call in mocked_run.call_args_list:
            self.assertEqual(call.kwargs["cwd"], "/work")
            self.assertTrue(call.kwargs["check"])

    def test_empirical_flag_is_forwarded_to_hist(self) -> None:
        """The empirical option adds -e to the hist step only."""
        # Arrange
        with patch(
            "analysis.blamm.blamm_tfbs_counts.subprocess.run"
        ) as mocked_run:
            # Act
            run_blamm(
                blamm_executable="blamm",
                motifs_path="/data/motifs.jaspar",
                manifest_path="/work/sequences.mf",
                work_dir="/work",
                p_value=0.0001,
                empirical_histograms=True,
            )

        # Assert
        commands = [call.args[0] for call in mocked_run.call_args_list]
        self.assertIn("-e", commands[1])
        self.assertNotIn("-e", commands[2])


class ParseOccurrencesTest(unittest.TestCase):
    """Tests for reading the blamm occurrence table."""

    def setUp(self) -> None:
        """Create a temporary working directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Remove the temporary working directory."""
        shutil.rmtree(self.temp_dir)

    def _write_occurrences(self, lines: list) -> str:
        """Write occurrence lines to a file and return its path."""
        path = os.path.join(self.temp_dir, "occurrences.txt")
        with open(path, "w") as handle:
            handle.write("".join(f"{line}\n" for line in lines))
        return path

    def test_splits_gene_and_variant(self) -> None:
        """The variant suffix is separated from the gene identifier."""
        # Arrange
        path = self._write_occurrences(
            [
                _occurrence_line(GENE_ID, "reference", "MA0001.1"),
                _occurrence_line(GENE_ID, "mutated", "MA0002.1"),
            ]
        )

        # Act
        occurrences = parse_occurrences(path)

        # Assert
        self.assertEqual(
            list(occurrences["gene"].unique()), [GENE_ID]
        )
        self.assertEqual(list(occurrences["variant"]), ["reference", "mutated"])
        self.assertEqual(list(occurrences["motif_id"]), ["MA0001.1", "MA0002.1"])

    def test_empty_file_yields_empty_frame(self) -> None:
        """A scan without hits produces an empty frame, not an error."""
        # Arrange
        path = self._write_occurrences([])

        # Act
        occurrences = parse_occurrences(path)

        # Assert
        self.assertTrue(occurrences.empty)
        self.assertIn("gene", occurrences.columns)
        self.assertIn("variant", occurrences.columns)

    def test_unknown_variant_suffix_raises(self) -> None:
        """A sequence identifier without a known variant suffix is an error."""
        # Arrange
        path = self._write_occurrences(
            [_occurrence_line("gene_a", "something_else", "MA0001.1")]
        )

        # Act / Assert
        with self.assertRaises(ValueError):
            parse_occurrences(path)


class CountMotifHitsTest(unittest.TestCase):
    """Tests for per-gene motif counting."""

    def test_counts_both_variants_and_fills_missing_with_zero(self) -> None:
        """A motif seen in one variant only gets a zero count in the other."""
        # Arrange
        occurrences = pd.DataFrame(
            {
                "gene": ["gene_a", "gene_a", "gene_a", "gene_a"],
                "variant": ["reference", "reference", "mutated", "mutated"],
                "motif_id": ["MA0001.1", "MA0001.1", "MA0001.1", "MA0002.1"],
                "start": [1, 20, 1, 30],
                "end": [8, 27, 8, 37],
                "score": [1.0, 1.0, 1.0, 1.0],
                "strand": ["+", "-", "+", "+"],
            }
        )

        # Act
        counts = count_motif_hits(occurrences, MOTIF_METADATA)

        # Assert
        alpha = counts[counts["motif_id"] == "MA0001.1"].iloc[0]
        beta = counts[counts["motif_id"] == "MA0002.1"].iloc[0]
        self.assertEqual(alpha["count_reference"], 2)
        self.assertEqual(alpha["count_mutated"], 1)
        self.assertEqual(alpha["diff"], -1)
        self.assertEqual(beta["count_reference"], 0)
        self.assertEqual(beta["count_mutated"], 1)
        self.assertEqual(beta["diff"], 1)
        self.assertEqual(beta["motif_name"], "BetaTF")
        self.assertEqual(beta["source_db"], "dap.meme")

    def test_log2_fold_uses_pseudocount(self) -> None:
        """A motif introduced from zero yields a finite log2 fold change."""
        # Arrange
        occurrences = pd.DataFrame(
            {
                "gene": ["gene_a"],
                "variant": ["mutated"],
                "motif_id": ["MA0001.1"],
                "start": [1],
                "end": [8],
                "score": [1.0],
                "strand": ["+"],
            }
        )

        # Act
        counts = count_motif_hits(occurrences, MOTIF_METADATA)

        # Assert
        self.assertTrue(math.isfinite(counts.loc[0, "log2_fold"]))
        self.assertAlmostEqual(counts.loc[0, "log2_fold"], math.log2(2 / 1))


class AggregateOverGenesTest(unittest.TestCase):
    """Tests for aggregation across genes."""

    def setUp(self) -> None:
        """Build per-gene counts for two genes and one motif."""
        self.per_gene_counts = pd.DataFrame(
            {
                "gene": ["gene_a", "gene_b"],
                "motif_id": ["MA0001.1", "MA0001.1"],
                "motif_name": ["AlphaTF", "AlphaTF"],
                "source_db": ["jaspar.meme", "jaspar.meme"],
                "count_reference": [2, 1],
                "count_mutated": [5, 4],
                "diff": [3, 3],
                "log2_fold": [1.0, 1.0],
            }
        )

    def test_sums_counts_and_normalises_by_all_analysed_genes(self) -> None:
        """diff_per_gene divides by every analysed gene, not only hit genes."""
        # Act
        aggregated = aggregate_over_genes(self.per_gene_counts, n_genes_analysed=4)

        # Assert
        self.assertEqual(len(aggregated), 1)
        row = aggregated.iloc[0]
        self.assertEqual(row["total_reference"], 3)
        self.assertEqual(row["total_mutated"], 9)
        self.assertEqual(row["total_diff"], 6)
        self.assertEqual(row["n_genes_analysed"], 4)
        self.assertEqual(row["n_genes_with_hit"], 2)
        self.assertAlmostEqual(row["diff_per_gene"], 1.5)
        self.assertAlmostEqual(row["log2_fold"], math.log2(10 / 4))

    def test_rejects_non_positive_gene_count(self) -> None:
        """A zero denominator is an error rather than a division by zero."""
        # Act / Assert
        with self.assertRaises(ValueError):
            aggregate_over_genes(self.per_gene_counts, n_genes_analysed=0)

    def test_sorted_by_total_difference(self) -> None:
        """Motifs are ordered from most introduced to most removed."""
        # Arrange
        removed = self.per_gene_counts.copy()
        removed["motif_id"] = "MA0002.1"
        removed["motif_name"] = "BetaTF"
        removed["count_reference"] = [9, 9]
        removed["count_mutated"] = [0, 0]
        combined = pd.concat([self.per_gene_counts, removed], ignore_index=True)

        # Act
        aggregated = aggregate_over_genes(combined, n_genes_analysed=2)

        # Assert
        self.assertEqual(list(aggregated["motif_id"]), ["MA0001.1", "MA0002.1"])


class ParseMutationCountTest(unittest.TestCase):
    """Tests for the --mutation-count argument type."""

    def test_accepts_max_sentinel_case_insensitively(self) -> None:
        """Both 'MAX' and 'max' select the maximum mutation count."""
        # Act / Assert
        self.assertEqual(_parse_mutation_count("MAX"), MAX_MUTATIONS)
        self.assertEqual(_parse_mutation_count("max"), MAX_MUTATIONS)

    def test_accepts_non_negative_integer(self) -> None:
        """An integer is returned as an int."""
        # Act / Assert
        self.assertEqual(_parse_mutation_count("5"), 5)

    def test_rejects_negative_and_non_numeric(self) -> None:
        """Invalid values raise an argparse error."""
        # Act / Assert
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_mutation_count("-1")
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_mutation_count("many")


if __name__ == "__main__":
    unittest.main()
