"""Tests for building the STARR-seq v2 oligo library from the selected front entries."""
import json
import os
import tempfile
import unittest
from typing import Any, Dict, List

import pandas as pd

from workflows.starrseq_v2.qa.check_upload_sheets import SEQUENCE_LENGTH, UPLOAD_COLUMNS
from workflows.starrseq_v2.selection.build_library import (
    add_reference_pointers,
    assign_numbers,
    attach_sequences,
    build_identifier,
    build_library,
    check_references_selected,
    config_source_paths,
    parse_record_name,
    write_upload_sheet,
)

ARA_RECORD = "4_AT4G24540_gene:12674106-12670896"
# the chromosome itself carries underscores and a dot, and repeats inside the gene id
NTAB_RECORD = "NW_015791515.1_Nicotiana_tabacum_NW_015791515.1_000002_gene:14248-7986"


def sequence_of(prefix: str) -> str:
    """A 170 bp ACGT sequence beginning with prefix."""
    return (prefix + "ACGT" * SEQUENCE_LENGTH)[:SEQUENCE_LENGTH]


def entry(**overrides: Any) -> Dict[str, Any]:
    """One front-table row with sensible defaults."""
    row = {
        "run_dir": "config_a_261008_000000_000000/4_AT4G24540_gene:12674106-12670896_261008_000000_000000",
        "front_index": 0,
        "selection_reason": "reference",
        "gene_id": "AT4G24540",
        "config_name": "config_a",
        "record_name": ARA_RECORD,
        "species": "arabidopsis",
        "is_reference": True,
        "n_mutations": 0,
        "sequence": sequence_of("ACGT"),
    }
    row.update(overrides)
    return row


def frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """A front table from a list of rows."""
    return pd.DataFrame(rows)


class TestParseRecordName(unittest.TestCase):
    def test_arabidopsis(self) -> None:
        self.assertEqual(parse_record_name(ARA_RECORD, "AT4G24540"), ("4", "12674106-12670896"))

    def test_chromosome_with_underscores_and_dots(self) -> None:
        chromosome, span = parse_record_name(NTAB_RECORD, "NW_015791515.1_000002")
        self.assertEqual(chromosome, "NW_015791515.1_Nicotiana_tabacum")
        self.assertEqual(span, "14248-7986")

    def test_offsite_suffix_is_stripped(self) -> None:
        self.assertEqual(
            parse_record_name(f"{ARA_RECORD}_off_440", "AT4G24540"), ("4", "12674106-12670896")
        )

    def test_span_order_is_preserved_for_the_minus_strand(self) -> None:
        """The reversed span is how the extraction encodes the strand, so it must survive."""
        _, span = parse_record_name("1_AT1G10300_gene:3380414-3377877", "AT1G10300")
        self.assertEqual(span, "3380414-3377877")

    def test_gene_id_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_record_name(ARA_RECORD, "AT1G99999")


class TestBuildIdentifier(unittest.TestCase):
    def test_arabidopsis_identifier(self) -> None:
        identifier = build_identifier(
            entry(front_index=12, n_mutations=37, selection_reason="random"), 123
        )
        self.assertEqual(
            identifier,
            "TAIR10_4:12674106-12670896_AT4G24540.config-a.f12.m37.random_GS_000123",
        )

    def test_mutation_count_is_two_digits_zero_padded(self) -> None:
        self.assertIn(".m00.", build_identifier(entry(n_mutations=0), 1))
        self.assertIn(".m07.", build_identifier(entry(n_mutations=7), 1))
        self.assertIn(".m60.", build_identifier(entry(n_mutations=60), 1))

    def test_entries_tied_at_one_count_differ_only_in_front_index(self) -> None:
        """Two front entries can share a mutation count; the front index still separates them."""
        first = build_identifier(entry(front_index=21, n_mutations=37), 1)
        second = build_identifier(entry(front_index=22, n_mutations=37), 2)
        self.assertIn(".f21.m37.", first)
        self.assertIn(".f22.m37.", second)
        self.assertNotEqual(first, second)

    def test_serial_is_six_digits_zero_padded(self) -> None:
        self.assertTrue(build_identifier(entry(), 7).endswith("_GS_000007"))

    def test_underscores_and_bars_are_flattened(self) -> None:
        identifier = build_identifier(
            entry(config_name="ensemble_ara_model_maximize", selection_reason="half_max|max"), 1
        )
        self.assertIn("ensemble-ara-model-maximize", identifier)
        self.assertIn("half-max+max", identifier)

    def test_ntab_identifier_keeps_five_underscore_fields(self) -> None:
        identifier = build_identifier(
            entry(
                record_name=NTAB_RECORD,
                gene_id="NW_015791515.1_000002",
                species="ntab",
                front_index=3,
                n_mutations=8,
                selection_reason="half_max",
            ),
            456,
        )
        self.assertEqual(
            identifier,
            "Ntab-TN90_NW-015791515.1-Nicotiana-tabacum:14248-7986"
            "_NW-015791515.1-000002.config-a.f03.m08.half-max_GS_000456",
        )
        self.assertEqual(len(identifier.split("_")), 5)


class TestCheckReferencesSelected(unittest.TestCase):
    def test_passes_with_one_reference_per_run(self) -> None:
        check_references_selected(frame([
            entry(run_dir="a", front_index=0),
            entry(run_dir="a", front_index=1, is_reference=False),
            entry(run_dir="b", front_index=0),
        ]))

    def test_raises_when_a_run_has_no_reference(self) -> None:
        with self.assertRaises(ValueError):
            check_references_selected(frame([entry(run_dir="a", is_reference=False)]))


class TestAssignNumbers(unittest.TestCase):
    def test_numbers_follow_the_deterministic_sort(self) -> None:
        """Gene and config are crossed on purpose, so this pins which key sorts first."""
        table = frame([
            entry(gene_id="AT2G00002", config_name="config_a", front_index=1),
            entry(gene_id="AT1G00001", config_name="config_b", front_index=0),
            entry(gene_id="AT2G00002", config_name="config_a", front_index=0),
            entry(gene_id="AT1G00001", config_name="config_a", front_index=0),
        ])
        # the identifier is built from the gene id, so keep the record name consistent
        table["record_name"] = ["4_" + gene + "_gene:1-2" for gene in table["gene_id"]]
        ordered = assign_numbers(table)
        self.assertEqual(list(ordered["sequence_number"]), [1, 2, 3, 4])
        self.assertEqual(
            list(zip(ordered["config_name"], ordered["gene_id"], ordered["front_index"])),
            [
                ("config_a", "AT1G00001", 0),
                ("config_a", "AT2G00002", 0),
                ("config_a", "AT2G00002", 1),
                ("config_b", "AT1G00001", 0),
            ],
        )

    def test_rerun_reproduces_the_same_numbers(self) -> None:
        table = frame([entry(front_index=i, is_reference=i == 0) for i in range(5)])
        first = assign_numbers(table)
        second = assign_numbers(table.iloc[::-1].reset_index(drop=True))
        self.assertEqual(list(first["identifier"]), list(second["identifier"]))


class TestReferencePointers(unittest.TestCase):
    def test_variant_points_at_its_own_runs_reference(self) -> None:
        table = assign_numbers(frame([
            entry(run_dir="a", front_index=0),
            entry(run_dir="b", front_index=0),
            entry(run_dir="b", front_index=1, is_reference=False),
        ]))
        result = add_reference_pointers(table)
        reference_b = result[(result["run_dir"] == "b") & (result["front_index"] == 0)].iloc[0]
        variant = result[result["front_index"] == 1].iloc[0]
        self.assertEqual(variant["reference_id"], reference_b["identifier"])

    def test_reference_points_at_itself(self) -> None:
        result = add_reference_pointers(assign_numbers(frame([entry(run_dir="a", front_index=0)])))
        self.assertEqual(result["reference_id"].iloc[0], result["identifier"].iloc[0])


class TestConfigSourcePaths(unittest.TestCase):
    def test_finds_configs_and_ignores_the_bins_copy(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            for folder in ["baseline", "arms", "grid", "starrseq_v1", "bins"]:
                os.makedirs(os.path.join(root, folder))
            open(os.path.join(root, "baseline", "config_a.json"), "w").close()
            open(os.path.join(root, "grid", "config_b.json"), "w").close()
            open(os.path.join(root, "baseline", "parameter_overview.json"), "w").close()
            open(os.path.join(root, "bins", "config_a.json"), "w").close()
            found = config_source_paths(["config_a", "config_b"], root)
            self.assertEqual(found["config_a"], os.path.join(root, "baseline", "config_a.json"))
            self.assertEqual(found["config_b"], os.path.join(root, "grid", "config_b.json"))

    def test_missing_config_raises(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            for folder in ["baseline", "arms", "grid", "starrseq_v1"]:
                os.makedirs(os.path.join(root, folder))
            with self.assertRaises(FileNotFoundError):
                config_source_paths(["absent"], root)


def write_front(runs_root: str, run_dir: str, entries: List[List[Any]]) -> None:
    """Write one synthetic pareto_front.json."""
    directory = os.path.join(runs_root, run_dir, "saved_populations")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "pareto_front.json"), "w") as f:
        json.dump(entries, f)


class TestAttachSequences(unittest.TestCase):
    def test_reads_the_selected_indices(self) -> None:
        with tempfile.TemporaryDirectory() as runs_root:
            write_front(runs_root, "run_a", [
                [sequence_of("AAAA").lower(), 1.0, 0.0],
                [sequence_of("CCCC"), 2.0, 5.0],
            ])
            selected = pd.DataFrame({"run_dir": ["run_a", "run_a"], "front_index": [1, 0]})
            result = attach_sequences(selected, runs_root)
            self.assertEqual(list(result["sequence"]), [sequence_of("CCCC"), sequence_of("AAAA")])


class TestWriteUploadSheet(unittest.TestCase):
    def test_writes_every_entry_as_the_five_columns_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = os.path.join(directory, "upload.csv")
            table = pd.DataFrame([
                {
                    "sequence_number": number,
                    "identifier": f"ID_{number}",
                    "reference_id": "ID_1",
                    "sequence": sequence_of("ACGT"),
                    "path_to_config": "configs_gernot/config_a.json",
                }
                for number in [1, 2]
            ])
            write_upload_sheet(table, destination)

            written = pd.read_csv(destination, dtype=str)
            self.assertEqual(list(written.columns), UPLOAD_COLUMNS)
            self.assertEqual(
                list(written.iloc[1]),
                ["2", "ID_2", "ID_1", sequence_of("ACGT"), "configs_gernot/config_a.json"],
            )


class TestBuildLibraryEndToEnd(unittest.TestCase):
    # run -> its front, as (sequence, fitness, mutation count)
    FRONTS = {
        "run_good": [(sequence_of("AAAA"), 1.0, 0.0), (sequence_of("CCCC"), 2.0, 4.0)],
        "run_shared": [(sequence_of("AAAA"), 1.0, 0.0), (sequence_of("GGGG"), 2.0, 3.0)],
        "run_bad_reference": [(sequence_of("GGTCTC"), 1.0, 0.0), (sequence_of("TTTT"), 2.0, 6.0)],
    }

    def _tree(self, runs_root: str) -> None:
        for run_dir, front in self.FRONTS.items():
            write_front(runs_root, run_dir, [list(entry) for entry in front])

    def _inputs(self) -> tuple:
        rows = []
        for run_dir, front in self.FRONTS.items():
            for index, (_, _, n_mutations) in enumerate(front):
                rows.append({
                    "run_dir": run_dir,
                    "front_index": index,
                    "selection_reason": "reference" if index == 0 else "random",
                    "gene_id": "AT4G24540",
                    "config_name": "config_a",
                    "record_name": ARA_RECORD,
                    "species": "arabidopsis",
                    "is_reference": index == 0,
                    "n_mutations": int(n_mutations),
                })
        table = pd.DataFrame(rows)
        selected = table[["run_dir", "front_index", "selection_reason"]]
        front_table = table.drop(columns=["selection_reason"])
        return selected, front_table

    def test_every_entry_is_kept_numbered_and_pointed_at_its_reference(self) -> None:
        """Screening and deduplication happen in the pooled QA, so nothing is dropped here."""
        with tempfile.TemporaryDirectory() as runs_root:
            self._tree(runs_root)
            selected, front_table = self._inputs()
            table = build_library(selected, front_table, runs_root)

            self.assertEqual(len(table), 6)
            self.assertEqual(list(table["sequence_number"]), [1, 2, 3, 4, 5, 6])
            self.assertEqual(table["identifier"].nunique(), 6)
            self.assertTrue((table["path_to_config"] == "configs_gernot/config_a.json").all())
            references = table.loc[table["is_reference"]].set_index("run_dir")["identifier"]
            self.assertEqual(list(table["reference_id"]), list(table["run_dir"].map(references)))


if __name__ == "__main__":
    unittest.main()
