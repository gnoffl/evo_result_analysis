"""Tests for building the STARR-seq v2 oligo library from the selected front entries."""
import json
import os
import tempfile
import unittest
from typing import Any, Dict, List

import pandas as pd

from workflows.starrseq_v2.selection.build_library import (
    SEQUENCE_LENGTH,
    SORT_KEY,
    UPLOAD_COLUMNS,
    add_reference_pointers,
    assign_numbers,
    attach_sequences,
    build_identifier,
    build_library,
    check_references_selected,
    config_source_paths,
    deduplicate,
    drop_restriction_sites,
    has_restriction_site,
    parse_record_name,
    validate,
    write_fasta,
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


class TestHasRestrictionSite(unittest.TestCase):
    def test_clean_sequence(self) -> None:
        self.assertFalse(has_restriction_site("ACGTACGTACGT"))

    def test_internal_site_and_its_reverse_complement(self) -> None:
        self.assertTrue(has_restriction_site("ACGGTCTCACG"))
        self.assertTrue(has_restriction_site("ACGAGACCACG"))

    def test_site_completed_by_the_cloning_suffix_at_the_end(self) -> None:
        """Only the trailing C is missing, so the flank completes the site."""
        self.assertTrue(has_restriction_site("ACGTGGTCT"))
        self.assertTrue(has_restriction_site("ACGTGAGAC"))

    def test_partial_site_in_the_middle_is_not_a_hit(self) -> None:
        self.assertFalse(has_restriction_site("ACGTGGTCTAAA"))
        self.assertFalse(has_restriction_site("ACGTGAGACAAA"))


class TestDropRestrictionSites(unittest.TestCase):
    def test_entry_with_its_own_site_is_dropped_alone(self) -> None:
        table = frame([
            entry(front_index=0, sequence=sequence_of("ACGT")),
            entry(front_index=1, is_reference=False, sequence=sequence_of("GGTCTC")),
            entry(front_index=2, is_reference=False, sequence=sequence_of("TTTT")),
        ])
        survivors, dropped = drop_restriction_sites(table)
        self.assertEqual(sorted(survivors["front_index"]), [0, 2])
        self.assertEqual(dropped["entries_dropped_for_their_own_site"], 1)
        self.assertEqual(dropped["runs_without_an_orderable_reference"], 0)

    def test_run_whose_reference_carries_a_site_loses_every_entry(self) -> None:
        table = frame([
            entry(run_dir="bad", front_index=0, sequence=sequence_of("GGTCTC")),
            entry(run_dir="bad", front_index=1, is_reference=False, sequence=sequence_of("TTTT")),
            entry(run_dir="good", front_index=0, sequence=sequence_of("ACGT")),
        ])
        survivors, dropped = drop_restriction_sites(table)
        self.assertEqual(list(survivors["run_dir"]), ["good"])
        self.assertEqual(dropped["runs_without_an_orderable_reference"], 1)
        self.assertEqual(dropped["entries_lost_with_their_reference"], 2)

    def test_a_clean_variant_of_an_unusable_reference_is_still_dropped(self) -> None:
        """A mutation may destroy the site, but the baseline is gone either way."""
        table = frame([
            entry(run_dir="bad", front_index=0, sequence=sequence_of("GGTCTC")),
            entry(run_dir="bad", front_index=1, is_reference=False, sequence=sequence_of("AAAA")),
        ])
        survivors, _ = drop_restriction_sites(table)
        self.assertTrue(survivors.empty)


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


class TestDeduplicate(unittest.TestCase):
    def test_lowest_number_represents_the_group(self) -> None:
        shared = sequence_of("TTTT")
        table = assign_numbers(frame([
            entry(front_index=0, sequence=shared),
            entry(front_index=1, is_reference=False, sequence=sequence_of("GGGG")),
            entry(front_index=2, is_reference=False, sequence=shared),
        ]))
        result = deduplicate(table)
        self.assertEqual(list(result["representative_number"]), [1, 2, 1])
        self.assertEqual(list(result["is_representative"]), [True, True, False])
        self.assertEqual(result["representative_id"].iloc[2], result["identifier"].iloc[0])

    def test_every_entry_keeps_its_own_number(self) -> None:
        """Numbering before deduplication is what keeps duplicate entries addressable."""
        shared = sequence_of("TTTT")
        table = assign_numbers(frame([
            entry(front_index=0, sequence=shared),
            entry(front_index=1, is_reference=False, sequence=shared),
        ]))
        result = deduplicate(table)
        self.assertEqual(list(result["sequence_number"]), [1, 2])
        self.assertEqual(result["identifier"].nunique(), 2)


class TestReferencePointers(unittest.TestCase):
    def test_pointer_resolves_through_deduplication(self) -> None:
        shared = sequence_of("TTTT")
        table = deduplicate(assign_numbers(frame([
            entry(run_dir="a", front_index=0, sequence=shared),
            entry(run_dir="b", front_index=0, sequence=shared),
            entry(run_dir="b", front_index=1, is_reference=False, sequence=sequence_of("GGGG")),
        ])))
        result = add_reference_pointers(table)
        reference_a = result[result["run_dir"] == "a"].iloc[0]
        reference_b = result[(result["run_dir"] == "b") & (result["front_index"] == 0)].iloc[0]
        variant = result[result["front_index"] == 1].iloc[0]
        # run b's own reference is a duplicate of run a's, so the representative pointer leads to run a's entry
        self.assertEqual(variant["reference_id"], reference_b["identifier"])
        self.assertEqual(variant["reference_representative_id"], reference_a["identifier"])

    def test_reference_points_at_itself(self) -> None:
        table = deduplicate(assign_numbers(frame([entry(run_dir="a", front_index=0)])))
        result = add_reference_pointers(table)
        self.assertEqual(result["reference_id"].iloc[0], result["identifier"].iloc[0])


class TestValidate(unittest.TestCase):
    def _complete(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        return add_reference_pointers(deduplicate(assign_numbers(frame(rows))))

    def test_accepts_a_consistent_table(self) -> None:
        table = self._complete([
            entry(front_index=0),
            entry(front_index=1, is_reference=False, sequence=sequence_of("GGGG")),
        ])
        validate(table, table.loc[table["is_representative"]])

    def test_rejects_a_wrong_length(self) -> None:
        table = self._complete([entry()])
        table["sequence"] = "ACGT"
        with self.assertRaises(ValueError):
            validate(table, table)

    def test_rejects_a_non_acgt_base(self) -> None:
        table = self._complete([entry()])
        table["sequence"] = "N" + sequence_of("ACGT")[1:]
        with self.assertRaises(ValueError):
            validate(table, table)

    def test_rejects_a_dangling_reference_pointer(self) -> None:
        table = self._complete([entry()])
        table["reference_representative_id"] = "not_in_the_sheet"
        with self.assertRaises(ValueError):
            validate(table, table.loc[table["is_representative"]])


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
    def test_writes_the_five_columns_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = os.path.join(directory, "upload.csv")
            representatives = pd.DataFrame([{
                "identifier": "ID_ONE",
                "reference_representative_id": "REF_ONE",
                "sequence": sequence_of("ACGT"),
                "path_to_config": "configs_gernot/config_a.json",
            }])
            write_upload_sheet(representatives, destination)

            written = pd.read_csv(destination, dtype=str)
            self.assertEqual(list(written.columns), UPLOAD_COLUMNS)
            self.assertEqual(
                list(written.iloc[0]),
                ["1", "ID_ONE", "REF_ONE", sequence_of("ACGT"), "configs_gernot/config_a.json"],
            )

    def test_numbering_is_contiguous_over_the_representatives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = os.path.join(directory, "upload.csv")
            representatives = pd.DataFrame([
                {
                    "identifier": f"ID_{i}",
                    "reference_representative_id": "REF",
                    "sequence": sequence_of("ACGT"),
                    "path_to_config": "configs_gernot/config_a.json",
                }
                for i in range(3)
            ])
            write_upload_sheet(representatives, destination)
            self.assertEqual(list(pd.read_csv(destination)["Number"]), [1, 2, 3])


class TestWriteFasta(unittest.TestCase):
    def test_one_record_per_representative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "library.fasta")
            write_fasta(pd.DataFrame([
                {"identifier": "ID_ONE", "sequence": sequence_of("AAAA")},
                {"identifier": "ID_TWO", "sequence": sequence_of("CCCC")},
            ]), path)
            lines = open(path).read().splitlines()
            self.assertEqual(lines[0], ">ID_ONE")
            self.assertEqual(lines[1], sequence_of("AAAA"))
            self.assertEqual(lines[2], ">ID_TWO")


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

    def test_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as runs_root:
            self._tree(runs_root)
            selected, front_table = self._inputs()
            table, dropped = build_library(selected, front_table, runs_root)

            self.assertEqual(dropped["runs_without_an_orderable_reference"], 1)
            self.assertEqual(dropped["entries_lost_with_their_reference"], 2)
            self.assertEqual(sorted(table["run_dir"].unique()), ["run_good", "run_shared"])

            # the two runs share their reference sequence, so four entries become three oligos
            self.assertEqual(len(table), 4)
            representatives = table.loc[table["is_representative"]]
            self.assertEqual(len(representatives), 3)
            validate(table, representatives)

            self.assertTrue((table["path_to_config"] == "configs_gernot/config_a.json").all())
            self.assertEqual(list(table["sequence_number"]), [1, 2, 3, 4])

    def test_shared_reference_collapses_but_both_entries_survive(self) -> None:
        with tempfile.TemporaryDirectory() as runs_root:
            self._tree(runs_root)
            selected, front_table = self._inputs()
            table, _ = build_library(selected, front_table, runs_root)
            references = table.loc[table["is_reference"]]
            self.assertEqual(len(references), 2)
            self.assertEqual(references["representative_id"].nunique(), 1)
            self.assertEqual(references["identifier"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
