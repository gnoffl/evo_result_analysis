"""Tests for cleaning the pooled STARR-seq v2 upload sheets."""
import os
import tempfile
import unittest
from typing import Any, Dict, List

import pandas as pd

from workflows.starrseq_v2.qa.check_upload_sheets import (
    FULL_TABLE_COLUMNS,
    SEQUENCE_LENGTH,
    UPLOAD_COLUMNS,
    clean,
    drop_reason_flags,
    has_restriction_site,
    load_sheets,
    validate,
    write_fasta,
    write_outputs,
)

CONFIG = "configs_gernot/a.json"


def padded(prefix: str) -> str:
    """prefix extended with A to the full insert length."""
    return prefix + "A" * (SEQUENCE_LENGTH - len(prefix))


def row(identifier: str, reference: str, product: str, **overrides: Any) -> Dict[str, Any]:
    """One upload-sheet row pointing at the default config."""
    entry = {"Number": "1", "id": identifier, "reference": reference, "product": product, "path_to_config": CONFIG}
    return {**entry, **overrides}


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


class SheetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.directory, "configs_gernot"))
        open(os.path.join(self.directory, CONFIG), "w").close()

    def write_sheet(self, name: str, rows: List[Dict[str, Any]]) -> str:
        path = os.path.join(self.directory, name)
        pd.DataFrame(rows)[UPLOAD_COLUMNS].to_csv(path, index=False)
        return path

    def cleaned(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        table = load_sheets([self.write_sheet("sheet.csv", rows)])
        return clean(table, drop_reason_flags(table, self.directory)).set_index("id", drop=False)


class TestLoadSheets(SheetTestCase):
    def test_pools_files_and_renumbers_continuously(self) -> None:
        first = self.write_sheet("first.csv", [row("ref", "ref", padded("AAAA"), Number="1")])
        second = self.write_sheet("second.csv", [
            row("variant", "ref", padded("aatt"), Number="1"),
            row("other", "ref", padded("CCCC"), Number="7"),
        ])
        table = load_sheets([first, second])
        self.assertEqual(list(table["source_file"]), ["first", "second", "second"])
        self.assertEqual(list(table["Number"]), [1, 2, 3])
        self.assertEqual(table["product"].iloc[1], padded("AATT"))

    def test_ods_is_read_from_its_first_sheet(self) -> None:
        path = os.path.join(self.directory, "sheet.ods")
        with pd.ExcelWriter(path, engine="odf") as writer:
            pd.DataFrame([row("ref", "ref", padded("AAAA"))])[UPLOAD_COLUMNS].to_excel(writer, sheet_name="upload", index=False)
            pd.DataFrame({"id": ["ignored"]}).to_excel(writer, sheet_name="notes", index=False)
        table = load_sheets([path])
        self.assertEqual(list(table["id"]), ["ref"])
        self.assertEqual(list(table["source_file"]), ["sheet"])

    def test_wrong_columns_rejected(self) -> None:
        path = os.path.join(self.directory, "bad.csv")
        pd.DataFrame({"id": ["x"]}).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            load_sheets([path])


class TestDropReasons(SheetTestCase):
    def test_clean_rows_are_kept(self) -> None:
        result = self.cleaned([row("ref", "ref", padded("AAAA")), row("variant", "ref", padded("AATT"))])
        self.assertEqual(list(result["drop_reasons"]), ["", ""])

    def test_duplicate_ids_drop_every_copy(self) -> None:
        result = self.cleaned([
            row("ref", "ref", padded("AAAA")),
            row("dup", "ref", padded("CCCC")),
            row("dup", "ref", padded("GGGG")),
        ])
        self.assertEqual(list(result["drop_reasons"]), ["", "duplicate_id", "duplicate_id"])

    def test_each_row_level_reason(self) -> None:
        result = self.cleaned([
            row("ref", "ref", padded("AAAA")),
            row("n_base", "ref", padded("AANA")),
            row("short", "ref", "ACGT"),
            row("site", "ref", padded("aaggtctcaa")),
            row("dangling", "missing", padded("CCCC")),
            row("no_config", "ref", padded("GGGG"), path_to_config="configs_gernot/b.json"),
        ])
        self.assertEqual(result.loc["n_base", "drop_reasons"], "not_acgt")
        self.assertEqual(result.loc["short", "drop_reasons"], "wrong_length")
        self.assertEqual(result.loc["site", "drop_reasons"], "restriction_site")
        self.assertEqual(result.loc["dangling", "drop_reasons"], "reference_missing")
        self.assertEqual(result.loc["no_config", "drop_reasons"], "unresolved_config")

    def test_a_single_blank_id_is_dropped(self) -> None:
        """One blank id is not a duplicate, so it needs its own check."""
        result = self.cleaned([row("ref", "ref", padded("AAAA")), row(" ", "ref", padded("CCCC"))])
        self.assertIn("empty_id", result["drop_reasons"].iloc[1].split("|"))

    def test_whitespace_in_id_and_in_references_to_it(self) -> None:
        result = self.cleaned([
            row("ref", "ref", padded("AAAA")),
            row("bad\tid", "bad\tid", padded("CCCC")),
            row("variant", "bad\tid", padded("GGGG")),
            row("trailing ", "ref", padded("TTTT")),
        ])
        self.assertEqual(list(result["drop_reasons"]), [
            "",
            "whitespace_in_id|whitespace_in_reference",
            "whitespace_in_reference",
            "whitespace_in_id",
        ])

    def test_whitespace_in_reference_is_named_explicitly(self) -> None:
        result = self.cleaned([row("ref", "ref", padded("AAAA")), row("variant", "ref ", padded("CCCC"))])
        self.assertIn("whitespace_in_reference", result["drop_reasons"].iloc[1].split("|"))

    def test_na_values_count_as_blank(self) -> None:
        result = self.cleaned([row("ref", "ref", padded("AAAA")), row("N/A", "ref", padded("CCCC"))])
        self.assertEqual(list(result["drop_reasons"]), ["", "empty_id"])

    def test_blank_reference_is_dropped_even_when_a_blank_id_exists(self) -> None:
        """A blank reference would otherwise resolve to a row with a blank id."""
        result = self.cleaned([
            row("", "", padded("AAAA")),
            row("variant", "", padded("CCCC")),
            row("ref", "ref", padded("GGGG")),
        ])
        self.assertEqual(result["drop_reasons"].iloc[1], "empty_reference")
        self.assertEqual(result["drop_reasons"].iloc[2], "")

    def test_several_reasons_are_joined(self) -> None:
        result = self.cleaned([row("ref", "ref", padded("AAAA")), row("bad", "ref", "GGTCTN")])
        self.assertEqual(result.loc["bad", "drop_reasons"], "not_acgt|wrong_length")

    def test_a_dropped_reference_takes_its_variants_along(self) -> None:
        """A mutation may destroy the site, but the baseline is gone either way."""
        result = self.cleaned([
            row("ref", "ref", padded("GGTCTC")),
            row("variant", "ref", padded("CCCC")),
            row("other_ref", "other_ref", padded("TTTT")),
        ])
        self.assertEqual(list(result["drop_reasons"]), ["restriction_site", "reference_dropped", ""])

    def test_the_cascade_follows_chained_references(self) -> None:
        result = self.cleaned([
            row("ref", "ref", padded("GGTCTC")),
            row("middle", "ref", padded("CCCC")),
            row("end", "middle", padded("TTTT")),
        ])
        self.assertEqual(result.loc["end", "drop_reasons"], "reference_dropped")


class TestDeduplication(SheetTestCase):
    def test_lowest_number_represents_across_files(self) -> None:
        first = self.write_sheet("first.csv", [row("ref", "ref", padded("AAAA"))])
        second = self.write_sheet("second.csv", [
            row("their_ref", "their_ref", padded("CCCC")),
            row("their_copy", "their_ref", padded("aaaa")),
        ])
        table = load_sheets([first, second])
        result = clean(table, drop_reason_flags(table, self.directory))
        self.assertEqual(list(result["representative_id"]), ["ref", "their_ref", "ref"])
        self.assertEqual(list(result["drop_reasons"]), ["", "", ""])

    def test_a_dropped_row_never_represents(self) -> None:
        result = self.cleaned([
            row("dup", "dup", padded("AAAA")),
            row("dup", "dup", padded("CCCC")),
            row("ref", "ref", padded("AAAA")),
        ])
        self.assertEqual(list(result["representative_id"]), ["", "", "ref"])

    def test_columns_follow_the_template(self) -> None:
        result = self.cleaned([row("ref", "ref", padded("AAAA"))])
        self.assertEqual(list(result.columns), FULL_TABLE_COLUMNS)


class TestValidate(SheetTestCase):
    def test_rejects_a_gap_in_the_numbering(self) -> None:
        result = self.cleaned([row("a", "a", padded("AAAA")), row("b", "b", padded("CCCC"))])
        result["Number"] = [1, 3]
        with self.assertRaises(ValueError):
            validate(result)

    def test_rejects_duplicate_representative_sequences(self) -> None:
        result = self.cleaned([row("a", "a", padded("AAAA")), row("b", "b", padded("AAAA"))])
        result["representative_id"] = ["a", "b"]
        with self.assertRaises(ValueError):
            validate(result)

    def test_rejects_a_kept_row_with_a_dropped_reference(self) -> None:
        result = self.cleaned([row("a", "a", padded("AAAA")), row("b", "a", padded("CCCC"))])
        result.loc["a", "drop_reasons"] = "restriction_site"
        result.loc["a", "representative_id"] = ""
        with self.assertRaises(ValueError):
            validate(result)


class TestWriteOutputs(SheetTestCase):
    def test_one_removed_file_per_source_and_representatives_only_in_fasta(self) -> None:
        first = self.write_sheet("first.csv", [row("ref", "ref", padded("AAAA"))])
        second = self.write_sheet("second.csv", [
            row("site", "site", padded("GGTCTC")),
            row("copy", "copy", padded("AAAA")),
        ])
        table = load_sheets([first, second])
        out_dir = os.path.join(self.directory, "out")
        os.makedirs(out_dir)
        write_outputs(clean(table, drop_reason_flags(table, self.directory)), out_dir)

        self.assertTrue(pd.read_csv(os.path.join(out_dir, "removed_first.csv")).empty)
        self.assertEqual(list(pd.read_csv(os.path.join(out_dir, "removed_second.csv"))["id"]), ["site"])
        self.assertEqual(len(pd.read_csv(os.path.join(out_dir, "metadata.csv"))), 3)
        with open(os.path.join(out_dir, "candidates.fa")) as f:
            self.assertEqual(f.read(), f">ref\n{padded('AAAA')}\n")

    def test_write_fasta(self) -> None:
        path = os.path.join(self.directory, "out.fasta")
        write_fasta(pd.DataFrame({"id": ["a", "b"], "product": ["AC", "GT"]}), path)
        with open(path) as f:
            self.assertEqual(f.read(), ">a\nAC\n>b\nGT\n")


if __name__ == "__main__":
    unittest.main()
