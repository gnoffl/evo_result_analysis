"""Clean the pooled STARR-seq v2 upload sheets from all contributors and write the oligo FASTA.

Every sheet has the columns of `UPLOAD_COLUMNS`. The sheets are pooled first, since IDs,
sequences and reference pointers must hold across contributors. Rows failing a check are
dropped with their reasons recorded, and so is every row whose reference was dropped. The
survivors are deduplicated on the exact sequence: duplicates stay in the table, pointing at
their representative, but only representatives are ordered.

`Number` is rewritten to run 1..N over the pooled rows in the order the sheets are given, so
the representative of a duplicate group — the lowest `Number` — is the first occurrence.

Usage:
    python -m workflows.starrseq_v2.qa.check_upload_sheets <configs_root> <out_dir> <sheet.csv>...

`configs_root` is the folder the `path_to_config` values are relative to, i.e. the one
holding `configs_gernot/` and the other contributors' config folders.
"""
import argparse
import os
import re
from typing import List

import pandas as pd

UPLOAD_COLUMNS = ["Number", "id", "reference", "product", "path_to_config"]
FULL_TABLE_COLUMNS = UPLOAD_COLUMNS + ["source_file", "drop_reasons", "representative_id"]
SEQUENCE_LENGTH = 170
# BsaI and its reverse complement, each also matching at the very end of the insert where
# the cloning suffix supplies the final C
RESTRICTION_SITE = re.compile(r"GGTCT(C|$)|GAGAC(C|$)")
BUDGET = 100_000

def has_restriction_site(sequence: str) -> bool:
    """Whether the sequence carries a BsaI site, counting one the cloning suffix would complete."""
    return RESTRICTION_SITE.search(sequence) is not None


def read_sheet(path: str) -> pd.DataFrame:
    """Read a .csv, or the first sheet of an .ods, as strings with NA values turned into ""."""
    if path.endswith(".ods"):
        return pd.read_excel(path, sheet_name=0, engine="odf", dtype=str).fillna("")
    return pd.read_csv(path, dtype=str).fillna("")


def load_sheets(paths: List[str]) -> pd.DataFrame:
    """Concatenate the sheets, record each row's `source_file` and renumber `Number` 1..N.

    Raises:
        ValueError: If a sheet's columns differ from UPLOAD_COLUMNS.
    """
    sheets = []
    for path in paths:
        sheet = read_sheet(path)
        if list(sheet.columns) != UPLOAD_COLUMNS:
            raise ValueError(f"{path} has columns {list(sheet.columns)}")
        sheets.append(sheet.assign(source_file=os.path.splitext(os.path.basename(path))[0]))
    pooled = pd.concat(sheets, ignore_index=True)
    pooled["product"] = pooled["product"].str.upper()
    pooled["Number"] = pooled.index + 1
    return pooled


def is_blank(table: pd.DataFrame, column: str) -> pd.Series:
    """Whether each value is empty or whitespace; `load_sheets` turns NA values into empty strings."""
    return table[column].str.strip() == ""


def own_drop_reasons(table: pd.DataFrame, configs_root: str) -> pd.DataFrame:
    """One boolean column per row-level reason, aligned with table, True where the row fails it."""
    config_found = table["path_to_config"].apply(lambda path: os.path.isfile(os.path.join(configs_root, path)))
    return pd.DataFrame({
        "empty_id": is_blank(table, "id"),
        "empty_reference": is_blank(table, "reference"),
        "whitespace_in_id": table["id"].str.contains(r"\s"),
        "whitespace_in_reference": table["reference"].str.contains(r"\s"),
        "duplicate_id": table["id"].duplicated(keep=False),
        "not_acgt": ~table["product"].str.fullmatch("[ACGT]+"),
        "wrong_length": table["product"].str.len() != SEQUENCE_LENGTH,
        "restriction_site": table["product"].apply(has_restriction_site),
        "reference_missing": ~table["reference"].isin(table["id"]),
        "unresolved_config": ~config_found,
    })


def reference_dropped(table: pd.DataFrame, dropped: pd.Series) -> pd.Series:
    """Rows not dropped themselves whose chain of references reaches a dropped row."""
    lost = dropped.copy()
    while True:
        newly_lost = table["reference"].isin(table.loc[lost, "id"]) & ~lost
        if not newly_lost.any():
            return lost & ~dropped
        lost |= newly_lost


def drop_reason_flags(table: pd.DataFrame, configs_root: str) -> pd.DataFrame:
    """All drop reasons as boolean columns, the reference cascade included."""
    flags = own_drop_reasons(table, configs_root)
    dropped = pd.Series(flags.to_numpy().any(axis=1), index=flags.index)
    flags["reference_dropped"] = reference_dropped(table, dropped)
    return flags


def join_reasons(flags: pd.DataFrame) -> pd.Series:
    """The `|`-joined names of the reasons each row fails, empty for kept rows."""
    names = list(flags.columns)
    return pd.Series(
        ["|".join(name for name, hit in zip(names, row) if hit) for row in flags.to_numpy()],
        index=flags.index,
    )


def assign_representatives(table: pd.DataFrame, kept: pd.Series) -> pd.Series:
    """The id of the lowest-numbered kept row with the same product, empty for dropped rows."""
    survivors = table.loc[kept]
    lowest = survivors.groupby("product")["Number"].transform("min")
    id_by_number = dict(zip(survivors["Number"], survivors["id"]))
    representatives = pd.Series([id_by_number[number] for number in lowest], index=survivors.index)
    return representatives.reindex(table.index, fill_value="")


def validate(full_table: pd.DataFrame) -> None:
    """Raise if numbering, deduplication or the reference cascade left an inconsistency."""
    if list(full_table["Number"]) != list(range(1, len(full_table) + 1)):
        raise ValueError("Number is not 1..N without gaps or duplicates")
    kept = full_table.loc[full_table["drop_reasons"] == ""]
    representatives = kept.loc[kept["id"] == kept["representative_id"]]
    if kept["product"].apply(has_restriction_site).any():
        raise ValueError("kept rows have a BsaI site")
    if representatives["product"].duplicated().any():
        raise ValueError("representative sequences are not unique")
    if not kept["representative_id"].isin(representatives["id"]).all():
        raise ValueError("a kept row points at a representative that is not ordered")
    if not kept["reference"].isin(kept["id"]).all():
        raise ValueError("a kept row points at a dropped reference")


def clean(table: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Add `drop_reasons` and `representative_id` to the pooled table and validate the result.

    Args:
        flags: The output of `drop_reason_flags` for table.
    """
    reasons = join_reasons(flags)
    full_table = table.assign(
        drop_reasons=reasons,
        representative_id=assign_representatives(table, reasons == ""),
    ).reindex(columns=FULL_TABLE_COLUMNS)
    validate(full_table)
    return full_table


def write_fasta(table: pd.DataFrame, path: str) -> None:
    """Write one record per row, headed by its id."""
    with open(path, "w") as f:
        for identifier, sequence in zip(table["id"], table["product"]):
            f.write(f">{identifier}\n{sequence}\n")


def write_outputs(full_table: pd.DataFrame, out_dir: str) -> None:
    """Write the full table, one removed-rows CSV per source file and the representatives' FASTA."""
    full_table.to_csv(os.path.join(out_dir, "library_full_table.csv"), index=False)
    removed = full_table.loc[full_table["drop_reasons"] != ""]
    for source_file in full_table["source_file"].unique():
        rows = removed.loc[removed["source_file"] == source_file]
        rows.to_csv(os.path.join(out_dir, f"removed_{source_file}.csv"), index=False)
    representatives = full_table.loc[full_table["id"] == full_table["representative_id"]]
    write_fasta(representatives, os.path.join(out_dir, "library.fasta"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean the pooled STARR-seq v2 upload sheets.")
    parser.add_argument("configs_root")
    parser.add_argument("out_dir")
    parser.add_argument("sheets", nargs="+")
    arguments = parser.parse_args()
    return arguments


def main() -> None:
    """Clean the pooled sheets, print the drop counts per source file and write all outputs."""
    arguments = parse_args()
    table = load_sheets(arguments.sheets)
    flags = drop_reason_flags(table, arguments.configs_root)
    full_table = clean(table, flags)
    write_outputs(full_table, arguments.out_dir)

    print(f"{len(table)} rows from {len(arguments.sheets)} sheets")
    print("rows dropped per reason (a row can fail several):")
    print(flags.groupby(table["source_file"]).sum().T.to_string())
    kept = full_table["drop_reasons"] == ""
    print(f"kept rows: {int(kept.sum())}")
    n_representatives = int((full_table["id"] == full_table["representative_id"]).sum())
    print(f"oligos to order: {n_representatives} of the {BUDGET} budget")


if __name__ == "__main__":
    main()
