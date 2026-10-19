"""Cross-check the built library against the raw fronts, independently of build_library.

Re-reads every selected entry from its pareto_front.json and re-derives the screen, the
numbering, the deduplication and the reference pointers, then compares them against the
written table, FASTA and upload sheet. Prints every failed check. Run after build_library.
"""
import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, List

import pandas as pd

from workflows.starrseq_v2.selection import build_library as bl

IDENTIFIER = re.compile(r"^(?P<genome>[^_]+)_(?P<chromosome>[^_]+)_(?P<setup>[^_]+)_(?P<author>[^_]+)_(?P<serial>\d{6})$")

failures: List[str] = []


def check(condition: bool, message: str) -> None:
    """Record a failed check."""
    if not condition:
        failures.append(message)


def raw_sequences(runs_root: str, run_dirs: List[str]) -> Dict[str, List[str]]:
    """The sequence of every front entry of each run, by front index."""
    fronts = {}
    for run_dir in run_dirs:
        path = os.path.join(runs_root, run_dir, "saved_populations", "pareto_front.json")
        with open(path) as f:
            fronts[run_dir] = [entry[0].upper() for entry in json.load(f)]
    return fronts


def read_fasta(path: str) -> Dict[str, str]:
    """Identifier to sequence."""
    records: Dict[str, str] = {}
    name = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
            elif name is not None:
                records[name] = line
    return records


def check_sequences_against_the_fronts(table: pd.DataFrame, fronts: Dict[str, List[str]]) -> None:
    """Every written sequence must be the raw front entry it claims to come from."""
    wrong = sum(
        fronts[row["run_dir"]][row["front_index"]] != row["sequence"]
        for row in table.to_dict("records")
    )
    check(wrong == 0, f"{wrong} sequences do not match their raw front entry")
    lengths = {len(sequence) for sequence in table["sequence"]}
    check(lengths == {bl.SEQUENCE_LENGTH}, f"sequence lengths present: {sorted(lengths)}")


def check_the_screen(selected: pd.DataFrame, fronts: Dict[str, List[str]], table: pd.DataFrame) -> None:
    """Re-derive which entries the screen should have kept, from the raw sequences."""
    carries = {
        (row["run_dir"], row["front_index"]): bl.has_restriction_site(
            fronts[row["run_dir"]][row["front_index"]]
        )
        for row in selected.to_dict("records")
    }
    references = {
        row["run_dir"]: row["front_index"]
        for row in selected.to_dict("records")
        if row["selection_reason"].startswith("reference")
    }
    check(
        len(references) == selected["run_dir"].nunique(),
        "not every selected run contributes a reference",
    )
    unusable = {run_dir for run_dir, index in references.items() if carries[(run_dir, index)]}
    expected = {key for key, hit in carries.items() if not hit and key[0] not in unusable}
    written = set(zip(table["run_dir"], table["front_index"]))
    check(expected == written, f"screen kept {len(written)} entries, expected {len(expected)}")
    still_carrying = sum(bl.has_restriction_site(sequence) for sequence in table["sequence"])
    check(still_carrying == 0, f"{still_carrying} written sequences still carry a site")


def check_numbering(table: pd.DataFrame) -> None:
    """Numbers must be 1..N in the deterministic sort order, and match the identifier serial."""
    check(list(table["sequence_number"]) == list(range(1, len(table) + 1)), "numbers are not 1..N")
    ordered = table.sort_values(bl.SORT_KEY, kind="mergesort")
    check(
        list(ordered["sequence_number"]) == list(table["sequence_number"]),
        "rows are not in the deterministic sort order",
    )
    bad_serial = 0
    bad_field = 0
    for identifier, number in zip(table["identifier"], table["sequence_number"]):
        match = IDENTIFIER.match(identifier)
        if match is None:
            bad_field += 1
            continue
        if int(match.group("serial")) != number or match.group("author") != bl.AUTHOR:
            bad_serial += 1
        if match.group("genome") not in bl.GENOME_BY_SPECIES.values():
            bad_field += 1
    check(bad_field == 0, f"{bad_field} identifiers do not have five fields with a known genome")
    check(bad_serial == 0, f"{bad_serial} identifiers carry the wrong serial or author")
    wrong_setup = sum(
        f".f{int(row['front_index']):02d}.m{int(row['n_mutations']):02d}." not in row["identifier"]
        for row in table.to_dict("records")
    )
    check(wrong_setup == 0, f"{wrong_setup} identifiers carry the wrong front index or mutation count")


def check_deduplication(table: pd.DataFrame) -> None:
    """Each group of identical sequences must be represented by its lowest-numbered entry."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for sequence, number in zip(table["sequence"], table["sequence_number"]):
        groups[sequence].append(number)
    expected = {
        number: min(numbers) for numbers in groups.values() for number in numbers
    }
    written = dict(zip(table["sequence_number"], table["representative_number"]))
    check(expected == written, "representatives are not the lowest-numbered entry of their group")
    representatives = table.loc[table["is_representative"]]
    check(
        len(representatives) == len(groups),
        f"{len(representatives)} representatives for {len(groups)} distinct sequences",
    )
    identifier_by_number = dict(zip(table["sequence_number"], table["identifier"]))
    mismatched = sum(
        identifier_by_number[row["representative_number"]] != row["representative_id"]
        for row in table.to_dict("records")
    )
    check(mismatched == 0, f"{mismatched} representative_id values do not match their number")


def check_reference_pointers(table: pd.DataFrame) -> None:
    """Every entry must point at the 0-mutation entry of its own run, and at its representative."""
    by_run = {
        row["run_dir"]: row
        for row in table.to_dict("records")
        if row["selection_reason"].startswith("reference")
    }
    wrong_id = 0
    wrong_representative = 0
    for row in table.to_dict("records"):
        reference = by_run.get(row["run_dir"])
        if reference is None:
            wrong_id += 1
            continue
        if reference["identifier"] != row["reference_id"]:
            wrong_id += 1
        if reference["representative_id"] != row["reference_representative_id"]:
            wrong_representative += 1
    check(wrong_id == 0, f"{wrong_id} entries point at the wrong reference")
    check(wrong_representative == 0, f"{wrong_representative} entries point at the wrong representative")


def check_reference_is_the_zero_mutation_entry(table: pd.DataFrame, front_table: pd.DataFrame) -> None:
    """The entry each run calls its reference must be the one with 0 mutations."""
    counts = dict(zip(zip(front_table["run_dir"], front_table["front_index"]), front_table["n_mutations"]))
    references = table.loc[table["selection_reason"].str.startswith("reference")]
    wrong = sum(
        counts[(row["run_dir"], row["front_index"])] != 0 for row in references.to_dict("records")
    )
    check(wrong == 0, f"{wrong} reference rows do not have 0 mutations")


def check_fasta(table: pd.DataFrame, path: str) -> None:
    """The FASTA must hold exactly the representatives."""
    records = read_fasta(path)
    representatives = table.loc[table["is_representative"]]
    expected = dict(zip(representatives["identifier"], representatives["sequence"]))
    check(records == expected, f"FASTA holds {len(records)} records, expected {len(expected)}")


def check_upload_sheet(table: pd.DataFrame, path: str) -> None:
    """The sheet must hold the representatives in order."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    representatives = table.loc[table["is_representative"]].to_dict("records")
    check(rows[0] == ["Number", "id", "reference", "product", "path_to_config"], "header changed")
    check(len(rows) - 1 == len(representatives), f"{len(rows) - 1} sheet rows for {len(representatives)} representatives")
    identifiers = {row["identifier"] for row in table.to_dict("records") if row["is_representative"]}
    wrong = 0
    for position, (written, entry) in enumerate(zip(rows[1:], representatives), start=1):
        if written != [
            str(position), entry["identifier"], entry["reference_representative_id"],
            entry["sequence"], entry["path_to_config"],
        ]:
            wrong += 1
        if written[2] not in identifiers:
            wrong += 1
    check(wrong == 0, f"{wrong} sheet rows disagree with the table")


def check_config_paths(table: pd.DataFrame) -> None:
    """Every config path must name a file that exists at its source."""
    sources = bl.config_source_paths(table["config_name"].unique(), bl.CONFIGS)
    missing = [path for path in sources.values() if not os.path.isfile(path)]
    check(not missing, f"{len(missing)} config source files do not exist")
    expected = {bl.CONFIG_PREFIX + name + ".json" for name in sources}
    check(set(table["path_to_config"]) == expected, "path_to_config values do not match the configs")


def main() -> None:
    """Re-derive everything and report."""
    output = bl.OUTPUT
    table = pd.read_csv(os.path.join(output, "library_table.csv"))
    selected = pd.read_csv(os.path.join(output, "selected_keys.csv"))
    front_table = pd.read_csv(  # pyright: ignore[reportCallIssue]
        os.path.join(output, "front_table.csv.gz"),
        float_precision="round_trip",  # pyright: ignore[reportArgumentType]
    )
    fronts = raw_sequences(bl.RUNS, sorted(set(selected["run_dir"])))

    check_sequences_against_the_fronts(table, fronts)
    check_the_screen(selected, fronts, table)
    check_numbering(table)
    check_deduplication(table)
    check_reference_pointers(table)
    check_reference_is_the_zero_mutation_entry(table, front_table)
    check_fasta(table, os.path.join(output, "library.fasta"))
    check_upload_sheet(table, os.path.join(output, "library_upload.csv"))
    check_config_paths(table)

    print(f"entries {len(table)}, representatives {int(table['is_representative'].sum())}")
    print(f"runs {table['run_dir'].nunique()}, genes {table['gene_id'].nunique()}")
    if failures:
        print(f"\n{len(failures)} FAILED CHECKS")
        for message in failures:
            print(f"  {message}")
    else:
        print("\nall checks passed")


if __name__ == "__main__":
    main()
