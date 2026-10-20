"""Cross-check the built library against the raw fronts, independently of build_library.

Re-reads every selected entry from its pareto_front.json and re-derives the numbering and
the reference pointers, then compares them against the written table and upload sheet.
Prints every failed check. Run after build_library.
"""
import csv
import json
import os
import re
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


def check_sequences_against_the_fronts(table: pd.DataFrame, fronts: Dict[str, List[str]]) -> None:
    """Every written sequence must be the raw front entry it claims to come from."""
    wrong = sum(
        fronts[row["run_dir"]][row["front_index"]] != row["sequence"]
        for row in table.to_dict("records")
    )
    check(wrong == 0, f"{wrong} sequences do not match their raw front entry")


def check_every_selected_entry_written(selected: pd.DataFrame, table: pd.DataFrame) -> None:
    """Nothing is dropped before the pooled QA, so the table must hold exactly the selected keys."""
    references = {
        row["run_dir"] for row in selected.to_dict("records")
        if row["selection_reason"].startswith("reference")
    }
    check(
        len(references) == selected["run_dir"].nunique(),
        "not every selected run contributes a reference",
    )
    expected = set(zip(selected["run_dir"], selected["front_index"]))
    written = set(zip(table["run_dir"], table["front_index"]))
    check(expected == written and len(table) == len(selected),
          f"table holds {len(table)} entries, expected {len(selected)}")


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


def check_reference_pointers(table: pd.DataFrame) -> None:
    """Every entry must point at the reference entry of its own run."""
    by_run = {
        row["run_dir"]: row["identifier"]
        for row in table.to_dict("records")
        if row["selection_reason"].startswith("reference")
    }
    wrong = sum(by_run.get(row["run_dir"]) != row["reference_id"] for row in table.to_dict("records"))
    check(wrong == 0, f"{wrong} entries point at the wrong reference")


def check_reference_is_the_zero_mutation_entry(table: pd.DataFrame, front_table: pd.DataFrame) -> None:
    """The entry each run calls its reference must be the one with 0 mutations."""
    counts = dict(zip(zip(front_table["run_dir"], front_table["front_index"]), front_table["n_mutations"]))
    references = table.loc[table["selection_reason"].str.startswith("reference")]
    wrong = sum(
        counts[(row["run_dir"], row["front_index"])] != 0 for row in references.to_dict("records")
    )
    check(wrong == 0, f"{wrong} reference rows do not have 0 mutations")


def check_upload_sheet(table: pd.DataFrame, path: str) -> None:
    """The sheet must hold every entry in order."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    entries = table.to_dict("records")
    check(rows[0] == ["Number", "id", "reference", "product", "path_to_config"], "header changed")
    check(len(rows) - 1 == len(entries), f"{len(rows) - 1} sheet rows for {len(entries)} entries")
    identifiers = set(table["identifier"])
    wrong = 0
    for position, (written, entry) in enumerate(zip(rows[1:], entries), start=1):
        if written != [
            str(position), entry["identifier"], entry["reference_id"],
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
    check_every_selected_entry_written(selected, table)
    check_numbering(table)
    check_reference_pointers(table)
    check_reference_is_the_zero_mutation_entry(table, front_table)
    check_upload_sheet(table, os.path.join(output, "library_upload.csv"))
    check_config_paths(table)

    print(f"entries {len(table)}")
    print(f"runs {table['run_dir'].nunique()}, genes {table['gene_id'].nunique()}")
    if failures:
        print(f"\n{len(failures)} FAILED CHECKS")
        for message in failures:
            print(f"  {message}")
    else:
        print("\nall checks passed")


if __name__ == "__main__":
    main()
