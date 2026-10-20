"""Build the raw STARR-seq v2 upload sheet from the selected front entries.

Fetches the sequence of every selected entry, numbers the entries and writes the analysis
table and the upload sheet, one row per entry. Restriction screening and deduplication
happen afterwards in `qa.check_upload_sheets`, pooled with the other contributors' sheets;
its full table maps every `identifier` here to the oligo that measures it.
"""
import json
import os
from typing import Any, Dict, Iterable, Tuple

import pandas as pd

from workflows.starrseq_v2.qa.check_upload_sheets import UPLOAD_COLUMNS

RUNS = "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/starrseq_v2/full_run"
CONFIGS = "/home/gernot/Code/PhD_Code/Evolution/configs/starrseq_v2"
OUTPUT = os.path.join(os.path.dirname(__file__), "output")

AUTHOR = "GS"
CONFIG_PREFIX = "configs_gernot/"
GENOME_BY_SPECIES = {"arabidopsis": "TAIR10", "ntab": "Ntab-TN90"}
# bins/ holds a second, byte-identical copy of all 68 configs, partitioned into batches
CONFIG_FOLDERS = ["baseline", "arms", "grid", "starrseq_v1"]
# unique, since (run_dir, front_index) keys the front table; the leading fields only group
SORT_KEY = ["config_name", "gene_id", "run_dir", "front_index"]


def load_json(path: str) -> Any:
    """Read one JSON file."""
    with open(path) as f:
        return json.load(f)


def flatten_field(value: str) -> str:
    """Make a value safe for one underscore-separated field of the identifier."""
    return value.replace("_", "-").replace("|", "+")


def parse_record_name(record_name: str, gene_id: str) -> Tuple[str, str]:
    """Split a record name into its chromosome and the gene's genomic span.

    Args:
        record_name: `<chromosome>_<gene_id>_gene:<start>-<end>`, optionally with the
            `_off_<start>` suffix of an off-site run. Chromosome names contain underscores
            and dots in N. tabacum, so the gene id is used to find the split point.

    Returns:
        The chromosome, and the span as it appears in the name — reversed for genes on the
        minus strand, which is how the extraction encodes the strand.
    """
    base = record_name.rsplit("_off_", 1)[0]
    left, span = base.rsplit(":", 1)
    suffix = f"_{gene_id}_gene"
    if not left.endswith(suffix):
        raise ValueError(f"record name {record_name!r} does not carry gene id {gene_id!r}")
    return left[: -len(suffix)], span


def build_identifier(entry: Dict[str, Any], serial: int) -> str:
    """The readable identifier of one entry, in the upload template's five-field form."""
    chromosome, span = parse_record_name(entry["record_name"], entry["gene_id"])
    setup = ".".join([
        flatten_field(entry["gene_id"]),
        flatten_field(entry["config_name"]),
        f"f{int(entry['front_index']):02d}",
        f"m{int(entry['n_mutations']):02d}",
        flatten_field(entry["selection_reason"]),
    ])
    genome = GENOME_BY_SPECIES[entry["species"]]
    return f"{genome}_{flatten_field(chromosome)}:{span}_{setup}_{AUTHOR}_{serial:06d}"


def attach_sequences(selected: pd.DataFrame, runs_root: str) -> pd.DataFrame:
    """Add the 170 bp sequence of every selected entry, reading each front once."""
    sequences = []
    for run_dir, rows in selected.groupby("run_dir", sort=False):
        front = load_json(os.path.join(runs_root, str(run_dir), "saved_populations", "pareto_front.json"))
        sequences.append(pd.Series(
            [front[index][0].upper() for index in rows["front_index"]], index=rows.index
        ))
    return selected.assign(sequence=pd.concat(sequences))


def check_references_selected(table: pd.DataFrame) -> None:
    """Every run must contribute exactly one reference; the reference pointers rely on it."""
    references = table.loc[table["is_reference"], "run_dir"]
    if references.duplicated().any():
        raise ValueError("a run contributes more than one reference")
    without = set(table["run_dir"]) - set(references)
    if without:
        raise ValueError(f"{len(without)} runs have no selected reference")


def assign_numbers(table: pd.DataFrame) -> pd.DataFrame:
    """Sort deterministically, number the entries from 1 and build their identifiers."""
    ordered = table.sort_values(SORT_KEY, kind="mergesort").reset_index(drop=True)
    ordered["sequence_number"] = ordered.index + 1
    ordered["identifier"] = [
        build_identifier(entry, serial)
        for entry, serial in zip(ordered.to_dict("records"), ordered["sequence_number"])
    ]
    return ordered


def add_reference_pointers(table: pd.DataFrame) -> pd.DataFrame:
    """Add `reference_id`, the identifier of each entry's run reference."""
    references = table.loc[table["is_reference"]]
    identifier_by_run = dict(zip(references["run_dir"], references["identifier"]))
    return table.assign(reference_id=[identifier_by_run[run_dir] for run_dir in table["run_dir"]])


def config_source_paths(config_names: Iterable[str], configs_root: str) -> Dict[str, str]:
    """Locate one config file per config name, outside the duplicated bins/ copies.

    Raises:
        FileNotFoundError: If a config name has no file.
    """
    wanted = set(config_names)
    found: Dict[str, str] = {}
    for folder in CONFIG_FOLDERS:
        directory = os.path.join(configs_root, folder)
        for file_name in sorted(os.listdir(directory)):
            name, extension = os.path.splitext(file_name)
            if extension == ".json" and name in wanted:
                found.setdefault(name, os.path.join(directory, file_name))
    missing = wanted - set(found)
    if missing:
        raise FileNotFoundError(f"no config file for {sorted(missing)}")
    return found


def write_upload_sheet(table: pd.DataFrame, destination: str) -> None:
    """Write every entry as one row of the five-column upload CSV."""
    pd.DataFrame({
        "Number": list(table["sequence_number"]),
        "id": list(table["identifier"]),
        "reference": list(table["reference_id"]),
        "product": list(table["sequence"]),
        "path_to_config": list(table["path_to_config"]),
    }).reindex(columns=UPLOAD_COLUMNS).to_csv(destination, index=False)


def build_library(selected: pd.DataFrame, front_table: pd.DataFrame, runs_root: str) -> pd.DataFrame:
    """Run the whole step from the selected keys to one numbered row per entry."""
    table = selected.merge(front_table, on=["run_dir", "front_index"], how="left", validate="one_to_one")
    table = attach_sequences(table, runs_root)
    check_references_selected(table)
    table = assign_numbers(table)
    table = add_reference_pointers(table)
    table["path_to_config"] = CONFIG_PREFIX + table["config_name"] + ".json"
    return table


def main() -> None:
    """Build the sheet from the step-2 output and write the table, the sheet and the config list to OUTPUT."""
    selected = pd.read_csv(os.path.join(OUTPUT, "selected_keys.csv"))
    front_table = pd.read_csv(  # pyright: ignore[reportCallIssue]
        os.path.join(OUTPUT, "front_table.csv.gz"),
        float_precision="round_trip",  # pyright: ignore[reportArgumentType]
    )
    table = build_library(selected, front_table, RUNS)

    table.to_csv(os.path.join(OUTPUT, "library_table.csv"), index=False)
    write_upload_sheet(table, os.path.join(OUTPUT, "library_upload.csv"))
    sources = config_source_paths(table["config_name"].unique(), CONFIGS)
    with open(os.path.join(OUTPUT, "configs_to_copy.txt"), "w") as f:
        for name in sorted(sources):
            f.write(f"{sources[name]}\n")

    print(f"entries {len(table)} from {table['run_dir'].nunique()} runs")
    print(f"configs to copy into {CONFIG_PREFIX}: {len(sources)}")


if __name__ == "__main__":
    main()
