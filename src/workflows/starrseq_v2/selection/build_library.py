"""Build the STARR-seq v2 oligo library from the selected front entries.

Fetches the sequence of every selected entry, drops what carries a BsaI site, numbers the
survivors, deduplicates on the exact sequence and writes the analysis table, the upload
sheet and the FASTA.

A run whose reference carries a site loses all of its entries: with no orderable baseline,
none of its variants can be interpreted.

Numbers are assigned before deduplication, so every selected entry stays individually
addressable in its own run context. Several numbers therefore map onto one ordered oligo
and one measurement — join assay results on representative_id, never on sequence_number.
"""
import json
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

RUNS = "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/starrseq_v2/full_run"
CONFIGS = "/home/gernot/Code/PhD_Code/Evolution/configs/starrseq_v2"
OUTPUT = os.path.join(os.path.dirname(__file__), "output")

# BsaI and its reverse complement, each also matching at the very end of the insert where
# the cloning suffix supplies the final C
RESTRICTION_SITE = re.compile(r"GGTCT(C|$)|GAGAC(C|$)")
SEQUENCE_LENGTH = 170
AUTHOR = "GS"
CONFIG_PREFIX = "configs_gernot/"
UPLOAD_COLUMNS = ["Number", "id", "reference", "product", "path_to_config"]
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
    """Every run must contribute exactly one reference; the screen and the pointers rely on it."""
    references = table.loc[table["is_reference"], "run_dir"]
    if references.duplicated().any():
        raise ValueError("a run contributes more than one reference")
    without = set(table["run_dir"]) - set(references)
    if without:
        raise ValueError(f"{len(without)} runs have no selected reference")


def has_restriction_site(sequence: str) -> bool:
    """Whether the sequence carries a BsaI site, counting one the cloning suffix would complete."""
    return RESTRICTION_SITE.search(sequence) is not None


def drop_restriction_sites(table: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop entries carrying a site, and every entry of a run whose reference carries one.

    Returns:
        The survivors, and how many were lost to each cause.
    """
    carries = table["sequence"].apply(has_restriction_site)
    unusable_runs = table.loc[carries & table["is_reference"], "run_dir"].unique()
    from_unusable_run = table["run_dir"].isin(unusable_runs)
    survivors = table.loc[~from_unusable_run & ~carries].copy()
    return survivors, {
        "runs_without_an_orderable_reference": len(unusable_runs),
        "entries_lost_with_their_reference": int(from_unusable_run.sum()),
        "entries_dropped_for_their_own_site": int((carries & ~from_unusable_run).sum()),
    }


def assign_numbers(table: pd.DataFrame) -> pd.DataFrame:
    """Sort deterministically, number the survivors from 1 and build their identifiers."""
    ordered = table.sort_values(SORT_KEY, kind="mergesort").reset_index(drop=True)
    ordered["sequence_number"] = ordered.index + 1
    ordered["identifier"] = [
        build_identifier(entry, serial)
        for entry, serial in zip(ordered.to_dict("records"), ordered["sequence_number"])
    ]
    return ordered


def deduplicate(table: pd.DataFrame) -> pd.DataFrame:
    """Mark one representative per distinct sequence, the one with the lowest number."""
    representative = list(table.groupby("sequence")["sequence_number"].transform("min"))
    identifier_by_number = dict(zip(table["sequence_number"], table["identifier"]))
    return table.assign(
        representative_number=representative,
        representative_id=[identifier_by_number[number] for number in representative],
        is_representative=[
            number == own for number, own in zip(representative, table["sequence_number"])
        ],
    )


def add_reference_pointers(table: pd.DataFrame) -> pd.DataFrame:
    """Point every entry at its run's reference, as that entry and as its representative."""
    references = table.loc[table["is_reference"]]
    identifier_by_run = dict(zip(references["run_dir"], references["identifier"]))
    representative_by_run = dict(zip(references["run_dir"], references["representative_id"]))
    return table.assign(
        reference_id=[identifier_by_run[run_dir] for run_dir in table["run_dir"]],
        reference_representative_id=[representative_by_run[run_dir] for run_dir in table["run_dir"]],
    )


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


def validate(table: pd.DataFrame, representatives: pd.DataFrame) -> None:
    """Raise on anything that would make the library unorderable or the upload sheet inconsistent."""
    lengths = sorted(table["sequence"].str.len().unique())
    if lengths != [SEQUENCE_LENGTH]:
        raise ValueError(f"sequence lengths present: {lengths}")
    if not all(table["sequence"].str.fullmatch("[ACGT]+")):
        raise ValueError("sequences outside the ACGT alphabet")
    if table["sequence_number"].duplicated().any() or table["identifier"].duplicated().any():
        raise ValueError("sequence numbers or identifiers are not unique")
    if representatives["sequence"].duplicated().any():
        raise ValueError("representative sequences are not unique")
    dangling = set(table["reference_representative_id"]) - set(representatives["identifier"])
    if dangling:
        raise ValueError(f"{len(dangling)} reference pointers are absent from the upload sheet")


def write_upload_sheet(representatives: pd.DataFrame, destination: str) -> None:
    """Write the representatives as the five-column upload CSV, numbered from 1."""
    pd.DataFrame({
        "Number": range(1, len(representatives) + 1),
        "id": list(representatives["identifier"]),
        "reference": list(representatives["reference_representative_id"]),
        "product": list(representatives["sequence"]),
        "path_to_config": list(representatives["path_to_config"]),
    }, columns=UPLOAD_COLUMNS).to_csv(destination, index=False)


def write_fasta(representatives: pd.DataFrame, path: str) -> None:
    """Write one record per representative sequence."""
    with open(path, "w") as f:
        for entry in representatives.to_dict("records"):
            f.write(f">{entry['identifier']}\n{entry['sequence']}\n")


def build_library(selected: pd.DataFrame, front_table: pd.DataFrame, runs_root: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Run the whole pipeline from the selected keys to the finished table.

    Returns:
        One row per surviving selected entry, and how many entries the screen dropped.
    """
    table = selected.merge(front_table, on=["run_dir", "front_index"], how="left", validate="one_to_one")
    table = attach_sequences(table, runs_root)
    check_references_selected(table)
    table, dropped = drop_restriction_sites(table)
    table = assign_numbers(table)
    table = deduplicate(table)
    table = add_reference_pointers(table)
    table["path_to_config"] = CONFIG_PREFIX + table["config_name"] + ".json"
    return table, dropped


def main() -> None:
    """Build the library from the step-2 output and write all four files to OUTPUT."""
    selected = pd.read_csv(os.path.join(OUTPUT, "selected_keys.csv"))
    front_table = pd.read_csv(  # pyright: ignore[reportCallIssue]
        os.path.join(OUTPUT, "front_table.csv.gz"),
        float_precision="round_trip",  # pyright: ignore[reportArgumentType]
    )
    table, dropped = build_library(selected, front_table, RUNS)
    representatives = table.loc[table["is_representative"]]
    validate(table, representatives)

    table.to_csv(os.path.join(OUTPUT, "library_table.csv"), index=False)
    write_upload_sheet(representatives, os.path.join(OUTPUT, "library_upload.csv"))
    write_fasta(representatives, os.path.join(OUTPUT, "library.fasta"))
    sources = config_source_paths(table["config_name"].unique(), CONFIGS)
    with open(os.path.join(OUTPUT, "configs_to_copy.txt"), "w") as f:
        for name in sorted(sources):
            f.write(f"{sources[name]}\n")

    print(f"selected {len(selected)} entries")
    for cause, count in dropped.items():
        print(f"  {cause}: {count}")
    print(f"surviving entries {len(table)} from {table['run_dir'].nunique()} runs")
    print(f"distinct sequences to order: {len(representatives)}")
    print(f"configs to copy into {CONFIG_PREFIX}: {len(sources)}")


if __name__ == "__main__":
    main()
