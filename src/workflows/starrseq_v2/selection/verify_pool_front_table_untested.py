"""Cross-check every source the pooled front table draws from, independently of pool_front_table.

Sources: config folder name, run folder name, parameters.json, pareto_front.json,
stats_<config>.json, candidate_gene_metadata.csv, the reference and frame FASTAs,
problem_configs.json, and the written output tables. Run after pool_front_table.
"""
import collections
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from evolution.write_run_script import (
    REFERENCE_PATHS,
    STARRSEQ_V2_FIXED,
    WEIGHTS_PER_DIRECTION,
    generate_checkpoint_generations,
    sequences_per_direction,
)
from pyfaidx import Fasta

from workflows.starrseq_v2.selection import pool_front_table as pft

EVOLUTION = "/home/gernot/Code/PhD_Code/Evolution"
V1_FRAMES = "data/starrseq_v2/starrseq_v1_extracted_genes.fa"
TIMESTAMP = r"_\d{6}_\d{6}_\d{6}"
MODEL_DIRS = {"ntab_model": "ntab", "ara_model": "Ara_ssr"}
FAILURES: Dict[str, List[Any]] = collections.defaultdict(list)
COUNTS: Dict[str, int] = collections.Counter()

# (run_dir, front_index, n_mutations, fitness, n_count) re-read from the raw fronts
EntryRow = Tuple[str, int, int, float, int]


def fail(check: str, detail: Any) -> None:
    """Record one failed check."""
    FAILURES[check].append(detail)


def evolution_path(relative_path: str) -> str:
    """Resolve a path relative to the Evolution repo, as the configs store them."""
    return os.path.join(EVOLUTION, relative_path)


def check_config_dirs(config_dirs: List[str], factors_by_name: Dict[str, Dict[str, str]]) -> None:
    """Config folder names parse, are unique per config, and match the analysis folders."""
    n_combinations = 1
    for levels in pft.CONFIG_LEVELS.values():
        n_combinations *= len(levels)
    if n_combinations != len(factors_by_name):
        fail("config name map injective", (n_combinations, len(factors_by_name)))
    config_names = [config_dir.rsplit("_", 3)[0] for config_dir in config_dirs]
    for config_dir, config_name in zip(config_dirs, config_names):
        if not re.fullmatch(".+" + TIMESTAMP, config_dir):
            fail("config dir has 3-field timestamp", config_dir)
        if config_name not in factors_by_name:
            fail("config name parses", config_dir)
    repeated = [name for name, count in collections.Counter(config_names).items() if count > 1]
    if repeated:
        fail("one dir per config name", repeated)
    analysis_dirs = set(os.listdir(pft.ANALYSIS))
    if analysis_dirs != set(config_names):
        fail("analysis dirs == config names", analysis_dirs ^ set(config_names))


def expected_records(factors: Dict[str, str], group_metadata: pd.DataFrame, gene_ids: Set[str]) -> Set[str]:
    """Rebuild the record names a config was written for, as write_run_script.resolve does."""
    reference_path = REFERENCE_PATHS[(factors["background"], factors["gene_sp"], factors["location"])]
    per_direction = sequences_per_direction(evolution_path(reference_path), group_metadata)
    records = per_direction.get(factors["direction"], [])
    subset_column = {"arms": "arm_subset", "grid": "full_grid_subset"}.get(pft.group_of(factors))
    if subset_column is None:
        return set(records)
    allowed = set(group_metadata.loc[group_metadata[subset_column], "gene_id"])
    return {record for record in records if pft.gene_id_from_record(record, gene_ids) in allowed}


def check_parameters(parameters: Dict[str, Any], factors: Dict[str, str], config_dir: str, run_name: str) -> None:
    """parameters.json agrees with the config and run folder names."""
    record_name = parameters["sequence_name"]
    if not re.fullmatch(re.escape(record_name) + TIMESTAMP, run_name):
        fail("run folder == sequence_name + timestamp", run_name)
    if os.path.basename(parameters["base_folder"]) != config_dir:
        fail("base_folder == config dir", (config_dir, parameters["base_folder"]))
    if (len(parameters["models"]) == 1) != (factors["models"] == "single"):
        fail("models", run_name)
    model_dirs = {os.path.basename(os.path.dirname(model["path"])) for model in parameters["models"]}
    if model_dirs != {MODEL_DIRS[factors["model_sp"]]}:
        fail("model_sp", (run_name, model_dirs))
    reference_path = REFERENCE_PATHS[(factors["background"], factors["gene_sp"], factors["location"])]
    if parameters["reference_sequence_initial"] != reference_path:
        fail("background/gene_sp/location -> reference file", run_name)
    if (parameters["vcf_path"] is None) != (factors["mutations"] == "unconstrained"):
        fail("mutations <-> vcf_path", run_name)
    if parameters["vcf_path"] is not None and os.path.basename(parameters["vcf_path"]) != record_name + ".vcf":
        fail("vcf file belongs to record", run_name)
    if parameters["weights"] != WEIGHTS_PER_DIRECTION[factors["direction"]]:
        fail("direction <-> weights", run_name)
    generations = parameters["number_of_generations"]
    if generations != (300 if factors["mutations"] == "natural" else 1000):
        fail("generations", run_name)
    if parameters["checkpoint_generations"] != generate_checkpoint_generations(generations):
        fail("checkpoint generations", run_name)
    for name, value in STARRSEQ_V2_FIXED.items():
        if parameters[name] != value:
            fail("STARRSEQ_V2_FIXED", (run_name, name))
    construct_offsite = factors["background"] == "construct" and factors["location"] == "offsite"
    if ("_off_" in record_name) != construct_offsite:
        fail("_off_ suffix <-> construct+offsite", run_name)
    window = (parameters["mutation_start"], parameters["mutation_end"])
    if factors["background"] == "natural" and factors["location"] == "offsite":
        if window[0] == 780 or window[1] - window[0] != pft.WINDOW_LENGTH:
            fail("natural+offsite window", (run_name, window))
    elif window != (780, 950):
        fail("insert window", (run_name, window))


def check_metadata(
    factors: Dict[str, str],
    source: Tuple[int, int],
    gene_rows: pd.DataFrame,
    group_gene_rows: pd.DataFrame,
    run_name: str,
) -> None:
    """The run agrees with its gene's metadata rows.

    Args:
        gene_rows: All metadata rows of the run's gene.
        group_gene_rows: The subset its config was written from (v1 rows or the rest).
    """
    if group_gene_rows.empty:
        fail("gene in its group's metadata rows", run_name)
    if factors["direction"] not in set(group_gene_rows["direction"]):
        fail("direction among gene's metadata directions", run_name)
    subset_column = {"arms": "arm_subset", "grid": "full_grid_subset"}.get(pft.group_of(factors))
    if subset_column is not None and not bool(group_gene_rows[subset_column].all()):
        fail("gene in group's subset", (run_name, subset_column))
    if factors["location"] == "correct":
        return
    windows = metadata_windows(gene_rows)
    if source not in windows:
        fail("offsite window among gene's metadata windows", run_name)
    elif source not in metadata_windows(group_gene_rows):
        # known: build_starrseq_v2_library splices v1 windows into the ara off-target FASTA too
        COUNTS["offsite runs on a window from another group's metadata rows"] += 1
    if len(windows) > 1:
        COUNTS["offsite runs on a gene with >1 metadata window"] += 1


def metadata_windows(gene_rows: pd.DataFrame) -> Set[Tuple[int, int]]:
    """The full-length off-site windows listed for a gene."""
    usable = gene_rows.loc[gene_rows["full_length_window"]]
    return set(zip(usable["alternative_start"].astype(int), usable["alternative_end"].astype(int)))


def check_front(
    front: List[list],
    stats: Dict[str, Any],
    reference_window: str,
    source_sequence: str,
    run_name: str,
) -> None:
    """The front is well formed, starts from the right sequence, and agrees with its stats.

    Args:
        reference_window: The run's reference sequence cut to its mutation window.
        source_sequence: The natural gene frame cut to the pooled source window.
    """
    counts = [n_mutations for _, _, n_mutations in front]
    references = [entry for entry in front if entry[2] == 0]
    if len(references) != 1:
        fail("exactly one 0-mutation entry", (run_name, len(references)))
    reference_sequence, reference_fitness, _ = references[0]
    if reference_sequence.upper() != reference_window.upper():
        fail("0-mutation entry == reference[mutation window]", run_name)
    if reference_sequence.upper() != source_sequence.upper():
        fail("0-mutation entry == natural frame[source window]", run_name)
    for index, (sequence, _, n_mutations) in enumerate(front):
        if len(sequence) != pft.WINDOW_LENGTH:
            fail("170 bp", (run_name, index))
        if set(sequence) - set("ACGTN"):
            fail("alphabet ACGTN", (run_name, index))
        if n_mutations != int(n_mutations) or not 0 <= n_mutations <= 60:
            fail("mutation count integral in 0..60", (run_name, index))
        if sum(base != reference_base for base, reference_base in zip(sequence, reference_sequence)) != n_mutations:
            COUNTS["entries whose Hamming distance to the reference != n_mutations"] += 1
    COUNTS["fronts with tied mutation counts"] += len(set(counts)) != len(counts)
    COUNTS["fronts with gaps"] += len(set(counts)) != max(counts) + 1
    if stats["start_fitness"] != reference_fitness:
        fail("stats start_fitness == 0-mutation fitness", run_name)
    if stats["max_mutations"] != max(counts):
        fail("stats max_mutations == front max", run_name)
    if stats["final_fitness"] not in [fitness for _, fitness, n_mutations in front if n_mutations == max(counts)]:
        fail("stats final_fitness == fitness at max count", run_name)
    if stats["num_mutations_half_max_effect"] not in counts:
        fail("half-max count in front", run_name)


def check_config(
    config_dir: str,
    factors: Dict[str, str],
    metadata: pd.DataFrame,
    fastas: Dict[str, Fasta],
    fixed_seen: Dict[str, Set[str]],
) -> List[EntryRow]:
    """Run every per-run check on one config and return its raw front entries.

    Args:
        fastas: Opened FASTAs keyed by their path relative to the Evolution repo.
        fixed_seen: Collects every value each fixed parameter takes, updated in place.
    """
    config_name = config_dir.rsplit("_", 3)[0]
    config_path = os.path.join(pft.RUNS, config_dir)
    stats = pft.load_json(os.path.join(pft.ANALYSIS, config_name, f"stats_{config_name}.json"))
    names = sorted(os.listdir(config_path))
    run_names = [name for name in names if os.path.isdir(os.path.join(config_path, name))]
    other_files = set(names) - set(run_names)
    if other_files - {"problem_configs.json", "problem_configs_errors.json"}:
        fail("only problem_configs next to runs", (config_dir, other_files))
    failed_records: Set[str] = set()
    if "problem_configs.json" in other_files:
        problems = pft.load_json(os.path.join(config_path, "problem_configs.json"))
        failed_records = {problem["sequence_name"] for problem in problems}
        COUNTS["runs that never started (problem_configs)"] += len(failed_records)
    if set(stats) != set(run_names):
        fail("stats keys == run folders", (config_name, set(stats) ^ set(run_names)))

    gene_ids = set(metadata["gene_id"])
    is_v1 = metadata["reason"] == "starrseq_v1"
    group_metadata = metadata.loc[is_v1 if factors["gene_sp"] == "v1_genes" else ~is_v1]
    reference_path = REFERENCE_PATHS[(factors["background"], factors["gene_sp"], factors["location"])]
    frame_path = V1_FRAMES if factors["gene_sp"] == "v1_genes" else REFERENCE_PATHS[("natural", factors["gene_sp"], "correct")]
    rows: List[EntryRow] = []
    present_records: List[str] = []
    for run_name in run_names:
        COUNTS["runs"] += 1
        run_path = os.path.join(config_path, run_name)
        parameters = pft.load_json(os.path.join(run_path, "parameters.json"))
        record_name = parameters["sequence_name"]
        present_records.append(record_name)
        check_parameters(parameters, factors, config_dir, run_name)
        for name in pft.FIXED_PARAMETERS:
            fixed_seen[name].add(json.dumps(parameters[name], sort_keys=True))

        gene_id = pft.gene_id_from_record(record_name, gene_ids)
        source_start, source_end = pft.source_window(factors, parameters)
        gene_rows = metadata.loc[metadata["gene_id"] == gene_id]
        group_gene_rows = group_metadata.loc[group_metadata["gene_id"] == gene_id]
        check_metadata(factors, (source_start, source_end), gene_rows, group_gene_rows, run_name)

        reference = str(fastas[reference_path][record_name])
        if parameters["reference_sequence"] != reference:
            fail("parameters reference_sequence == FASTA record", run_name)
        run_fasta = Fasta(os.path.join(run_path, "reference_sequence.fa"))
        if str(run_fasta[list(run_fasta.keys())[0]]) != reference:
            fail("run reference_sequence.fa == FASTA record", run_name)
        frame = str(fastas[frame_path][record_name.split("_off_")[0]])

        front = pft.load_json(os.path.join(run_path, "saved_populations", "pareto_front.json"))
        reference_window = reference[parameters["mutation_start"]:parameters["mutation_end"]]
        check_front(front, stats[run_name], reference_window, frame[source_start:source_end], run_name)
        rows.extend(
            (f"{config_dir}/{run_name}", index, round(n_mutations), fitness, sequence.count("N"))
            for index, (sequence, fitness, n_mutations) in enumerate(front)
        )

    present = set(present_records)
    if len(present) != len(present_records):
        fail("no duplicate record within config", config_name)
    if present & failed_records:
        fail("failed records absent", (config_name, present & failed_records))
    expected = expected_records(factors, group_metadata, gene_ids)
    if expected != present | failed_records:
        fail("present + failed == records the config was written for",
             (config_name, expected - present - failed_records, (present | failed_records) - expected))
    return rows


def check_outputs(rows: List[EntryRow]) -> None:
    """The written tables hold exactly the raw front entries, with consistent derived columns."""
    # the default float parser is not exact; round_trip reads back the written value
    # (valid in pandas 2.0, missing from the bundled stubs)
    table = pd.read_csv(  # pyright: ignore[reportCallIssue]
        os.path.join(pft.OUTPUT, "front_table.csv.gz"),
        float_precision="round_trip",  # pyright: ignore[reportArgumentType]
    )
    overview = pd.read_csv(os.path.join(pft.OUTPUT, "run_overview.csv"))
    raw = pd.DataFrame(rows, columns=pd.Index(["run_dir", "front_index", "n_mutations", "fitness", "n_count"]))
    merged = table.merge(raw, on=["run_dir", "front_index"], how="outer", suffixes=("", "_raw"), indicator=True)
    if (merged["_merge"] != "both").any():
        fail("table keys == raw keys", merged["_merge"].value_counts().to_dict())
    for column in ["n_mutations", "fitness", "n_count"]:
        n_differing = int((merged[column] != merged[column + "_raw"]).sum())
        if n_differing:
            fail(f"table {column} == raw", n_differing)
    if table.duplicated(["run_dir", "front_index"]).any():
        fail("(run_dir, front_index) unique", "")
    if len(overview) != overview["run_dir"].nunique() or set(overview["run_dir"]) != set(table["run_dir"]):
        fail("overview runs == table runs", "")
    if overview.duplicated(["config_name", "record_name"]).any():
        fail("(config_name, record_name) unique", "")
    if (table.loc[table["is_reference"], "fitness_delta_vs_reference"] != 0).any():
        fail("delta 0 on reference rows", "")
    if overview[["gene_id", "species"]].isna().to_numpy().any():
        fail("every run has metadata", "")
    COUNTS["table rows"] = len(table)
    COUNTS["overview rows"] = len(overview)
    COUNTS["rows tied_at_count"] = int(table["tied_at_count"].sum())
    COUNTS["rows with N"] = int((table["n_count"] > 0).sum())
    COUNTS["rows is_half_max"] = int(table["is_half_max"].sum())
    COUNTS["rows is_max_mutations"] = int(table["is_max_mutations"].sum())
    print("configs per group:", overview.drop_duplicates("config_name")["group"].value_counts().to_dict())
    print("runs per group:", overview["group"].value_counts().to_dict())
    print("gene_sp x species:", overview.groupby(["gene_sp", "species"]).size().to_dict())
    print("location x background:", overview.groupby(["location", "background"]).size().to_dict())


def main() -> None:
    """Run all checks and print the counts and every failed check."""
    metadata = pd.read_csv(pft.METADATA)
    n_species = metadata.groupby("gene_id")["species"].nunique()
    if (n_species > 1).any():
        fail("species constant within gene", [gene for gene, n in n_species.items() if n > 1])
    factors_by_name = pft.config_factors()
    config_dirs = sorted(os.listdir(pft.RUNS))
    check_config_dirs(config_dirs, factors_by_name)
    fasta_paths = set(REFERENCE_PATHS.values()) | {V1_FRAMES}
    fastas = {path: Fasta(evolution_path(path)) for path in fasta_paths}
    fixed_seen: Dict[str, Set[str]] = collections.defaultdict(set)
    rows: List[EntryRow] = []
    for config_dir in config_dirs:
        factors = factors_by_name[config_dir.rsplit("_", 3)[0]]
        rows.extend(check_config(config_dir, factors, metadata, fastas, fixed_seen))
    for name, values in fixed_seen.items():
        if len(values) != 1:
            fail("fixed parameter constant", (name, values))
    COUNTS["front entries"] = len(rows)
    check_outputs(rows)

    for key, value in COUNTS.items():
        print(f"{key}: {value}")
    print("\nALL CHECKS PASSED" if not FAILURES else "")
    for check, details in FAILURES.items():
        print(f"FAIL {check}: {len(details)}  e.g. {details[:3]}")


if __name__ == "__main__":
    main()
