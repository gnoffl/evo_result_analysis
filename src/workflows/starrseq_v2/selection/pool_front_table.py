"""Pool every final Pareto front of the STARR-seq v2 full run into one overview table."""
import itertools
import json
import os
from typing import Any, Dict, Set, Tuple

import numpy as np
import pandas as pd
from evolution.write_run_script import (
    BASELINE,
    FACTORS,
    STARRSEQ_V2_INSERT_END,
    STARRSEQ_V2_INSERT_START,
    gene_id_from_record,
)

RUNS = "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/starrseq_v2/full_run"
ANALYSIS = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/starrseq_v2/full_run"
METADATA = "/home/gernot/Code/PhD_Code/Evolution/data/starrseq_v2/candidate_gene_metadata.csv"
OUTPUT = os.path.join(os.path.dirname(__file__), "output")

WINDOW_LENGTH = STARRSEQ_V2_INSERT_END - STARRSEQ_V2_INSERT_START
# config names are the factor levels in FACTORS order, then the direction;
# the v1 genes only appear as their own gene species, outside FACTORS
CONFIG_LEVELS = {
    **FACTORS,
    "gene_sp": FACTORS["gene_sp"] + ["v1_genes"],
    "direction": ["maximize", "minimize"],
}
# identical in every run, so written once instead of once per row
FIXED_PARAMETERS = [
    "population_size", "mutation_rate", "mutation_probability", "crossover_probability",
    "tournament_size", "size_of_hall_of_fame", "initial_number_mutations",
    "initial_mutation_strategy", "mutation_strategy", "evaluate_population", "eval_pop_kwargs",
    "max_number_mutations", "max_N", "limit_mutations", "maximize_greedy",
    "sort_before_crossover", "verbose",
]


def load_json(path: str) -> Any:
    """Read one JSON file."""
    with open(path) as f:
        return json.load(f)


def config_factors() -> Dict[str, Dict[str, str]]:
    """Map every possible config name (without timestamp) to its factor levels and direction."""
    return {
        "_".join(levels): dict(zip(CONFIG_LEVELS, levels))
        for levels in itertools.product(*CONFIG_LEVELS.values())
    }


def group_of(factors: Dict[str, str]) -> str:
    """Name the design group a config belongs to, by how far it is from the baseline."""
    if factors["gene_sp"] == "v1_genes":
        return "starrseq_v1"
    # every group runs on both gene species, so gene_sp does not count
    n_differing = sum(factors[factor] != BASELINE[factor] for factor in BASELINE if factor != "gene_sp")
    return {0: "baseline", 1: "arms"}.get(n_differing, "grid")


def source_window(factors: Dict[str, str], parameters: Dict[str, Any]) -> Tuple[int, int]:
    """Where the optimised 170 bp comes from in the 3020 bp gene extraction window, as [start, end)."""
    if factors["location"] == "correct":
        return STARRSEQ_V2_INSERT_START, STARRSEQ_V2_INSERT_END
    if factors["background"] == "construct":
        # the construct always mutates the insert site; the origin is in the record name
        start = int(parameters["sequence_name"].rsplit("_off_", 1)[1])
        return start, start + WINDOW_LENGTH
    return parameters["mutation_start"], parameters["mutation_end"]


def run_record(run_path: str, factors: Dict[str, str], stats: Dict[str, Any], gene_ids: Set[str]) -> Dict[str, Any]:
    """The per-run columns of one run.

    Args:
        stats: This run's entry in the config's stats file.
    """
    parameters = load_json(os.path.join(run_path, "parameters.json"))
    source_start, source_end = source_window(factors, parameters)
    model_names = sorted(os.path.basename(model["path"]) for model in parameters["models"])
    return {
        "record_name": parameters["sequence_name"],
        "gene_id": gene_id_from_record(parameters["sequence_name"], gene_ids),
        **factors,
        "source_start": source_start,
        "source_end": source_end,
        "number_of_generations": parameters["number_of_generations"],
        "n_models": len(model_names),
        "model_paths": "|".join(model_names),
        "has_vcf": parameters["vcf_path"] is not None,
        "start_fitness": stats["start_fitness"],
        "final_fitness": stats["final_fitness"],
        "max_mutations_reached": round(stats["max_mutations"]),
        "num_mutations_half_max_effect": stats["num_mutations_half_max_effect"],
    }


def front_entries(front_path: str) -> pd.DataFrame:
    """The entries of one Pareto front in file order, without their sequences."""
    front = load_json(front_path)
    entries = pd.DataFrame({
        "front_index": range(len(front)),
        "n_mutations": [round(n_mutations) for _, _, n_mutations in front],
        "fitness": [fitness for _, fitness, _ in front],
        "n_count": [sequence.upper().count("N") for sequence, _, _ in front],
    })
    # non-dominated entries with identical fitness can share a mutation count
    entries["tied_at_count"] = entries["n_mutations"].duplicated(keep=False)
    return entries


def collapse_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per gene: its species, all reasons joined with '|'."""
    per_gene = metadata.groupby("gene_id")
    return pd.DataFrame({
        "species": per_gene["species"].first(),
        "reason": per_gene["reason"].agg(lambda reasons: "|".join(sorted(set(reasons)))),
    }).reset_index()


def add_entry_flags(front_table: pd.DataFrame) -> pd.DataFrame:
    """Add the per-entry flags and the fitness gain over the reference.

    The gain is signed so that positive always means the optimisation worked.
    """
    sign = np.where(front_table["direction"] == "maximize", 1, -1)
    return front_table.assign(
        is_reference=front_table["n_mutations"] == 0,
        is_half_max=front_table["n_mutations"] == front_table["num_mutations_half_max_effect"],
        is_max_mutations=front_table["n_mutations"] == front_table["max_mutations_reached"],
        fitness_delta_vs_reference=sign * (front_table["fitness"] - front_table["start_fitness"]),
    )


def pool_fronts(runs_root: str, analysis_root: str, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the per-run overview and the per-entry front table.

    Returns:
        run_overview, one row per run; front_table, one row per front entry,
        keyed by (run_dir, front_index).
    """
    gene_ids = set(metadata["gene_id"])
    factors_by_name = config_factors()
    runs, fronts = [], []
    for config_dir_name in sorted(os.listdir(runs_root)):
        # strip the three-field timestamp
        config_name = config_dir_name.rsplit("_", 3)[0]
        factors = factors_by_name[config_name]
        stats = load_json(os.path.join(analysis_root, config_name, f"stats_{config_name}.json"))
        config_dir_path = os.path.join(runs_root, config_dir_name)
        # the config folder also holds problem_configs*.json for runs that never started
        run_names = sorted(name for name in os.listdir(config_dir_path) if os.path.isdir(os.path.join(config_dir_path, name)))
        for run_name in run_names:
            run_dir = f"{config_dir_name}/{run_name}"
            run_path = os.path.join(runs_root, run_dir)
            runs.append({
                "run_dir": run_dir,
                "group": group_of(factors),
                "config_name": config_name,
                **run_record(run_path, factors, stats[run_name], gene_ids),
            })
            front = front_entries(os.path.join(run_path, "saved_populations", "pareto_front.json"))
            fronts.append(front.assign(run_dir=run_dir))
    run_overview = pd.DataFrame(runs).merge(collapse_metadata(metadata), on="gene_id", how="left")
    front_table = add_entry_flags(run_overview.merge(pd.concat(fronts), on="run_dir"))
    return run_overview, front_table


def main() -> None:
    """Pool the full run and write the tables and the fixed parameters to OUTPUT."""
    run_overview, front_table = pool_fronts(RUNS, ANALYSIS, pd.read_csv(METADATA))
    os.makedirs(OUTPUT, exist_ok=True)
    run_overview.to_csv(os.path.join(OUTPUT, "run_overview.csv"), index=False)
    front_table.to_csv(os.path.join(OUTPUT, "front_table.csv.gz"), index=False)
    parameters = load_json(os.path.join(RUNS, run_overview["run_dir"].iloc[0], "parameters.json"))
    with open(os.path.join(OUTPUT, "fixed_parameters.json"), "w") as f:
        json.dump({name: parameters[name] for name in FIXED_PARAMETERS}, f, indent=2)


if __name__ == "__main__":
    main()
