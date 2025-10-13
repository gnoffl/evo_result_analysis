import os
import argparse
import json
import bisect
from typing import List, Optional, Tuple

from analysis.simple_result_stats import calculate_half_max_mutations, get_min_mutation_count_for_fitness
import numpy as np
import pandas as pd

def get_sequence_at_mutation_count(pareto_front: List[Tuple[str, float, float]], target_mutation_count: int) -> str:
    # implement binary search for the mutation count
    mutation_counts = [round(item[2]) for item in pareto_front]
    index = bisect.bisect_left(mutation_counts, target_mutation_count)
    if index < len(pareto_front) and mutation_counts[index] == target_mutation_count:
        return pareto_front[index][0]
    else:
        print(mutation_counts)
        print(target_mutation_count)
        raise ValueError(f"Mutation count {target_mutation_count} not found in pareto front.")


def select_candidates(results_folder: str, output_path: str, starting_fitness_max: Optional[float] = None, starting_fitness_min: Optional[float] = None) -> None:
    """
    Select genes with the lowest starting values from analysis results.
    """
    if not os.path.exists(results_folder):
        raise FileNotFoundError(f"The specified results folder does not exist: {results_folder}")

    gene_folders = [os.path.join(results_folder, d) for d in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, d))]
    pareto_paths = [os.path.join(gene_folder, "saved_populations", "pareto_front.json") for gene_folder in gene_folders]
    pareto_paths = [p for p in pareto_paths if os.path.isfile(p)]

    selected = {}
    for pareto_path in pareto_paths:
        selected.setdefault("gene", []).append(os.path.basename(os.path.dirname(os.path.dirname(pareto_path))))
        print(selected["gene"][-1])
        with open(pareto_path, 'r') as file:
            pareto_front = json.load(file)
        start_fitness = pareto_front[-1][1]
        final_fitness = pareto_front[0][1]
        max_num_mutations = pareto_front[0][2]
        half_max_mutations = calculate_half_max_mutations(pareto_front)
        
        for fitness in np.arange(0, 1, 0.1):
            if fitness < start_fitness or fitness > final_fitness:
                value = np.nan
            else:
                value = get_min_mutation_count_for_fitness(pareto_front, fitness)
            selected.setdefault(f"mutations_for_fitness_{fitness:.1f}", []).append(value)
            selected.setdefault(f"sequence_for_fitness_{fitness:.1f}", []).append(
                get_sequence_at_mutation_count(pareto_front, round(value)) if not np.isnan(value) else ""
            )
        selected.setdefault("reference_sequence", []).append(pareto_front[-1][0])
        selected.setdefault("max_num_mutations", []).append(max_num_mutations)
        selected.setdefault("start_fitness", []).append(start_fitness)
        selected.setdefault("final_fitness", []).append(final_fitness)
        selected.setdefault("half_max_mutations", []).append(half_max_mutations)
    
    selected = pd.DataFrame(selected)
    selected = selected.set_index("gene")
    if starting_fitness_max is not None:
        selected = selected[selected["start_fitness"] <= starting_fitness_max]
    if starting_fitness_min is not None:
        selected = selected[selected["start_fitness"] >= starting_fitness_min]
    selected = selected.sort_values(by="start_fitness", ascending=True)
    print(selected.describe())
    selected.to_csv(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select genes with the lowest starting values from analysis results.")
    parser.add_argument("--results_folder", "-r", type=str, help="Path to the folder containing analysis results.")
    parser.add_argument("--output_path", "-o", type=str, help="Path to save the selected genes CSV file.")
    parser.add_argument("--starting_fitness_max", "-max", type=float, default=None, help="Maximum starting fitness to filter genes.")
    parser.add_argument("--starting_fitness_min", "-min", type=float, default=None, help="Minimum starting fitness to filter genes.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    select_candidates(args.results_folder, args.output_path, args.starting_fitness_max, args.starting_fitness_min)