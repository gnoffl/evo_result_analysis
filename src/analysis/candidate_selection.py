import os
import argparse
import json
import bisect
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from analysis.simple_result_stats import calculate_half_max_mutations, get_min_mutation_count_for_fitness
import numpy as np
import pandas as pd

def get_data_at_mutation_count(pareto_front: List[Tuple[str, float, float]], target_mutation_count: int) -> Tuple[str, float, float]:
    # implement binary search for the mutation count
    mutation_counts = [round(item[2]) for item in pareto_front]
    mutation_counts_asc = mutation_counts[::-1]
    index = bisect.bisect_left(mutation_counts_asc, target_mutation_count)
    if index < len(pareto_front) and mutation_counts_asc[index] == target_mutation_count:
        reversed_index = len(pareto_front) - 1 - index
        return pareto_front[reversed_index]
    else:
        print(mutation_counts)
        print(target_mutation_count)
        print(index)
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
        selected["gene"][-1] = selected["gene"][-1].split("_")[1]
        print(selected["gene"][-1])
        with open(pareto_path, 'r') as file:
            pareto_front = json.load(file)
        start_fitness = pareto_front[-1][1]
        final_fitness = pareto_front[0][1]
        max_num_mutations = pareto_front[0][2]
        half_max_mutations = calculate_half_max_mutations(pareto_front)

        for mutations in [1, 2, 3, 5, 7, 10, 15, 20]:
            value = get_data_at_mutation_count(pareto_front, mutations)[1]
            selected.setdefault(f"fitness_delta_for_{mutations}_mutations", []).append(value - start_fitness)
            selected.setdefault(f"sequence_for_{mutations}_mutations", []).append(get_data_at_mutation_count(pareto_front, mutations)[0])
        
        for fitness in np.arange(0, 1, 0.1):
            if fitness < start_fitness or fitness > final_fitness:
                value = np.nan
            else:
                value = get_min_mutation_count_for_fitness(pareto_front, fitness)
            selected.setdefault(f"mutations_for_fitness_{fitness:.1f}", []).append(value)
            selected.setdefault(f"sequence_for_fitness_{fitness:.1f}", []).append(
                get_data_at_mutation_count(pareto_front, round(value))[0] if not np.isnan(value) else ""
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


def load_selected_genes(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"The specified CSV file does not exist: {csv_path}")
    selected = pd.read_csv(csv_path, index_col="gene")
    return selected


def compare_trajectories_mutations():
    selected_single = load_selected_genes("/home/gernot/ARCitect/ARCs/genRE/assays/Evolution/protocols/paper/analysis/GOF_LOF/GOF/GOF_single_mutation/candidate_genes_GOF_single.csv")
    selected_multi = load_selected_genes("/home/gernot/ARCitect/ARCs/genRE/assays/Evolution/protocols/paper/analysis/GOF_LOF/GOF/GOF_multi_mutation/candidate_genes_GOF_multi.csv")
    merged = selected_single.merge(selected_multi, left_index=True, right_index=True, suffixes=("_single", "_multi"))
    vis_folder = os.path.join("visualization", "GOF_LOF", "GOF", "compare_trajectories")
    os.makedirs(vis_folder, exist_ok=True)
    for gene , row in merged.iterrows():
        mutation_cols_single = [col for col in merged.columns if col.startswith("mutations_for_fitness_") and col.endswith("_single")]
        mutation_cols_multi = [col for col in merged.columns if col.startswith("mutations_for_fitness_") and col.endswith("_multi")]
        single_points = []
        multi_points = []
        for col in mutation_cols_single:
            single_points.append((float(col.split("_")[-2]), row[col]))
        for col in mutation_cols_multi:
            multi_points.append((float(col.split("_")[-2]), row[col]))
        single_points.append((row["start_fitness_single"], 0))
        single_points.append((row["final_fitness_single"], row["max_num_mutations_single"]))
        multi_points.append((row["start_fitness_multi"], 0))
        multi_points.append((row["final_fitness_multi"], row["max_num_mutations_multi"]))
        single_points = sorted(single_points, key=lambda x: x[0])
        multi_points = sorted(multi_points, key=lambda x: x[0])
        single_mutations = [point[1] for point in single_points]
        multi_mutations = [point[1] for point in multi_points]
        output_path = os.path.join(vis_folder, f"{gene}_trajectory_comparison.png")
        plt.figure()
        plt.plot([point[0] for point in single_points], single_mutations, marker='o', label='Single Mutation')
        plt.plot([point[0] for point in multi_points], multi_mutations, marker='o', label='Multi Mutation')
        plt.xlabel('Fitness')
        plt.ylabel('Number of Mutations')
        plt.xlim(-0.03, 1.03)
        plt.ylim(-3, 93)
        plt.title(f'Fitness vs. Number of Mutations for {gene}')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()



def one_off_error_correction(path):
    data = pd.read_csv(path, index_col="gene")
    data["corrected_gene"] = data.index.str.split("_").str[1]
    data = data.set_index("corrected_gene")
    #rename corrected_gene to gene
    data.index.name = "gene"
    data.to_csv(path)


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
    # compare_trajectories_mutations()
    # one_off_error_correction("/home/gernot/ARCitect/ARCs/genRE/assays/Evolution/protocols/paper/analysis/GOF_LOF/GOF/GOF_multi_mutation/candidate_genes_GOF_multi.csv")