import argparse
import os
import json
from analysis.summarize_mutations import MutationsGene
import pandas as pd
import numpy as np
from typing import Any, List, Dict, Optional, Tuple
from matplotlib import pyplot as plt

from analysis.simple_result_stats import expand_pareto_front


def compare_methods_progress(results_paths: List[str]) -> None:
    pass

def check_genes_present(gene_paths: Dict[str, Dict[str, str]], methods: List[str]) -> None:
    missing = []
    for gene, method_paths in gene_paths.items():
        for method in methods:
            if method not in method_paths:
                missing.append((gene, method))
    if missing:
        missing = sorted(missing, key=lambda x: x[0])
        for gene, method in missing:
            print(f"Missing gene {gene} for method {method}")
        raise ValueError("Some genes are missing for certain methods.")

def get_gene_paths(gene_folder_paths: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
    gene_paths = {}
    for method, gene_folders in gene_folder_paths.items():
        for gene_folder in gene_folders:
            gene_name = "_".join(os.path.basename(gene_folder).split("_")[:2])
            curr_dict = gene_paths.get(gene_name, {})
            curr_dict[method] = os.path.join(gene_folder, "saved_populations", "pareto_front.json")
            if not os.path.isfile(curr_dict[method]):
                raise FileNotFoundError(f"Warning: Missing pareto front file for gene {gene_name} and method {method}")
            gene_paths[gene_name] = curr_dict
    return gene_paths

def calculate_differences_between_fronts(fronts: Dict[str, List[Tuple[str, float, int]]]) -> Dict[str, Dict[str, float]]:
    areas = {method: sum([item[1] for item in front]) for method, front in fronts.items()}
    maximization = fronts[list(fronts.keys())[0]][0][1] < fronts[list(fronts.keys())[0]][-1][1]
    if maximization:
        reference = max(areas, key=areas.get)   #type: ignore
    else:
        reference = min(areas, key=areas.get)   #type: ignore
    # get method with max area as reference
    differences = {"summed_diff": {method: abs(areas[reference] - area) for method, area in areas.items()}}
    for method, front in fronts.items():
        if method == reference:
            differences.setdefault("abs_diff", {})[method] = 0.
            continue
        abs_diff = sum([abs(item1[1] - item2[1]) for item1, item2 in zip(fronts[reference], front)])
        differences.setdefault("abs_diff", {})[method] = abs_diff
    return differences

def add_normalized_fronts(fronts: Dict[str, List[Tuple[str, float, int]]], normalized_fronts_all: Dict[str, List[List[Tuple[float, int]]]]) -> None:
    min_vals, max_vals = [], []
    for method, front in fronts.items():
        min_vals.append(min([item[1] for item in front]))
        max_vals.append(max([item[1] for item in front]))
    global_max = max(max_vals)
    global_min = min(min_vals)
    for method, front in fronts.items():
        normalized_front = []
        for _, fitness, mutations in front:
            if global_max - global_min == 0:
                norm_fitness = 0.0
            else:
                norm_fitness = (fitness - global_min) / (global_max - global_min)
            normalized_front.append((norm_fitness, mutations))
        existing_fronts = normalized_fronts_all.get(method, [])
        existing_fronts.append(normalized_front)
        normalized_fronts_all[method] = existing_fronts

def get_plot_vals_normalized_fronts(normalized_fronts: List[List[Tuple[float, int]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fitnesses = [[fitness for fitness, _ in front] for front in normalized_fronts]
        fitnesses = np.array(fitnesses)
        mutations = [[mutations for _, mutations in front] for front in normalized_fronts]
        mutations = np.array(mutations)
        average_fitnesses = np.mean(fitnesses, axis=0)
        std_fitnesses = np.std(fitnesses, axis=0)
        average_mutations = np.mean(mutations, axis=0)
        return average_mutations, average_fitnesses, std_fitnesses

def plot_normalized_fronts(normalized_fronts: Dict[str, List[List[Tuple[float, int]]]], output_dir: str) -> None:
    plt.clf()
    average_fronts = {}
    for method, fronts in normalized_fronts.items():
        average_mutations, average_fitnesses, std_fitnesses = get_plot_vals_normalized_fronts(fronts)
        average_fronts[method] = [("", fitness, mutation) for fitness, mutation in zip(average_fitnesses, average_mutations)]
        plt.errorbar(average_mutations, average_fitnesses, yerr=std_fitnesses, label=method)
    plt.xlabel("Number of Mutations")
    plt.ylabel("Normalized Fitness")
    plt.title("Normalized Pareto Fronts Comparison")
    plt.xlim(-1, 91)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "normalized_pareto_fronts_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    plot_differences_between_fronts(fronts=average_fronts, gene_name="average fitness", tag="normalized", output_dir=output_dir, output_name="normalized_pareto_fronts_differences.png")

def plot_interesting_pareto_fronts_values(fronts: Dict[str, List[Tuple[str, float, int]]], gene_name: str, tag: str, output_dir: str) -> None:
    plt.clf()
    for method, front in fronts.items():
        mutations = [item[2] for item in front]
        fitnesses = [item[1] for item in front]
        plt.plot(mutations, fitnesses, label=method)
    plt.xlabel("Number of Mutations")
    plt.ylabel("Fitness")
    plt.title(f"Pareto Fronts Comparison for {gene_name} ({tag})")
    plt.xlim(-1, 91)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"pareto_fronts_comparison_{gene_name}_{tag}.png"), dpi=300, bbox_inches='tight')

def get_differences_and_mutations(fronts: Dict[str, List[Tuple[str, float, int]]]) -> Dict[str, Tuple[List[float], List[int]]]:
    maximization = fronts[list(fronts.keys())[0]][0][1] < fronts[list(fronts.keys())[0]][-1][1]
    if maximization:
        reference_method = max(fronts.keys(), key=lambda m: sum([item[1] for item in fronts[m]]))  #type: ignore
    else:
        reference_method = min(fronts.keys(), key=lambda m: sum([item[1] for item in fronts[m]]))  #type: ignore
    reference_front = fronts[reference_method]
    differences_dict = {}
    for method, front in fronts.items():
        differences = [item1[1] - item2[1] for item1, item2 in zip(front, reference_front)]
        mutations = [item[2] for item in front]
        differences_dict[method] = (differences, mutations)
    return differences_dict

def plot_differences_between_fronts(fronts: Dict[str, List[Tuple[str, float, int]]], gene_name: str, tag: str, output_dir: str, output_name: Optional[str] = None) -> None:
    # all fronts need to have the same lengths and same mutations
    lengths = [len(front) for front in fronts.values()]
    if len(set(lengths)) != 1:
        raise ValueError("Cannot plot differences between fronts of different lengths.")
    mutations = [[item[2] for item in front] for front in fronts.values()]
    for muts in mutations[1:]:
        if muts != mutations[0]:
            raise ValueError("Cannot plot differences between fronts with different mutation counts.")
    plt.clf()
    differences_dict = get_differences_and_mutations(fronts)
    for method, (differences, mutations) in differences_dict.items():
        plt.plot(mutations, differences, label=f"{method}")
    plt.xlabel("Number of Mutations")
    plt.ylabel("Absolute Fitness Difference")
    plt.title(f"Differences Between Pareto Fronts for {gene_name} ({tag})")
    plt.xlim(-1, 91)
    plt.legend()
    output_name = output_name if output_name is not None else f"pareto_fronts_differences_{gene_name}_{tag}.png"
    plt.savefig(os.path.join(output_dir, output_name), dpi=300, bbox_inches='tight')

def plot_interesting_pareto_fronts(fronts: Dict[str, List[Tuple[str, float, int]]], gene_name: str, tag: str, output_dir: str) -> None:
    plot_interesting_pareto_fronts_values(fronts, gene_name, tag, output_dir)
    plot_differences_between_fronts(fronts, gene_name, tag, output_dir)


def update_ranks_and_areas(ranks_and_areas: Dict[str, Dict[str, Any]], differences: Dict[str, float]) -> None:
    sorted_differences = sorted(differences.items(), key=lambda x: x[1])
    for rank, (method, value) in enumerate(sorted_differences):
        ranks_and_areas["ranks"] = ranks_and_areas.get("ranks", {})
        ranks_and_areas["areas"] = ranks_and_areas.get("areas", {})
        ranks_and_areas["best"] = ranks_and_areas.get("best", {})
        ranks_and_areas["ranks"][method] = ranks_and_areas["ranks"].get(method, 0) + rank + 1
        ranks_and_areas["areas"][method] = ranks_and_areas["areas"].get(method, 0) + value
        ranks_and_areas["best"][method] = ranks_and_areas["best"].get(method, 0)
        if rank == 0:
            ranks_and_areas["best"][method] += 1


def compare_methods_final(results_paths: Dict[str, str], output_dir: str, max_mutations: int = 90) -> None:
    gene_folder_paths = {}
    for method, path in results_paths.items():
        gene_folder_paths[method] = [os.path.join(path, folder) for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    gene_paths = get_gene_paths(gene_folder_paths)
    check_genes_present(gene_paths, methods=list(results_paths.keys()))
    min_diff_data, max_abs_diff_data, max_summed_diff_data, max_delta_max_summed_data = None, None, None, None
    min_diff, max_abs_diff, max_summed_diff, max_delta_max_summed = float("inf"), -1, -1, -1
    normalized_fronts = {}
    ranks_and_areas = {}
    for gene, method_paths in gene_paths.items():
        fronts = {}
        for method, path in method_paths.items():
            with open(path, 'r') as f:
                fronts[method] = expand_pareto_front(json.load(f), max_number_mutation=max_mutations)
        differences = calculate_differences_between_fronts(fronts)
        update_ranks_and_areas(ranks_and_areas, differences["summed_diff"])
        differences = {measurement: sum([value]) for measurement, values in differences.items() for value in values.values()}
        if differences["abs_diff"] < min_diff:
            min_diff = differences["abs_diff"]
            min_diff_data = (gene, fronts)
        if differences["abs_diff"] > max_abs_diff:
            max_abs_diff = differences["abs_diff"]
            max_abs_diff_data = (gene, fronts)
        if differences["summed_diff"] > max_summed_diff:
            max_summed_diff = differences["summed_diff"]
            max_summed_diff_data = (gene, fronts)
        delta_max_summed = differences["abs_diff"] - differences["summed_diff"]
        if delta_max_summed > max_delta_max_summed:
            max_delta_max_summed = delta_max_summed
            max_delta_max_summed_data = (gene, fronts)
        add_normalized_fronts(fronts, normalized_fronts_all=normalized_fronts)
    # normalize ranks and best counts
    ranks_and_areas["ranks"] = {method: rank / len(gene_paths) for method, rank in ranks_and_areas["ranks"].items()}
    ranks_and_areas["best"] = {method: best / len(gene_paths) for method, best in ranks_and_areas["best"].items()}
    plot_normalized_fronts(normalized_fronts, output_dir=output_dir)
    plot_interesting_pareto_fronts(fronts=min_diff_data[1], gene_name=min_diff_data[0], tag="min_diff", output_dir=output_dir) #type: ignore
    plot_interesting_pareto_fronts(fronts=max_abs_diff_data[1], gene_name=max_abs_diff_data[0], tag="max_abs", output_dir=output_dir) #type: ignore
    plot_interesting_pareto_fronts(fronts=max_summed_diff_data[1], gene_name=max_summed_diff_data[0], tag="max_sum", output_dir=output_dir) #type: ignore
    plot_interesting_pareto_fronts(fronts=max_delta_max_summed_data[1], gene_name=max_delta_max_summed_data[0], tag="max_delta", output_dir=output_dir) #type: ignore
    with open(os.path.join(output_dir, "ranks_and_areas.json"), 'w') as f:
        json.dump(ranks_and_areas, f, indent=2)

def load_mutation_data(input_data):
    method_dict = {}
    genes = {}
    for method, (results_path, mutation_path) in input_data.items():
        with open(mutation_path, 'r') as f:
            mutation_data = json.load(f)
            method_dict[method] = mutation_data
            genes[method] = sorted(list(mutation_data.keys()))
    gene_names = genes[list(genes.keys())[0]]
    for method, gene_list in genes.items():
        if gene_list != gene_names:
            raise ValueError(f"Gene lists do not match between methods. Method {method} has different genes.")
    for method in method_dict.keys():
        mutation_data = method_dict[method]
        for gene, data in mutation_data.items():
            mutation_data[gene] = MutationsGene.from_dict(data)
    return method_dict, gene_names

def count_mutations_per_method(method_dict, gene):
    curr_dict = {}
    for method, mutation_data in method_dict.items():
        gene_data: MutationsGene = mutation_data[gene]
        generations = sorted([int(gen) for gen in gene_data.generation_dict.keys()])
        final_generation = generations[-1]
        mutation_list = gene_data.get_all_mutations_generation(generation=final_generation)
        mutation_count = len(mutation_list)
        curr_dict[method] = mutation_count
    return curr_dict

def get_ranks_from_sorted(curr_dict: Dict[str, float]) -> List[float]:
    sorted_methods = sorted(curr_dict.items(), key=lambda x: x[1], reverse=True)
    duplicated = {}
    for method, mutation_count in sorted_methods:
        duplicated[mutation_count] = duplicated.get(mutation_count, []) + [method]
    ranks = []
    last_rank = 0
    for mutation_count, methods in sorted(duplicated.items(), key=lambda x: x[0], reverse=True):
        length = len(methods)
        curr_rank =  (2 * last_rank + 1 + length) / 2
        ranks.extend([curr_rank] * length)
        last_rank += length
    return ranks

def update_comparison_dict(comparison_dict, curr_dict):
    sorted_methods = sorted(curr_dict.items(), key=lambda x: x[1], reverse=True)
    ranks = get_ranks_from_sorted(curr_dict)

    for i, (method, mutation_count) in enumerate(sorted_methods):
        comparison_dict.setdefault("mutation_count", {}).setdefault(method, 0)
        comparison_dict.setdefault("ranks", {}).setdefault(method, 0)
        comparison_dict.setdefault("best", {}).setdefault(method, 0)
        comparison_dict["mutation_count"][method] += mutation_count
        comparison_dict["ranks"][method] += ranks[i]
        if ranks[i] == ranks[0]:
            comparison_dict["best"][method] += 1 / ranks.count(ranks[i])

def rank_by_mutation_count(method_dict, gene_names):
    comparison_dict = {}
    for gene in gene_names:
        curr_dict = count_mutations_per_method(method_dict, gene)
        # sort curr_dict by mutation count
        update_comparison_dict(comparison_dict, curr_dict)
    # normalize ranks and best counts
    comparison_dict["ranks"] = {method: rank / len(gene_names) for method, rank in comparison_dict["ranks"].items()}
    comparison_dict["best"] = {method: best / len(gene_names) for method, best in comparison_dict["best"].items()}
    return comparison_dict

def compare_diversity_methods(input_data: Dict[str, Tuple[str, str]], output_dir: str) -> None:
    method_dict, gene_names = load_mutation_data(input_data)
    comparison_dict = rank_by_mutation_count(method_dict, gene_names)
    with open(os.path.join(output_dir, "diversity_comparison.json"), 'w') as f:
        json.dump(comparison_dict, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare evolutionary methods based on their results.")
    parser.add_argument("--results_paths", "-r", type=str, nargs='+', required=True, help="Paths to the results of different methods.")
    parser.add_argument("--mutation_data", "-mu", type=str, nargs='*', help="Path to mutation data if needed for diversity comparison.", default=[])
    parser.add_argument("--methods", "-me", type=str, nargs='+', required=True, help="Names of the methods corresponding to the results paths.")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory to save the output plots.")

    parser.add_argument("--final", "-f", action='store_true', help="Compare final results of methods.")
    parser.add_argument("--diversity", "-d", action='store_true', help="Compare diversity of methods based on mutation data.")
    parser.add_argument("--progress", "-p", action='store_true', help="Compare progress of methods over generations.")
    parser.add_argument("--all", "-a", action='store_true', help="Compare all aspects of methods.")
    parsed = parser.parse_args()
    if len(parsed.results_paths) != len(parsed.methods):
        raise ValueError("Number of results paths must match number of methods.")
    if parsed.mutation_data and len(parsed.mutation_data) != len(parsed.methods):
        raise ValueError("Number of mutation data paths must match number of methods.")
    mutation_data = {}
    if parsed.mutation_data:
        mutation_data = {method: (results_path, mutation_path) for method, mutation_path, results_path in zip(parsed.methods, parsed.mutation_data, parsed.results_paths)}
    results_paths = {method: path for method, path in zip(parser.parse_args().methods, parser.parse_args().results_paths)}
    return parsed, results_paths, mutation_data


#TODO: compare avg loss over generations (linear and log scale)
#TODO: compare avg final fronts (next to each other and differences)

if __name__ == "__main__":
    args, results_paths, mutation_data = parse_args()
    run_final = args.final or args.all
    run_diversity = args.diversity or args.all
    run_progress = args.progress or args.all
    if not (run_final or run_diversity or run_progress):
        raise ValueError("At least one of --final, --diversity, --progress, or --all must be specified.")
    # compare_methods_progress(results_paths=results_paths)
    if run_final:
        compare_methods_final(results_paths=results_paths, output_dir=args.output_dir)
    if run_diversity:
        compare_diversity_methods(input_data=mutation_data, output_dir=args.output_dir)
    if run_progress:
        raise NotImplementedError("Progress comparison not yet implemented.")
        compare_methods_progress(results_paths=results_paths)