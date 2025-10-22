import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
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
    # get method with max area as reference
    reference = max(areas, key=areas.get)   #type: ignore
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
    for method, fronts in normalized_fronts.items():
        average_mutations, average_fitnesses, std_fitnesses = get_plot_vals_normalized_fronts(fronts)
        plt.errorbar(average_mutations, average_fitnesses, yerr=std_fitnesses, label=method)
    plt.xlabel("Number of Mutations")
    plt.ylabel("Normalized Fitness")
    plt.title("Normalized Pareto Fronts Comparison")
    plt.xlim(-1, 91)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "normalized_pareto_fronts_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

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
    reference_method = max(fronts.keys(), key=lambda m: sum([item[1] for item in fronts[m]]))  #type: ignore
    reference_front = fronts[reference_method]
    differences_dict = {}
    for method, front in fronts.items():
        differences = [item1[1] - item2[1] for item1, item2 in zip(reference_front, front)]
        mutations = [item[2] for item in front]
        differences_dict[method] = (differences, mutations)
    return differences_dict

def plot_differences_between_fronts(fronts: Dict[str, List[Tuple[str, float, int]]], gene_name: str, tag: str, output_dir: str) -> None:
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
    plt.savefig(os.path.join(output_dir, f"pareto_fronts_differences_{gene_name}_{tag}.png"), dpi=300, bbox_inches='tight')

def plot_interesting_pareto_fronts(fronts: Dict[str, List[Tuple[str, float, int]]], gene_name: str, tag: str, output_dir: str) -> None:
    plot_interesting_pareto_fronts_values(fronts, gene_name, tag, output_dir)
    plot_differences_between_fronts(fronts, gene_name, tag, output_dir)


def compare_methods_final(results_paths: Dict[str, str], output_dir: str, max_mutations: int = 90) -> None:
    gene_folder_paths = {}
    for method, path in results_paths.items():
        gene_folder_paths[method] = [os.path.join(path, folder) for folder in os.listdir(path)]
    gene_paths = get_gene_paths(gene_folder_paths)
    check_genes_present(gene_paths, methods=list(results_paths.keys()))
    min_diff_data, max_abs_diff_data, max_summed_diff_data, max_delta_max_summed_data = None, None, None, None
    min_diff, max_abs_diff, max_summed_diff, max_delta_max_summed = float("inf"), -1, -1, -1
    normalized_fronts = {}
    for gene, method_paths in gene_paths.items():
        fronts = {}
        for method, path in method_paths.items():
            with open(path, 'r') as f:
                fronts[method] = expand_pareto_front(json.load(f), max_number_mutation=max_mutations)
        differences = calculate_differences_between_fronts(fronts)
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
    plot_normalized_fronts(normalized_fronts, output_dir=output_dir)
    plot_interesting_pareto_fronts(fronts=min_diff_data[1], gene_name=min_diff_data[0], tag="min_diff", output_dir=output_dir) #type: ignore
    plot_interesting_pareto_fronts(fronts=max_abs_diff_data[1], gene_name=max_abs_diff_data[0], tag="max_abs", output_dir=output_dir) #type: ignore
    plot_interesting_pareto_fronts(fronts=max_summed_diff_data[1], gene_name=max_summed_diff_data[0], tag="max_sum", output_dir=output_dir) #type: ignore
    plot_interesting_pareto_fronts(fronts=max_delta_max_summed_data[1], gene_name=max_delta_max_summed_data[0], tag="max_delta", output_dir=output_dir) #type: ignore


#TODO: compare avg loss over generations (linear and log scale)
#TODO: compare avg final fronts (next to each other and differences)
# log gene with max difference between methods (both progress and final)
    # account for both abs and summed deviation between fronts