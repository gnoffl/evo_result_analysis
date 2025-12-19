import argparse
import os
import json
import random
import pandas as pd
import re
import numpy as np
import math
from typing import Any, List, Dict, Optional, Tuple
from matplotlib import pyplot as plt
import sklearn.decomposition as decomposition

from analysis.simple_result_stats import expand_pareto_front
from analysis.summarize_mutations import MutatedSequence, MutationsGene
from analysis.analyze_mutations import count_mutations_single_gene, calculate_conservation_statistic
import tqdm


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

def plot_normalized_fronts(normalized_fronts: Dict[str, List[List[Tuple[float, int]]]], output_dir: str, output_format: str) -> None:
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
    plt.savefig(os.path.join(output_dir, f"normalized_pareto_fronts_comparison.{output_format}"), dpi=300, bbox_inches='tight')
    plt.close()
    plot_differences_between_fronts(fronts=average_fronts, gene_name="average fitness", tag="normalized", output_dir=output_dir, output_name="normalized_pareto_fronts_differences.png", output_format=output_format)

def plot_interesting_pareto_fronts_values(fronts: Dict[str, List[Tuple[str, float, int]]], gene_name: str, tag: str, output_dir: str, output_format: str) -> None:
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
    plt.savefig(os.path.join(output_dir, f"pareto_fronts_comparison_{gene_name}_{tag}.{output_format}"), dpi=300, bbox_inches='tight')

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

def plot_differences_between_fronts(fronts: Dict[str, List[Tuple[str, float, int]]], gene_name: str, tag: str, output_dir: str, output_format: str, output_name: Optional[str] = None) -> None:
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
    output_name = output_name if output_name is not None else f"pareto_fronts_differences_{gene_name}_{tag}.{output_format}"
    plt.savefig(os.path.join(output_dir, output_name), dpi=300, bbox_inches='tight')

def plot_interesting_pareto_fronts(fronts: Dict[str, List[Tuple[str, float, int]]], gene_name: str, tag: str, output_dir: str, output_format: str) -> None:
    plot_interesting_pareto_fronts_values(fronts=fronts, gene_name=gene_name, tag=tag, output_dir=output_dir, output_format=output_format)
    plot_differences_between_fronts(fronts=fronts, gene_name=gene_name, tag=tag, output_dir=output_dir, output_format=output_format)


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


def compare_methods_final(results_paths: Dict[str, str], output_dir: str, output_format: str, max_mutations: int = 90) -> None:
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
    plot_normalized_fronts(normalized_fronts, output_dir=output_dir, output_format=output_format)
    plot_interesting_pareto_fronts(fronts=min_diff_data[1], gene_name=min_diff_data[0], tag="min_diff", output_dir=output_dir, output_format=output_format) #type: ignore
    plot_interesting_pareto_fronts(fronts=max_abs_diff_data[1], gene_name=max_abs_diff_data[0], tag="max_abs", output_dir=output_dir, output_format=output_format) #type: ignore
    plot_interesting_pareto_fronts(fronts=max_summed_diff_data[1], gene_name=max_summed_diff_data[0], tag="max_sum", output_dir=output_dir, output_format=output_format) #type: ignore
    plot_interesting_pareto_fronts(fronts=max_delta_max_summed_data[1], gene_name=max_delta_max_summed_data[0], tag="max_delta", output_dir=output_dir, output_format=output_format) #type: ignore
    with open(os.path.join(output_dir, "ranks_and_areas.json"), 'w') as f:
        json.dump(ranks_and_areas, f, indent=2)


def normalize_gene_name(gene_name: str) -> str:
    # use RE to find whether date is present in gene name (3 blocks of 6 digits separated by _) and if so, remove it
    pattern = r'_(\d{6}_\d{6}_\d{6})$'
    match = re.search(pattern, gene_name)
    if match:
        return gene_name[:match.start()]
    return gene_name

def load_mutation_data(input_data):
    method_dict = {}
    genes = {}
    for method, (results_path, mutation_path) in input_data.items():
        with open(mutation_path, 'r') as f:
            mutation_data = json.load(f)
            method_dict[method] = mutation_data
            #normalize gene names by splitting off the date (format: _YYYYMMDD_HHMMSS_msmsms)
            genes[method] = {gene: normalize_gene_name(gene) for gene in mutation_data.keys()}
    gene_names = set(genes[list(genes.keys())[0]].values())
    for method, gene_list in genes.items():
        curr_genes = set(gene_list.values())
        if curr_genes != gene_names:
            missing_in_method = curr_genes - gene_list
            extra_in_method = gene_list - curr_genes
            if missing_in_method:
                print(f"Genes missing in method {method}: {missing_in_method}")
            if extra_in_method:
                print(f"Extra genes in method {method}: {extra_in_method}")
            raise ValueError(f"Gene lists do not match between methods. Method {method} has different genes.")
    for method in method_dict.keys():
        mutation_data = method_dict[method]
        loaded_mutation_data = {}
        for gene, data in mutation_data.items():
            loaded_mutation_data[genes[method][gene]] = MutationsGene.from_dict(data)
        method_dict[method] = loaded_mutation_data
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

def update_comparison_dict(comparison_dict, curr_dict, max_best: bool = True, summation_field: str = "mutation_count"):
    sorted_methods = sorted(curr_dict.items(), key=lambda x: x[1], reverse=max_best)
    ranks = get_ranks_from_sorted(curr_dict)

    for i, (method, mutation_count) in enumerate(sorted_methods):
        comparison_dict.setdefault(summation_field, {}).setdefault(method, 0)
        comparison_dict.setdefault("ranks", {}).setdefault(method, 0)
        comparison_dict.setdefault("best", {}).setdefault(method, 0)
        comparison_dict[summation_field][method] += mutation_count
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


def get_sampled_pareto_fronts(pareto_front: List[MutatedSequence], max_bootstrap: int) -> List[List[MutatedSequence]]:
    options_per_mutation_count = {}
    for individual in pareto_front:
        options_per_mutation_count.setdefault(len(individual.mutations), []).append(individual)
    choices = [len(options) for options in options_per_mutation_count.values()]
    possible_combinations = math.prod(choices)
    bootstraps = min(possible_combinations, max_bootstrap)
    sampled_fronts = []
    for _ in range(bootstraps):
        selected_individuals = []
        for options in options_per_mutation_count.values():
            selected_individual = np.random.choice(options)
            selected_individuals.append(selected_individual)
        sampled_fronts.append(selected_individuals)
    return sampled_fronts


def calculate_conservation_statistic_pareto_front(pareto_front: List[MutatedSequence], max_bootstrap: int, mutable_positions: int) -> float:
    stats = []
    sampled_fronts = get_sampled_pareto_fronts(pareto_front, max_bootstrap)
    for curr_pareto_front in sampled_fronts:
        mutation_counts, individual_mutation_counts = count_mutations_single_gene(curr_pareto_front)
        statistic = calculate_conservation_statistic(mutation_counts=mutation_counts, individual_mutation_counts=individual_mutation_counts, mutable_positions=mutable_positions)
        stats.append(statistic)
    average_statistic = sum(stats) / len(stats)
    return average_statistic


def calculate_conservation_per_method(method_dict: Dict[str, Any], gene: str, max_bootstrap: int, mutable_positions: int):
    method_stats = {}
    for method, mutation_data in method_dict.items():
        gene_data: MutationsGene = mutation_data[gene]
        generations = sorted([int(gen) for gen in gene_data.generation_dict.keys()])
        final_generation = generations[-1]
        pareto_front = gene_data.generation_dict[final_generation]
        diversity_measure = calculate_conservation_statistic_pareto_front(pareto_front, max_bootstrap=max_bootstrap, mutable_positions=mutable_positions)
        method_stats[method] = diversity_measure
    return method_stats


def rank_by_conservation(method_dict, gene_names: List[str], max_bootstrap: int, mutable_positions: int):
    comparison_dict = {}
    for gene in gene_names:
        curr_dict = calculate_conservation_per_method(method_dict, gene=gene, max_bootstrap=max_bootstrap, mutable_positions=mutable_positions)
        # sort curr_dict by mutation count
        update_comparison_dict(comparison_dict, curr_dict, max_best=False, summation_field="conservation_measure")
    # normalize ranks and best counts
    comparison_dict["ranks"] = {method: rank / len(gene_names) for method, rank in comparison_dict["ranks"].items()}
    comparison_dict["best"] = {method: best / len(gene_names) for method, best in comparison_dict["best"].items()}
    return comparison_dict

def compare_diversity_methods(input_data: Dict[str, Tuple[str, str]], output_dir: str, max_bootstrap: int, mutable_positions: int) -> None:
    method_dict, gene_names = load_mutation_data(input_data)
    comparison_dict_mutation_count = rank_by_mutation_count(method_dict, gene_names)
    comparison_dict_conservation_measure = rank_by_conservation(method_dict, gene_names, max_bootstrap, mutable_positions)

    results_dict = {
        "mutation_count_comparison": comparison_dict_mutation_count,
        "conservation_measure_comparison": comparison_dict_conservation_measure
    }
    with open(os.path.join(output_dir, "diversity_comparison.json"), 'w') as f:
        json.dump(results_dict, f, indent=2)

def get_all_mutations(method_dict, gene) -> set:
    all_mutations = set()
    for method, mutation_data in method_dict.items():
        gene_data: MutationsGene = mutation_data[gene]
        final_generation = max([int(gen) for gen in gene_data.generation_dict.keys()])
        mutation_list = gene_data.get_all_mutations_generation(generation=final_generation)
        all_mutations.update(mutation_list)
    return all_mutations

def get_method_vectors(method_dict, gene, all_mutations):
    method_vectors = {}
    for method, mutation_data in method_dict.items():
        gene_data: MutationsGene = mutation_data[gene]
        final_generation = max([int(gen) for gen in gene_data.generation_dict.keys()])
        vectors = []
        for element in gene_data.generation_dict[final_generation]:
            vector_representation = [1 if (mutation in element.mutations) else 0 for mutation in all_mutations]
            vectors.append(vector_representation)
        method_vectors[method] = vectors
    return method_vectors

def pca_transform_all_methods(method_vectors: Dict[str, List[List[int]]]):
    PCA_all_methods = decomposition.PCA(n_components=2)
    all_methods_vectors = [vector for vectors in method_vectors.values() for vector in vectors]
    all_methods_vectors = np.array(all_methods_vectors)
    PCA_all_methods.fit(all_methods_vectors)
    transformed_vectors = {}
    for method, vectors in method_vectors.items():
        vectors = np.array(vectors)
        transformed = PCA_all_methods.transform(vectors)
        transformed_vectors[method] = transformed
    return PCA_all_methods, transformed_vectors

def plot_pca_all_methods(method_vectors: Dict[str, List[List[int]]], gene: str, output_dir: str, output_format: str):
    PCA_all_methods, transformed_vectors = pca_transform_all_methods(method_vectors)
    plt.clf()
    for method, transformed in transformed_vectors.items():
        plt.scatter(transformed[:,0], transformed[:,1], label=method, alpha=0.5)
    explained_variance = PCA_all_methods.explained_variance_ratio_
    plt.xlabel(f"PCA Component 1 ({explained_variance[0] * 100:.1f}% variance)")
    plt.ylabel(f"PCA Component 2 ({explained_variance[1] * 100:.1f}% variance)")
    plt.title(f"PCA of Mutation Vectors for {gene} (All Methods)")
    plt.legend()
    output_folder = os.path.join(output_dir, "pca_results")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"pca_{gene}_all_methods.{output_format}"), dpi=300, bbox_inches='tight')
    plt.close()

def pca_transform_single_method(vectors: List[List[int]]):
    PCA_method = decomposition.PCA(n_components=2)
    np_vectors = np.array(vectors)
    PCA_method.fit(np_vectors)
    transformed = PCA_method.transform(np_vectors)
    return PCA_method, transformed

def plot_pca_single_method(vectors: List[List[int]], gene: str, method: str, output_dir: str, output_format: str):
    PCA_method, transformed = pca_transform_single_method(vectors)
    explained_variance = PCA_method.explained_variance_ratio_
    plt.clf()
    plt.scatter(transformed[:,0], transformed[:,1], alpha=0.5)
    plt.xlabel(f"PCA Component 1 ({explained_variance[0] * 100:.1f}% variance)")
    plt.ylabel(f"PCA Component 2 ({explained_variance[1] * 100:.1f}% variance)")
    plt.title(f"PCA of Mutation Vectors for {gene} ({method})")
    output_folder = os.path.join(output_dir, "pca_results")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"pca_{gene}_{method}.{output_format}"), dpi=300, bbox_inches='tight')
    plt.close()

def pca_visualization(input_data: Dict[str, Tuple[str, str]], output_dir: str) -> None:
    method_dict, gene_names = load_mutation_data(input_data)
    tested_genes = random.sample(gene_names, min(4, len(gene_names)))
    for gene in tested_genes:
        all_mutations = get_all_mutations(method_dict, gene)
        method_vectors = get_method_vectors(method_dict, gene, all_mutations)
        plot_pca_all_methods(method_vectors=method_vectors, gene=gene, output_dir=output_dir, output_format="png")
        for method, vectors in method_vectors.items():
            plot_pca_single_method(vectors=vectors, gene=gene, method=method, output_dir=output_dir, output_format="png")

def get_best_area(method_paths: Dict[str, str], max_mutations: int) -> Tuple[float, bool]:
    fronts = {}
    for method, path in method_paths.items():
        with open(path, 'r') as f:
            fronts[method] = expand_pareto_front(json.load(f), max_number_mutation=max_mutations)
    areas = {method: sum([item[1] for item in front]) for method, front in fronts.items()}
    maximization = fronts[list(fronts.keys())[0]][0][1] < fronts[list(fronts.keys())[0]][-1][1]
    if maximization:
        reference = max(areas.values())   #type: ignore
    else:
        reference = min(areas.values())   #type: ignore
    return reference, maximization

def get_area(pareto_front_path: str, max_mutations: int) -> float:
    expanded_pareto_front = expand_pareto_front(json.load(open(pareto_front_path, 'r')), max_number_mutation=max_mutations)
    area = sum([item[1] for item in expanded_pareto_front])
    return area

def visualize_progress(generation_losses: Dict[str, Dict[int, List[float]]], output_dir: str, output_format: str, logarithmic: bool = True) -> None:
    plt.clf()
    if logarithmic:
        plt.yscale("log")
    for method, gen_losses in generation_losses.items():
        generations = sorted(gen_losses.keys())
        avg_losses = [np.mean(gen_losses[gen]) for gen in generations]
        std_losses = [np.std(gen_losses[gen]) for gen in generations]
        std_losses = np.array(std_losses)
        for i in range(len(std_losses)):
            if i % 1000 != -10:
                std_losses[i] = np.nan
        plt.errorbar(generations, avg_losses, yerr=std_losses, label=method)
    plt.xlabel("Generation")
    plt.ylabel("Average Loss to Best Front")
    plt.title("Method Progress Over Generations")
    plt.legend()
    name = "method_progress_over_generations_log" if logarithmic else "method_progress_over_generations_linear"
    plt.savefig(os.path.join(output_dir, f"{name}.{output_format}"), dpi=300, bbox_inches='tight')
    plt.close()


def compare_method_progress(results_paths: Dict[str, str], max_mutations: int, output_format: str, output_dir: str) -> None:
    gene_folder_paths = {}
    for method, path in results_paths.items():
        gene_folder_paths[method] = [os.path.join(path, folder) for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    gene_paths = get_gene_paths(gene_folder_paths)
    check_genes_present(gene_paths, methods=list(results_paths.keys()))
    generation_losses_dict = {}
    for gene, method_paths in tqdm.tqdm(gene_paths.items()):
        best_area, maximization = get_best_area(method_paths=method_paths, max_mutations=max_mutations)
        for method, path in method_paths.items():
            fronts_folder = os.path.dirname(path)
            generation_fronts = [os.path.join(fronts_folder, f) for f in os.listdir(fronts_folder) if f.startswith("pareto_front_gen_") and f.endswith(".json")]
            generation_fronts = {int(f.split("_")[-1].split(".")[0]): f for f in generation_fronts}
            generation_areas = {gen: get_area(path, max_mutations=max_mutations) for gen, path in generation_fronts.items()}
            generation_losses_curr = {gen: (best_area - area) if maximization else (area - best_area) for gen, area in generation_areas.items()}
            for gen, loss in generation_losses_curr.items():
                generation_losses_dict.setdefault(method, {}).setdefault(gen, []).append(loss)
    visualize_progress(generation_losses=generation_losses_dict, output_dir=output_dir, output_format=output_format, logarithmic=True)
    visualize_progress(generation_losses=generation_losses_dict, output_dir=output_dir, output_format=output_format, logarithmic=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare evolutionary methods based on their results.")
    parser.add_argument("--results_paths", "-r", type=str, nargs='+', required=True, help="Paths to the results of different methods.")
    parser.add_argument("--mutation_data", "-mu", type=str, nargs='*', help="Path to mutation data if needed for diversity comparison.", default=[])
    parser.add_argument("--methods", "-me", type=str, nargs='+', required=True, help="Names of the methods corresponding to the results paths.")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory to save the output plots.")
    parser.add_argument("--max_bootstrap", "-mb", type=int, default=100, help="Maximum number of bootstrap samples for diversity calculation.")
    parser.add_argument("--mutable_positions", "-mp", type=int, default=3000, help="Number of mutable positions for diversity calculation.")
    parser.add_argument("--output_format", "-fmt", type=str, default="png", help="Output format for plots (e.g., png, pdf). Default is png.")
    parser.add_argument("--max_mutations", "-mm", type=int, default=90, help="Maximum number of mutations to consider in pareto fronts.")

    parser.add_argument("--final", "-f", action='store_true', help="Compare final results of methods.")
    parser.add_argument("--diversity", "-d", action='store_true', help="Compare diversity of methods based on mutation data.")
    parser.add_argument("--mutation_pca", "-pca", action='store_true', help="Perform PCA visualization of mutation data.")
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
    results_paths = {method: path for method, path in zip(parsed.methods, parsed.results_paths)}
    parsed.output_format = parsed.output_format.lower().strip('.').strip()
    if parsed.output_format not in ['png', 'pdf', 'jpg', 'jpeg', 'tiff', "svg"]:
        raise ValueError("Unsupported output format. Supported formats are: png, pdf, jpg, jpeg, tiff, svg.")
    return parsed, results_paths, mutation_data


#TODO: compare avg loss over generations (linear and log scale)
#TODO: compare avg final fronts (next to each other and differences)
#TODO: at least for some examples look at clustering of the data

if __name__ == "__main__":
    args, results_paths, mutation_data = parse_args()
    run_final = args.final or args.all
    run_diversity = args.diversity or args.all
    run_progress = args.progress or args.all
    run_pca = args.mutation_pca or args.all
    if not (run_final or run_diversity or run_progress or run_pca):
        raise ValueError("At least one of --final, --diversity, --progress, --mutation_pca, or --all must be specified.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if run_final:
        compare_methods_final(results_paths=results_paths, output_dir=args.output_dir, output_format=args.output_format, max_mutations=args.max_mutations)
    if run_diversity:
        compare_diversity_methods(input_data=mutation_data, output_dir=args.output_dir, max_bootstrap=args.max_bootstrap, mutable_positions=args.mutable_positions)
    if run_pca:
        pca_visualization(input_data=mutation_data, output_dir=args.output_dir)
    if run_progress:
        compare_method_progress(results_paths=results_paths, max_mutations=args.max_mutations, output_format=args.output_format, output_dir=args.output_dir)