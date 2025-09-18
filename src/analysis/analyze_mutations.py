import math
from typing import Dict, List, Tuple
import numpy as np
from analysis.summarize_mutations import MutationsGene
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def count_mutations_single_gene(mutation_data: Dict, gene_name: str, generation: int = 1999) -> Tuple[List[int], List[int]]:
    if gene_name not in mutation_data:
        raise ValueError(f"Gene {gene_name} not found in mutation data.")
    data_for_all_generations = MutationsGene.from_dict(mutation_data[gene_name])
    if generation not in data_for_all_generations.generation_dict.keys():
        raise ValueError(f"Generation {generation} not found in mutation data for gene {gene_name}.")
    pareto_front_current_generation = data_for_all_generations.generation_dict[generation]
    mutation_counts = {}
    individual_mutation_counts = []
    for individual in pareto_front_current_generation:
        individual_mutation_counts.append(len(individual.mutations))
        for pos, ref, mut in individual.mutations:
            mutation_counts[str(pos) + str(mut)] = mutation_counts.get(str(pos) + str(mut), 0) + 1
    # print mutation counts sorted by frequency
    # sorted_mutations = sorted(mutation_counts.items(), key=lambda x: x[1], reverse=False)
    # print(f"Mutations for gene {gene_name} in generation {generation}:")
    # for mutation, count in sorted_mutations:
    #     print(f"Mutation: {mutation}, Count: {count}")
    return list(mutation_counts.values()), sorted(individual_mutation_counts, reverse=True)


def count_mutations_all_genes(mutation_data_path: str, generation: int = 1999) -> Dict[str, Tuple[List[int], List[int]]]:
    with open(mutation_data_path, 'r') as f:
        mutation_data = json.load(f)
    mutation_counts = {}
    for gene_name in mutation_data.keys():
        try:
            counts, individual_mutation_counts = count_mutations_single_gene(mutation_data, gene_name, generation)
            mutation_counts[gene_name] = (counts, individual_mutation_counts)
        except ValueError as e:
            print(f"Skipping {gene_name}: {e}")
    return mutation_counts


def calculate_mutation_stats(mutation_data_path: str, generation: int = 1999) -> Tuple[List[Tuple[str, int, int]], float, float, float, float, np.ndarray]:
    mutation_counts = count_mutations_all_genes(mutation_data_path=mutation_data_path, generation=generation)
    sorted_by_count = sorted([(gene, max(counts), len(individual_mutation_counts)) for gene, (counts, individual_mutation_counts) in mutation_counts.items()], key=lambda x: x[1], reverse=False)
    counts = np.array([max_count for _, max_count, _ in sorted_by_count])
    n_individuals = np.array([n_individuals for _, _, n_individuals in sorted_by_count])
    ratios = counts / (n_individuals - 1)
    average_max_count = np.mean(counts)
    average_n_individuals = np.mean(n_individuals)
    std_max_count = np.std(counts)
    std_n_individuals = np.std(n_individuals)
    return sorted_by_count, float(average_max_count), float(std_max_count), float(average_n_individuals), float(std_n_individuals), ratios


def print_mutation_stats(mutation_data_path: str, generation: int = 1999) -> None:
    sorted_by_count, average_max_count, std_max_count, average_n_individuals, std_n_individuals, ratios = calculate_mutation_stats(mutation_data_path=mutation_data_path, generation=generation)
    print(f"max max_count: {sorted_by_count[-1][1]} for gene {sorted_by_count[-1][0]} with {sorted_by_count[-1][2]} individuals")
    print(f"min max_count: {sorted_by_count[0][1]} for gene {sorted_by_count[0][0]} with {sorted_by_count[0][2]} individuals")
    print(f"Average max count: {average_max_count}, Std: {std_max_count}")
    print(f"Average number of individuals: {average_n_individuals}, Std: {std_n_individuals}")
    print(f"Average ratio of max count to number of individuals: {np.mean(ratios)}, Std: {np.std(ratios)}")


def create_ideal_distribution(individual_mutation_counts: List[int], target_length: int) -> np.ndarray:
    """Create an ideal distribution of mutations meaning that all individuals have the maximum possible overlap in mutations.

    Args:
        individual_mutation_counts (List[int]): List of mutation counts for each individual.
        target_length (int): Target length for the distribution.

    Returns:
        np.ndarray: Ideal distribution of mutations.
    """
    output = np.zeros(target_length, dtype=int)
    for i in range(len(individual_mutation_counts)):
        current_mutation_count = individual_mutation_counts[i]
        output[:current_mutation_count] += 1
    return output


def create_worst_case_distribution(mutation_counts, mutable_positions):
    n_mutations_total = sum(mutation_counts)
    possible_number_of_mutations = 3 * mutable_positions
    worst_case_base = n_mutations_total // possible_number_of_mutations
    remainder = n_mutations_total % possible_number_of_mutations
    worst_case_distribution = [worst_case_base + 1] * remainder
    if worst_case_base > 0:
        worst_case_distribution.extend([worst_case_base] * (possible_number_of_mutations - remainder))
    worst_case_distribution = np.array(worst_case_distribution)
    return worst_case_distribution


def calculate_conservation_statistic(mutation_counts: List[int], individual_mutation_counts: List[int], mutable_positions: int) -> float:
    """Calculate a conservation statistic based on mutation counts.
    
    Args:
        mutation_counts (List[int]): List of mutation counts for each gene.
        individual_mutation_counts (List[int]): List of mutation counts for each individual of the current pareto front / gene.
        sequence_length (int): Length of the sequence.

    Returns:
        float: Conservation statistic.
    """
    worst_case_distribution = create_worst_case_distribution(mutation_counts, mutable_positions)
    ideal_distribution = create_ideal_distribution(individual_mutation_counts, len(worst_case_distribution))
    if np.array_equal(ideal_distribution, worst_case_distribution):
        return 1.0
    mutation_count_distribution = np.zeros(len(worst_case_distribution), dtype=int)
    mutation_count_distribution[:len(mutation_counts)] = mutation_counts

    difference_current_dist = np.abs(ideal_distribution - mutation_count_distribution).sum()
    difference_worst_case_dist = np.abs(ideal_distribution - worst_case_distribution).sum()
    stat = 1 - (difference_current_dist / difference_worst_case_dist)
    return stat


def calc_conservation_stat_stats(conservation_stats: Dict[str, float]) -> Tuple[float, float, str, str, float, float]:
    #get min, max avg and std of conservation statistics with corresponding gene each
    min_stat, max_stat = 1, 0
    min_gene, max_gene = "", ""

    for gene, stat in conservation_stats.items():
        if stat < min_stat:
            min_stat = stat
            min_gene = gene
        if stat > max_stat:
            max_stat = stat
            max_gene = gene
            
    avg_stat = np.mean(list(conservation_stats.values()))
    std_stat = np.std(list(conservation_stats.values()))
    return min_stat, max_stat, min_gene, max_gene, float(avg_stat), float(std_stat)


def calculate_conservation_statistics(mutation_data_path: str, name: str, generation: int = 1999, mutable_positions: int = 3000, output_folder: str = ".") -> Dict[str, float]:
    """Calculate conservation statistics for all genes in the mutation data.
    
    Args:
        mutation_data_path (str): Path to the mutation data JSON file.
        generation (int): Generation to analyze. Defaults to 1999.
        mutable_positions (int): Number of positions in which mutations can occur. Defaults to 3000.

    Returns:
        Dict[str, float]: Dictionary with gene names as keys and conservation statistics as values.
    """
    mutation_counts = count_mutations_all_genes(mutation_data_path=mutation_data_path, generation=generation)
    conservation_stats = {}
    for gene_name, (counts, individual_mutation_counts) in mutation_counts.items():
        conservation_stat = calculate_conservation_statistic(counts, individual_mutation_counts, mutable_positions)
        conservation_stats[gene_name] = conservation_stat
    
    min_stat, max_stat, min_gene, max_gene, avg_stat, std_stat = calc_conservation_stat_stats(conservation_stats)
    print(f"Conservation statistics for generation {generation}:")
    print(f"Min: {min_stat} ({min_gene})\nMax: {max_stat} ({max_gene})\nAvg: {avg_stat}\nStd: {std_stat}")
    with open(os.path.join(output_folder, f"conservation_statistics_{name}_gen_{generation}.json"), 'w') as f:
        json.dump(conservation_stats, f, indent=2)
    return conservation_stats


def plot_dict_as_stacked_bars(data_dict: Dict, title: str, xlabel: str, ylabel: str, file_path: str):
    """Plot mutations from data_dict as stacked histogram.
    
    Args:
        data_dict (Dict): Dictionary containing mutation data.
        title (str): Title for the plot.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        file_path (str): Filename for the output file.
    """
    # plot mutations from from_dict as stacked histogram 
    # for each mutation number, a stacked bar should appear showing the different mutations in different colors
    x_labels = sorted(data_dict.keys())
    bottom_line = np.zeros(len(x_labels))

    letter_counts_from = {
        'A': np.array([data_dict[x].get("A", 0) for x in x_labels]),
        'C': np.array([data_dict[x].get("C", 0) for x in x_labels]),
        'G': np.array([data_dict[x].get("G", 0) for x in x_labels]),
        'T': np.array([data_dict[x].get("T", 0) for x in x_labels])
    }

    plt.clf()
    fig, ax = plt.subplots()
    for letter, counts in letter_counts_from.items():
        ax.bar(x_labels, counts, bottom=bottom_line, label=letter)
        bottom_line += counts

    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')

def make_line_plot_rolling_window(data_dict: Dict, name: str, window_size: int = 11, output_folder: str = "."):
    """Create a rolling window line plot of mutations.
    
    Args:
        data_dict (Dict): Dictionary containing mutation data.
        name (str): Name to distinguish the output file.
        window_size (int): Size of the rolling window. Defaults to 11.
        output_folder (str): Path to the output folder for saving results. Defaults to ".".
    """
    SEQUENCE_LENGTH = 3020
    PADDING = 20
    results = {
        "A": np.empty(SEQUENCE_LENGTH),
        "C": np.empty(SEQUENCE_LENGTH),
        "G": np.empty(SEQUENCE_LENGTH),
        "T": np.empty(SEQUENCE_LENGTH)
    }
    colors = {"A": "green", "C": "blue", "G": "red", "T": "orange", "Sum": "black"}
    for letter in results.keys():
        counts = np.array([data_dict.get(x, {}).get(letter, 0) for x in range(SEQUENCE_LENGTH)])
        rolling_mean = np.convolve(counts, np.ones(window_size)/window_size, mode='valid')
        results[letter] = rolling_mean
    sum_over_all_letters = np.sum([results[letter] for letter in results.keys()], axis=0)
    results["Sum"] = sum_over_all_letters
    cutoff = window_size / 2 - 0.5
    indexes = np.arange(len(sum_over_all_letters)) + cutoff
    # cut out the area in the middle of the array that is not valid
    start = math.ceil((len(results["Sum"]) - PADDING) // 2 - cutoff)
    end = math.ceil((len(results["Sum"]) + PADDING) // 2 + cutoff)
    results = {letter: np.concatenate((values[:start], np.array([np.nan]* (end - start)), values[end:])) for letter, values in results.items()}

    if name.endswith("_to"):
        title = "Introduced Nucleotides"
    elif name.endswith("_from"):
        title = "Removed Nucleotides"
    elif name.endswith("_diff"):
        title = "Net Change in Nucleotides"
    else:
        title = ""
    # title = f'Rolling Mean of Mutations for {name}; Window Size: {window_size}'
    plt.clf()
    plt.figure(figsize=(8, 5.333))
    for letter, values in results.items():
        plt.plot(indexes, values, label=letter, color=colors[letter])
    plt.xlabel('Position in Sequence', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.title(title, fontsize=15)
    plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000], [0, 500, "1000\nTSS", 1500, "2000\nTTS", 2500, 3000], fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(output_folder, f"rolling_mean_mutations_{name}_{window_size}.pdf"), dpi=300, bbox_inches='tight')
        

def plot_hist_half_max_mutations_stacked(mutation_data_path: str, name: str, output_folder: str = "."):
    """Generate stacked histogram of mutations at half max fitness.
    
    Args:
        mutation_data_path (str): Path to the mutation data JSON file.
        name (str): Name to distinguish output files.
        output_folder (str): Path to the output folder for saving results. Defaults to ".".
    """
    from_dict, to_dict = {}, {}
    with open(mutation_data_path, 'r') as f:
        mutation_data = json.load(f)
    for gene, data in tqdm(mutation_data.items(), desc="Processing genes"):
        generation_mutations = MutationsGene.from_dict(data)
        generations = sorted([int(gen) for gen in generation_mutations.generation_dict.keys()])
        final_generation = generations[-1]
        initial_fit, optimal_fit_generation = generation_mutations.get_init_and_optimal_fitness_generation(final_generation)
        half_max_fitness = (initial_fit + optimal_fit_generation) / 2.0
        half_max_seq = generation_mutations.get_equal_or_next_closest_fitness(final_generation, half_max_fitness)
        half_max_mut = half_max_seq.get_mutation_number()
        for position, ref_base, mut_base in half_max_seq.mutations:
            mut_dict_from = from_dict.setdefault(half_max_mut, {})     
            mut_dict_from[ref_base] = mut_dict_from.get(ref_base, 0) + 1
            mut_dict_to = to_dict.setdefault(half_max_mut, {})
            mut_dict_to[mut_base] = mut_dict_to.get(mut_base, 0) + 1
    # plot mutations from from_dict as stacked histogram 
    # for each mutation number, a stacked bar should appear showing the different mutations in different colors
    plot_dict_as_stacked_bars(from_dict,
                              title=f'Histogram of Mutations at Half Max Fitness for {name} (From)',
                              xlabel='Mutations at Half Max Fitness',
                              ylabel='Frequency',
                              file_path=os.path.join(output_folder, f"hist_half_max_mutations_stacked_{name}_from.pdf"))
    # plot mutations from to_dict as stacked histogram
    # for each mutation number, a stacked bar should appear showing the different mutations in different colors
    plot_dict_as_stacked_bars(to_dict,
                              title=f'Histogram of Mutations at Half Max Fitness for {name} (To)',
                              xlabel='Mutations at Half Max Fitness',
                              ylabel='Frequency',
                              file_path=os.path.join(output_folder, f"hist_half_max_mutations_stacked_{name}_to.pdf"))


def plot_mutations_location(mutation_data_path: str, name: str, window_size: int = 11, plot_stacked: bool = True, plot_rolling: bool = True, output_folder: str = "."):
    """Generate mutation location plots.
    
    Args:
        mutation_data_path (str): Path to the mutation data JSON file.
        name (str): Name to distinguish output files.
        window_size (int): Window size for rolling mean plots. Defaults to 11.
        plot_stacked (bool): Whether to generate stacked bar plots. Defaults to True.
        plot_rolling (bool): Whether to generate rolling window plots. Defaults to True.
        output_folder (str): Path to the output folder for saving results. Defaults to ".".
    """
    if not plot_stacked and not plot_rolling:
        raise ValueError("At least one of plot_stacked or plot_rolling must be True.")
    from_dict, to_dict = {}, {}
    with open(mutation_data_path, 'r') as f:
        mutation_data = json.load(f)
    for gene, data in tqdm(mutation_data.items(), desc="Processing genes"):
        generation_mutations = MutationsGene.from_dict(data)
        generations = sorted([int(gen) for gen in generation_mutations.generation_dict.keys()])
        final_generation = generations[-1]
        min_fit, max_fit = generation_mutations.get_init_and_optimal_fitness_generation(final_generation)
        max_seq = generation_mutations.get_equal_or_next_closest_fitness(final_generation, max_fit)
        for position, ref_base, mut_base in max_seq.mutations:
            mut_dict_from = from_dict.setdefault(position, {})
            mut_dict_from[ref_base] = mut_dict_from.get(ref_base, 0) + 1
            mut_dict_to = to_dict.setdefault(position, {})
            mut_dict_to[mut_base] = mut_dict_to.get(mut_base, 0) + 1

    diff_dict = {
        position: {base: to_dict[position].get(base, 0) - from_dict[position].get(base, 0) for base in ["A", "C", "G", "T"]} for position in from_dict.keys()
    }
    
    if plot_stacked:
        plot_dict_as_stacked_bars(from_dict,
                                  title=f'Histogram of Mutations Location for {name} (From)',
                                  xlabel='Position',
                                  ylabel='Frequency',
                                  file_path=os.path.join(output_folder, f"hist_mutations_location_stacked_{name}_from.pdf"))
        plot_dict_as_stacked_bars(to_dict,
                                  title=f'Histogram of Mutations Location for {name} (To)',
                                  xlabel='Position',
                                  ylabel='Frequency',
                                  file_path=os.path.join(output_folder, f"hist_mutations_location_stacked_{name}_to.pdf"))
        plot_dict_as_stacked_bars(diff_dict,
                                  title=f'Histogram of Mutations Location for {name} (Diff)',
                                  xlabel='Position',
                                  ylabel='Frequency',
                                  file_path=os.path.join(output_folder, f"hist_mutations_location_stacked_{name}_diff.pdf"))
    
    if plot_rolling:
        make_line_plot_rolling_window(from_dict, f"{name}_from", window_size=window_size, output_folder=output_folder)
        make_line_plot_rolling_window(to_dict, f"{name}_to", window_size=window_size, output_folder=output_folder)
        make_line_plot_rolling_window(diff_dict, f"{name}_diff", window_size=window_size, output_folder=output_folder)


def plot_hist_mutation_conservation(mutation_data_path: str, name: str, generation: int = 1999, mutable_positions: int = 3000, output_folder: str = ".") -> None:
    out_path = os.path.join(output_folder, f"conservation_statistics_{name}_gen_{generation}.json")
    if os.path.isfile(out_path):
        with open(out_path, 'r') as f:
            conservation_stats = json.load(f)
    else:
        conservation_stats = calculate_conservation_statistics(mutation_data_path=mutation_data_path, name=name, generation=generation, mutable_positions=mutable_positions, output_folder=output_folder)
            
    stats = list(conservation_stats.values())
    
    # Create the histogram
    plt.figure(figsize=(12, 6))
    plt.hist(stats, bins=30, range=(0, 1))
    plt.xlabel('Conservation Statistic')
    plt.ylabel('Frequency')
    plt.title(f'Conservation Statistics for {name} at Generation {generation}')
    plt.savefig(os.path.join(output_folder, f"hist_mutation_conservation_{name}_gen_{generation}.pdf"), dpi=300, bbox_inches='tight')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze mutations from evolution results and generate visualizations')
    parser.add_argument('--mutation_data', '-m', help='Path to the mutation data JSON file', required=True)
    parser.add_argument('--name', '-n', help='Name to distinguish output files', required=True)
    parser.add_argument('--output_folder', '-o', help='Path to the output folder for saving results', default='.', required=False)
    parser.add_argument('--window_size', '-w', type=int, default=31, help='Window size for rolling mean plots (default: 31)')
    parser.add_argument('--generation', '-g', type=int, default=1999, help='Generation to analyze (default: 1999). Currently relevant for conservation statistics of mutations only.')
    parser.add_argument("--mutable_positions", type=int, default=3000, help='Number of positions in which mutations can occur (default: 3000). Currently relevant for conservation statistics of mutations only.')

    # Analysis control flags - explicit inclusion
    parser.add_argument('--plot_half_max_stacked', action='store_true', help='Generate stacked histogram of mutations at half max fitness')
    parser.add_argument('--plot_mutations_location', action='store_true', help='Generate mutation location plots (both stacked and rolling by default)')
    parser.add_argument('--plot_stacked_only', action='store_true', help='Generate only stacked bar plots for mutation locations')
    parser.add_argument('--plot_rolling_only', action='store_true', help='Generate only rolling window plots for mutation locations')
    parser.add_argument("--plot_mutation_conservation", action='store_true', help='Calculate conservation statistics for mutations in each gene')
    parser.add_argument('--all', action='store_true', help='Run all analysis steps')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mutation_data):
        raise FileNotFoundError(f"Mutation data file not found: {args.mutation_data}")
    
    return args


def main():
    args = parse_args()
    
    # Determine which functions to run based on explicit flags
    run_half_max_stacked = args.plot_half_max_stacked or args.all
    run_mutations_location = args.plot_mutations_location or args.plot_stacked_only or args.plot_rolling_only or args.all
    run_mutation_conservation = args.plot_mutation_conservation or args.all

    if not (run_half_max_stacked or run_mutations_location or run_mutation_conservation):
        raise ValueError("Currently no analysis is selected for execution, so nothing is done. Use --all to run all available analyses or use -h to get an overview of available analyses.")
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    if run_half_max_stacked:
        try:
            plot_hist_half_max_mutations_stacked(args.mutation_data, args.name, args.output_folder)
        except Exception as e:
            print(f"Error while plotting half max mutations stacked: {e}")
    
    if run_mutations_location:
        # Determine plot types based on specific flags
        if args.plot_stacked_only:
            plot_stacked, plot_rolling = True, False
        elif args.plot_rolling_only:
            plot_stacked, plot_rolling = False, True
        else:
            # Default: both plots for --plot_mutations_location or --all
            plot_stacked, plot_rolling = True, True

        try:
            plot_mutations_location(args.mutation_data, args.name, 
                                  window_size=args.window_size,
                                  plot_stacked=plot_stacked,
                                  plot_rolling=plot_rolling,
                                  output_folder=args.output_folder)
        except Exception as e:
            print(f"Error while plotting mutations location: {e}")
        
    if run_mutation_conservation:
        try:
            plot_hist_mutation_conservation(mutation_data_path=args.mutation_data, name=args.name, generation=args.generation,
                                             mutable_positions=args.mutable_positions, output_folder=args.output_folder)
        except Exception as e:
            print(f"Error while plotting mutation conservation: {e}")

if __name__ == "__main__":
    main()
