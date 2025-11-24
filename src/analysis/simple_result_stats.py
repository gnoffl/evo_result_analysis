import matplotlib.pyplot as plt
import os
import json
import numpy as np
import argparse
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm


def get_pareto_front(gene_path: str) -> Optional[List[Tuple[str, float, int]]]:
    pareto_front_path = os.path.join(gene_path, 'saved_populations', 'pareto_front.json')
    if not os.path.exists(pareto_front_path):
        print(f"Skipping {pareto_front_path}, not found.")
        return None
    with open(pareto_front_path, 'r') as f:
        pareto_front = json.load(f)
    return pareto_front


def add_basic_stats(stats: Dict[str, Dict[str, Any]], fitnesses: List[float], num_mutations: List[int], gene: str) -> None:
        stats[gene] = {
            'final_fitness': fitnesses[0],
            "start_fitness": fitnesses[-1],
            "max_mutations": max(num_mutations),
            "origin_chromosome": gene.split("_")[0],
        }

def get_min_mutation_count_for_fitness(pareto_front: List[Tuple[str, float, float]], target_fitness: float) -> float:
    EPSILON = 1e-9
    for i, item in enumerate(pareto_front):
        if pareto_front[0][1] > pareto_front[-1][1]:
            if item[1] < target_fitness - EPSILON:
                return pareto_front[i-1][2]
        else:
            if item[1] > target_fitness + EPSILON:
                return pareto_front[i-1][2]
    return pareto_front[-1][2]

def calculate_half_max_mutations(pareto_front: List[Tuple[str, float, float]]) -> int:
    half_max_fitness = (pareto_front[0][1] + pareto_front[-1][1]) / 2
    # items in the pareto front are sorted descending regarding their fitness
    return round(get_min_mutation_count_for_fitness(pareto_front, half_max_fitness))

# for folder structure
# -results_folder
#    - gene_A
#      - saved_populations
#        - pareto_front.json
#        - ...
#      - ...
#    - gene_B
#      - ...
# and structure of the pareto_front.json:
# [[sequence_as_string, fitness, number_of_mutations], ...]
# calculate the average and std of the maximal fitness and number of mutations

def get_stats_per_gene(results_folder: str, name: str, output_folder: str = ".") -> Dict[str, Dict[str, Any]]:
    """Calculate statistics for each gene in the results folder.

    Args:
        results_folder (str): Path to the results folder containing gene directories.
        name (str): Name to distinguish output files.
        output_folder (str): Path to the output folder for saving results. Defaults to ".".

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing statistics for each gene.
    """
    print(f"creating stats for {name} in {output_folder}")
    output_path = os.path.join(output_folder, f'stats_{name}.json')
    if os.path.exists(output_path):
        print(f"Stats file {output_path} already exists. If you dont want calculations to be redone, you can specify --stats_file next time.")

    stats = {}
    print(f"Analyzing results in {results_folder}...")
    for gene in tqdm(os.listdir(results_folder)):
        gene_path = os.path.join(results_folder, gene)
        if not os.path.isdir(gene_path):
            print(f"Skipping {gene_path}, not a directory.")
            continue

        pareto_front = get_pareto_front(gene_path)
        if pareto_front is None:
            continue
        # only_stats = [(fitness, num_mutations) for (seq, fitness, num_mutations) in pareto_front]
        # print(json.dumps(only_stats, indent=2))

        fitnesses = [item[1] for item in pareto_front]
        num_mutations = [item[2] for item in pareto_front]

        add_basic_stats(stats, fitnesses, num_mutations, gene)
    
        #add number of mutations for half max fitness
        mutations_half_max = calculate_half_max_mutations(pareto_front=pareto_front) # type: ignore
        stats[gene]['num_mutations_half_max_effect'] = mutations_half_max

    print(f"Stats for {name} calculated, saving to {output_path}")
    json.dump(stats, open(output_path, 'w'), indent=2)
    print("finished saving stats.")
    return stats


def summary_stat_calculation(stats):
    summary = {
        'final_fitness': [],
        'start_fitness': [],
        'max_mutations': [],
        'num_mutations_half_max_effect': []
    }

    for gene, stat in stats.items():
        summary['final_fitness'].append(stat['final_fitness'])
        summary['start_fitness'].append(stat['start_fitness'])
        summary['max_mutations'].append(stat['max_mutations'])
        if 'num_mutations_half_max_effect' in stat:
            summary['num_mutations_half_max_effect'].append(stat['num_mutations_half_max_effect'])

    summarized_stats = {
        'final_fitness_mean': np.mean(summary['final_fitness']),
        'final_fitness_std': np.std(summary['final_fitness']),
        'start_fitness_mean': np.mean(summary['start_fitness']),
        'start_fitness_std': np.std(summary['start_fitness']),
        'max_mutations_mean': np.mean(summary['max_mutations']),
        'max_mutations_std': np.std(summary['max_mutations']),
        'num_mutations_half_max_mean': np.mean(summary['num_mutations_half_max_effect']) if summary['num_mutations_half_max_effect'] else None,
        'num_mutations_half_max_std': np.std(summary['num_mutations_half_max_effect']) if summary['num_mutations_half_max_effect'] else None
    }
    
    return summarized_stats


def split_stats(stats):
    main_chrom_stats = {}
    organelle_scaffold_stats = {}
    for gene, stat in stats.items():
        origin = stat['origin_chromosome']
        if origin.isnumeric():
            main_chrom_stats[gene] = stat
        else:
            organelle_scaffold_stats[gene] = stat
    return main_chrom_stats,organelle_scaffold_stats


def summarize_stats(stats: Dict[str, Dict[str, Any]], name: str, output_folder: str, group_origin: bool = True) -> str:
    summarized_stats = summary_stat_calculation(stats)
    result = f"Summary for {name}:\n"
    for key, value in summarized_stats.items():
        result += f"  {key}: {value}\n"
    if group_origin:
        main_chrom_stats, organelle_scaffold_stats = split_stats(stats)
        result += summarize_stats(stats=main_chrom_stats, name=f"{name}_main_chromosome", output_folder=output_folder, group_origin=False)
        result += summarize_stats(stats=organelle_scaffold_stats, name=f"{name}_organelle_scaffold", output_folder=output_folder, group_origin=False)
    with open(os.path.join(output_folder, f'summary_{name}.txt'), 'w') as f:
        f.write(result)
    return result


def visualize_start_vs_max_fitness(stats: Dict[str, Dict[str, Any]], name: str, output_format: str, output_folder: str = ".") -> None:
    """create a scatter plot of the start fitness vs max fitness

    Args:
        stats (Dict[str, Dict[str, Any]]): Dictionary containing statistics for each gene.
        name (str): Name to distinguish the output file.
        output_folder (str): Path to the output folder for saving results. Defaults to ".".
    """
    plt.clf()
    
    # Separate data by groups
    main_chromosome_start = []
    main_chromosome_max = []
    organelle_scaffold_start = []
    organelle_scaffold_max = []
    
    for stat in stats.values():
        if stat['origin_chromosome'].isnumeric():
            main_chromosome_start.append(stat['start_fitness'])
            main_chromosome_max.append(stat['final_fitness'])
        else:
            organelle_scaffold_start.append(stat['start_fitness'])
            organelle_scaffold_max.append(stat['final_fitness'])

    # Create scatter plots with different colors for each group
    plt.scatter(main_chromosome_start, main_chromosome_max, 
               c='blue', alpha=0.6, label='Main Chromosome')
    plt.scatter(organelle_scaffold_start, organelle_scaffold_max, 
               c='red', alpha=0.6, label='Organelle or Scaffold')
    
    plt.xlabel('Start Fitness', fontsize=13)
    plt.ylabel('Final Fitness', fontsize=13)
    plt.title('Start Fitness vs Final Fitness', fontsize=15)
    # make axis tick labels size 12
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(output_folder, f'start_vs_final_fitness_{name}.{output_format}'), bbox_inches='tight')


def visualize_start_vs_max_fitness_by_mutations(stats: Dict[str, Dict[str, Any]], name: str, output_format: str, output_folder: str = ".") -> None:
    """create a scatter plot of the start fitness vs max fitness colored by mutations at half max fitness

    Args:
        stats (Dict[str, Dict[str, Any]]): Dictionary containing statistics for each gene.
        name (str): Name to distinguish the output file.
        output_folder (str): Path to the output folder for saving results. Defaults to ".".
    """
    draw_visualize_start_vs_max_fitness_by_mutations(stats, f"{name}_absolute", relative=False, output_folder=output_folder, output_format=output_format)
    draw_visualize_start_vs_max_fitness_by_mutations(stats, f"{name}_relative", relative=True, output_folder=output_folder, output_format=output_format)


def draw_visualize_start_vs_max_fitness_by_mutations(stats: Dict[str, Dict[str, Any]], name: str, output_format: str, relative: bool = False, output_folder: str = ".") -> None:
    """create a scatter plot of the start fitness vs max fitness colored by mutations at half max fitness

    Args:
        stats (Dict[str, Dict[str, Any]]): Dictionary containing statistics for each gene.
        name (str): Name to distinguish the output file.
        relative (bool): determines whether the absolute number for mutations at half max fitness is used or the relative value (i.e. number of mutations at half max fitness divided by the maximum number of mutations). Defaults to False.
        output_folder (str): Path to the output folder for saving results. Defaults to ".".
    """
    plt.clf()
    
    start_fitness = []
    max_fitness = []
    mutations_half_max = []
    
    for stat in stats.values():
        if 'num_mutations_half_max_effect' in stat:
            start_fitness.append(stat['start_fitness'])
            max_fitness.append(stat['final_fitness'])
            if relative:
                # Calculate relative mutations at half max fitness
                relative_mutations = stat['num_mutations_half_max_effect'] / stat['max_mutations'] if stat['max_mutations'] > 0 else 0
                mutations_half_max.append(relative_mutations)
            else:
                mutations_half_max.append(stat['num_mutations_half_max_effect'])

    # Create scatter plot with color mapped to mutations at half max fitness
    scatter = plt.scatter(start_fitness, max_fitness, 
                         c=mutations_half_max, alpha=0.6, cmap='viridis')
    
    plt.xlabel('Start Fitness')
    plt.ylabel('Final Fitness')
    plt.title('Start Fitness vs Final Fitness (Colored by Mutations at Half Max)')
    plt.colorbar(scatter, label='Mutations at Half Max Effect')
    plt.savefig(os.path.join(output_folder, f'start_vs_final_fitness_by_mutations_{name}.{output_format}'), bbox_inches='tight')


def plot_pareto_front(pareto_path: str, out_path: str) -> None:
    """Show the pareto front from the results folder.

    Args:
        pareto_path (str): Path to the pareto front JSON file.
        out_path (str): Path for the output files.
    """
    if not os.path.exists(pareto_path):
        print(f"Skipping {pareto_path}, not found.")
        return

    with open(pareto_path, 'r') as f:
        pareto_front = json.load(f)

    plt.clf()
    plt.figure()
    fitnesses = [item[1] for item in pareto_front]
    num_mutations = [item[2] for item in pareto_front]
    plt.scatter(num_mutations, fitnesses)
    plt.xlabel('Number of Mutations')
    plt.ylabel('Fitness')
    gene_folder_name = os.path.basename(os.path.dirname(os.path.dirname(pareto_path)))
    gene_name = "_".join(gene_folder_name.split("_")[:2])
    plt.title(f'Pareto Front for {gene_name}')
    plt.savefig(out_path, bbox_inches='tight')


def show_random_fronts(results_folder: str, output_format: str, num_samples: int = 4, output_folder: str = "."):
    """Show random pareto fronts from the results folder.

    Args:
        results_folder (str): Path to the results folder.
        num_samples (int): Number of random samples to show. Defaults to 4.
        output_folder (str): Path to the output folder for saving results. Defaults to ".".
    """
    genes = [gene for gene in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, gene))]
    selected_genes = np.random.choice(genes, size=min(num_samples, len(genes)), replace=False)
    out_folder = os.path.join(output_folder, f"random_pareto_fronts_{os.path.basename(results_folder)}")
    os.makedirs(out_folder, exist_ok=True)
    print(f"Selected genes: {selected_genes}")
    for gene in selected_genes:
        plot_pareto_front(os.path.join(results_folder, gene, 'saved_populations', 'pareto_front.json'), os.path.join(out_folder, f'pareto_front_{gene}.{output_format}'))


def deduplicate_pareto_front(pareto_front: List[Tuple[str, float, int]]) -> List[Tuple[str, float, int]]:
    pareto_front = sorted(pareto_front, key=lambda x: x[2], reverse=False)
    deduplicated = [pareto_front[0]]
    for item in pareto_front:
        if item[2] != deduplicated[-1][2]:
            deduplicated.append(item)
    return deduplicated


def expand_pareto_front(pareto_front: List[Tuple[str, float, int]], max_number_mutation: int) -> List[Tuple[str, float, int]]:
    """Expand the pareto front to include all points with the same fitness.

    Args:
        pareto_front (List[Tuple[str, float, int]]): The pareto front to expand.

    Returns:
        List[Tuple[str, float, int]]: The expanded pareto front.
    """
    pareto_front = deduplicate_pareto_front(pareto_front)
    expanded_front = [pareto_front[0]]
    next_index_old_front = 1
    while len(expanded_front) < max_number_mutation + 1:
        try:
            next_item_old_front = pareto_front[next_index_old_front]
        except IndexError:
            next_item_old_front = ("", 0, max_number_mutation + 10)
        curr_item_new_front = expanded_front[-1]
        if next_item_old_front[2] == curr_item_new_front[2] + 1:
            expanded_front.append(next_item_old_front)
            next_index_old_front += 1
        else:
            # add a new item with the same fitness but one more mutation
            new_item = ("", curr_item_new_front[1], curr_item_new_front[2] + 1)
            expanded_front.append(new_item)
    return expanded_front


def normalize_front(pareto_front: List[Tuple[str, float, int]]) -> List[Tuple[str, float, int]]:
    """Normalize the pareto front to have a maximum fitness of 1.

    Args:
        pareto_front (List[Tuple[str, float, int]]): The pareto front to normalize.

    Returns:
        List[Tuple[str, float, int]]: The normalized pareto front.
    """
    min_fitness = min(pareto_front, key=lambda x: x[1])[1]
    shifted_front = [(item[0], item[1] - min_fitness, item[2]) for item in pareto_front]
    max_fitness = max(shifted_front, key=lambda x: x[1])[1]
    if max_fitness == 0:
        raise ValueError("Maximum fitness is zero, cannot normalize.")
    normalized_front = [(item[0], item[1] / max_fitness, item[2]) for item in shifted_front]

    return normalized_front


def show_average_pareto_front(results_folder: str, output_format: str, output_folder: str = ".", max_number_mutation: int = 90) -> None:
    """Show the average pareto front from the results folder.

    Args:
        results_folder (str): Path to the results folder.
        output_folder (str): Path to the output folder for saving results. Defaults to ".".
    """
    genes = [gene for gene in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, gene))]
    all_fitnesses = []
    all_mutations = []

    for gene in tqdm(genes):
        pareto_path = os.path.join(results_folder, gene, 'saved_populations', 'pareto_front.json')
        if not os.path.exists(pareto_path):
            print(f"Skipping {pareto_path}, not found.")
            continue

        with open(pareto_path, 'r') as f:
            pareto_front = json.load(f)
        
        full_front = expand_pareto_front(pareto_front, max_number_mutation=max_number_mutation)
        normalized = normalize_front(full_front)
        
        fitnesses = [item[1] for item in normalized]
        num_mutations = [item[2] for item in normalized]
        
        all_fitnesses.append(fitnesses)
        all_mutations.append(num_mutations)

    fitnesses = np.array(all_fitnesses)
    avg_mutations = np.mean(all_mutations, axis=0)

    # plot the average pareto front with strandard deviation
    plt.clf()
    plt.figure()
    # make the dots the same shade of blue as in the show_random_fronts, but the errorbars black
    plt.errorbar(avg_mutations, np.mean(fitnesses, axis=0),
                    yerr=np.std(fitnesses, axis=0), fmt='o', capsize=5, label='Average Pareto Front', color='#1f77b4', ecolor='black')
    plt.xlabel('Number of Mutations', fontsize=25)
    plt.ylabel('Normalized DeepCRE Output', fontsize=25)
    plt.title('Average Pareto Front', fontsize=30)
    run_name = os.path.basename(results_folder)
    plt.savefig(os.path.join(output_folder, f'average_pareto_front_{run_name}.{output_format}'), bbox_inches='tight', dpi=1000)


def calculate_loss_pareto_front(current_front: List[Tuple[str, float, int]], target_front: List[Tuple[str, float, int]], max_number_mutation: int) -> float:
    """Calculate the loss between the current pareto front and the target pareto front.

    Args:
        current_front (List[Tuple[str, float, int]]): The current pareto front.
        target_front (List[Tuple[str, float, int]]): The target pareto front.

    Returns:
        float: The loss value.
    """
    target_length = max_number_mutation + 1
    if len(current_front) > target_length or len(target_front) > target_length:
        raise ValueError(f"Both pareto fronts must at most have a length of {target_length}! Found {len(current_front)} and {len(target_front)}.")
    
    current_expanded = expand_pareto_front(current_front, max_number_mutation=max_number_mutation)
    target_expanded = expand_pareto_front(target_front, max_number_mutation=max_number_mutation)
    loss = 0.0
    for curr, target in zip(current_expanded, target_expanded):
        if curr[2] != target[2]:
            raise ValueError(f"Number of mutations must be aligned between both pareto fronts! Found {curr[2]} and {target[2]}.")
        loss += target[1] - curr[1]
    return abs(loss)

def calculate_loss_over_generations_single_gene(results_folder: str, gene: str, max_number_mutation: int, last_generation: int = 1999) -> Dict[int, float]:
    """Calculate the loss over generations for a single gene.

    Args:
        results_folder (str): Path to the results folder.
        gene (str): The gene to analyze.
        last_generation (int, optional): The last generation to consider. Defaults to 2000.

    Returns:
        Dict[int, float]: A dictionary mapping generation numbers to loss values.
    """
    loss_over_generations = {}
    saved_populations = os.listdir(os.path.join(results_folder, gene, 'saved_populations'))
    pareto_fronts = [f for f in saved_populations if f.startswith('pareto_front_gen')]
    last_front = os.path.join(results_folder, gene, 'saved_populations', "pareto_front.json")
    with open(last_front, 'r') as f:
        last_front = json.load(f)

    for generation in pareto_fronts:
        gen_num = int(generation.split('_')[-1].split('.')[0])
        if gen_num > last_generation:
            print(f"Skipping generation {gen_num}, exceeds last generation {last_generation}.")
            continue
        current_front_path = os.path.join(results_folder, gene, 'saved_populations', generation)
        with open(current_front_path, 'r') as f:
            current_front = json.load(f)
        
        loss = calculate_loss_pareto_front(current_front, last_front, max_number_mutation=max_number_mutation)
        loss_over_generations[gen_num] = loss
    loss_over_generations[last_generation] = 0
    return loss_over_generations


def calculate_loss_over_generations(results_folder: str, max_number_mutation: int, last_generation: int = 1999) -> Dict[str, Dict[int, float]]:
    """Calculate the loss over generations for all genes in the results folder.

    Args:
        results_folder (str): Path to the results folder.
        last_generation (int, optional): The last generation to consider. Defaults to 2000.

    Returns:
        Dict[str, Dict[int, float]]: A dictionary mapping gene names to dictionaries of generation numbers and loss values.
    """
    loss_over_generations = {}
    genes = [gene for gene in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, gene))]
    
    for gene in tqdm(genes):
        try:
            loss_over_generations[gene] = calculate_loss_over_generations_single_gene(results_folder, gene, max_number_mutation=max_number_mutation, last_generation=last_generation)
        except Exception as e:
            print(f"Error processing gene {gene}: {e}")
    
    return loss_over_generations


def join_losses_for_visualization(loss_data: Dict[str, Dict[int, float]]) -> Tuple[List[int], List[float], List[float]]:
    full_losses = {}
    for loss_dict in loss_data.values():
        for gen, loss in loss_dict.items():
            if gen not in full_losses:
                full_losses[gen] = []
            full_losses[gen].append(loss)

    generations = sorted(full_losses.keys())
    losses = [full_losses[gen] for gen in generations]
    loss_averages = [np.mean(loss) for loss in losses]
    loss_stds = [np.std(loss) for loss in losses]
    return generations, loss_averages, [float(x) for x in loss_stds]


def plot_loss_over_generations(results_folder: str, name: str, max_number_mutation: int, output_format: str, last_generation: int = 1999, output_folder: str = ".") -> None:
    """Plot the average loss over generations for all genes.

    Args:
        results_folder (str): Path to the results folder.
        name (str): Name to distinguish the output file.
        last_generation (int, optional): The last generation to consider. Defaults to 1999.
        output_folder (str): Path to the output folder for saving results. Defaults to ".".
    """
    loss_data = calculate_loss_over_generations(results_folder=results_folder, last_generation=last_generation, max_number_mutation=max_number_mutation)
    generations, loss_averages, loss_stds = join_losses_for_visualization(loss_data)
    """
    random_gene = list(loss_data.keys())[0]
    generations = sorted(loss_data[random_gene].keys())
    losses =  []
    for gen in generations:
        loss_curr_gen = []
        for loss_per_gen in loss_data.values():
            loss_curr_gen.append(loss_per_gen[gen])
        losses.append(loss_curr_gen)
    losses = np.array(losses)
    avg_loss_per_gen = np.mean(losses, axis=1)
    std_loss_per_gen = np.std(losses, axis=1)
    loss_for_vis = [np.nan for _ in range(len(std_loss_per_gen))]
    for i in range(100, len(std_loss_per_gen), 20):
        loss_for_vis[i] = std_loss_per_gen[i]
    """
    # print(loss_for_vis)
    # print(std_loss_per_gen)
    plt.clf()
    plt.figure()
    # plot average loss
    # plot error bars every 100 generations
    # plt.plot(generations, avg_loss_per_gen, label='Average Loss', color='blue')
    plt.errorbar(generations, loss_averages,
                 yerr=loss_stds, capsize=3, label='Average Loss', ecolor='black')
    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.title("Average Loss Over Generations")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"loss_over_generations_{name}.{output_format}"), bbox_inches='tight')
    plt.close()


def plot_half_max_mutations_vs_initial_fitness(stats: Dict[str, Dict[str, Any]], name: str, output_format: str, output_folder: str = ".") -> None:
    """Plot the number of mutations at half max fitness against the initial fitness.

    Args:
        stats (Dict[str, Dict[str, Any]]): Dictionary containing statistics for each gene.
        name (str): Name to distinguish the output file.
        output_folder (str): Path to the output folder for saving results. Defaults to ".".
    """
    plt.clf()
    
    initial_fitness = []
    half_max_mutations = []
    
    for stat in stats.values():
        if 'num_mutations_half_max_effect' in stat:
            initial_fitness.append(stat['start_fitness'])
            half_max_mutations.append(stat['num_mutations_half_max_effect'])
        else:
            print(f"Skipping gene {stat['origin_chromosome']}, no mutations at half max effect found.")

    plt.scatter(initial_fitness, half_max_mutations, alpha=0.6)
    plt.xlabel('Initial Fitness')
    plt.ylabel('Mutations at Half Max Effect')
    plt.title('Initial Fitness vs Mutations at Half Max Effect')
    plt.savefig(os.path.join(output_folder, f'half_max_mutations_vs_initial_fitness_{name}.{output_format}'), bbox_inches='tight')


def hist_half_max_mutations(stats: Dict[str, Dict[str, Any]], name: str, output_format: str, output_folder: str = ".") -> None:
    """Create a histogram of the number of mutations at half max effect.

    Args:
        stats (Dict[str, Dict[str, Any]]): Dictionary containing statistics for each gene.
        name (str): Name to distinguish the output file.
        output_folder (str): Path to the output folder for saving results. Defaults to ".".
    """
    plt.clf()
    
    half_max_mutations = []
    
    for stat in stats.values():
        if 'num_mutations_half_max_effect' in stat:
            half_max_mutations.append(stat['num_mutations_half_max_effect'])
        else:
            print(f"Skipping gene {stat['origin_chromosome']}, no mutations at half max effect found.")

    max_mutations = round(max(half_max_mutations)) if half_max_mutations else 0
    bins = np.arange(0, max_mutations + 2) - 0.5  # to center bins on integers
    
    plt.hist(half_max_mutations, bins=bins, alpha=0.7) #type:ignore
    plt.xlabel('Mutations at Half Max Effect', fontsize=25)
    plt.ylabel('Frequency', fontsize=25)
    plt.title('Histogram of Mutations at Half Max Effect', fontsize=30)
    plt.savefig(os.path.join(output_folder, f'hist_half_max_mutations_{name}.{output_format}'), bbox_inches='tight', dpi=1000)


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze evolution results and generate visualizations')
    parser.add_argument('--results_folder', "-r", help='Path to the results folder containing gene directories', required=False, default=None)
    parser.add_argument('--name', "-n", help='name to distinguish output files')
    parser.add_argument('--stats_file', "-s", help='path to an already saved stats file, if not provided, it will be generated', default=None, required=False)
    parser.add_argument('--output_folder', "-o", help='Path to the output folder for saving results', default='.', required=False)
    parser.add_argument("--number", "-N", type=int, default=4, help="Number of random pareto fronts to draw from the results folder. Default is 4.")
    parser.add_argument("--max_number_mutation", "-m", type=int, default=90, help="Maximum number of mutations to consider when expanding the pareto front. Default is 90.")
    parser.add_argument("--output_format", "-f", type=str, default="png", help="Output format for plots (e.g., pdf, png). Default is png.")
    
    # Analysis control flags - explicit inclusion
    parser.add_argument('--summary', action='store_true', help='Generate and print summary statistics')
    parser.add_argument('--fitness_plot', action='store_true', help='Generate start vs max fitness plot')
    parser.add_argument('--mutations_plot', action='store_true', help='Generate mutations colored plot')
    parser.add_argument('--plot_random_pareto', action='store_true', help='Draw random pareto fronts from the results folder')
    parser.add_argument('--plot_average_pareto', action='store_true', help='Draw average pareto front from the results folder')
    parser.add_argument('--plot_average_loss', action='store_true', help='Draw average loss over generations from the results folder')
    parser.add_argument('--plot_half_max_mutations', action='store_true', help='Draw half max mutations vs initial fitness from the results folder')
    parser.add_argument('--plot_half_max_mutations_hist', action='store_true', help='Draw histogram of half max mutations from the results folder')
    parser.add_argument('--all', action='store_true', help='Run all analysis steps')
    
    args = parser.parse_args()
    if args.results_folder is None and args.stats_file is None:
        raise ValueError("You must provide either a results folder or a stats file.")
    args.output_format = args.output_format.lower().strip('.').strip()
    if args.output_format not in ['png', 'pdf', 'jpg', 'jpeg', 'svg', "tiff"]:
        raise ValueError(f"Unsupported output format: {args.output_format}. Supported formats are png, pdf, jpg, jpeg, svg.")
    return args


def main():
    args = parse_args()
    
    # Determine which functions to run based on explicit flags
    run_summary = args.summary or args.all
    run_fitness_plot = args.fitness_plot or args.all
    run_mutations_plot = args.mutations_plot or args.all
    run_random_pareto = args.plot_random_pareto or args.all
    run_average_pareto = args.plot_average_pareto or args.all
    run_half_max_mutations = args.plot_half_max_mutations or args.all
    run_hist_half_max_mutations = args.plot_half_max_mutations_hist or args.all
    plot_average_loss = args.plot_average_loss or args.all

    if not (run_summary or run_fitness_plot or run_mutations_plot or run_half_max_mutations or run_hist_half_max_mutations or run_random_pareto or run_average_pareto or plot_average_loss):
        raise ValueError("No analysis step was selected to run, so nothing is done. Use --all to run all steps or use -h to get an overview of available options.")
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    if run_summary or run_fitness_plot or run_mutations_plot or run_half_max_mutations or run_hist_half_max_mutations:
        if args.stats_file:
            with open(args.stats_file, 'r') as f:
                stats = json.load(f)
        else:
            stats = get_stats_per_gene(args.results_folder, args.name, args.output_folder)

    if run_summary:
        try:
            print("Summarizing stats...")
            summarize_stats(stats=stats, name=args.name, output_folder=args.output_folder)
            print("Stats summarized successfully.")
        except Exception as e:
            print("failed to summarize stats.")
            print(f"Error summarizing stats: {e}")

    if run_fitness_plot:
        try:
            print("Visualizing start vs max fitness...")
            visualize_start_vs_max_fitness(stats=stats, name=args.name, output_folder=args.output_folder, output_format=args.output_format)
            print("Start vs max fitness visualization completed successfully.")
        except Exception as e:
            print("failed to visualize start vs max fitness.")
            print(f"Error while plotting fitness: {e}")

    if run_mutations_plot:
        try:
            print("Visualizing start vs max fitness by mutations...")
            visualize_start_vs_max_fitness_by_mutations(stats=stats, name=args.name, output_folder=args.output_folder, output_format=args.output_format)
            print("Start vs max fitness by mutations visualization completed successfully.")
        except Exception as e:
            print("failed to visualize start vs max fitness by mutations.")
            print(f"Error while plotting mutations: {e}")

    if run_half_max_mutations:
        try:
            print("Plotting half max mutations vs initial fitness...")
            plot_half_max_mutations_vs_initial_fitness(stats=stats, name=args.name, output_format=args.output_format, output_folder=args.output_folder)
            print("Half max mutations vs initial fitness plot completed successfully.")
        except Exception as e:
            print("failed to plot half max mutations vs initial fitness.")
            print(f"Error while plotting half max mutations: {e}")

    if run_hist_half_max_mutations:
        try:
            print("Plotting histogram of half max mutations...")
            hist_half_max_mutations(stats=stats, name=args.name, output_format=args.output_format, output_folder=args.output_folder)
            print("Histogram of half max mutations plot completed successfully.")
        except Exception as e:
            print("failed to plot histogram of half max mutations.")
            print(f"Error while plotting histogram of half max mutations: {e}")

    if run_random_pareto:
        try:
            print("Drawing random pareto fronts...")
            if not args.results_folder:
                raise ValueError("You must provide a results folder to draw random pareto fronts.")
            show_random_fronts(results_folder=args.results_folder, num_samples=args.number, output_folder=args.output_folder, output_format=args.output_format)
            print("Random pareto fronts drawn successfully.")
        except Exception as e:
            print("failed to draw random pareto fronts.")
            print(f"Error while plotting random pareto fronts: {e}")
    
    if run_average_pareto:
        try:
            print("Drawing average pareto front...")
            if not args.results_folder:
                raise ValueError("You must provide a results folder to draw the average pareto front.")
            show_average_pareto_front(results_folder=args.results_folder, output_format=args.output_format, output_folder=args.output_folder, max_number_mutation=args.max_number_mutation)
            print("Average pareto front drawn successfully.")
        except Exception as e:
            print("failed to draw average pareto front.")
            print(f"Error while plotting average pareto front: {e}")

    if plot_average_loss:
        try:
            print("Plotting average loss...")
            if not args.results_folder:
                raise ValueError("You must provide a results folder to draw the average loss.")
            plot_loss_over_generations(results_folder=args.results_folder, name=args.name, output_folder=args.output_folder, max_number_mutation=args.max_number_mutation, output_format=args.output_format)
            print("Average loss plot completed successfully.")
        except Exception as e:
            print("failed to plot average loss.")
            print(f"Error while plotting average loss: {e}")


if __name__ == "__main__":
    main()
    # result = calculate_loss_over_generations_single_gene(results_folder="/home/gernot/Code/PhD_Code/Evolution/results/diverse_genes_arabidopsis_250611_105634_010904", gene="1_AT1G09440_gene:3048006-3045258_250611_121447_109863")
    # result = [(k, v) for k, v in result.items()]
    # result = sorted(result, key=lambda x: x[0])
    # # plot results
    # plt.clf()
    # plt.figure()
    # # add grid in bg
    # plt.plot([k for k, v in result], [v for k, v in result])
    # plt.grid()
    # plt.xlabel('Generation')
    # plt.ylabel('Loss')
    # plt.title('Loss over Generations for Gene 1_AT1G09440')
    # plt.savefig('loss_over_generations.png', bbox_inches='tight')
    # plot_loss_over_generations(results_folder="/home/gernot/Code/PhD_Code/Evolution/results/diverse_genes_arabidopsis_250611_105634_010904", name='arabidopsis')
