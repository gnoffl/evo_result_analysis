import os
import argparse
import json
import bisect
import numpy as np
import pandas as pd
from pyfaidx import Fasta
import re
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union

from analysis.simple_result_stats import calculate_half_max_mutations, get_min_mutation_count_for_fitness

def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement[base] for base in reversed(seq))

def get_restriction_regexes(restriction_site_path: str) -> Dict[str, re.Pattern]:
    restriction_sites = Fasta(restriction_site_path)
    restriction_regexes = {}
    for record in restriction_sites:
        site_name = record.name
        site_seq = str(record).upper()
        site_regex = site_seq.replace('N', '[ACGTN]')
        restriction_regexes[site_name] = re.compile(site_regex)
        
        # Also add reverse complement
        site_seq_revcomp = reverse_complement(site_seq)
        site_regex_revcomp = site_seq_revcomp.replace('N', '[ACGTN]')
        restriction_regexes[f"{site_name}_reverse"] = re.compile(site_regex_revcomp)
    return restriction_regexes

def get_sequences_without_restriction_sites(sequence_path: str, restriction_site_path: str, output_path: str,
                                         prefixes: Optional[Dict[str, str]] = None, postfixes: Optional[Dict[str, str]] = None) -> None:
    restriction_regexes = get_restriction_regexes(restriction_site_path)
    sequences = Fasta(sequence_path)
    keepers = {}
    for sequence in sequences:
        seq_str = str(sequence).upper()
        seq_name = sequence.name
        if prefixes is not None and seq_name in prefixes:
            seq_str = prefixes[seq_name] + seq_str
        if postfixes is not None and seq_name in postfixes:
            seq_str = seq_str + postfixes[seq_name]
        found_sites = []
        for site_name, site_regex in restriction_regexes.items():
            match = site_regex.search(seq_str)
            if match:
                found_sites.append((site_name, match.start() + 1, match.group(0)))
        if found_sites:
            print(f"restriction sites found in sequence {seq_name}: {found_sites}")
        else:
            keepers[seq_name] = seq_str
    with open(output_path, 'w') as outfile:
        print(f"Writing {len(keepers)} sequences without restriction sites out of {len(sequences.keys())} sequences to {output_path}")
        for name, seq in keepers.items():
            outfile.write(f">{name}\n{seq}\n")

def get_data_at_mutation_count(pareto_front: List[Tuple[str, float, float]], target_mutation_count: int) -> Tuple[str, float, float]:
    # implement binary search for the mutation count
    mutation_counts = [round(item[2]) for item in pareto_front]
    mutation_counts_asc = mutation_counts[::-1]
    index = bisect.bisect_left(mutation_counts_asc, target_mutation_count)
    if index < len(pareto_front) and mutation_counts_asc[index] == target_mutation_count:
        reversed_index = len(pareto_front) - 1 - index
        return pareto_front[reversed_index]
    else:
        raise ValueError(f"Mutation count {target_mutation_count} not found in pareto front.")

def get_pareto_front_paths(results_folder: str) -> List[str]:
    if not os.path.exists(results_folder):
        raise FileNotFoundError(f"The specified results folder does not exist: {results_folder}")

    gene_folders = [os.path.join(results_folder, d) for d in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, d))]
    pareto_paths = [os.path.join(gene_folder, "saved_populations", "pareto_front.json") for gene_folder in gene_folders]
    pareto_paths = [p for p in pareto_paths if os.path.isfile(p)]
    return pareto_paths

def get_gene_name_from_path(pareto_path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.dirname(pareto_path))).split("_")[1]

def process_mutations(pareto_front: List[Tuple[str, float, float]]) -> Dict[str, Optional[float]]:
    mutation_data = {}
    start_fitness = pareto_front[-1][1]
    for mutations in [1, 2, 3, 5, 7, 10, 15, 20]:
        try:
            sequence, fitness, _ = get_data_at_mutation_count(pareto_front, mutations)
            mutation_data[f"fitness_delta_for_{mutations}_mutations"] = abs(fitness - start_fitness)
            mutation_data[f"sequence_for_{mutations}_mutations"] = sequence
        except ValueError:
            mutation_data[f"fitness_delta_for_{mutations}_mutations"] = np.nan
            mutation_data[f"sequence_for_{mutations}_mutations"] = ""
    return mutation_data

def process_fitness_thresholds(pareto_front: List[Tuple[str, float, float]]) -> Dict[str, List[float]]:
    fitness_data = {}
    min_fitness = min(pareto_front[0][1], pareto_front[-1][1])
    max_fitness = max(pareto_front[0][1], pareto_front[-1][1])
    for fitness in np.arange(0, 1.01, 0.1):
        if fitness < min_fitness or fitness > max_fitness:
            mutation_count = np.nan
            sequence = ""
        else:
            mutation_count = get_min_mutation_count_for_fitness(pareto_front, fitness)
            try:
                sequence, _, _ = get_data_at_mutation_count(pareto_front, round(mutation_count))
            except ValueError:
                sequence = ""
        fitness_data[f"mutations_for_fitness_{fitness:.1f}"] = mutation_count
        fitness_data[f"sequence_for_fitness_{fitness:.1f}"] = sequence
    return fitness_data

def process_single_gene(pareto_path: str) -> Tuple[Dict, bool]:
    gene_name = get_gene_name_from_path(pareto_path)
    with open(pareto_path, 'r') as file:
        pareto_front = json.load(file)
    start_fitness = pareto_front[-1][1]
    final_fitness = pareto_front[0][1]
    max_num_mutations = pareto_front[0][2]
    half_max_mutations = calculate_half_max_mutations(pareto_front)

    gene_data = {
        "gene": gene_name,
        "reference_sequence": pareto_front[-1][0],
        "max_num_mutations": max_num_mutations,
        "start_fitness": start_fitness,
        "final_fitness": final_fitness,
        "half_max_mutations": half_max_mutations,
    }
    gene_data.update(process_mutations(pareto_front=pareto_front))
    gene_data.update(process_fitness_thresholds(pareto_front=pareto_front))
    return gene_data, final_fitness > start_fitness


def filter_and_sort_results(selected_df: pd.DataFrame, starting_fitness_max: Optional[float], starting_fitness_min: Optional[float], ascending: bool) -> pd.DataFrame:
    if starting_fitness_max is not None:
        selected_df = selected_df[selected_df["start_fitness"] <= starting_fitness_max]
    if starting_fitness_min is not None:
        selected_df = selected_df[selected_df["start_fitness"] >= starting_fitness_min]
    selected_df = selected_df.sort_values(by="start_fitness", ascending=ascending)
    return selected_df

def preliminary_selection(results_folder: str, output_path: str, starting_fitness_max: Optional[float] = None, starting_fitness_min: Optional[float] = None) -> None:
    """
    Select genes with the lowest starting values from analysis results.
    """
    pareto_paths = get_pareto_front_paths(results_folder)

    all_gene_data = []
    for pareto_path in pareto_paths:
        gene_data, increasing = process_single_gene(pareto_path)
        all_gene_data.append(gene_data)

    selected_df = pd.DataFrame(all_gene_data).set_index("gene")
    selected_df = filter_and_sort_results(selected_df, starting_fitness_max, starting_fitness_min, ascending=increasing)
    print(selected_df.describe())
    selected_df.to_csv(output_path)



def load_selected_genes(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"The specified CSV file does not exist: {csv_path}")
    selected = pd.read_csv(csv_path, index_col="gene")
    return selected

def draw_line_plot(vis_folder, gene, single_points, multi_points):
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

def get_line_plot_data(row):
    mutation_cols_single = [col for col in row.index if col.startswith("mutations_for_fitness_") and col.endswith("_single")]
    mutation_cols_multi = [col for col in row.index if col.startswith("mutations_for_fitness_") and col.endswith("_multi")]
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
    return single_points,multi_points


def compare_trajectories_mutations(merged: pd.DataFrame):
    vis_folder = os.path.join("visualization", "GOF_LOF", "GOF", "compare_trajectories")
    os.makedirs(vis_folder, exist_ok=True)
    for gene , row in merged.iterrows():
        single_points, multi_points = get_line_plot_data(row)
        draw_line_plot(vis_folder, gene, single_points, multi_points)

def get_best_fitness_and_seq(row, mutations, single_col, multi_col) -> Tuple[float, str]:
    if pd.isna(row[single_col]) and pd.isna(row[multi_col]):
        return np.nan, ""
    elif pd.isna(row[single_col]):
        return row[multi_col], row[f"sequence_for_{mutations}_mutations_multi"]
    elif pd.isna(row[multi_col]):
        return row[single_col], row[f"sequence_for_{mutations}_mutations_single"]
    else:
        if row[single_col] > row[multi_col]:
            return row[single_col], row[f"sequence_for_{mutations}_mutations_single"]
        else:
            return row[multi_col], row[f"sequence_for_{mutations}_mutations_multi"]

def find_best_mutation_deltas(merged: pd.DataFrame):
    for mutations in [1, 2, 3, 5, 7, 10, 15, 20]:
        best_fitnesses, best_sequences = [], []
        single_col = f"fitness_delta_for_{mutations}_mutations_single"
        multi_col = f"fitness_delta_for_{mutations}_mutations_multi"
        for gene, row in merged.iterrows():
            best_fitness, best_sequence = get_best_fitness_and_seq(row=row, mutations=mutations, single_col=single_col, multi_col=multi_col)
            best_fitnesses.append(best_fitness)
            best_sequences.append(best_sequence)
        merged[f"best_fitness_delta_for_{mutations}_mutations"] = best_fitnesses
        merged[f"per_mutation_delta_for_{mutations}_mutations"] = [val/mutations if not pd.isna(val) else np.nan for val in best_fitnesses]
        merged[f"best_sequence_for_{mutations}_mutations"] = best_sequences

def get_sorted_values(row, per_mutation_columns: List[str], min_delta: float) -> List[Tuple[str, float]]:
    values = []
    for col in per_mutation_columns:
        if pd.isna(row[col]):
            values.append((col, -np.inf))
            continue
        mutation_count = int(col.split("_")[-2])
        total_delta = row[f"best_fitness_delta_for_{mutation_count}_mutations"]
        if pd.isna(total_delta):
            values.append((col, -np.inf))
            continue
        if total_delta >= min_delta - 1e-6:
            values.append((col, row[col]))
        else:
            values.append((col, -np.inf))
    values = sorted(values, key=lambda x: x[1], reverse=True)
    return values

def find_best_mutation_count_per_gene(row: pd.Series, rank: int = 1, min_delta: float = 0.3) -> Union[int, float]:
    per_mutation_columns = [col for col in row.index if col.startswith("per_mutation_delta_for_") and col.endswith("_mutations")]
    sorted_values = get_sorted_values(row, per_mutation_columns, min_delta)
    if rank > len(sorted_values):
        return np.nan
    best_col_tuple = sorted_values[rank - 1]
    mutation_count = int(best_col_tuple[0].split("_")[-2])
    return mutation_count if row[f"best_fitness_delta_for_{mutation_count}_mutations"] >= min_delta - 1e-6 else np.nan

def find_best_mutations_per_gene(merged: pd.DataFrame, n_candidates: int):
    for results_rank in range(1, n_candidates + 1):
        merged[f"best_mutation_count_rank_{results_rank}"] = merged.apply(lambda row: find_best_mutation_count_per_gene(row, rank=results_rank), axis=1)

def selected_per_mutation(merged: pd.DataFrame, n_candidates: int) -> pd.DataFrame:
    maximization = merged.iloc[0]["final_fitness_single"] > merged.iloc[0]["start_fitness_single"]
    mutations = [1, 2, 3, 5, 7, 10, 15, 20]
    best_deltas, best_delta_genes, best_delta_sequences, references, mutation_col, start_fitnesses, fitnesses = [], [], [], [], [], [], []
    rel_cols = [col for col in merged.columns if col.startswith("best_fitness_delta_for_") or col.startswith("best_sequence_for_") or (col.startswith("sequence_for_") and col.endswith("_mutations")) or col.startswith("reference")]
    relevant_merged = merged[rel_cols]
    best_delta_cols = [f"best_fitness_delta_for_{m}_mutations" for m in mutations]
    for col in best_delta_cols:
        curr_df = relevant_merged.sort_values(by=col, ascending=False).iloc[:n_candidates]
        for gene in curr_df.index:
            mutation_col.append(int(col.split("_")[-2]))
            best_deltas.append(curr_df.loc[gene, col])
            best_delta_genes.append(gene)
            best_delta_sequences.append(curr_df.loc[gene, col.replace("best_fitness_delta", "best_sequence")])
            references.append(curr_df.loc[gene, "reference_sequence_single"])
            start_fitnesses.append(merged.loc[gene, "start_fitness_single"])
            fitness = start_fitnesses[-1] + best_deltas[-1] if maximization else start_fitnesses[-1] - best_deltas[-1]
            fitnesses.append(fitness)
    summary = pd.DataFrame({
        "gene": best_delta_genes,
        "mutations": mutation_col,
        "start_fitness": start_fitnesses,
        "best_fitness_delta": best_deltas,
        "fitness": fitnesses,
        "best_sequence": best_delta_sequences,
        "reference_sequence": references,
    })
    summary["selection_reason"] = "top_per_mutation"
    return summary


def selected_per_gene(merged: pd.DataFrame, n_candidates: int) -> pd.DataFrame:
    best_deltas, best_delta_genes, best_delta_sequences, references, mutation_col, start_fitnesses, fitnesses = [], [], [], [], [], [], []
    for gene, row in merged.iterrows():
        for mutation_col_rank in [f"best_mutation_count_rank_{i}" for i in range(1, n_candidates + 1)]:
            best_mutation_count = float(row[mutation_col_rank])
            if pd.isna(best_mutation_count):
                break
            best_mutation_count = round(best_mutation_count)
            best_deltas.append(row[f"best_fitness_delta_for_{best_mutation_count}_mutations"])
            best_delta_genes.append(gene)
            best_delta_sequences.append(row[f"best_sequence_for_{best_mutation_count}_mutations"])
            references.append(row["reference_sequence_single"])
            start_fitnesses.append(row["start_fitness_single"])
            fitness = start_fitnesses[-1] + best_deltas[-1] if row["final_fitness_single"] > row["start_fitness_single"] else start_fitnesses[-1] - best_deltas[-1]
            fitnesses.append(fitness)
            mutation_col.append(best_mutation_count)
    summary = pd.DataFrame({
        "gene": best_delta_genes,
        "mutations": mutation_col,
        "start_fitness": start_fitnesses,
        "best_fitness_delta": best_deltas,
        "fitness": fitnesses,
        "best_sequence": best_delta_sequences,
        "reference_sequence": references,
    })
    summary["selection_reason"] = "top_per_gene"
    return summary


def combine_selections(per_gene_bests: pd.DataFrame, per_mutation_bests: pd.DataFrame) -> pd.DataFrame:
    per_mutation_bests = per_mutation_bests.copy()
    for i, row in per_mutation_bests.iterrows():
        gene, mutations = row["gene"], row["mutations"]
        per_gene_rows = per_gene_bests[(per_gene_bests["gene"] == gene) & (per_gene_bests["mutations"] == mutations)]
        if not per_gene_rows.empty:
            per_mutation_bests.at[i, "selection_reason"] = "both"
            per_gene_bests = per_gene_bests.drop(per_gene_rows.index)
    final_selection = pd.concat([per_gene_bests, per_mutation_bests]).drop_duplicates(subset=["gene", "mutations"]).reset_index(drop=True)
    return final_selection


def final_selection(single_path: str, multi_path: str, output_path: str, n_candidates: int, min_delta: float = 0.3) -> None:
    EPSILON = 1e-6
    selected_single = load_selected_genes(single_path)
    selected_multi = load_selected_genes(multi_path)
    merged = selected_single.merge(selected_multi, left_index=True, right_index=True, suffixes=("_single", "_multi"))
    find_best_mutation_deltas(merged)
    find_best_mutations_per_gene(merged, n_candidates=n_candidates)
    per_gene_bests = selected_per_gene(merged, n_candidates=n_candidates)
    per_gene_bests = per_gene_bests[per_gene_bests["best_fitness_delta"] >= min_delta - EPSILON]
    per_mutation_bests = selected_per_mutation(merged, n_candidates=n_candidates)
    final_selection = combine_selections(per_gene_bests, per_mutation_bests)
    final_selection.to_csv(output_path, index=False)


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
    parser.add_argument("--n_candidates", "-n", type=int, default=3, help="Number of candidate genes to select per mutation count and per gene.")
    parser.add_argument("--single_path", "-s", type=str, help="Path to the CSV file with single mutation results.")
    parser.add_argument("--multi_path", "-m", type=str, help="Path to the CSV file with multi mutation results.")
    parser.add_argument("--starting_fitness_max", "-max", type=float, default=None, help="Maximum starting fitness to filter genes.")
    parser.add_argument("--starting_fitness_min", "-min", type=float, default=None, help="Minimum starting fitness to filter genes.")
    parser.add_argument("--final_selection", "-f", action="store_true", help="Perform final selection step.")
    parser.add_argument("--preliminary_selection", "-p", action="store_true", help="Perform preliminary selection step.")
    return parser.parse_args()

if __name__ == "__main__":
    # PARAMETERS FOR GOF CANDIDATE SELECTION
    # final_selection(single_path="/home/gernot/ARCitect/ARCs/genRE/assays/Evolution/protocols/paper/analysis/GOF_LOF/GOF/GOF_single_mutation/candidate_genes_GOF_single.csv",
    #                 multi_path="/home/gernot/ARCitect/ARCs/genRE/assays/Evolution/protocols/paper/analysis/GOF_LOF/GOF/GOF_multi_mutation/candidate_genes_GOF_multi.csv",
    #                 output_path="data/selected_genes_GOF.csv",
    #                 n_candidates=3)
    args = parse_args()
    if not args.final_selection and not args.preliminary_selection:
        raise ValueError("Either --final_selection (-f) or --preliminary_selection (-p) must be specified.")
    if args.final_selection and args.preliminary_selection:
        raise ValueError("Only one of --final_selection (-f) or --preliminary_selection (-p) can be specified at a time.")
    if args.preliminary_selection:
        preliminary_selection(args.results_folder, args.output_path, args.starting_fitness_max, args.starting_fitness_min)
    if args.final_selection:
        final_selection(single_path=args.single_path,
                        multi_path=args.multi_path,
                        output_path=args.output_path,
                        n_candidates=args.n_candidates)