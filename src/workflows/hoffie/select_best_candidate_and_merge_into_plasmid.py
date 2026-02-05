import os
import json
from pyfaidx import Fasta

def get_run_paths():
    data_path = os.path.join(os.path.dirname(__file__), "data", "ubi", "run_results")
    run_paths = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    return run_paths

def find_best_sequence():
    run_paths = get_run_paths()
    differences = {}
    for run_path in run_paths:
        for gene in ["T01", "T02", "T03"]:
            with open(os.path.join(run_path, gene, "pareto_front.json"), 'r') as f:
                sequences = json.load(f)
            optimized_fitness = sequences[0][1]
            starting_fitness = sequences[-1][1]
            difference = abs(optimized_fitness - starting_fitness)
            differences[gene] = differences.get(gene, 0) + difference
    
    print("Total fitness differences per gene across all runs:")
    for gene, total_diff in differences.items():
        print(f"  Gene: {gene}, Total Difference: {total_diff}")


def merge_sequences_into_plasmid(selected_sequence: str = "T03"):
    plasmid = Fasta(os.path.join(os.path.dirname(__file__), "data", "ubi", "pIK74.fa"))
    sequence = plasmid["PLASMID"]
    plasmid_start = sequence[:1900]
    plasmid_end = sequence[3302:]
    run_paths = get_run_paths()
    pareto_paths = [os.path.join(run_path, selected_sequence, "pareto_front.json") for run_path in run_paths]
    content = {}
    reference_head = ">5_GRMZM2G409726_T03_gene:84403652-84400792_reference"

    for pareto_path in pareto_paths:
        with open(pareto_path, 'r') as f:
            sequences = json.load(f)
        optimized_sequence, fitness, mutations = sequences[0]
        reference, _, _ = sequences[-1]
        # sequences contain trailing N that needs to be removed
        optimized_sequence = optimized_sequence.rstrip('N')
        reference = reference.rstrip('N')
        print(len(optimized_sequence), len(reference), len(plasmid_start), len(plasmid_end)) #type:ignore
        merged_sequence = str(plasmid_start) + optimized_sequence + str(plasmid_end)
        merged_reference = str(plasmid_start) + reference + str(plasmid_end)
        print(len(merged_sequence), len(merged_reference))
        run = os.path.basename(os.path.dirname(os.path.dirname(pareto_path)))
        content[f">5_GRMZM2G409726_T03_gene:84403652-84400792_{run}_{round(fitness, 5)}_{mutations:03}"] = merged_sequence
        if reference_head in content:
            prev_reference = content[reference_head]
            if merged_reference != prev_reference:
                raise ValueError(f"Warning: Reference sequences do not match for run {run}")
        else:
            content[reference_head] = merged_reference
    
    output_file = os.path.join(os.path.dirname(__file__), "data", "ubi", "merged_plasmids.fa")
    with open(output_file, 'w') as out:
        seq = content.pop(reference_head)
        print(len(seq))
        out.write(f"{reference_head}\n{seq}\n")
        for header, seq in content.items():
            print(len(seq))
            out.write(f"{header}\n{seq}\n")

if __name__ == "__main__":
    # find_best_sequence()
    merge_sequences_into_plasmid()