import argparse
import json
import os
from typing import Dict, List, Optional, Tuple
from pyfaidx import Fasta
import re
from tqdm import tqdm


class MutatedSequence:
    reference_sequence: str
    mutations: List[Tuple[int, str, str]]  # List of tuples (position, reference_base, mutated_base)
    mutated_sequence: str
    fitness: float

    def __init__(self, reference_sequence: str, mutated_sequence: str, fitness: float) -> None:
        self.reference_sequence = reference_sequence
        self.mutations = []
        self.mutated_sequence = mutated_sequence
        self.fitness = fitness
        self._find_mutations()

    def _find_mutations(self) -> None:
        if len(self.mutated_sequence) != len(self.reference_sequence):
            raise ValueError("Mutated sequence length does not match reference sequence length.")
        for i, (ref_base, mut_base) in enumerate(zip(self.reference_sequence, self.mutated_sequence)):
            if ref_base != mut_base:
                self.mutations.append((i, ref_base, mut_base))

    def _parse_string(self, mutation_string: str):
        mutations_dict = {}
        mutation_string = mutation_string.strip()
        substrings = mutation_string.split('|')
        if len(substrings) != 2:
            raise ValueError(f"mutation string did not contain exactly one '|' character: {mutation_string}")
        mutation_string = substrings[0].strip()  # Get the part before the '|
        fitness_str = substrings[1].strip()
        fitness = float(fitness_str)
        self.fitness = fitness

        regex = r"(\d+)([ACGT])([ACGT])"
        regex = re.compile(regex)
        while match := regex.match(mutation_string):
            position = int(match.group(1))
            ref_base = match.group(2)
            mut_base = match.group(3)
            if position in mutations_dict:
                if mutations_dict[position] != (ref_base, mut_base):
                    raise ValueError(f"Position {position} has conflicting mutations: {mutations_dict[position]} vs {ref_base}->{mut_base}.")
            mutations_dict[position] = (ref_base, mut_base)
            mutation_string = mutation_string[match.end():]  # Move past the matched part
        if mutation_string:
            raise ValueError(f"mutation string not parseable starting here: {mutation_string}")
        mutations = [(pos, ref, mut) for pos, (ref, mut) in sorted(mutations_dict.items())]
        self.mutations = mutations
    
    def _apply_mutations(self) -> None:
        mutated_seq = list(self.reference_sequence)
        for pos, ref_base, mut_base in self.mutations:
            if mutated_seq[pos] != ref_base:
                raise ValueError(f"Reference base at position {pos} does not match expected base {ref_base}.")
            mutated_seq[pos] = mut_base
        mutated_seq =  ''.join(mutated_seq)
        self.mutated_sequence = mutated_seq
    
    def get_mutation_number(self) -> int:
        return len(self.mutations)
    
    @classmethod
    def from_string(cls, reference_sequence: str, mutation_string: str) -> 'MutatedSequence':
        mutated_obj = cls.__new__(cls)
        mutated_obj.reference_sequence = reference_sequence
        mutated_obj._parse_string(mutation_string)
        mutated_obj._apply_mutations()
        return mutated_obj
    
    def __str__(self) -> str:
        mutation_str = ", ".join(f"{pos}:{ref}->{mut}" for pos, ref, mut in self.mutations) + f", fitness: {self.fitness}"
        return f"MutatedSequence({mutation_str})"
    
    def __repr__(self) -> str:
        mutation_str = "".join(f"{pos}{ref}{mut}" for pos, ref, mut in self.mutations) + f"|{self.fitness}"
        return mutation_str
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MutatedSequence):
            return False
        return (self.reference_sequence == other.reference_sequence and
                self.mutations == other.mutations and
                self.mutated_sequence == other.mutated_sequence and
                self.fitness == other.fitness)


#class that contains a dictionary with the generations of the evolutionary algorithm as key, and a list of MutatedSequence objects as value
class MutationsGene:
    reference_sequence: str  # the reference sequence used for mutation
    generation_dict: Dict[int, List['MutatedSequence']]  # key: generation, value: list of MutatedSequence objects

    def __init__(self, gene_folder_path: str, final_generation: int = 1999, generation: Optional[int] = None) -> None:
        ref_seq = Fasta(os.path.join(gene_folder_path, "reference_sequence.fa"))
        ref_seq = ref_seq[1][:].seq # type:ignore
        populations = [os.path.join(gene_folder_path, "saved_populations", file) for file in os.listdir(os.path.join(gene_folder_path, "saved_populations")) if file.startswith("pareto_front_gen")]
        results = {}
        # if generation is final generation, we can directly take the final pareto front without searching
        if generation is None or generation < final_generation:
            for population in populations:
                gen_num = int(population.split('_')[-1].split('.')[0])
                if generation is not None and gen_num != generation:
                    continue
                
                with open(population, 'r') as f:
                    pareto_front = json.load(f)
                
                for sequence, fitness, mutations in pareto_front:
                    mutated_seq = MutatedSequence(reference_sequence=ref_seq, mutated_sequence=sequence, fitness=fitness)
                    results.setdefault(gen_num, []).append(mutated_seq)
        
        # add data from the final pareto front
        if generation is None or generation == final_generation:
            with open(os.path.join(gene_folder_path, "saved_populations", "pareto_front.json"), 'r') as f:
                pareto_front = json.load(f)
            for sequence, fitness, mutations in pareto_front:
                mutated_seq = MutatedSequence(reference_sequence=ref_seq, mutated_sequence=sequence, fitness=fitness)
                results.setdefault(final_generation, []).append(mutated_seq)
        
        if generation is not None and generation not in results:
            raise ValueError(f"Generation {generation} not found in data. Available generations: {list(results.keys())}")

        self.generation_dict = results
        self.reference_sequence = ref_seq
    
    # method to convert this into a serializeable dictionary
    def to_dict(self) -> Dict:
        results = {}
        results['reference_sequence'] = self.reference_sequence
        for generation, sequences in self.generation_dict.items():
            results[generation] = [repr(seq) for seq in sequences]
        return results

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MutationsGene):
            return False
        if self.reference_sequence != other.reference_sequence:
            return False
        if len(self.generation_dict) != len(other.generation_dict):
            return False
        for gen, sequences in self.generation_dict.items():
            if gen not in other.generation_dict:
                return False
            if len(sequences) != len(other.generation_dict[gen]):
                return False
            for seq in sequences:
                if seq not in other.generation_dict[gen]:
                    return False
            for seq in other.generation_dict[gen]:
                if seq not in sequences:
                    return False
        return True
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MutationsGene':
        gene_obj = cls.__new__(cls)
        gene_obj.generation_dict = {}
        reference_sequence = data.pop('reference_sequence')
        new_data = {int(k): v for k, v in data.items()}  # Convert generation keys to int
        gene_obj.reference_sequence = reference_sequence
        for gen, sequences in new_data.items():
            gene_obj.generation_dict[gen] = [MutatedSequence.from_string(reference_sequence, seq) for seq in sequences]
        return gene_obj
    
    def get_init_and_optimal_fitness_generation(self, generation: int) -> Tuple[float, float]:
        """Get the initial and optimal fitness values for a specific generation.

        Args:
            generation (int): The generation to get the fitness values for.

        Returns:
            Tuple[float, float]: A tuple containing the initial and optimal fitness values.
        """
        if generation not in self.generation_dict:
            raise ValueError(f"Generation {generation} not found in data.")
        sorted_sequences = sorted(self.generation_dict[generation], key=lambda seq: len(seq.mutations))
        return sorted_sequences[0].fitness, sorted_sequences[-1].fitness

    def get_equal_or_next_closest_fitness(self, generation: int, fitness: float) -> MutatedSequence:
        sequences = self.generation_dict.get(generation, [])
        if not sequences:
            raise ValueError(f"No sequences found for generation {generation}.")
        # create sorted version of sequences by mutations
        # binary search for the fitness
        # if searched fitness is between two fitness values, return the one with more mutations
        sequences = sorted(sequences, key=lambda seq: len(seq.mutations))
        fitness_increasing = sequences[0].fitness <= sequences[-1].fitness
        left, right = 0, len(sequences) - 1
        while left <= right:
            mid = (left + right) // 2
            if sequences[mid].fitness < fitness:
                if fitness_increasing:
                    left = mid + 1
                else:
                    right = mid - 1
            elif sequences[mid].fitness > fitness:
                if fitness_increasing:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                return sequences[mid]
        # if we reach here, the fitness was not found, return the next larger one
        try:
            return sequences[left]
        except IndexError:
            raise ValueError(f"No sequence found with appropriate fitness relative to {fitness} in generation {generation}. Optimal fitness in generation {generation} is {sequences[-1].fitness}.")

    def search_mutation_count(self, mutation_count: int, generation: Optional[int] = None) -> Dict[str, List['MutatedSequence']]:
        """Search for mutated sequences with a specific number of mutations.

        Args:
            mutation_count (int): The number of mutations to search for.
            generation (Optional[int]): The generation to filter by. If None, search all generations.

        Returns:
            List[MutatedSequence]: A list of MutatedSequence objects with the specified mutation count.
        """
        results = {}
        if generation is not None:
            sequences = self.generation_dict.get(generation, [])
            results[generation] = [seq for seq in sequences if seq.get_mutation_number() == mutation_count]
        else:
            for gen, sequences in self.generation_dict.items():
                results[gen] = [seq for seq in sequences if seq.get_mutation_number() == mutation_count]
        return results

    def search_mutation_fitness(self, fitness: float, generation: Optional[int] = None) -> Dict[int, List['MutatedSequence']]:
        """Search for mutated sequences with a specific fitness.

        Args:
            fitness (float): The fitness value to search for.
            generation (Optional[int]): The generation to filter by. If None, search all generations.

        Returns:
            List[MutatedSequence]: A list of MutatedSequence objects with the specified fitness.
        """
        results = {}
        if generation is not None:
            sequences = self.generation_dict.get(generation, [])
            results[generation] = [seq for seq in sequences if seq.fitness == fitness]
        else:
            for gen, sequences in self.generation_dict.items():
                results[gen] = [seq for seq in sequences if seq.fitness == fitness]
        return results
    

    def get_all_mutations_generation(self, generation: int) -> List[Tuple[int, str, str]]:
        mutations = self.generation_dict[generation]
        total_mutations = set([mut for seq in mutations for mut in seq.mutations])
        return list(total_mutations)


def load_mutations_from_json(json_file: str) -> Dict[str, MutationsGene]:
    """Load mutations from a JSON file with multiple sequences.
    
    Args:
        json_file: Path to JSON file containing multiple sequence entries
        
    Returns:
        Dictionary mapping sequence names to MutationsGene objects
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = {}
    for seq_name, seq_data in data.items():
        results[seq_name] = MutationsGene.from_dict(seq_data)
    
    return results


def summarize_mutations_all_folders(base_folder_path: str, name: str, final_generation: int, generation: Optional[int] = None, output_folder: str = ".") -> Dict[str, MutationsGene]:
    output_name = f"all_mutated_sequences_{name}"
    if generation is not None:
        output_name += f"_gen{generation}"
    output_name += ".json"
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, output_name)
    if os.path.exists(save_path):
        raise FileExistsError(f"Output file {save_path} already exists. Please choose a different name or delete the existing file.")

    gene_folders = [os.path.join(base_folder_path, folder) for folder in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, folder))]
    all_results = {}
    
    for gene_folder in tqdm(gene_folders, desc="Processing genes"):
        try:
            gene_name = os.path.basename(gene_folder)
            gene_info = MutationsGene(gene_folder, final_generation=final_generation, generation=generation)
            all_results[gene_name] = gene_info
        except Exception as e:
            print(f"Error processing {gene_name}: {e}")
    with open(save_path, "w") as f:
        json.dump({gene: gene_info.to_dict() for gene, gene_info in all_results.items()}, f, indent=2)
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize mutations from all gene folders.")
    parser.add_argument("--results_folder", "-r", type=str, help="Path to the folder containing gene folders.")
    parser.add_argument("--name", "-n", type=str, default="summary", help="Name for the output summary file.")
    parser.add_argument("--output_folder", "-o", type=str, default=".", help="Folder for the output summary file.")
    parser.add_argument("--final_generation", "-f", type=int, default=1999, help="Final generation to consider for the summary.")
    parser.add_argument("--generation", "-g", type=int, default=None, help="Specific generation to filter results by. If not provided, all generations will be considered.")
    args = parser.parse_args()
    if not os.path.isdir(args.results_folder):
        raise ValueError(f"The provided results folder '{args.results_folder}' does not exist or is not a directory.")
    if not args.name:
        raise ValueError("Please provide a name for the output summary file using the --name argument.")
    return args


def main():
    args = parse_args()
    summarize_mutations_all_folders(args.results_folder, args.name, args.final_generation, args.generation, output_folder=args.output_folder)


if __name__ == "__main__":
    main()
    # test_obj = MutationsGene("/home/gernot/Code/PhD_Code/Evolution/results/diverse_genes_arabidopsis_250611_105634_010904/1_AT1G09440_gene:3048006-3045258_250611_121447_109863")
    # with open("test_mutations.json", "w") as f:
    #     json.dump(test_obj.to_dict(), f, indent=2)
    # loaded = MutationsGene.from_dict(json.load(open("test_mutations.json", "r")))
    # print(loaded.data)
    # print(loaded.reference_sequence)
    # print(loaded.data.keys())