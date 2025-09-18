import json
import shutil
import unittest
import os
import tempfile
from analysis.summarize_mutations import MutatedSequence, MutationsGene, summarize_mutations_all_folders

class TestMutatedSequence(unittest.TestCase):
    """Test cases for the MutatedSequence class."""
    
    def setUp(self):
        """Set up test fixtures with common test data."""
        self.ref_seq = "AAAAAAAAAAAAAAAAAAAAAAAATTTTTTTTTTTTTTTTTTTTTTTTT"
        mut_seqs = list(self.ref_seq)
        mut_seqs[2] = 'T'  # Mutation at position 2
        mut_seqs[10] = 'G'  # Mutation at position 10
        mut_seqs[30] = 'A'  # Mutation at position 30
        mut_seqs[42] = 'C'  # Mutation at position 42
        self.mut_seq = "".join(mut_seqs)
        self.expected_mutations = [
            (2, 'A', 'T'),
            (10, 'A', 'G'),
            (30, 'T', 'A'),
            (42, 'T', 'C')
        ]
        
    def test_init_with_valid_sequences(self):
        """Test initialization with valid reference and mutated sequences."""
        mutated_seq = MutatedSequence(self.ref_seq, self.mut_seq, 0.5)
        
        self.assertEqual(mutated_seq.reference_sequence, self.ref_seq)
        self.assertEqual(mutated_seq.mutated_sequence, self.mut_seq)
        self.assertEqual(mutated_seq.mutations, self.expected_mutations)
        self.assertEqual(mutated_seq.fitness, 0.5)
    
    def test_init_with_mismatched_length(self):
        """Test initialization with sequences of different lengths."""
        short_seq = "AAAA"
        
        with self.assertRaises(ValueError) as context:
            MutatedSequence(self.ref_seq, short_seq, 0.5)
        
        self.assertIn("does not match reference sequence length", str(context.exception))
    
    def test_find_mutations_no_mutations(self):
        """Test _find_mutations with identical sequences."""
        identical_seq = self.ref_seq
        mutated_seq = MutatedSequence(self.ref_seq, identical_seq, 0.5)
        
        self.assertEqual(mutated_seq.mutations, [])
    
    def test_find_mutations_single_mutation(self):
        """Test _find_mutations with a single mutation."""
        single_mut_seq = self.ref_seq[:5] + "G" + self.ref_seq[6:]
        mutated_seq = MutatedSequence(self.ref_seq, single_mut_seq, 0.5)
        
        expected = [(5, 'A', 'G')]
        self.assertEqual(mutated_seq.mutations, expected)
    
    def test_parse_mutations_valid_string(self):
        """Test _parse_mutations with valid mutation string."""
        mutated_seq = MutatedSequence.__new__(MutatedSequence)
        mutation_string = "2AT10AG30TA42TC|0.5"
        
        mutated_seq._parse_string(mutation_string)
        
        self.assertEqual(mutated_seq.mutations, self.expected_mutations)
        self.assertEqual(mutated_seq.fitness, 0.5)
    
    def test_parse_mutations_invalid_string(self):
        """Test _parse_mutations with invalid mutation string."""
        mutated_seq = MutatedSequence.__new__(MutatedSequence)
        invalid_strings = ["2AT10AGXYZ|0.5", "2AT10AT2AG|0.5", "2AT10AT2TG|0.5", "invalid"]

        for invalid_string in invalid_strings:
            with self.assertRaises(ValueError):
                mutated_seq._parse_string(invalid_string)
    
    def test_parse_mutations_empty_string(self):
        """Test _parse_mutations with empty mutation string."""
        mutated_seq = MutatedSequence.__new__(MutatedSequence)
        
        mutated_seq._parse_string("|0.5")
        
        self.assertEqual(mutated_seq.mutations, [])
        self.assertEqual(mutated_seq.fitness, 0.5)
    
    def test_apply_mutations_valid(self):
        """Test _apply_mutations with valid mutations."""
        mutated_seq = MutatedSequence.__new__(MutatedSequence)
        mutated_seq.reference_sequence = self.ref_seq
        mutated_seq.mutations = self.expected_mutations
        
        mutated_seq._apply_mutations()
        
        self.assertEqual(mutated_seq.mutated_sequence, self.mut_seq)
    
    def test_apply_mutations_invalid_reference(self):
        """Test _apply_mutations with mismatched reference base."""
        mutated_seq = MutatedSequence.__new__(MutatedSequence)
        mutated_seq.reference_sequence = self.ref_seq
        # Create mutation with wrong reference base
        mutated_seq.mutations = [(2, 'G', 'T')]  # Position 2 should be 'A', not 'G'
        
        with self.assertRaises(ValueError) as context:
            mutated_seq._apply_mutations()
        
        self.assertIn("Reference base at position 2 does not match expected base", str(context.exception))
    
    def test_from_string_classmethod(self):
        """Test from_string class method."""
        mutation_string = "2AT10AG30TA42TC|0.5"
        mutated_seq = MutatedSequence.from_string(self.ref_seq, mutation_string)
        
        self.assertEqual(mutated_seq.reference_sequence, self.ref_seq)
        self.assertEqual(mutated_seq.mutations, self.expected_mutations)
        self.assertEqual(mutated_seq.mutated_sequence, self.mut_seq)
        self.assertEqual(mutated_seq.fitness, 0.5)
    
    def test_from_string_empty_mutations(self):
        """Test from_string with no mutations."""
        mutated_seq = MutatedSequence.from_string(self.ref_seq, "|0.5")
        
        self.assertEqual(mutated_seq.reference_sequence, self.ref_seq)
        self.assertEqual(mutated_seq.mutations, [])
        self.assertEqual(mutated_seq.mutated_sequence, self.ref_seq)
        self.assertEqual(mutated_seq.fitness, 0.5)
    
    def test_str_representation(self):
        """Test __str__ method."""
        mutated_seq = MutatedSequence(self.ref_seq, self.mut_seq, 0.5)
        str_repr = str(mutated_seq)
        
        expected = "MutatedSequence(2:A->T, 10:A->G, 30:T->A, 42:T->C, fitness: 0.5)"
        self.assertEqual(str_repr, expected)
    
    def test_str_representation_no_mutations(self):
        """Test __str__ method with no mutations."""
        mutated_seq = MutatedSequence(self.ref_seq, self.ref_seq, 0.5)
        str_repr = str(mutated_seq)
        
        expected = "MutatedSequence(, fitness: 0.5)"
        self.assertEqual(str_repr, expected)
    
    def test_repr_representation(self):
        """Test __repr__ method."""
        mutated_seq = MutatedSequence(self.ref_seq, self.mut_seq, 0.5)
        repr_str = repr(mutated_seq)
        
        expected = "2AT10AG30TA42TC|0.5"
        self.assertEqual(repr_str, expected)
    
    def test_repr_representation_no_mutations(self):
        """Test __repr__ method with no mutations."""
        mutated_seq = MutatedSequence(self.ref_seq, self.ref_seq, 0.5)
        repr_str = repr(mutated_seq)
        
        expected = "|0.5"
        self.assertEqual(repr_str, expected)
    
    def test_roundtrip_conversion(self):
        """Test that creating from sequences and then from string gives same result."""
        # Create from sequences
        mutated_seq1 = MutatedSequence(self.ref_seq, self.mut_seq, 0.5)
        
        # Get string representation
        repr_str = repr(mutated_seq1)
        
        # Create from string
        mutated_seq2 = MutatedSequence.from_string(self.ref_seq, repr_str)
        
        # Compare
        self.assertTrue(mutated_seq1 == mutated_seq2)
        self.assertEqual(str(mutated_seq1), str(mutated_seq2))
        self.assertEqual(repr(mutated_seq1), repr(mutated_seq2))
    
    def test_equality(self):
        """Test equality operator."""
        mutated_seq1 = MutatedSequence(self.ref_seq, self.mut_seq, 0.5)
        mutated_seq2 = MutatedSequence(self.ref_seq, self.mut_seq, 0.5)
        
        self.assertEqual(mutated_seq1, mutated_seq2)
        self.assertNotEqual(mutated_seq1, MutatedSequence(self.ref_seq, "A" * len(self.ref_seq), 0.5))
        self.assertNotEqual(mutated_seq1, MutatedSequence("A" * len(self.ref_seq), self.mut_seq, 0.5))
        self.assertNotEqual(mutated_seq1, MutatedSequence("A" * len(self.ref_seq), self.mut_seq, 0.4))
    
    def test_get_mutation_number(self):
        """Test get_mutation_number method."""
        mutated_seq = MutatedSequence(self.ref_seq, self.mut_seq, 0.5)
        self.assertEqual(mutated_seq.get_mutation_number(), 4)
        
        no_mut_seq = MutatedSequence(self.ref_seq, self.ref_seq, 0.5)
        self.assertEqual(no_mut_seq.get_mutation_number(), 0)



class TestMutationsGene(unittest.TestCase):
    """Test cases for the MutationsGene class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.gene_folder_min = os.path.join(self.temp_dir, "test_gene_minimization")
        self.gene_folder_max = os.path.join(self.temp_dir, "test_gene_maximization")
        os.makedirs(os.path.join(self.gene_folder_min, "saved_populations"))
        os.makedirs(os.path.join(self.gene_folder_max, "saved_populations"))
        
        # Create reference sequence file
        self.ref_seq = "AAAAAATTTTTTT"
        with open(os.path.join(self.gene_folder_min, "reference_sequence.fa"), 'w') as f:
            f.write(">test_sequence\n" + self.ref_seq + "\n")
        with open(os.path.join(self.gene_folder_max, "reference_sequence.fa"), 'w') as f:
            f.write(">test_sequence\n" + self.ref_seq + "\n")
        
        # Create test pareto front data - minimization case (lower fitness is better, fewer mutations is better)
        self.min_pareto_data = [
            ["AAGAAATTTTCTA", 0.4, 3],  
            ["AACAAATTGTTTT", 0.6, 2],  
            ["AAAAAATTTTTTT", 0.8, 0]   
        ]

        # Maximization case (higher fitness is better, fewer mutations is better)
        self.max_pareto_data = [
            ["AAGAAATTTTCTA", 0.8, 3],  
            ["AACAAATTGTTTT", 0.6, 2],  
            ["AAAAAATTTTTTT", 0.4, 0]   
        ]
        
        # Use minimization data by default
        with open(os.path.join(self.gene_folder_min, "saved_populations", "pareto_front.json"), 'w') as f:
            json.dump(self.min_pareto_data, f)
        
        # Create generation-specific pareto front
        with open(os.path.join(self.gene_folder_min, "saved_populations", "pareto_front_gen_100.json"), 'w') as f:
            json.dump(self.min_pareto_data[1:], f)
        
        with open(os.path.join(self.gene_folder_max, "saved_populations", "pareto_front.json"), 'w') as f:
            json.dump(self.max_pareto_data, f)
        
        with open(os.path.join(self.gene_folder_max, "saved_populations", "pareto_front_gen_100.json"), 'w') as f:
            json.dump(self.max_pareto_data[1:], f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_all_generations(self):
        """Test initialization with all generations."""
        gene = MutationsGene(self.gene_folder_min, final_generation=1999)
        
        self.assertIn(1999, gene.generation_dict)
        self.assertIn(100, gene.generation_dict)
        self.assertEqual(len(gene.generation_dict[1999]), 3)
        self.assertEqual(len(gene.generation_dict[100]), 2)
    
    def test_init_specific_generation(self):
        """Test initialization with specific generation."""
        gene = MutationsGene(self.gene_folder_min, final_generation=1999, generation=100)
        
        self.assertIn(100, gene.generation_dict)
        self.assertNotIn(1999, gene.generation_dict)
        self.assertEqual(len(gene.generation_dict[100]), 2)
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        gene = MutationsGene(self.gene_folder_min, final_generation=1999)
        gene_dict = gene.to_dict()
        
        reconstructed_gene = MutationsGene.from_dict(gene_dict)
        
        self.assertEqual(gene.reference_sequence, reconstructed_gene.reference_sequence)
        self.assertEqual(list(gene.generation_dict.keys()), list(reconstructed_gene.generation_dict.keys()))
        for gen, sequences in gene.generation_dict.items():
            self.assertEqual(len(sequences), len(reconstructed_gene.generation_dict[gen]))
            for seq in sequences:
                self.assertIn(seq, reconstructed_gene.generation_dict[gen])
    
    def test_get_init_and_optimal_fitness_generation_maximization(self):
        """Test fitness retrieval for maximization problem."""
        # Update to use maximization data
        gene = MutationsGene(self.gene_folder_max, final_generation=1999)
        init_fit, opt_fit = gene.get_init_and_optimal_fitness_generation(1999)
        
        # For maximization: sorted by mutation count, so lowest mutations (0) = initial fitness, highest mutations (3) = optimal
        self.assertEqual(init_fit, 0.4)  # 0 mutations
        self.assertEqual(opt_fit, 0.8)   # 3 mutations
    
    def test_get_init_and_optimal_fitness_generation_minimization(self):
        """Test fitness retrieval for minimization problem."""
        gene = MutationsGene(self.gene_folder_min, final_generation=1999)
        init_fit, opt_fit = gene.get_init_and_optimal_fitness_generation(1999)
        
        # For minimization: sorted by mutation count, so lowest mutations (0) = initial fitness, highest mutations (3) = final
        self.assertEqual(init_fit, 0.8)  # 0 mutations
        self.assertEqual(opt_fit, 0.4)   # 3 mutations
    
    def test_search_mutation_count(self):
        """Test search by mutation count."""
        gene = MutationsGene(self.gene_folder_min, final_generation=1999)
        results = gene.search_mutation_count(2)
        
        self.assertIn(1999, results)
        self.assertIn(100, results)
        for gen in [1999, 100]:
            self.assertEqual(len(results[gen]), 1)  # type: ignore
            self.assertEqual(results[gen][0].mutated_sequence, self.min_pareto_data[1][0])      # type: ignore
            self.assertEqual(results[gen][0].get_mutation_number(), 2)                          # type: ignore
            for mutation in results[gen][0].mutations:  # type: ignore  
                self.assertIn(mutation, [(2, 'A', 'C'), (8, 'T', 'G')])

        # Test with 0 mutations
        results_zero = gene.search_mutation_count(0)
        self.assertIn(1999, results_zero)
        self.assertIn(100, results_zero)
        for gen in [1999, 100]:
            self.assertEqual(len(results_zero[gen]), 1)  # type: ignore
            self.assertEqual(results_zero[gen][0].mutated_sequence, self.ref_seq)     # type: ignore
            self.assertEqual(results_zero[gen][0].get_mutation_number(), 0)         # type: ignore

    def test_search_mutation_fitness(self):
        """Test search by fitness."""
        gene = MutationsGene(self.gene_folder_min, final_generation=1999)
        results = gene.search_mutation_fitness(0.6)
        
        self.assertIn(1999, results)
        self.assertIn(100, results)
        self.assertEqual(len(results[1999]), 1)  # One sequence with fitness 0.6
        self.assertEqual(results[1999][0].mutated_sequence, self.min_pareto_data[1][0])
        self.assertEqual(results[100][0].mutated_sequence, self.min_pareto_data[1][0])
    
    def test_get_equal_or_next_closest_fitness_minimization(self):
        """Test fitness search for minimization problem."""
        gene = MutationsGene(self.gene_folder_min, final_generation=1999)
        
        # Search for exact fitness
        seq = gene.get_equal_or_next_closest_fitness(1999, 0.6)
        self.assertEqual(seq.fitness, 0.6)
        self.assertEqual(seq.get_mutation_number(), 2)
        
        # Search for fitness between values - should get next lower (better) fitness
        seq = gene.get_equal_or_next_closest_fitness(1999, 0.5)
        self.assertEqual(seq.fitness, 0.4)
        self.assertEqual(seq.get_mutation_number(), 3)
    
    def test_get_equal_or_next_closest_fitness_maximization(self):
        """Test fitness search for maximization problem."""
        # Update to use maximization data
        gene = MutationsGene(self.gene_folder_max, final_generation=1999)

        # Search for exact fitness
        seq = gene.get_equal_or_next_closest_fitness(1999, 0.6)
        self.assertEqual(seq.fitness, 0.6)
        self.assertEqual(seq.get_mutation_number(), 2)
        
        # Search for fitness between values - should get next higher (better) fitness
        seq = gene.get_equal_or_next_closest_fitness(1999, 0.7)
        self.assertEqual(seq.fitness, 0.8)
        self.assertEqual(seq.get_mutation_number(), 3)

class TestSummarizeMutations(unittest.TestCase):
    """Test cases for summarize_mutations module functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_folder = os.path.join(self.temp_dir, "results")
        os.makedirs(self.base_folder)
        
        # Create test gene folders
        for i, gene_name in enumerate(["gene1", "gene2"]):
            gene_folder = os.path.join(self.base_folder, gene_name)
            os.makedirs(os.path.join(gene_folder, "saved_populations"))
            
            # Create reference sequence
            ref_seq = "A" * 10
            with open(os.path.join(gene_folder, "reference_sequence.fa"), 'w') as f:
                f.write(f">{gene_name}\n{ref_seq}\n")
            
            # Create pareto front data
            pareto_data = [
                ["ATAAAAAAAA", 0.8 + i * 0.1, 1],
                ["ATAAACAAAA", 0.6 + i * 0.1, 2]
            ]
            
            with open(os.path.join(gene_folder, "saved_populations", "pareto_front.json"), 'w') as f:
                json.dump(pareto_data, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_summarize_mutations_all_folders(self):
        """Test summarizing mutations from all folders."""
        output_file = os.path.join(self.temp_dir, "all_mutated_sequences_test.json")
        
        data_direct = summarize_mutations_all_folders(
            self.base_folder, 
            "test", 
            final_generation=1999,
            output_folder=self.temp_dir
        )
        
        self.assertTrue(os.path.exists(output_file))
        
        with open(output_file, 'r') as f:
            data_direct = json.load(f)
        
        self.assertIn("gene1", data_direct)
        self.assertIn("gene2", data_direct)
        self.assertEqual(data_direct["gene1"], data_direct["gene1"])
        self.assertEqual(data_direct["gene2"], data_direct["gene2"])
        gene1 = MutationsGene.from_dict(data_direct["gene1"])
        gene2 = MutationsGene.from_dict(data_direct["gene2"])
        self.assertEqual(gene1.reference_sequence, "A" * 10)
        self.assertEqual(gene2.reference_sequence, "A" * 10)
        for gene in [gene1, gene2]:
            one_mut_seq = gene.search_mutation_count(1)
            self.assertEqual(len(one_mut_seq), 1)
            self.assertEqual(one_mut_seq[1999][0].mutated_sequence, "ATAAAAAAAA")       #type: ignore
            self.assertIn(one_mut_seq[1999][0].fitness, [0.8, 0.9]) # type: ignore      
            two_mut_seq = gene.search_mutation_count(2)
            self.assertEqual(len(two_mut_seq), 1)
            self.assertEqual(two_mut_seq[1999][0].mutated_sequence, "ATAAACAAAA")       # type: ignore
            self.assertIn(two_mut_seq[1999][0].fitness, [0.6, 0.7]) # type: ignore  