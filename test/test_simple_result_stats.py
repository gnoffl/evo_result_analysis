import unittest
import tempfile
import os
import json
import shutil
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import numpy as np
from unittest.mock import patch, MagicMock
from analysis.simple_result_stats import (
    get_pareto_front, add_basic_stats, calculate_half_max_mutations,
    get_stats_per_gene, summary_stat_calculation, split_stats, summarize_stats, 
    visualize_start_vs_max_fitness, visualize_start_vs_max_fitness_by_mutations,
    draw_visualize_start_vs_max_fitness_by_mutations, plot_pareto_front, show_random_fronts,
    show_average_pareto_front, expand_pareto_front, normalize_front, calculate_loss_pareto_front,
    calculate_loss_over_generations_single_gene, calculate_loss_over_generations,
    plot_loss_over_generations, plot_half_max_mutations_vs_initial_fitness, hist_half_max_mutations,
    deduplicate_pareto_front
)


class TestSimpleResultStats(unittest.TestCase):
    """Test cases for simple_result_stats module functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_folder = os.path.join(self.temp_dir, "results")
        self.results_loss_vis = os.path.join(self.temp_dir, "results_loss_vis")
        os.makedirs(self.results_folder)
        os.makedirs(self.results_loss_vis)
        
        # Create test gene folders with pareto fronts
        # Test both maximization and minimization scenarios
        test_cases = [
            ("1_gene1", "maximization"),  # Chromosome 1, maximization
            ("2_gene2", "minimization"),  # Chromosome 2, minimization  
            ("C_gene3", "minimization")   # Chloroplast, minimization
        ]
        
        for gene_name, optimization_type in test_cases:
            gene_folder = os.path.join(self.results_folder, gene_name)
            os.makedirs(os.path.join(gene_folder, "saved_populations"))
            
            if optimization_type == "maximization":
                # Higher fitness is better, fewer mutations typically give lower initial fitness
                pareto_data = [
                    ["TTTTTA", 0.9, 5],  # Best fitness, most mutations
                    ["TTTAAA", 0.7, 3],   # Medium fitness, some mutations
                    ["TTAAAA", 0.5, 2],   # Lower fitness, fewer mutations
                    ["TAAAAA", 0.1, 1],    # Worst fitness, least mutations
                    ["AAAAAA", 0., 0]    # Worst fitness, least mutations
                ]
            else:  # minimization
                # Lower fitness is better, fewer mutations typically give higher initial fitness
                pareto_data = [
                    ["TTTTTA", 0.1, 5],  # Best fitness, most mutations
                    ["TTTAAA", 0.3, 3],   # Medium fitness, some mutations
                    ["TTAAAA", 0.5, 2],   # Higher fitness, fewer mutations
                    ["TAAAAA", 0.9, 1],    # Worst fitness, least mutations
                    ["AAAAAA", 1., 0]    # Worst fitness, least mutations
                ]

            
            with open(os.path.join(gene_folder, "saved_populations", "pareto_front.json"), 'w') as f:
                json.dump(pareto_data, f)
            
            # Create generation-specific fronts for loss calculation
            with open(os.path.join(gene_folder, "saved_populations", "pareto_front_gen_100.json"), 'w') as f:
                json.dump(pareto_data[2:], f)
        
        vis_gene_folder = os.path.join(self.results_loss_vis, "1_gene1")
        os.makedirs(os.path.join(vis_gene_folder, "saved_populations"))
        with open(os.path.join(vis_gene_folder, "saved_populations", "pareto_front.json"), 'w') as f:
            json.dump([
                ["TTTTA", 1, 4],
                ["TTAAA", 0.9, 2],
                ["TAAAA", 0.8, 1],
                ["AAAAA", 0, 0],
            ], f)
        with open(os.path.join(vis_gene_folder, "saved_populations", "pareto_front_gen_1000.json"), 'w') as f:
            json.dump([
                ["CCCCC", 1, 5],
                ["CCCAA", 0.9, 3],
                ["CCAAA", 0.8, 2],
                ["CAAAA", 0.7, 1],
                ["AAAAA", 0, 0],
            ], f)
        with open(os.path.join(vis_gene_folder, "saved_populations", "pareto_front_gen_100.json"), 'w') as f:
            json.dump([
                ["GGGGG", 0.9, 5],
                ["GGGGA", 0.8, 4],
                ["GGGAA", 0.7, 3],
                ["GGAAA", 0.6, 2],
                ["GAAAA", 0.5, 1],
                ["AAAAA", 0, 0],
            ], f)
        with open(os.path.join(vis_gene_folder, "saved_populations", "pareto_front_gen_10.json"), 'w') as f:
            json.dump([
                ["TCGGT", 0.6, 5],
                ["TGCGA", 0.5, 4],
                ["ACTGA", 0.4, 3],
                ["AGACA", 0.3, 2],
                ["AGAAA", 0.2, 1],
                ["AAAAA", 0, 0],
            ], f)

        vis_gene_folder = os.path.join(self.results_loss_vis, "2_gene2")
        os.makedirs(os.path.join(vis_gene_folder, "saved_populations"))
        with open(os.path.join(vis_gene_folder, "saved_populations", "pareto_front.json"), 'w') as f:
            json.dump([
                ["TTTTT", 1, 5],
                ["TTTAA", 0.9, 4],
                ["TTAAA", 0.8, 2],
                ["TAAAA", 0.6, 1],
                ["AAAAA", 0, 0],
            ], f)
        with open(os.path.join(vis_gene_folder, "saved_populations", "pareto_front_gen_1000.json"), 'w') as f:
            json.dump([
                ["CCCCC", 0.9, 5],
                ["CCCAA", 0.8, 3],
                ["CCAAA", 0.7, 2],
                ["CAAAA", 0.4, 1],
                ["AAAAA", 0, 0],
            ], f)
        with open(os.path.join(vis_gene_folder, "saved_populations", "pareto_front_gen_100.json"), 'w') as f:
            json.dump([
                ["GGGGG", 0.7, 5],
                ["GGGGA", 0.6, 4],
                ["GGGAA", 0.5, 3],
                ["GGAAA", 0.4, 2],
                ["GAAAA", 0.3, 1],
                ["AAAAA", 0, 0],
            ], f)
        with open(os.path.join(vis_gene_folder, "saved_populations", "pareto_front_gen_10.json"), 'w') as f:
            json.dump([
                ["TCGGT", 0.5, 5],
                ["TGCGA", 0.4, 4],
                ["ACTGA", 0.3, 3],
                ["AGACA", 0.2, 2],
                ["AGAAA", 0.1, 1],
                ["AAAAA", 0, 0],
            ], f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_get_stats_per_gene(self):
        """Test getting statistics per gene."""
        stats = get_stats_per_gene(self.results_folder, "test", self.temp_dir)
        
        self.assertIn("1_gene1", stats)
        self.assertIn("2_gene2", stats)
        self.assertIn("C_gene3", stats)
        
        # Check maximization gene (1_gene1)
        gene1_stats = stats["1_gene1"]
        self.assertEqual(gene1_stats["final_fitness"], 0.9)  # First item (best fitness)
        self.assertEqual(gene1_stats["start_fitness"], 0.0)  # Last item (worst fitness)
        self.assertEqual(gene1_stats["max_mutations"], 5)
        self.assertEqual(gene1_stats["origin_chromosome"], "1")
        self.assertEqual(gene1_stats["num_mutations_half_max_effect"], 2)
        
        # Check minimization gene (2_gene2)
        gene2_stats = stats["2_gene2"]
        self.assertEqual(gene2_stats["final_fitness"], 0.1)  # First item (best fitness)
        self.assertEqual(gene2_stats["start_fitness"], 1.)  # Last item (worst fitness)
        self.assertEqual(gene2_stats["max_mutations"], 5)
        self.assertEqual(gene2_stats["origin_chromosome"], "2")
        self.assertEqual(gene2_stats["num_mutations_half_max_effect"], 2)
    
    def test_normalize_front_maximization(self):
        """Test normalizing front for maximization problem."""
        # Maximization: higher fitness is better
        pareto_front = [("seq1", 0.9, 3), ("seq2", 0.5, 2), ("seq3", 0.1, 1)]
        normalized = normalize_front(pareto_front)
        
        fitnesses = [item[1] for item in normalized]
        self.assertAlmostEqual(max(fitnesses), 1.0)
        self.assertAlmostEqual(min(fitnesses), 0.0)
        # Check that order is preserved and correct
        self.assertAlmostEqual(fitnesses[0], 1.0)  # 0.9 -> 1.0
        self.assertAlmostEqual(fitnesses[1], 0.5)  # 0.5 -> 0.5  
        self.assertAlmostEqual(fitnesses[2], 0.0)  # 0.1 -> 0.0
    
    def test_normalize_front_minimization(self):
        """Test normalizing front for minimization problem."""
        # Minimization: lower fitness is better (but normalization treats it the same)
        pareto_front = [("seq1", 0.1, 3), ("seq2", 0.5, 2), ("seq3", 0.9, 1)]
        normalized = normalize_front(pareto_front)
        
        fitnesses = [item[1] for item in normalized]
        self.assertAlmostEqual(max(fitnesses), 1.0)
        self.assertAlmostEqual(min(fitnesses), 0.0)
        # Check that order is preserved and correct  
        self.assertAlmostEqual(fitnesses[0], 0.0)  # 0.1 -> 0.0
        self.assertAlmostEqual(fitnesses[1], 0.5)  # 0.5 -> 0.5
        self.assertAlmostEqual(fitnesses[2], 1.0)  # 0.9 -> 1.0
    
    def test_get_pareto_front(self):
        """Test reading pareto front from file."""
        gene_path = os.path.join(self.results_folder, "1_gene1")
        pareto_front = get_pareto_front(gene_path)
        
        self.assertIsNotNone(pareto_front)
        if pareto_front is not None:  # Type guard for mypy
            self.assertEqual(len(pareto_front), 5)
            self.assertEqual(pareto_front[0], ["TTTTTA", 0.9, 5])
        
        # Test non-existent path
        non_existent_path = os.path.join(self.temp_dir, "non_existent")
        result = get_pareto_front(non_existent_path)
        self.assertIsNone(result)
    
    def test_add_basic_stats(self):
        """Test adding basic statistics."""
        stats = {}
        fitnesses = [0.9, 0.7, 0.5, 0.1, 0.0]
        num_mutations = [5, 3, 2, 1, 0]
        gene = "1_gene1_test"
        
        add_basic_stats(stats, fitnesses, num_mutations, gene)
        
        self.assertIn("1_gene1_test", stats)
        gene_stats = stats["1_gene1_test"]
        self.assertEqual(gene_stats['final_fitness'], 0.9)
        self.assertEqual(gene_stats['start_fitness'], 0.0)
        self.assertEqual(gene_stats['max_mutations'], 5)
        self.assertEqual(gene_stats['origin_chromosome'], "1")
    
    def test_calculate_half_max_mutations(self):
        """Test calculating half max mutations."""
        # Test maximization scenario
        pareto_front = [("seq1", 0.9, 5.), ("seq2", 0.7, 3.), ("seq3", 0.5, 2.), ("seq4", 0.1, 1.)]

        result = calculate_half_max_mutations(pareto_front)
        self.assertEqual(result, 2)
        
        # Test minimization scenario
        pareto_front_min = [("seq1", 0.1, 5.), ("seq2", 0.3, 3.), ("seq3", 0.5, 2.), ("seq4", 0.9, 1.)]
        
        result_min = calculate_half_max_mutations(pareto_front_min)
        self.assertEqual(result_min, 2)
    
    def test_summary_stat_calculation(self):
        """Test summary statistics calculation."""
        stats = {
            "gene1": {
                'final_fitness': 0.9,
                'start_fitness': 0.1,
                'max_mutations': 5,
                'num_mutations_half_max_effect': 3
            },
            "gene2": {
                'final_fitness': 0.8,
                'start_fitness': 0.2,
                'max_mutations': 4,
                'num_mutations_half_max_effect': 2
            }
        }
        
        summary = summary_stat_calculation(stats)
        
        self.assertAlmostEqual(summary['final_fitness_mean'], 0.85)
        self.assertAlmostEqual(summary['start_fitness_mean'], 0.15)
        self.assertAlmostEqual(summary['max_mutations_mean'], 4.5)
        self.assertAlmostEqual(summary['num_mutations_half_max_mean'], 2.5)
        
        # Check standard deviations exist
        self.assertAlmostEqual(summary['final_fitness_std'], 0.05)
        self.assertAlmostEqual(summary['start_fitness_std'], 0.05)
        self.assertAlmostEqual(summary['max_mutations_std'], 0.5)
        self.assertAlmostEqual(summary['num_mutations_half_max_std'], 0.5)

    def test_split_stats(self):
        """Test splitting statistics by chromosome origin."""
        stats = {
            "1_gene1": {'origin_chromosome': "1"},
            "2_gene2": {'origin_chromosome': "2"},
            "C_gene3": {'origin_chromosome': "C"},
            "M_gene4": {'origin_chromosome': "M"}
        }
        
        main_chrom, organelle_scaffold = split_stats(stats)
        
        self.assertIn("1_gene1", main_chrom)
        self.assertIn("2_gene2", main_chrom)
        self.assertIn("C_gene3", organelle_scaffold)
        self.assertIn("M_gene4", organelle_scaffold)
        self.assertEqual(len(main_chrom), 2)
        self.assertEqual(len(organelle_scaffold), 2)
    
    def test_summarize_stats(self):
        """Test summarizing statistics with file output."""
        stats = {
            "1_gene1": {
                'final_fitness': 0.9,
                'start_fitness': 0.1,
                'max_mutations': 5,
                'origin_chromosome': "1"
            },
            "C_gene2": {
                'final_fitness': 0.8,
                'start_fitness': 0.2,
                'max_mutations': 4,
                'origin_chromosome': "C"
            }
        }
        
        result = summarize_stats(stats, "test", self.temp_dir, group_origin=False)
        
        self.assertIsInstance(result, str)
        self.assertIn("Summary for test:", result)
        
        # Check that output file was created
        output_file = os.path.join(self.temp_dir, "summary_test.txt")
        self.assertTrue(os.path.exists(output_file))
        
        # Check file content
        with open(output_file, 'r') as f:
            content = f.read()
        self.assertIn("Summary for test:", content)
        self.assertEqual(content, result)
    
    def test_deduplicate_pareto_front(self):
        pareto_front_min = [("seq1", 0.9, 0), ("seq2", 0.7, 2), ("seq3", 0.5, 5), ("seq4", 0.7, 2)]
        deduped = deduplicate_pareto_front(pareto_front_min)
        self.assertEqual(len(deduped), 3)
        self.assertEqual(deduped[0][1], 0.9)
        self.assertEqual(deduped[1][1], 0.7)
        self.assertEqual(deduped[2][1], 0.5)
        self.assertEqual(deduped[0][2], 0)
        self.assertEqual(deduped[1][2], 2)
        self.assertEqual(deduped[2][2], 5)

        pareto_front_max = [("seq1", 0.9, 5), ("seq2", 0.7, 2), ("seq3", 0.5, 0), ("seq4", 0.7, 2)]
        deduped = deduplicate_pareto_front(pareto_front_max)
        self.assertEqual(len(deduped), 3)
        self.assertEqual(deduped[0][1], 0.5)
        self.assertEqual(deduped[1][1], 0.7)
        self.assertEqual(deduped[2][1], 0.9)
        self.assertEqual(deduped[0][2], 0)
        self.assertEqual(deduped[1][2], 2)
        self.assertEqual(deduped[2][2], 5)
        
    
    def test_expand_pareto_front(self):
        """Test expanding pareto front to include all mutation numbers."""
        # minimization case
        pareto_front_min = [("seq1", 0.9, 0), ("seq2", 0.7, 2), ("seq3", 0.5, 5), ("seq4", 0.7, 2)]
        max_mutations = 7
        
        expanded = expand_pareto_front(pareto_front_min, max_mutations)
        
        self.assertEqual(len(expanded), 8)  # 0 to 7 mutations
        
        # Check that all mutation numbers from 0 to max_mutations are present
        mutation_numbers = [item[2] for item in expanded]
        self.assertEqual(mutation_numbers, [0, 1, 2, 3, 4, 5, 6, 7])
        
        # Check that expansion worked correctly
        self.assertEqual(expanded[0][1], 0.9)  # Original fitness at 0 mutations
        self.assertEqual(expanded[1][1], 0.9)  # Original fitness at 0 mutations
        self.assertEqual(expanded[2][1], 0.7)  # Original fitness at 2 mutations
        self.assertEqual(expanded[3][1], 0.7)  # Original fitness at 2 mutations
        self.assertEqual(expanded[4][1], 0.7)  # Original fitness at 2 mutations
        self.assertEqual(expanded[5][1], 0.5)  # Original fitness at 5 mutations
        self.assertEqual(expanded[6][1], 0.5)  # Original fitness at 5 mutations
        self.assertEqual(expanded[7][1], 0.5)  # Original fitness at 5 mutations

        # maximization case
        pareto_front_max = [("seq1", 0.1, 0), ("seq2", 0.5, 2), ("seq3", 0.9, 5), ("seq4", 0.5, 2)]
        max_mutations_max = 7
        expanded_max = expand_pareto_front(pareto_front_max, max_mutations_max)
        self.assertEqual(len(expanded_max), 8)  # 0 to 7 mutations
        mutation_numbers_max = [item[2] for item in expanded_max]
        self.assertEqual(mutation_numbers_max, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(expanded_max[0][1], 0.1)  # Original fitness at 0 mutations
        self.assertEqual(expanded_max[1][1], 0.1)  # Original fitness at 0 mutations
        self.assertEqual(expanded_max[2][1], 0.5)  # Original fitness at 2 mutations
        self.assertEqual(expanded_max[3][1], 0.5)  # Original fitness at 2 mutations
        self.assertEqual(expanded_max[4][1], 0.5)  # Original fitness at 2 mutations
        self.assertEqual(expanded_max[5][1], 0.9)  # Original fitness at 5 mutations
        self.assertEqual(expanded_max[6][1], 0.9)  # Original fitness at 5 mutations
        self.assertEqual(expanded_max[7][1], 0.9)  # Original fitness at 5 mutations

    def test_normalize_front_edge_cases(self):
        """Test normalize front with edge cases."""
        # Test with zero fitness range
        pareto_front_zero = [("seq1", 0.5, 1), ("seq2", 0.5, 2)]
        with self.assertRaises(ValueError):
            normalize_front(pareto_front_zero)
        
        # Test with single point
        pareto_front_single = [("seq1", 0.5, 1)]
        with self.assertRaises(ValueError):
            normalize_front(pareto_front_single)
    
    def test_calculate_loss_pareto_front(self):
        """Test calculating loss between pareto fronts."""
        # minimization case
        current_front = [("seq1", 0.8, 0), ("seq2", 0.6, 1), ("seq3", 0.4, 2), ("seq4", 0.2, 4)]
        target_front = [("seq1", 0.8, 0), ("seq2", 0.5, 1), ("seq3", 0.3, 2), ("seq4", 0.2, 3)]
        max_mutations = 5
        
        minimization_loss = calculate_loss_pareto_front(current_front, target_front, max_mutations)
        
        # Loss should be sum of differences: (0.8-0.8) + (0.5-0.6) + (0.3-0.4) + (0.2-0.4) + (0.2-0.2) = 0.4
        self.assertAlmostEqual(minimization_loss, 0.4, places=5)

        # maximization case
        current_front_max = [("seq1", 0.2, 0), ("seq2", 0.4, 1), ("seq3", 0.6, 2), ("seq4", 0.8, 4)]
        target_front_max = [("seq1", 0.2, 0), ("seq2", 0.5, 1), ("seq3", 0.7, 2), ("seq4", 0.8, 3)]
        max_mutations = 5

        maximization_loss = calculate_loss_pareto_front(current_front_max, target_front_max, max_mutations)
        # Loss should be sum of differences: (0.2-0.2) + (0.5-0.4) + (0.7-0.6) + (0.8-0.6) + (0.8-0.8) = 0.4
        self.assertAlmostEqual(maximization_loss, 0.4, places=5)

    def test_calculate_loss_over_generations_single_gene(self):
        """Test calculating loss over generations for a single gene."""
        gene_name = "1_gene1"
        
        # The test setup creates pareto_front_gen_100.json with partial data
        loss_dict = calculate_loss_over_generations_single_gene(
            self.results_folder, gene_name, max_number_mutation=5, last_generation=1999
        )
        
        self.assertIn(100, loss_dict)
        self.assertIn(1999, loss_dict)
        self.assertAlmostEqual(loss_dict[1999], 0)  # Final generation should have 0 loss
        self.assertAlmostEqual(loss_dict[100], .8)  # Earlier generation should have some loss
    
    def test_calculate_loss_over_generations(self):
        """Test calculating loss over generations for all genes."""
        loss_data = calculate_loss_over_generations(
            self.results_folder, max_number_mutation=90, last_generation=1999
        )
        
        self.assertIn("1_gene1", loss_data)
        self.assertIn("2_gene2", loss_data)
        self.assertIn("C_gene3", loss_data)
        
        # Check that each gene has loss data
        for gene, gene_loss in loss_data.items():
            self.assertIn(1999, gene_loss)
            self.assertEqual(gene_loss[1999], 0)  # Final generation should have 0 loss
            if gene_loss.get(100) is not None:
                self.assertGreater(gene_loss[100], 0)  # Earlier generation should have some loss
    
    def test_visualize_start_vs_max_fitness(self):
        """Test start vs max fitness visualization."""
        stats = get_stats_per_gene(self.results_folder, "test", self.temp_dir)
        
        visualize_start_vs_max_fitness(stats, "test", output_folder=self.temp_dir, output_format="pdf")
        
        # Check that output file was created
        output_file = os.path.join(self.temp_dir, "start_vs_final_fitness_test.pdf")
        self.assertTrue(os.path.exists(output_file))
    
    def test_visualize_start_vs_max_fitness_by_mutations(self):
        """Test start vs max fitness by mutations visualization."""
        stats = get_stats_per_gene(self.results_folder, "test", self.temp_dir)
        
        visualize_start_vs_max_fitness_by_mutations(stats, "test", output_folder=self.temp_dir, output_format="png")
        
        # Check that both output files were created
        absolute_file = os.path.join(self.temp_dir, "start_vs_final_fitness_by_mutations_test_absolute.png")
        relative_file = os.path.join(self.temp_dir, "start_vs_final_fitness_by_mutations_test_relative.png")
        self.assertTrue(os.path.exists(absolute_file))
        self.assertTrue(os.path.exists(relative_file))
    
    def test_draw_visualize_start_vs_max_fitness_by_mutations(self):
        """Test drawing start vs max fitness by mutations."""
        stats = {
            "gene1": {
                'start_fitness': 0.1,
                'final_fitness': 0.9,
                'max_mutations': 5,
                'num_mutations_half_max_effect': 3
            },
            "gene2": {
                'start_fitness': 0.2,
                'final_fitness': 0.8,
                'max_mutations': 4,
                'num_mutations_half_max_effect': 2
            }
        }
        
        # Test absolute values
        draw_visualize_start_vs_max_fitness_by_mutations(stats, "test_abs", relative=False, output_folder=self.temp_dir, output_format="png")
        output_file_abs = os.path.join(self.temp_dir, "start_vs_final_fitness_by_mutations_test_abs.png")
        self.assertTrue(os.path.exists(output_file_abs))
        
        # Test relative values
        draw_visualize_start_vs_max_fitness_by_mutations(stats, "test_rel", relative=True, output_folder=self.temp_dir, output_format="png")
        output_file_rel = os.path.join(self.temp_dir, "start_vs_final_fitness_by_mutations_test_rel.png")
        self.assertTrue(os.path.exists(output_file_rel))
    
    def test_plot_pareto_front(self):
        """Test plotting individual pareto front."""
        pareto_path = os.path.join(self.results_folder, "1_gene1", "saved_populations", "pareto_front.json")
        output_path = os.path.join(self.temp_dir, "test_pareto.png")
        
        plot_pareto_front(pareto_path, output_path)
        
        # Check that output file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Test with non-existent file
        non_existent_path = os.path.join(self.temp_dir, "non_existent.json")
        output_path_2 = os.path.join(self.temp_dir, "test_pareto_2.png")
        plot_pareto_front(non_existent_path, output_path_2)  # Should not crash
        self.assertFalse(os.path.exists(output_path_2))
    
    def test_show_random_fronts(self):
        """Test showing random pareto fronts."""
        show_random_fronts(self.results_folder, num_samples=2, output_folder=self.temp_dir, output_format="png")
        
        # Check that output folder and files were created
        output_folder = os.path.join(self.temp_dir, f"random_pareto_fronts_{os.path.basename(self.results_folder)}")
        self.assertTrue(os.path.exists(output_folder))
        
        # Check that at least some pareto front files were created
        png_files = [f for f in os.listdir(output_folder) if f.endswith('.png')]
        self.assertEqual(len(png_files), 2)
    
    def test_show_average_pareto_front(self):
        """Test showing average pareto front."""
        show_average_pareto_front(self.results_folder, output_folder=self.temp_dir, max_number_mutation=10, output_format="pdf")
        
        # Check that output file was created
        run_name = os.path.basename(self.results_folder)
        output_file = os.path.join(self.temp_dir, f"average_pareto_front_{run_name}.pdf")
        self.assertTrue(os.path.exists(output_file))
    
    def test_plot_loss_over_generations(self):
        """Test plotting loss over generations."""
        plot_loss_over_generations(
            self.results_loss_vis, "test", max_number_mutation=5, 
            last_generation=1999, output_folder=self.temp_dir, output_format="png"
        )
        
        # Check that output file was created
        output_file = os.path.join(self.temp_dir, "loss_over_generations_test.png")
        self.assertTrue(os.path.exists(output_file))
    
    def test_plot_half_max_mutations_vs_initial_fitness(self):
        """Test plotting half max mutations vs initial fitness."""
        stats = get_stats_per_gene(self.results_folder, "test", self.temp_dir)
        
        plot_half_max_mutations_vs_initial_fitness(stats, "test", output_folder=self.temp_dir, output_format="png")
        
        # Check that output file was created
        output_file = os.path.join(self.temp_dir, "half_max_mutations_vs_initial_fitness_test.png")
        self.assertTrue(os.path.exists(output_file))
    
    def test_hist_half_max_mutations(self):
        """Test histogram of half max mutations."""
        stats = get_stats_per_gene(self.results_folder, "test", output_folder=self.temp_dir)
        
        hist_half_max_mutations(stats, "test", output_folder=self.temp_dir, output_format="png")
        
        # Check that output file was created
        output_file = os.path.join(self.temp_dir, "hist_half_max_mutations_test.png")
        self.assertTrue(os.path.exists(output_file))
    
    def test_get_stats_per_gene_file_exists_error(self):
        """Test that get_stats_per_gene raises error when output file exists."""
        # Create an existing stats file
        existing_file = os.path.join(self.temp_dir, "stats_test_exists.json")
        with open(existing_file, 'w') as f:
            json.dump({}, f)
        
        with self.assertRaises(FileExistsError):
            get_stats_per_gene(self.results_folder, "test_exists", output_folder=self.temp_dir)
    
    def test_get_stats_per_gene_skip_non_directory(self):
        """Test that get_stats_per_gene skips non-directory files."""
        # Create a file (not directory) in results folder
        non_dir_file = os.path.join(self.results_folder, "not_a_gene.txt")
        with open(non_dir_file, 'w') as f:
            f.write("not a gene folder")
        
        # Should not crash and should skip the file
        stats = get_stats_per_gene(self.results_folder, "test_skip", self.temp_dir)
        
        # Should still have the 3 gene directories
        self.assertEqual(len(stats), 3)
        self.assertNotIn("not_a_gene.txt", stats)
    
    def test_summary_stat_calculation_missing_half_max(self):
        """Test summary statistics when some genes are missing half_max_effect."""
        stats = {
            "gene1": {
                'final_fitness': 0.9,
                'start_fitness': 0.1,
                'max_mutations': 5,
                'num_mutations_half_max_effect': 3
            },
            "gene2": {
                'final_fitness': 0.8,
                'start_fitness': 0.2,
                'max_mutations': 4
                # Missing num_mutations_half_max_effect
            }
        }
        
        summary = summary_stat_calculation(stats)
        
        # Should still calculate other stats
        self.assertAlmostEqual(summary['final_fitness_mean'], 0.85)
        # Should handle missing half_max data gracefully
        self.assertAlmostEqual(summary['num_mutations_half_max_mean'], 3.0)  # Only gene1 contributes
    
    def test_summarize_stats_with_grouping(self):
        """Test summarize_stats with origin grouping."""
        stats = {
            "1_gene1": {
                'final_fitness': 0.9,
                'start_fitness': 0.1,
                'max_mutations': 5,
                'origin_chromosome': "1"
            },
            "C_gene2": {
                'final_fitness': 0.8,
                'start_fitness': 0.2,
                'max_mutations': 4,
                'origin_chromosome': "C"
            }
        }
        
        result = summarize_stats(stats, "test_group", self.temp_dir, group_origin=True)
        
        self.assertIsInstance(result, str)
        self.assertIn("Summary for test_group:", result)
        self.assertIn("Summary for test_group_main_chromosome:", result)
        self.assertIn("Summary for test_group_organelle_scaffold:", result)
        self.assertIn("final_fitness_mean: 0.9", result)
        
        # Check that multiple output files were created
        base_file = os.path.join(self.temp_dir, "summary_test_group.txt")
        main_file = os.path.join(self.temp_dir, "summary_test_group_main_chromosome.txt")
        organelle_file = os.path.join(self.temp_dir, "summary_test_group_organelle_scaffold.txt")
        
        self.assertTrue(os.path.exists(base_file))
        self.assertTrue(os.path.exists(main_file))
        self.assertTrue(os.path.exists(organelle_file))
    
    def test_expand_pareto_front_edge_cases(self):
        """Test edge cases for expand_pareto_front."""
        # Test when pareto front already has all points
        pareto_front = [("seq1", 0.9, 0), ("seq2", 0.7, 1), ("seq3", 0.5, 2)]
        max_mutations = 2
        
        expanded = expand_pareto_front(pareto_front, max_mutations)
        self.assertEqual(expanded, pareto_front)  # Should remain the same
        
        # Test with larger gaps
        pareto_front_gaps = [("seq1", 0.9, 0), ("seq2", 0.5, 5)]
        max_mutations = 5
        
        expanded_gaps = expand_pareto_front(pareto_front_gaps, max_mutations)
        self.assertEqual(len(expanded_gaps), 6)  # 0 to 5 mutations
        
        # Check that intermediate points have same fitness as previous point
        self.assertEqual(expanded_gaps[1][1], 0.9)  # Gap filling should use previous fitness
        self.assertEqual(expanded_gaps[2][1], 0.9)
        self.assertEqual(expanded_gaps[3][1], 0.9)
        self.assertEqual(expanded_gaps[4][1], 0.9)
        self.assertEqual(expanded_gaps[5][1], 0.5)  # Original point
    
    def test_calculate_loss_pareto_front_edge_cases(self):
        """Test edge cases for calculate_loss_pareto_front."""
        # Test with identical fronts
        front1 = [("seq1", 0.9, 0), ("seq2", 0.7, 1)]
        front2 = [("seq1", 0.9, 0), ("seq2", 0.7, 1)]
        max_mutations = 1
        
        loss = calculate_loss_pareto_front(front1, front2, max_mutations)
        self.assertAlmostEqual(loss, 0.0)
        
        # Test with fronts that are too long
        long_front = [("seq" + str(i), 0.5, i) for i in range(100)]
        with self.assertRaises(ValueError):
            calculate_loss_pareto_front(long_front, front1, max_mutations)
    
    def test_plot_half_max_mutations_vs_initial_fitness_missing_data(self):
        """Test plotting when some genes are missing half_max data."""
        stats = {
            "gene1": {
                'start_fitness': 0.1,
                'final_fitness': 0.9,
                'num_mutations_half_max_effect': 3,
                "origin_chromosome": "1"
            },
            "gene2": {
                'start_fitness': 0.2,
                'final_fitness': 0.8,
                "origin_chromosome": "1"
                # Missing num_mutations_half_max_effect
            }
        }
        
        # Should not crash and should only plot gene1
        plot_half_max_mutations_vs_initial_fitness(stats, "test_missing", output_folder=self.temp_dir, output_format="png")
        
        # Check that output file was created
        output_file = os.path.join(self.temp_dir, "half_max_mutations_vs_initial_fitness_test_missing.png")
        self.assertTrue(os.path.exists(output_file))
    
    def test_hist_half_max_mutations_missing_data(self):
        """Test histogram when some genes are missing half_max data."""
        stats = {
            "gene1": {
                'num_mutations_half_max_effect': 3
            },
            "gene2": {
                'final_fitness': 0.8,
                "origin_chromosome": "1"
                # Missing num_mutations_half_max_effect
            }
        }
        
        # Should not crash and should only use gene1
        hist_half_max_mutations(stats, "test_missing", output_folder=self.temp_dir, output_format="png")
        
        # Check that output file was created
        output_file = os.path.join(self.temp_dir, "hist_half_max_mutations_test_missing.png")
        self.assertTrue(os.path.exists(output_file))
    
    def test_hist_half_max_mutations_empty_data(self):
        """Test histogram with no valid data."""
        stats = {
            "gene1": {
                'final_fitness': 0.8,
                "origin_chromosome": "1"
                # Missing num_mutations_half_max_effect
            }
        }
        
        # Should not crash even with no valid data
        hist_half_max_mutations(stats, "test_empty", output_folder=self.temp_dir, output_format="png")
        
        # Check that output file was created (even if plot is empty)
        output_file = os.path.join(self.temp_dir, "hist_half_max_mutations_test_empty.png")
        self.assertTrue(os.path.exists(output_file))
    
    def test_show_random_fronts_more_samples_than_genes(self):
        """Test show_random_fronts when requesting more samples than available genes."""
        # Request more samples than we have genes (3)
        show_random_fronts(self.results_folder, num_samples=10, output_folder=self.temp_dir, output_format="png")
        
        # Should still work and show all available genes
        output_folder = os.path.join(self.temp_dir, f"random_pareto_fronts_{os.path.basename(self.results_folder)}")
        self.assertTrue(os.path.exists(output_folder))
        
        # Should have files for all 3 genes (or fewer if some failed)
        png_files = [f for f in os.listdir(output_folder) if f.endswith('.png')]
        self.assertLessEqual(len(png_files), 3)  # Can't be more than total genes
        self.assertGreater(len(png_files), 0)   # Should have at least some files


if __name__ == '__main__':
    unittest.main()
    # test_obj = TestSimpleResultStats()
    # test_obj.setUp()
    # test_obj.test_show_random_fronts_more_samples_than_genes()
    # test_obj.tearDown()
