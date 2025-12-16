import os
import unittest
import tempfile
import json
import shutil
from analysis.simple_result_stats import expand_pareto_front
from analysis.summarize_mutations import MutatedSequence, MutationsGene
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
from unittest.mock import patch, MagicMock
from typing import Dict, List, Tuple

from analysis.compare_methods import (
    compare_methods_progress, check_genes_present, get_gene_paths,
    calculate_differences_between_fronts, add_normalized_fronts,
    get_plot_vals_normalized_fronts, plot_normalized_fronts,
    plot_interesting_pareto_fronts_values, get_differences_and_mutations,
    plot_differences_between_fronts, plot_interesting_pareto_fronts,
    compare_methods_final, update_ranks_and_areas,
    load_mutation_data, count_mutations_per_method, update_comparison_dict,
    rank_by_mutation_count, compare_diversity_methods, get_ranks_from_sorted,
    calculate_conservation_statistic_pareto_front, calculate_conservation_per_method,
    get_sampled_pareto_fronts, rank_by_conservation,
    get_all_mutations, get_method_vectors,
    pca_transform_single_method, plot_pca_single_method,
    pca_visualization,
)
from analysis.compare_methods import pca_transform_all_methods, plot_pca_all_methods


class TestCompareMethods(unittest.TestCase):
    """Test cases for compare_methods module functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data structure
        self.sample_fronts = {
            "method1": [("seq1", 0.8, 10), ("seq2", 0.6, 20), ("seq3", 0.6, 20), ("seq4", 0.2, 40)],
            "method2": [("seq5", 0.7, 12), ("seq6", 0.5, 22), ("seq7", 0.3, 32)]
        }
        
        # Create test gene folder structure
        self.create_test_gene_structure()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_gene_structure(self):
        """Create test gene folder structure with pareto front files."""
        methods = ["method1", "method2"]
        genes = ["gene1_variant1_asdf1234", "gene2_variant2"]
        
        for method in methods:
            method_dir = os.path.join(self.temp_dir, method)
            os.makedirs(method_dir)
            
            for gene in genes:
                gene_dir = os.path.join(method_dir, gene)
                populations_dir = os.path.join(gene_dir, "saved_populations")
                os.makedirs(populations_dir)
                
                # Create sample pareto front data
                pareto_data = [
                    {"sequence": f"ATCG{gene}{method}", "fitness": 0.8, "mutations": 10},
                    {"sequence": f"GCTA{gene}{method}", "fitness": 0.6, "mutations": 20},
                    {"sequence": f"TTAA{gene}{method}", "fitness": 0.4, "mutations": 30}
                ]
                
                pareto_file = os.path.join(populations_dir, "pareto_front.json")
                with open(pareto_file, 'w') as f:
                    json.dump(pareto_data, f)
    
    def test_check_genes_present_success(self):
        """Test check_genes_present with all genes present."""
        gene_paths = {
            "gene1": {"method1": "path1", "method2": "path2"},
            "gene2": {"method1": "path3", "method2": "path4"}
        }
        methods = ["method1", "method2"]
        
        # Should not raise any exception
        check_genes_present(gene_paths, methods)
    
    def test_check_genes_present_missing_genes(self):
        """Test check_genes_present with missing genes."""
        gene_paths = {
            "gene1": {"method1": "path1"},  # missing method2
            "gene2": {"method1": "path3", "method2": "path4"}
        }
        methods = ["method1", "method2"]
        
        with self.assertRaises(ValueError) as context:
            check_genes_present(gene_paths, methods)
        
        self.assertIn("Some genes are missing", str(context.exception))
    
    def test_get_gene_paths(self):
        """Test getting gene paths from folder structure."""
        gene_folder_paths = {
            "method1": [
                os.path.join(self.temp_dir, "method1", "gene1_variant1_asdf1234"),
                os.path.join(self.temp_dir, "method1", "gene2_variant2")
            ],
            "method2": [
                os.path.join(self.temp_dir, "method2", "gene1_variant1_asdf1234"),
                os.path.join(self.temp_dir, "method2", "gene2_variant2")
            ]
        }
        
        gene_paths = get_gene_paths(gene_folder_paths)
        
        # Check structure
        self.assertIn("gene1_variant1", gene_paths)
        self.assertIn("gene2_variant2", gene_paths)
        
        # Check that gene names are correctly extracted
        for method in ["gene1_variant1", "gene2_variant2"]:
            self.assertIn("method1", gene_paths[method])
            self.assertIn("method2", gene_paths[method])
            
            # Check that paths point to pareto_front.json files
            for gene in ["method1", "method2"]:
                path = gene_paths[method][gene]
                self.assertTrue(path.endswith("pareto_front.json"))
                self.assertTrue(os.path.exists(path))
    
    def test_get_gene_paths_missing_file(self):
        """Test error handling when pareto front file is missing."""
        # Create a gene folder without pareto front file
        method_dir = os.path.join(self.temp_dir, "method3")
        gene_dir = os.path.join(method_dir, "gene3_variant3")
        populations_dir = os.path.join(gene_dir, "saved_populations")
        os.makedirs(populations_dir)
        # Don't create pareto_front.json file
        
        gene_folder_paths = {
            "method3": [gene_dir]
        }
        
        with self.assertRaises(FileNotFoundError) as context:
            get_gene_paths(gene_folder_paths)
        
        self.assertIn("Missing pareto front file", str(context.exception))
    
    def test_calculate_differences_between_fronts(self):
        """Test calculating differences between pareto fronts."""
        fronts = {
            "method1": [("seq1", 0.8, 0), ("seq2", 0.6, 1), ("seq3", 0.4, 2)],
            "method2": [("seq1", 0.8, 0), ("seq5", 0.5, 1), ("seq6", 0.45, 2)]
        }
        
        differences = calculate_differences_between_fronts(fronts)
        
        # Check that we get expected keys
        self.assertIn("abs_diff", differences)
        self.assertIn("summed_diff", differences)
        
        # Check that differences are calculated
        self.assertIsInstance(differences["abs_diff"], (dict))
        self.assertIsInstance(differences["summed_diff"], (dict))
        
        # Differences should be non-negative
        expected_abs = {"method1": 0.15, "method2": 0.0}
        expected_summed = {"method1": 0.05, "method2": 0.00}
        for method in expected_abs:
            self.assertAlmostEqual(differences["abs_diff"][method], expected_abs[method])
            self.assertAlmostEqual(differences["summed_diff"][method], expected_summed[method])
    
    def test_add_normalized_fronts(self):
        """Test adding normalized fronts to collection."""
        fronts = {
            "method1": [("seq1", 0.8, 0), ("seq2", 0.6, 1), ("seq3", 0.4, 2)],
            "method2": [("seq1", 0.8, 0), ("seq4", 0.2, 1), ("seq5", 0.0, 2)]
        }
        normalized_fronts_all = {}
        
        expected = {
            "method1": [[(1.0, 0), (0.75, 1), (0.5, 2)]],
            "method2": [[(1.0, 0), (0.25, 1), (0.0, 2)]]
        }
        add_normalized_fronts(fronts, normalized_fronts_all)

        for method in expected:
            self.assertIn(method, normalized_fronts_all)
            for expected_point, actual_point in zip(expected[method][0], normalized_fronts_all[method][0]):
                self.assertAlmostEqual(expected_point[0], actual_point[0])
                self.assertEqual(expected_point[1], actual_point[1])
        # Check that normalized fronts were added
        self.assertIn("method1", normalized_fronts_all)
        self.assertIn("method2", normalized_fronts_all)
        
        # Check structure of normalized fronts
        for method in ["method1", "method2"]:
            self.assertEqual(len(normalized_fronts_all[method]), 1)  # One front added
            front = normalized_fronts_all[method][0]
            self.assertEqual(len(front), 3)  # Three points in front
            
            # Check that each point has fitness and mutations
            for fitness, mutations in front:
                self.assertIsInstance(fitness, float)
                self.assertIsInstance(mutations, int)
                self.assertGreaterEqual(fitness, 0)
                self.assertLessEqual(fitness, 1)  # Should be normalized
    
    def test_add_normalized_fronts_edge_case(self):
        """Test normalized fronts with equal min/max values."""
        fronts = {
            "method1": [("seq1", 0.5, 10), ("seq2", 0.5, 20)]  # Same fitness values
        }
        normalized_fronts_all = {}
        
        add_normalized_fronts(fronts, normalized_fronts_all)
        
        # Should handle division by zero case
        front = normalized_fronts_all["method1"][0]
        for fitness, mutations in front:
            self.assertEqual(fitness, 0.0)  # Should be 0 when min == max
    
    def test_get_plot_vals_normalized_fronts(self):
        """Test getting plot values from normalized fronts."""
        normalized_fronts = [
            [(0.8, 10), (0.6, 20), (0.4, 30)],
            [(0.7, 12), (0.5, 22), (0.3, 32)]
        ]

        expected_avg_mutations = np.array([11.0, 21.0, 31.0])
        expected_avg_fitness = np.array([0.75, 0.55, 0.35])
        expected_std_fitness = np.array([0.05, 0.05, 0.05])
        
        avg_mutations, avg_fitness, std_fitness = get_plot_vals_normalized_fronts(normalized_fronts)
        
        # Check return types and shapes
        self.assertIsInstance(avg_mutations, np.ndarray)
        self.assertIsInstance(avg_fitness, np.ndarray)
        self.assertIsInstance(std_fitness, np.ndarray)
        
        # Check that all arrays have same length
        self.assertEqual(len(avg_mutations), len(avg_fitness))
        self.assertEqual(len(avg_fitness), len(std_fitness))
        self.assertEqual(len(avg_mutations), 3)  # Three points per front
        
        # Check that values are reasonable
        self.assertTrue(np.all(avg_mutations > 0))
        self.assertTrue(np.all(avg_fitness >= 0))
        self.assertTrue(np.all(std_fitness >= 0))
    
        # Check that calculated values match expected values
        np.testing.assert_array_almost_equal(avg_mutations, expected_avg_mutations)
        np.testing.assert_array_almost_equal(avg_fitness, expected_avg_fitness)
        np.testing.assert_array_almost_equal(std_fitness, expected_std_fitness)
    
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.errorbar')
    def test_plot_normalized_fronts(self, mock_errorbar, mock_legend, mock_close):
        """Test plotting normalized fronts."""
        normalized_fronts = {
            "method1": [[(0.8, 10), (0.6, 20)], [(0.7, 12), (0.5, 22)]],
            "method2": [[(0.6, 10), (0.4, 20)], [(0.5, 12), (0.3, 22)]]
        }
        
        plot_normalized_fronts(normalized_fronts, self.temp_dir, output_format="png")
        
        # Check that matplotlib functions were called
        self.assertEqual(mock_errorbar.call_count, 2)  # One call per method
        self.assertEqual(mock_legend.call_count, 2)  # One call per method
        mock_close.assert_called_once()
        
        # Check that savefig was called with correct path
        expected_path = os.path.join(self.temp_dir, "normalized_pareto_fronts_comparison.png")
        self.assertTrue(os.path.exists(expected_path))
    
    @patch('matplotlib.pyplot.plot')
    def test_plot_interesting_pareto_fronts_values(self, mock_plot):
        """Test plotting interesting pareto fronts values."""
        fronts = self.sample_fronts
        gene_name = "test_gene"
        tag = "test_tag"
        
        plot_interesting_pareto_fronts_values(fronts, gene_name, tag, self.temp_dir, output_format="png")
        
        # Check that plot was called for each method
        self.assertEqual(mock_plot.call_count, 2)
        
        # Check that savefig was called with correct filename
        expected_filename = f"pareto_fronts_comparison_{gene_name}_{tag}.png"
        expected_path = os.path.join(self.temp_dir, expected_filename)
        self.assertTrue(os.path.exists(expected_path))
    
    def test_get_differences_and_mutations(self):
        """Test getting differences and mutations between fronts."""
        fronts = {
            "method1": [("seq1", 0.8, 10), ("seq2", 0.6, 20), ("seq3", 0.4, 30)],
            "method2": [("seq1", 0.7, 10), ("seq4", 0.5, 20), ("seq5", 0.3, 30)]
        }
        
        differences_dict = get_differences_and_mutations(fronts)
        
        expected = {
            "method1": ([0.1, 0.1, 0.1], [10, 20, 30]),
            "method2": ([0.0, 0.0, 0.0], [10, 20, 30])
        }

        for method in expected:
            self.assertIn(method, differences_dict)
            expected_diffs, expected_muts = expected[method]
            actual_diffs, actual_muts = differences_dict[method]
            self.assertEqual(expected_muts, actual_muts)
            for exp_diff, act_diff in zip(expected_diffs, actual_diffs):
                self.assertAlmostEqual(exp_diff, act_diff)
    
    @patch('matplotlib.pyplot.plot')
    def test_plot_differences_between_fronts(self, mock_plot):
        """Test plotting differences between fronts."""
        fronts = {
            "method1": [("seq1", 0.8, 10), ("seq2", 0.6, 20), ("seq3", 0.4, 30)],
            "method2": [("seq1", 0.7, 10), ("seq4", 0.5, 20), ("seq5", 0.3, 30)]
        }
        gene_name = "test_gene"
        tag = "test_tag"
        
        plot_differences_between_fronts(fronts, gene_name, tag, self.temp_dir, output_format="png")
        
        # Check that plot was called for each method
        self.assertEqual(mock_plot.call_count, 2)
        
        # Check that savefig was called
        expected_filename = f"pareto_fronts_differences_{gene_name}_{tag}.png"
        expected_path = os.path.join(self.temp_dir, expected_filename)
        self.assertTrue(os.path.isfile(expected_path))
    
    @patch('analysis.compare_methods.plot_differences_between_fronts')
    @patch('analysis.compare_methods.plot_interesting_pareto_fronts_values')
    def test_plot_interesting_pareto_fronts(self, mock_plot_values, mock_plot_differences):
        """Test plotting interesting pareto fronts (calls both sub-functions)."""
        fronts = self.sample_fronts
        gene_name = "test_gene"
        tag = "test_tag"
        
        plot_interesting_pareto_fronts(fronts, gene_name, tag, self.temp_dir, output_format="png")
        
        # Check that both plotting functions were called
        mock_plot_values.assert_called_once_with(fronts=fronts, gene_name=gene_name, tag=tag, output_dir=self.temp_dir, output_format="png")
        mock_plot_differences.assert_called_once_with(fronts=fronts, gene_name=gene_name, tag=tag, output_dir=self.temp_dir, output_format="png")
    
    def test_compare_methods_progress(self):
        """Test compare_methods_progress function."""
        # This function is currently a pass, so just test that it doesn't crash
        results_paths = ["path1", "path2", "path3"]
        
        try:
            compare_methods_progress(results_paths)
        except Exception as e:
            self.fail(f"compare_methods_progress raised an exception: {e}")
    
    @patch('analysis.compare_methods.plot_interesting_pareto_fronts')
    @patch('analysis.compare_methods.plot_normalized_fronts')
    @patch('analysis.compare_methods.expand_pareto_front')
    def test_compare_methods_final(self, mock_expand, mock_plot_normalized, mock_plot_interesting):
        """Test compare_methods_final function."""
        # Mock expand_pareto_front to return our sample data
        mock_expand.side_effect = lambda data, max_number_mutation: [
            ("seq1", 0.8, 10), ("seq2", 0.6, 20), ("seq3", 0.4, 30)
        ]
        
        results_paths = {
            "method1": os.path.join(self.temp_dir, "method1"),
            "method2": os.path.join(self.temp_dir, "method2")
        }
        
        compare_methods_final(results_paths, self.temp_dir, max_mutations=90, output_format="png")
        
        # Check that plotting functions were called
        mock_plot_normalized.assert_called_once()
        self.assertEqual(mock_plot_interesting.call_count, 4)  # Called for 4 different scenarios
        
        # Check that expand_pareto_front was called for each gene-method combination
        expected_calls = 4  # 2 genes × 2 methods
        self.assertEqual(mock_expand.call_count, expected_calls)
    
    def test_compare_methods_final_missing_directory(self):
        """Test compare_methods_final with missing directory."""
        results_paths = {
            "method1": "/nonexistent/path",
            "method2": "/another/nonexistent/path"
        }
        
        with self.assertRaises(FileNotFoundError):
            compare_methods_final(results_paths, self.temp_dir, output_format="png")
    
    def test_update_ranks_and_areas(self):
        """Test updating ranks and areas aggregation."""
        ranks_and_areas = {}
        differences = {"methodA": 0.2, "methodB": 0.5, "methodC": 0.0}
        
        # First update
        update_ranks_and_areas(ranks_and_areas, differences)
        
        # Expected order by difference ascending: methodC (0.1) -> rank 1, methodA (0.2) -> rank 2, methodB (0.5) -> rank 3
        expected_ranks_first = {"methodC": 1, "methodA": 2, "methodB": 3}
        expected_areas_first = {"methodA": 0.2, "methodB": 0.5, "methodC": 0.0}
        expected_best_first = {"methodC": 1, "methodA": 0, "methodB": 0}
        
        self.assertIn("ranks", ranks_and_areas)
        self.assertIn("areas", ranks_and_areas)
        self.assertEqual(ranks_and_areas["ranks"], expected_ranks_first)
        self.assertEqual(ranks_and_areas["best"], expected_best_first)
        for method in expected_areas_first:
            self.assertAlmostEqual(ranks_and_areas["areas"][method], expected_areas_first[method])
        
        # Call again to ensure aggregation (ranks sum, areas sum)
        update_ranks_and_areas(ranks_and_areas, differences)
        expected_ranks_second = {k: v * 2 for k, v in expected_ranks_first.items()}
        expected_areas_second = {k: v * 2 for k, v in expected_areas_first.items()}
        expected_best_second = {k: v * 2 for k, v in expected_best_first.items()}
        
        self.assertEqual(ranks_and_areas["ranks"], expected_ranks_second)
        self.assertEqual(ranks_and_areas["best"], expected_best_second)
        for method in expected_areas_second:
            self.assertAlmostEqual(ranks_and_areas["areas"][method], expected_areas_second[method])
    

class TestCompareDiversity(unittest.TestCase):
    """Dedicated tests for diversity comparison utilities (load/update/rank/compare)."""

    def setUp(self):
        """Create temporary folder and sample mutation JSONs used by multiple tests."""
        self.temp_dir = tempfile.mkdtemp()
        # Define two methods with identical gene lists
        self.genes = ["geneA", "geneB"]
        self.ref_seq = "AAAAAA"
        self.method1_data = {
            "geneA": {"reference_sequence": self.ref_seq, "1999": ["1AT|0.5", "|0.4"]},
            "geneB": {"reference_sequence": self.ref_seq, "1999": ["2AT3AG|0.6", "1AG|0.5", "|0.4"]}
        }
        self.method2_data = {
            "geneA": {"reference_sequence": self.ref_seq, "1999": ["|0.4"]},
            "geneB": {"reference_sequence": self.ref_seq, "1999": ["2AT|0.6", "|0.4"]}
        }
        self.m1_path = os.path.join(self.temp_dir, "method1_mut.json")
        self.m2_path = os.path.join(self.temp_dir, "method2_mut.json")
        with open(self.m1_path, 'w') as f:
            json.dump(self.method1_data, f)
        with open(self.m2_path, 'w') as f:
            json.dump(self.method2_data, f)

        # Input mapping expected by load_mutation_data: method -> (results_path, mutation_path)
        self.input_data = {
            "method1": (os.path.join(self.temp_dir, "method1_results"), self.m1_path),
            "method2": (os.path.join(self.temp_dir, "method2_results"), self.m2_path)
        }
        # create dummy results folders
        os.makedirs(self.input_data["method1"][0], exist_ok=True)
        os.makedirs(self.input_data["method2"][0], exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_mutation_data(self):
        """Test load_mutation_data returns expected structures and gene list."""
        method_dict, gene_names = load_mutation_data(self.input_data)
        # Basic checks
        self.assertIn("method1", method_dict)
        self.assertIn("method2", method_dict)
        self.assertEqual(sorted(gene_names), sorted(self.genes))
        # Each method value should be a dict mapping gene->MutationsGene
        for method in ["method1", "method2"]:
            self.assertIsInstance(method_dict[method], dict)
            for gene in self.genes:
                self.assertIn(gene, method_dict[method])
                mg = method_dict[method][gene]
                self.assertTrue(hasattr(mg, "generation_dict"))
                self.assertIn(1999, mg.generation_dict)

    def test_count_mutations_per_method(self):
        """Test count_mutations_per_method returns correct per-method counts for a gene."""
        method_dict, gene_names = load_mutation_data(self.input_data)
        counts = count_mutations_per_method(method_dict, "geneB")
        # method1 geneB has three unique mutation positions, method2 has one
        self.assertEqual(counts["method1"], 3)
        self.assertEqual(counts["method2"], 1)
        method_dict = {
            "method1": {
                "geneA": MutationsGene.from_dict({
                    "reference_sequence": "AAAAAA",
                    1999: ["1AT|0.5", "|0.4"]
                }),
            },
            "method2": {
                "geneA": MutationsGene.from_dict({
                    "reference_sequence": "AAAAAA",
                    1999: ["|0.4"]
                }),
            }
        }
        counts = count_mutations_per_method(method_dict, "geneA")
        self.assertEqual(counts["method1"], 1)
        self.assertEqual(counts["method2"], 0)
    
    def test_get_ranks_from_sorted(self):
        input_dict = {
            "A": 10.,
            "B": 2.,
            "C": 5.,
            "D": 2.,
            "E": 7.,
            "F": 3.,
            "G": 3.,
            "H": 3.,
            "I": 3.,
            "J": 2.,
        }
        calculated_ranks = get_ranks_from_sorted(input_dict)
        expected_ranks = [1, 2, 3, 5.5, 5.5, 5.5, 5.5, 9, 9, 9]
        self.assertEqual(calculated_ranks, expected_ranks)

    def test_update_comparison_dict(self):
        """Test update_comparison_dict increments mutation_count, ranks and best correctly."""
        comparison = {}
        curr = {"m1": 5, "m2": 2, "m3": 7, "m4": 7}
        update_comparison_dict(comparison, curr)
        # mutation_count aggregated
        self.assertEqual(comparison["mutation_count"]["m1"], 5)
        self.assertEqual(comparison["mutation_count"]["m2"], 2)
        self.assertEqual(comparison["mutation_count"]["m3"], 7)
        self.assertEqual(comparison["mutation_count"]["m4"], 7)
        # ranks should reflect sorted order ascending by mutation_count: m2 (rank1), m1 (rank2), m3 (rank3)
        self.assertEqual(comparison["ranks"]["m1"], 3)
        self.assertEqual(comparison["ranks"]["m2"], 4)
        self.assertEqual(comparison["ranks"]["m3"], 1.5)
        self.assertEqual(comparison["ranks"]["m4"], 1.5)
        # best count should set 1 for the best (lowest) method
        self.assertEqual(comparison["best"]["m1"], 0)
        self.assertEqual(comparison["best"]["m2"], 0.)
        self.assertEqual(comparison["best"]["m3"], 0.5)
        self.assertEqual(comparison["best"]["m4"], 0.5)
        # call again to ensure aggregation
        update_comparison_dict(comparison, {"m1": 1, "m2": 2, "m3": 3, "m4": 4})
        # mutation_count aggregated
        self.assertEqual(comparison["mutation_count"]["m1"], 6)
        self.assertEqual(comparison["mutation_count"]["m2"], 4)
        self.assertEqual(comparison["mutation_count"]["m3"], 10)
        self.assertEqual(comparison["mutation_count"]["m4"], 11)
        # ranks should reflect sorted order ascending by mutation_count: m2 (rank1), m1 (rank2), m3 (rank3)
        self.assertEqual(comparison["ranks"]["m1"], 7)
        self.assertEqual(comparison["ranks"]["m2"], 7)
        self.assertEqual(comparison["ranks"]["m3"], 3.5)
        self.assertEqual(comparison["ranks"]["m4"], 2.5)
        # best count should set 1 for the best (lowest) method
        self.assertEqual(comparison["best"]["m1"], 0)
        self.assertEqual(comparison["best"]["m2"], 0.)
        self.assertEqual(comparison["best"]["m3"], 0.5)
        self.assertEqual(comparison["best"]["m4"], 1.5)

    def test_rank_by_mutation_count(self):
        """Test rank_by_mutation_count aggregates counts, ranks and best correctly across genes."""
        method_dict, gene_names = load_mutation_data(self.input_data)
        comparison = rank_by_mutation_count(method_dict, gene_names)
        self.assertIn("mutation_count", comparison)
        self.assertIn("ranks", comparison)
        self.assertIn("best", comparison)
        # mutation counts should sum to total across genes: method1 = 1 (geneA) + 3 (geneB) = 4
        self.assertEqual(comparison["mutation_count"]["method1"], 4)
        self.assertEqual(comparison["mutation_count"]["method2"], 1)
        self.assertEqual(comparison["ranks"]["method1"], 1)
        self.assertEqual(comparison["ranks"]["method2"], 2)
        self.assertEqual(comparison["best"]["method1"], 1)
        self.assertEqual(comparison["best"]["method2"], 0.)

    def test_compare_diversity_methods_writes_file(self):
        """Test compare_diversity_methods creates diversity_comparison.json with expected keys."""
        compare_diversity_methods(self.input_data, self.temp_dir, max_bootstrap=5, mutable_positions=10)
        out_file = os.path.join(self.temp_dir, "diversity_comparison.json")
        self.assertTrue(os.path.exists(out_file))
        with open(out_file, 'r') as f:
            content = json.load(f)
        self.assertIn("mutation_count_comparison", content)
        self.assertIn("conservation_measure_comparison", content)
    

    def test_get_sampled_pareto_fronts(self):
        """get_sampled_pareto_front should return a sampled subset of the pareto front up to max_samples."""
        class DummyInd:
            def __init__(self, muts):
                self.mutations = muts
        pareto_front = [DummyInd([i]) for i in range(10)]
        # Test with max_bootstrap greater than front size
        sampled = get_sampled_pareto_fronts(pareto_front, max_bootstrap=15)
        self.assertEqual(len(sampled), 10)

        pareto_front.extend([DummyInd([1, i]) for i in range(10)])
        sampled = get_sampled_pareto_fronts(pareto_front, max_bootstrap=1000)
        self.assertEqual(len(sampled), 100)

        pareto_front.append(DummyInd([1, 2, 3]))
        pareto_front.append(DummyInd([1, 2, 3, 4]))
        pareto_front.append(DummyInd([1, 2, 3, 4, 5]))
        sampled = get_sampled_pareto_fronts(pareto_front, max_bootstrap=1000)
        self.assertEqual(len(sampled), 100)
        # Test that sampled individuals are from original front
        for front in sampled:
            for ind in front :
                self.assertIn(ind, pareto_front)
        
        sampled = get_sampled_pareto_fronts(pareto_front, max_bootstrap=5)
        self.assertEqual(len(sampled), 5)
    
    def test_calculate_conservation_statistic_pareto_front(self):
        """calculate_conservation_statistic_pareto_front should return a float conservation measure."""
        reference_sequence = "AAAAAA"
        pareto_front = [
            MutatedSequence(reference_sequence=reference_sequence, mutated_sequence="AAAAAA", fitness=0),
            MutatedSequence(reference_sequence=reference_sequence, mutated_sequence="GAAAAA", fitness=0),
            MutatedSequence(reference_sequence=reference_sequence, mutated_sequence="GGAAAA", fitness=0),
        ]
        cons = calculate_conservation_statistic_pareto_front(
            pareto_front,
            max_bootstrap=10,
            mutable_positions=6
        )
        self.assertIsInstance(cons, float)
        self.assertAlmostEqual(cons, 1)

        pareto_front = [
            MutatedSequence(reference_sequence=reference_sequence, mutated_sequence="AAAAAA", fitness=0),
            MutatedSequence(reference_sequence=reference_sequence, mutated_sequence="GAAAAA", fitness=0),
            MutatedSequence(reference_sequence=reference_sequence, mutated_sequence="AGGAAA", fitness=0),
        ]
        cons = calculate_conservation_statistic_pareto_front(
            pareto_front,
            max_bootstrap=10,
            mutable_positions=6
        )
        self.assertIsInstance(cons, float)
        self.assertAlmostEqual(cons, 0)

        pareto_front = [
            MutatedSequence(reference_sequence=reference_sequence, mutated_sequence="AAAAAA", fitness=0),
            MutatedSequence(reference_sequence=reference_sequence, mutated_sequence="GAAAAA", fitness=0),
            MutatedSequence(reference_sequence=reference_sequence, mutated_sequence="AGAAAA", fitness=0),
            MutatedSequence(reference_sequence=reference_sequence, mutated_sequence="AGGAAA", fitness=0),
            MutatedSequence(reference_sequence=reference_sequence, mutated_sequence="GGGAAA", fitness=0),
        ]
        cons = calculate_conservation_statistic_pareto_front(
            pareto_front,
            max_bootstrap=10,
            mutable_positions=6
        )
        self.assertIsInstance(cons, float)
        self.assertGreater(cons, 0.6)
        self.assertLessEqual(cons, 1)

    @patch('analysis.compare_methods.calculate_conservation_statistic_pareto_front')
    def test_calculate_diversity_per_method(self, mock_cons_pf):
        """calculate_diversity_per_method should call conservation function for each method and return mapping."""
        # Build dummy MutationsGene-like object with generation_dict holding a pareto front
        # Patch conservation to return deterministic values per method-call
        mock_cons_pf.side_effect = [0.33, 0.66]
        inputs_dict = {}
        for method, item in self.input_data.items():
            (_, mutation_data_path) = item
            with open(mutation_data_path, 'r') as f:
                mut_data = json.load(f)
            full_method_dict = {gene: MutationsGene.from_dict(mut_data[gene]) for gene in mut_data}
            inputs_dict[method] = full_method_dict
        out = calculate_conservation_per_method(inputs_dict, gene="geneA", max_bootstrap=5, mutable_positions=10)
        self.assertIn("method1", out)
        self.assertIn("method2", out)
        self.assertAlmostEqual(out["method1"], 0.33)
        self.assertAlmostEqual(out["method2"], 0.66)
        self.assertEqual(mock_cons_pf.call_count, 2)

    @patch('analysis.compare_methods.calculate_conservation_per_method')
    def test_rank_by_conservation_aggregates_and_normalizes(self, mock_calc_div):
        """rank_by_conservation should aggregate conservation measures across genes, compute ranks and normalize them."""
        # Two genes, two methods. Prepare per-gene conservation measures to hit different ordering.
        # For gene1: m1=0.1, m2=0.2 -> m1 better (lower)
        # For gene2: m1=0.3, m2=0.05 -> m2 better
        mock_calc_div.side_effect = [
            {"m1": 0.1, "m2": 0.2},
            {"m1": 0.3, "m2": 0.05}
        ]
        comparison = rank_by_conservation(method_dict={}, gene_names=["g1", "g2"], max_bootstrap=5, mutable_positions=10)
        # Should contain aggregated conservation_measure sums, ranks and best normalized by number of genes (2)
        self.assertIn("conservation_measure", comparison)
        self.assertIn("ranks", comparison)
        self.assertIn("best", comparison)
        # Aggregated sums
        self.assertAlmostEqual(comparison["conservation_measure"]["m1"], 0.4)
        self.assertAlmostEqual(comparison["conservation_measure"]["m2"], 0.25)
        # After two genes, ranks aggregated were (m1:3, m2:3) then normalized by 2 -> 1.5
        self.assertAlmostEqual(comparison["ranks"]["m1"], 1.5)
        self.assertAlmostEqual(comparison["ranks"]["m2"], 1.5)
        # Best counts: each method was best once -> normalized 1/2 = 0.5
        self.assertAlmostEqual(comparison["best"]["m1"], 0.5)
        self.assertAlmostEqual(comparison["best"]["m2"], 0.5)

class TestMutationVectorsAndPCA(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.add_mutation_data()
        # Prepare input_data mapping for pca_visualization
        self.input_data = {
            method: (f"/dummy/path/{method}_results", f"/dummy/path/{method}_mutations.json")
            for method in self.method_dict.keys()
        }
    
    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)
    
    def add_mutation_data(self):
        """Create sample mutation data for two methods and one gene."""
        self.method_dict = {
            "m1": {
                "g1": MutationsGene.from_dict({
                    "reference_sequence": "AAAAAA",
                    2000: ["1AT4AG|0.6", "3AC|0.5", "1AT|0.4", "|0.3"],
                    1001: ["2AT|0.7"]
                })
            },
            "m2": {
                "g1": MutationsGene.from_dict({
                    "reference_sequence": "AAAAAA",
                    2000: ["1AT|0.4", "2AG|0.35", "|0.3"],
                    1001: ["1AT|0.6"]
                })
            }
        }

    def test_get_all_mutations(self):
        """Ensure all mutations union and vectors per element match membership in element.mutations."""
        gene = "g1"
        # get_all_mutations should return union across methods (expect 3 unique positions)
        all_muts = get_all_mutations(self.method_dict, gene)
        self.assertIsInstance(all_muts, set)
        self.assertEqual(len(all_muts), 4)
        for mut in [(1, "A", "T"), (4, "A", "G"), (3, "A", "C"), (2, "A", "G")]:
            self.assertIn(mut, all_muts)

    def test_get_method_vectors(self):
        gene = "g1"
        # Stable order for deterministic vectors
        all_muts = [(1, "A", "T"), (2, "A", "G"), (3, "A", "C"), (4, "A", "G")]
        vectors = get_method_vectors(self.method_dict, gene, all_muts)
        self.assertIn("m1", vectors)
        self.assertIn("m2", vectors)
        # For each method, for each individual, sum(vector) equals len(element.mutations)
        self.assertEqual(vectors["m1"], [[1, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(vectors["m2"], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])

    def test_pca_transform_single_method(self):
        """pca_transform_single_method should fit and transform and return expected shapes."""
        vectors = [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]]
        PCA_obj, transformed = pca_transform_single_method(vectors)
        self.assertEqual(transformed.shape, (4, 2))
        self.assertIsNotNone(PCA_obj)
        # first two vectors should be represented by one principal component, same for last two
        self.assertTrue((transformed[0, 0] == 0 and transformed[1, 0] == 0) or (transformed[0, 1] == 0 and transformed[1, 1] == 0))
        self.assertTrue((transformed[2, 0] == 0 and transformed[3, 0] == 0) or (transformed[2, 1] == 0 and transformed[3, 1] == 0))
    

    def test_pca_transform_all_methods(self):
        """pca_transform_all_methods should fit and transform and return expected shapes."""
        method_vectors = {
            "m1": [[1, 0, 0], [0, 1, 0], [1, 1, 0]],
            "m2": [[0, 0, 1], [0, 0, 2]],
        }
        PCA_obj, transformed = pca_transform_all_methods(method_vectors)
        self.assertIsNotNone(PCA_obj)
        self.assertIn("m1", transformed)
        self.assertIn("m2", transformed)
        self.assertEqual(transformed["m1"].shape, (3, 2))
        self.assertEqual(transformed["m2"].shape, (2, 2))

    def test_plot_pca_single_method_writes_file(self):
        """plot_pca_single_method should write a file."""
        # Use the real function with simple vectors by mocking the transform function
        with patch('analysis.compare_methods.pca_transform_single_method') as mock_tx:
            mock_pca = MagicMock()
            mock_pca.explained_variance_ratio_ = np.array([0.6, 0.4])
            transformed = np.array([[0.1, 0.2], [0.3, 0.4]])
            mock_tx.return_value = (mock_pca, transformed)
            plot_pca_single_method([[0, 1], [1, 0]], gene="g1", method="m1", output_dir=self.temp_dir, output_format="png")
        out = os.path.join(self.temp_dir, "pca_m1_g1.png")
        self.assertTrue(os.path.exists(out))

    def test_plot_pca_all_methods_writes_file(self):
        """plot_pca_all_methods should write a file."""
        with patch('analysis.compare_methods.pca_transform_all_methods') as mock_tx:
            mock_pca = MagicMock()
            mock_pca.explained_variance_ratio_ = np.array([0.55, 0.45])
            mock_tx.return_value = (mock_pca, {
                "m1": np.array([[0.1, 0.2], [0.3, 0.4]]),
                "m2": np.array([[0.5, 0.6]]),
            })
            method_vectors = {"m1": [[0, 1], [1, 1]], "m2": [[1, 0]]}
            plot_pca_all_methods(method_vectors, gene="g1", output_dir=self.temp_dir, output_format="png")
        out = os.path.join(self.temp_dir, "pca_all_methods_g1.png")
        self.assertTrue(os.path.exists(out))

    @patch('analysis.compare_methods.plot_pca_single_method')
    @patch('analysis.compare_methods.plot_pca_all_methods')
    @patch('analysis.compare_methods.load_mutation_data')
    @patch('analysis.compare_methods.random.sample')
    def test_pca_visualization_calls_plots(self, mock_sample, mock_load, mock_plot_all, mock_plot_single):
        """pca_visualization should call both plotting routines for sampled genes."""
        # Mock load_mutation_data to return our pre-built method_dict
        mock_load.return_value = (self.method_dict, ["g1"])
        # Make sure we sample both genes
        mock_sample.side_effect = lambda genes, k: list(genes)
        
        # Use dummy input_data since load_mutation_data is mocked
        dummy_input = {"m1": ("/dummy/path", "/dummy/path/m1_mutations.json")}
        pca_visualization(dummy_input, output_dir=self.temp_dir)
        
        # We have 2 genes; plot_all called once per gene
        self.assertEqual(mock_plot_all.call_count, 1)
        # plot_single called once per method per gene (2 methods * 2 genes)
        self.assertEqual(mock_plot_single.call_count, 2)


class TestCompareMethodsIntegration(unittest.TestCase):
    """Integration tests for compare_methods module."""
    
    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.create_complex_test_structure()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_complex_test_structure(self):
        """Create a more complex test structure with multiple genes and methods."""
        methods = ["nsga2", "spea2", "moead"]
        genes = {"gene1_v1": 0.95, "gene1_v2": 0.9, "gene2_v1": 0.85}
        
        for method in methods:
            method_dir = os.path.join(self.temp_dir, method)
            os.makedirs(method_dir)

            for i, (gene, base_fitness) in enumerate(genes.items()):
                gene_dir = os.path.join(method_dir, gene)
                populations_dir = os.path.join(gene_dir, "saved_populations")
                os.makedirs(populations_dir)
                
                # Create diverse pareto front data with method-specific characteristics
                fitness_decay = 0.2 if method == "nsga2" else (0.15 if method == "spea2" else 0.1)
                
                pareto_data = [( f"{method}_{gene}_seq_0", base_fitness, 0)]
                for j in range(1, 5):  # 5 points per front
                    mutations = j * 3 + i  # Slight variation per gene
                    if method == "moead":
                        fitness_decay = [0.1, 0.2, 0.3, 0.3][j-1]
                    fitness = base_fitness - j * fitness_decay
                    sequence = f"{method}_{gene}_seq_{j}"
                    pareto_data.append((sequence, max(0.01, fitness), mutations))
                
                # add duplicate
                pareto_data.insert(2, pareto_data[2])
                
                pareto_file = os.path.join(populations_dir, "pareto_front.json")
                with open(pareto_file, 'w') as f:
                    json.dump(pareto_data, f, indent=2)
    
    def test_integration_full_comparison(self):
        """Integration test for full method comparison."""
        # Mock expand_pareto_front to return realistic data
        results_paths = {
            "nsga2": os.path.join(self.temp_dir, "nsga2"),
            "spea2": os.path.join(self.temp_dir, "spea2"),
            "moead": os.path.join(self.temp_dir, "moead")
        }
        
        # This should complete without errors
        compare_methods_final(results_paths, self.temp_dir, max_mutations=15, output_format="png")
        
        # Check that output files were created
        expected_file = os.path.join(self.temp_dir, "normalized_pareto_fronts_comparison.png")
        self.assertTrue(os.path.exists(expected_file))
        
        # Check that multiple comparison plots were created
        comparison_files = [f for f in os.listdir(self.temp_dir) if f.startswith("pareto_fronts_comparison_")]
        self.assertGreater(len(comparison_files), 0)
        
        difference_files = [f for f in os.listdir(self.temp_dir) if f.startswith("pareto_fronts_differences_")]
        self.assertGreater(len(difference_files), 0)
    
    def test_integration_gene_path_extraction(self):
        """Integration test for gene path extraction from complex structure."""
        gene_folder_paths = {}
        for method in ["nsga2", "spea2", "moead"]:
            method_path = os.path.join(self.temp_dir, method)
            gene_folder_paths[method] = [ os.path.join(method_path, folder) for folder in os.listdir(method_path) ]
        
        gene_paths = get_gene_paths(gene_folder_paths)
        
        expected_genes = ["gene1_v1", "gene1_v2", "gene2_v1"]
        expected_methods = ["nsga2", "spea2", "moead"]
        # Check that all methods are present
        for gene in expected_genes:
            self.assertIn(gene, gene_paths)
        
        # Check that all genes are present for each method
        for method in expected_methods:
            for gene in expected_genes:
                self.assertIn(method, gene_paths[gene])
                self.assertTrue(os.path.exists(gene_paths[gene][method]))

        # Verify that check_genes_present doesn't raise an error
        try:
            check_genes_present(gene_paths, ["nsga2", "spea2", "moead"])
        except ValueError:
            self.fail("check_genes_present raised ValueError unexpectedly")
    
    def test_integration_differences_calculation(self):
        """Integration test for differences calculation with realistic data."""
        # Create fronts with different characteristics
        fronts = {
            "nsga2": [("seq1", 0.9, 0), ("seq2", 0.8, 3), ("seq3", 0.5, 5)],
            "spea2": [("seq1", 0.9, 0), ("seq5", 0.75, 4), ("seq6", 0.65, 6)],
            "moead": [("seq1", 0.9, 0), ("seq8", 0.7, 2), ("seq9", 0.6, 7)]
        }
        fronts = {k: expand_pareto_front(v, 7) for k, v in fronts.items()}
        
        differences = calculate_differences_between_fronts(fronts)

        expected = {
            "abs_diff": {
                "nsga2": 0.0,
                "spea2": 0.7,
                "moead": 0.9
            },
            "summed_diff": {
                "nsga2": 0.0,
                "spea2": 0.6,
                "moead": 0.1
            }
        }
        
        # Check that differences are calculated correctly
        self.assertIn("abs_diff", differences)
        self.assertIn("summed_diff", differences)
        
        # Differences should be meaningful
        for method in expected["abs_diff"]:
            self.assertAlmostEqual(differences["abs_diff"][method], expected["abs_diff"][method])
            self.assertAlmostEqual(differences["summed_diff"][method], expected["summed_diff"][method])
        
        # Test normalized fronts addition
        normalized_fronts_all = {}
        add_normalized_fronts(fronts, normalized_fronts_all)
        
        # Check that normalization worked
        for method in fronts.keys():
            self.assertIn(method, normalized_fronts_all)
            normalized_front = normalized_fronts_all[method][0]
            self.assertEqual(len(normalized_front), len(fronts[method]))
            self.assertEqual([mut for _, mut in normalized_front], [0, 1, 2, 3, 4, 5, 6, 7])
            
        expected_nsga2_normal = [(1.0, 0), (1, 1), (1, 2), (0.75, 3), (0.75, 4), (0.0, 5), (0.0, 6), (0.0, 7)]
        expected_spea2_normal = [(1.0, 0), (1, 1), (1, 2), (1, 3), (0.625, 4), (0.625, 5), (0.375, 6), (0.375, 7)]
        expected_moead_normal = [(1.0, 0), (1, 1), (0.5, 2), (0.5, 3), (0.5, 4), (0.5, 5), (0.5, 6), (0.25, 7)]
        for expected, actual in zip(expected_nsga2_normal, normalized_fronts_all["nsga2"][0]):
            self.assertAlmostEqual(expected[0], actual[0])
            self.assertEqual(expected[1], actual[1])
        for expected, actual in zip(expected_spea2_normal, normalized_fronts_all["spea2"][0]):
            self.assertAlmostEqual(expected[0], actual[0])
            self.assertEqual(expected[1], actual[1])
        for expected, actual in zip(expected_moead_normal, normalized_fronts_all["moead"][0]):
            self.assertAlmostEqual(expected[0], actual[0])
            self.assertEqual(expected[1], actual[1])


if __name__ == '__main__':
    unittest.main()
    # testObj = TestCompareDiversity()
    # testObj.setUp()
    # testObj.test_rank_by_conservation_aggregates_and_normalizes()
    # testObj.tearDown()
    # print("Success!")
