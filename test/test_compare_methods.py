import os
import unittest
import tempfile
import json
import shutil
from analysis.simple_result_stats import expand_pareto_front
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
    compare_methods_final
)


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
        expected_abs = {"method1": 0.0, "method2": 0.15}
        expected_summed = {"method1": 0.0, "method2": 0.05}
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
            "method2": [[(0.6, 11), (0.4, 21)], [(0.5, 13), (0.3, 23)]]
        }
        
        plot_normalized_fronts(normalized_fronts, self.temp_dir)
        
        # Check that matplotlib functions were called
        self.assertEqual(mock_errorbar.call_count, 2)  # One call per method
        mock_legend.assert_called_once()
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
        
        plot_interesting_pareto_fronts_values(fronts, gene_name, tag, self.temp_dir)
        
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
            "method1": ([0.0, 0.0, 0.0], [10, 20, 30]),
            "method2": ([0.1, 0.1, 0.1], [10, 20, 30])
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
        
        plot_differences_between_fronts(fronts, gene_name, tag, self.temp_dir)
        
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
        
        plot_interesting_pareto_fronts(fronts, gene_name, tag, self.temp_dir)
        
        # Check that both plotting functions were called
        mock_plot_values.assert_called_once_with(fronts, gene_name, tag, self.temp_dir)
        mock_plot_differences.assert_called_once_with(fronts, gene_name, tag, self.temp_dir)
    
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
        
        compare_methods_final(results_paths, self.temp_dir, max_mutations=90)
        
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
            compare_methods_final(results_paths, self.temp_dir)


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
        compare_methods_final(results_paths, self.temp_dir, max_mutations=100)
        
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
                "nsga2": 0.7,
                "spea2": 0.0,
                "moead": 0.6
            },
            "summed_diff": {
                "nsga2": 0.6,
                "spea2": 0.0,
                "moead": 0.5
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
    # testObj = TestCompareMethodsIntegration()
    # testObj.setUp()
    # testObj.test_integration_differences_calculation()
    # testObj.tearDown()
    # print("Success!")
