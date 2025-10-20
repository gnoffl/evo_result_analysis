import os
import unittest
import tempfile
import json
import shutil
import random
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
from unittest.mock import patch, MagicMock, mock_open
from typing import List
from analysis.summarize_mutations import MutationsGene
from analysis.analyze_mutations import (
    count_mutations_single_gene, count_mutations_all_genes, print_mutation_stats, calculate_mutation_stats,
    create_ideal_distribution, create_worst_case_distribution, calculate_conservation_statistic,
    calculate_conservation_statistics, calc_conservation_stat_stats, plot_dict_as_stacked_bars, make_line_plot_rolling_window,
    plot_hist_half_max_mutations_stacked, plot_mutations_location, plot_hist_mutation_conservation,
    calculate_mutation_distances_single_gene, calculate_mutation_distances, plot_mutation_distances,
    load_mutation_data, analyze_mutation_distances, get_random_mutation_distributions, analyze_range_single_gene,
    plot_dist_hist,
)
from analysis.summarize_mutations import MutatedSequence
import numpy as np


class TestAnalyzeMutations(unittest.TestCase):
    """Test cases for analyze_mutations module functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        mutation_data = {
            "gene1": {
                "reference_sequence": "AAAAAAAAAA",
                "1999": ["0AT1AG5AC|0.8", "0AT1AG|0.6", "|0.5"],
                "100": ["0AT1AG|0.6", "|0.5"]
            },
            "gene2": {
                "reference_sequence": "TTTTTTTTTT", 
                "1999": ["0TC1TG|0.7", "|0.4"],
                "100": ["0TG1TG|0.6", "|0.4"]
            }
        }
        
        self.mutation_file = os.path.join(self.temp_dir, "mutations.json")
        with open(self.mutation_file, 'w') as f:
            json.dump(mutation_data, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_count_mutations_single_gene(self):
        """Test counting mutations for a single gene."""
        with open(self.mutation_file, 'r') as f:
            mutation_data = json.load(f)
        counts, individual_counts = count_mutations_single_gene(mutation_data, "gene1", generation=1999)

        # Check that we get correct number of mutation types and individual counts
        # gene1 has: "0AT1AG2AC|0.8" (3 mutations), "0AT1AG|0.6" (2 mutations), "|0.5" (0 mutations)
        # Mutations: position 0 A->T, position 1 A->G, position 2 A->C
        # Expected individual_counts: [3, 2, 0] (sorted descending)
        # Expected unique mutation keys: "0T", "1G", "2C" with counts [2, 2, 1]
        self.assertEqual(individual_counts, [3, 2, 0])
        self.assertEqual(len(counts), 3)  # 3 unique mutations
        self.assertEqual(sorted(counts), [1, 2, 2])  # Counts should be [2, 2, 1] in some order
    
    def test_count_mutations_single_gene_not_found(self):
        """Test error handling when gene is not found."""
        with open(self.mutation_file, 'r') as f:
            mutation_data = json.load(f)
        with self.assertRaises(ValueError) as context:
            count_mutations_single_gene(mutation_data, "nonexistent_gene", generation=1999)
        self.assertIn("Gene nonexistent_gene not found", str(context.exception))
    
    def test_count_mutations_single_gene_generation_not_found(self):
        """Test error handling when generation is not found."""
        with open(self.mutation_file, 'r') as f:
            mutation_data = json.load(f)
        with self.assertRaises(ValueError) as context:
            count_mutations_single_gene(mutation_data, "gene1", generation=9999)
        self.assertIn("Generation 9999 not found", str(context.exception))
    
    def test_count_mutations_all_genes(self):
        """Test counting mutations for all genes."""
        all_counts = count_mutations_all_genes(self.mutation_file)
        
        self.assertIn("gene1", all_counts)
        self.assertIn("gene2", all_counts)
        self.assertEqual(len(all_counts), 2)
        
        # Check structure of returned data
        counts, individual_counts = all_counts["gene1"]
        self.assertEqual(sorted(counts), [1, 2, 2])
        self.assertEqual(individual_counts, [3, 2, 0])
    
    def test_calculate_mutation_stats(self):
        """Test calculating mutation statistics."""
        sorted_by_count, average_max_count, std_max_count, average_n_individuals, std_n_individuals, ratios = calculate_mutation_stats(
            self.mutation_file, generation=1999
        )
        
        # Check structure of returned data
        self.assertIsInstance(sorted_by_count, list)
        self.assertIsInstance(average_max_count, float)
        self.assertIsInstance(std_max_count, float)
        self.assertIsInstance(average_n_individuals, float)
        self.assertIsInstance(std_n_individuals, float)
        self.assertIsInstance(ratios, np.ndarray)
        
        # Check that we have data for both genes
        self.assertEqual(len(sorted_by_count), 2)
        
        # Check structure of sorted_by_count entries (gene_name, max_count, n_individuals)
        for entry in sorted_by_count:
            self.assertEqual(len(entry), 3)
            self.assertIsInstance(entry[0], str)  # gene name
            self.assertIsInstance(entry[1], int)  # max count
            self.assertIsInstance(entry[2], int)  # number of individuals
        
        # Check that statistics are reasonable
        self.assertGreater(average_max_count, 0)
        self.assertGreaterEqual(std_max_count, 0)
        self.assertGreater(average_n_individuals, 0)
        self.assertGreaterEqual(std_n_individuals, 0)
        self.assertEqual(len(ratios), 2)  # One ratio per gene

        self.assertEqual(sorted_by_count[0][0], "gene2")
        self.assertEqual(sorted_by_count[0][1], 1)
        self.assertEqual(sorted_by_count[0][2], 2)

    def test_create_ideal_distribution(self):
        """Test creating ideal mutation distribution."""
        individual_counts = [4, 2, 1]
        ideal_dist = create_ideal_distribution(individual_counts, 5)
        
        expected = np.array([3, 2, 1, 1, 0])
        np.testing.assert_array_equal(ideal_dist, expected)
    
    def test_create_ideal_distribution_edge_cases(self):
        """Test ideal distribution with edge cases."""
        # Empty input
        ideal_dist = create_ideal_distribution([], 3)
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(ideal_dist, expected)
        
        # Single individual
        ideal_dist = create_ideal_distribution([2], 3)
        expected = np.array([1, 1, 0])
        np.testing.assert_array_equal(ideal_dist, expected)
    
    def test_create_worst_case_distribution(self):
        """Test creating worst case mutation distribution."""
        mutation_counts = [2, 2, 1, 1, 1]  # Total = 7
        worst_dist_1 = create_worst_case_distribution(mutation_counts, 8)
        worst_dist_2 = create_worst_case_distribution(mutation_counts, 1)

        expected_1 = np.array([1, 1, 1, 1, 1, 1, 1])  # Evenly distributed
        expected_2 = np.array([3, 2, 2])  # Evenly distributed
        np.testing.assert_array_equal(worst_dist_1, expected_1)
        np.testing.assert_array_equal(worst_dist_2, expected_2)
    
    def test_create_worst_case_distribution_edge_cases(self):
        """Test worst case distribution with edge cases."""
        # Zero mutations
        worst_dist = create_worst_case_distribution([], 5)
        expected = np.array([])
        np.testing.assert_array_equal(worst_dist, expected)
        
        # Single mutation
        worst_dist = create_worst_case_distribution([1], 10)
        expected = np.array([1])
        np.testing.assert_array_equal(worst_dist, expected)

    def test_calculate_conservation_statistic(self):
        """Test conservation statistic calculation."""
        mutation_counts = [3, 2, 1]
        individual_counts = [3, 2, 1]
        stat = calculate_conservation_statistic(mutation_counts, individual_counts, 3)
        
        self.assertEqual(stat, 1)

        mutation_counts = [2, 2, 1, 1, 1]
        individual_counts = [4, 2, 1]
        stat = calculate_conservation_statistic(mutation_counts, individual_counts, 8)
        self.assertAlmostEqual(stat, 0.666666666666, places=4)
    
    def test_calculate_conservation_statistics(self):
        """Test calculating conservation statistics for all genes."""
        with patch('builtins.print'):  # Suppress print output
            stats = calculate_conservation_statistics(
                self.mutation_file, "test", generation=1999, 
                mutable_positions=10, output_folder=self.temp_dir
            )
        
        self.assertIsInstance(stats, dict)
        self.assertIn("gene1", stats)
        self.assertIn("gene2", stats)
        
        # Check that all values are between 0 and 1
        for gene, stat in stats.items():
            self.assertGreaterEqual(stat, 0)
            self.assertLessEqual(stat, 1)
        
        # Check that output file was created
        output_file = os.path.join(self.temp_dir, "conservation_statistics_test_gen_1999.json")
        self.assertTrue(os.path.exists(output_file))
    
    def test_calc_conservation_stat_stats(self):
        """Test calculating statistics on conservation statistics."""
        test_stats = {
            "gene1": 0.8,
            "gene2": 0.6,
            "gene3": 0.9,
            "gene4": 0.2,
            "gene5": 0.7
        }
        
        min_stat, max_stat, min_gene, max_gene, avg_stat, std_stat = calc_conservation_stat_stats(test_stats)
        
        # Check return types
        self.assertIsInstance(min_stat, float)
        self.assertIsInstance(max_stat, float)
        self.assertIsInstance(min_gene, str)
        self.assertIsInstance(max_gene, str)
        self.assertIsInstance(avg_stat, float)
        self.assertIsInstance(std_stat, float)
        
        # Check min/max values and corresponding genes
        self.assertEqual(min_stat, 0.2)
        self.assertEqual(max_stat, 0.9)
        self.assertEqual(min_gene, "gene4")
        self.assertEqual(max_gene, "gene3")
        
        # Check statistical calculations
        expected_avg = np.mean(list(test_stats.values()))
        expected_std = np.std(list(test_stats.values()))
        self.assertAlmostEqual(avg_stat, expected_avg, places=5)
        self.assertAlmostEqual(std_stat, expected_std, places=5)
        
        # Test edge case with single gene
        single_stats = {"gene1": 0.5}
        min_stat, max_stat, min_gene, max_gene, avg_stat, std_stat = calc_conservation_stat_stats(single_stats)
        self.assertEqual(min_stat, 0.5)
        self.assertEqual(max_stat, 0.5)
        self.assertEqual(min_gene, "gene1")
        self.assertEqual(max_gene, "gene1")
        self.assertEqual(avg_stat, 0.5)
        self.assertEqual(std_stat, 0.0)
    
    def test_plot_dict_as_stacked_bars(self):
        """Test plotting dictionary as stacked bars."""
        test_data = {
            1: {"A": 5, "C": 3, "G": 2, "T": 1},
            2: {"A": 2, "C": 4, "G": 3, "T": 2},
            3: {"A": 1, "C": 2, "G": 4, "T": 3}
        }
        
        output_file = os.path.join(self.temp_dir, "test_plot.pdf")
        plot_dict_as_stacked_bars(
            test_data, "Test Title", "X Label", "Y Label", output_file
        )
        
        # Check that the output file was created
        self.assertTrue(os.path.exists(output_file))
    
    def test_make_line_plot_rolling_window(self):
        """Test making rolling window line plot."""
        # Create test data with correct structure
        test_data = {}
        for i in range(100):
            test_data[i] = {"A": np.random.randint(0, 5), "C": np.random.randint(0, 5), 
                           "G": np.random.randint(0, 5), "T": np.random.randint(0, 5)}
        
        make_line_plot_rolling_window(
            test_data, "test", window_size=11, output_folder=self.temp_dir
        )
        
        # Check that the output file was created
        expected_file = os.path.join(self.temp_dir, "rolling_mean_mutations_test_11.pdf")
        self.assertTrue(os.path.exists(expected_file))
    
    @patch('analysis.analyze_mutations.MutationsGene')
    @patch('tqdm.tqdm')
    def test_plot_hist_half_max_mutations_stacked(self, mock_tqdm, mock_mutations_gene):
        """Test plotting histogram of half max mutations."""
        # Mock tqdm to return the input as-is
        mock_tqdm.side_effect = lambda x, desc=None: x
        
        # Mock MutationsGene behavior
        mock_gene_instance = MagicMock()
        mock_gene_instance.generation_dict = {1999: []}
        mock_gene_instance.get_init_and_optimal_fitness_generation.return_value = (0.5, 0.9)
        mock_seq = MagicMock()
        mock_seq.get_mutation_number.return_value = 2
        mock_seq.mutations = [(0, 'A', 'T'), (1, 'G', 'C')]
        mock_gene_instance.get_equal_or_next_closest_fitness.return_value = mock_seq
        mock_mutations_gene.from_dict.return_value = mock_gene_instance
        
        plot_hist_half_max_mutations_stacked(
            self.mutation_file, "test", output_folder=self.temp_dir
        )
        
        # Check that the output files were created
        expected_from_file = os.path.join(self.temp_dir, "hist_half_max_mutations_stacked_test_from.pdf")
        expected_to_file = os.path.join(self.temp_dir, "hist_half_max_mutations_stacked_test_to.pdf")
        self.assertTrue(os.path.exists(expected_from_file))
        self.assertTrue(os.path.exists(expected_to_file))
    
    @patch('analysis.analyze_mutations.MutationsGene')
    @patch('tqdm.tqdm')
    def test_plot_mutations_location(self, mock_tqdm, mock_mutations_gene):
        """Test plotting mutation locations."""
        mock_tqdm.side_effect = lambda x, desc=None: x
        
        # Mock MutationsGene behavior
        mock_gene_instance = MagicMock()
        mock_gene_instance.generation_dict = {1999: []}
        mock_gene_instance.get_init_and_optimal_fitness_generation.return_value = (0.5, 0.9)
        mock_seq = MagicMock()
        mock_seq.mutations = [(0, 'A', 'T'), (1, 'G', 'C')]
        mock_gene_instance.get_equal_or_next_closest_fitness.return_value = mock_seq
        mock_mutations_gene.from_dict.return_value = mock_gene_instance
        
        # Test with both plots enabled
        plot_mutations_location(
            self.mutation_file, "test", window_size=11, 
            plot_stacked=True, plot_rolling=True, output_folder=self.temp_dir
        )
        
        # Check that the stacked plot files were created (3 files: from, to, diff)
        expected_stacked_files = [
            os.path.join(self.temp_dir, "hist_mutations_location_stacked_test_from.pdf"),
            os.path.join(self.temp_dir, "hist_mutations_location_stacked_test_to.pdf"),
            os.path.join(self.temp_dir, "hist_mutations_location_stacked_test_diff.pdf")
        ]
        for file_path in expected_stacked_files:
            self.assertTrue(os.path.exists(file_path))
        
        # Check that the rolling window plot files were created (3 files: from, to, diff)
        expected_rolling_files = [
            os.path.join(self.temp_dir, "rolling_mean_mutations_test_from_11.pdf"),
            os.path.join(self.temp_dir, "rolling_mean_mutations_test_to_11.pdf"),
            os.path.join(self.temp_dir, "rolling_mean_mutations_test_diff_11.pdf")
        ]
        for file_path in expected_rolling_files:
            self.assertTrue(os.path.exists(file_path))
    
    def test_plot_mutations_location_error(self):
        """Test error handling when no plot type is selected."""
        with self.assertRaises(ValueError) as context:
            plot_mutations_location(
                self.mutation_file, "test", plot_stacked=False, plot_rolling=False
            )
        self.assertIn("At least one of plot_stacked or plot_rolling must be True", str(context.exception))
    
    @patch('analysis.analyze_mutations.calculate_conservation_statistics')
    def test_plot_hist_mutation_conservation_new_calculation(self, mock_calc_stats):
        """Test plotting mutation conservation histogram with new calculation."""
        mock_stats = {"gene1": 0.8, "gene2": 0.6, "gene3": 0.9}
        mock_calc_stats.return_value = mock_stats
        
        plot_hist_mutation_conservation(
            self.mutation_file, "test", generation=1999, 
            mutable_positions=3000, output_folder=self.temp_dir
        )
        
        mock_calc_stats.assert_called_once()
        
        # Check that the output file was created
        expected_file = os.path.join(self.temp_dir, "hist_mutation_conservation_test_gen_1999.pdf")
        self.assertTrue(os.path.exists(expected_file))
    
    def test_plot_hist_mutation_conservation_existing_file(self):
        """Test plotting mutation conservation histogram with existing stats file."""
        # Create existing stats file
        stats_file = os.path.join(self.temp_dir, "conservation_statistics_test_gen_1999.json")
        test_stats = {"gene1": 0.8, "gene2": 0.6, "gene3": 0.9}
        with open(stats_file, 'w') as f:
            json.dump(test_stats, f)
        
        plot_hist_mutation_conservation(
            self.mutation_file, "test", generation=1999, 
            mutable_positions=3000, output_folder=self.temp_dir
        )
        
        # Check that the output file was created
        expected_file = os.path.join(self.temp_dir, "hist_mutation_conservation_test_gen_1999.pdf")
        self.assertTrue(os.path.exists(expected_file))
    
    def test_calculate_mutation_distances_single_gene(self):
        """Test calculating mutation distances for a single gene."""
        # Create a mock MutatedSequence with specific mutations
        distances = calculate_mutation_distances_single_gene([1, 3, 7, 10, 5])
        
        # Expected distances: 3-1=2, 5-3=2, 7-5=2, 10-7=3
        expected_distances = Counter([2, 2, 2, 3])
        
        self.assertEqual(distances, expected_distances)
        self.assertIsInstance(distances, Counter)
    
    def test_calculate_mutation_distances_single_gene_edge_cases(self):
        """Test edge cases for mutation distance calculation."""
        # Test with no mutations
        distances_empty = calculate_mutation_distances_single_gene([])
        self.assertEqual(distances_empty, Counter())
        
        # Test with single mutation
        distances_single = calculate_mutation_distances_single_gene([5])
        self.assertEqual(distances_single, Counter())
        
        # Test with two adjacent mutations
        distances_adjacent = calculate_mutation_distances_single_gene([10, 11])
        expected_adjacent = Counter([1])
        self.assertEqual(distances_adjacent, expected_adjacent)
    
    def test_calculate_mutation_distances(self):
        """Test calculating mutation distances for all genes."""
        
        # Mock MutationsGene behavior for multiple genes
        mock_gene_instance = MagicMock()
        mock_gene_instance.generation_dict = {1999: []}
        mock_gene_instance.get_init_and_optimal_fitness_generation.return_value = (0.5, 0.9)
        
        # Create different mock sequences for different genes
        mock_seq1 = MagicMock()
        mock_seq1.mutations = [(0, 'A', 'T'), (2, 'G', 'C'), (5, 'T', 'A')]  # distances: [2, 3]
        
        mock_seq2 = MagicMock() 
        mock_seq2.mutations = [(1, 'C', 'G'), (3, 'A', 'T'), (6, 'G', 'A')]  # distances: [2, 3]

        distances = calculate_mutation_distances(load_mutation_data(self.mutation_file))

        # Expected combined distances: [2, 3] + [2, 3] = Counter({2: 2, 3: 2})
        expected_distances = Counter({1: 2, 4: 1})
        
        self.assertEqual(distances, expected_distances)
        self.assertIsInstance(distances, Counter)
        
        # Verify that MutationsGene.from_dict was called for each gene in test data
    
    @patch('analysis.analyze_mutations.calculate_mutation_distances')
    def test_plot_mutation_distances(self, mock_calc_distances):
        """Test plotting mutation distances."""
        mock_distances = Counter({1: 5, 2: 8, 3: 3, 5: 2, 10: 1})
        mock_calc_distances.return_value = mock_distances
        
        # Import the function for testing
        
        plot_mutation_distances(self.mutation_file, "test", self.temp_dir)
        
        # Verify that calculate_mutation_distances was called
        mock_calc_distances.assert_called_once()
        
        # Check that the output file was created
        expected_file = os.path.join(self.temp_dir, "mutation_distances_test.pdf")
        expected_file_2 = os.path.join(self.temp_dir, "mutation_distances_test_smaller_distances.pdf")
        self.assertTrue(os.path.exists(expected_file))
        self.assertTrue(os.path.exists(expected_file_2))

    def test_load_mutation_data(self):
        """Test loading mutation data from file."""
        mutation_data = load_mutation_data(self.mutation_file)
        
        # Check that we get MutationsGene instances
        self.assertIn("gene1", mutation_data)
        self.assertIn("gene2", mutation_data)
        self.assertEqual(len(mutation_data), 2)
        
        # Check that each value is a MutationsGene instance
        for gene_name, mutations_gene in mutation_data.items():
            self.assertIsInstance(mutations_gene, MutationsGene)
            self.assertIn(1999, mutations_gene.generation_dict)

    @patch('analysis.analyze_mutations.load_mutation_data')
    @patch('analysis.analyze_mutations.analyze_range_single_gene')
    def test_analyze_mutation_distances(self, mock_analyze_range, mock_load_data):
        """Test analyzing mutation distances."""
        # Mock load_mutation_data
        mock_gene_instance = MagicMock()
        mock_gene_instance.generation_dict = {1999: []}
        mock_gene_instance.get_init_and_optimal_fitness_generation.return_value = (0.5, 0.9)
        mock_seq = MagicMock()
        mock_seq.mutations = [(0, 'A', 'T'), (2, 'G', 'C'), (5, 'T', 'A')]
        mock_gene_instance.get_equal_or_next_closest_fitness.return_value = mock_seq
        mock_load_data.return_value = {"gene1": mock_gene_instance, "gene2": mock_gene_instance}
        
        # Mock analyze_range_single_gene
        mock_analyze_range.return_value = (100, 200, 120, 180, 140, 160)  # first, last, start_90, end_90, start_50, end_50
        
        analyze_mutation_distances(mock_load_data.return_value, self.temp_dir, "test")
        
        # Check that output file was created
        output_file = os.path.join(self.temp_dir, "mutation_ranges_test.txt")
        self.assertTrue(os.path.exists(output_file))
        
        # Check content of output file
        with open(output_file, 'r') as f:
            content = f.read()
            self.assertIn("90% quantile start", content)
            self.assertIn("90% quantile end", content)
            self.assertIn("50% quantile start", content)
            self.assertIn("50% quantile end", content)

    def test_get_random_mutation_distributions(self):
        """Test generating random mutation distributions."""
        # Use smaller numbers for testing
        random_dist = get_random_mutation_distributions(mutable_positions=100, num_samples=10)
        
        self.assertIsInstance(random_dist, Counter)
        # Should have some distances
        self.assertGreater(len(random_dist), 0)
        # All distances should be positive
        for distance in random_dist.keys():
            self.assertGreater(distance, 0)

    def test_analyze_range_single_gene(self):
        """Test analyzing range for a single gene."""
        # Test with a simple case
        mutations = [1, 2, 3, 4, 50, 60, 70, 80, 90, 100]  # 10 mutations
        first, last, start_90, end_90, start_50, end_50 = analyze_range_single_gene(mutations)
        
        # Check that we get valid ranges
        self.assertIsInstance(start_90, int)
        self.assertIsInstance(end_90, int)
        self.assertIsInstance(start_50, int)
        self.assertIsInstance(end_50, int)
        
        # Check that ranges make sense
        self.assertLessEqual(start_90, end_90)
        self.assertLessEqual(start_50, end_50)
        self.assertLessEqual(end_50 - start_50, end_90 - start_90)  # 50% range should be smaller
        self.assertEqual(start_90, 1)
        self.assertEqual(end_90, 90)
        self.assertEqual(start_50, 1)
        self.assertEqual(end_50, 50)
        self.assertEqual(first, 1)
        self.assertEqual(last, 100)

    def test_analyze_range_single_gene_edge_cases(self):
        """Test edge cases for analyze_range_single_gene."""
        # Test with minimum viable mutations (need at least 2 for 50% quantile)
        mutations = [10]
        first, last, start_90, end_90, start_50, end_50 = analyze_range_single_gene(mutations)
        
        # With only 2 mutations, both quantiles should be the same
        self.assertEqual(first, 10)
        self.assertEqual(last, 10)
        self.assertEqual(start_90, 10)
        self.assertEqual(end_90, 10)
        self.assertEqual(start_50, 10)
        self.assertEqual(end_50, 10)
        
        # Test with empty list should raise error
        with self.assertRaises((ValueError, IndexError)):
            analyze_range_single_gene([])

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.figure')
    def test_plot_dist_hist(self, mock_figure, mock_clf, mock_savefig):
        """Test plotting distribution histogram."""
        distances = [1, 2, 3, 4, 5]
        counts = [10, 8, 6, 4, 2]
        
        plot_dist_hist("test", self.temp_dir, distances, counts)
        
        # Check that matplotlib functions were called
        mock_figure.assert_called_once()
        mock_clf.assert_called_once()
        mock_savefig.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.figure')
    def test_plot_dist_hist_with_random(self, mock_figure, mock_clf, mock_savefig):
        """Test plotting distribution histogram with random distribution."""
        distances = [1, 2, 3, 4, 5]
        counts = [10, 8, 6, 4, 2]
        random_dist = Counter({1: 5, 2: 4, 3: 3, 4: 2, 5: 1})
        
        plot_dist_hist("test", self.temp_dir, distances, counts, random_dist)
        
        # Check that matplotlib functions were called
        mock_figure.assert_called_once()
        mock_clf.assert_called_once()
        mock_savefig.assert_called_once()


    def test_calculate_mutation_distances_single_gene_with_positions(self):
        """Test calculating mutation distances with position list."""
        positions = [1, 3, 7, 10]
        distances = calculate_mutation_distances_single_gene(positions)
        
        # Expected distances: 3-1=2, 7-3=4, 10-7=3
        expected_distances = Counter([2, 4, 3])
        
        self.assertEqual(distances, expected_distances)
        self.assertIsInstance(distances, Counter)

class TestAnalyzeMutationsIntegration(unittest.TestCase):
    """Integration tests using more complex test data."""
    
    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @staticmethod
    def get_mutated_sequence_random(n_mutations: int, fitness: float) -> MutatedSequence:
        """Generate a random mutated sequence with a specified number of mutations."""
        base_sequence = "A" * 3000
        mutation_locations = random.sample(range(len(base_sequence)), n_mutations)
        mutated_sequence = list(base_sequence)
        for loc in mutation_locations:
            mutated_sequence[loc] = random.choice(["C", "G", "T"])
        return MutatedSequence(base_sequence, "".join(mutated_sequence), fitness=fitness)

    @staticmethod
    def get_mutated_sequence_ordered(n_mutations: int, fitness: float) -> MutatedSequence:
        """Generate a mutated sequence with mutations at the first n_mutations positions."""
        base_sequence = "A" * 3000
        mutated_sequence = list(base_sequence)
        for i in range(n_mutations):
            mutated_sequence[i] = "T"
        return MutatedSequence(base_sequence, "".join(mutated_sequence), fitness=fitness)

    def create_pareto_distribution(self, random_mutations: bool = True, minimization: bool = False) -> List[MutatedSequence]:
        """Create a pareto distribution for testing, supporting both maximization and minimization."""
        number_of_elements = np.random.randint(40, 90)
        mutation_counts = [i for i in range(10)]
        mutation_counts += random.sample(range(10, 90), number_of_elements - 10)
        mutation_counts.sort()  # Sort ascending for proper pareto front
        max_mutation_count = max(mutation_counts)
        mutated_sequences = []
        
        for i, mutation_count in enumerate(mutation_counts):
            if minimization:
                # For minimization: fewer mutations = worse fitness (higher values)
                # More mutations = better fitness (lower values)  
                fitness = 1.0 - (mutation_count / max_mutation_count)
            else:
                # For maximization: fewer mutations = worse fitness (lower values)
                # More mutations = better fitness (higher values)
                fitness = mutation_count / max_mutation_count
            
            mutated_sequence = self.get_mutated_sequence_random(mutation_count, fitness) if random_mutations else self.get_mutated_sequence_ordered(mutation_count, fitness)
            mutated_sequences.append(mutated_sequence)
        
        # Sort by mutations ascending (pareto front should go from few to many mutations)
        mutated_sequences.sort(key=lambda x: x.get_mutation_number())
        return mutated_sequences

    def generate_test_mutation_data(self, mutation_type: str, minimization: bool = False, num_genes: int = 5):
        """Generate test data for mutation conservation with support for minimization."""
        mutation_stats = {}
        for i in range(num_genes):
            random_mutations = True if mutation_type == "random" else False
            mutated_genes = self.create_pareto_distribution(random_mutations=random_mutations, minimization=minimization)
            if mutation_type == "one-off":
                random_sequence = random.choice(mutated_genes)
                if random_sequence.mutations:  # Only modify if there are mutations
                    random_mutation_index = random.randint(0, len(random_sequence.mutations) - 1)
                    old_mutation = random_sequence.mutations[random_mutation_index]
                    random_sequence.mutations[random_mutation_index] = (old_mutation[0], old_mutation[1], "G")
            
            mutation_stats[f"gene_{i}"] = {
                "reference_sequence": "A" * 3000,
                "1999": [repr(seq) for seq in mutated_genes]
            }
        
        return mutation_stats

    def test_mutation_conservation_integration_random(self):
        """Integration test for mutation conservation with random mutations."""
        # Generate test data with random mutations
        mutation_data = self.generate_test_mutation_data("random", minimization=False, num_genes=3)
        
        # Save to file
        test_file = os.path.join(self.temp_dir, "test_random.json")
        with open(test_file, 'w') as f:
            json.dump(mutation_data, f)
        
        # Test conservation statistics calculation
        with patch('builtins.print'):  # Suppress print output
            stats = calculate_conservation_statistics(
                test_file, "random_test", generation=1999, 
                mutable_positions=3000, output_folder=self.temp_dir
            )
        
        # Verify results
        self.assertEqual(len(stats), 3)  # Should have 3 genes
        for gene, stat in stats.items():
            self.assertGreaterEqual(stat, 0)
            self.assertLessEqual(stat, 1)
        
        # Check that output file was created
        output_file = os.path.join(self.temp_dir, "conservation_statistics_random_test_gen_1999.json")
        self.assertTrue(os.path.exists(output_file))

    def test_mutation_conservation_integration_ordered(self):
        """Integration test for mutation conservation with ordered mutations."""
        # Generate test data with ordered mutations (should have higher conservation)
        mutation_data = self.generate_test_mutation_data("ordered", minimization=False, num_genes=3)
        
        # Save to file
        test_file = os.path.join(self.temp_dir, "test_ordered.json")
        with open(test_file, 'w') as f:
            json.dump(mutation_data, f)
        
        # Test conservation statistics calculation
        with patch('builtins.print'):  # Suppress print output
            stats = calculate_conservation_statistics(
                test_file, "ordered_test", generation=1999, 
                mutable_positions=3000, output_folder=self.temp_dir
            )
        
        # Verify results - ordered mutations should generally have higher conservation
        self.assertEqual(len(stats), 3)
        for gene, stat in stats.items():
            self.assertGreaterEqual(stat, 0)
            self.assertLessEqual(stat, 1)
            # Ordered mutations should typically have higher conservation than random
            # (though this is probabilistic, so we just check it's reasonable)
            self.assertGreater(stat, 0.3)  # Should be reasonably high for ordered

    def test_conservation_histogram_integration(self):
        """Integration test for conservation histogram plotting."""
        # Generate test data
        mutation_data = self.generate_test_mutation_data("random", minimization=False, num_genes=5)
        
        # Save to file
        test_file = os.path.join(self.temp_dir, "test_histogram.json")
        with open(test_file, 'w') as f:
            json.dump(mutation_data, f)
        
        # Test histogram plotting
        plot_hist_mutation_conservation(
            test_file, "histogram_test", generation=1999, 
            mutable_positions=3000, output_folder=self.temp_dir
        )
        
        # Check that both files were created
        stats_file = os.path.join(self.temp_dir, "conservation_statistics_histogram_test_gen_1999.json")
        plot_file = os.path.join(self.temp_dir, "hist_mutation_conservation_histogram_test_gen_1999.pdf")
        
        self.assertTrue(os.path.exists(stats_file))
        self.assertTrue(os.path.exists(plot_file))
        
        # Verify the stats file content
        with open(stats_file, 'r') as f:
            saved_stats = json.load(f)
        self.assertEqual(len(saved_stats), 5)  # Should have 5 genes
        for gene, stat in saved_stats.items():
            self.assertIsInstance(stat, (int, float))
            self.assertGreaterEqual(stat, 0)
            self.assertLessEqual(stat, 1)

    def test_complex_mutation_counting_integration(self):
        """Integration test for mutation counting with complex data."""
        # Generate test data with varying mutation patterns
        mutation_data = self.generate_test_mutation_data("random", minimization=False, num_genes=4)
        
        # Save to file
        test_file = os.path.join(self.temp_dir, "test_counting.json")
        with open(test_file, 'w') as f:
            json.dump(mutation_data, f)
        
        # Test all genes counting
        all_counts = count_mutations_all_genes(test_file)
        
        # Verify structure and content
        self.assertEqual(len(all_counts), 4)  # Should have 4 genes
        
        for gene_name, (counts, individual_counts) in all_counts.items():
            self.assertIsInstance(counts, list)
            self.assertIsInstance(individual_counts, list)
            self.assertGreater(len(individual_counts), 0)  # Should have some individuals
            
            # Individual counts should be sorted in descending order
            self.assertEqual(individual_counts, sorted(individual_counts, reverse=True))
            
            # All mutation counts should be positive
            for count in counts:
                self.assertGreater(count, 0)
        
        # Test mutation statistics calculation
        sorted_by_count, avg_max, std_max, avg_individuals, std_individuals, ratios = calculate_mutation_stats(test_file)
        
        # Verify statistics make sense
        self.assertEqual(len(sorted_by_count), 4)
        self.assertIsInstance(avg_max, float)
        self.assertIsInstance(std_max, float)
        self.assertGreater(avg_max, 0)
        self.assertGreaterEqual(std_max, 0)
        self.assertEqual(len(ratios), 4)

if __name__ == '__main__':
    unittest.main()
    # testObj = TestAnalyzeMutations()
    # testObj.setUp()
    # testObj.test_analyze_range_single_gene_edge_cases()
    # testObj.tearDown()
    # print("Tests ran successfully.")