import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import numpy as np
from typing import List, Tuple

# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.candidate_selection import (
    get_data_at_mutation_count,
    preliminary_selection,
    load_selected_genes,
    draw_line_plot,
    get_line_plot_data,
    compare_trajectories_mutations,
    get_best_fitness_and_seq,
    find_best_mutation_deltas,
    find_best_mutation_count_per_gene,
    find_best_mutations_per_gene,
    selected_per_mutation,
    selected_per_gene,
    final_selection,
    one_off_error_correction,
    get_pareto_front_paths,
    get_gene_name_from_path,
    process_mutations,
    process_fitness_thresholds,
    process_single_gene,
    filter_and_sort_results,
    combine_selections,
    get_sorted_values
)


class TestCandidateSelection(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.sample_pareto_front_max = [
            ("AAAAAACC", 0.85, 20.),
            ("AAAAAACA", 0.75, 15),
            ("AAAAAAAT", 0.5, 10),
            ("AAAAAAAG", 0.3, 5),
            ("AAAAAAAC", 0.1, 1),
            ("AAAAAAAA", 0.05, 0.)
        ]
        self.sample_pareto_front_min = [
            ("AAAAAACC", 0.05, 20.),
            ("AAAAAACA", 0.1, 15),
            ("AAAAAAAT", 0.3, 10),
            ("AAAAAAAG", 0.5, 5),
            ("AAAAAAAC", 0.75, 1),
            ("AAAAAAAA", 0.85, 0.)
        ]

        self.results_max_folder = tempfile.mkdtemp()
        self.results_min_folder = tempfile.mkdtemp()
        data_max = {
            "1_gene1_gene:123-234_12342143_12341234_312341": [("AAAAAACC", 0.85, 20.), ("AAAAAACA", 0.75, 15), ("AAAAAAAT", 0.5, 10), ("AAAAAAAG", 0.3, 5), ("AAAAAAAC", 0.1, 1), ("AAAAAAAA", 0.05, 0.)],
            "2_gene2_gene:234-345_23452345_23452345_23452345": [("AAAAAAAG", 0.3, 15), ("AAAAAAAC", 0.1, 10), ("AAAAAAAA", 0.05, 0.)],
            "3_gene3_gene:345-456_34563456_34563456_34563456": [("AAAAAAAT", 0.75, 10), ("AAAAAAAA", 0.6, 0.)],
            "4_gene4_gene:456-567_45674567_45674567_45674567": []
        }
        data_min = {
            "1_gene1_gene:123-234_12342143_12341234_312341": [("AAAAAACC", 0.05, 20.), ("AAAAAACA", 0.1, 15), ("AAAAAAAT", 0.3, 10), ("AAAAAAAG", 0.5, 5), ("AAAAAAAC", 0.75, 1), ("AAAAAAAA", 0.85, 0.)],
            "2_gene2_gene:234-345_23452345_23452345_23452345": [("AAAAAAAG", 0.05, 15), ("AAAAAAAC", 0.1, 10), ("AAAAAAAA", 0.3, 0.)],
            "3_gene3_gene:345-456_34563456_34563456_34563456": [("AAAAAAAT", 0.3, 10), ("AAAAAAAA", 0.4, 0.)],
            "4_gene4_gene:456-567_45674567_45674567_45674567": []
        }

        for gene, pf in data_max.items():
            gene_dir = os.path.join(self.results_max_folder, gene, "saved_populations")
            os.makedirs(gene_dir, exist_ok=True)
            if pf:  # Only write if there is data
                with open(os.path.join(gene_dir, "pareto_front.json"), 'w') as f:
                    json.dump(pf, f)
        for gene, pf in data_min.items():
            gene_dir = os.path.join(self.results_min_folder, gene, "saved_populations")
            os.makedirs(gene_dir, exist_ok=True)
            if pf:  # Only write if there is data
                with open(os.path.join(gene_dir, "pareto_front.json"), 'w') as f:
                    json.dump(pf, f)
        
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.results_max_folder, ignore_errors=True)
        shutil.rmtree(self.results_min_folder, ignore_errors=True)

    def test_get_data_at_mutation_count_success(self):
        """Test successful retrieval of data at specific mutation count."""
        result = get_data_at_mutation_count(self.sample_pareto_front_max, 10)
        self.assertEqual(result, ("AAAAAAAT", 0.5, 10))
        
        result = get_data_at_mutation_count(self.sample_pareto_front_max, 0)
        self.assertEqual(result, ("AAAAAAAA", 0.05, 0))

    def test_get_data_at_mutation_count_not_found(self):
        """Test error when mutation count not found."""
        with self.assertRaises(ValueError):
            get_data_at_mutation_count(self.sample_pareto_front_max, 17)
        with self.assertRaises(ValueError):
            get_data_at_mutation_count(self.sample_pareto_front_max, 25)
        with self.assertRaises(ValueError):
            get_data_at_mutation_count(self.sample_pareto_front_max, -1)

    @patch('analysis.candidate_selection.os.path.exists')
    def test_preliminary_selection_folder_not_exists(self, mock_exists):
        """Test error when results folder doesn't exist."""
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            preliminary_selection("nonexistent_folder", "output.csv")


    def test_get_pareto_front_paths_folder_not_exists(self):
        """Test error when results folder doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            get_pareto_front_paths("nonexistent_folder")
    

    def test_get_pareto_front_paths_success(self):
        """Test successful retrieval of pareto front paths."""
        result = get_pareto_front_paths(self.results_max_folder)
        
        expected_paths = [
            os.path.join(self.results_max_folder, "1_gene1_gene:123-234_12342143_12341234_312341", "saved_populations", "pareto_front.json"),
            os.path.join(self.results_max_folder, "2_gene2_gene:234-345_23452345_23452345_23452345", "saved_populations", "pareto_front.json"),
            os.path.join(self.results_max_folder, "3_gene3_gene:345-456_34563456_34563456_34563456", "saved_populations", "pareto_front.json"),
        ]
        self.assertEqual(len(result), len(expected_paths))
        for expected in expected_paths:
            self.assertIn(expected, result)


    def test_get_gene_name_from_path(self):
        """Test extraction of gene name from pareto front path."""
        test_path = "/some/path/1_gene1_gene:123-234_12342143_12341234_312341/saved_populations/pareto_front.json"
        result = get_gene_name_from_path(test_path)
        self.assertEqual(result, "gene1")
        
        test_path2 = "lul/another/path/2_gene2_gene:234-345_23452345_23452345_23452345/saved_populations/pareto_front.json"
        result2 = get_gene_name_from_path(test_path2)
        self.assertEqual(result2, "gene2")

    def test_process_mutations(self):
        """Test processing of mutation data from pareto front."""
        result_max = process_mutations(self.sample_pareto_front_max)
        result_min = process_mutations(self.sample_pareto_front_min)
        
        expected_results_min = {
            "fitness_delta_for_1_mutations": 0.1,
            "sequence_for_1_mutations": "AAAAAAAC",
            "fitness_delta_for_2_mutations": np.nan,
            "sequence_for_2_mutations": "",
            "fitness_delta_for_3_mutations": np.nan,
            "sequence_for_3_mutations": "",
            "fitness_delta_for_5_mutations": 0.35,
            "sequence_for_5_mutations": "AAAAAAAG",
            "fitness_delta_for_7_mutations": np.nan,
            "sequence_for_7_mutations": "",
            "fitness_delta_for_10_mutations": 0.55,
            "sequence_for_10_mutations": "AAAAAAAT",
            "fitness_delta_for_15_mutations": 0.75,
            "sequence_for_15_mutations": "AAAAAACA",
            "fitness_delta_for_20_mutations": 0.8,
            "sequence_for_20_mutations": "AAAAAACC"
        }
        expected_results_max = {
            "fitness_delta_for_1_mutations": 0.05,
            "sequence_for_1_mutations": "AAAAAAAC",
            "fitness_delta_for_2_mutations": np.nan,
            "sequence_for_2_mutations": "",
            "fitness_delta_for_3_mutations": np.nan,
            "sequence_for_3_mutations": "",
            "fitness_delta_for_5_mutations": 0.25,
            "sequence_for_5_mutations": "AAAAAAAG",
            "fitness_delta_for_7_mutations": np.nan,
            "sequence_for_7_mutations": "",
            "fitness_delta_for_10_mutations": 0.45,
            "sequence_for_10_mutations": "AAAAAAAT",
            "fitness_delta_for_15_mutations": 0.7,
            "sequence_for_15_mutations": "AAAAAACA",
            "fitness_delta_for_20_mutations": 0.8,
            "sequence_for_20_mutations": "AAAAAACC"
        }
        self.assertEqual(len(result_max), len(expected_results_max))
        self.assertEqual(len(result_min), len(expected_results_min))
        for key, value in expected_results_max.items():
            if pd.isna(value):
                self.assertTrue(pd.isna(result_max[key]), f"Expected NaN for {key} max")
            else:
                found = result_max[key]
                if isinstance(found, float):
                    self.assertAlmostEqual(found, value, msg=f"Mismatch for {key} max")
                else:
                    self.assertEqual(found, value, f"Mismatch for {key} max")

        for key, value in expected_results_min.items():
            if pd.isna(value):
                self.assertTrue(pd.isna(result_min[key]), f"Expected NaN for {key} min")
            else:
                found = result_min[key]
                if isinstance(found, float):
                    self.assertAlmostEqual(found, value, msg=f"Mismatch for {key} min")
                else:
                    self.assertEqual(found, value, f"Mismatch for {key} min")

    def test_process_fitness_thresholds(self):
        """Test processing of fitness threshold data."""
        result_max = process_fitness_thresholds(self.sample_pareto_front_max)
        result_min = process_fitness_thresholds(self.sample_pareto_front_min)

        expected_results_max = {
            "mutations_for_fitness_0.0": np.nan,
            "sequence_for_fitness_0.0": "",
            "mutations_for_fitness_0.1": 1.0,
            "sequence_for_fitness_0.1": "AAAAAAAC",
            "mutations_for_fitness_0.2": 5.0,
            "sequence_for_fitness_0.2": "AAAAAAAG",
            "mutations_for_fitness_0.3": 5.0,
            "sequence_for_fitness_0.3": "AAAAAAAG",
            "mutations_for_fitness_0.4": 10.0,
            "sequence_for_fitness_0.4": "AAAAAAAT",
            "mutations_for_fitness_0.5": 10.0,
            "sequence_for_fitness_0.5": "AAAAAAAT",
            "mutations_for_fitness_0.6": 15.0,
            "sequence_for_fitness_0.6": "AAAAAACA",
            "mutations_for_fitness_0.7": 15.0,
            "sequence_for_fitness_0.7": "AAAAAACA",
            "mutations_for_fitness_0.8": 20.0,
            "sequence_for_fitness_0.8": "AAAAAACC",
            "mutations_for_fitness_0.9": np.nan,
            "sequence_for_fitness_0.9": "",
            "mutations_for_fitness_1.0": np.nan,
            "sequence_for_fitness_1.0": "",
        }
        expected_results_min = {
            "mutations_for_fitness_0.0": np.nan,
            "sequence_for_fitness_0.0": "",
            "mutations_for_fitness_0.1": 15.0,
            "sequence_for_fitness_0.1": "AAAAAACA",
            "mutations_for_fitness_0.2": 15.0,
            "sequence_for_fitness_0.2": "AAAAAACA",
            "mutations_for_fitness_0.3": 10.0,
            "sequence_for_fitness_0.3": "AAAAAAAT",
            "mutations_for_fitness_0.4": 10.0,
            "sequence_for_fitness_0.4": "AAAAAAAT",
            "mutations_for_fitness_0.5": 5.0,
            "sequence_for_fitness_0.5": "AAAAAAAG",
            "mutations_for_fitness_0.6": 5.0,
            "sequence_for_fitness_0.6": "AAAAAAAG",
            "mutations_for_fitness_0.7": 5.0,
            "sequence_for_fitness_0.7": "AAAAAAAG",
            "mutations_for_fitness_0.8": 1.0,
            "sequence_for_fitness_0.8": "AAAAAAAC",
            "mutations_for_fitness_0.9": np.nan,
            "sequence_for_fitness_0.9": "",
            "mutations_for_fitness_1.0": np.nan,
            "sequence_for_fitness_1.0": "",
        }

        self.assertEqual(len(result_max), len(expected_results_max))
        for key, value in expected_results_max.items():
            if pd.isna(value):
                self.assertTrue(pd.isna(result_max[key]), f"Expected NaN for {key} max")
            else:
                found = result_max[key]
                if isinstance(found, float):
                    self.assertAlmostEqual(found, value, msg=f"Mismatch for {key} max")
                else:
                    self.assertEqual(found, value, f"Mismatch for {key} max")

        self.assertEqual(len(result_min), len(expected_results_min))
        for key, value in expected_results_min.items():
            if pd.isna(value):
                self.assertTrue(pd.isna(result_min[key]), f"Expected NaN for {key} min")
            else:
                found = result_min[key]
                if isinstance(found, float):
                    self.assertAlmostEqual(found, value, msg=f"Mismatch for {key} min")
                else:
                    self.assertEqual(found, value, f"Mismatch for {key} min")

    def test_process_single_gene(self):
        """Test processing of a single gene's pareto front data."""
        result_max, increasing_max = process_single_gene(os.path.join(self.results_max_folder, "1_gene1_gene:123-234_12342143_12341234_312341", "saved_populations", "pareto_front.json"))
        result_min, increasing_min = process_single_gene(os.path.join(self.results_min_folder, "1_gene1_gene:123-234_12342143_12341234_312341", "saved_populations", "pareto_front.json"))

        expected_result_max = {
            "gene": "gene1",
            "reference_sequence": "AAAAAAAA",
            "max_num_mutations": 20,
            "start_fitness": 0.05,
            "final_fitness": 0.85,
            "half_max_mutations": 10.0,
            "mutations_for_fitness_0.0": np.nan,
            "sequence_for_fitness_0.0": "",
            "mutations_for_fitness_0.1": 1.0,
            "sequence_for_fitness_0.1": "AAAAAAAC",
            "mutations_for_fitness_0.2": 5.0,
            "sequence_for_fitness_0.2": "AAAAAAAG",
            "mutations_for_fitness_0.3": 5.0,
            "sequence_for_fitness_0.3": "AAAAAAAG",
            "mutations_for_fitness_0.4": 10.0,
            "sequence_for_fitness_0.4": "AAAAAAAT",
            "mutations_for_fitness_0.5": 10.0,
            "sequence_for_fitness_0.5": "AAAAAAAT",
            "mutations_for_fitness_0.6": 15.0,
            "sequence_for_fitness_0.6": "AAAAAACA",
            "mutations_for_fitness_0.7": 15.0,
            "sequence_for_fitness_0.7": "AAAAAACA",
            "mutations_for_fitness_0.8": 20.0,
            "sequence_for_fitness_0.8": "AAAAAACC",
            "mutations_for_fitness_0.9": np.nan,
            "sequence_for_fitness_0.9": "",
            "mutations_for_fitness_1.0": np.nan,
            "sequence_for_fitness_1.0": "",
            "fitness_delta_for_1_mutations": 0.05,
            "sequence_for_1_mutations": "AAAAAAAC",
            "fitness_delta_for_2_mutations": np.nan,
            "sequence_for_2_mutations": "",
            "fitness_delta_for_3_mutations": np.nan,
            "sequence_for_3_mutations": "",
            "fitness_delta_for_5_mutations": 0.25,
            "sequence_for_5_mutations": "AAAAAAAG",
            "fitness_delta_for_7_mutations": np.nan,
            "sequence_for_7_mutations": "",
            "fitness_delta_for_10_mutations": 0.45,
            "sequence_for_10_mutations": "AAAAAAAT",
            "fitness_delta_for_15_mutations": 0.7,
            "sequence_for_15_mutations": "AAAAAACA",
            "fitness_delta_for_20_mutations": 0.8,
            "sequence_for_20_mutations": "AAAAAACC"
        }
        expected_result_min = {
            "gene": "gene1",
            "reference_sequence": "AAAAAAAA",
            "max_num_mutations": 20,
            "start_fitness": 0.85,
            "final_fitness": 0.05,
            "half_max_mutations": 10.0,
            "mutations_for_fitness_0.0": np.nan,
            "sequence_for_fitness_0.0": "",
            "mutations_for_fitness_0.1": 15.0,
            "sequence_for_fitness_0.1": "AAAAAACA",
            "mutations_for_fitness_0.2": 15.0,
            "sequence_for_fitness_0.2": "AAAAAACA",
            "mutations_for_fitness_0.3": 10.0,
            "sequence_for_fitness_0.3": "AAAAAAAT",
            "mutations_for_fitness_0.4": 10.0,
            "sequence_for_fitness_0.4": "AAAAAAAT",
            "mutations_for_fitness_0.5": 5.0,
            "sequence_for_fitness_0.5": "AAAAAAAG",
            "mutations_for_fitness_0.6": 5.0,
            "sequence_for_fitness_0.6": "AAAAAAAG",
            "mutations_for_fitness_0.7": 5.0,
            "sequence_for_fitness_0.7": "AAAAAAAG",
            "mutations_for_fitness_0.8": 1.0,
            "sequence_for_fitness_0.8": "AAAAAAAC",
            "mutations_for_fitness_0.9": np.nan,
            "sequence_for_fitness_0.9": "",
            "mutations_for_fitness_1.0": np.nan,
            "sequence_for_fitness_1.0": "",
            "fitness_delta_for_1_mutations": 0.1,
            "sequence_for_1_mutations": "AAAAAAAC",
            "fitness_delta_for_2_mutations": np.nan,
            "sequence_for_2_mutations": "",
            "fitness_delta_for_3_mutations": np.nan,
            "sequence_for_3_mutations": "",
            "fitness_delta_for_5_mutations": 0.35,
            "sequence_for_5_mutations": "AAAAAAAG",
            "fitness_delta_for_7_mutations": np.nan,
            "sequence_for_7_mutations": "",
            "fitness_delta_for_10_mutations": 0.55,
            "sequence_for_10_mutations": "AAAAAAAT",
            "fitness_delta_for_15_mutations": 0.75,
            "sequence_for_15_mutations": "AAAAAACA",
            "fitness_delta_for_20_mutations": 0.8,
            "sequence_for_20_mutations": "AAAAAACC"
        }
        self.assertTrue(increasing_max)
        self.assertFalse(increasing_min)
        self.assertEqual(len(result_max), len(expected_result_max))
        self.assertEqual(len(result_min), len(expected_result_min))
        for key, value in expected_result_max.items():
            if pd.isna(value):
                self.assertTrue(pd.isna(result_max[key]), f"Expected NaN for {key} max")
            else:
                found = result_max[key]
                if isinstance(found, float):
                    self.assertAlmostEqual(found, value, msg=f"Mismatch for {key} max")
                else:
                    self.assertEqual(found, value, f"Mismatch for {key} max")
        for key, value in expected_result_min.items():
            if pd.isna(value):
                self.assertTrue(pd.isna(result_min[key]), f"Expected NaN for {key} min")
            else:
                found = result_min[key]
                if isinstance(found, float):
                    self.assertAlmostEqual(found, value, msg=f"Mismatch for {key} min")
                else:
                    self.assertEqual(found, value, f"Mismatch for {key} min")

    def test_filter_and_sort_results(self):
        """Test filtering and sorting of results DataFrame."""
        data = {
            "start_fitness": [0.1, 0.05, 0.15, 0.08],
            "gene": ["A", "B", "C", "D"]
        }
        df = pd.DataFrame(data)
        
        # Test with max filter
        result = filter_and_sort_results(df, starting_fitness_max=0.1, starting_fitness_min=None, ascending=True)
        expected_genes = ["B", "D", "A"]  # Sorted by start_fitness ascending
        self.assertEqual(list(result["gene"]), expected_genes)
        
        # Test with min filter
        result = filter_and_sort_results(df, starting_fitness_max=None, starting_fitness_min=0.08, ascending=True)
        expected_genes = ["D", "A", "C"]  # Sorted by start_fitness ascending
        self.assertEqual(list(result["gene"]), expected_genes)
        
        # Test with both filters
        result = filter_and_sort_results(df, starting_fitness_max=0.12, starting_fitness_min=0.06, ascending=True)
        expected_genes = ["D", "A"]  # Only D and A meet criteria
        self.assertEqual(list(result["gene"]), expected_genes)
        
        # Test with no filters
        result = filter_and_sort_results(df, starting_fitness_max=None, starting_fitness_min=None, ascending=False)
        expected_genes = ["C", "A", "D", "B"]  # All genes, sorted
        self.assertEqual(list(result["gene"]), expected_genes)


    @patch('analysis.candidate_selection.filter_and_sort_results')
    @patch('analysis.candidate_selection.process_single_gene')
    @patch('analysis.candidate_selection.get_pareto_front_paths')
    @patch('pandas.DataFrame.to_csv')
    def test_preliminary_selection_success(self, mock_to_csv, mock_get_paths, 
                                         mock_process_gene, mock_filter_sort):
        """Test successful preliminary selection with new structure."""
        # Setup mocks
        mock_paths = [
            "/path/gene_ABC123/saved_populations/pareto_front.json",
            "/path/gene_DEF456/saved_populations/pareto_front.json"
        ]
        mock_get_paths.return_value = mock_paths
        
        mock_gene_data = [
            ({"gene": "ABC123", "start_fitness": 0.05, "other": "data1"}, True),
            ({"gene": "DEF456", "start_fitness": 0.1, "other": "data2"}, True)
        ]
        mock_process_gene.side_effect = mock_gene_data

        mock_filtered_df = pd.DataFrame([data for data, _ in mock_gene_data]).set_index("gene")
        mock_filter_sort.return_value = mock_filtered_df
        
        with patch('builtins.print'):  # Mock print to avoid output during test
            preliminary_selection("test_folder", "output.csv", 
                                starting_fitness_max=0.2, starting_fitness_min=0.01)
        
        # Verify function calls
        mock_get_paths.assert_called_once_with("test_folder")
        self.assertEqual(mock_process_gene.call_count, 2)
        mock_filter_sort.assert_called_once()
        mock_to_csv.assert_called_once_with("output.csv")
    
    def test_load_selected_genes_file_not_exists(self):
        """Test error when CSV file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            load_selected_genes("nonexistent.csv")

    def test_load_selected_genes_success(self):
        """Test successful loading of selected genes."""
        mock_df = pd.DataFrame({"gene": ["ABC", "DEF"], "value": [1, 2]}).set_index("gene")
        mock_df.to_csv(os.path.join(self.temp_dir, "test.csv"))

        result = load_selected_genes(os.path.join(self.temp_dir,"test.csv"))
        #check that the result is the same as mock_df
        pd.testing.assert_frame_equal(result, mock_df)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('os.path.join')
    def test_draw_line_plot(self, mock_join, mock_figure, mock_close, mock_savefig):
        """Test drawing line plot functionality."""
        mock_join.return_value = "test_path.png"
        single_points = [(0.1, 5), (0.5, 10), (0.9, 20)]
        multi_points = [(0.2, 3), (0.6, 8), (0.8, 15)]
        
        draw_line_plot("vis_folder", "test_gene", single_points, multi_points)
        
        mock_savefig.assert_called_once_with("test_path.png", bbox_inches='tight')
        mock_close.assert_called_once()

    def test_get_line_plot_data(self):
        """Test extraction of line plot data from DataFrame row."""
        data = {
            "mutations_for_fitness_0.1_single": 5,
            "mutations_for_fitness_0.5_single": 10,
            "mutations_for_fitness_0.1_multi": 3,
            "mutations_for_fitness_0.5_multi": 8,
            "start_fitness_single": 0.0,
            "final_fitness_single": 1.0,
            "max_num_mutations_single": 20,
            "start_fitness_multi": 0.0,
            "final_fitness_multi": 0.95,
            "max_num_mutations_multi": 18
        }
        row = pd.Series(data)
        
        single_points, multi_points = get_line_plot_data(row)
        
        expected_single = [(0.0, 0), (0.1, 5), (0.5, 10), (1.0, 20)]
        expected_multi = [(0.0, 0), (0.1, 3), (0.5, 8), (0.95, 18)]
        self.assertEqual(len(single_points), 4)  # 2 fitness points + start + final
        self.assertEqual(len(multi_points), 4)
        self.assertEqual(single_points, expected_single)
        self.assertEqual(multi_points, expected_multi)

    @patch('analysis.candidate_selection.draw_line_plot')
    @patch('os.makedirs')
    def test_compare_trajectories_mutations(self, mock_makedirs, mock_draw):
        """Test trajectory comparison functionality."""
        data = {
            "mutations_for_fitness_0.1_single": [5],
            "mutations_for_fitness_0.1_multi": [3],
            "start_fitness_single": [0.0],
            "final_fitness_single": [1.0],
            "max_num_mutations_single": [20],
            "start_fitness_multi": [0.0],
            "final_fitness_multi": [1.0],
            "max_num_mutations_multi": [18]
        }
        merged = pd.DataFrame(data, index=["gene1"])
        
        compare_trajectories_mutations(merged)
        
        mock_makedirs.assert_called_once()
        mock_draw.assert_called_once()

    def test_get_best_fitness_and_seq_both_na(self):
        """Test get_best_fitness_and_seq when both values are NaN."""
        row = pd.Series({
            "single_col": np.nan,
            "multi_col": np.nan
        })
        
        fitness, seq = get_best_fitness_and_seq(row, 5, "single_col", "multi_col")
        
        self.assertTrue(np.isnan(fitness))
        self.assertEqual(seq, "")

    def test_get_best_fitness_and_seq_one_na(self):
        """Test get_best_fitness_and_seq when one of the values is NaN."""
        row = pd.Series({
            "single_col": np.nan,
            "multi_col": 0.6,
            "sequence_for_5_mutations_multi": "GCTA",
            "sequence_for_5_mutations_single": "ATCG"
        })
        
        fitness_delta, seq = get_best_fitness_and_seq(row, 5, "single_col", "multi_col")
        
        self.assertEqual(fitness_delta, 0.6)
        self.assertEqual(seq, "GCTA")

    def test_get_best_fitness_and_seq_single_better(self):
        """Test get_best_fitness_and_seq when single is better."""
        row = pd.Series({
            "single_col": 0.8,
            "multi_col": 0.6,
            "sequence_for_5_mutations_single": "ATCG",
            "sequence_for_5_mutations_multi": "GCTA"
        })
        
        fitness, seq = get_best_fitness_and_seq(row, 5, "single_col", "multi_col")
        
        self.assertEqual(fitness, 0.8)
        self.assertEqual(seq, "ATCG")

    def test_find_best_mutation_deltas(self):
        """Test finding best mutation deltas across methods."""
        data = {
        }
        for mutation in [1, 2, 3, 5, 7, 10, 15, 20]:
            data[f"fitness_delta_for_{mutation}_mutations_single"] = [0.5, 0.3]
            data[f"fitness_delta_for_{mutation}_mutations_multi"] = [0.4, 0.6]
            data[f"sequence_for_{mutation}_mutations_single"] = ["SEQ1", "SEQ3"]
            data[f"sequence_for_{mutation}_mutations_multi"] = ["SEQ2", "SEQ4"]
        merged = pd.DataFrame(data, index=["gene1", "gene2"])
        
        find_best_mutation_deltas(merged)
        
        self.assertIn("best_fitness_delta_for_2_mutations", merged.columns)
        self.assertIn("per_mutation_delta_for_2_mutations", merged.columns)
        self.assertEqual(merged.loc["gene1", "best_fitness_delta_for_2_mutations"], 0.5)
        self.assertEqual(merged.loc["gene2", "best_fitness_delta_for_2_mutations"], 0.6)
        self.assertEqual(merged.loc["gene1", "best_sequence_for_2_mutations"], "SEQ1")
        self.assertEqual(merged.loc["gene2", "best_sequence_for_2_mutations"], "SEQ4")
        self.assertEqual(merged.loc["gene1", "per_mutation_delta_for_2_mutations"], 0.25)
        self.assertEqual(merged.loc["gene2", "per_mutation_delta_for_2_mutations"], 0.3)

    def test_find_best_mutation_count_per_gene(self):
        """Test finding best mutation count for a gene."""
        data = {
            "per_mutation_delta_for_1_mutations": 0.5,
            "per_mutation_delta_for_2_mutations": 0.3,
            "per_mutation_delta_for_5_mutations": 0.7,
            "best_fitness_delta_for_1_mutations": 0.2,
            "best_fitness_delta_for_2_mutations": 0.4,
            "best_fitness_delta_for_5_mutations": 0.6
        }
        row = pd.Series(data)
        
        result = find_best_mutation_count_per_gene(row)
        self.assertEqual(result, 5)
        
        result = find_best_mutation_count_per_gene(row, rank=2)
        self.assertEqual(result, 2)

        result = find_best_mutation_count_per_gene(row, rank=3)
        self.assertTrue(pd.isna(result))

    def test_find_best_mutations_per_gene(self):
        """Test finding best mutations per gene for multiple ranks."""
        data = {
            "per_mutation_delta_for_1_mutations": [0.5, 0.3],
            "per_mutation_delta_for_2_mutations": [0.3, 0.6],
            "per_mutation_delta_for_5_mutations": [0.7, 0.2],
            "best_fitness_delta_for_1_mutations": [0.1, 0.3],
            "best_fitness_delta_for_2_mutations": [0.2, 0.6],
            "best_fitness_delta_for_5_mutations": [0.7, 0.2],
        }
        merged = pd.DataFrame(data, index=["gene1", "gene2"])
        
        find_best_mutations_per_gene(merged, n_candidates=2)
        
        self.assertIn("best_mutation_count_rank_1", merged.columns)
        self.assertIn("best_mutation_count_rank_2", merged.columns)
        self.assertEqual(merged.loc["gene1", "best_mutation_count_rank_1"], 5)
        self.assertEqual(merged.loc["gene2", "best_mutation_count_rank_1"], 2)
        self.assertTrue(pd.isna(merged.loc["gene1", "best_mutation_count_rank_2"]))
        self.assertEqual(merged.loc["gene2", "best_mutation_count_rank_2"], 1)

    def test_selected_per_mutation(self):
        """Test selection of best candidates per mutation count."""
        data = {}
        for mutation in [1, 2, 3, 5, 7, 10, 15, 20]:
            data[f"best_fitness_delta_for_{mutation}_mutations"] = [0.5, 0.3, 0.8]
            data[f"best_sequence_for_{mutation}_mutations"] = ["SEQ1", "SEQ2", "SEQ3"]
            data[f"reference_sequence_single"] = ["REF1", "REF2", "REF3"]
            data[f"start_fitness_single"] = [0.1, 0.2, 0.0]
            data["final_fitness_single"] = [0.6, 0.5, 0.8]
        merged = pd.DataFrame(data, index=["gene1", "gene2", "gene3"])
        
        result = selected_per_mutation(merged, n_candidates=2)
        expected_result = pd.DataFrame({
            "gene": ["gene3", "gene1"] * 8,
            "mutations": [1, 1, 2, 2, 3, 3, 5, 5, 7, 7, 10, 10, 15, 15, 20, 20],
            "start_fitness": [0.0, 0.1] * 8,
            "best_fitness_delta": [0.8, 0.5] * 8,
            "fitness": [0.8, 0.6] * 8,
            "best_sequence": ["SEQ3", "SEQ1"] * 8,
            "reference_sequence": ["REF3", "REF1"] * 8,
        })
        expected_result["selection_reason"] = "top_per_mutation"
        pd.testing.assert_frame_equal(result, expected_result)

    def test_selected_per_gene(self):
        """Test selection of best candidates per gene."""
        data = {
            "best_mutation_count_rank_1": ["1", "2"],
            "best_mutation_count_rank_2": ["2", "1"],
            "best_fitness_delta_for_1_mutations": [0.5, 0.3],
            "best_fitness_delta_for_2_mutations": [0.3, 0.6],
            "best_sequence_for_1_mutations": ["SEQ1", "SEQ3"],
            "best_sequence_for_2_mutations": ["SEQ2", "SEQ4"],
            "reference_sequence_single": ["REF1", "REF2"],
            "start_fitness_single": [0.1, 0.2],
            "final_fitness_single": [0.6, 0.8]
        }
        merged = pd.DataFrame(data, index=["gene1", "gene2"])
        
        result = selected_per_gene(merged, n_candidates=2)
        expected = pd.DataFrame({
            "gene": ["gene1", "gene1", "gene2", "gene2"],
            "mutations": [1, 2, 2, 1],
            "start_fitness": [0.1, 0.1, 0.2, 0.2],
            "best_fitness_delta": [0.5, 0.3, 0.6, 0.3],
            "fitness": [0.6, 0.4, 0.8, 0.5],
            "best_sequence": ["SEQ1", "SEQ2", "SEQ4", "SEQ3"],
            "reference_sequence": ["REF1", "REF1", "REF2", "REF2"],
        })
        expected["selection_reason"] = "top_per_gene"
        pd.testing.assert_frame_equal(result, expected)
        

    @patch('analysis.candidate_selection.load_selected_genes')
    def test_final_selection(self, mock_load):
        """Test final selection process."""
        # Mock loaded data
        single_data = pd.DataFrame({
            "fitness_delta_for_1_mutations": [0.1, 0.2],
            "fitness_delta_for_2_mutations": [0.3, 0.2],
            "fitness_delta_for_3_mutations": [0.4, 0.5],
            "fitness_delta_for_5_mutations": [0.5, 0.7],
            "fitness_delta_for_7_mutations": [0.8, 0.75],
            "fitness_delta_for_10_mutations": [np.nan, 0.85],
            "fitness_delta_for_15_mutations": [np.nan, np.nan],
            "fitness_delta_for_20_mutations": [np.nan, np.nan],
            "sequence_for_1_mutations": ["SEQ1", "SEQ9"],
            "sequence_for_2_mutations": ["SEQ2", "SEQ10"],
            "sequence_for_3_mutations": ["SEQ3", "SEQ11"],
            "sequence_for_5_mutations": ["SEQ4", "SEQ12"],
            "sequence_for_7_mutations": ["SEQ5", "SEQ13"],
            "sequence_for_10_mutations": ["SEQ6", "SEQ14"],
            "sequence_for_15_mutations": ["SEQ7", "SEQ15"],
            "sequence_for_20_mutations": ["SEQ8", "SEQ16"],
            "reference_sequence": ["REF1", "REF2"],
            "start_fitness": [0.1, 0.1],
            "final_fitness": [0.6, 0.95]
        }, index=["gene1", "gene2"])
        
        multi_data = pd.DataFrame({
            "fitness_delta_for_1_mutations": [0.09, 0.19],
            "fitness_delta_for_2_mutations": [0.29, 0.19],
            "fitness_delta_for_3_mutations": [0.41, 0.55],
            "fitness_delta_for_5_mutations": [0.49, 0.69],
            "fitness_delta_for_7_mutations": [0.79, 0.74],
            "fitness_delta_for_10_mutations": [np.nan, 0.84],
            "fitness_delta_for_15_mutations": [np.nan, np.nan],
            "fitness_delta_for_20_mutations": [np.nan, np.nan],
            "sequence_for_1_mutations": ["SEQ21", "SEQ29"],
            "sequence_for_2_mutations": ["SEQ22", "SEQ210"],
            "sequence_for_3_mutations": ["SEQ23", "SEQ211"],
            "sequence_for_5_mutations": ["SEQ24", "SEQ212"],
            "sequence_for_7_mutations": ["SEQ25", "SEQ213"],
            "sequence_for_10_mutations": ["SEQ26", "SEQ214"],
            "sequence_for_15_mutations": ["SEQ27", "SEQ215"],
            "sequence_for_20_mutations": ["SEQ28", "SEQ216"],
            "reference_sequence": ["REF1", "REF2"],
            "start_fitness": [0.1, 0.1],
            "final_fitness": [0.65, 0.9]
        }, index=["gene1", "gene2"])

        mock_load.side_effect = [single_data, multi_data]
        
        final_selection("single.csv", "multi.csv", os.path.join(self.temp_dir, "output.csv"), n_candidates=1)
        expected = pd.DataFrame({
            "gene": ["gene2", "gene1", "gene2", "gene2", "gene1", "gene2", "gene1", "gene1"],
            "mutations": [1, 2, 3, 5, 7, 10, 15, 20],
            "start_fitness": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "best_fitness_delta": [0.2, 0.3, .55, .7, .8, .85, np.nan, np.nan],
            "fitness": [0.3, 0.4, .65, .8, .9, .95, np.nan, np.nan],
            "best_sequence": ["SEQ9", "SEQ2", "SEQ211", "SEQ12", "SEQ5", "SEQ14", pd.NA, pd.NA],
            "reference_sequence": ["REF2", "REF1", "REF2", "REF2", "REF1", "REF2", "REF1", "REF1"],
            "selection_reason": ["top_per_mutation", "both", "both", "top_per_mutation", "top_per_mutation", "top_per_mutation", "top_per_mutation", "top_per_mutation"]
        })
        result = pd.read_csv(os.path.join(self.temp_dir, "output.csv"))
        for i, row in expected.iterrows():
            row_in_result = result[(result["gene"] == row["gene"]) & (result["mutations"] == row["mutations"])]
            self.assertEqual(len(row_in_result), 1, f"Row for gene {row['gene']} with mutations {row['mutations']} not found or duplicated.")
            for col in expected.columns:
                if pd.isna(row[col]):
                    self.assertTrue(pd.isna(row_in_result.iloc[0][col]), f"Expected NaN for {col} in gene {row['gene']} mutations {row['mutations']}")
                else:
                    if isinstance(row[col], float):
                        self.assertAlmostEqual(row_in_result.iloc[0][col], row[col], msg=f"Mismatch in column {col} for gene {row['gene']} mutations {row['mutations']}")
                    else:
                        self.assertEqual(row_in_result.iloc[0][col], row[col], f"Mismatch in column {col} for gene {row['gene']} mutations {row['mutations']}")


    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_csv')
    def test_one_off_error_correction(self, mock_to_csv, mock_read_csv):
        """Test one-off error correction functionality."""
        data = pd.DataFrame({
            "value": [1, 2]
        }, index=["prefix_gene1", "prefix_gene2_postfix"])
        data.index.name = "gene"
        
        mock_read_csv.return_value = data
        
        one_off_error_correction("test.csv")
        
        mock_to_csv.assert_called_once_with("test.csv")

    def test_combine_selections(self):
        """Test combining per-gene and per-mutation selections."""
        per_gene_data = pd.DataFrame({
            "gene": ["gene1", "gene1", "gene2"],
            "mutations": [1, 2, 3],
            "start_fitness": [0.1, 0.1, 0.2],
            "best_fitness_delta": [0.5, 0.3, 0.6],
            "fitness": [0.6, 0.4, 0.8],
            "best_sequence": ["SEQ1", "SEQ2", "SEQ3"],
            "reference_sequence": ["REF1", "REF1", "REF2"],
            "selection_reason": ["top_per_gene", "top_per_gene", "top_per_gene"]
        })
        
        per_mutation_data = pd.DataFrame({
            "gene": ["gene1", "gene3"],
            "mutations": [1, 2],
            "start_fitness": [0.1, 0.3],
            "best_fitness_delta": [0.5, 0.4],
            "fitness": [0.6, 0.7],
            "best_sequence": ["SEQ1", "SEQ4"],
            "reference_sequence": ["REF1", "REF3"],
            "selection_reason": ["top_per_mutation", "top_per_mutation"]
        })
        
        result = combine_selections(per_gene_data, per_mutation_data)
        
        # Should have 3 rows: gene1-mutation1 (both), gene1-mutation2 (per_gene), gene2-mutation3 (per_gene), gene3-mutation2 (per_mutation)
        expected_rows = 4
        self.assertEqual(len(result), expected_rows)
        
        # Check that gene1-mutation1 is marked as "both"
        both_row = result[(result["gene"] == "gene1") & (result["mutations"] == 1)]
        self.assertEqual(len(both_row), 1)
        self.assertEqual(both_row.iloc[0]["selection_reason"], "both")
        
        # Check that gene1-mutation2 is still "top_per_gene"
        per_gene_row = result[(result["gene"] == "gene1") & (result["mutations"] == 2)]
        self.assertEqual(len(per_gene_row), 1)
        self.assertEqual(per_gene_row.iloc[0]["selection_reason"], "top_per_gene")
        
        # Check that gene3-mutation2 keeps "top_per_mutation"
        per_mutation_row = result[(result["gene"] == "gene3") & (result["mutations"] == 2)]
        self.assertEqual(len(per_mutation_row), 1)
        self.assertEqual(per_mutation_row.iloc[0]["selection_reason"], "top_per_mutation")

    def test_get_values_for_sorting(self):
        """Test getting values for sorting with minimum delta threshold."""
        row = pd.Series({
            "per_mutation_delta_for_1_mutations": 0.2,
            "per_mutation_delta_for_2_mutations": 0.2,
            "per_mutation_delta_for_5_mutations": .15,
            "per_mutation_delta_for_10_mutations": np.nan,
            "best_fitness_delta_for_1_mutations": 0.2,
            "best_fitness_delta_for_2_mutations": 0.4,
            "best_fitness_delta_for_5_mutations": 0.75,
            "best_fitness_delta_for_10_mutations": np.nan,
        })
        
        per_mutation_columns = [
            "per_mutation_delta_for_1_mutations",
            "per_mutation_delta_for_2_mutations", 
            "per_mutation_delta_for_5_mutations",
            "per_mutation_delta_for_10_mutations",
        ]
        min_delta = 0.3
        
        result = get_sorted_values(row, per_mutation_columns, min_delta)
        expected = [
            ("per_mutation_delta_for_2_mutations", 0.2),
            ("per_mutation_delta_for_5_mutations", .15),
            ("per_mutation_delta_for_1_mutations", -np.inf),
            ("per_mutation_delta_for_10_mutations", -np.inf),
        ]
        
        # Should return tuples of (column, value)
        self.assertEqual(len(result), 4)
        
        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1], expected[1])
        for i in range(2, 4):
            self.assertTrue(np.isnan(result[i][1]) or result[i][1] == -np.inf)
        self.assertTrue(all(col in [r[0] for r in result] for col in per_mutation_columns))

if __name__ == '__main__':
    unittest.main()
    # test_obj = TestCandidateSelection()
    # test_obj.setUp()
    # test_obj.test_find_best_mutation_count_per_gene()
    # test_obj.tearDown()
    # print("Test completed.")
