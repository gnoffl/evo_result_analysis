import unittest
import pandas as pd
import numpy as np
import os
import json
import tempfile
import math
import shutil
from pandas.testing import assert_series_equal
from unittest.mock import patch, mock_open, MagicMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.analyze_mapping import (
    get_epm_tfbs_mapping_old,
    get_epm_tfbs_mapping_new,
    add_epm,
    get_epm_count_table,
    get_min_max_dfs,
    get_difference_number_epm,
    statistics_epm_before_after,
    add_tf_names,
    compare_initial_and_final_distribution,
    add_fitness,
    find_diff_tf,
    add_diff_tf,
    add_original_sequence_and_number_mutations,
    analyze_mutations_effect
)


class TestGetEpmTfbsMapping(unittest.TestCase):
    
    def setUp(self):
        self.sample_old_data = pd.DataFrame({
            'epm': ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5', 'epm_ara_msr_max_corrected_p1m10'],
            'tf_name': ['AraC', 'CRP', 'AraC'],
            'loc': ['gene1_mutations_0', 'gene1_mutations_1', 'gene2_mutations_0'],
            'type': ['gene', 'gene', 'gene']
        })
        
        self.sample_new_data = pd.DataFrame({
            'motif': ['epm_ara_msr_max_corrected_p1m10F_1', 'epm_crp_p2m5F_2', 'epm_ara_msr_max_corrected_p1m10R_1'],
            'sequence_id': ['run1__gene1_mutations_0:0_seed', 'run1__gene1_mutations_1:1_seed', 'run1__gene2_mutations_0:0_seed']
        })
    
    def test_get_epm_tfbs_mapping_old(self):
        with patch('pandas.read_csv', return_value=self.sample_old_data):
            result = get_epm_tfbs_mapping_old('dummy_path.csv')
            
        expected = pd.DataFrame({
            'epm': ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5'],
            'tf_name': ['AraC', 'CRP']
        })
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_get_epm_tfbs_mapping_new(self):
        with patch('pandas.read_csv', return_value=self.sample_new_data):
            result = get_epm_tfbs_mapping_new('dummy_path.csv')

        self.assertIn('epm', result.columns)
        self.assertIn('tf_name', result.columns)
        self.assertTrue(all(result['tf_name'] == 'not_implemented'))


class TestAddEpm(unittest.TestCase):
    
    def test_add_epm_success(self):
        data = pd.DataFrame({
            'motif': ['epm_ara_msr_max_corrected_p1m10F_1', 'epm_crp_p2m5R_2']
        })
        add_epm(data)
        
        expected_epm = ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5']
        self.assertEqual(data['epm'].tolist(), expected_epm)
    
    def test_add_epm_already_exists(self):
        data = pd.DataFrame({
            'motif': ['epm_ara_msr_max_corrected_p1m10F_1'],
            'epm': ['existing_epm']
        })
        add_epm(data)
        
        # Should not overwrite existing epm column
        self.assertEqual(data['epm'].tolist(), ['existing_epm'])


class TestGetEpmCountTable(unittest.TestCase):
    
    def setUp(self):
        self.sample_new_data = pd.DataFrame({
            'motif': ['epm_ara_msr_max_corrected_p1m10F_1', 'epm_crp_p2m5F_2', 'epm_crp_p2m5R_2', 'epm_ara_msr_max_corrected_p1m10F_1', 'epm_ara_msr_max_corrected_p1m10F_1'],
            'sequence_id': ['run1__gene1_mutations_0:0_seed', 'run1__gene1_mutations_1:1_seed', 'run1__gene1_mutations_0:0_seed', 'run1__gene1_mutations_1:1_seed', 'run1__gene1_mutations_1:1_seed']
        })
        
        self.sample_old_data = pd.DataFrame({
            'epm': ['epm_ara_msr_max_corrected_p1m10F_1', 'epm_crp_p2m5F_2', 'epm_crp_p2m5F_2', 'epm_ara_msr_max_corrected_p1m10F_1', 'epm_ara_msr_max_corrected_p1m10R_1'],
            'tf_name': ['AraC', 'CRP', "CRP", 'AraC', 'AraC'],
            'loc': ['run1__gene1_mutations_0:0_seed', 'run1__gene1_mutations_1:1_seed', 'run1__gene1_mutations_0:0_seed', 'run1__gene1_mutations_1:1_seed', 'run1__gene1_mutations_1:1_seed'],
            'type': ['gene', 'gene', 'gene', 'gene', 'gene'],
            'motif': ['epm_ara_msr_max_corrected_p1m10p1m10', 'epm_crp_p2m5', 'epm_crp_p2m5', 'epm_ara_msr_max_corrected_p1m10p1m10', 'epm_ara_msr_max_corrected_p1m10p1m10']
        })
    
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_csv')
    def test_get_epm_count_table_new(self, mock_to_csv, mock_read_csv):
        mock_read_csv.return_value = self.sample_new_data
        
        result = get_epm_count_table('dummy_path.csv', new=True)
        
        self.assertIn('sequence', result.columns)
        self.assertTrue(any(col.startswith('epm_') for col in result.columns))
        mock_to_csv.assert_called_once()
        self.assertIn('epm_ara_msr_max_corrected_p1m10', result.columns)
        self.assertIn('epm_crp_p2m5', result.columns)
        self.assertIn('run1__gene1_mutations_0:0_seed', result["sequence"].values)
        self.assertIn('run1__gene1_mutations_1:1_seed', result["sequence"].values)
        self.assertEqual(result.loc[result["sequence"] == "run1__gene1_mutations_0:0_seed"]['epm_ara_msr_max_corrected_p1m10'].item(), 1)
        self.assertEqual(result.loc[result["sequence"] == "run1__gene1_mutations_0:0_seed", 'epm_crp_p2m5'].item(), 0)
        self.assertEqual(result.loc[result["sequence"] == "run1__gene1_mutations_1:1_seed", 'epm_ara_msr_max_corrected_p1m10'].item(), 2)
        self.assertEqual(result.loc[result["sequence"] == "run1__gene1_mutations_1:1_seed", 'epm_crp_p2m5'].item(), 1)

    
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_csv')
    def test_get_epm_count_table_old(self, mock_to_csv, mock_read_csv):
        mock_read_csv.return_value = self.sample_old_data
        
        result = get_epm_count_table('dummy_path.csv', new=False)
        
        self.assertIn('sequence', result.columns)
        mock_to_csv.assert_called_once()


class TestGetMinMaxDfs(unittest.TestCase):
    
    def setUp(self):
        self.sample_epm_counts = pd.DataFrame({
            'sequence': ['gene1_mutations_0', 'gene1_mutations_1', 'gene1_mutations_2'],
            'epm_ara_msr_max_corrected_p1m10': [2, 1, 3],
            'epm_crp_p2m5': [1, 2, 0],
        })
    
    def test_get_min_max_dfs_raises_error_when_not_remembered(self):
        with self.assertRaisesRegex(ValueError, "Remember that for some reason"):
            get_min_max_dfs(self.sample_epm_counts, remembered_issues=False)
    
    def test_get_min_max_dfs_with_remembered_issues(self):
        min_df, max_df = get_min_max_dfs(self.sample_epm_counts, remembered_issues=True)
        
        self.assertIsInstance(min_df, pd.DataFrame)
        self.assertIsInstance(max_df, pd.DataFrame)
        expected_min = pd.DataFrame({
            'sequence': ['gene1_mutations_0'],
            'epm_ara_msr_max_corrected_p1m10': [2],
            'epm_crp_p2m5': [1],
            'number_mutations': [0],
            'original_sequence': ['gene1'],
            "max_number_mutations": [2],
        }).set_index("original_sequence")
        expected_max = pd.DataFrame({
            'sequence': ['gene1_mutations_2'],
            'epm_ara_msr_max_corrected_p1m10': [3],
            'epm_crp_p2m5': [0],
            'number_mutations': [2],
            'original_sequence': ['gene1'],
            "max_number_mutations": [2],
        }).set_index("original_sequence")

        pd.testing.assert_frame_equal(min_df, expected_min, check_like=True, check_index_type=False)
        pd.testing.assert_frame_equal(max_df, expected_max, check_like=True, check_index_type=False)

        self.assertTrue('original_sequence' in min_df.index.names or min_df.index.name == 'original_sequence')


class TestStatisticsEpmBeforeAfter(unittest.TestCase):
    
    def setUp(self):
        self.sample_epm_counts = pd.DataFrame({
            'sequence': ['1_mutations_0', '1_mutations_1', '1_mutations_2', "2_mutations_0", "2_mutations_3", "3_mutations_0", "3_mutations_1", "4_mutations_0", "4_mutations_5", "5_mutations_0", "5_mutations_1", "6_mutations_0", "6_mutations_2", "7_mutations_0", "7_mutations_1", "8_mutations_0", "8_mutations_2"],

            # always increases by one from min mutations to max mutations --> significant
            'epm_ara_msr_max_corrected_p1m10': [0, 1, 1, 1, 2, 1, 2, 2, 3, 2, 3, 2, 3, 3, 4, 3, 4],
            # always decreases by one from min mutations to max mutations --> significant
            'epm_crp_p2m5': [1, 2, 0, 2, 1, 2, 1, 3, 2, 3, 2, 3, 2, 4, 3, 4, 3],
            # alternates between increasing and decreasing by one --> not significant
            'epm_asd_p1m6': [1, 2, 0, 0, 1, 2, 1, 1, 2, 3, 2, 2, 3, 4, 3, 3, 4],

            'number_mutations': [0, 1, 2, 0, 3, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 2],
            'original_sequence': ['1', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5', '6', '6', '7', '7', '8', '8']
        })
    
    def test_statistics_epm_before_after(self):
        result = statistics_epm_before_after(self.sample_epm_counts)
        
        self.assertIsInstance(result, dict)
        self.assertIn('epm_ara_msr_max_corrected_p1m10', result)
        self.assertIn('epm_crp_p2m5', result)
        self.assertIn('epm_asd_p1m6', result)
        self.assertTrue(all(isinstance(v, float) for v in result.values()))
        self.assertLess(result["epm_ara_msr_max_corrected_p1m10"], 0.05)
        self.assertLess(result["epm_crp_p2m5"], 0.05)
        self.assertGreater(result["epm_asd_p1m6"], 0.05)


class TestAddTfNames(unittest.TestCase):
    
    def setUp(self):
        self.sample_mapping = pd.DataFrame({
            'epm': ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5'],
            'tf_name': ['AraC', 'CRP']
        })
    
    def test_add_tf_names(self):
        df = pd.DataFrame({
            'counts': [1, 2, 3]
        }, index=['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5', 'epm_unknown'])
        
        result = add_tf_names(df, self.sample_mapping)
        
        self.assertIn('tf_name', result.columns)
        self.assertEqual(result.loc['epm_ara_msr_max_corrected_p1m10', 'tf_name'], 'AraC')
        self.assertEqual(result.loc['epm_crp_p2m5', 'tf_name'], 'CRP')
        self.assertTrue(math.isnan(result.loc['epm_unknown', 'tf_name'])) #type:ignore


class TestCompareInitialAndFinalDistribution(unittest.TestCase):
    
    def setUp(self):
        self.sample_mapping = pd.DataFrame({
            'epm': ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5'],
            'tf_name': ['AraC', 'CRP']
        })
    
    @patch('analysis.analyze_mapping.get_epm_count_table')
    @patch('analysis.analyze_mapping.statistics_epm_before_after')
    def test_compare_initial_and_final_distribution(self, mock_statistics, mock_get_epm_count_table):
        epm_counts = pd.DataFrame({
            'sequence': ['gene1_mutations_0', 'gene1_mutations_5', "gene2_mutations_0", "gene2_mutations_3"],
            'epm_ara_msr_max_corrected_p1m10': [5, 10, 4, 12],
            'epm_crp_p2m5': [3, 2, 4, 3]
        })
        
        mock_get_epm_count_table.return_value = epm_counts
        mock_statistics.return_value = {'epm_ara_msr_max_corrected_p1m10': 0.005, 'epm_crp_p2m5': 0.03}
        
        with tempfile.TemporaryDirectory() as tmp_path:
            result = compare_initial_and_final_distribution(
                input_path='dummy_path.csv',
                mapping=self.sample_mapping,
                output_folder=tmp_path,
                new=True
            )
            
            output_file = os.path.join(tmp_path, "initial_vs_final_per_epm_counts.csv")
            self.assertTrue(os.path.exists(output_file))
            expected_columns = ["initial", "final", "difference", "log_ratio", "significance", "tf_name"]
            self.assertTrue(all(col in result.columns for col in expected_columns))
            epm_ara_row = pd.Series({
                "initial": 9,
                "final": 22,
                "difference": 13,
                "log_ratio": np.log2(22/9),
                "significance": 0.005,
                "tf_name": "AraC"
            }, name='epm_ara_msr_max_corrected_p1m10')
            epm_crp_row = pd.Series({
                "initial": 7,
                "final": 5,
                "difference": -2,
                "log_ratio": np.log2(5/7),
                "significance": 0.03,
                "tf_name": "CRP"
            }, name='epm_crp_p2m5')
            sum_row = pd.Series({
                "initial": 16,
                "final": 27,
                "difference": 11,
                "log_ratio": np.log2(27/16),
                "significance": 0.035,
                "tf_name": pd.NA
            }, name='sum')
            assert_series_equal(result.loc['epm_ara_msr_max_corrected_p1m10'], epm_ara_row, check_exact=False)
            assert_series_equal(result.loc['epm_crp_p2m5'], epm_crp_row, check_exact=False)
            assert_series_equal(result.loc['sum'], sum_row, check_exact=False)
            
            # Verify the mocks were called
            mock_get_epm_count_table.assert_called_once_with('dummy_path.csv', new=True)
            mock_statistics.assert_called_once()

class TestAddFitness(unittest.TestCase):

    def setUp(self):
        self.rundir_max = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self.rundir_max.name, "run1", "gene1", "saved_populations"), exist_ok=True)
        os.makedirs(os.path.join(self.rundir_max.name, "run1", "gene2", "saved_populations"), exist_ok=True)
        gene1_pareto = [['seq1', 0.5, 0], ['seq2', 0.8, 1], ['seq3', 0.9, 2]]
        gene2_pareto = [['seq1', 0.5, 0], ['seq2', 0.8, 1], ['seq3', 0.9, 3]]
        with open(os.path.join(self.rundir_max.name, "run1", "gene1", "saved_populations", "gene1_pareto_front.json"), 'w') as f:
            json.dump(gene1_pareto, f)
        with open(os.path.join(self.rundir_max.name, "run1", "gene2", "saved_populations", "pareto_front.json"), 'w') as f:
            json.dump(gene2_pareto, f)
        self.rundir_min = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self.rundir_min.name, "run1", "gene1", "saved_populations"), exist_ok=True)
        os.makedirs(os.path.join(self.rundir_min.name, "run1", "gene2", "saved_populations"), exist_ok=True)
        gene1_pareto = [['seq1', 0.5, 0], ['seq2', 0.2, 1], ['seq3', 0.1, 2]]
        gene2_pareto = [['seq1', 0.5, 0], ['seq2', 0.2, 1], ['seq3', 0.1, 3]]
        with open(os.path.join(self.rundir_min.name, "run1", "gene1", "saved_populations", "gene1_pareto_front.json"), 'w') as f:
            json.dump(gene1_pareto, f)
        with open(os.path.join(self.rundir_min.name, "run1", "gene2", "saved_populations", "pareto_front.json"), 'w') as f:
            json.dump(gene2_pareto, f)
    
    def tearDown(self):
        shutil.rmtree(self.rundir_max.name)
        shutil.rmtree(self.rundir_min.name)
    
    def test_add_fitness_max_success(self):
        df = pd.DataFrame({
            'sequence': ['run1__gene1_mutations_0', 'run1__gene1_mutations_1', 'run1__gene1_mutations_2', 'run1__gene2_mutations_0', 'run1__gene2_mutations_1', 'run1__gene2_mutations_3'],
            'number_mutations': [0, 1, 2, 0, 1, 3]
        })
        
        add_fitness(df, self.rundir_max.name)
        
        self.assertIn('fitness', df.columns)
        self.assertIn('diff_fitness', df.columns)
        self.assertIn('diff_fitness_normalized', df.columns)
        expected_fitness = pd.Series([0.5, 0.8, 0.9, 0.5, 0.8, 0.9], name='fitness', index=df.index)
        expected_diff_fitness = pd.Series([np.nan, 0.3, 0.1, np.nan, 0.3, 0.1], name='diff_fitness', index=df.index)
        expected_diff_fitness_normalized = pd.Series([np.nan, 0.3, 0.1, np.nan, 0.3, 0.05], name='diff_fitness_normalized', index=df.index)
        assert_series_equal(df['fitness'], expected_fitness)
        assert_series_equal(df['diff_fitness'], expected_diff_fitness)
        assert_series_equal(df['diff_fitness_normalized'], expected_diff_fitness_normalized)


    def test_add_fitness_min_success(self):
        df = pd.DataFrame({
            'sequence': ['run1__gene1_mutations_0', 'run1__gene1_mutations_1', 'run1__gene1_mutations_2', 'run1__gene2_mutations_0', 'run1__gene2_mutations_1', 'run1__gene2_mutations_3'],
            'number_mutations': [0, 1, 2, 0, 1, 3]
        })
        
        add_fitness(df, self.rundir_min.name)
        
        self.assertIn('fitness', df.columns)
        self.assertIn('diff_fitness', df.columns)
        self.assertIn('diff_fitness_normalized', df.columns)
        expected_fitness = pd.Series([0.5, 0.2, 0.1, 0.5, 0.2, 0.1], name='fitness', index=df.index)
        expected_diff_fitness = pd.Series([np.nan, -0.3, -0.1, np.nan, -0.3, -0.1], name='diff_fitness', index=df.index)
        expected_diff_fitness_normalized = pd.Series([np.nan, -0.3, -0.1, np.nan, -0.3, -0.05], name='diff_fitness_normalized', index=df.index)
        assert_series_equal(df['fitness'], expected_fitness)
        assert_series_equal(df['diff_fitness'], expected_diff_fitness)
        assert_series_equal(df['diff_fitness_normalized'], expected_diff_fitness_normalized)

    def test_add_fitness_file_not_found(self):
        df = pd.DataFrame({
            'sequence': ['run1__gene3_mutations_0', 'run1__gene3_mutations_1'],
            'number_mutations': [0, 1]
        })
        
        add_fitness(df, self.rundir_max.name)
        self.assertIn('fitness', df.columns)
        self.assertIn('diff_fitness', df.columns)
        self.assertIn('diff_fitness_normalized', df.columns)
        expected_fitness = pd.Series([np.nan, np.nan], name='fitness', index=df.index)
        expected_diff_fitness = pd.Series([np.nan, np.nan], name='diff_fitness', index=df.index)
        expected_diff_fitness_normalized = pd.Series([np.nan, np.nan], name='diff_fitness_normalized', index=df.index)
        assert_series_equal(df['fitness'], expected_fitness)
        assert_series_equal(df['diff_fitness'], expected_diff_fitness)
        assert_series_equal(df['diff_fitness_normalized'], expected_diff_fitness_normalized)
        


class TestFindDiffTf(unittest.TestCase):
    
    def test_find_diff_tf_single_change_pos(self):
        epm_cols = ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5']
        diff_epms = pd.Series({'epm_ara_msr_max_corrected_p1m10': 1, 'epm_crp_p2m5': 0})
        
        tf, counter = find_diff_tf(epm_cols, diff_epms)
        
        self.assertEqual(tf, 'epm_ara_msr_max_corrected_p1m10')
        self.assertEqual(counter, 1)
    
    def test_find_diff_tf_single_change_neg(self):
        epm_cols = ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5']
        diff_epms = pd.Series({'epm_ara_msr_max_corrected_p1m10': 0, 'epm_crp_p2m5': -1})
        
        tf, counter = find_diff_tf(epm_cols, diff_epms)
        
        self.assertEqual(tf, 'epm_crp_p2m5')
        self.assertEqual(counter, -1)

    def test_find_diff_tf_multi_change(self):
        epm_cols = ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5', 'epm_asd_p1m6']
        diff_epms = pd.Series({'epm_ara_msr_max_corrected_p1m10': 1, 'epm_crp_p2m5': -1, "epm_asd_p1m6": 1})
        
        tf, counter = find_diff_tf(epm_cols, diff_epms)
        
        self.assertIsNone(tf)
        self.assertEqual(counter, 1)

    def test_find_diff_tf_multiple_changes(self):
        epm_cols = ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5']
        diff_epms = pd.Series({'epm_ara_msr_max_corrected_p1m10': 1, 'epm_crp_p2m5': 1})
        
        tf, counter = find_diff_tf(epm_cols, diff_epms)
        
        self.assertIsNone(tf)
        self.assertEqual(counter, 2)

    def test_find_diff_tf_multiple_changes_1(self):
        epm_cols = ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5']
        diff_epms = pd.Series({'epm_ara_msr_max_corrected_p1m10': 0, 'epm_crp_p2m5': 2})
        
        tf, counter = find_diff_tf(epm_cols, diff_epms)
        
        self.assertIsNone(tf)
        self.assertEqual(counter, 2)

    def test_find_diff_tf_no_changes(self):
        epm_cols = ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5']
        diff_epms = pd.Series({'epm_ara_msr_max_corrected_p1m10': 0, 'epm_crp_p2m5': 0})

        tf, counter = find_diff_tf(epm_cols, diff_epms)
        
        self.assertIsNone(tf)
        self.assertEqual(counter, 0)


class TestAddDiffTf(unittest.TestCase):
    
    def test_add_diff_tf(self):
        df = pd.DataFrame({
            'sequence': ['gene1_mutations_0', 'gene1_mutations_1', 'gene1_mutations_2', 'gene2_mutations_0', 'gene2_mutations_1', 'gene2_mutations_2'],
            'number_mutations': [0, 1, 2, 0, 1, 2],
            'epm_ara_msr_max_corrected_p1m10': [1, 2, 1, 1, 1, 1],
            'epm_crp_p2m5': [1, 1, 1, 1, 1, 0]
        })
        
        result = add_diff_tf(df)
        
        self.assertIn('diff_tf', result.columns)
        expected_diff_tf = pd.Series([pd.NA, 'epm_ara_msr_max_corrected_p1m10', "epm_ara_msr_max_corrected_p1m10", pd.NA, pd.NA, 'epm_crp_p2m5'], name='diff_tf', index=df.index)
        expected_introduced_removed = pd.Series([pd.NA, "+", "-", pd.NA, pd.NA, '-'], name='introduced_removed', index=df.index)
        assert_series_equal(result['diff_tf'], expected_diff_tf)
        assert_series_equal(result['introduced_removed'], expected_introduced_removed)


class TestAddOriginalSequenceAndNumberMutations(unittest.TestCase):
    
    def test_add_original_sequence_and_number_mutations(self):
        df = pd.DataFrame({
            'sequence': ['gene1_mutations_0:0_seed', 'gene1_mutations_5:5_seed']
        })
        
        add_original_sequence_and_number_mutations(df)
        
        self.assertIn('original_sequence', df.columns)
        self.assertIn('number_mutations', df.columns)
        self.assertEqual(df['original_sequence'].tolist(), ['gene1', 'gene1'])
        self.assertEqual(df['number_mutations'].tolist(), [0, 5])


class TestAnalyzeMutationsEffect(unittest.TestCase):
    
    def setUp(self):
        self.sample_mapping = pd.DataFrame({
            'epm': ['epm_dummy_p1m10', 'epm_dummy_p1m11'],
            'tf_name': ['DMY10', pd.NA]
        })

        self.rundir_max = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self.rundir_max.name, "run1", "gene1", "saved_populations"), exist_ok=True)
        os.makedirs(os.path.join(self.rundir_max.name, "run1", "gene2", "saved_populations"), exist_ok=True)
        gene1_pareto = [['seq1', 0.1, 0], ['seq2', 0.2, 1], ['seq3', 0.3, 2], ['seq3', 0.4, 3], ['seq3', 0.5, 4], ['seq3', 0.6, 5], ['seq3', 0.7, 6], ['seq3', 0.8, 7], ['seq3', 0.90, 8], ['seq3', 0.902, 9], ['seq3', 0.903, 10], ['seq3', 0.904, 11], ['seq3', 0.905, 12]]
        gene2_pareto = [['seq1', 0.01, 0], ['seq2', 0.02, 1], ['seq3', 0.13, 2], ['seq3', 0.14, 3], ['seq3', 0.25, 4], ['seq3', 0.26, 5], ['seq3', 0.37, 6], ['seq3', 0.38, 7], ['seq3', 0.39, 8], ['seq3', 0.50, 9], ['seq3', 0.61, 10], ['seq3', 0.62, 11], ['seq3', 0.73, 12], ['seq3', 0.74, 13], ['seq3', 0.85, 14], ['seq3', 0.86, 15], ['seq3', 0.97, 16]]
        with open(os.path.join(self.rundir_max.name, "run1", "gene1", "saved_populations", "gene1_pareto_front.json"), 'w') as f:
            json.dump(gene1_pareto, f)
        with open(os.path.join(self.rundir_max.name, "run1", "gene2", "saved_populations", "pareto_front.json"), 'w') as f:
            json.dump(gene2_pareto, f)
    
    def tearDown(self):
        shutil.rmtree(self.rundir_max.name)
    
    @patch('analysis.analyze_mapping.get_epm_count_table')
    def test_analyze_mutations_effect(self, mock_get_epm_count_table):
        # Setup mock return values
        mock_epm_counts = pd.DataFrame({
            'sequence': ['run1__gene1_mutations_0', 'run1__gene1_mutations_1', 'run1__gene1_mutations_2', 'run1__gene1_mutations_3', 'run1__gene1_mutations_4', 'run1__gene1_mutations_5', 'run1__gene1_mutations_6', 'run1__gene1_mutations_7', 'run1__gene1_mutations_8', 'run1__gene1_mutations_9', 'run1__gene1_mutations_10', 'run1__gene1_mutations_11', 'run1__gene1_mutations_12', 'run1__gene2_mutations_0', 'run1__gene2_mutations_1', 'run1__gene2_mutations_2', 'run1__gene2_mutations_3', 'run1__gene2_mutations_4', 'run1__gene2_mutations_5', 'run1__gene2_mutations_6', 'run1__gene2_mutations_7', 'run1__gene2_mutations_8', 'run1__gene2_mutations_9', 'run1__gene2_mutations_10', 'run1__gene2_mutations_11', 'run1__gene2_mutations_12', 'run1__gene2_mutations_13', 'run1__gene2_mutations_14', 'run1__gene2_mutations_15', 'run1__gene2_mutations_16'],
            'epm_dummy_p1m10': [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'epm_dummy_p1m11': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 3, 2, 1, 2, 1, 2, 1, 2, 1],
        })
        mock_get_epm_count_table.return_value = mock_epm_counts
        
        with tempfile.TemporaryDirectory() as tmp_path:
            result = analyze_mutations_effect(
                input_path='dummy_path.csv',
                output_folder=tmp_path,
                mapping=self.sample_mapping,
                data_path=self.rundir_max.name,
            )
            self.assertTrue(os.path.exists(os.path.join(tmp_path, "effect_per_mutation_epm.csv")))

        self.assertIsInstance(result, pd.DataFrame)
        expected_effects = pd.Series([0.11, 0.01, 0.1], index=[2, 1, 0], name="avg_effect")
        assert_series_equal(result['avg_effect'], expected_effects, check_exact=False)
        for pval in result["pval"]:
            self.assertLessEqual(pval, 1)
            self.assertGreaterEqual(pval, 0)


class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        self.sample_old_data = pd.DataFrame({
            'epm': ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5'],
            'tf_name': ['AraC', 'CRP'],
            'loc': ['gene1_mutations_0', 'gene1_mutations_1'],
            'type': ['gene', 'gene']
        })
        
        self.sample_new_data = pd.DataFrame({
            'motif': ['epm_ara_msr_max_corrected_p1m10F_1', 'epm_crp_p2m5F_2'],
            'sequence_id': ['run1__gene1_mutations_0:0_seed', 'run1__gene1_mutations_1:1_seed']
        })
    
    def test_full_pipeline_old_format(self):
        """Test the full pipeline with old format data"""
        with tempfile.TemporaryDirectory() as tmp_path:
            input_file = os.path.join(tmp_path, "input.csv")
            self.sample_old_data.to_csv(input_file, index=False)
            
            with patch('pandas.read_csv', return_value=self.sample_old_data):
                mapping = get_epm_tfbs_mapping_old(input_file)
                
            self.assertGreater(len(mapping), 0)
            self.assertIn('epm', mapping.columns)
            self.assertIn('tf_name', mapping.columns)
    
    def test_full_pipeline_new_format(self):
        """Test the full pipeline with new format data"""
        with tempfile.TemporaryDirectory() as tmp_path:
            input_file = os.path.join(tmp_path, "input.csv")
            self.sample_new_data.to_csv(input_file, index=False)
            
            with patch('pandas.read_csv', return_value=self.sample_new_data):
                mapping = get_epm_tfbs_mapping_new(input_file)
                
            self.assertGreater(len(mapping), 0)
            self.assertIn('epm', mapping.columns)
            self.assertIn('tf_name', mapping.columns)


if __name__ == "__main__":
    unittest.main()
    

