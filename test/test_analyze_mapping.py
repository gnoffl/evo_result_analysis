import unittest
import pandas as pd
import numpy as np
import os
import json
import tempfile
import math
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


class TestCompareInitialAndFinalDistributionsCore(unittest.TestCase):
    
    def setUp(self):
        self.sample_mapping = pd.DataFrame({
            'epm': ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5'],
            'tf_name': ['AraC', 'CRP']
        })
    
    def test_compare_initial_and_final_distributions_core(self):
        epm_counts = pd.DataFrame({
            'sequence': ['gene1_mutations_0', 'gene2_mutations_5'],
            'epm_ara_msr_max_corrected_p1m10': [2, 3],
            'epm_crp_p2m5': [1, 0]
        })
        
        with tempfile.TemporaryDirectory() as tmp_path:
            with patch('analysis.analyze_mapping.statistics_epm_before_after', return_value={'epm_ara_msr_max_corrected_p1m10': 0.05}):
                compare_initial_and_final_distribution(
                    mapping=self.sample_mapping,
                    output_folder=tmp_path,
                    epm_counts=epm_counts
                )
            
            output_file = os.path.join(tmp_path, "initial_vs_final_per_epm_counts.csv")
            self.assertTrue(os.path.exists(output_file))


class TestAddFitness(unittest.TestCase):
    
    def test_add_fitness_success(self):
        df = pd.DataFrame({
            'sequence': ['run1__gene1_mutations_0:0_seed', 'run1__gene1_mutations_1:1_seed'],
            'number_mutations': [0, 1]
        })
        
        with tempfile.TemporaryDirectory() as tmp_path:
            # Create mock pareto front file
            pareto_data = [['seq1', 0.5, 0], ['seq2', 0.8, 1]]
            pareto_dir = os.path.join(tmp_path, "run1", "gene1", "saved_populations")
            os.makedirs(pareto_dir, exist_ok=True)
            pareto_file = os.path.join(pareto_dir, "gene1_pareto_front.json")
            with open(pareto_file, 'w') as f:
                json.dump(pareto_data, f)
            
            add_fitness(df, tmp_path)
            
            self.assertIn('fitness', df.columns)
            self.assertIn('diff_fitness', df.columns)
            self.assertIn('diff_fitness_normalized', df.columns)
    
    def test_add_fitness_file_not_found(self):
        df = pd.DataFrame({
            'sequence': ['run1__gene1_mutations_0:0_seed'],
            'number_mutations': [0]
        })
        
        with tempfile.TemporaryDirectory() as tmp_path:
            with patch('builtins.print') as mock_print:
                add_fitness(df, tmp_path)
            
            mock_print.assert_called()
            self.assertTrue(pd.isna(df.loc[0, 'fitness']))


class TestFindDiffTf(unittest.TestCase):
    
    def test_find_diff_tf_single_change(self):
        epm_cols = ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5']
        diff_epms = pd.Series({'epm_ara_msr_max_corrected_p1m10': 1, 'epm_crp_p2m5': 0})
        
        tf, counter = find_diff_tf(epm_cols, diff_epms)
        
        self.assertEqual(tf, 'epm_ara_msr_max_corrected_p1m10')
        self.assertEqual(counter, 1)
    
    def test_find_diff_tf_multiple_changes(self):
        epm_cols = ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5']
        diff_epms = pd.Series({'epm_ara_msr_max_corrected_p1m10': 1, 'epm_crp_p2m5': 1})
        
        tf, counter = find_diff_tf(epm_cols, diff_epms)
        
        self.assertIsNone(tf)
        self.assertEqual(counter, 2)


class TestAddDiffTf(unittest.TestCase):
    
    def test_add_diff_tf(self):
        df = pd.DataFrame({
            'sequence': ['gene1_mutations_0', 'gene1_mutations_1'],
            'number_mutations': [0, 1],
            'epm_ara_msr_max_corrected_p1m10': [1, 2],
            'epm_crp_p2m5': [1, 1]
        })
        
        result = add_diff_tf(df)
        
        self.assertIn('diff_tf', result.columns)
        self.assertEqual(result.loc[1, 'diff_tf'], 'epm_ara_msr_max_corrected_p1m10')


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
            'epm': ['epm_ara_msr_max_corrected_p1m10', 'epm_crp_p2m5'],
            'tf_name': ['AraC', 'CRP']
        })
    
    @patch('analysis.analyze_mapping.get_epm_count_table')
    @patch('analysis.analyze_mapping.add_fitness')
    @patch('analysis.analyze_mapping.add_diff_tf')
    @patch('pandas.DataFrame.to_csv')
    def test_analyze_mutations_effect(self, mock_to_csv, mock_add_diff_tf, mock_add_fitness, mock_get_epm_count_table):
        # Setup mock return values
        mock_epm_counts = pd.DataFrame({
            'sequence': ['gene1_mutations_0', 'gene1_mutations_1'],
            'epm_ara_msr_max_corrected_p1m10': [1, 2],
            'diff_fitness_normalized': [np.nan, 0.1],
            'epm': ['epm_ara_msr_max_corrected_p1m10', 'epm_ara_msr_max_corrected_p1m10']
        })
        mock_get_epm_count_table.return_value = mock_epm_counts
        mock_add_diff_tf.return_value = mock_epm_counts
        
        with tempfile.TemporaryDirectory() as tmp_path:
            with patch('builtins.print'):
                result = analyze_mutations_effect(
                    'dummy_path.csv',
                    tmp_path,
                    self.sample_mapping,
                    'dummy_data_path'
                )
            
            self.assertIsInstance(result, pd.DataFrame)
            mock_to_csv.assert_called_once()


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
    # unittest.main()
    testObj = TestAddTfNames()
    testObj.setUp()
    testObj.test_add_tf_names()
    print("Tests ran successfully.")

