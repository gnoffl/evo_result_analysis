"""Comprehensive test suite for deepcis_scanner module.

Tests are organized in the same order as functions/classes appear in the module:
1. GeneRunData class
2. find_gene_folders
3. get_full_sequence
4. _get_zero_mutation_entry
5. _get_max_mutation_entry
6. slide_windows
7. load_deepcis_model
8. predict_windows
9. scan_single_gene_folder
10. scan_all_genes
"""

import unittest
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open, ANY
import json
import os

from evolution.sequences import one_hot_decode

from analysis.motives.deepcis_scanner import (
    GeneRunData,
    find_gene_folders,
    find_padding_region,
    get_full_sequence,
    _get_max_mutation_entry,
    slide_windows,
    load_deepcis_model,
    predict_windows,
    scan_single_gene_folder,
    scan_all_genes,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STEP_SIZE,
    DEFAULT_BATCH_SIZE,
    TF_FAMILY_NAMES,
    N_TF_FAMILIES,
)


# ============================================================================
# Mock Model for Testing
# ============================================================================

class MockDeepCISModel:
    """Mock deepCIS model that detects simple motifs instead of running real predictions."""
    
    def predict(self, windows_onehot, batch_size=None, verbose=0):
        """
        Mock prediction that checks for specific motifs.
        
        Args:
            windows_onehot: Array of shape (batch, length, 4) with one-hot encoded sequences
            batch_size: Ignored by mock
            verbose: Ignored by mock
            
        Returns:
            np.ndarray of shape (batch, 2) with motif detection results:
            - Column 0: 1.0 if "CTC" is present, 0.0 otherwise
            - Column 1: 1.0 if "AGA" is present, 0.0 otherwise
        """
        batch_size_actual = windows_onehot.shape[0]
        output = np.zeros((batch_size_actual, 2), dtype=np.float32)
        
        for i in range(batch_size_actual):
            # Decode one-hot to nucleotide string
            seq = one_hot_decode(windows_onehot[i])
            
            # Check for motifs
            output[i, 0] = 1.0 if "CTC" in seq else 0.0
            output[i, 1] = 1.0 if "AGA" in seq else 0.0
        
        return output


# ============================================================================
# GeneRunData Class Tests
# ============================================================================

class TestGeneRunDataStaticMethods(unittest.TestCase):
    """Tests for GeneRunData static and class methods."""

    def test_load_pareto_front_basic(self):
        """Test loading pareto front from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gene_folder = tmpdir
            pareto_dir = os.path.join(gene_folder, "saved_populations")
            os.makedirs(pareto_dir)
            
            pareto_data = [
                ["ACGTACGT", 0.5, 0],
                ["MUTMUTMUT", 0.8, 5],
            ]
            pareto_path = os.path.join(pareto_dir, "pareto_front.json")
            with open(pareto_path, "w") as f:
                json.dump(pareto_data, f)
            
            result = GeneRunData.load_pareto_front(gene_folder)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], ("ACGTACGT", 0.5, 0))
            self.assertEqual(result[1], ("MUTMUTMUT", 0.8, 5))

    def test_load_pareto_front_type_coercion(self):
        """Test that pareto front types are properly coerced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gene_folder = tmpdir
            pareto_dir = os.path.join(gene_folder, "saved_populations")
            os.makedirs(pareto_dir)
            
            # Test with mixed types
            pareto_data = [["seq", "0.9", "10"]]
            pareto_path = os.path.join(pareto_dir, "pareto_front.json")
            with open(pareto_path, "w") as f:
                json.dump(pareto_data, f)
            
            result = GeneRunData.load_pareto_front(gene_folder)
            self.assertEqual(result[0][0], "seq")
            self.assertIsInstance(result[0][1], float)
            self.assertIsInstance(result[0][2], int)
            self.assertEqual(result[0], ("seq", 0.9, 10))

    def test_load_pareto_front_file_not_found(self):
        """Test error handling when pareto front file is missing."""
        with self.assertRaises(FileNotFoundError):
            GeneRunData.load_pareto_front("/nonexistent/folder")

    def test_load_pareto_front_empty(self):
        """Test loading empty pareto front."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gene_folder = tmpdir
            pareto_dir = os.path.join(gene_folder, "saved_populations")
            os.makedirs(pareto_dir)
            
            pareto_path = os.path.join(pareto_dir, "pareto_front.json")
            with open(pareto_path, "w") as f:
                json.dump([], f)
            
            result = GeneRunData.load_pareto_front(gene_folder)
            self.assertEqual(len(result), 0)

    def test_load_gene_params_basic(self):
        """Test loading gene parameters from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            params_file = os.path.join(tmpdir, "parameters.json")
            params = {"mutation_start": 100, "mutation_end": 200, "extra": "data"}
            with open(params_file, "w") as f:
                json.dump(params, f)
            
            result = GeneRunData.load_gene_params(tmpdir)
            self.assertEqual(result["mutation_start"], 100)
            self.assertEqual(result["mutation_end"], 200)
            self.assertEqual(result["extra"], "data")

    def test_load_gene_params_file_not_found(self):
        """Test error handling when parameters file is missing."""
        with self.assertRaises(FileNotFoundError):
            GeneRunData.load_gene_params("/nonexistent/folder")

    def test_load_reference_sequence_full_with_key(self):
        """Test loading reference sequence with explicit 'reference_sequence_full' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gene_folder = tmpdir
            with open(os.path.join(gene_folder, "reference_sequence.fa"), "w") as f:
                target_seq = "AAAACCCC"
                f.write(f">distraction\nAAAAAAAA\n>reference_sequence_full\n{target_seq}\n")
            
            loaded_ref = GeneRunData.load_reference_sequence_full(gene_folder)
            self.assertEqual(loaded_ref, target_seq)

    def test_load_reference_sequence_full_fallback_to_first_key(self):
        """Test loading reference sequence when key doesn't exist (fallback to first key)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gene_folder = tmpdir
            with open(os.path.join(gene_folder, "reference_sequence.fa"), "w") as f:
                target_seq = "AAAACCCC"
                f.write(f">target\n{target_seq}\n>distraction\nAAAAAAAA\n")
            
            loaded_ref = GeneRunData.load_reference_sequence_full(gene_folder)
            self.assertEqual(loaded_ref, target_seq)

    def test_load_reference_sequence_full_import_error(self):
        """Test error handling when pyfaidx is not available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gene_folder = tmpdir
            
            with self.assertRaises(Exception):
                loaded_ref = GeneRunData.load_reference_sequence_full(gene_folder)

    def test_load_gene_run_data_full_integration(self):
        """Test the complete load_gene_run_data class method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gene_folder = tmpdir
            
            # Create pareto front
            pareto_dir = os.path.join(gene_folder, "saved_populations")
            os.makedirs(pareto_dir)
            pareto_path = os.path.join(pareto_dir, "pareto_front.json")
            with open(pareto_path, "w") as f:
                json.dump([["REFSEQ", 0.5, 0], ["MUTSEQ", 0.9, 5]], f)
            
            # Create parameters
            params_path = os.path.join(gene_folder, "parameters.json")
            with open(params_path, "w") as f:
                json.dump({"mutation_start": 100, "mutation_end": 150}, f)
            
            with open(os.path.join(gene_folder, "reference_sequence.fa"), "w") as f:
                f.write(">reference_sequence_full\n" + "A" * 1000 + "\n")
            
            result = GeneRunData.load_gene_run_data(gene_folder)
            
            self.assertEqual(result.gene_name, os.path.basename(gene_folder))
            self.assertEqual(result.gene_folder, gene_folder)
            self.assertEqual(len(result.pareto_front), 2)
            self.assertEqual(result.mutation_start, 100)
            self.assertEqual(result.mutation_end, 150)
            self.assertEqual(result.reference_sequence_full, "A" * 1000)

    def test_gene_run_data_dataclass_initialization(self):
        """Test direct GeneRunData initialization."""
        pareto_front = [("ACGT", 0.5, 0), ("MUTSEQ", 0.9, 5)]
        gene_data = GeneRunData(
            gene_name="test_gene",
            gene_folder="/path/to/gene",
            pareto_front=pareto_front,
            reference_sequence_full="A" * 1000,
            mutation_start=100,
            mutation_end=150
        )
        
        self.assertEqual(gene_data.gene_name, "test_gene")
        self.assertEqual(len(gene_data.pareto_front), 2)
        self.assertEqual(gene_data.mutation_start, 100)
        self.assertEqual(gene_data.mutation_end, 150)
        self.assertEqual(gene_data.reference_sequence_full, "A" * 1000)
        self.assertEqual(gene_data.pareto_front[0], ("ACGT", 0.5, 0))
        self.assertEqual(gene_data.gene_folder, "/path/to/gene")


# ============================================================================
# find_gene_folders Tests
# ============================================================================

class TestFindGeneFolders(unittest.TestCase):
    """Tests for the find_gene_folders function."""

    def test_find_gene_folders_valid(self):
        """Test discovery of valid gene folders with required files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folders = ["gene1", "gene2", "not_a_folder", "gene3"]
            for folder in folders:
                os.makedirs(os.path.join(tmpdir, folder))
                if folder in ["gene1", "gene3"]:
                    os.makedirs(os.path.join(tmpdir, folder, "saved_populations"))
                    with open(os.path.join(tmpdir, folder, "parameters.json"), "w") as f:
                        json.dump({"mutation_start": 100, "mutation_end": 200}, f)
                    with open(os.path.join(tmpdir, folder, "saved_populations", "pareto_front.json"), "w") as f:
                        json.dump([["SEQ", 0.5, 0]], f)
                    with open(os.path.join(tmpdir, folder, "reference_sequence.fa"), "w") as f:
                        f.write(">reference_sequence_full\n" + "A" * 1000 + "\n")
                with open(os.path.join(tmpdir, "file.txt"), "w") as f:
                    f.write("test")

            result = find_gene_folders(tmpdir)
        
        self.assertEqual(len(result), 2)
        self.assertTrue(any("gene1" in p for p in result))
        self.assertTrue(any("gene3" in p for p in result))

    def test_find_gene_folders_no_valid_folders(self):
        """Test when no valid gene folders exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "file.txt"), "w") as f:
                f.write("test")
            os.makedirs(os.path.join(tmpdir, "empty_dir"))

            result = find_gene_folders(tmpdir)
        self.assertEqual(len(result), 0)

    def test_find_gene_folders_missing_pareto_front(self):
        """Test that folders without pareto_front.json are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "gene1", "saved_populations"))
            with open(os.path.join(tmpdir, "gene1", "parameters.json"), "w") as f:
                json.dump({"mutation_start": 100, "mutation_end": 200}, f)
            with open(os.path.join(tmpdir, "gene1", "saved_populations", "pareto_front.json"), "w") as f:
                json.dump([["SEQ", 0.5, 0]], f)
            with open(os.path.join(tmpdir, "gene1", "reference_sequence.fa"), "w") as f:
                f.write(">reference_sequence_full\n" + "A" * 1000 + "\n")

            os.makedirs(os.path.join(tmpdir, "gene2", "saved_populations"))
            with open(os.path.join(tmpdir, "gene2", "parameters.json"), "w") as f:
                json.dump({"mutation_start": 100, "mutation_end": 200}, f)
            with open(os.path.join(tmpdir, "gene2", "reference_sequence.fa"), "w") as f:
                f.write(">reference_sequence_full\n" + "A" * 1000 + "\n")

            result = find_gene_folders(tmpdir)
        
        self.assertEqual(len(result), 1)
        self.assertTrue("gene1" in result[0])

    def test_find_gene_folders_missing_parameters(self):
        """Test that folders without parameters.json are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "gene1", "saved_populations"))
            with open(os.path.join(tmpdir, "gene1", "parameters.json"), "w") as f:
                json.dump({"mutation_start": 100, "mutation_end": 200}, f)
            with open(os.path.join(tmpdir, "gene1", "saved_populations", "pareto_front.json"), "w") as f:
                json.dump([["SEQ", 0.5, 0]], f)
            with open(os.path.join(tmpdir, "gene1", "reference_sequence.fa"), "w") as f:
                f.write(">reference_sequence_full\n" + "A" * 1000 + "\n")

            os.makedirs(os.path.join(tmpdir, "gene2", "saved_populations"))
            with open(os.path.join(tmpdir, "gene2", "saved_populations", "pareto_front.json"), "w") as f:
                json.dump([["SEQ", 0.5, 0]], f)
            with open(os.path.join(tmpdir, "gene2", "reference_sequence.fa"), "w") as f:
                f.write(">reference_sequence_full\n" + "A" * 1000 + "\n")

            result = find_gene_folders(tmpdir)

        self.assertEqual(len(result), 1)
        self.assertTrue("gene1" in result[0])


# ============================================================================
# get_full_sequence Tests
# ============================================================================

class TestGetFullSequence(unittest.TestCase):
    """Tests for the get_full_sequence function."""

    def test_get_full_sequence_basic(self):
        """Test basic sequence replacement in the middle."""
        ref_full = "AAAAAGGGGGTTTTTCCCCC"
        mutable = "XXXX"
        result = get_full_sequence(
            mutable_sequence=mutable,
            reference_sequence_full=ref_full,
            mutation_start=5,
            mutation_end=9
        )
        expected = "AAAAAXXXXGTTTTTCCCCC"
        self.assertEqual(result, expected)

    def test_get_full_sequence_at_start(self):
        """Test replacing mutable region at sequence start."""
        ref_full = "AAAAAGGGGGTTTTTCCCCC"
        mutable = "XX"
        result = get_full_sequence(
            mutable_sequence=mutable,
            reference_sequence_full=ref_full,
            mutation_start=0,
            mutation_end=2
        )
        expected = "XXAAAGGGGGTTTTTCCCCC"
        self.assertEqual(result, expected)

    def test_get_full_sequence_at_end(self):
        """Test replacing mutable region at sequence end."""
        ref_full = "AAAAAGGGGGTTTTTCCCCC"
        mutable = "YYY"
        result = get_full_sequence(
            mutable_sequence=mutable,
            reference_sequence_full=ref_full,
            mutation_start=17,
            mutation_end=20
        )
        expected = "AAAAAGGGGGTTTTTCCYYY"
        self.assertEqual(result, expected)

    def test_get_full_sequence_entire_sequence(self):
        """Test replacing entire sequence."""
        ref_full = "AAAAAGGGGGTTTTTCCCCC"
        mutable = "TTTTTTTTTTTTTTTTTTTT"
        result = get_full_sequence(
            mutable_sequence=mutable,
            reference_sequence_full=ref_full,
            mutation_start=0,
            mutation_end=20
        )
        self.assertEqual(result, mutable)

    def test_get_full_sequence_single_nucleotide(self):
        """Test replacing single nucleotide."""
        ref_full = "ACGTACGTACGT"
        mutable = "T"
        result = get_full_sequence(
            mutable_sequence=mutable,
            reference_sequence_full=ref_full,
            mutation_start=5,
            mutation_end=6
        )
        expected = "ACGTATGTACGT"
        self.assertEqual(result, expected)

    def test_get_full_sequence_preserves_length(self):
        """Test that output length equals reference length."""
        ref_full = "ACGTACGTACGT" * 10
        mutable = "T" * 50
        result = get_full_sequence(
            mutable_sequence=mutable,
            reference_sequence_full=ref_full,
            mutation_start=30,
            mutation_end=80
        )
        self.assertEqual(len(result), len(ref_full))

# ============================================================================
# _get_max_mutation_entry Tests
# ============================================================================

class TestGetMaxMutationEntry(unittest.TestCase):
    """Tests for the _get_max_mutation_entry function."""

    def test_get_max_mutation_entry_basic(self):
        """Test finding entry with most mutations."""
        pareto_front = [
            ("SEQSEQSEQ", 0.5, 0),
            ("MUTMUTMUT1", 0.8, 3),
            ("MUTMUTMUT2", 0.9, 5),
            ("MUTMUTMUT3", 0.7, 2),
        ]
        result = _get_max_mutation_entry(pareto_front)
        self.assertEqual(result[2], 5)
        self.assertEqual(result[1], 0.9)
        self.assertEqual(result[0], "MUTMUTMUT2")

    def test_get_max_mutation_entry_single_entry(self):
        """Test with single entry."""
        pareto_front = [("ONLYSEQ", 0.7, 3)]
        result = _get_max_mutation_entry(pareto_front)
        self.assertEqual(result, ("ONLYSEQ", 0.7, 3))

    def test_get_max_mutation_entry_all_same_count(self):
        """Test when all entries have same mutation count (returns first)."""
        pareto_front = [
            ("SEQ1", 0.5, 5),
            ("SEQ2", 0.8, 5),
            ("SEQ3", 0.9, 5),
        ]
        result = _get_max_mutation_entry(pareto_front)
        self.assertEqual(result[2], 5)

    def test_get_max_mutation_entry_empty_pareto_front(self):
        """Test with empty pareto front raises ValueError."""
        with self.assertRaises(ValueError):
            _get_max_mutation_entry([])


class TestFindPaddingRegion(unittest.TestCase):
    """Tests for the find_padding_region function."""

    def test_find_padding_region_basic(self):
        """Test finding a single padding region."""
        seq = "AAAANNNNNTTTT"
        result = find_padding_region(seq)
        self.assertEqual(result, (4, 9))

    def test_find_padding_region_multiple_regions(self):
        """Test finding longest padding region among multiple."""
        seq = "NNNAAAANNNNTTTNNNNN"
        result = find_padding_region(seq)
        self.assertEqual(result, (14, 19))

    def test_find_padding_single_n(self):
        """Test finding longest padding region among multiple."""
        seq = "AAANAAA"
        result = find_padding_region(seq)
        self.assertEqual(result, (3, 4))

    def test_find_padding_region_no_padding(self):
        """Test when no padding is present."""
        seq = "ACGTACGT"
        result = find_padding_region(seq)
        self.assertEqual(result, (-1, -1))

    def test_find_padding_region_all_padding(self):
        """Test when entire sequence is padding."""
        seq = "NNNNNNNN"
        result = find_padding_region(seq)
        self.assertEqual(result, (0, 8))


# ============================================================================
# slide_windows Tests
# ============================================================================

class TestSlideWindows(unittest.TestCase):
    """Tests for the slide_windows function."""

    def test_slide_windows_basic(self):
        """Test basic window generation with standard parameters."""
        seq = "AANNAAA"
        windows = slide_windows(
            sequence=seq,
            window_size=3,
            step=2,
            padding_start=2,
            padding_end=4
        )
        
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[0][1], 3)
        self.assertEqual(windows[0][2], "AAN")
        self.assertTrue(windows[0][3])
        self.assertEqual(windows[2][0], 4)
        self.assertEqual(windows[2][1], 7)
        self.assertEqual(windows[2][2], "AAA")
        self.assertFalse(windows[2][3])

    def test_slide_windows_additional_window_at_end(self):
        """Test basic window generation with standard parameters."""
        seq = "AAANNAAA"
        windows = slide_windows(
            sequence=seq,
            window_size=3,
            step=2,
        )
        
        self.assertEqual(len(windows), 4)

        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[0][1], 3)
        self.assertEqual(windows[0][2], "AAA")
        self.assertFalse(windows[0][3])

        self.assertEqual(windows[2][0], 4)
        self.assertEqual(windows[2][1], 7)
        self.assertEqual(windows[2][2], "NAA")
        self.assertTrue(windows[2][3])

        self.assertEqual(windows[3][0], 5)
        self.assertEqual(windows[3][1], 8)
        self.assertEqual(windows[3][2], "AAA")
        self.assertFalse(windows[3][3])

    def test_slide_windows_no_padding(self):
        """Test window generation without explicit padding."""
        seq = "AACCGGTT"
        windows = slide_windows(
            sequence=seq,
            window_size=3,
            step=2,
        )
        self.assertEqual(len(windows), 4)
        self.assertTrue(all(not w[3] for w in windows))  # No windows should contain padding

    def test_slide_windows_exact_fit(self):
        """Test when sequence length equals window size."""
        seq = "A" * 250
        windows = slide_windows(sequence=seq, window_size=250, step=50)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[0][1], 250)

    def test_slide_windows_small_sequence(self):
        """Test with sequence smaller than window size (no windows)."""
        seq = "A" * 100
        windows = slide_windows(sequence=seq, window_size=250, step=50)
        self.assertEqual(len(windows), 0)

    def test_slide_windows_default_window_size(self):
        """Test using default window size."""
        seq = "A" * 1000
        windows = slide_windows(seq, step=50)
        
        # Should use DEFAULT_WINDOW_SIZE
        for start, end, subseq, _ in windows:
            self.assertEqual(end - start, DEFAULT_WINDOW_SIZE)
            self.assertEqual(len(subseq), DEFAULT_WINDOW_SIZE)

    def test_slide_windows_all_tuples_have_correct_format(self):
        """Test that all window tuples have correct format and types."""
        seq = "ACGT" * 100
        windows = slide_windows(seq, window_size=50, step=20)
        
        for window in windows:
            self.assertEqual(len(window), 4)
            start, end, subseq, contains_padding = window
            self.assertIsInstance(start, int)
            self.assertIsInstance(end, int)
            self.assertIsInstance(subseq, str)
            self.assertIsInstance(contains_padding, bool)
            self.assertEqual(end - start, 50)
            self.assertEqual(len(subseq), 50)


# ============================================================================
# load_deepcis_model Tests
# ============================================================================

class TestLoadDeepCISModel(unittest.TestCase):
    """Tests for the load_deepcis_model function."""

    def test_load_deepcis_model_returns_model_object(self):
        """Test that returned object is a model."""
        model = load_deepcis_model("models/deepcis/deepCIS_model_chrom_1_model.h5")
        self.assertTrue(hasattr(model, "predict"))

    def test_load_deepcis_model_raises(self):
        """Test that returned object is a model."""
        with self.assertRaises(Exception):
            load_deepcis_model("non_existent_folder/non_existent_model.h6")


# ============================================================================
# predict_windows Tests
# ============================================================================

class TestPredictWindows(unittest.TestCase):
    """Tests for the predict_windows function."""

    def test_predict_windows_shape(self):
        """Test that predict_windows returns correct output shape."""
        model = load_deepcis_model("models/deepcis/deepCIS_model_chrom_1_model.h5")
        
        windows = np.random.randn(10, 250, 4)
        result = predict_windows(model, windows, batch_size=5)
        
        self.assertEqual(result.shape, (10, 46))
        self.assertEqual(result.dtype, np.float32)

    def test_predict_windows_batch_size_parameter(self):
        """Test that batch_size is passed to model.predict."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros((5, 46), dtype=np.float32)
        
        windows = np.random.randn(5, 250, 4)
        predict_windows(mock_model, windows, batch_size=2)
        
        # Check that batch_size was passed
        call_kwargs = mock_model.predict.call_args[1]
        self.assertEqual(call_kwargs["batch_size"], 2)

    def test_predict_windows_single_window(self):
        """Test prediction on a single window."""
        model = load_deepcis_model("models/deepcis/deepCIS_model_chrom_1_model.h5")
        
        windows = np.random.randn(1, 250, 4)
        result = predict_windows(model, windows)
        
        self.assertEqual(result.shape, (1, 46))

    def test_predict_windows_many_windows(self):
        """Test prediction on many windows."""
        model = load_deepcis_model("models/deepcis/deepCIS_model_chrom_1_model.h5")
        n_windows = 1000
        
        windows = np.random.randn(n_windows, 250, 4)
        result = predict_windows(model, windows, batch_size=32)
        
        self.assertEqual(result.shape, (n_windows, 46))

# ============================================================================
# scan_single_gene_folder Tests
# ============================================================================

class TestScanSingleGeneFolder(unittest.TestCase):
    """Tests for the scan_single_gene_folder function."""

    def _create_mock_gene_data(self):
        """Helper to create mock GeneRunData."""
        full_seq = "A" * 10 + 5 * "N" + "G" * 10
        return GeneRunData(
            gene_name="test_gene",
            gene_folder="/mock/folder",
            pareto_front=[
                ("A" * 10, 0.5, 0),
                ("AACTCAAAAA", 0.9, 3)
            ],
            reference_sequence_full=full_seq,
            mutation_start=0,
            mutation_end=10
        )

    @patch("analysis.motives.deepcis_scanner.N_TF_FAMILIES", 2)
    @patch("analysis.motives.deepcis_scanner.TF_FAMILY_NAMES", ["CTC", "AGA"])
    def test_scan_single_gene_folder_basic(self):
        """Test basic scanning of a single gene folder."""
        model = MockDeepCISModel()
        gene_data = self._create_mock_gene_data()
        
        df = scan_single_gene_folder(
            model=model,
            gene_data=gene_data,
            window_size=10,
            step=5
        )
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("gene", df.columns)
        self.assertIn("sequence_type", df.columns)
        self.assertIn("window_start", df.columns)
        self.assertIn("window_end", df.columns)
        self.assertIn("contains_padding", df.columns)
        self.assertIn("CTC", df.columns)
        self.assertIn("AGA", df.columns)
        self.assertEqual(len(df.columns), 5 + 2)  # 6 metadata columns + 2 TF columns

    @patch("analysis.motives.deepcis_scanner.N_TF_FAMILIES", 2)
    @patch("analysis.motives.deepcis_scanner.TF_FAMILY_NAMES", ["CTC", "AGA"])
    def test_scan_single_gene_folder_sequence_types(self):
        """Test that both reference and max_mutated sequences are scanned."""
        model = MockDeepCISModel()
        gene_data = self._create_mock_gene_data()
        
        df = scan_single_gene_folder(
            model=model,
            gene_data=gene_data,
            window_size=25,
            step=10
        )
        
        sequence_types = set(df["sequence_type"].unique())
        self.assertEqual(sequence_types, {"reference", "max_mutated"})

    @patch("analysis.motives.deepcis_scanner.N_TF_FAMILIES", 2)
    @patch("analysis.motives.deepcis_scanner.TF_FAMILY_NAMES", ["CTC", "AGA"])
    def test_scan_single_gene_folder_full_df(self):
        """Test that all TF columns are present in output."""
        
        model = MockDeepCISModel()
        gene_data = self._create_mock_gene_data()
        
        df = scan_single_gene_folder(
            model=model,
            gene_data=gene_data,
            window_size=10,
            step=7
        )

        expected = {
            "gene": ["test_gene"] * 8,
            "sequence_type": ["reference"] * 4 + ["max_mutated"] * 4,
            "window_start": [0, 7, 14, 15] * 2,
            "window_end": [10, 17, 24, 25] * 2,
            "contains_padding": [False, True, True, False] * 2,
            "CTC": [0.0] * 4 + [1.0] + [0.0] * 3,
            "AGA": [0.0] * 8
        }

        self.assertTrue(df.equals(pd.DataFrame(expected)))

# ============================================================================
# scan_all_genes Tests
# ============================================================================

class TestScanAllGenes(unittest.TestCase):
    """Tests for the scan_all_genes function."""

    def _setup_gene_folders(self, tmpdir):
        """
        Create actual gene folders with all required files.
        Returns a list of gene folder paths.
        
        Gene 1: Contains "CTC" motif in mutated sequence only
        Gene 2: Contains "AGA" motif in mutated sequence only
        """
        gene_folders = []
        
        # Gene 1 setup
        gene1_folder = os.path.join(tmpdir, "test_gene_1")
        os.makedirs(os.path.join(gene1_folder, "saved_populations"))
        
        # Gene 1 parameters
        with open(os.path.join(gene1_folder, "parameters.json"), "w") as f:
            json.dump({"mutation_start": 0, "mutation_end": 10}, f)
        
        # Gene 1 pareto front
        with open(os.path.join(gene1_folder, "saved_populations", "pareto_front.json"), "w") as f:
            json.dump([
                ["A" * 10, 0.5, 0],
                ["AACTCAAAAA", 0.9, 3]  # Contains "CTC" at positions 2-4
            ], f)
        
        # Gene 1 reference sequence
        gene1_seq = "A" * 10 + "N" * 5 + "G" * 10
        with open(os.path.join(gene1_folder, "reference_sequence.fa"), "w") as f:
            f.write(">reference_sequence_full\n" + gene1_seq + "\n")
        
        gene_folders.append(gene1_folder)
        
        # Gene 2 setup
        gene2_folder = os.path.join(tmpdir, "test_gene_2")
        os.makedirs(os.path.join(gene2_folder, "saved_populations"))
        
        # Gene 2 parameters
        with open(os.path.join(gene2_folder, "parameters.json"), "w") as f:
            json.dump({"mutation_start": 0, "mutation_end": 10}, f)
        
        # Gene 2 pareto front
        with open(os.path.join(gene2_folder, "saved_populations", "pareto_front.json"), "w") as f:
            json.dump([
                ["G" * 10, 0.5, 0],
                ["GGGGAGACGG", 0.9, 3]  # Contains "AGA" at positions 4-6
            ], f)
        
        # Gene 2 reference sequence
        gene2_seq = "G" * 10 + "N" * 5 + "A" * 10
        with open(os.path.join(gene2_folder, "reference_sequence.fa"), "w") as f:
            f.write(">reference_sequence_full\n" + gene2_seq + "\n")
        
        gene_folders.append(gene2_folder)


    @patch("analysis.motives.deepcis_scanner.N_TF_FAMILIES", 2)
    @patch("analysis.motives.deepcis_scanner.TF_FAMILY_NAMES", ["CTC", "AGA"])
    @patch("analysis.motives.deepcis_scanner.load_deepcis_model")
    def test_scan_all_genes_comprehensive(self, mock_load_model):
        """
        Comprehensive test for scan_all_genes with two genes:
        - Gene 1: Contains "CTC" motif in mutated sequence only
        - Gene 2: Contains "AGA" motif in mutated sequence only
        Both use simple sequences with known windowing behavior.
        """
        mock_load_model.return_value = MockDeepCISModel()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create actual gene folders and files
            self._setup_gene_folders(tmpdir)
            
            # Call scan_all_genes with the parent folder (find_gene_folders will discover them)
            result_df, genes_data = scan_all_genes(
                model_path="/model.h5",
                run_folder=tmpdir,
                output_path=tmpdir,
                name="test_comprehensive",
                window_size=10,
                step=7,
                overwrite=True
            )

            expected_output_path = os.path.join(tmpdir, f"deepcis_window_scan_test_comprehensive.csv")
            self.assertTrue(os.path.exists(expected_output_path), "Output CSV file should be created")
            loaded_df = pd.read_csv(expected_output_path)
            self.assertTrue(loaded_df.equals(result_df), "Saved CSV should match returned DataFrame")

            result_df = result_df.sort_values(by=["gene"]).reset_index(drop=True)

            expected_1 = {
                "gene": ["test_gene_1"] * 8,
                "sequence_type": ["reference"] * 4 + ["max_mutated"] * 4,
                "window_start": [0, 7, 14, 15] * 2,
                "window_end": [10, 17, 24, 25] * 2,
                "contains_padding": [False, True, True, False] * 2,
                "CTC": [0.0] * 4 + [1.0] + [0.0] * 3,
                "AGA": [0.0] * 8
            }
            expected_2 = {
                "gene": ["test_gene_2"] * 8,
                "sequence_type": ["reference"] * 4 + ["max_mutated"] * 4,
                "window_start": [0, 7, 14, 15] * 2,
                "window_end": [10, 17, 24, 25] * 2,
                "contains_padding": [False, True, True, False] * 2,
                "CTC": [0.0] * 8,
                "AGA": [0.0] * 4 + [1.0] + [0.0] * 3
            }
            df_1 = pd.DataFrame(expected_1)
            df_2 = pd.DataFrame(expected_2)
            expected_full = pd.concat([df_1, df_2], ignore_index=True).reset_index(drop=True)
            
            # Verify structure
            self.assertIsInstance(result_df, pd.DataFrame)
            self.assertEqual(len(genes_data), 2)
            self.assertIn("test_gene_1", genes_data)
            self.assertIn("test_gene_2", genes_data)
            
            # Verify both genes are in result
            self.assertIn("test_gene_1", result_df["gene"].values)
            self.assertIn("test_gene_2", result_df["gene"].values)
            
            # Verify columns
            expected_cols = {
                "gene", "sequence_type", "window_start", "window_end",
                "contains_padding", "CTC", "AGA"
            }
            self.assertEqual(set(result_df.columns), expected_cols)
            
            # Separate genes for verification
            gene1_df = result_df[result_df["gene"] == "test_gene_1"].reset_index(drop=True)
            gene2_df = result_df[result_df["gene"] == "test_gene_2"].reset_index(drop=True)
            
            self.assertGreater(len(gene1_df), 0)
            self.assertGreater(len(gene2_df), 0)

            print(gene2_df.head(10))
            print(df_2.head(10))

            self.assertTrue(gene1_df.equals(df_1), "Gene 1 results should match expected")
            self.assertTrue(gene2_df.equals(df_2), "Gene 2 results should match expected")
            self.assertTrue(result_df.equals(expected_full), "Full result should match expected combined DataFrame")

if __name__ == "__main__":
    unittest.main()
