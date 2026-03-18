import unittest
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open
import json
import os

from analysis.motives.deepcis_scanner import (
    slide_windows,
    GeneRunData,
    scan_gene_folder,
    find_gene_folders,
    get_full_sequence,
    _get_max_mutation_entry,
    _get_zero_mutation_entry,
    load_deepcis_model,
    predict_windows,
    scan_all_genes,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STEP_SIZE,
    TF_FAMILY_NAMES,
    N_TF_FAMILIES,
)


class TestSlideWindows(unittest.TestCase):
    """Tests for the slide_windows function."""

    def test_slide_windows_basic(self):
        """Test basic window generation with standard parameters."""
        seq = "A" * 500  # 500 bp sequence
        windows = slide_windows(
            sequence=seq,
            window_size=250,
            step=50,
            padding_start=100,
            padding_end=120
        )
        
        # Check number of windows: (500 - 250) // 50 + 1 = 6 windows
        self.assertEqual(len(windows), 6)
        
        # Window 0: 0-250 (overlaps 100-120 padding) -> contains_padding=True
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[0][1], 250)
        self.assertTrue(windows[0][3])
        self.assertEqual(windows[0][2], seq[0:250])

        # Window 5: 250-500 (does not overlap 100-120 padding) -> contains_padding=False
        self.assertEqual(windows[5][0], 250)
        self.assertEqual(windows[5][1], 500)
        self.assertFalse(windows[5][3])
        self.assertEqual(windows[5][2], seq[250:500])

    def test_slide_windows_no_padding(self):
        """Test window generation without explicit padding."""
        seq = "ACGT" * 250  # 1000 bp
        windows = slide_windows(
            sequence=seq,
            window_size=250,
            step=100,
            extragenic=1000,
            intragenic=500,
            central_padding=20
        )
        # (1000 - 250) // 100 + 1 = 8 windows
        self.assertEqual(len(windows), 8)
        self.assertEqual(all(start + 250, end for start, end, _, _ in windows))

    def test_slide_windows_edge_case_exact_fit(self):
        """Test when sequence length equals window size."""
        seq = "A" * 250
        windows = slide_windows(sequence=seq, window_size=250, step=50)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[0][1], 250)


class TestGetFullSequence(unittest.TestCase):
    """Tests for the get_full_sequence function."""

    def test_get_full_sequence_basic(self):
        """Test basic sequence replacement."""
        ref_full = "AAAAAGGGGGTTTTTCCCCC"
        mutable = "XXXX"
        result = get_full_sequence(
            mutable_sequence=mutable,
            reference_sequence_full=ref_full,
            mutation_start=5,
            mutation_end=9
        )
        expected = "AAAAXXXXXTTTTTCCCCC"
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
        expected = "AAAAAGGGGGTTTTTCYYY"
        self.assertEqual(result, expected)


class TestMaxMinMutationEntry(unittest.TestCase):
    """Tests for _get_max_mutation_entry and _get_zero_mutation_entry."""

    def test_get_max_mutation_entry(self):
        """Test finding entry with most mutations."""
        pareto_front = [
            ("SEQSEQSEQ", 0.5, 0),
            ("MUTMUTMUT", 0.8, 3),
            ("MUTMUTMUT", 0.9, 5),
            ("MUTMUTMUT", 0.7, 2),
        ]
        result = _get_max_mutation_entry(pareto_front)
        self.assertEqual(result[2], 5  # mutation count)
        self.assertEqual(result[1], 0.9  # fitness)

    def test_get_zero_mutation_entry_exact(self):
        """Test finding entry with exactly zero mutations."""
        pareto_front = [
            ("SEQSEQSEQ", 0.5, 0),
            ("MUTMUTMUT", 0.8, 3),
        ]
        result = _get_zero_mutation_entry(pareto_front)
        self.assertEqual(result[2], 0  # mutation count)

    def test_get_zero_mutation_entry_fallback(self):
        """Test fallback to minimum mutations when zero unavailable."""
        pareto_front = [
            ("SEQSEQSEQ", 0.5, 2),
            ("MUTMUTMUT", 0.8, 3),
        ]
        result = _get_zero_mutation_entry(pareto_front)
        self.assertEqual(result[2], 2  # minimum mutation count)


class TestFindGeneFolders(unittest.TestCase):
    """Tests for the find_gene_folders function."""

    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("os.path.isfile")
    def test_find_gene_folders_valid(self, mock_isfile, mock_isdir, mock_listdir):
        """Test discovery of valid gene folders."""
        mock_listdir.return_value = ["gene1", "gene2", "not_a_folder", "gene3"]
        
        def isdir_side_effect(path):
            return path.endswith(("gene1", "gene2", "gene3"))
        
        def isfile_side_effect(path):
            # Only gene1 and gene3 have required files
            return (path.endswith("parameters.json") and ("gene1" in path or "gene3" in path)) or \
                   (path.endswith("pareto_front.json") and ("gene1" in path or "gene3" in path))
        
        mock_isdir.side_effect = isdir_side_effect
        mock_isfile.side_effect = isfile_side_effect
        
        result = find_gene_folders("/run/folder")
        self.assertEqual(len(result), 2)
        assert any("gene1" in p for p in result)
        assert any("gene3" in p for p in result)

    @patch("os.listdir")
    @patch("os.path.isdir")
    @patch("os.path.isfile")
    def test_find_gene_folders_empty(self, mock_isfile, mock_isdir, mock_listdir):
        """Test with no valid gene folders."""
        mock_listdir.return_value = ["file.txt", "empty_dir"]
        mock_isdir.return_value = False
        
        result = find_gene_folders("/run/folder")
        self.assertEqual(len(result), 0)


@pytest.fixture
def mock_gene_data():
    """Create a mock GeneRunData for testing."""
    # 3020 length: 1000 extra + 500 intra + 20 pad + 500 intra + 1000 extra
    full_seq = "A" * 1000 + "C" * 500 + "N" * 20 + "G" * 500 + "T" * 1000
    # Let mutable region be just a 10 bp window somewhere
    mutable_seq_ref = "A" * 10
    mutable_seq_mut = "A" * 5 + "C" * 5
    
    return GeneRunData(
        gene_name="test_gene",
        gene_folder="/mock/folder",
        pareto_front=[
            (mutable_seq_ref, 0.5, 0),
            (mutable_seq_mut, 0.9, 5)
        ],
        reference_sequence_full=full_seq,
        mutation_start=1500,
        mutation_end=1510
    )


class TestGeneRunData(unittest.TestCase):
    """Tests for GeneRunData class methods."""

    @patch("analysis.motives.deepcis_scanner.Fasta")
    def test_load_reference_sequence_full_with_key(self, mock_fasta_class, tmp_path):
        """Test loading reference sequence with explicit key."""
        mock_fasta = MagicMock()
        mock_fasta.__contains__ = MagicMock(return_value=True)
        mock_fasta.__getitem__ = MagicMock(return_value="ACGTACGTACGT")
        mock_fasta_class.return_value = mock_fasta
        
        result = GeneRunData.load_reference_sequence_full("/gene/folder")
        self.assertEqual(result, "ACGTACGTACGT")

    @patch("builtins.open", new_callable=mock_open, read_data='[["MUTSEQ", 0.8, 5]]')
    def test_load_pareto_front(self, mock_file):
        """Test loading pareto front from JSON."""
        result = GeneRunData.load_pareto_front("/gene/folder")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("MUTSEQ", 0.8, 5))

    @patch("builtins.open", new_callable=mock_open, read_data='{"mutation_start": 100, "mutation_end": 200}')
    def test_load_gene_params(self, mock_file):
        """Test loading gene parameters from JSON."""
        result = GeneRunData.load_gene_params("/gene/folder")
        self.assertEqual(result["mutation_start"], 100)
        self.assertEqual(result["mutation_end"], 200)


class TestScanGeneFolder(unittest.TestCase):
    """Tests for the scan_gene_folder function."""

    @patch("analysis.motives.deepcis_scanner.predict_windows")
    def test_scan_gene_folder(self, mock_predict, mock_gene_data):
        """Test scanning a single gene folder."""
        # Mock model prediction: return an array of 0.5s matching (n_windows, 46)
        def mock_predict_side_effect(model, encoded_windows, batch_size):
            return np.full((encoded_windows.shape[0], 46), 0.5, dtype=np.float32)
        mock_predict.side_effect = mock_predict_side_effect
        
        dummy_model = MagicMock()
        
        df = scan_gene_folder(
            model=dummy_model,
            gene_data=mock_gene_data,
            window_size=250,
            step=50,
            extragenic=1000,
            intragenic=500,
            central_padding=20
        )
        
        # 3020 bp sequence with win=250, step=50 => (3020 - 250) // 50 + 1 = 56 windows
        # Done for both reference and max_mutated => 112 rows
        self.assertEqual(len(df), 112)
        assert "sequence_type" in df.columns
        self.assertEqual(set(df["sequence_type"].unique()), {"reference", "max_mutated"})
        
        # Ensure all 46 TF columns are present and correctly named
        self.assertEqual(len(TF_FAMILY_NAMES), N_TF_FAMILIES)
        for tf_name in TF_FAMILY_NAMES:
            col_name = f"tf_{tf_name}"
            assert col_name in df.columns
            assert df[col_name].iloc[0] == 0.5

    @patch("analysis.motives.deepcis_scanner.predict_windows")
    def test_scan_gene_folder_output_columns(self, mock_predict, mock_gene_data):
        """Test that scan_gene_folder includes all required columns."""
        mock_predict.return_value = np.zeros((10, 46), dtype=np.float32)
        dummy_model = MagicMock()
        
        df = scan_gene_folder(
            model=dummy_model,
            gene_data=mock_gene_data,
            window_size=250,
            step=50
        )
        
        required_cols = {"gene", "sequence_type", "window_start", "window_end", "contains_padding"}
        assert required_cols.issubset(df.columns)


class TestLoadModel(unittest.TestCase):
    """Tests for the load_deepcis_model function."""

    @patch("tensorflow.keras.models.load_model")
    def test_load_deepcis_model(self, mock_load_model):
        """Test loading a deepCIS model."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        result = load_deepcis_model("/path/to/model.h5")
        self.assertEqual(result, mock_model)
        mock_load_model.assert_called_once_with("/path/to/model.h5")


class TestPredictWindows(unittest.TestCase):
    """Tests for the predict_windows function."""

    def test_predict_windows_shape(self):
        """Test that predict_windows returns correct output shape."""
        mock_model = MagicMock()
        # Mock returns shape (n_windows, 46)
        mock_model.predict.return_value = np.ones((10, 46), dtype=np.float32)
        
        windows = np.random.randn(10, 250, 4)
        result = predict_windows(mock_model, windows, batch_size=5)
        
        self.assertEqual(result.shape, (10, 46))
        self.assertEqual(result.dtype, np.float32)


class TestScanAllGenes(unittest.TestCase):
    """Tests for the scan_all_genes function."""

    @patch("analysis.motives.deepcis_scanner.find_gene_folders")
    @patch("analysis.motives.deepcis_scanner.load_deepcis_model")
    @patch("analysis.motives.deepcis_scanner.GeneRunData.load_gene_run_data")
    @patch("analysis.motives.deepcis_scanner.scan_gene_folder")
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_scan_all_genes_no_cache(
        self, mock_exists, mock_makedirs, mock_scan_gene,
        mock_load_gene, mock_load_model, mock_find_folders, tmp_path
    ):
        """Test scanning all genes when no cache exists."""
        mock_exists.return_value = False
        mock_find_folders.return_value = ["/gene1", "/gene2"]
        mock_load_model.return_value = MagicMock()
        
        # Set up mock gene data
        mock_gene1 = MagicMock()
        mock_gene1.gene_name = "gene1"
        mock_gene2 = MagicMock()
        mock_gene2.gene_name = "gene2"
        mock_load_gene.side_effect = [mock_gene1, mock_gene2]
        
        # Set up mock scan results
        df1 = pd.DataFrame({"gene": ["gene1"], "value": [1]})
        df2 = pd.DataFrame({"gene": ["gene2"], "value": [2]})
        mock_scan_gene.side_effect = [df1, df2]
        
        result_df, genes_data = scan_all_genes(
            model_path="/model.h5",
            run_folder="/run",
            output_path=str(tmp_path),
            name="test_run",
            overwrite=True
        )
        
        # Should process both genes
        self.assertEqual(len(genes_data), 2)
        assert "gene1" in genes_data
        assert "gene2" in genes_data

    @patch("os.path.exists")
    @patch("pandas.read_csv")
    def test_scan_all_genes_with_cache(self, mock_read_csv, mock_exists):
        """Test loading existing output without re-running."""
        mock_exists.return_value = True
        mock_df = pd.DataFrame({"gene": ["cached_result"]})
        mock_read_csv.return_value = mock_df
        
        result_df, genes_data = scan_all_genes(
            model_path="/model.h5",
            run_folder="/run",
            output_path="/output",
            overwrite=False
        )
        
        # Should load from cache
        self.assertEqual(len(genes_data), 0  # Empty because not re-running)
        pd.testing.assert_frame_equal(result_df, mock_df)
