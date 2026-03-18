import unittest
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from typing import List, Tuple

from analysis.motives.deepcis_annotation import (
    _mutations_from_ohe,
    compare_sequences_df,
    annotate_mutations_in_windows,
    add_genomic_coordinates,
    DEFAULT_DELTA_THRESHOLD,
    DEFAULT_EXTRAGENIC,
    DEFAULT_INTRAGENIC,
    DEFAULT_CENTRAL_PADDING,
    TF_FAMILY_NAMES,
)
from analysis.motives.deepcis_scanner import GeneRunData


class TestMutationsFromOHE(unittest.TestCase):
    """Tests for the _mutations_from_ohe function."""

    def test_mutations_from_ohe_single_mutation(self):
        """Test detecting a single mutation."""
        # Reference: ACGT
        ref_ohe = np.array([
            [1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # C
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1],  # T
        ], dtype=np.float32)
        
        # Mutated: ATGT (C->T at position 1)
        max_ohe = np.array([
            [1, 0, 0, 0],  # A
            [0, 0, 0, 1],  # T
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1],  # T
        ], dtype=np.float32)
        
        mutations = _mutations_from_ohe(ref_ohe, max_ohe)
        self.assertEqual(len(mutations), 1)
        self.assertEqual(mutations[0], (1, "C", "T"))

    def test_mutations_from_ohe_multiple_mutations(self):
        """Test detecting multiple mutations."""
        # Reference: AAAA
        ref_ohe = np.ones((4, 4), dtype=np.float32)
        ref_ohe[:] = 0
        ref_ohe[:, 0] = 1  # All A's
        
        # Mutated: ACGA (positions 1->C, 2->G, 3->A)
        max_ohe = np.zeros((4, 4), dtype=np.float32)
        max_ohe[0, 0] = 1  # A
        max_ohe[1, 1] = 1  # C
        max_ohe[2, 2] = 1  # G
        max_ohe[3, 0] = 1  # A
        
        mutations = _mutations_from_ohe(ref_ohe, max_ohe)
        self.assertEqual(len(mutations), 2)
        assert (1, "A", "C") in mutations
        assert (2, "A", "G") in mutations

    def test_mutations_from_ohe_no_mutations(self):
        """Test when sequences are identical."""
        ohe = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        
        mutations = _mutations_from_ohe(ohe, ohe)
        self.assertEqual(len(mutations), 0)


class TestCompareSequencesDF(unittest.TestCase):
    """Tests for the compare_sequences_df function."""

    @pytest.fixture
    def scan_data(self):
        """Create mock scan data with reference and mutated sequences."""
        # Create data with 2 TF families for simplicity
        data = {
            "gene": ["gene1", "gene1", "gene2", "gene2"],
            "sequence_type": ["reference", "max_mutated", "reference", "max_mutated"],
            "window_start": [0, 0, 100, 100],
            "window_end": [250, 250, 350, 350],
            "contains_padding": [False, False, True, True],
        }
        
        # Add TF columns
        for tf_name in TF_FAMILY_NAMES[:2]:
            tf_col = f"tf_{tf_name}"
            data[tf_col] = [0.1, 0.3, 0.5, 0.6, 0.2, 0.25, 0.8, 0.9]
        
        return pd.DataFrame(data)

    def test_compare_sequences_df_basic(self, scan_data):
        """Test basic comparison of reference vs mutated sequences."""
        result = compare_sequences_df(scan_data, threshold=0.1)
        
        # Should have one row per gene/window pair
        self.assertEqual(len(result), 2)
        assert "binding_changed" in result.columns
        assert "max_delta" in result.columns
        
        # Check delta columns exist
        for tf_name in TF_FAMILY_NAMES[:2]:
            assert f"delta_tf_{tf_name}" in result.columns

    def test_compare_sequences_df_threshold(self, scan_data):
        """Test that threshold correctly flags binding changes."""
        result = compare_sequences_df(scan_data, threshold=0.2)
        
        # First row: |0.3 - 0.1| = 0.2 for first TF, should be flagged
        self.assertTrue(result.iloc[0]["binding_changed"])
        
        # Adjust threshold higher
        result = compare_sequences_df(scan_data, threshold=0.5)
        # Now first row should not be flagged (max delta is 0.2)
        self.assertFalse(result.iloc[0]["binding_changed"])

    def test_compare_sequences_df_columns_structure(self, scan_data):
        """Test that output has correct column structure."""
        result = compare_sequences_df(scan_data)
        
        # Should have ref and max_mut versions of each TF column
        for tf_name in TF_FAMILY_NAMES[:2]:
            assert f"tf_{tf_name}_ref" in result.columns
            assert f"tf_{tf_name}_max_mut" in result.columns


class TestAnnotateMutationsInWindows(unittest.TestCase):
    """Tests for the annotate_mutations_in_windows function."""

    @pytest.fixture
    def changed_df(self):
        """Create a mock DataFrame of changed windows."""
        return pd.DataFrame({
            "gene": ["test_gene1", "test_gene1"],
            "window_start": [100, 200],
            "window_end": [350, 450],
            "binding_changed": [True, True],
        })

    @pytest.fixture
    def genes_data(self):
        """Create mock gene data."""
        full_seq = "A" * 1000
        return {
            "test_gene1": GeneRunData(
                gene_name="test_gene1",
                gene_folder="/mock",
                pareto_front=[
                    ("A" * 10, 0.5, 0),
                    ("A" * 5 + "C" * 5, 0.9, 5),
                ],
                reference_sequence_full=full_seq,
                mutation_start=500,
                mutation_end=510,
            )
        }

    @patch("analysis.motives.deepcis_annotation.one_hot_encode")
    @patch("analysis.motives.deepcis_annotation._mutations_from_ohe")
    def test_annotate_mutations_in_windows(
        self, mock_mutations, mock_encode, changed_df, genes_data
    ):
        """Test annotation of mutations within windows."""
        mock_encode.return_value = np.zeros((1000, 4), dtype=np.float32)
        mock_mutations.return_value = [
            (150, "A", "C"),
            (250, "A", "G"),
            (500, "A", "T"),  # Outside window
        ]
        
        result = annotate_mutations_in_windows(changed_df, genes_data)
        
        assert "mutations_in_window" in result.columns
        assert "n_mutations_in_window" in result.columns
        
        # Window 1: [100, 350) should contain mutations at 150 and 250
        self.assertEqual(result.iloc[0]["n_mutations_in_window"], 2)
        
        # Window 2: [200, 450) should contain mutations at 250
        self.assertEqual(result.iloc[1]["n_mutations_in_window"], 1)

    def test_annotate_mutations_in_windows_missing_gene(self, changed_df):
        """Test handling of missing gene data."""
        result = annotate_mutations_in_windows(changed_df, {})
        
        # Should have empty lists for mutations
        self.assertEqual(result.iloc[0]["n_mutations_in_window"], 0)
        self.assertEqual(result.iloc[1]["n_mutations_in_window"], 0)


class TestAddGenomicCoordinates(unittest.TestCase):
    """Tests for the add_genomic_coordinates function."""

    @pytest.fixture
    def window_df(self):
        """Create a DataFrame with sequence-relative coordinates."""
        return pd.DataFrame({
            "gene": ["Chr1_100_500_+", "Chr2_1000_2000_-"],
            "window_start": [100, 200],
            "window_end": [350, 450],
        })

    @patch("analysis.motives.deepcis_annotation.AnnotatedMutatedSequence")
    def test_add_genomic_coordinates_forward_strand(
        self, mock_sequence_class, window_df
    ):
        """Test coordinate mapping on forward strand."""
        mock_sequence_class.parse_sequence_name.return_value = (
            "Chr1_gene", "Chr1", 100, 500
        )
        
        result = add_genomic_coordinates(
            window_df,
            extragenic=1000,
            intragenic=500,
            central_padding=20,
        )
        
        assert "gene_id" in result.columns
        assert "chromosome" in result.columns
        assert "strand" in result.columns
        assert "genomic_window_start" in result.columns
        assert "genomic_window_end" in result.columns

    @patch("analysis.motives.deepcis_annotation.AnnotatedMutatedSequence")
    def test_add_genomic_coordinates_invalid_gene_name(
        self, mock_sequence_class, window_df
    ):
        """Test handling of unparseable gene names."""
        mock_sequence_class.parse_sequence_name.side_effect = Exception("Parse error")
        
        result = add_genomic_coordinates(window_df)
        
        # Should fill with defaults
        self.assertEqual(result.iloc[0]["gene_id"], "Chr1_100_500_+")
        self.assertEqual(result.iloc[0]["chromosome"], "unknown")
        self.assertEqual(result.iloc[0]["strand"], "unknown")
        assert result.iloc[0]["genomic_window_start"] is None

    @patch("analysis.motives.deepcis_annotation.AnnotatedMutatedSequence")
    def test_add_genomic_coordinates_reverse_strand(
        self, mock_sequence_class, window_df
    ):
        """Test coordinate mapping on reverse strand."""
        # For reverse strand, gene_end < gene_start
        mock_sequence_class.parse_sequence_name.return_value = (
            "Chr2_gene", "Chr2", 2000, 1000
        )
        
        result = add_genomic_coordinates(window_df)
        
        # Reverse strand should have different coordinate logic
        self.assertEqual(result.iloc[1]["strand"], "-")


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple functions."""

    @patch("analysis.motives.deepcis_annotation._mutations_from_ohe")
    @patch("analysis.motives.deepcis_annotation.one_hot_encode")
    @patch("analysis.motives.deepcis_annotation.AnnotatedMutatedSequence")
    def test_full_annotation_pipeline(
        self, mock_sequence_class, mock_encode, mock_mutations
    ):
        """Test the full annotation pipeline."""
        # Create mock scan results
        scan_df = pd.DataFrame({
            "gene": ["gene1", "gene1"],
            "sequence_type": ["reference", "max_mutated"],
            "window_start": [0, 0],
            "window_end": [250, 250],
            "contains_padding": [False, False],
        })
        
        # Add TF columns
        for tf_name in TF_FAMILY_NAMES[:2]:
            tf_col = f"tf_{tf_name}"
            scan_df[tf_col] = [0.5, 0.8, 0.3, 0.2]
        
        # Setup mocks
        mock_encode.return_value = np.zeros((1000, 4))
        mock_mutations.return_value = [(100, "A", "C")]
        mock_sequence_class.parse_sequence_name.return_value = (
            "gene1", "Chr1", 100, 500
        )
        
        # Step 1: Compare sequences
        compared = compare_sequences_df(scan_df, threshold=0.1)
        self.assertEqual(len(compared), 1)
        assert "binding_changed" in compared.columns
        
        # Step 2: Annotate mutations
        genes_data = {
            "gene1": GeneRunData(
                gene_name="gene1",
                gene_folder="/mock",
                pareto_front=[("AAAAAAAAAA", 0.5, 0), ("AAACCCCAAA", 0.8, 5)],
                reference_sequence_full="A" * 1000,
                mutation_start=500,
                mutation_end=510,
            )
        }
        
        annotated = annotate_mutations_in_windows(compared, genes_data)
        assert "mutations_in_window" in annotated.columns
        
        # Step 3: Add genomic coordinates
        final = add_genomic_coordinates(annotated)
        assert "genomic_window_start" in final.columns
