"""Tests for deepCIS visualization module."""

import os
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from analysis.motives.deepcis_visualize import (
    extract_plot_data,
    get_tf_columns,
    load_scan_results,
    _get_padding_regions,
    _validate_and_set_defaults,
    _load_input_data,
    _resolve_output_directory,
    parse_arguments,
    visualize_scan_results,
)


class TestExtractPlotData(unittest.TestCase):
    """Tests for the extract_plot_data function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample deepCIS scan data for testing
        self.sample_scan_data = pd.DataFrame({
            "gene": ["gene1", "gene1", "gene1", "gene1"],
            "sequence_type": ["reference", "reference", "max_mutated", "max_mutated"],
            "window_start": [0, 50, 0, 50],
            "window_end": [250, 300, 250, 300],
            "contains_padding": [False, True, False, True],
            "tf_0": [0.1, 0.5, 0.2, 0.6],
            "tf_1": [0.3, 0.4, 0.35, 0.45],
            "tf_2": [0.2, 0.2, 0.25, 0.3],
        })

    def test_extract_plot_data_basic(self):
        """Test basic extraction of plot data."""
        gene_df = self.sample_scan_data[self.sample_scan_data["gene"] == "gene1"]
        ref_x, ref_y, mut_x, mut_y, x_min, x_max = extract_plot_data(gene_df, "tf_0")

        # Check that we get two reference points and two mutated points
        self.assertEqual(len(ref_x), 2)
        self.assertEqual(len(ref_y), 2)
        self.assertEqual(len(mut_x), 2)
        self.assertEqual(len(mut_y), 2)

        # Check x values are at window centers
        # Window 1: [0, 250) center = 125.5
        # Window 2: [50, 300) center = 175.5
        self.assertTrue(np.allclose(ref_x, [125.5, 175.5]))
        self.assertTrue(np.allclose(mut_x, [125.5, 175.5]))

    def test_extract_plot_data_y_values(self):
        """Test that y values are correctly extracted."""
        gene_df = self.sample_scan_data[self.sample_scan_data["gene"] == "gene1"]
        ref_x, ref_y, mut_x, mut_y, x_min, x_max = extract_plot_data(gene_df, "tf_0")

        # Reference: [0.1, 0.5]
        # Mutated: [0.2, 0.6]
        self.assertTrue(np.allclose(ref_y, [0.1, 0.5]))
        self.assertTrue(np.allclose(mut_y, [0.2, 0.6]))

    def test_extract_plot_data_x_range(self):
        """Test that x_min and x_max are correctly determined."""
        gene_df = self.sample_scan_data[self.sample_scan_data["gene"] == "gene1"]
        ref_x, ref_y, mut_x, mut_y, x_min, x_max = extract_plot_data(gene_df, "tf_0")

        # x_min should be minimum window_start (0)
        # x_max should be maximum window_end (300)
        self.assertEqual(x_min, 0)
        self.assertEqual(x_max, 300)

    def test_extract_plot_data_different_tf(self):
        """Test extraction with different TF columns."""
        gene_df = self.sample_scan_data[self.sample_scan_data["gene"] == "gene1"]

        # Test tf_1
        ref_x, ref_y, mut_x, mut_y, x_min, x_max = extract_plot_data(gene_df, "tf_1")
        self.assertTrue(np.allclose(ref_y, [0.3, 0.4]))
        self.assertTrue(np.allclose(mut_y, [0.35, 0.45]))

        # Test tf_2
        ref_x, ref_y, mut_x, mut_y, x_min, x_max = extract_plot_data(gene_df, "tf_2")
        self.assertTrue(np.allclose(ref_y, [0.2, 0.2]))
        self.assertTrue(np.allclose(mut_y, [0.25, 0.3]))

    def test_extract_plot_data_single_window(self):
        """Test extraction with only one window."""
        data = {
            "gene": ["gene1", "gene1"],
            "sequence_type": ["reference", "max_mutated"],
            "window_start": [100, 100],
            "window_end": [350, 350],
            "contains_padding": [False, True],
            "tf_0": [0.5, 0.7],
        }
        df = pd.DataFrame(data)
        ref_x, ref_y, mut_x, mut_y, x_min, x_max = extract_plot_data(df, "tf_0")

        # Single reference point
        self.assertEqual(len(ref_x), 1)
        self.assertEqual(len(mut_x), 1)

        # Window [100, 350) center = 225.5
        self.assertTrue(np.allclose(ref_x, [225.5]))
        self.assertTrue(np.allclose(mut_x, [225.5]))

        # X range
        self.assertEqual(x_min, 100)
        self.assertEqual(x_max, 350)

    def test_extract_plot_data_empty_reference(self):
        """Test extraction when reference sequence is missing."""
        data = {
            "gene": ["gene1"],
            "sequence_type": ["max_mutated"],
            "window_start": [0],
            "window_end": [250],
            "contains_padding": [False],
            "tf_0": [0.7],
        }
        df = pd.DataFrame(data)
        ref_x, ref_y, mut_x, mut_y, x_min, x_max = extract_plot_data(df, "tf_0")

        # No reference data
        self.assertEqual(len(ref_x), 0)
        self.assertEqual(len(ref_y), 0)

        # Mutated data present
        self.assertEqual(len(mut_x), 1)
        self.assertEqual(len(mut_y), 1)

    def test_extract_plot_data_multiple_windows(self):
        """Test extraction with many windows."""
        data = {
            "gene": (["gene1"] * 6),
            "sequence_type": ["reference"] * 3 + ["max_mutated"] * 3,
            "window_start": [0, 50, 100, 0, 50, 100],
            "window_end": [250, 300, 350, 250, 300, 350],
            "contains_padding": [False] * 6,
            "tf_0": [0.1, 0.2, 0.3, 0.15, 0.25, 0.35],
        }
        df = pd.DataFrame(data)
        ref_x, ref_y, mut_x, mut_y, x_min, x_max = extract_plot_data(df, "tf_0")

        # Three windows per sequence type
        self.assertEqual(len(ref_x), 3)
        self.assertEqual(len(mut_x), 3)

        # Check x values (window centers)
        expected_x = [125.5, 175.5, 225.5]
        self.assertTrue(np.allclose(ref_x, expected_x))
        self.assertTrue(np.allclose(mut_x, expected_x))

        # Check y values (should be sorted by window_start)
        self.assertTrue(np.allclose(ref_y, [0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(mut_y, [0.15, 0.25, 0.35]))


class TestGetTfColumns(unittest.TestCase):
    """Tests for the get_tf_columns function."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_scan_data = pd.DataFrame({
            "gene": ["gene1", "gene1", "gene1", "gene1"],
            "sequence_type": ["reference", "reference", "max_mutated", "max_mutated"],
            "window_start": [0, 50, 0, 50],
            "window_end": [250, 300, 250, 300],
            "contains_padding": [False, True, False, True],
            "tf_0": [0.1, 0.5, 0.2, 0.6],
            "tf_1": [0.3, 0.4, 0.35, 0.45],
            "tf_2": [0.2, 0.2, 0.25, 0.3],
        })

    def test_get_tf_columns_basic(self):
        """Test tf column extraction from data with fixed columns."""
        tf_cols = get_tf_columns(self.sample_scan_data)
        self.assertEqual(tf_cols, ["tf_0", "tf_1", "tf_2"])

    def test_get_tf_columns_sorted(self):
        """Test that tf columns are sorted."""
        data = {
            "gene": ["gene1"],
            "sequence_type": ["reference"],
            "window_start": [0],
            "window_end": [250],
            "tf_15": [0.1],
            "tf_3": [0.2],
            "tf_8": [0.3],
            "tf_0": [0.4],
        }
        df = pd.DataFrame(data)
        tf_cols = get_tf_columns(df)
        # Note: string sorting, so "tf_0" < "tf_15" < "tf_3" < "tf_8"
        self.assertEqual(tf_cols, ["tf_0", "tf_15", "tf_3", "tf_8"])

    def test_get_tf_columns_no_tf(self):
        """Test with DataFrame containing no TF columns (only fixed columns)."""
        data = {
            "gene": ["gene1"],
            "sequence_type": ["reference"],
            "window_start": [0],
            "window_end": [250],
            "contains_padding": [False],
        }
        df = pd.DataFrame(data)
        tf_cols = get_tf_columns(df)
        self.assertEqual(tf_cols, [])

    def test_get_tf_columns_alternative_names(self):
        """Test that function works with any TF naming scheme (not just tf_ prefix)."""
        data = {
            "gene": ["gene1"],
            "sequence_type": ["reference"],
            "window_start": [0],
            "window_end": [250],
            "contains_padding": [False],
            "BHLH": [0.1],
            "WRKY": [0.2],
            "MYB": [0.3],
        }
        df = pd.DataFrame(data)
        tf_cols = get_tf_columns(df)
        # Should extract all non-fixed columns
        self.assertEqual(set(tf_cols), {"BHLH", "WRKY", "MYB"})
        # Should be sorted
        self.assertEqual(tf_cols, sorted(["BHLH", "WRKY", "MYB"]))





class TestGetPaddingRegions(unittest.TestCase):
    """Tests for the _get_padding_regions function."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_scan_data = pd.DataFrame({
            "gene": ["gene1", "gene1", "gene1", "gene1"],
            "sequence_type": ["reference", "reference", "max_mutated", "max_mutated"],
            "window_start": [0, 50, 0, 50],
            "window_end": [250, 300, 250, 300],
            "contains_padding": [False, True, False, True],
            "tf_0": [0.1, 0.5, 0.2, 0.6],
            "tf_1": [0.3, 0.4, 0.35, 0.45],
            "tf_2": [0.2, 0.2, 0.25, 0.3],
        })

    def test_get_padding_regions_basic(self):
        """Test extraction of padding regions."""
        regions = _get_padding_regions(self.sample_scan_data)
        # Should have 1 region: [50, 300) (deduplicated)
        self.assertEqual(len(regions), 1)
        self.assertIn((50, 300), regions)

    def test_get_padding_regions_no_padding_column(self):
        """Test when padding column is missing."""
        data = {
            "gene": ["gene1"],
            "sequence_type": ["reference"],
            "window_start": [0],
            "window_end": [250],
            "tf_0": [0.1],
        }
        df = pd.DataFrame(data)
        regions = _get_padding_regions(df)
        self.assertEqual(regions, [])

    def test_get_padding_regions_no_padding(self):
        """Test when no padding regions exist."""
        # All padding flags are False
        data = self.sample_scan_data.copy()
        data["contains_padding"] = False
        regions = _get_padding_regions(data)
        self.assertEqual(regions, [])

    def test_get_padding_regions_multiple(self):
        """Test with multiple separate padding regions."""
        data = {
            "gene": ["gene1"] * 4,
            "sequence_type": ["reference"] * 2 + ["max_mutated"] * 2,
            "window_start": [0, 100, 0, 100],
            "window_end": [250, 350, 250, 350],
            "contains_padding": [True, False, False, True],
            "tf_0": [0.1, 0.2, 0.3, 0.4],
        }
        df = pd.DataFrame(data)
        regions = _get_padding_regions(df)
        # Should have 2 unique regions
        self.assertEqual(len(regions), 2)
        self.assertIn((0, 250), regions)
        self.assertIn((100, 350), regions)

    def test_get_padding_regions_sorted(self):
        """Test that padding regions are sorted."""
        data = {
            "gene": ["gene1"] * 2,
            "sequence_type": ["reference", "max_mutated"],
            "window_start": [100, 0],
            "window_end": [350, 250],
            "contains_padding": [True, True],
            "tf_0": [0.1, 0.2],
        }
        df = pd.DataFrame(data)
        regions = _get_padding_regions(df)
        # Should be sorted: (0, 250) before (100, 350)
        self.assertEqual(regions, [(0, 250), (100, 350)])


class TestValidateAndSetDefaults(unittest.TestCase):
    """Tests for the _validate_and_set_defaults function."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_scan_data = pd.DataFrame({
            "gene": ["gene1", "gene1", "gene1", "gene1"],
            "sequence_type": ["reference", "reference", "max_mutated", "max_mutated"],
            "window_start": [0, 50, 0, 50],
            "window_end": [250, 300, 250, 300],
            "contains_padding": [False, True, False, True],
            "tf_0": [0.1, 0.5, 0.2, 0.6],
            "tf_1": [0.3, 0.4, 0.35, 0.45],
            "tf_2": [0.2, 0.2, 0.25, 0.3],
        })

        self.sample_scan_data_two_genes = pd.DataFrame({
            "gene": [
                "geneA", "geneA", "geneA", "geneA",
                "geneB", "geneB", "geneB", "geneB"
            ],
            "sequence_type": ["reference", "reference", "max_mutated", "max_mutated"] * 2,
            "window_start": [0, 50, 0, 50, 100, 150, 100, 150],
            "window_end": [250, 300, 250, 300, 350, 400, 350, 400],
            "contains_padding": [False, True, False, True, False, False, True, True],
            "tf_0": [0.1, 0.5, 0.2, 0.6, 0.15, 0.45, 0.25, 0.55],
            "tf_1": [0.3, 0.4, 0.35, 0.45, 0.32, 0.42, 0.37, 0.47],
        })

    def test_validate_and_set_defaults_both_none(self):
        """Test with both genes and tfs as None."""
        genes, tfs = _validate_and_set_defaults(self.sample_scan_data, None, None)
        self.assertEqual(genes, ["gene1"])
        self.assertEqual(tfs, ["tf_0", "tf_1", "tf_2"])

    def test_validate_and_set_defaults_specific_genes(self):
        """Test with specific gene selection."""
        genes, tfs = _validate_and_set_defaults(
            self.sample_scan_data_two_genes, ["geneA"], None
        )
        self.assertEqual(genes, ["geneA"])
        self.assertEqual(tfs, ["tf_0", "tf_1"])

    def test_validate_and_set_defaults_specific_tfs(self):
        """Test with specific TF selection."""
        genes, tfs = _validate_and_set_defaults(self.sample_scan_data, None, ["tf_0", "tf_1"])
        self.assertEqual(genes, ["gene1"])
        self.assertEqual(tfs, ["tf_0", "tf_1"])

    def test_validate_and_set_defaults_invalid_gene(self):
        """Test that invalid gene raises error."""
        with self.assertRaises(ValueError):
            _validate_and_set_defaults(self.sample_scan_data, ["nonexistent"], None)

    def test_validate_and_set_defaults_invalid_tf(self):
        """Test that invalid TF raises error."""
        with self.assertRaises(ValueError):
            _validate_and_set_defaults(self.sample_scan_data, None, ["tf_nonexistent"])

    def test_validate_and_set_defaults_filters_invalid(self):
        """Test that invalid items are filtered out."""
        with self.assertRaises(ValueError):
            genes, tfs = _validate_and_set_defaults(
                self.sample_scan_data_two_genes,
                ["geneA", "nonexistent", "geneB"],
                ["tf_0", "tf_nonexistent", "tf_1"],
            )


class TestLoadInputData(unittest.TestCase):
    """Tests for the _load_input_data function."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_scan_data = pd.DataFrame({
            "gene": ["gene1", "gene1", "gene1", "gene1"],
            "sequence_type": ["reference", "reference", "max_mutated", "max_mutated"],
            "window_start": [0, 50, 0, 50],
            "window_end": [250, 300, 250, 300],
            "contains_padding": [False, True, False, True],
            "tf_0": [0.1, 0.5, 0.2, 0.6],
            "tf_1": [0.3, 0.4, 0.35, 0.45],
            "tf_2": [0.2, 0.2, 0.25, 0.3],
        })

    def test_load_input_data_from_file(self):
        """Test loading from CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "scan.csv")
            self.sample_scan_data.to_csv(csv_path, index=False)
            df = _load_input_data(csv_path)
            self.assertEqual(len(df), 4)
            self.assertTrue(df.equals(self.sample_scan_data))

    def test_load_input_data_from_dataframe(self):
        """Test passing DataFrame directly."""
        df = _load_input_data(self.sample_scan_data)
        self.assertIs(df, self.sample_scan_data)

    def test_load_input_data_invalid_type(self):
        """Test with invalid input type."""
        with self.assertRaises(ValueError):
            _load_input_data(123)

    def test_load_input_data_empty_dataframe(self):
        """Test with empty DataFrame."""
        with self.assertRaises(ValueError):
            _load_input_data(pd.DataFrame())

    def test_load_input_data_nonexistent_file(self):
        """Test with nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            _load_input_data("/nonexistent/path.csv")


class TestResolveOutputDirectory(unittest.TestCase):
    """Tests for the _resolve_output_directory function."""

    def test_resolve_output_directory_explicit(self):
        """Test with explicit output directory."""
        result = _resolve_output_directory("data/file.csv", "custom_dir")
        self.assertEqual(result, "custom_dir")

    def test_resolve_output_directory_from_file(self):
        """Test resolving from file path."""
        result = _resolve_output_directory("data/file.csv", None)
        self.assertEqual(result, os.path.join("data", "deepcis_scan_plots"))

    def test_resolve_output_directory_from_dataframe(self):
        """Test with input as DataFrame (not file)."""
        result = _resolve_output_directory(None, None)
        self.assertEqual(result, os.path.join(os.getcwd(), "deepcis_scan_plots"))


class TestVisualizeScanResults(unittest.TestCase):
    """Tests for the main visualize_scan_results entry point."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_scan_data = pd.DataFrame({
            "gene": ["gene1", "gene1", "gene1", "gene1"],
            "sequence_type": ["reference", "reference", "max_mutated", "max_mutated"],
            "window_start": [0, 50, 0, 50],
            "window_end": [250, 300, 250, 300],
            "contains_padding": [False, True, False, True],
            "tf_0": [0.1, 0.5, 0.2, 0.6],
            "tf_1": [0.3, 0.4, 0.35, 0.45],
            "tf_2": [0.2, 0.2, 0.25, 0.3],
        })

    def test_visualize_from_file(self):
        """Test visualization from CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "scan.csv")
            self.sample_scan_data.to_csv(csv_path, index=False)

            visualize_scan_results(
                csv_path,
                genes=["gene1"],
                tfs=["tf_0"],
                output_dir=os.path.join(tmpdir, "output"),
            )

            # Check file was created
            output_file = os.path.join(tmpdir, "output", "gene1", "gene1_TF_0.png")
            self.assertTrue(os.path.exists(output_file))

    def test_visualize_from_dataframe(self):
        """Test visualization from DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualize_scan_results(
                self.sample_scan_data,
                genes=["gene1"],
                tfs=["tf_0"],
                output_dir=tmpdir,
            )

            output_file = os.path.join(tmpdir, "gene1", "gene1_TF_0.png")
            self.assertTrue(os.path.exists(output_file))

    def test_visualize_default_output_directory_from_file(self):
        """Test that default output directory is 'plots' next to input file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "scan.csv")
            self.sample_scan_data.to_csv(csv_path, index=False)

            # No output_dir specified
            visualize_scan_results(
                csv_path,
                genes=["gene1"],
                tfs=["tf_0"],
            )

            # Should create plots in tmpdir/plots
            expected_file = os.path.join(tmpdir, "deepcis_scan_plots", "gene1", "gene1_TF_0.png")
            self.assertTrue(os.path.exists(expected_file))


if __name__ == '__main__':
    unittest.main()
