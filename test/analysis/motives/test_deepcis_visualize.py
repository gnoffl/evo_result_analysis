"""Tests for deepCIS visualization module."""

import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from analysis.motives.deepcis_visualize import (
    extract_plot_data,
    get_tf_columns,
    load_scan_results,
    load_peak_results,
    _get_padding_regions,
    _get_peak_background_regions,
    _get_peak_background_regions_from_peaks,
    _validate_and_set_defaults,
    _load_input_data,
    _resolve_output_directory,
    _select_random_subset,
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
        gene_df: pd.DataFrame = self.sample_scan_data[self.sample_scan_data["gene"] == "gene1"]         #type:ignore
        ref_x, ref_y, mut_x, mut_y, _, _, x_min, x_max = extract_plot_data(gene_df, "tf_0")

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
        gene_df: pd.DataFrame = self.sample_scan_data[self.sample_scan_data["gene"] == "gene1"]         #type:ignore
        ref_x, ref_y, mut_x, mut_y, _, _, x_min, x_max = extract_plot_data(gene_df, "tf_0")

        # Reference: [0.1, 0.5]
        # Mutated: [0.2, 0.6]
        self.assertTrue(np.allclose(ref_y, [0.1, 0.5]))
        self.assertTrue(np.allclose(mut_y, [0.2, 0.6]))

    def test_extract_plot_data_x_range(self):
        """Test that x_min and x_max are correctly determined."""
        gene_df: pd.DataFrame = self.sample_scan_data[self.sample_scan_data["gene"] == "gene1"]         #type:ignore
        ref_x, ref_y, mut_x, mut_y, _, _, x_min, x_max = extract_plot_data(gene_df, "tf_0")

        # x_min should be minimum window_start (0)
        # x_max should be maximum window_end (300)
        self.assertEqual(x_min, 0)
        self.assertEqual(x_max, 300)

    def test_extract_plot_data_different_tf(self):
        """Test extraction with different TF columns."""
        gene_df: pd.DataFrame = self.sample_scan_data[self.sample_scan_data["gene"] == "gene1"]         #type:ignore

        # Test tf_1
        ref_x, ref_y, mut_x, mut_y, _, _, x_min, x_max = extract_plot_data(gene_df, "tf_1")
        self.assertTrue(np.allclose(ref_y, [0.3, 0.4]))
        self.assertTrue(np.allclose(mut_y, [0.35, 0.45]))

        # Test tf_2
        ref_x, ref_y, mut_x, mut_y, _, _, x_min, x_max = extract_plot_data(gene_df, "tf_2")
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
        ref_x, ref_y, mut_x, mut_y, _, _, x_min, x_max = extract_plot_data(df, "tf_0")

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
        ref_x, ref_y, mut_x, mut_y, _, _, x_min, x_max = extract_plot_data(df, "tf_0")

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
        ref_x, ref_y, mut_x, mut_y, _, _, x_min, x_max = extract_plot_data(df, "tf_0")

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

    def test_get_tf_columns_ignores_peak_annotation_columns(self):
        data = {
            "gene": ["gene1"],
            "sequence_type": ["reference"],
            "window_start": [0],
            "window_end": [250],
            "contains_padding": [False],
            "window_id": [0],
            "tf_0": [0.4],
            "tf_0__reference__in_peak": [True],
            "tf_0__reference__region_idx_list": [[0]],
            "tf_0__reference__peak_rank_list": [[0]],
        }
        df = pd.DataFrame(data)
        tf_cols = get_tf_columns(df)
        self.assertEqual(tf_cols, ["tf_0"])





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


class TestPeakBackgroundRegions(unittest.TestCase):
    """Tests for peak background extraction from merged wide output."""

    def test_get_peak_background_regions_merges_overlapping_windows(self):
        df = pd.DataFrame(
            {
                "gene": ["gene1", "gene1", "gene1", "gene1"],
                "sequence_type": ["reference", "reference", "max_mutated", "max_mutated"],
                "window_start": [0, 50, 0, 50],
                "window_end": [250, 300, 250, 300],
                "contains_padding": [False, False, False, False],
                "tf_0": [0.1, 0.5, 0.2, 0.6],
                "tf_0__reference__in_peak": [False, True, False, False],
                "tf_0__max_mutated__in_peak": [False, False, True, True],
                "tf_0__difference__in_peak": [False, False, False, True],
            }
        )

        regions = _get_peak_background_regions(df, "tf_0")

        self.assertIn("reference", regions)
        self.assertIn("max_mutated", regions)
        self.assertIn("difference", regions)
        self.assertEqual(regions["reference"], [(50.0, 300.0)])
        self.assertEqual(regions["max_mutated"], [(0.0, 300.0)])
        self.assertEqual(regions["difference"], [(50.0, 300.0)])

    def test_get_peak_background_regions_respects_requested_types(self):
        df = pd.DataFrame(
            {
                "gene": ["gene1"],
                "sequence_type": ["reference"],
                "window_start": [0],
                "window_end": [250],
                "contains_padding": [False],
                "tf_0": [0.1],
                "tf_0__reference__in_peak": [True],
            }
        )

        regions = _get_peak_background_regions(df, "tf_0", signal_types=["difference"])
        self.assertEqual(regions, {})


class TestPeakBackgroundRegionsFromPeaks(unittest.TestCase):
    """Tests for peak background extraction from raw peak-scanner output."""

    def test_get_peak_background_regions_from_peaks_merges_overlapping_peaks(self):
        peak_df = pd.DataFrame(
            {
                "gene": ["gene1", "gene1", "gene1", "gene1"],
                "tf": ["tf_0", "tf_0", "tf_0", "tf_0"],
                "signal_type": ["reference", "reference", "max_mutated", "difference"],
                "peak_start": [50, 80, 0, 60],
                "peak_end": [120, 300, 250, 300],
                "score": [0.1, 0.2, 0.3, 0.4],
                "region_idx": [0, 1, 0, 2],
                "peak_rank": [0, 1, 0, 0],
                "edge_peak": [False, False, False, False],
            }
        )

        regions = _get_peak_background_regions_from_peaks(peak_df, "gene1", "tf_0")

        self.assertIn("reference", regions)
        self.assertIn("max_mutated", regions)
        self.assertIn("difference", regions)
        self.assertEqual(regions["reference"], [(175.0, 425.0)])
        self.assertEqual(regions["max_mutated"], [(125.0, 375.0)])
        self.assertEqual(regions["difference"], [(185.0, 425.0)])

    def test_get_peak_background_regions_from_peaks_respects_requested_types(self):
        peak_df = pd.DataFrame(
            {
                "gene": ["gene1"],
                "tf": ["tf_0"],
                "signal_type": ["reference"],
                "peak_start": [0],
                "peak_end": [250],
                "score": [0.1],
                "region_idx": [0],
                "peak_rank": [0],
                "edge_peak": [False],
            }
        )

        regions = _get_peak_background_regions_from_peaks(
            peak_df,
            "gene1",
            "tf_0",
            signal_types=["difference"],
        )
        self.assertEqual(regions, {})


class TestLoadPeakResults(unittest.TestCase):
    """Tests for loading raw peak-scanner output."""

    def test_load_peak_results_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "peaks.csv")
            pd.DataFrame(
                {
                    "gene": ["gene1"],
                    "tf": ["tf_0"],
                    "signal_type": ["reference"],
                    "peak_start": [0],
                    "peak_end": [250],
                    "score": [0.1],
                    "region_idx": [0],
                    "peak_rank": [0],
                    "edge_peak": [False],
                }
            ).to_csv(csv_path, index=False)

            df = load_peak_results(csv_path)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "peak_start"], 0)

    def test_load_peak_results_validates_required_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "peaks.csv")
            pd.DataFrame(
                {
                    "gene": ["gene1"],
                    "tf": ["tf_0"],
                    "peak_start": [0],
                    "peak_end": [250],
                }
            ).to_csv(csv_path, index=False)

            with self.assertRaises(KeyError):
                load_peak_results(csv_path)


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
            _load_input_data(123)               #type:ignore

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
        result = _resolve_output_directory(pd.DataFrame(), None)
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
        self.sample_peak_data = pd.DataFrame({
            "gene": ["gene1", "gene1", "gene1"],
            "tf": ["tf_0", "tf_0", "tf_0"],
            "signal_type": ["reference", "max_mutated", "difference"],
            "peak_start": [0, 50, 100],
            "peak_end": [250, 300, 350],
            "score": [0.1, 0.2, 0.3],
            "region_idx": [0, 1, 2],
            "peak_rank": [0, 0, 0],
            "edge_peak": [False, False, False],
        })

    def test_visualize_from_file(self):
        """Test visualization from CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "scan.csv")
            self.sample_scan_data.to_csv(csv_path, index=False)
            peaks_path = os.path.join(tmpdir, "peaks.csv")
            pd.DataFrame(
                {
                    "gene": ["gene1", "gene1"],
                    "tf": ["tf_0", "tf_0"],
                    "signal_type": ["reference", "max_mutated"],
                    "peak_start": [0, 50],
                    "peak_end": [250, 300],
                    "score": [0.1, 0.2],
                    "region_idx": [0, 1],
                    "peak_rank": [0, 0],
                    "edge_peak": [False, False],
                }
            ).to_csv(peaks_path, index=False)

            visualize_scan_results(
                csv_path,
                genes=["gene1"],
                tfs=["tf_0"],
                output_dir=os.path.join(tmpdir, "output"),
                peak_input=peaks_path,
            )

            # Check file was created
            output_file = os.path.join(tmpdir, "output", "gene1", "gene1_TF_0.png")
            self.assertTrue(os.path.exists(output_file))

    def test_visualize_from_dataframe(self):
        """Test visualization from DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            peaks_df = pd.DataFrame(
                {
                    "gene": ["gene1"],
                    "tf": ["tf_0"],
                    "signal_type": ["reference"],
                    "peak_start": [0],
                    "peak_end": [250],
                    "score": [0.1],
                    "region_idx": [0],
                    "peak_rank": [0],
                    "edge_peak": [False],
                }
            )
            visualize_scan_results(
                self.sample_scan_data,
                genes=["gene1"],
                tfs=["tf_0"],
                output_dir=tmpdir,
                peak_input=peaks_df,
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
                peak_input=pd.DataFrame(
                    {
                        "gene": ["gene1"],
                        "tf": ["tf_0"],
                        "signal_type": ["reference"],
                        "peak_start": [0],
                        "peak_end": [250],
                        "score": [0.1],
                        "region_idx": [0],
                        "peak_rank": [0],
                        "edge_peak": [False],
                    }
                ),
            )

            # Should create plots in tmpdir/plots
            expected_file = os.path.join(tmpdir, "deepcis_scan_plots", "gene1", "gene1_TF_0.png")
            self.assertTrue(os.path.exists(expected_file))

    def test_visualize_separate_images_for_multiple_signal_types(self):
        """Test that multiple peak signal types are written as separate images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualize_scan_results(
                self.sample_scan_data,
                genes=["gene1"],
                tfs=["tf_0"],
                output_dir=tmpdir,
                highlight_peaks=True,
                peak_input=self.sample_peak_data,
            )

            base_dir = os.path.join(tmpdir, "gene1")
            expected_files = {
                os.path.join(base_dir, "gene1_TF_0_reference.png"),
                os.path.join(base_dir, "gene1_TF_0_max_mutated.png"),
                os.path.join(base_dir, "gene1_TF_0_difference.png"),
            }
            for expected_file in expected_files:
                self.assertTrue(os.path.exists(expected_file))

    def test_visualize_random_subset_uses_subset_and_separate_images(self):
        """Test random subset selection with multiple signal types."""
        data = pd.DataFrame({
            "gene": ["geneA", "geneA", "geneB", "geneB", "geneC", "geneC"],
            "sequence_type": ["reference", "max_mutated"] * 3,
            "window_start": [0, 0, 0, 0, 0, 0],
            "window_end": [250, 250, 250, 250, 250, 250],
            "contains_padding": [False, False, False, False, False, False],
            "tf_0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "tf_1": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "tf_2": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "tf_3": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        })
        peak_data = pd.DataFrame({
            "gene": ["geneA", "geneA", "geneA"],
            "tf": ["tf_0", "tf_0", "tf_0"],
            "signal_type": ["reference", "max_mutated", "difference"],
            "peak_start": [0, 50, 100],
            "peak_end": [250, 300, 350],
            "score": [0.1, 0.2, 0.3],
            "region_idx": [0, 1, 2],
            "peak_rank": [0, 0, 0],
            "edge_peak": [False, False, False],
        })

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "analysis.motives.deepcis_visualize.random.sample",
            side_effect=lambda values, k: list(values)[:k],
        ):
            visualize_scan_results(
                data,
                output_dir=tmpdir,
                highlight_peaks=True,
                peak_input=peak_data,
                random_subset=True,
            )

            for gene in ["geneA", "geneB", "geneC"][:3]:
                gene_dir = os.path.join(tmpdir, gene)
                self.assertTrue(os.path.isdir(gene_dir))

            expected_file = os.path.join(tmpdir, "geneA", "geneA_TF_0_reference.png")
            self.assertTrue(os.path.exists(expected_file))


class TestRandomSubsetHelper(unittest.TestCase):
    """Tests for random subset selection."""

    def test_select_random_subset_limits_to_three(self):
        with patch("analysis.motives.deepcis_visualize.random.sample", side_effect=lambda values, k: list(values)[:k]):
            genes, tfs = _select_random_subset(["gene1", "gene2", "gene3", "gene4"], ["tf_0", "tf_1", "tf_2", "tf_3"])

        self.assertEqual(genes, ["gene1", "gene2", "gene3"])
        self.assertEqual(tfs, ["tf_0", "tf_1", "tf_2"])

    def test_select_random_subset_handles_small_inputs(self):
        with patch("analysis.motives.deepcis_visualize.random.sample", side_effect=lambda values, k: list(values)[:k]):
            genes, tfs = _select_random_subset(["gene1"], ["tf_0", "tf_1"])

        self.assertEqual(genes, ["gene1"])
        self.assertEqual(tfs, ["tf_0", "tf_1"])


class TestParseArguments(unittest.TestCase):
    """Tests for the CLI parser."""

    def test_parse_arguments_defaults_include_background_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "scan.csv")
            pd.DataFrame(
                {
                    "gene": ["gene1"],
                    "sequence_type": ["reference"],
                    "window_start": [0],
                    "window_end": [250],
                    "contains_padding": [False],
                    "tf_0": [0.1],
                }
            ).to_csv(input_path, index=False)

            args = parse_arguments(["--input", input_path])

            self.assertTrue(args.highlight_padding)
            self.assertTrue(args.highlight_peaks)
            self.assertIsNone(args.peak_signals)

    def test_parse_arguments_peak_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "scan.csv")
            pd.DataFrame(
                {
                    "gene": ["gene1"],
                    "sequence_type": ["reference"],
                    "window_start": [0],
                    "window_end": [250],
                    "contains_padding": [False],
                    "tf_0": [0.1],
                }
            ).to_csv(input_path, index=False)

            args = parse_arguments([
                "--input",
                input_path,
                "--no-highlight-padding",
                "--no-highlight-peaks",
                "--peak-signals",
                "reference",
                "difference",
            ])

            self.assertFalse(args.highlight_padding)
            self.assertFalse(args.highlight_peaks)
            self.assertEqual(args.peak_signals, ["reference", "difference"])

    def test_parse_arguments_random_subset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "scan.csv")
            pd.DataFrame(
                {
                    "gene": ["gene1"],
                    "sequence_type": ["reference"],
                    "window_start": [0],
                    "window_end": [250],
                    "contains_padding": [False],
                    "tf_0": [0.1],
                }
            ).to_csv(input_path, index=False)

            args = parse_arguments(["--input", input_path, "--random-subset"])

            self.assertTrue(args.random_subset)


if __name__ == '__main__':
    unittest.main()
