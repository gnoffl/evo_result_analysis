"""Comprehensive tests for peak_scanner module."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from analysis.motives.peak_annotation import PeakAnnotator
from analysis.motives.peak_scanner import (
    DeepCISPeakScanner,
    _build_parser,
    _parse_list_arg,
    main,
)


class _PeakScannerTestBase(unittest.TestCase):
    """Shared fixtures for peak scanner tests."""

    def setUp(self):
        self.sample_df = pd.DataFrame(
            {
                "gene": ["geneA", "geneA", "geneA", "geneA", "geneB", "geneB"],
                "sequence_type": [
                    "reference",
                    "reference",
                    "optimized",
                    "optimized",
                    "reference",
                    "optimized",
                ],
                "window_start": [0, 10, 0, 10, 0, 0],
                "window_end": [20, 30, 20, 30, 20, 20],
                "contains_padding": [False, False, False, False, False, False],
                "tf_1": [0.1, 0.2, 0.4, 0.8, 0.5, 0.9],
                "tf_2": [0.3, 0.4, 0.6, 0.7, 0.2, 0.6],
            }
        )


class TestDeepCISPeakScannerInit(_PeakScannerTestBase):
    """Tests for DeepCISPeakScanner.__init__."""

    def test_init_defaults(self):
        scanner = DeepCISPeakScanner()
        self.assertEqual(scanner.signal_type, "difference")
        self.assertEqual(scanner.output_dir, Path("data") / "peak_annotations")
        self.assertEqual(scanner.annotator_window_size, 250)
        self.assertIsNone(scanner.annotator_step_size)
        self.assertEqual(scanner.annotator_threshold_peak, 0.2)
        self.assertEqual(scanner.annotator_sigma, 10.0)
        self.assertEqual(scanner.annotator_lambda_weight, 1.0)

    def test_init_custom_values(self):
        scanner = DeepCISPeakScanner(
            genes=["geneA"],
            tfs=["tf_1"],
            signal_type="reference",
            output_path="custom_dir/custom_file.end",
            annotator_window_size=100,
            annotator_step_size=10,
            annotator_threshold_peak=0.2,
            annotator_sigma=5.0,
            annotator_lambda_weight=0.5,
        )
        self.assertEqual(scanner.genes, ["geneA"])
        self.assertEqual(scanner.tfs, ["tf_1"])
        self.assertEqual(scanner.signal_type, "reference")
        self.assertEqual(scanner.output_path, Path("custom_dir") / "custom_file.end")
        self.assertEqual(scanner.output_dir, Path("custom_dir"))
        self.assertEqual(scanner.annotator_window_size, 100)
        self.assertEqual(scanner.annotator_step_size, 10)
        self.assertEqual(scanner.annotator_threshold_peak, 0.2)
        self.assertEqual(scanner.annotator_sigma, 5.0)
        self.assertEqual(scanner.annotator_lambda_weight, 0.5)

    def test_init_invalid_signal_type_raises(self):
        with self.assertRaises(ValueError):
            DeepCISPeakScanner(signal_type="mutated")


class TestLoadScannerData(_PeakScannerTestBase):
    """Tests for DeepCISPeakScanner._load_scanner_data."""

    def test_load_scanner_data_from_dataframe_returns_copy(self):
        loaded = DeepCISPeakScanner._load_scanner_data(self.sample_df)
        self.assertTrue(loaded.equals(self.sample_df))
        self.assertIsNot(loaded, self.sample_df)

    def test_load_scanner_data_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "scanner.csv"
            self.sample_df.to_csv(csv_path, index=False)

            loaded = DeepCISPeakScanner._load_scanner_data(csv_path)
            self.assertEqual(len(loaded), len(self.sample_df))
            self.assertEqual(list(loaded.columns), list(self.sample_df.columns))
            self.assertTrue(loaded.equals(self.sample_df))

    def test_load_scanner_data_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            DeepCISPeakScanner._load_scanner_data("/nonexistent/scanner.csv")


class TestValidateScannerInput(_PeakScannerTestBase):
    """Tests for DeepCISPeakScanner._validate_scanner_input."""

    def test_validate_scanner_input_success(self):
        scanner = DeepCISPeakScanner()
        scanner._validate_scanner_input(self.sample_df)

    def test_validate_scanner_input_missing_required_columns_raises(self):
        scanner = DeepCISPeakScanner()
        bad_df = self.sample_df.drop(columns=["sequence_type"])
        with self.assertRaises(KeyError):
            scanner._validate_scanner_input(bad_df)

    def test_validate_scanner_input_stale_max_mutated_raises(self):
        scanner = DeepCISPeakScanner()
        stale_df = self.sample_df.copy()
        stale_df["sequence_type"] = stale_df["sequence_type"].replace(
            "optimized", "max_mutated"
        )
        with self.assertRaises(ValueError):
            scanner._validate_scanner_input(stale_df)


class TestGetAvailableTfs(_PeakScannerTestBase):
    """Tests for DeepCISPeakScanner._get_available_tfs."""

    def test_get_available_tfs_excludes_metadata(self):
        tfs = DeepCISPeakScanner._get_available_tfs(self.sample_df)
        self.assertEqual(tfs, ["tf_1", "tf_2"])

    def test_get_available_tfs_without_optional_metadata_column(self):
        df = self.sample_df.drop(columns=["contains_padding"])
        tfs = DeepCISPeakScanner._get_available_tfs(df)
        self.assertEqual(tfs, ["tf_1", "tf_2"])
    
    def test_get_available_tfs_with_only_metadata_columns_returns_empty(self):
        metadata_df = self.sample_df.copy().drop(columns=["tf_1", "tf_2"])
        tfs = DeepCISPeakScanner._get_available_tfs(metadata_df)
        self.assertEqual(tfs, [])


class TestValidateGenes(_PeakScannerTestBase):
    """Tests for DeepCISPeakScanner.validate_genes."""

    def test_validate_genes_returns_all_when_none(self):
        scanner = DeepCISPeakScanner(genes=None)
        genes = scanner.validate_genes(self.sample_df)
        self.assertEqual(set(genes), {"geneA", "geneB"})

    def test_validate_genes_filters_and_keeps_order(self):
        scanner = DeepCISPeakScanner(genes=["geneB"])
        genes = scanner.validate_genes(self.sample_df)
        self.assertEqual(genes, ["geneB"])

    def test_validate_genes_missing_gene_raises(self):
        scanner = DeepCISPeakScanner(genes=["geneA", "missing"])
        with self.assertRaises(ValueError):
            scanner.validate_genes(self.sample_df)


class TestValidateTfs(_PeakScannerTestBase):
    """Tests for DeepCISPeakScanner.validate_tfs."""

    def test_validate_tfs_returns_all_when_none(self):
        scanner = DeepCISPeakScanner(tfs=None)
        selected = scanner.validate_tfs(self.sample_df, ["tf_1", "tf_2"])
        self.assertEqual(selected, ["tf_1", "tf_2"])

    def test_validate_tfs_returns_requested(self):
        scanner = DeepCISPeakScanner(tfs=["tf_2"])
        selected = scanner.validate_tfs(self.sample_df, ["tf_1", "tf_2"])
        self.assertEqual(selected, ["tf_2"])

    def test_validate_tfs_missing_tf_raises(self):
        scanner = DeepCISPeakScanner(tfs=["tf_3"])
        with self.assertRaises(ValueError):
            scanner.validate_tfs(self.sample_df, ["tf_1", "tf_2"])


class TestComputeDifferenceSignal(_PeakScannerTestBase):
    """Tests for DeepCISPeakScanner._compute_difference_signal."""

    def test_compute_difference_signal_missing_sequence_type_raises(self):
        scanner = DeepCISPeakScanner(signal_type="difference")
        only_ref: pd.DataFrame = self.sample_df[self.sample_df["sequence_type"] == "reference"] #type: ignore
        with self.assertRaises(ValueError):
            scanner._compute_difference_signal(only_ref, "tf_1")

    def test_compute_difference_signal_mismatched_counts_raises(self):
        scanner = DeepCISPeakScanner(signal_type="difference")
        # delete single line in df
        gene_a = self.sample_df[self.sample_df["gene"] == "geneA"]
        shortened_df = gene_a.drop(index=0)
        with self.assertRaises(ValueError):
            scanner._compute_difference_signal(shortened_df, "tf_1")
    
    def test_compute_difference_signal_zero_length_returns_none(self):
        scanner = DeepCISPeakScanner(signal_type="difference")
        empty_df: pd.DataFrame = self.sample_df.iloc[0:0] #type:ignore
        with self.assertRaises(ValueError):
            result = scanner._compute_difference_signal(empty_df, "tf_1")
    
    def test_compute_difference_signal_merge_with_different_window_order(self):
        scanner = DeepCISPeakScanner(signal_type="difference")
        gene_a_df = self.sample_df[self.sample_df["gene"] == "geneA"]
        gene_a_df = gene_a_df.sample(frac=1).reset_index(drop=True)

        result: pd.DataFrame = scanner._compute_difference_signal(gene_a_df, "tf_1") #type: ignore

        self.assertIsNotNone(result)
        self.assertEqual(list(result.columns), ["tf_1", "window_start", "window_end"])
        result = result.sort_values("window_start").reset_index(drop=True)
        self.assertAlmostEqual(result["tf_1"].tolist()[0], 0.3)
        self.assertAlmostEqual(result["tf_1"].tolist()[1], 0.6)

    def test_compute_difference_signal_success(self):
        scanner = DeepCISPeakScanner(signal_type="difference")
        gene_a_df: pd.DataFrame = self.sample_df[self.sample_df["gene"] == "geneA"] #type: ignore

        result = scanner._compute_difference_signal(gene_a_df, "tf_1")

        self.assertIsNotNone(result)
        if result is None:
            raise AssertionError("Expected non-None result")
        self.assertEqual(list(result.columns), ["tf_1", "window_start", "window_end"])
        self.assertAlmostEqual(result["tf_1"].tolist()[0], 0.3)
        self.assertAlmostEqual(result["tf_1"].tolist()[1], 0.6)

    def test_compute_difference_signal_mutated_smaller(self):
        scanner = DeepCISPeakScanner(signal_type="difference")
        gene_a_df: pd.DataFrame = self.sample_df[self.sample_df["gene"] == "geneA"] #type: ignore
        gene_a_df.iloc[2, gene_a_df.columns.get_loc("tf_1")] = 0.05
        gene_a_df.iloc[3, gene_a_df.columns.get_loc("tf_1")] = 0.05

        result = scanner._compute_difference_signal(gene_a_df, "tf_1")

        self.assertIsNotNone(result)
        if result is None:
            raise AssertionError("Expected non-None result")
        self.assertEqual(list(result.columns), ["tf_1", "window_start", "window_end"])
        self.assertAlmostEqual(result["tf_1"].tolist()[0], -0.05)
        self.assertAlmostEqual(result["tf_1"].tolist()[1], -0.15)
    
    def test_compute_difference_signal_negative_values(self):
        scanner = DeepCISPeakScanner(signal_type="difference")
        gene_a_df: pd.DataFrame = self.sample_df[self.sample_df["gene"] == "geneA"] #type: ignore
        gene_a_df.iloc[2, gene_a_df.columns.get_loc("tf_1")] = -0.1
        gene_a_df.iloc[3, gene_a_df.columns.get_loc("tf_1")] = -0.2

        result = scanner._compute_difference_signal(gene_a_df, "tf_1")

        self.assertIsNotNone(result)
        if result is None:
            raise AssertionError("Expected non-None result")
        self.assertEqual(list(result.columns), ["tf_1", "window_start", "window_end"])
        self.assertAlmostEqual(result["tf_1"].tolist()[0], -0.2)
        self.assertAlmostEqual(result["tf_1"].tolist()[1], -0.4)

    def test_compute_difference_signal_mismatched_windows_raises(self):
        scanner = DeepCISPeakScanner(signal_type="difference")
        #mismatch in window coordinates between reference and mutated for geneA
        gene_a_df = pd.DataFrame(
            {
                "gene": ["geneA", "geneA", "geneA", "geneA"],
                "sequence_type": [
                    "reference",
                    "reference",
                    "optimized",
                    "optimized",
                ],
                "window_start": [0, 10, 0, 20],
                "window_end": [20, 30, 20, 40],
                "contains_padding": [False, False, False, False],
                "tf_1": [0.1, 0.2, 0.4, 0.8],
                "tf_2": [0.3, 0.4, 0.6, 0.7],
            }
        )

        with self.assertRaises(ValueError):
            try:
                scanner._compute_difference_signal(gene_a_df, "tf_1")
            except ValueError as e:
                message = str(e)
                self.assertIn("Reference and mutated windows do not perfectly match for gene", message)
                raise


class TestComputeGeneTfSignal(_PeakScannerTestBase):
    """Tests for DeepCISPeakScanner._compute_gene_tf_signal."""

    def test_compute_gene_tf_signal_reference_success(self):
        scanner = DeepCISPeakScanner(signal_type="reference")
        gene_a_df: pd.DataFrame = self.sample_df[self.sample_df["gene"] == "geneA"] #type: ignore
        result = scanner._compute_gene_tf_signal(gene_a_df, "tf_1")
        self.assertIsNotNone(result)
        if result is None:
            raise AssertionError("Expected non-None result")
        self.assertEqual(list(result.columns), ["tf_1", "window_start", "window_end"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result["tf_1"].tolist(), [0.1, 0.2])

    def test_compute_gene_tf_signal_reference_missing_returns_none(self):
        scanner = DeepCISPeakScanner(signal_type="reference")
        only_mut: pd.DataFrame = self.sample_df[self.sample_df["sequence_type"] == "optimized"] #type: ignore
        with self.assertRaises(ValueError):
            result = scanner._compute_gene_tf_signal(only_mut, "tf_1")

    def test_compute_gene_tf_signal_optimized_success(self):
        scanner = DeepCISPeakScanner(signal_type="optimized")
        gene_a_df: pd.DataFrame = self.sample_df[self.sample_df["gene"] == "geneA"] #type: ignore
        result = scanner._compute_gene_tf_signal(gene_a_df, "tf_2")
        self.assertIsNotNone(result)
        if result is None:
            raise AssertionError("Expected non-None result")
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result.columns), ["tf_2", "window_start", "window_end"])
        self.assertEqual(result["tf_2"].tolist(), [0.6, 0.7])

    def test_compute_gene_tf_signal_difference_delegates(self):
        scanner = DeepCISPeakScanner(signal_type="difference")
        gene_a_df: pd.DataFrame = self.sample_df[self.sample_df["gene"] == "geneA"] #type: ignore
        expected = pd.DataFrame({"signal": [0.5], "window_start": [0], "window_end": [20]})
        with patch.object(scanner, "_compute_difference_signal", return_value=expected) as mock_diff:
            result = scanner._compute_gene_tf_signal(gene_a_df, "tf_1")
        mock_diff.assert_called_once()
        self.assertIs(result, expected)


class TestDetectPeaksForGeneTf(_PeakScannerTestBase):
    """Tests for DeepCISPeakScanner._detect_peaks_for_gene_tf."""

    def setUp(self):
        super().setUp()
        self.signal_df = pd.DataFrame(
            {
                "signal": [0.1, 0.5, 0.2],
                "window_start": [0, 10, 20],
                "window_end": [20, 30, 40],
            }
        )

    def test_detect_peaks_for_gene_tf_success(self):
        scanner = DeepCISPeakScanner(signal_type="difference")
        mock_peaks = pd.DataFrame(
            {
                "peak_start": [10],
                "peak_end": [30],
                "peak_middle_start": [15],
                "peak_middle_end": [25],
                "peak_area": [0.75],
                "region_idx": [0],
                "peak_rank": [0],
                "edge_peak": [True],
            }
        )

        with patch("analysis.motives.peak_scanner.PeakAnnotator") as mock_annotator_cls:
            mock_annotator = MagicMock()
            mock_annotator.detect_peaks.return_value = mock_peaks
            mock_annotator_cls.return_value = mock_annotator

            result = scanner._detect_peaks_for_gene_tf(self.signal_df, "geneA", "tf_1")

        self.assertIsNotNone(result)
        if result is None:
            raise AssertionError("Expected non-None result")
        self.assertEqual(
            set(result.columns),
            {"gene", "tf", "signal_type", "peak_start", "peak_end", "peak_area", "region_idx", "peak_rank", "edge_peak", "peak_middle_start", "peak_middle_end"},
        )
        self.assertEqual(result.iloc[0]["gene"], "geneA")
        self.assertEqual(result.iloc[0]["tf"], "tf_1")
        self.assertEqual(result.iloc[0]["signal_type"], "difference")
        self.assertEqual(result.iloc[0]["peak_start"], 10)
        self.assertEqual(result.iloc[0]["peak_end"], 30)
        self.assertEqual(result.iloc[0]["peak_area"], 0.75)
        self.assertEqual(result.iloc[0]["region_idx"], 0)
        self.assertEqual(result.iloc[0]["peak_rank"], 0)
        self.assertEqual(result.iloc[0]["edge_peak"], True)
        mock_annotator.detect_peaks.assert_called_once_with(signal_column="tf_1")
        # assert that the annotator was called with the correct inputs
        mock_annotator_cls.assert_called_once_with(
            df=self.signal_df,
            window_size=250,
            step_size=None,
            threshold_peak=0.2,
            sigma=10.0,
            lambda_weight=1.0,
        )


    def test_detect_peaks_for_gene_tf_no_peaks_returns_none(self):
        scanner = DeepCISPeakScanner()
        with patch("analysis.motives.peak_scanner.PeakAnnotator") as mock_annotator_cls:
            mock_annotator = MagicMock()
            mock_annotator.detect_peaks.return_value = pd.DataFrame(columns=["peak_start", "peak_end", "peak_area", "region_idx", "peak_rank", "edge_peak", "peak_middle_start", "peak_middle_end"])
            mock_annotator_cls.return_value = mock_annotator

            result = scanner._detect_peaks_for_gene_tf(self.signal_df, "geneA", "tf_1")
        if result is None:
            raise AssertionError("Expected empty DataFrame, got None")
        self.assertTrue(result.empty)


class TestSavePeaksResults(_PeakScannerTestBase):
    """Tests for DeepCISPeakScanner._save_peaks_results."""

    def test_save_peaks_results_with_file_input_uses_stem(self):
        scanner = DeepCISPeakScanner(signal_type="reference")
        result_df = pd.DataFrame(
            {
                "gene": ["geneA"],
                "tf": ["tf_1"],
                "signal_type": ["reference"],
                "peak_start": [0],
                "peak_end": [20],
                "score": [0.9],
                "region_idx": [0],
                "peak_rank": [1],
                "edge_peak": [False],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scanner.output_dir = Path(tmpdir)
            with patch("analysis.motives.peak_scanner.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20260101_010203"
                scanner._save_peaks_results(result_df, Path("data/input.csv"))

            expected_path = Path(tmpdir) / "input_annotated_peaks_reference_20260101_010203.csv"
            self.assertTrue(expected_path.exists())
            loaded = pd.read_csv(expected_path)
            self.assertEqual(len(loaded), 1)
            pd.testing.assert_frame_equal(loaded, result_df)

    def test_save_peaks_results_with_dataframe_input_uses_default_basename(self):
        scanner = DeepCISPeakScanner(signal_type="difference")
        result_df = pd.DataFrame(columns=["gene", "tf", "signal_type", "peak_start", "peak_end", "score", "region_idx", "peak_rank"])

        with tempfile.TemporaryDirectory() as tmpdir:
            scanner.output_dir = Path(tmpdir)
            with patch("analysis.motives.peak_scanner.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20260101_000000"
                scanner._save_peaks_results(result_df, self.sample_df, signal_types=[ "reference", "difference"])

            expected_path = Path(tmpdir) / "annotated_peaks_difference_reference_20260101_000000.csv"
            self.assertTrue(expected_path.exists())


class TestScanAllCombinations(_PeakScannerTestBase):
    """Tests for DeepCISPeakScanner._scan_all_combinations."""

    def test_scan_all_combinations_collects_only_non_none_peaks(self):
        scanner = DeepCISPeakScanner(signal_type="difference")

        signal_side_effects = [
            pd.DataFrame({"signal": [0.1], "window_start": [0], "window_end": [20]}),
            None,
            pd.DataFrame({"signal": [0.2], "window_start": [0], "window_end": [20]}),
            pd.DataFrame({"signal": [0.3], "window_start": [0], "window_end": [20]}),
        ]
        peaks_side_effects = [
            pd.DataFrame(
                {
                    "gene": ["geneA"],
                    "tf": ["tf_1"],
                    "signal_type": ["difference"],
                    "peak_start": [0],
                    "peak_end": [20],
                    "score": [0.8],
                    "region_idx": [0],
                    "peak_rank": [1],
                }
            ),
            pd.DataFrame(
                {
                    "gene": ["geneB"],
                    "tf": ["tf_1"],
                    "signal_type": ["difference"],
                    "peak_start": [5],
                    "peak_end": [25],
                    "score": [0.6],
                    "region_idx": [0],
                    "peak_rank": [1],
                }
            ),
            pd.DataFrame(columns=["gene", "tf", "signal_type", "peak_start", "peak_end", "score", "region_idx", "peak_rank"]),
        ]

        with patch.object(scanner, "_compute_gene_tf_signal", side_effect=signal_side_effects):
            with patch.object(scanner, "_detect_peaks_for_gene_tf", side_effect=peaks_side_effects):
                result = scanner._scan_all_combinations(
                    self.sample_df,
                    selected_genes=["geneA", "geneB"],
                    selected_tfs=["tf_1", "tf_2"],
                )
                actual_peak_calls_detect_peaks = scanner._detect_peaks_for_gene_tf.call_args_list       #type:ignore
                actual_calls_compute_signal = scanner._compute_gene_tf_signal.call_args_list            #type:ignore

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].iloc[0]["gene"], "geneA")
        self.assertEqual(result[1].iloc[0]["gene"], "geneB")
        # assert all combinations of gene and tf were used as input parameters for the respective methods
        expected_calls = [
            (self.sample_df[self.sample_df["gene"] == "geneA"].reset_index(drop=True), "tf_1"),
            (self.sample_df[self.sample_df["gene"] == "geneA"].reset_index(drop=True), "tf_2"),
            (self.sample_df[self.sample_df["gene"] == "geneB"].reset_index(drop=True), "tf_1"),
            (self.sample_df[self.sample_df["gene"] == "geneB"].reset_index(drop=True), "tf_2"),
        ]
        for curr_df, curr_tf in expected_calls:
            actual_call = actual_calls_compute_signal.pop(0)
            self.assertEqual(actual_call[0][1], curr_tf)
            pd.testing.assert_frame_equal(actual_call[0][0], curr_df)               #type:ignore
        
        # parameters for _detect_peaks_for_gene_tf only called for non-None signals
        expected_peak_calls = [
            (signal_side_effects[0], "geneA", "tf_1"),
            (signal_side_effects[2], "geneB", "tf_1"),
            (signal_side_effects[3], "geneB", "tf_2"),
        ]
        for curr_signal, curr_gene, curr_tf in expected_peak_calls:
            actual_call = actual_peak_calls_detect_peaks.pop(0)
            self.assertEqual(actual_call[0][1], curr_gene)
            self.assertEqual(actual_call[0][2], curr_tf)
            pd.testing.assert_frame_equal(actual_call[0][0], curr_signal)


class TestScan(_PeakScannerTestBase):
    """Tests for DeepCISPeakScanner.scan."""

    def test_scan_end_to_end_without_patching_detects_expected_peak(self):
        signal = [0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0, 0.0]
        window_start = np.arange(0, len(signal) * 10, 10)
        window_end = window_start + 30

        scanner_df = pd.DataFrame(
            {
                "gene": ["geneA"] * (2 * len(signal)),
                "sequence_type": ["reference"] * len(signal) + ["optimized"] * len(signal),
                "window_start": np.concatenate([window_start, window_start]),
                "window_end": np.concatenate([window_end, window_end]),
                "contains_padding": [False] * (2 * len(signal)),
                "tf_1": [0.0] * len(signal) + signal,
            }
        )

        scanner = DeepCISPeakScanner(
            genes=["geneA"],
            tfs=["tf_1"],
            signal_type="difference",
            annotator_window_size=30,
            annotator_threshold_peak=0.4,
            annotator_sigma=10.0,
            annotator_lambda_weight=1,
        )

        result = scanner.scan(scanner_df)

        expected = pd.DataFrame({
            "gene": ["geneA", "geneA"],
            "tf": ["tf_1", "tf_1"],
            "signal_type": ["difference", "difference"],
            "peak_start": [20, 50],
            "peak_end": [49, 79],
            "peak_middle_start": [30, 60],
            "peak_middle_end": [40, 70],
            "peak_area": [3.0, -3.0],
            "region_idx": [0, 1],
            "peak_rank": [0, 0],
            "edge_peak": [True, True],
        })

        self.assertFalse(result.empty)
        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_scan_returns_concatenated_results(self):
        scanner = DeepCISPeakScanner()
        peaks_1 = pd.DataFrame(
            {
                "gene": ["geneA"],
                "tf": ["tf_1"],
                "signal_type": ["difference"],
                "peak_start": [0],
                "peak_end": [20],
                "peak_area": [0.8],
                "region_idx": [0],
                "peak_rank": [1],
            }
        )
        peaks_2 = pd.DataFrame(
            {
                "gene": ["geneB"],
                "tf": ["tf_2"],
                "signal_type": ["difference"],
                "peak_start": [10],
                "peak_end": [30],
                "peak_area": [0.9],
                "region_idx": [0],
                "peak_rank": [1],
            }
        )

        with patch.object(scanner, "_load_scanner_data", return_value=self.sample_df):
            with patch.object(scanner, "_validate_scanner_input"):
                with patch.object(scanner, "_get_available_tfs", return_value=["tf_1", "tf_2"]):
                    with patch.object(scanner, "validate_genes", return_value=["geneA", "geneB"]):
                        with patch.object(scanner, "validate_tfs", return_value=["tf_1", "tf_2"]):
                            with patch.object(scanner, "_scan_all_combinations", return_value=[peaks_1, peaks_2]):
                                result = scanner.scan(self.sample_df)

        self.assertEqual(len(result), 2)
        self.assertEqual(list(result["gene"]), ["geneA", "geneB"])

    def test_scan_returns_empty_dataframe_with_standard_columns_when_no_peaks(self):
        scanner = DeepCISPeakScanner()

        with patch.object(scanner, "_load_scanner_data", return_value=self.sample_df):
            with patch.object(scanner, "_validate_scanner_input"):
                with patch.object(scanner, "_get_available_tfs", return_value=["tf_1", "tf_2"]):
                    with patch.object(scanner, "validate_genes", return_value=["geneA"]):
                        with patch.object(scanner, "validate_tfs", return_value=["tf_1"]):
                            with patch.object(scanner, "_scan_all_combinations", return_value=[]):
                                result = scanner.scan(self.sample_df)

        expected_cols = [
            "gene",
            "tf",
            "signal_type",
            "peak_start",
            "peak_end",
            "score",
            "region_idx",
            "peak_rank",
        ]
        self.assertEqual(list(result.columns), expected_cols)
        self.assertTrue(result.empty)

class TestDeepCISPeakScannerRepr(_PeakScannerTestBase):
    """Tests for DeepCISPeakScanner.__repr__."""

    def test_repr_contains_configuration(self):
        scanner = DeepCISPeakScanner(
            genes=["geneA"],
            tfs=["tf_1"],
            signal_type="reference",
            output_path="out",
            annotator_window_size=123,
            annotator_step_size=10,
            annotator_threshold_peak=0.11,
            annotator_sigma=3.5,
            annotator_lambda_weight=0.7,
        )
        rep = repr(scanner)
        self.assertIn("DeepCISPeakScanner", rep)
        self.assertIn("genes=['geneA']", rep)
        self.assertIn("tfs=['tf_1']", rep)
        self.assertIn("signal_type='reference'", rep)
        self.assertIn("annotator_window_size=123", rep)


class TestParseListArg(unittest.TestCase):
    """Tests for _parse_list_arg."""

    def test_parse_list_arg_none(self):
        self.assertIsNone(_parse_list_arg(None))

    def test_parse_list_arg_repeated_and_comma_separated(self):
        parsed = _parse_list_arg(["geneA,geneB", "geneC", " geneD , "])
        self.assertEqual(parsed, ["geneA", "geneB", "geneC", "geneD"])

    def test_parse_list_arg_empty_values_returns_none(self):
        parsed = _parse_list_arg(["", "   ", ",,", " , "])
        self.assertIsNone(parsed)


if __name__ == "__main__":
    # run single specific test method:
    # suite = unittest.TestSuite()
    # suite.addTest(TestComputeDifferenceSignal("test_compute_difference_signal_nan_after_join_raises"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
    unittest.main()