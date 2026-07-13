"""Integration tests for analyze_peaks.py analysis pipeline.

Tests the two main analysis modes end-to-end with minimal mocking:
1. Overlap analysis: WRKY peaks vs target mutation regions
2. Peak summarization: Peak counts and scores by gene and signal type
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.motives.analyze_peaks import (
    PARAMETERS_FILE_NAME,
    _load_peaks,
    analyze_wrky_peak_overlaps,
    compute_symmetric_limit,
    prepare_diff_calc_data,
    select_top_bottom,
    summarize_peaks,
    _parse_mutation_region,
)


def _normalize_for_csv_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame to match CSV-loaded format.
    
    Converts pd.NA to np.nan and casts numeric columns appropriately.
    """
    df = df.copy()
    
    # Replace pd.NA with np.nan
    df = df.where(pd.notna(df), np.nan)
    
    # Cast columns to match CSV dtypes
    for col in df.columns:
        if df[col].dtype == "object":
            # Check if column contains numeric or boolean data
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
                
            first_val = non_null.iloc[0]
            
            # Handle boolean columns
            if isinstance(first_val, (bool, np.bool_)):
                df[col] = df[col].astype("bool")
            # Handle integer columns (those without decimals)
            elif isinstance(first_val, (int, np.integer)):
                df[col] = pd.to_numeric(df[col], errors="ignore")
            # Handle float columns
            else:
                try:
                    float(first_val)
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except (ValueError, TypeError):
                    pass
    
    return df


class AnalyzePeaksIntegrationBase(unittest.TestCase):
    """Base class with shared test data generation."""

    def _create_simple_peaks_df(self):
        """Create a minimal peaks DataFrame with reference and differential signals."""
        return pd.DataFrame(
            {
                "gene": [
                    "AT1G01_01", "AT1G01_01", "AT1G01_01",
                    "AT2G02_02", "AT2G02_02",
                ],
                "tf": [
                    "WRKY40", "WRKY40", "bHLH74",
                    "WRKY40", "WRKY40",
                ],
                "peak_start": [100, 150, 200, 300, 350],
                "peak_end": [120, 170, 220, 320, 370],
                "peak_area": [0.5, 0.3, 0.7, 0.6, -0.4],
                "signal_type": [
                    "reference", "difference", "optimized",
                    "reference", "difference",
                ],
            }
        )

    def _write_gene_parameters(self, root: Path, gene_name: str, mutation_start: int, mutation_end: int) -> None:
        """Write a gene folder with parameters.json."""
        gene_dir = root / gene_name
        gene_dir.mkdir(parents=True, exist_ok=True)
        with open(gene_dir / PARAMETERS_FILE_NAME, "w", encoding="utf-8") as handle:
            json.dump({"mutation_start": mutation_start, "mutation_end": mutation_end}, handle)


class TestOverlapAnalysisMode(AnalyzePeaksIntegrationBase):
    """Integration tests for Mode 1: overlap analysis with run directory."""

    def test_overlap_analysis_end_to_end_with_mixed_genes(self):
        """Test full overlap analysis pipeline with genes that have and lack overlaps."""
        peaks_df = self._create_simple_peaks_df()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            output_dir = Path(tmpdir) / "output"

            # Gene with overlap: target 105-125 overlaps with peak 100-120
            self._write_gene_parameters(run_dir, "AT1G01_01", 105, 126)

            # Gene without overlap: target 1000-1020 (far beyond peaks)
            # Note: peaks extend by 250bp, so peak 350-370 extends to 620
            self._write_gene_parameters(run_dir, "AT2G02_02", 1000, 1021)

            # Run the analysis
            results, mapping = analyze_wrky_peak_overlaps(
                peaks=peaks_df,
                run_directory=run_dir,
                output_folder=output_dir,
                tf_substring="WRKY",
            )

            # Validate output structure
            assert len(results) == 2, f"Expected 2 results, got {len(results)}"
            assert "gene" in results.columns
            assert "target_start" in results.columns
            assert "overlaps_peak" in results.columns
            assert "matched_peak_start" in results.columns

            # Validate overlap result
            row_with_overlap = results.loc[results["gene"] == "AT1G01_01"].iloc[0]
            assert row_with_overlap["overlaps_peak"] == True
            assert row_with_overlap["matched_peak_start"] == 100
            assert row_with_overlap["matched_peak_end"] == 120

            # Validate no-overlap result
            row_no_overlap = results.loc[results["gene"] == "AT2G02_02"].iloc[0]
            assert row_no_overlap["overlaps_peak"] == False
            assert pd.isna(row_no_overlap["matched_peak_start"])
            assert pd.isna(row_no_overlap["matched_peak_end"])

            # Validate mapping output
            assert len(mapping) == 2
            assert "gene_name" in mapping.columns
            assert "target_start" in mapping.columns
            assert "target_end" in mapping.columns
            
            row_gene_1 = mapping.loc[mapping["gene_name"] == "01"].iloc[0]
            assert row_gene_1["gene_name"] == "01"
            assert row_gene_1["target_start"] == 105
            assert row_gene_1["target_end"] == 125

            # Validate files were written
            assert (output_dir / "reference.csv").exists()
            assert (output_dir / "reference_expected_peak_locations.csv").exists()
            loaded_res = pd.read_csv(output_dir / "reference.csv")
            loaded_mapping = pd.read_csv(output_dir / "reference_expected_peak_locations.csv")
            
            # Normalize results for comparison: convert pd.NA to np.nan and fix dtypes
            results_normalized = _normalize_for_csv_comparison(results)
            mapping_normalized = _normalize_for_csv_comparison(mapping)
            
            pd.testing.assert_frame_equal(loaded_res, results_normalized)
            pd.testing.assert_frame_equal(loaded_mapping, mapping_normalized)

    def test_overlap_analysis_filters_tf_correctly(self):
        """Test that TF filtering works correctly (WRKY only, not bHLH)."""
        peaks_df = pd.DataFrame(
            {
                "gene": ["AT1G01_01", "AT1G01_01", "AT1G01_01"],
                "tf": ["WRKY40", "bHLH74", "WRKY40"],
                "peak_start": [100, 150, 200],
                "peak_end": [120, 170, 220],
                "peak_area": [0.5, 0.3, 0.7],
                "signal_type": ["reference", "reference", "reference"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            output_dir = Path(tmpdir) / "output"

            # Gene with both WRKY and bHLH peaks
            self._write_gene_parameters(run_dir, "AT1G01_01", 100, 230)

            results, _ = analyze_wrky_peak_overlaps(
                peaks=peaks_df,
                run_directory=run_dir,
                output_folder=output_dir,
                tf_substring="WRKY",
            )

            row = results.iloc[0]
            # Should match WRKY peak, not bHLH peak
            assert row["matched_peak_tf"] == "WRKY40"
            # Should have 2 WRKY peaks in overlap count (100-120 and 200-220)
            assert row["overlap_count"] == 2

            assert "bHLH74" not in results["matched_peak_tf"].values
            assert all(results["matched_peak_tf"].dropna().str.contains("WRKY")), "All matched peaks should be WRKY"

    def test_overlap_analysis_with_multiple_overlaps_selects_best(self):
        """Test that when multiple peaks overlap, one is selected deterministically."""
        peaks_df = pd.DataFrame(
            {
                "gene": ["geneA", "geneA", "geneA", "geneA", "geneA"],
                "tf": ["WRKY40", "WRKY40", "WRKY40", "WRKY40", "WRKY40"],
                "peak_start": [0, 30, 100, 150, 200],
                "peak_end": [20, 50, 120, 170, 220],
                "peak_area": [0.1, 0.5, 0.9, 0.8, 0.7],
                "signal_type": ["reference", "reference", "reference", "reference", "reference"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            output_dir = Path(tmpdir) / "output"

            # Target range overlaps all three peaks
            self._write_gene_parameters(run_dir, "geneA", 110, 210)

            results, _ = analyze_wrky_peak_overlaps(
                peaks=peaks_df,
                run_directory=run_dir,
                output_folder=output_dir,
            )

            self.assertEqual(len(results), 1, f"Expected 1 result, got {len(results)}")

            row = results.iloc[0]
            # Should select one peak (the overlap selection logic picks the one with best metrics)
            # Verify that only one peak is selected
            assert row["overlaps_peak"] == True
            assert row["overlap_count"] == 5  # All five peaks overlap
            # The selected peak should be one of the five peaks
            assert row["matched_peak_start"] == 30
            assert row["matched_peak_end"] == 50


class TestLoadPeaks(unittest.TestCase):
    """Unit tests for loading and filtering peak tables."""

    def test_load_peaks_excludes_genes_from_json_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            peaks_path = tmp_path / "peaks.csv"
            excluded_path = tmp_path / "excluded_genes.json"

            peaks_df = pd.DataFrame(
                {
                    "gene": ["AT1G01_01", "AT2G02_02", "AT3G03_03"],
                    "tf": ["WRKY40", "WRKY40", "WRKY40"],
                    "peak_start": [100, 200, 300],
                    "peak_end": [120, 220, 320],
                    "peak_area": [0.5, 0.6, 0.7],
                }
            )
            peaks_df.to_csv(peaks_path, index=False)

            # _load_peaks filters by prefix before the last underscore groups.
            with open(excluded_path, "w", encoding="utf-8") as handle:
                json.dump(["AT2G02"], handle)

            filtered = _load_peaks(str(peaks_path), str(excluded_path))

            assert len(filtered) == 2
            assert "AT2G02_02" not in filtered["gene"].tolist()
            assert set(filtered["gene"].tolist()) == {"AT1G01_01", "AT3G03_03"}


class TestPeakSummarizationMode(AnalyzePeaksIntegrationBase):
    """Integration tests for Mode 2: peak summarization by gene and signal type."""

    def test_summarization_end_to_end_with_reference_and_diff_signals(self):
        """Test full summarization pipeline with mixed signal types."""
        peaks_df = self._create_simple_peaks_df()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Run summarization without expected peak locations
            summary = summarize_peaks(peaks_df, output_dir, pd.DataFrame())

            # Wide format: one row per TF, one column per signal type
            assert len(summary) > 0
            assert "tf" in summary.columns
            assert "diff_calc" in summary.columns
            assert "signal_type" not in summary.columns
            assert "peak_count" not in summary.columns

            # Signal types become columns
            for col in ("reference", "diff_added", "diff_removed", "optimized"):
                assert col in summary.columns, f"Expected column '{col}' in summary"

            # WRKY40: 2 reference peaks, 1 diff_added, 1 diff_removed, diff_calc=0
            assert "WRKY40" in summary["tf"].values
            wrky_row = summary[summary["tf"] == "WRKY40"].iloc[0]
            self.assertEqual(wrky_row["reference"], 2)
            self.assertEqual(wrky_row["diff_added"], 1)
            self.assertEqual(wrky_row["diff_removed"], 1)
            self.assertEqual(wrky_row["optimized"], 0)
            self.assertEqual(wrky_row["diff_calc"], -2)

            # Output is sorted by diff_calc descending
            self.assertTrue((summary["diff_calc"].diff().dropna() <= 0).all())

            # Validate file was written
            assert (output_dir / "peak_summary.csv").exists()

    def test_summarization_with_expected_peak_locations(self):
        """Test summarization that filters peaks to only those overlapping expected regions."""
        peaks_df = pd.DataFrame(
            {
                "gene": [
                    "AT1G01_01", "AT1G01_01", "AT1G01_01", "AT1G01_01",
                    "AT2G02_02", "AT2G02_02",
                ],
                "tf": [
                    "WRKY40", "WRKY40", "bHLH74", "bHLH74",
                    "WRKY40", "WRKY40",
                ],
                "peak_start": [100, 150, 200, 210, 350, 400],
                "peak_end": [120, 170, 220, 230, 370, 420],
                "peak_area": [0.5, 0.3, 0.7, 0.6, -0.4, 0.2],
                "signal_type": [
                    "reference", "difference", "optimized", "difference",
                    "reference", "difference",
                ],
            }
        )

        # Expected peak locations (mapping) - only genes with AT1G01 prefix
        expected_locations = pd.DataFrame(
            {
                "gene_name": ["01"],
                "target_start": [240],
                "target_end": [340],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Run summarization with expected peak locations
            summary = summarize_peaks(peaks_df, output_dir, expected_locations)

            # After filtering to target region [240, 340] for gene "01":
            #   AT1G01_01 WRKY40 150-170 (middle=285) → retained → diff_added (area=0.3>0)
            #   AT1G01_01 bHLH74 200-220 (middle=335) → retained → optimized
            # AT2G02_02 peaks fall outside the region and are excluded.
            # Wide format: one row per TF (WRKY40, bHLH74).
            self.assertEqual(len(summary), 2)
            self.assertEqual(summary["tf"].tolist(), ["bHLH74", "WRKY40"])  # sorted by diff_calc desc

            wrky_row = summary[summary["tf"] == "WRKY40"].iloc[0]
            self.assertEqual(wrky_row["diff_added"], 1)
            self.assertEqual(wrky_row["optimized"], 0)
            self.assertEqual(wrky_row["diff_calc"], 0)

            bhlh_row = summary[summary["tf"] == "bHLH74"].iloc[0]
            self.assertEqual(bhlh_row["diff_added"], 0)
            self.assertEqual(bhlh_row["optimized"], 1)
            self.assertEqual(bhlh_row["diff_calc"], 1)

            assert (output_dir / "peak_summary.csv").exists()

            # Read the summary to verify it was filtered
            written_summary = pd.read_csv(output_dir / "peak_summary.csv")
            written_summary_normalized = _normalize_for_csv_comparison(written_summary).reset_index(drop=True)
            returned_summary_normalized = _normalize_for_csv_comparison(summary).reset_index(drop=True)
            pd.testing.assert_frame_equal(written_summary_normalized, returned_summary_normalized, )


class TestPrepareDiffCalcData(unittest.TestCase):
    """Unit tests for the diff_calc plotting-data preparation."""

    def test_keeps_only_tf_and_diff_calc_sorted_descending(self):
        # Arrange
        summary = pd.DataFrame(
            {
                "tf": ["A_tnt", "B_tnt", "C_tnt"],
                "reference": [10, 5, 8],
                "optimized": [12, 3, 8],
                "diff_calc": [2, -2, 0],
            }
        )

        # Act
        plot_df = prepare_diff_calc_data(summary)

        # Assert
        self.assertEqual(list(plot_df.columns), ["tf", "diff_calc"])
        self.assertEqual(plot_df["tf"].tolist(), ["A_tnt", "C_tnt", "B_tnt"])
        self.assertEqual(plot_df["diff_calc"].tolist(), [2.0, 0.0, -2.0])
        # Index is reset to a clean range
        self.assertEqual(plot_df.index.tolist(), [0, 1, 2])

    def test_diff_calc_is_cast_to_float(self):
        # Arrange
        summary = pd.DataFrame({"tf": ["A_tnt"], "diff_calc": [3]})

        # Act
        plot_df = prepare_diff_calc_data(summary)

        # Assert
        self.assertEqual(plot_df["diff_calc"].dtype, np.dtype("float64"))

    def test_raises_when_required_columns_missing(self):
        # Arrange
        summary = pd.DataFrame({"tf": ["A_tnt"], "reference": [1]})

        # Act / Assert
        with self.assertRaises(KeyError):
            prepare_diff_calc_data(summary)


class TestComputeSymmetricLimit(unittest.TestCase):
    """Unit tests for the symmetric color/axis limit calculation."""

    def test_returns_max_absolute_value(self):
        # Arrange
        values = pd.Series([2.0, -5.0, 1.0])

        # Act
        limit = compute_symmetric_limit(values)

        # Assert
        self.assertEqual(limit, 5.0)

    def test_returns_one_when_all_zero(self):
        # Arrange
        values = np.array([0.0, 0.0])

        # Act
        limit = compute_symmetric_limit(values)

        # Assert
        self.assertEqual(limit, 1.0)

    def test_returns_one_when_empty(self):
        # Arrange
        values = np.array([])

        # Act
        limit = compute_symmetric_limit(values)

        # Assert
        self.assertEqual(limit, 1.0)

    def test_ignores_nan_values(self):
        # Arrange
        values = np.array([np.nan, -3.0, 2.0])

        # Act
        limit = compute_symmetric_limit(values)

        # Assert
        self.assertEqual(limit, 3.0)


class TestSelectTopBottom(unittest.TestCase):
    """Unit tests for select_top_bottom."""

    def _make_df(self, values: list) -> pd.DataFrame:
        """Return a descending-sorted diff_calc DataFrame matching prepare_diff_calc_data output."""
        sorted_values = sorted(values, reverse=True)
        return pd.DataFrame({"tf": [f"TF{i}" for i in range(len(sorted_values))], "diff_calc": sorted_values})

    def test_returns_top_and_bottom_n(self):
        # Arrange — 7 rows, n=2: expect rows 0,1 and 5,6
        df = self._make_df([10, 8, 6, 4, 2, -2, -4])

        # Act
        result = select_top_bottom(df, 2)

        # Assert
        self.assertEqual(len(result), 4)
        self.assertEqual(result["diff_calc"].tolist(), [10, 8, -2, -4])

    def test_preserves_descending_order(self):
        # Arrange
        df = self._make_df([5, 3, 1, -1, -3])

        # Act
        result = select_top_bottom(df, 2)

        # Assert — top 2 then bottom 2, descending throughout
        self.assertEqual(result["diff_calc"].tolist(), [5, 3, -1, -3])

    def test_returns_full_frame_when_rows_equal_twice_n(self):
        # Arrange — exactly 4 rows, n=2
        df = self._make_df([4, 2, -2, -4])

        # Act
        result = select_top_bottom(df, 2)

        # Assert
        self.assertEqual(len(result), 4)

    def test_returns_full_frame_when_fewer_rows_than_twice_n(self):
        # Arrange — 3 rows, n=3
        df = self._make_df([3, 1, -1])

        # Act
        result = select_top_bottom(df, 3)

        # Assert
        self.assertEqual(len(result), 3)

    def test_raises_on_non_positive_n(self):
        df = self._make_df([1, -1])
        with self.assertRaises(ValueError):
            select_top_bottom(df, 0)

    def test_resets_index(self):
        # Arrange
        df = self._make_df([10, 5, 1, -1, -5, -10])

        # Act
        result = select_top_bottom(df, 2)

        # Assert — clean 0-based index
        self.assertEqual(result.index.tolist(), [0, 1, 2, 3])



if __name__ == "__main__":
    unittest.main()
    # # run single specific test method:
    # suite = unittest.TestSuite()
    # suite.addTest(TestOverlapAnalysisMode("test_overlap_analysis_end_to_end_with_mixed_genes"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
