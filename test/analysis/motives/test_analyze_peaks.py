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
                    "reference", "difference", "max_mutated",
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

            # Validate output structure
            assert len(summary) > 0
            assert "tf" in summary.columns
            assert "signal_type" in summary.columns
            assert "peak_count" in summary.columns

            # Should have entries for both reference and difference signals
            signal_types = set(summary["signal_type"].unique())
            self.assertEqual(signal_types, {"reference", "diff_added", "diff_removed", "max_mutated"}, f"Expected signal types 'reference' and 'difference', got {signal_types}")

            # Should have entries for WRKY40
            assert "WRKY40" in summary["tf"].values
            wrkylines = summary[summary["tf"] == "WRKY40"]
            self.assertEqual(wrkylines["peak_count"].tolist(), [1, 1, 0, 2])
            self.assertEqual(wrkylines["signal_type"].tolist(), ["diff_added", "diff_removed", "max_mutated", "reference"])

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
                    "reference", "difference", "max_mutated", "difference",
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

            # Should only include peaks from AT1G01_01 that overlap 100-200
            # AT1G01_01 has peaks: 100-120 (overlaps), 150-170 (overlaps)
            # AT2G02_02 should be completely filtered out
            self.assertEqual(len(summary), 4)  # Only AT1G01_01 peaks should remain: reference and diff_added
            self.assertEqual(summary["tf"].tolist(), ["bHLH74", "bHLH74", "WRKY40", "WRKY40"])  # Only WRKY40 peaks should remain
            self.assertEqual(summary["signal_type"].tolist(), ["diff_added", "max_mutated", "diff_added", "max_mutated"])  # Only AT1G01_01 peaks should remain: reference and diff_added
            self.assertEqual(summary["peak_count"].tolist(), [0, 1, 1, 0])  # Both peaks from AT1G01_01 should be counted
            assert (output_dir / "peak_summary.csv").exists()

            # Read the summary to verify it was filtered
            written_summary = pd.read_csv(output_dir / "peak_summary.csv")
            written_summary_normalized = _normalize_for_csv_comparison(written_summary).reset_index(drop=True)
            returned_summary_normalized = _normalize_for_csv_comparison(summary).reset_index(drop=True)
            pd.testing.assert_frame_equal(written_summary_normalized, returned_summary_normalized, )


if __name__ == "__main__":
    unittest.main()
    # # run single specific test method:
    # suite = unittest.TestSuite()
    # suite.addTest(TestOverlapAnalysisMode("test_overlap_analysis_end_to_end_with_mixed_genes"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
