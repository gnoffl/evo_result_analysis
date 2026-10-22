"""Unit tests for the position-resolved species comparison.

The curves are synthetic, with peaks placed at known positions, so the peak
location and in-region measurements can be asserted exactly. The reused
per-window fitting in ``_common`` is mocked where the caching path is exercised.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import pandas as pd

from workflows.overlap_analysis.model_species_comparison import position_by_species
from workflows.overlap_analysis.model_species_comparison._models import SpeciesModel

ARA_MODEL = SpeciesModel("Ara", "NC_003070.9", "/models/ara_1.h5")
SLYC_MODEL = SpeciesModel("Slyc", "NC_015438.3", "/models/slyc_1.h5")


def _curve(species, chromosome, positions, correlations, count_fragments=None):
    """Build a tidy curve frame; the rolling column mirrors the raw one.

    ``count_fragments`` defaults to well-supported windows so tests that do not
    care about the support filter are unaffected by it.
    """
    if count_fragments is None:
        count_fragments = [800] * len(positions)
    return pd.DataFrame(
        {
            "species": species,
            "held_out_chromosome": chromosome,
            "position": positions,
            "correlation": correlations,
            "slope": [0.1] * len(positions),
            "p_value": [0.05] * len(positions),
            "correlation_rolling": correlations,
            "slope_rolling": [0.1] * len(positions),
            "p_value_rolling": [0.05] * len(positions),
            "count_fragments": count_fragments,
            "count_genes": [100] * len(positions),
        }
    )


class TestSummarizePeak(unittest.TestCase):
    def test_finds_peak_inside_the_region(self) -> None:
        # Arrange: the maximum sits at 900, inside the 800-1000 region.
        curve_df = _curve(
            "Ara", "NC_1", [700.0, 900.0, 1200.0], [0.05, 0.30, 0.10]
        )

        # Act
        summary = position_by_species.summarize_peak(curve_df, region=(800.0, 1000.0))

        # Assert
        self.assertEqual(summary["peak_position"], 900.0)
        self.assertAlmostEqual(summary["peak_correlation"], 0.30)
        self.assertTrue(summary["peak_in_region"])
        self.assertAlmostEqual(summary["max_correlation_in_region"], 0.30)
        self.assertEqual(summary["position_of_region_max"], 900.0)
        self.assertEqual(summary["count_windows_in_region"], 1)

    def test_flags_peak_outside_the_region_and_still_measures_inside(self) -> None:
        # Arrange: the global maximum is at 1500, outside the region; the best
        # in-region window is the weaker one at 850.
        curve_df = _curve(
            "Slyc", "NC_2", [850.0, 950.0, 1500.0], [0.12, 0.08, 0.40]
        )

        # Act
        summary = position_by_species.summarize_peak(curve_df, region=(800.0, 1000.0))

        # Assert
        self.assertEqual(summary["peak_position"], 1500.0)
        self.assertFalse(summary["peak_in_region"])
        self.assertAlmostEqual(summary["max_correlation_in_region"], 0.12)
        self.assertAlmostEqual(summary["mean_correlation_in_region"], 0.10)
        self.assertEqual(summary["count_windows_in_region"], 2)

    def test_region_bounds_are_inclusive(self) -> None:
        # Arrange: both windows sit exactly on the region boundaries.
        curve_df = _curve("Ara", "NC_1", [800.0, 1000.0], [0.20, 0.10])

        # Act
        summary = position_by_species.summarize_peak(curve_df, region=(800.0, 1000.0))

        # Assert
        self.assertEqual(summary["count_windows_in_region"], 2)
        self.assertAlmostEqual(summary["mean_correlation_in_region"], 0.15)

    def test_ignores_windows_without_a_correlation(self) -> None:
        # Arrange: one window was skipped for having too few points.
        curve_df = _curve("Ara", "NC_1", [850.0, 900.0], [0.20, None])

        # Act
        summary = position_by_species.summarize_peak(curve_df, region=(800.0, 1000.0))

        # Assert
        self.assertEqual(summary["count_windows"], 1)
        self.assertEqual(summary["peak_position"], 850.0)

    def test_raises_when_no_window_has_a_correlation(self) -> None:
        # Arrange
        curve_df = _curve("Ara", "NC_1", [850.0], [None])

        # Act / Assert
        with self.assertRaises(ValueError):
            position_by_species.summarize_peak(curve_df, region=(800.0, 1000.0))

    def test_raises_when_no_window_falls_in_the_region(self) -> None:
        # Arrange
        curve_df = _curve("Ara", "NC_1", [200.0, 2000.0], [0.2, 0.3])

        # Act / Assert
        with self.assertRaises(ValueError):
            position_by_species.summarize_peak(curve_df, region=(800.0, 1000.0))


class TestPeakSupportFilter(unittest.TestCase):
    """The sparse windows at the ends of the frame must not win the peak."""

    def test_sparse_edge_window_cannot_be_the_peak(self) -> None:
        # Arrange: the highest correlation sits in a 52-fragment edge window,
        # mimicking the real ~3010 window; the interior peak is at 927.
        curve_df = _curve(
            "Slyc", "NC_2",
            [927.0, 3010.0],
            [0.44, 0.76],
            count_fragments=[859, 52],
        )

        # Act
        summary = position_by_species.summarize_peak(
            curve_df, region=(800.0, 1000.0), min_support=200
        )

        # Assert: the reported peak is the supported interior one, while the
        # unrestricted peak still records the edge window.
        self.assertEqual(summary["peak_position"], 927.0)
        self.assertAlmostEqual(summary["peak_correlation"], 0.44)
        self.assertEqual(summary["count_fragments_at_peak"], 859)
        self.assertTrue(summary["peak_in_region"])
        self.assertEqual(summary["peak_position_any_support"], 3010.0)
        self.assertAlmostEqual(summary["peak_correlation_any_support"], 0.76)
        self.assertEqual(summary["count_windows_supported"], 1)

    def test_raises_when_no_window_meets_the_support_floor(self) -> None:
        # Arrange
        curve_df = _curve(
            "Ara", "NC_1", [850.0, 900.0], [0.2, 0.3], count_fragments=[10, 20]
        )

        # Act / Assert
        with self.assertRaises(ValueError) as context:
            position_by_species.summarize_peak(
                curve_df, region=(800.0, 1000.0), min_support=200
            )
        self.assertIn("200", str(context.exception))


class TestComputeWindowSupport(unittest.TestCase):
    def test_counts_fragments_and_genes_per_window(self) -> None:
        # Arrange: starts 0 and 10 fall in one 200 bp window together; 500 is
        # its own window. Genes repeat so the gene count differs from the
        # fragment count.
        enrichment_df = pd.DataFrame(
            {
                "group_overlap_start": [0, 10, 10, 500],
                "gene": ["g1", "g1", "g2", "g3"],
            }
        )

        # Act
        with patch.object(position_by_species._common, "BUCKET_SIZE", 200):
            support_df = position_by_species.compute_window_support(enrichment_df)

        # Assert: window centres are start + 100.
        self.assertEqual(list(support_df["position"]), [100.0, 110.0, 600.0])
        self.assertEqual(list(support_df["count_fragments"]), [3, 2, 1])
        self.assertEqual(list(support_df["count_genes"]), [2, 2, 1])


class TestBuildPeakSummary(unittest.TestCase):
    def test_one_row_per_model_sorted_by_species(self) -> None:
        # Arrange
        curves = [
            _curve("Slyc", "NC_2", [850.0, 1500.0], [0.12, 0.40]),
            _curve("Ara", "NC_1", [850.0, 1500.0], [0.30, 0.10]),
        ]

        # Act
        summary_df = position_by_species.build_peak_summary(
            curves, region=(800.0, 1000.0)
        )

        # Assert
        self.assertEqual(list(summary_df["species"]), ["Ara", "Slyc"])
        self.assertEqual(list(summary_df["peak_in_region"]), [True, False])
        self.assertEqual(list(summary_df["held_out_chromosome"]), ["NC_1", "NC_2"])


class TestPositionCurveForModel(unittest.TestCase):
    def test_reads_curve_cache_without_refitting(self) -> None:
        # Arrange: a curve cache already exists for this model.
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(position_by_species, "CACHE_DIR", temp_dir):
                cache_path = position_by_species.curve_cache_path_for(ARA_MODEL)
                _curve("Ara", "NC_003070.9", [900.0], [0.3]).to_csv(
                    cache_path, index=False
                )

                with patch.object(
                    position_by_species._common, "compute_correlation_by_position"
                ) as mock_compute:
                    # Act
                    curve_df = position_by_species.position_curve_for_model(ARA_MODEL)

                    # Assert
                    mock_compute.assert_not_called()
                    self.assertEqual(list(curve_df["position"]), [900.0])

    def test_raises_a_helpful_error_when_the_enrichment_cache_is_missing(self) -> None:
        # Arrange: neither the curve cache nor the enrichment cache exists.
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(position_by_species, "CACHE_DIR", temp_dir), \
                    patch.object(position_by_species, "enrichment_cache_path_for",
                                 return_value=os.path.join(temp_dir, "absent.csv")):
                # Act / Assert
                with self.assertRaises(FileNotFoundError) as context:
                    position_by_species.position_curve_for_model(SLYC_MODEL)
                self.assertIn("correlation_by_species", str(context.exception))

    def test_labels_the_computed_curve_and_writes_the_cache(self) -> None:
        # Arrange: the enrichment cache exists and the fit returns one window.
        fitted_curve = pd.DataFrame(
            {
                "position": [900.0],
                "correlation": [0.3],
                "slope": [0.1],
                "p_value": [0.05],
                "correlation_rolling": [0.3],
                "slope_rolling": [0.1],
                "p_value_rolling": [0.05],
                "label": ["Ara/NC_003070.9"],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            enrichment_path = os.path.join(temp_dir, "enrichment.csv")
            # start 800 gives window centre 900, matching the fitted curve, so
            # the support columns join onto it.
            pd.DataFrame(
                {
                    "group_overlap_start": [800, 800],
                    "gene": ["g1", "g2"],
                    "enrichment": [1.0, 2.0],
                }
            ).to_csv(enrichment_path, index=False)
            with patch.object(position_by_species, "CACHE_DIR", temp_dir), \
                    patch.object(position_by_species, "enrichment_cache_path_for",
                                 return_value=enrichment_path), \
                    patch.object(position_by_species._common,
                                 "compute_correlation_by_position",
                                 return_value=(None, None, None, fitted_curve)):
                # Act
                curve_df = position_by_species.position_curve_for_model(ARA_MODEL)

                # Assert: species labels added, plotting label dropped, support
                # columns joined on, and the curve cached.
                self.assertEqual(list(curve_df.columns)[:2],
                                 ["species", "held_out_chromosome"])
                self.assertNotIn("label", curve_df.columns)
                self.assertEqual(curve_df.iloc[0]["species"], "Ara")
                self.assertEqual(curve_df.iloc[0]["count_fragments"], 2)
                self.assertEqual(curve_df.iloc[0]["count_genes"], 2)
                self.assertTrue(
                    os.path.exists(position_by_species.curve_cache_path_for(ARA_MODEL))
                )


class TestPlots(unittest.TestCase):
    def setUp(self) -> None:
        self.all_curves_df = pd.concat(
            [
                _curve("Ara", "NC_1", [800.0, 900.0, 1000.0], [0.1, 0.3, 0.2]),
                _curve("Ara", "NC_2", [800.0, 900.0, 1000.0], [0.2, 0.4, 0.1]),
                _curve("Slyc", "NC_3", [800.0, 900.0, 1000.0], [0.0, 0.1, 0.05]),
                _curve("Slyc", "NC_4", [800.0, 900.0, 1000.0], [0.05, 0.15, 0.0]),
            ],
            ignore_index=True,
        )

    def test_curve_plot_writes_an_image(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "nested", "curves.png")

            # Act
            position_by_species.plot_position_curves(self.all_curves_df, output_path)

            # Assert
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_peak_plot_writes_an_image(self) -> None:
        # Arrange
        curves = [
            group for _, group in self.all_curves_df.groupby(
                ["species", "held_out_chromosome"]
            )
        ]
        summary_df = position_by_species.build_peak_summary(curves)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "nested", "peaks.png")

            # Act
            position_by_species.plot_peak_summary(summary_df, output_path)

            # Assert
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)


if __name__ == "__main__":
    unittest.main()
