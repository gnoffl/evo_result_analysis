import unittest

import numpy as np
import pandas as pd

from analysis.motives.peak_annotation import PeakAnnotator


class TestPeakAnnotator(unittest.TestCase):
    def _make_df(self, signal, step_size=10):
        return pd.DataFrame(
            {
                "window_start": np.arange(0, len(signal) * step_size, step_size),
                "signal": signal,
            }
        )

    # ===== constructor and validation =====

    def test_infer_step_size_success(self):
        df = self._make_df([0.0, 1.0, 2.0], step_size=20)
        annotator = PeakAnnotator(threshold_peak=0, df=df)
        self.assertEqual(annotator.step_size, 20)

    def test_infer_step_size_with_explicit_step_size(self):
        df = self._make_df([0.0, 1.0, 2.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, step_size=10)
        self.assertEqual(annotator.step_size, 10)

    def test_infer_step_size_errors(self):
        valid = PeakAnnotator(threshold_peak=0, df=self._make_df([0.0, 1.0]))

        with self.assertRaises(ValueError):
            valid._infer_step_size_from_dataframe(pd.DataFrame({"signal": [0.0, 1.0]}))

        with self.assertRaises(ValueError):
            valid._infer_step_size_from_dataframe(pd.DataFrame({"window_start": [10], "signal": [1.0]}))

        with self.assertRaises(ValueError):
            valid._infer_step_size_from_dataframe(
                pd.DataFrame({"window_start": [10, 10], "signal": [1.0, 2.0]})
            )

    def test_validate_signal_column_success_and_failures(self):
        df = self._make_df([1.0, 2.0, 3.0])
        annotator = PeakAnnotator(threshold_peak=0, df=df)
        annotator.validate_signal_column("signal")

        with self.assertRaises(ValueError):
            annotator.validate_signal_column("missing")

        bad_df = pd.DataFrame({"window_start": [0, 10, 20], "signal": ["a", "b", "c"]})
        bad_annotator = PeakAnnotator(threshold_peak=0, df=bad_df)
        with self.assertRaises(ValueError):
            bad_annotator.validate_signal_column("signal")

        annotator.df = [1, 2, 3]                        #type:ignore
        with self.assertRaises(TypeError):
            annotator.validate_signal_column("signal")

    def test_invalid_parameters_raise(self):
        df = self._make_df([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            PeakAnnotator(threshold_peak=0, df=df, window_size=0)
        with self.assertRaises(ValueError):
            PeakAnnotator(threshold_peak=0, df=df, sigma=0.0)
        with self.assertRaises(ValueError):
            PeakAnnotator(threshold_peak=0, df=df, lambda_weight=-0.1)

    def test_validate_inputs_direct_method(self):
        df = self._make_df([0.0, 1.0, 2.0])
        annotator = PeakAnnotator(threshold_peak=0, df=df)

        # Success path: should not raise.
        annotator._validate_inputs()

        # Explicit method-level validation branches.
        annotator.window_size = 0
        with self.assertRaises(ValueError):
            annotator._validate_inputs()

        annotator.window_size = 10
        annotator.sigma = 0.0
        with self.assertRaises(ValueError):
            annotator._validate_inputs()

        annotator.sigma = 1.0
        annotator.lambda_weight = -0.01
        with self.assertRaises(ValueError):
            annotator._validate_inputs()

    def test_validate_uniform_step_size_cases(self):
        inconsistent = pd.DataFrame(
            {"window_start": [0, 10, 25, 35], "signal": [0.0, 0.1, 0.2, 0.3]}
        )
        with self.assertRaises(ValueError):
            PeakAnnotator(threshold_peak=0, df=inconsistent)

        mismatch = self._make_df([0.0, 1.0, 2.0], step_size=10)
        with self.assertRaises(ValueError):
            PeakAnnotator(threshold_peak=0, df=mismatch, step_size=20)

        annotator = PeakAnnotator(threshold_peak=0, df=self._make_df([0.0, 1.0]))
        annotator.df = pd.DataFrame({"signal": [0.0, 1.0]})
        with self.assertRaises(ValueError):
            annotator._validate_uniform_step_size()

    def test_validate_uniform_step_size_single_row_is_allowed(self):
        one_row = pd.DataFrame({"window_start": [0], "signal": [0.2]})
        annotator = PeakAnnotator(threshold_peak=0, df=one_row, step_size=10)
        # Single-row input has no spacing to validate and should pass.
        annotator._validate_uniform_step_size()

    # ===== part A helpers =====

    def test_compute_moving_average_variants(self):
        signal = np.array([0.0, 1.0, 3.0, 1.0, 0.0], dtype=np.float64)
        annotator = PeakAnnotator(threshold_peak=0, df=self._make_df(signal, step_size=10), window_size=30)
        avg = annotator._compute_moving_average(signal)
        np.testing.assert_allclose(avg, np.array([4.0 / 3.0, 5.0 / 3.0, 4.0 / 3.0]))

        annotator = PeakAnnotator(threshold_peak=0, df=self._make_df(signal, step_size=10), window_size=10)
        same = annotator._compute_moving_average(signal)
        np.testing.assert_allclose(same, signal)

    def test_threshold_mask_edge(self):
        annotator = PeakAnnotator(threshold_peak=0.5, df=self._make_df([0.0, 1.0]), step_size=10)
        mask = annotator._threshold_mask(np.array([0.5, 0.5001, 0.49]))
        np.testing.assert_array_equal(mask, np.array([False, True, False]))

    def test_find_contiguous_regions_edges(self):
        self.assertEqual(PeakAnnotator._find_contiguous_regions(np.array([], dtype=bool)), [])
        self.assertEqual(PeakAnnotator._find_contiguous_regions(np.array([True, True, False])), [(0, 2)])
        self.assertEqual(PeakAnnotator._find_contiguous_regions(np.array([False, True, True])), [(1, 3)])
        self.assertEqual(PeakAnnotator._find_contiguous_regions(np.array([True, True, True])), [(0, 3)])
        self.assertEqual(PeakAnnotator._find_contiguous_regions(np.array([False, False, False])), [])
        self.assertEqual(
            PeakAnnotator._find_contiguous_regions(np.array([True, False, True, True, False])),
            [(0, 1), (2, 4)],
        )

    def test_detect_signal_regions_present_and_absent(self):
        high_signal_df = self._make_df([0.0, 0.1, 0.5, 0.3, 0.7, 0.2], step_size=10)
        annotator = PeakAnnotator(df=high_signal_df, window_size=30, threshold_peak=0.35)
        # weight should look like [0.2, 0.3, 0.5, 0.4]
        # mask should look like [false, false, true, true]
        # regions should be [(2, 4)]
        regions = annotator._detect_signal_regions(high_signal_df["signal"].to_numpy(dtype=np.float64))
        self.assertGreaterEqual(len(regions), 1)
        self.assertEqual(regions[0], (2, 4))

        zero_df = self._make_df([0.0, 0.0, 0.0, 0.0], step_size=10)
        annotator_zero = PeakAnnotator(df=zero_df, window_size=20, threshold_peak=10.0)
        self.assertEqual(
            annotator_zero._detect_signal_regions(zero_df["signal"].to_numpy(dtype=np.float64)),
            [],
        )

    def test_detect_signal_regions_returns_empty_when_window_exceeds_signal(self):
        df = self._make_df([0.1, 0.2], step_size=10)
        annotator = PeakAnnotator(df=df, window_size=100, threshold_peak=10.0)
        regions = annotator._detect_signal_regions(df["signal"].to_numpy(dtype=np.float64))
        self.assertEqual(regions, [])

    # ===== part B helpers =====

    def test_gaussian_smooth(self):
        signal = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
        annotator = PeakAnnotator(threshold_peak=0, df=self._make_df(signal, step_size=10), window_size=30, sigma=10)
        smooth = annotator._gaussian_smooth(signal)
        self.assertEqual(len(smooth), len(signal))
        np.testing.assert_allclose(smooth, np.array([2.0, 2.0, 2.0, 2.0]), atol=1e-6)

    def test_compute_forward_derivative(self):
        deriv = PeakAnnotator._compute_forward_derivative(np.array([1.0, 3.0, 2.0], dtype=np.float64))
        np.testing.assert_allclose(deriv, np.array([2.0, -1.0]))
        deriv = PeakAnnotator._compute_forward_derivative(np.array([0.0, 1.0, 3.0, 2.0, 0.0], dtype=np.float64))
        np.testing.assert_allclose(deriv, np.array([1.0, 2.0, -1.0, -2.]))

    def test_preprocess_signal_for_peaks_shapes(self):
        df = self._make_df([0.0, 1.0, 2.0, 1.0, 0.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=30, sigma=10.0)
        signal = df["signal"].to_numpy(dtype=np.float64)
        deriv = annotator._calculate_smooth_derivative(signal)

        self.assertEqual(len(deriv), len(signal) + 1)
        self.assertTrue(all(derivative >= 0 for derivative in deriv[:3]))
        self.assertTrue(all(derivative <= 0 for derivative in deriv[3:]))

    # ===== part C helpers =====

    def test_distribute_peaks_over_area(self):
        df = self._make_df([0.0, 1.0, 2.0, 1.0, 0.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=30)

        peaks = annotator.distribute_peaks_over_area(start=0, finish=20, num_peaks=1)
        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0][0], 0)
        self.assertEqual(peaks[0][1], 3)

        peaks = annotator.distribute_peaks_over_area(start=0, finish=2, num_peaks=1)
        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0][0], 0)
        self.assertEqual(peaks[0][1], 2)

        peaks = annotator.distribute_peaks_over_area(start=0, finish=9, num_peaks=3)
        self.assertEqual(len(peaks), 3)
        self.assertEqual(peaks[0][0], 0)
        self.assertEqual(peaks[0][1], 3)
        self.assertEqual(peaks[1][0], 3)
        self.assertEqual(peaks[1][1], 6)
        self.assertEqual(peaks[2][0], 6)
        self.assertEqual(peaks[2][1], 9)

        peaks = annotator.distribute_peaks_over_area(start=0, finish=8, num_peaks=3)
        self.assertEqual(len(peaks), 3)
        self.assertEqual(peaks[0][0], 0)
        self.assertEqual(peaks[0][1], 3)
        self.assertEqual(peaks[1][0], 3)
        self.assertEqual(peaks[1][1], 6)
        self.assertEqual(peaks[2][0], 5)
        self.assertEqual(peaks[2][1], 8)

        peaks = annotator.distribute_peaks_over_area(start=2, finish=10, num_peaks=3)
        self.assertEqual(len(peaks), 3)
        self.assertEqual(peaks, [(2, 5), (5, 8), (7, 10)])

        self.assertEqual(annotator.distribute_peaks_over_area(5, 5, 1), [])
        self.assertEqual(annotator.distribute_peaks_over_area(2, 10, 0), [])

        self.assertEqual(annotator.distribute_peaks_over_area(start=10, finish=2, num_peaks=3), [])
        self.assertEqual(annotator.distribute_peaks_over_area(start=2, finish=10, num_peaks=-1), [])


    
    def test_get_cumsum_signal(self):
        df = self._make_df([1.0, 2.0, 3.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=20)
        cumsum = annotator._get_cumsum_signal(df["signal"].to_numpy(dtype=np.float64))
        np.testing.assert_allclose(cumsum, np.array([0.0, 1.0, 3.0, 6.0, 6.0]), atol=1e-6)

    def test_get_middle_window_step_10(self):
        df = self._make_df([0.0, 1.0, 0.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=30)

        # Peak [20, 49] is centered at 35; middle window should be [30, 40].
        self.assertEqual(annotator.get_middle_window(20, 49), (30, 40))

        # Peak [25, 54] is centered at 40; middle window should be [35, 45].
        self.assertEqual(annotator.get_middle_window(25, 54), (35, 45))

        # Peak [10, 39] is centered at 25; the returned 10bp window should be [20, 30].
        self.assertEqual(annotator.get_middle_window(10, 39), (20, 30))

    def test_get_middle_window_with_larger_step_size(self):
        df = self._make_df([0.0, 1.0, 0.0], step_size=20)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=30)

        # For a 20bp step and peak [40, 99], middle window should be [60, 80].
        self.assertEqual(annotator.get_middle_window(40, 99), (60, 80))
    
    def test_calculate_reduced_mass_score(self):
        df = self._make_df([0.1, 0.2, 0.5, 0.6], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0.2, df=df, window_size=20)
        signal_cum_sum = annotator._get_cumsum_signal(df["signal"].to_numpy(dtype=np.float64))
        # l=0, r=2 -> (0.1 + 0.2) - (2 * 0.2) = 0.3 - 0.4 = -0.1
        self.assertAlmostEqual(annotator._calculate_reduced_mass_score(l=0, r=2, signal_cum_sum=signal_cum_sum), -0.1,)
        # l=1, r=3 -> (0.2 + 0.5) - (2 * 0.2) = 0.7 - 0.4 = 0.3
        self.assertAlmostEqual( annotator._calculate_reduced_mass_score(l=1, r=3, signal_cum_sum=signal_cum_sum), 0.3,)
        self.assertAlmostEqual( annotator._calculate_reduced_mass_score(l=0, r=4, signal_cum_sum=signal_cum_sum), 0.6,)
        self.assertAlmostEqual( annotator._calculate_reduced_mass_score(l=0, r=1, signal_cum_sum=signal_cum_sum), -0.1,)

    def test_calculate_mass_contributions(self):
        """Test basic mass contribution calculation with simple cumulative sum."""
        df = self._make_df([0.1, 0.2, 0.5, 0.6], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0.2, df=df, window_size=20)
        
        # Cumulative sum of [0.1, 0.2, 0.5, 0.6] is [0, 0.1, 0.3, 0.8, 1.4, 1.4] (with prepended 0)
        signal_cum_sum = annotator._get_cumsum_signal(df["signal"].to_numpy(dtype=np.float64))
        np.testing.assert_allclose(signal_cum_sum, np.array([0.0, 0.1, 0.3, 0.8, 1.4, 1.4]), atol=1e-6)
        
        # Test scan_range: region_start=0, region_end=2, left_end=1, right_end=3
        scan_range = (0, 2, 1, 3)
        mass_contributions = annotator.calculate_mass_contributions(signal_cum_sum, scan_range)
        
        # left_elements = 1 - 0 + 1 = 2 (indices 0, 1)
        # right_elements = 3 - 2 + 1 = 2 (indices 1, 2)
        # total = 2 * 2 = 4
        self.assertEqual(mass_contributions.shape, (2, 2))
        expected_values = [
            [
                # l=0, r=2: mass_term = cumsum[2] - cumsum[0] = 0.3 - 0 = 0.3, reduced = 0.3 - (2-0)*0.2 = -.1
                -0.1,
                #l=0, r=3: mass_term = cumsum[3] - cumsum[0] = 0.8 - 0 = 0.8, reduced = 0.8 - (3-0)*0.2 = 0.2
                0.2,
            ],
            [
                # l=1, r=2: mass_term = cumsum[2] - cumsum[1] = 0.3 - 0.1 = 0.2, reduced = 0.2 - (2-1)*0.2 = 0.0
                0.0,
                # l=1, r=3: mass_term = cumsum[3] - cumsum[1] = 0.8 - 0.1 = 0.7, reduced = 0.7 - (3-1)*0.2 = 0.3
                0.3,
            ]
        ]
        expected_values = np.array(expected_values)
        np.testing.assert_allclose(mass_contributions, expected_values, atol=1e-6)

    def test_calculate_scan_range(self):
        df = self._make_df([0.1, 0.2, 0.5, 0.6], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0.2, df=df, window_size=20)
        deriv = np.array([0.0, 0.1, 0.2, -0.1, -0.2], dtype=np.float64)

        # window_size=20, step_size=10 -> _window_size_elements=2
        self.assertEqual(
            annotator.calculate_scan_range(region_start=0, region_end=2, deriv=deriv),
            (0, 2, 1, 4),
        )

        # region_end is clipped to len(deriv)-1, and left/right bounds are clipped as well.
        self.assertEqual(
            annotator.calculate_scan_range(region_start=2, region_end=99, deriv=deriv),
            (2, 4, 3, 4),
        )

    def test_calculate_peak_score_matches_formula(self):
        df = self._make_df([0.1, 0.5, 1.0, 0.4], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=20, lambda_weight=1.0)

        deriv = np.array([0.1, 0.4, 0.5, -0.6, -0.4], dtype=np.float64)
        signal_cum_sum = annotator._get_cumsum_signal(df["signal"].to_numpy(dtype=np.float64))
        add_args = {
            "deriv": deriv,
            "mass_norm_term": 1,
            "deriv_norm_term": 1,
        }

        score = annotator._calculate_peak_score(
            peak_start_idx=1,
            peak_end_idx=3,
            reduced_mass_score=annotator._calculate_reduced_mass_score(l=1, r=3, signal_cum_sum=signal_cum_sum),
            **add_args
        )

        # lambda (=1.0) * sum of two middle values (left index included, right index excluded) divided by normalization term (1)
        expected_mass_term = 1.5
        # derivative at left index (0.5) - derivative at right index (-0.6) + mass term (1.5)
        # derivatives should be including left edge (difference getting TO first index and FROM second index)
        expected_score = 0.4 + 0.6 + expected_mass_term
        self.assertAlmostEqual(score, expected_score)

        score = annotator._calculate_peak_score(
            peak_start_idx=0,
            peak_end_idx=2,
            reduced_mass_score=annotator._calculate_reduced_mass_score(l=0, r=2, signal_cum_sum=signal_cum_sum),
            **add_args
            
        )
        expected_mass = 0.6
        expected_score = 0.1 - 0.5 + expected_mass
        self.assertAlmostEqual(score, expected_score)

        score = annotator._calculate_peak_score(
            peak_start_idx=1,
            peak_end_idx=1,
            reduced_mass_score=annotator._calculate_reduced_mass_score(l=1, r=1, signal_cum_sum=signal_cum_sum),
            **add_args
        )
        expected_score = float('-inf')
        self.assertEqual(score, expected_score)

        score = annotator._calculate_peak_score(
            peak_start_idx=2,
            peak_end_idx=4,
            reduced_mass_score=annotator._calculate_reduced_mass_score(l=2, r=4, signal_cum_sum=signal_cum_sum),
            **add_args
        )
        expected_mass = 1.4
        expected_score = 0.5 + 0.4 + expected_mass
        self.assertAlmostEqual(score, expected_score)

    def test_calculate_peak_score_lambda_weight_affects_mass_contribution(self):
        df = self._make_df([0.0, 0.5, 1.0, 0.5], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=20)
        deriv = np.array([0.0, 0.5, 0.5, -0.5, -.5], dtype=np.float64)
        signal_cum_sum = annotator._get_cumsum_signal(df["signal"].to_numpy(dtype=np.float64))

        no_mass = PeakAnnotator(threshold_peak=0, df=df, window_size=20, lambda_weight=0.0)
        with_mass = PeakAnnotator(threshold_peak=0, df=df, window_size=20, lambda_weight=3.0)
    
        reduced_mass = annotator._calculate_reduced_mass_score(l=1, r=3, signal_cum_sum=signal_cum_sum)

        score_no_mass = no_mass._calculate_peak_score(
            peak_start_idx=1,
            peak_end_idx=3,
            deriv=deriv,
            mass_norm_term=1,
            deriv_norm_term=1,
            reduced_mass_score=reduced_mass,
        )
        score_with_mass = with_mass._calculate_peak_score(
            peak_start_idx=1,
            peak_end_idx=3,
            deriv=deriv,
            mass_norm_term=1,
            deriv_norm_term=1,
            reduced_mass_score=reduced_mass,
        )

        expected_mass_term = (signal_cum_sum[3] - signal_cum_sum[1])
        self.assertAlmostEqual(
            score_with_mass - score_no_mass,
            (with_mass.lambda_weight - no_mass.lambda_weight) * expected_mass_term,
        )

    def test_find_multi_peak_edges_fallback_when_no_candidates(self):
        df = self._make_df([0.0, 1.0, 2.0, 1.0, 0.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=20)
        cumsum = np.cumsum(df["signal"].to_numpy(dtype=np.float64))

        deriv = annotator._calculate_smooth_derivative(df["signal"].to_numpy(dtype=np.float64))
        l_idx, r_idx, _, _ = annotator._find_multi_peak_edges(
            region_start=1, region_end=2, deriv=deriv, signal_cum_sum=cumsum
        )

        self.assertGreaterEqual(l_idx, 0)
        self.assertGreaterEqual(r_idx, l_idx + 1)

    def test_find_multi_peak_edges_clean_cut(self):
        df = self._make_df([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=30, lambda_weight=1, sigma=10.0)

        deriv = annotator._calculate_smooth_derivative(df["signal"].to_numpy(dtype=np.float64))
        cumsum = annotator._get_cumsum_signal(df["signal"].to_numpy(dtype=np.float64))

        l_idx, r_idx, _, _ = annotator._find_multi_peak_edges(
            region_start=2,
            region_end=5,
            deriv=deriv,
            signal_cum_sum=cumsum,
        )
        self.assertEqual(l_idx, 2)
        self.assertEqual(r_idx, 5)
    
    def test_find_multi_peak_edges_smooth_increase(self):
        df = self._make_df([0.0, 0.3, 1.0, 1.0, 1.0, 0.3, 0.0, 0.0, 1.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=30, lambda_weight=1, sigma=10.0)

        deriv = annotator._calculate_smooth_derivative(df["signal"].to_numpy(dtype=np.float64))
        cumsum = np.cumsum(df["signal"].to_numpy(dtype=np.float64))

        l_idx, r_idx, _, _ = annotator._find_multi_peak_edges(
            region_start=2,
            region_end=5,
            deriv=deriv,
            signal_cum_sum=cumsum,
        )
        self.assertEqual(l_idx, 2)
        self.assertEqual(r_idx, 5)

    def test_find_multi_peak_edges_wider(self):
        df = self._make_df([0.0, 0.3, 1.0, 1.0, 1.0, 1.0, 0.3, 0.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=30, lambda_weight=1, sigma=10.0)

        deriv = annotator._calculate_smooth_derivative(df["signal"].to_numpy(dtype=np.float64))
        cumsum = annotator._get_cumsum_signal(df["signal"].to_numpy(dtype=np.float64))

        l_idx, r_idx, _, _ = annotator._find_multi_peak_edges(
            region_start=2,
            region_end=5,
            deriv=deriv,
            signal_cum_sum=cumsum,
        )
        self.assertEqual(l_idx, 2)
        self.assertEqual(r_idx, 6)


    def test_find_multi_peak_left_edge(self):
        df = self._make_df([1.0, 1.3, 1.0, 1.0, 0.0, 0.0, 0.3, 0.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=30, lambda_weight=1, sigma=10.0)

        deriv = annotator._calculate_smooth_derivative(df["signal"].to_numpy(dtype=np.float64))
        cumsum = annotator._get_cumsum_signal(df["signal"].to_numpy(dtype=np.float64))

        l_idx, r_idx, _, _ = annotator._find_multi_peak_edges(
            region_start=0,
            region_end=3,
            deriv=deriv,
            signal_cum_sum=cumsum,
        )
        self.assertEqual(l_idx, 0)
        self.assertEqual(r_idx, 4)

    def test_find_multi_peak_right_edge(self):
        df = self._make_df([1.0, 1.3, 0.0, 0.0, 1.0, 1.0, 1.3, 1.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=30, lambda_weight=1, sigma=10.0)

        deriv = annotator._calculate_smooth_derivative(df["signal"].to_numpy(dtype=np.float64))
        cumsum = annotator._get_cumsum_signal(df["signal"].to_numpy(dtype=np.float64))

        l_idx, r_idx, _, _ = annotator._find_multi_peak_edges(
            region_start=3,
            region_end=8,
            deriv=deriv,
            signal_cum_sum=cumsum,
        )
        self.assertEqual(l_idx, 4)
        self.assertEqual(r_idx, 8)


    def test_find_multi_peak_small_signal_wide_peak(self):
        df = self._make_df([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=50, lambda_weight=1, sigma=10.0)

        deriv = annotator._calculate_smooth_derivative(df["signal"].to_numpy(dtype=np.float64))
        cumsum = annotator._get_cumsum_signal(df["signal"].to_numpy(dtype=np.float64))

        l_idx, r_idx, _, _ = annotator._find_multi_peak_edges(
            region_start=1,
            region_end=2,
            deriv=deriv,
            signal_cum_sum=cumsum,
        )
        self.assertEqual(l_idx, 1)
        self.assertEqual(r_idx, 6)

    def test_find_multi_peak_wide_wide_peak(self):
        df = self._make_df([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=30, lambda_weight=1, sigma=10.0)

        deriv = annotator._calculate_smooth_derivative(df["signal"].to_numpy(dtype=np.float64))
        cumsum = annotator._get_cumsum_signal(df["signal"].to_numpy(dtype=np.float64))

        l_idx, r_idx, _, _ = annotator._find_multi_peak_edges(
            region_start=0,
            region_end=4,
            deriv=deriv,
            signal_cum_sum=cumsum,
        )
        self.assertEqual(l_idx, 1)
        self.assertEqual(r_idx, 7)

    def test_select_peaks_in_region_single_peak(self):
        df = self._make_df([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=30)
        deriv = annotator._calculate_smooth_derivative(df["signal"].to_numpy(dtype=np.float64))
        cumsum = annotator._get_cumsum_signal(df["signal"].to_numpy(dtype=np.float64))

        peaks = annotator._select_peaks_in_region(3, 5, deriv, cumsum)
        self.assertGreaterEqual(len(peaks), 1)
        for start, end, _ in peaks:
            self.assertEqual(start, 3)
            self.assertEqual(end, 6)

    def test_select_peaks_in_region_multi_peak(self):
        df = self._make_df([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], step_size=10)
        annotator = PeakAnnotator(df=df, threshold_peak=0.0, window_size=30)
        deriv = annotator._calculate_smooth_derivative(df["signal"].to_numpy(dtype=np.float64))
        cumsum = annotator._get_cumsum_signal(df["signal"].to_numpy(dtype=np.float64))

        peaks = annotator._select_peaks_in_region(0, 6, deriv, cumsum)
        self.assertEqual(len(peaks), 2)
        for start, end, score in peaks:
            #peak should be (1, 4) and (4, 7)
            self.assertIn(start, [1, 4])
            self.assertIn(end, [4, 7])
            self.assertIsInstance(score, float)
            self.assertGreater(score, 0.0)

    def test_extract_and_format_peak_single_peak(self):
        df = self._make_df([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=30)
        signal = df["signal"].to_numpy(dtype=np.float64)
        deriv = annotator._calculate_smooth_derivative(signal)
        cumsum = annotator._get_cumsum_signal(signal)
        regions = [(2, 5)]
        start, finish, mass_norm_term, deriv_norm_term = annotator._find_multi_peak_edges(2, 5, deriv, cumsum)

        peaks = annotator._extract_and_format_peaks(
            regions=regions,
            deriv=deriv,
            signal_cum_sum=cumsum,
            window_starts=df["window_start"].to_numpy(),
        )
        self.assertEqual(len(peaks), 1)
        self.assertTrue({"peak_start", "peak_end", "score", "region_idx", "edge_peak"}.issubset(set(peaks.columns)))
        expected = pd.DataFrame([{
            "peak_start": 20,
            "peak_end": 49,
            "peak_middle_start": 30,
            "peak_middle_end": 40,
            "score": annotator._calculate_peak_score(2, 5, deriv, mass_norm_term=mass_norm_term, deriv_norm_term=deriv_norm_term, reduced_mass_score=3),
            "region_idx": 0,
            "peak_rank": 0,
            "edge_peak": True,
        }])
        self.assertTrue(peaks.equals(expected))

    def test_extract_and_format_peak_multi_peak(self):
        df = self._make_df([0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=30)
        signal = df["signal"].to_numpy(dtype=np.float64)
        deriv = annotator._calculate_smooth_derivative(signal)
        cumsum = annotator._get_cumsum_signal(signal)
        regions = [(1, 3), (7, 10)]
        _, _, mass_norm_term_a, deriv_norm_term_a = annotator._find_multi_peak_edges(1, 3, deriv, cumsum)
        _, _, mass_norm_term_b, deriv_norm_term_b = annotator._find_multi_peak_edges(7, 10, deriv, cumsum)

        peaks = annotator._extract_and_format_peaks(
            regions=regions,
            deriv=deriv,
            signal_cum_sum=cumsum,
            window_starts=df["window_start"].to_numpy(),
        )
        self.assertGreaterEqual(len(peaks), 1)
        self.assertTrue({"peak_start", "peak_end", "score", "region_idx", "edge_peak"}.issubset(set(peaks.columns)))
        expected = pd.DataFrame([{
                "peak_start": 10,
                "peak_end": 39,
                "peak_middle_start": 20,
                "peak_middle_end": 30,
                "score": annotator._calculate_peak_score(1, 4, deriv, mass_norm_term=mass_norm_term_a, deriv_norm_term=deriv_norm_term_a, reduced_mass_score=3),
                "region_idx": 0,
                "peak_rank": 0,
                "edge_peak": True,
            },
            {
                "peak_start": 20,
                "peak_end": 49,
                "peak_middle_start": 30,
                "peak_middle_end": 40,
                "score": annotator._calculate_peak_score(2, 5, deriv, mass_norm_term=mass_norm_term_a, deriv_norm_term=deriv_norm_term_a, reduced_mass_score=3),
                "region_idx": 0,
                "peak_rank": 1,
                "edge_peak": True,
            },
            {
                "peak_start": 70,
                "peak_end": 99,
                "peak_middle_start": 80,
                "peak_middle_end": 90,
                "score": annotator._calculate_peak_score(7, 10, deriv, mass_norm_term=mass_norm_term_b, deriv_norm_term=deriv_norm_term_b, reduced_mass_score=3),
                "region_idx": 1,
                "peak_rank": 0,
                "edge_peak": True,
            }
        ])
        print(peaks)
        print(expected)
        self.assertTrue(peaks.equals(expected))

    def test_extract_and_format_peaks_empty(self):
        df = self._make_df([0.0, 0.2, 0.7, 1.2, 0.8, 0.3, 0.1, 0.0], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df, window_size=20)
        signal = df["signal"].to_numpy(dtype=np.float64)
        deriv = annotator._calculate_smooth_derivative(signal)
        cumsum = annotator._get_cumsum_signal(signal)

        empty = annotator._extract_and_format_peaks([], deriv, cumsum, df["window_start"].to_numpy())
        self.assertTrue(empty.empty)

    def test_get_signal_returns_float64_and_empty(self):
        df = self._make_df([1, 2, 3], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df)
        signal = annotator._get_signal("signal")
        self.assertEqual(signal.dtype, np.float64)

        empty_df = pd.DataFrame({"window_start": [], "signal": []})
        empty_annotator = PeakAnnotator(threshold_peak=0, df=empty_df, step_size=10)
        empty_signal = empty_annotator._get_signal("signal")
        self.assertEqual(len(empty_signal), 0)

    # ===== public API =====

    def test_detect_peaks_validation_errors(self):
        df = self._make_df([0.0, 0.2, 0.7], step_size=10)
        annotator = PeakAnnotator(threshold_peak=0, df=df)
        with self.assertRaises(ValueError):
            annotator.detect_peaks(signal_column="missing")

        bad_df = pd.DataFrame({"window_start": [0, 10, 20], "signal": ["x", "y", "z"]})
        bad_annotator = PeakAnnotator(threshold_peak=0, df=bad_df)
        with self.assertRaises(ValueError):
            bad_annotator.detect_peaks(signal_column="signal")

    def test_detect_peaks_empty_dataframe_returns_empty(self):
        df = pd.DataFrame({"window_start": [], "signal": []})
        annotator = PeakAnnotator(threshold_peak=0, df=df, step_size=10)
        result = annotator.detect_peaks(signal_column="signal")

        self.assertTrue(result.empty)
        expected_columns = {"peak_start", "peak_end", "score", "region_idx", "peak_rank", "edge_peak", "peak_middle_start", "peak_middle_end"}
        self.assertEqual(
            set(result.columns),
            expected_columns,
        )
        expected_df = pd.DataFrame(columns=list(expected_columns))
        pd.testing.assert_frame_equal(result, expected_df, check_like=True)

    def test_detect_peaks_no_region_returns_empty(self):
        df = self._make_df([0.0, 0.0, 0.0, 0.0, 0.0], step_size=10)
        annotator = PeakAnnotator(df=df, window_size=20, threshold_peak=10.0)
        result = annotator.detect_peaks(signal_column="signal")
        expected_columns = ["peak_start", "peak_end", "peak_middle_start", "peak_middle_end", "score", "region_idx", "peak_rank", "edge_peak"]
        expected_df = pd.DataFrame(columns=expected_columns)
        pd.testing.assert_frame_equal(result, expected_df, check_like=True)

    def test_detect_peaks_basic(self):
        signal = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        df = self._make_df(signal, step_size=10)

        annotator = PeakAnnotator(
            df=df,
            window_size=30,
            threshold_peak=0.2,
            sigma=10.0,
            lambda_weight=0.5,
        )
        deriv = annotator._calculate_smooth_derivative(df["signal"].to_numpy(dtype=np.float64))
        cumsum = annotator._get_cumsum_signal(df["signal"].to_numpy(dtype=np.float64))
        _, _, mass_norm_term_a, deriv_norm_term_a = annotator._find_multi_peak_edges(0, 5, deriv, cumsum)

        result = annotator.detect_peaks(signal_column="signal")
        print(result)

        expected = pd.DataFrame([{
            "peak_start": 20,
            "peak_end": 49,
            "peak_middle_start": 30,
            "peak_middle_end": 40,
            "score": annotator._calculate_peak_score(2, 5, deriv, mass_norm_term=mass_norm_term_a, deriv_norm_term=deriv_norm_term_a, reduced_mass_score=annotator._calculate_reduced_mass_score(l=2, r=5, signal_cum_sum=cumsum)),
            "region_idx": 0,
            "peak_rank": 0,
            "edge_peak": True,
        }])
        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_detect_peaks_multi(self):
        signal = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        df = self._make_df(signal, step_size=10)

        annotator = PeakAnnotator(
            df=df,
            window_size=30,
            threshold_peak=0.2,
            sigma=10.0,
            lambda_weight=0.5,
        )

        result = annotator.detect_peaks(signal_column="signal")
        print(result)
        cum_sum = annotator._get_cumsum_signal(df["signal"].to_numpy(dtype=np.float64))
        deriv = annotator._calculate_smooth_derivative(df["signal"].to_numpy(dtype=np.float64))
        _, _, mass_norm_term_a, deriv_norm_term_a = annotator._find_multi_peak_edges(0, 5, deriv, cum_sum)
        _, _, mass_norm_term_b, deriv_norm_term_b = annotator._find_multi_peak_edges(6, 11, deriv, cum_sum)

        expected = pd.DataFrame({
            "peak_start": [20, 80, 100],
            "peak_end": [49, 109, 129],
            "peak_middle_start": [30, 90, 110],
            "peak_middle_end": [40, 100, 120],
            "score": [
                annotator._calculate_peak_score(2, 5, deriv, mass_norm_term=mass_norm_term_a, deriv_norm_term=deriv_norm_term_a, reduced_mass_score=annotator._calculate_reduced_mass_score(l=2, r=5, signal_cum_sum=cum_sum)),
                annotator._calculate_peak_score(8, 11, deriv, mass_norm_term=mass_norm_term_b, deriv_norm_term=deriv_norm_term_b, reduced_mass_score=annotator._calculate_reduced_mass_score(l=8, r=11, signal_cum_sum=cum_sum)),
                annotator._calculate_peak_score(10, 13, deriv, mass_norm_term=mass_norm_term_b, deriv_norm_term=deriv_norm_term_b, reduced_mass_score=annotator._calculate_reduced_mass_score(l=10, r=13, signal_cum_sum=cum_sum)),
            ],
            "region_idx": [0, 1, 1],
            "peak_rank": [0, 0, 1],
            "edge_peak": [True, True, True],
        })
        pd.testing.assert_frame_equal(result, expected, check_like=True)

    def test_repr_contains_configuration(self):
        df = self._make_df([0.0, 1.0, 2.0], step_size=10)
        annotator = PeakAnnotator(
            df=df,
            window_size=30,
            step_size=10,
            threshold_peak=0.3,
            sigma=12.0,
            lambda_weight=2.0,
        )
        text = repr(annotator)
        self.assertIn("window_size=30", text)
        self.assertIn("step_size=10", text)
        self.assertIn("threshold_peak=0.3", text)
        self.assertIn("sigma=12.0", text)
        self.assertIn("lambda_weight=2.0", text)

    # ===== calculate_mass_contributions tests =====

if __name__ == "__main__":
    unittest.main()
    #test specific test for debugging
    # suite = unittest.TestSuite()
    # suite.addTest(TestPeakAnnotator("test_repr_contains_configuration"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
