"""Peak detection algorithm for TF binding signals from deepCIS predictions.

This module provides the PeakAnnotator class which encapsulates a multi-part
peak detection algorithm designed for analyzing sliding-window deepCIS
transcription factor binding predictions.

Algorithm Overview:
- Part A: Region detection via moving average and thresholding
- Part B: Signal preprocessing (smoothing, derivatives, mass terms)
- Part C: Peak selection via greedy non-overlapping peak selection
- Part D: Output formatting with genomic coordinate mapping
"""

import math

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from scipy.ndimage import gaussian_filter1d

from analysis.motives.deepcis_scanner import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_STEP_SIZE,
)


class PeakAnnotator:
    window_size: int
    step_size: int
    threshold_peak: float
    sigma: float
    lambda_weight: float
    df: pd.DataFrame

    _window_size_elements: int
    _sigma_elements: float
    """Object-oriented peak detection with configurable multi-scale parameters.

    This class encapsulates a deterministic algorithm for identifying local maxima
    (peaks) in transcription factor binding signals. All algorithm parameters are
    stored as instance attributes for easy configuration and reuse.

    **Algorithm Parts:**
    - Part A: Detect broad peak regions via moving average and thresholding
    - Part B: Preprocess signal with Gaussian smoothing and edge detection
    - Part C: Greedily select non-overlapping peaks with scoring
    - Part D: Format results with genomic coordinate mapping

    Attributes:
        window_size: Sliding-window size in bp (default 250).
        step_size: Step size between window measurements in bp (default 10).
        threshold_peak: Threshold for region detection (default 0.0).
        sigma: Gaussian smoothing parameter in bp (default 20.0).
        lambda_weight: Balance between peak sharpness and mass (default 1.0).

    Example:
        >>> signal_df = pd.DataFrame({  # doctest: +SKIP
        ...     'signal': [...],
        ...     'window_start': [...],
        ... })
        >>> # Infer step_size from DataFrame at initialization
        >>> annotator = PeakAnnotator(window_size=250, sigma=20.0, df=signal_df)
        >>> peaks = annotator.detect_peaks(signal_df)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        threshold_peak: float,
        step_size: Optional[int] = None,
        sigma: float = 10.,
        window_size: int = DEFAULT_WINDOW_SIZE,
        lambda_weight: float = 1.0,
    ):
        """Initialize peak annotator with algorithm parameters.

        Args:
            window_size: Sliding-window size in bp (default 250).
            step_size: Step size between window measurements in bp (default 10).
                If None and df is provided, will be inferred from window_start column.
            threshold_peak: Threshold for region detection (default 0.0).
            sigma: Gaussian smoothing parameter in bp (default 20.0).
            lambda_weight: Balance between peak sharpness and mass (default 1.0).
            df: DataFrame with window_start column to infer step_size.
                If provided and step_size is None, step_size is inferred from data.
        """
        self.window_size = window_size
        self.threshold_peak = threshold_peak
        self.sigma = sigma
        self.lambda_weight = lambda_weight
        self.df = df

        # Initialize step_size from parameter or infer from DataFrame
        if step_size is not None:
            self.step_size = step_size
        elif self.df is not None:
            self.step_size = self._infer_step_size_from_dataframe(df)
        else:
            self.step_size = DEFAULT_STEP_SIZE

        self._window_size_elements = self.window_size // self.step_size
        self._sigma_elements = self.sigma / self.step_size
        
        self._validate_uniform_step_size()
        self._validate_inputs()
        
    # ========== VALIDATION METHODS ==========

    def _infer_step_size_from_dataframe(self, df: pd.DataFrame) -> int:
        """Infer step_size from window_start column in DataFrame.

        Computes the spacing between consecutive window_start values.

        Args:
            df: DataFrame with window_start column.

        Returns:
            Inferred step_size in bp (spacing between window measurements).

        Raises:
            ValueError: If window_start column is missing or has insufficient data.
        """
        if 'window_start' not in df.columns:
            raise ValueError(
                f"Cannot infer step_size: 'window_start' column not found. "
                f"Available columns: {list(df.columns)}"
            )
        
        if len(df) < 2:
            raise ValueError(
                f"Cannot infer step_size: DataFrame must have at least 2 rows, got {len(df)}"
            )
        
        window_starts = df['window_start'].values
        step_size = int(window_starts[1] - window_starts[0])
        
        if step_size <= 0:
            raise ValueError(
                f"Invalid window_start spacing: step_size must be > 0, "
                f"but got {step_size} from first two window_start values"
            )
        
        return step_size
    

    def validate_signal_column(self, signal_column: str) -> None:
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(self.df).__name__}")

        if signal_column not in self.df.columns:
            raise ValueError(
                f"Signal column '{signal_column}' not found in DataFrame. "
                f"Available columns: {list(self.df.columns)}"
            )

        if not np.issubdtype(self.df[signal_column].dtype, np.number):
            raise ValueError(
                f"Signal column '{signal_column}' must be numeric, "
                f"got {self.df[signal_column].dtype}"
            )

    def _validate_inputs(self,) -> None:
        """Validate input DataFrame and parameters.

        Args:
            df: Input DataFrame.
            signal_column: Name of signal column.

        Raises:
            TypeError: If df is not a DataFrame.
            ValueError: If columns or parameters are invalid.
        """
        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")

        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")

        if self.lambda_weight < 0:
            raise ValueError(f"lambda_weight must be >= 0, got {self.lambda_weight}")

    # ========== PART A: REGION DETECTION ==========

    def _compute_moving_average(self,signal: np.ndarray) -> np.ndarray:
        """Compute moving average using uniform convolution.

        Args:
            signal: 1D numpy array of signal values.

        Returns:
            1D array of moving averages (shorter by window_size_elements - 1).
        """
        kernel = np.ones(self._window_size_elements) / self._window_size_elements
        return np.convolve(signal, kernel, mode='valid')

    def _threshold_mask(self, avg_signal: np.ndarray) -> np.ndarray:
        """Create binary mask where moving average exceeds threshold.

        Args:
            avg_signal: 1D array of pre-smoothed values.

        Returns:
            Boolean 1D array.
        """
        return avg_signal > self.threshold_peak

    @staticmethod
    def _find_contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find contiguous stretches where mask == True.

        Args:
            mask: Boolean 1D array.

        Returns:
            List of (start, end) tuples where start is inclusive, end is exclusive.
        """
        regions = []
        in_region = False
        region_start = 0

        for i, value in enumerate(mask):
            if value and not in_region:
                region_start = i
                in_region = True
            elif not value and in_region:
                regions.append((region_start, i))
                in_region = False

        if in_region:
            regions.append((region_start, len(mask)))

        return regions

    def _detect_signal_regions(
        self,
        signal: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Detect broad peak regions via moving average and thresholding.

        Args:
            signal: 1D signal array.

        Returns:
            List of (region_start, region_end) tuples.
        """
        avg_signal = self._compute_moving_average(signal)
        if len(avg_signal) == 0:
            return []

        mask = self._threshold_mask(avg_signal)
        regions = self._find_contiguous_regions(mask)
        return regions

    # ========== PART B: SIGNAL PREPROCESSING ==========

    def _gaussian_smooth(self, signal: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing to signal.

        Args:
            signal: 1D numpy array of signal values.
            sigma: Standard deviation of Gaussian kernel.

        Returns:
            Smoothed 1D array, same length as input.
        """
        return gaussian_filter1d(signal, sigma=self._sigma_elements, mode='nearest')

    @staticmethod
    def _compute_forward_derivative(smooth_signal: np.ndarray) -> np.ndarray:
        """Compute forward difference (edge-detection signal).

        Args:
            smooth_signal: 1D smoothed signal array.

        Returns:
            1D array of forward differences, same length as input.
        """
        deriv = np.diff(smooth_signal)
        return deriv

    def _calculate_smooth_derivative(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """Preprocess signal for peak detection.

        Applies Gaussian smoothing, computes forward derivative and mass term.

        Args:
            signal: 1D signal array.

        Returns:
            Tuple of (deriv, mass) arrays.
        """
        #extend signal by one zero before and after to ensure derivative can be calculated for all possible edges
        signal = np.pad(signal, (1, 1), mode='constant', constant_values=0)
        smooth_signal = self._gaussian_smooth(signal)
        deriv = self._compute_forward_derivative(smooth_signal)
        return deriv

    # ========== PART C: PEAK SELECTION ==========

    def distribute_peaks_over_area(
        self,
        start: int,
        finish: int,
        num_peaks: int,
    ) -> List[Tuple[int, int]]:
        """distributes peaks evenly over the area of the region, ensuring full coverage of the region.

        peaks all have the self._window_size_elements size, and are distributed evenly across the region, with the first peak starting at start and the last peak ending at finish.

        Args:
            start: Inclusive start index of the region.
            finish: Exclusive end index of the region.
            num_peaks: Number of peaks to distribute.

        Returns:
            List of (peak_start, peak_end) tuples.
        """
        region_width = finish - start
        if num_peaks <= 0 or region_width <= 0:
            print(f"Warning: Invalid parameters for distribute_peaks_over_area: start={start}, finish={finish}, num_peaks={num_peaks}")
            return []
        step = region_width / num_peaks
        peaks = []

        for i in range(num_peaks):
            peak_start = round(start + i * step)
            peak_end = peak_start + self._window_size_elements
            if peak_end > finish:
                peak_end = finish
                peak_start = max(start, peak_end - self._window_size_elements)
            peaks.append((peak_start, peak_end))

        return peaks
    
    def _calculate_reduced_mass_score(self, l: int, r: int, signal_cum_sum: np.ndarray) -> float:
        """Calculate the reduced mass score for a candidate peak defined by (l, r).

        Args:
            l: Left index of the candidate peak.
            r: Right index of the candidate peak.
            signal_cum_sum: Cumulative sum array for mass term calculation.
        Returns:
            Reduced mass score for the candidate peak.
        """
        mass_term = signal_cum_sum[r] - signal_cum_sum[l]
        mass_term_reduced = mass_term - (r - l) * self.threshold_peak
        return mass_term_reduced
    
    def calculate_mass_contributions(self, signal_cum_sum: np.ndarray, scan_range: Tuple[int, int, int, int],) -> np.ndarray:
        """Calculate mass contributions for candidate peaks in the scan range.

        Args:
            deriv: Forward derivative array.
            signal_cum_sum: Cumulative sum array for mass term calculation.
            scan_range: Tuple of (region_start, region_end, left_end, right_end) indices.

        Returns:
            2D array of mass contributions for each (l, r) candidate pair.
        """
        region_start, region_end, left_end, right_end = scan_range
        left_elements = left_end - region_start + 1
        right_elements = right_end - region_end  + 1
        mass_contributions = np.full((left_elements, right_elements), dtype=np.float64, fill_value=np.nan)

        for i, l in enumerate(range(region_start, left_end + 1)):
            for j, r in enumerate(range(region_end, right_end + 1)):
                if r <= l:
                    continue
                mass_term_reduced = self._calculate_reduced_mass_score(l, r, signal_cum_sum)
                mass_contributions[i, j] = mass_term_reduced

        return mass_contributions
    
    def calculate_scan_range(self, region_start: int, region_end: int, deriv: np.ndarray) -> Tuple[int, int, int, int]:
        region_end = min(region_end, len(deriv) - 1)
        left_end = min(len(deriv) - 1, region_start + self._window_size_elements)
        right_end = min(len(deriv) - 1, region_end + self._window_size_elements)
        return region_start, region_end, left_end, right_end

    def _find_multi_peak_edges(self, region_start: int, region_end: int, deriv: np.ndarray, signal_cum_sum: np.ndarray) -> Tuple[int, int, float, float]:
        """Determine the edges of one connected region, which might contain multiple peaks.

        Returns:
            Tuple of (edge_start, edge_end) in element indices.
        """
        candidates: List[Tuple[float, int, int]] = []
        scan_range = self.calculate_scan_range(region_start, region_end, deriv)
        mass_contributions = self.calculate_mass_contributions(signal_cum_sum, scan_range)
        mass_norm_term = float(np.nanpercentile(mass_contributions, 90))
        deriv_norm_term = float(np.nanpercentile(abs(deriv), 90))
        region_start, region_end, left_end, right_end = scan_range

        for i, l in enumerate(range(region_start, left_end + 1)):
            for j, r in enumerate(range(region_end, right_end + 1)):
                if r <= l:
                    continue
                reduced_mass_score = mass_contributions[i, j]
                score = self._calculate_peak_score(peak_start_idx=l, peak_end_idx=r, deriv=deriv, deriv_norm_term=deriv_norm_term, mass_norm_term=mass_norm_term, reduced_mass_score=reduced_mass_score)
                candidates.append((score, l, r))

        if not candidates:
            fallback_start = max(0, min(region_start, len(deriv) - 1))
            fallback_end = max(fallback_start + 1, min(region_end, len(deriv) - 1))
            return fallback_start, fallback_end, mass_norm_term, deriv_norm_term

        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
        result = candidates[0]
        return result[1], result[2], mass_norm_term, deriv_norm_term
    
    def _calculate_peak_score(self, peak_start_idx: int, peak_end_idx: int, deriv: np.ndarray, mass_norm_term: float, deriv_norm_term: float, reduced_mass_score: float) -> float:
        """Calculate the score of a peak based on derivative and mass term.

        Args:
            peak_start_idx: Start index of the peak in element indices.
            peak_end_idx: End index of the peak in element indices.
            deriv: Forward derivative array.
            normalized_mass_term: Normalized mass term.
            deriv_norm_term: Normalized derivative term.

        Returns:
            Peak score.
        """
        if peak_end_idx <= peak_start_idx:
            return float('-inf')

        normalized_mass_term = reduced_mass_score / mass_norm_term
        lambda_mass = self.lambda_weight * normalized_mass_term
        left_flank = deriv[peak_start_idx]
        right_flank = -deriv[peak_end_idx]
        left_flank_norm = left_flank / deriv_norm_term
        right_flank_norm = right_flank / deriv_norm_term
        score = left_flank_norm + right_flank_norm + lambda_mass
        return score

    def _select_peaks_in_region(
        self,
        region_start: int,
        region_end: int,
        deriv: np.ndarray,
        signal_cum_sum: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """Greedily select non-overlapping peaks within one region.

        Args:
            region_start: Inclusive start index (element indices).
            region_end: Exclusive end index (element indices).
            deriv: Full derivative array.
            signal_cum_sum: Cumulative sum array.

        Returns:
            List of (peak_start, peak_end) tuples in element indices.
        """
        
        start, finish, mass_norm_term, deriv_norm_term = self._find_multi_peak_edges(region_start, region_end, deriv, signal_cum_sum)
        region_width = finish - start
        num_peaks = max(1, math.ceil(region_width / self._window_size_elements))

        selected_peaks = self.distribute_peaks_over_area(
            start, finish, num_peaks=num_peaks
        )

        selected_peaks = [(s, e, self._calculate_peak_score(peak_start_idx=s, peak_end_idx=e, deriv=deriv, mass_norm_term=mass_norm_term, deriv_norm_term=deriv_norm_term, reduced_mass_score=self._calculate_reduced_mass_score(s, e, signal_cum_sum))) for s, e in selected_peaks]

        return selected_peaks


    def _validate_uniform_step_size(self) -> None:
        """Validate that step_size is uniform across the DataFrame.

        Checks that the spacing between consecutive window_start values is consistent
        and matches the configured step_size.

        Args:
            df: Input DataFrame with window_start column.

        Raises:
            ValueError: If step_size is not uniform or does not match configuration.
        """
        if 'window_start' not in self.df.columns:
            raise ValueError(
                f"Cannot validate step_size: 'window_start' column not found. "
                f"Available columns: {list(self.df.columns)}"
            )

        window_starts = self.df['window_start'].values
        if len(window_starts) < 2:
            return

        inferred_step_size = int(window_starts[1] - window_starts[0])
        for i in range(2, len(window_starts)):
            current_step = int(window_starts[i] - window_starts[i - 1])
            if current_step != inferred_step_size:
                raise ValueError(
                    f"Inconsistent step_size detected at index {i}: "
                    f"expected {inferred_step_size}, got {current_step}"
                )

        if inferred_step_size != self.step_size:
            raise ValueError(
                f"Inferred step_size {inferred_step_size} does not match "
                f"configured step_size {self.step_size}"
            )

    def _get_signal( self, signal_column: str,) -> np.ndarray:
        """Validate inputs and cache derived values for peak detection.

        Performs all input validation and caches the derived element-based
        parameters for use by other methods. Step_size should be set in __init__.

        Args:
            df: Input DataFrame with signal and optional window_start columns.
            signal_column: Name of the signal column to analyze.

        Returns:
            Tuple of (signal_array, has_genomic_positions).
            Signal array is the signal column as float64 numpy array.
            has_genomic_positions is True if window_start column exists.

        Raises:
            ValueError: If validation fails.
            TypeError: If df is not a DataFrame.
        """
        # ======== Validate inputs ========
        if len(self.df) == 0:
            return np.array([], dtype=np.float64)
        signal = self.df[signal_column].values.astype(np.float64)
        return signal
    
    @staticmethod
    def empty_result_df() -> pd.DataFrame:
        """Create an empty result DataFrame with the correct columns."""
        return pd.DataFrame(
            columns=['peak_start', 'peak_end', 'peak_middle_start', 'peak_middle_end', 'score', 'region_idx', 'peak_rank', 'edge_peak']
        )
    
    def get_middle_window(self, start: int, end: int) -> Tuple[int, int]:
        """Calculate the middle window of a given region.

        Args:
            start: Inclusive start coordinate of the region.
            end: Exclusive end coordinate of the region.

        Returns:
            Tuple of (middle_start, middle_end) coordinates for the middle window.
        """
        middle = (start + end + 1) / 2
        middle_start = middle - self.step_size / 2
        middle_end = middle_start + self.step_size

        return int(middle_start), int(middle_end)

    def _extract_and_format_peaks(
        self,
        regions: List[Tuple[int, int]],
        deriv: np.ndarray,
        signal_cum_sum: np.ndarray,
        window_starts: np.ndarray,
    ) -> pd.DataFrame:
        """Execute Parts C and D: peak selection and output formatting.

        Part C: Greedily selects non-overlapping peaks within regions.
        Part D: Formats output with genomic coordinate mapping.

        Args:
            regions: List of (start, end) tuples from Part A.
            deriv: Forward derivative array from Part B.
            cum_sum: Cumulative sum array from Part B.
            window_starts: Optional window_start positions for mapping.

        Returns:
            DataFrame with detected peaks (peak_start, peak_end, score,
            region_idx, peak_rank columns). Empty DataFrame if no peaks found.
        """
        # ======== Part C: Extract and score peaks ========
        all_peaks: List[dict] = []

        for region_idx, (region_start, region_end) in enumerate(regions):

            peaks = self._select_peaks_in_region(
                region_start, region_end, deriv, signal_cum_sum
            )

            for peak_rank, (peak_start_idx, peak_end_idx, score) in enumerate(peaks):

                out_start = int(window_starts[peak_start_idx])
                out_end = int(window_starts[peak_end_idx - 1]) + self.step_size - 1

                middle_start, middle_end = self.get_middle_window(out_start, out_end)

                all_peaks.append({
                    'peak_start': out_start,
                    'peak_end': out_end,
                    'peak_middle_start': middle_start,
                    'peak_middle_end': middle_end,
                    'score': score,
                    'region_idx': region_idx,
                    'peak_rank': peak_rank,
                    # peaks at the edges should be perfectly centered around TFBS, peaks in between are kind of best guesses.
                    'edge_peak': peak_rank == 0 or peak_rank == len(peaks) - 1,
                })
        # ======== Part D: Format output ========
        if not all_peaks:
            return self.empty_result_df()
        result_df = pd.DataFrame(all_peaks)
        return result_df
    
    def _get_cumsum_signal(self, signal: np.ndarray) -> np.ndarray:
        """Compute cumulative sum of the signal for mass term calculation.

        Args:
            signal: 1D signal array.
        Returns:
            Cumulative sum array, same length as input.
        """
        summed_array = list(np.cumsum(signal))
        summed_array.insert(0, 0.0)
        summed_array.append(summed_array[-1])
        return np.array(summed_array)


    def detect_peaks( self, signal_column: str = "signal",) -> pd.DataFrame:
        """Detect peaks in a signal DataFrame using configured parameters.

        Applies the full 4-part peak detection algorithm:
        1. Region Detection (Part A): Moving average and thresholding
        2. Signal Preprocessing (Part B): Gaussian smoothing and derivatives
        3. Peak Selection (Part C): Greedy non-overlapping peak selection
        4. Output Formatting (Part D): Genomic coordinate mapping

        Args:
            df: Input DataFrame with numeric signal column and preferably
                a ``window_start`` column for genomic coordinate mapping.
            signal_column: Name of the column to analyze (default 'signal').

        Returns:
            DataFrame with detected peaks (columns: peak_start, peak_end,
            score, region_idx, peak_rank). Empty DataFrame if no peaks found.

        Raises:
            ValueError: If parameter or column validation fails.
            TypeError: If df is not a DataFrame.

        Example:
            >>> signal_df = pd.DataFrame({  # doctest: +SKIP
            ...     'signal': [0.1, 0.3, 0.8, 0.5, 0.2],
            ...     'window_start': [0, 10, 20, 30, 40],
            ... })
            >>> annotator = PeakAnnotator(window_size=250)
            >>> peaks = annotator.detect_peaks(signal_df)
            >>> print(peaks)  # doctest: +SKIP
               peak_start  peak_end    score  region_idx  peak_rank
            0        0     250  0.123456           0          1
        """
        self.validate_signal_column(signal_column)
        signal = self._get_signal(signal_column)
        if len(signal) == 0:
            return self.empty_result_df()

        regions = self._detect_signal_regions(signal)
        if not regions:
            return self.empty_result_df()

        deriv = self._calculate_smooth_derivative(signal)
        signal_cum_sum = self._get_cumsum_signal(signal)
        window_starts = self.df['window_start'].values
        result_df = self._extract_and_format_peaks(
            regions, deriv, signal_cum_sum, window_starts               # type: ignore
        )

        return result_df

    def __repr__(self) -> str:
        """String representation showing configuration."""
        return (
            f"PeakAnnotator(window_size={self.window_size}, "
            f"step_size={self.step_size}, threshold_peak={self.threshold_peak}, "
            f"sigma={self.sigma}, lambda_weight={self.lambda_weight})"
        )

#TODO: rethink logic with stepsize (dont rely on fixed stepsize, instead dynamically select windows within 250bp region)