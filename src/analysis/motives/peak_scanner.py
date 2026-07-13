"""High-level scanning interface for deepCIS predictions and peak detection.

This module provides the DeepCISPeakScanner class for orchestrating peak detection
across multiple genes and transcription factors from deepCIS sliding-window
predictions.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import pandas as pd

from analysis.motives.peak_annotation import PeakAnnotator
from tqdm import tqdm


logger = logging.getLogger(__name__)


# ============================================================================
# DeepCISPeakScanner class
# ============================================================================

class DeepCISPeakScanner:
    """Object-oriented interface for scanning deepCIS predictions for binding peaks.

    This class orchestrates peak detection across multiple genes and transcription
    factors from deepCIS sliding-window predictions. It uses a PeakAnnotator
    internally to perform the actual peak detection.

    Attributes:
        genes: List of genes to scan (None = all available).
        tfs: List of TFs to scan (None = all available).
        signal_type: Which signal to analyze: 'reference', 'mutated', or 'difference'.
        output_path: Full CSV file path to save results when explicitly set.
        save_results: Whether to save results to CSV.

    Example:
        >>> scanner = DeepCISPeakScanner(
        ...     genes=['gene1', 'gene2'],
        ...     tfs=['bHLH_tnt', 'WRKY_tnt'],
        ...     signal_type='difference',
        ... )
        >>> peaks = scanner.scan(
        ...     'data/deepcis_predictions.csv',
        ...     annotator=PeakAnnotator(window_size=250, sigma=20.0),
        ... )
    """

    def __init__(
        self,
        genes: Optional[List[str]] = None,
        tfs: Optional[List[str]] = None,
        signal_type: str = "difference",
        output_path: Optional[Union[str, Path]] = None,
        annotator_window_size: int = 250,
        annotator_step_size: Optional[int] = None,
        annotator_threshold_peak: float = 0.2,
        annotator_sigma: float = 10.0,
        annotator_lambda_weight: float = 1.0,
    ):
        """Initialize peak scanner parameters.

        Args:
            genes: Genes to include (default: all from data).
            tfs: TFs to include (default: all from data).
            signal_type: 'reference', 'mutated', or 'difference' (default).
            output_path: Full CSV file path to save results (default: generated
                under data/peak_annotations/).
            save_results: Whether to save results (default: True).
            annotator_window_size: PeakAnnotator window size in bp.
            annotator_step_size: PeakAnnotator step size in bp. If None, inferred
                from each signal DataFrame.
            annotator_threshold_peak: PeakAnnotator threshold for region detection.
            annotator_sigma: PeakAnnotator Gaussian smoothing sigma in bp.
            annotator_lambda_weight: PeakAnnotator lambda mass weight.

        Raises:
            ValueError: If signal_type is invalid.
        """
        valid_types = {"reference", "optimized", "difference"}
        if signal_type not in valid_types:
            raise ValueError(
                f"signal_type must be one of {valid_types}, got '{signal_type}'"
            )

        self.genes = genes
        self.tfs = tfs
        self.signal_type = signal_type
        self.output_path = Path(output_path) if output_path else None
        self.output_dir = self.output_path.parent if self.output_path else Path("data") / "peak_annotations"
        self.annotator_window_size = annotator_window_size
        self.annotator_step_size = annotator_step_size
        self.annotator_threshold_peak = annotator_threshold_peak
        self.annotator_sigma = annotator_sigma
        self.annotator_lambda_weight = annotator_lambda_weight

    # ========================================================================
    # DATA LOADING AND VALIDATION METHODS
    # ========================================================================

    @staticmethod
    def _load_scanner_data(
        scanner_data: Union[pd.DataFrame, str, Path],
    ) -> pd.DataFrame:
        """Load deepCIS scanner output from file or DataFrame.

        Args:
            scanner_data: Either a DataFrame or file path (str/Path).

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        if isinstance(scanner_data, (str, Path)):
            if not Path(scanner_data).exists():
                raise FileNotFoundError(f"Scanner data file not found: {scanner_data}")
            return pd.read_csv(scanner_data)
        else:
            return scanner_data.copy()

    def _validate_scanner_input(
        self,
        df: pd.DataFrame,
    ) -> None:
        """Validate input data and parameters for scanning.

        Args:
            df: Input DataFrame from scanner.

        Raises:
            ValueError: If signal_type is invalid.
            KeyError: If required columns are missing.
        """
        required_cols = {"gene", "sequence_type", "window_start"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise KeyError(
                f"Missing required columns: {missing_cols}. "
                f"Available: {list(df.columns)}"
            )

        # Reject scan output from the pre-"optimized" pipeline version. Such a
        # file still labels the optimized sequence "max_mutated"; the current
        # scanner would find no "optimized" rows and silently emit near-empty
        # peaks, so fail loudly instead.
        stale_label = "max_mutated"
        if stale_label in set(df["sequence_type"].unique()):
            raise ValueError(
                f"Scanner data is stale: its 'sequence_type' column contains "
                f"'{stale_label}', the label used by a previous pipeline version "
                f"(the current pipeline expects 'optimized'). Re-run deepcis_scanner "
                f"with --overwrite to regenerate this file, or use a fresh output "
                f"folder / delete the old deepcis_scan artifacts first."
            )


    # ========================================================================
    # DATA SELECTION METHODS
    # ========================================================================

    @staticmethod
    def _get_available_tfs(df: pd.DataFrame) -> List[str]:
        """Identify TF columns (exclude metadata columns).

        Args:
            df: Input DataFrame with TF and metadata columns.

        Returns:
            List of TF column names.
        """
        metadata_cols = {
            "gene", "sequence_type", "window_start", "window_end", "contains_padding"
        }
        return [col for col in df.columns if col not in metadata_cols]

    def validate_genes(
        self,
        df: pd.DataFrame,
    ) -> List[str]:
        """Select and validate requested genes against available data.

        Args:
            df: Input DataFrame.

        Returns:
            List of selected gene names.

        Raises:
            ValueError: If requested genes not in available genes.
        """
        available_genes = df["gene"].unique().tolist()
        if self.genes is None:
            return available_genes

        missing_genes = set(self.genes) - set(available_genes)
        if missing_genes:
            raise ValueError(
                f"Requested genes not found in data: {missing_genes}. "
                f"Available: {available_genes}"
            )
        selected_genes = [g for g in self.genes if g in available_genes]
        return selected_genes
        
    
    def validate_tfs(
        self,
        df,
        available_tfs: List[str],
    ) -> List[str]:
        """selects TFs from input values and validates against available TF columns in data
 
        Args:
            df (_type_): Input DataFrame.
            available_tfs (List[str]): List of available TF column names.

        Returns:
            List[str]: selected TF column names.
        """
        if self.tfs is None:
            return available_tfs

        required_cols = set(self.tfs)
        missing_tfs = required_cols - set(available_tfs)
        if missing_tfs:
            raise ValueError(
                f"Requested TF columns not found in data: {missing_tfs}. "
                f"Available: {list(df.columns)}"
            )

        return self.tfs

    # ========================================================================
    # SIGNAL COMPUTATION METHOD
    # ========================================================================

    def _compute_difference_signal(
        self,
        gene_df: pd.DataFrame,
        tf_name: str
    ) -> Optional[pd.DataFrame]:
        ref_df = gene_df[gene_df["sequence_type"] == "reference"]
        mut_df = gene_df[gene_df["sequence_type"] == "optimized"]

        n_ref = len(ref_df)
        n_mut = len(mut_df)

        # Ensure same number of windows
        if n_ref != n_mut:
            raise ValueError(
                f"Mismatched window counts: reference={n_ref}, mutated={n_mut}. "
                f"Using minimum."
            )

        if n_ref == 0:
            raise ValueError(
                f"No reference and mutation data found for tf \"{tf_name}\""
            )

        # merge on window_start and window_end to align reference and mutated signals
        signal_df_ref = ref_df[[tf_name, "window_start", "window_end"]].copy()
        signal_df_mut = mut_df[[tf_name, "window_start", "window_end"]].copy()
        signal_df = signal_df_ref.merge(
            signal_df_mut,
            on=["window_start", "window_end"],
            how="outer",
            suffixes=("", "_mut"),
            indicator=True,
            validate="one_to_one",
        )

        # Fail if any window exists only on one side
        if not (signal_df["_merge"] == "both").all():
            mismatches = signal_df.loc[signal_df["_merge"] != "both", ["window_start", "window_end", "_merge"]]
            raise ValueError(f"Reference and mutated windows do not perfectly match for gene {gene_df['gene'].iloc[0]} and TF {tf_name}. First mismatches:\n{mismatches.head(20).to_string(index=False)}")

        signal_df["signal"] = signal_df[f"{tf_name}_mut"] - signal_df[tf_name]
        signal_df = signal_df[["signal", "window_start", "window_end"]]
        signal_df = signal_df.rename(columns={"signal": tf_name})
        return signal_df

    def _compute_gene_tf_signal(
        self,
        gene_df: pd.DataFrame,
        tf_name: str,
    ) -> Optional[pd.DataFrame]:
        """Compute signal DataFrame for a single gene-TF pair.

        Args:
            gene_df: DataFrame containing only data for one gene
                (with 'reference' and 'optimized' rows).
            tf_name: TF column name.
            signal_type: 'reference', 'mutated', or 'difference'.

        Returns:
            Signal DataFrame with columns [signal, window_start, window_end],
            or None if data is insufficient.
        """
        if self.signal_type not in ["reference", "optimized", "difference"]:
            raise ValueError(f"Invalid signal_type: {self.signal_type}")

        elif self.signal_type in["reference", "optimized"]:
            ref_df = gene_df[gene_df["sequence_type"] == self.signal_type]
            if len(ref_df) == 0:
                raise ValueError(f"No '{self.signal_type}' data for gene {gene_df['gene'].iloc[0]} and TF {tf_name}")
            return ref_df[[tf_name, "window_start", "window_end"]].copy()

        #difference
        return self._compute_difference_signal(gene_df, tf_name)
    # ========================================================================
    # PEAK DETECTION METHOD
    # ========================================================================

    def _detect_peaks_for_gene_tf(
        self,
        signal_df: pd.DataFrame,
        gene_name: str,
        tf_name: str,
    ) -> Optional[pd.DataFrame]:
        """Detect peaks for a single gene-TF pair and add metadata.

        Args:
            signal_df: Signal DataFrame with columns [signal, window_start, window_end].
            gene_name: Gene identifier.
            tf_name: Transcription factor name.
            signal_type: Type of signal analyzed.
            annotator_window_size: Window size for auto-created PeakAnnotator.
            annotator_step_size: Step size for auto-created PeakAnnotator.
            annotator_threshold_peak: Threshold for auto-created PeakAnnotator.
            annotator_sigma: Sigma for auto-created PeakAnnotator.
            annotator_lambda_weight: Lambda for auto-created PeakAnnotator.

        Returns:
            DataFrame with detected peaks (with metadata) or None if detection fails.
        """
        run_annotator = PeakAnnotator(
            df=signal_df,
            window_size=self.annotator_window_size,
            step_size=self.annotator_step_size,
            threshold_peak=self.annotator_threshold_peak,
            sigma=self.annotator_sigma,
            lambda_weight=self.annotator_lambda_weight,
        )

        peaks_df = run_annotator.detect_peaks(signal_column=tf_name)

        # Add metadata columns
        peaks_df["gene"] = gene_name
        peaks_df["tf"] = tf_name
        peaks_df["signal_type"] = self.signal_type

        # Reorder columns to standard format
        return peaks_df[[                   #type: ignore
                "gene",
                "tf",
                "signal_type",
                "region_idx",
                "peak_rank",
                "peak_start",
                "peak_end",
                "peak_middle_start",
                "peak_middle_end",
                "edge_peak",
                "peak_area",
            ]
        ]

    # ========================================================================
    # OUTPUT HANDLING METHOD
    # ========================================================================

    def _save_peaks_results(
        self,
        result_df: pd.DataFrame,
        scanner_data: Union[pd.DataFrame, str, Path],
        signal_types: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Save peak detection results to a CSV file.

        Args:
            result_df: DataFrame with detected peaks.
        """
        if output_path is not None:
            output_file = Path(output_path)
            if output_file.suffix == "":
                output_file = output_file.with_suffix(".csv")
        else:
            if isinstance(scanner_data, (str, Path)):
                base_name = Path(scanner_data).stem
                base_name += "_annotated_peaks"
            else:
                base_name = "annotated_peaks"
            output_dir = self.output_path.parent if self.output_path else self.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            signal_type_str = "_".join(sorted(signal_types)) if signal_types else self.signal_type
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"{base_name}_{signal_type_str}_{timestamp}.csv"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(result_df)} peaks to {output_file}")
    
    def _scan_all_combinations(self, df: pd.DataFrame, selected_genes: List[str], selected_tfs: List[str]) -> List[pd.DataFrame]:
        """Scan all gene-TF combinations and collect detected peaks.

        Args:
            df (pd.DataFrame): Input DataFrame with deepCIS predictions.
            selected_genes (List[str]): List of gene names to scan.
            selected_tfs (List[str]): List of TF names to scan.

        Returns:
            List[pd.DataFrame]: List of DataFrames with detected peaks for each gene-TF combination.
        """
        all_peaks = []

        for gene_name in tqdm(selected_genes):
            gene_df = df[df["gene"] == gene_name].copy().reset_index(drop=True)

            for tf_name in selected_tfs:
                # Compute signal for this gene-TF pair
                try:
                    signal_df = self._compute_gene_tf_signal(gene_df, tf_name) #type:ignore
                except Exception as e:
                    logger.warning(f"Failed to compute signal for {gene_name}/{tf_name}: {e}")
                    continue

                if signal_df is None or len(signal_df) == 0:
                    logger.warning(f"No signal data for {gene_name}/{tf_name}, skipping peak detection.")
                    continue

                # Detect peaks and add metadata
                try:
                    peaks_df = self._detect_peaks_for_gene_tf(
                        signal_df,
                        gene_name,
                        tf_name,
                    )
                except Exception as e:
                    logger.warning(f"Failed to detect peaks for {gene_name}/{tf_name}: {e}")
                    continue

                all_peaks.append(peaks_df)

        return all_peaks


    # ========================================================================
    # PUBLIC API METHOD
    # ========================================================================

    def scan(
        self,
        scanner_data: Union[pd.DataFrame, str, Path],
    ) -> pd.DataFrame:
        """Scan deepCIS predictions for peaks in TF binding.

        Orchestrates peak detection across multiple genes and TFs using the
        provided PeakAnnotator (or creates a default one if not provided).

        Args:
            scanner_data: Either a DataFrame or file path to deepCIS predictions.

        Returns:
            DataFrame with detected peaks (columns: gene, tf, signal_type,
            peak_start, peak_end, score, region_idx, peak_rank).

        Raises:
            FileNotFoundError: If scanner_data is a path that doesn't exist.
            ValueError: If signal_type or other parameters are invalid.
            KeyError: If required columns are missing from data.

        Example:
            >>> scanner = DeepCISPeakScanner(genes=['gene1'])  # doctest: +SKIP
            >>> peaks = scanner.scan('data/predictions.csv')
            >>> print(f"Found {len(peaks)} peaks")
        """
        # ======== Step 1: Load and validate input ========
        df = self._load_scanner_data(scanner_data)
        self._validate_scanner_input(df)

        available_tfs = self._get_available_tfs(df)
        selected_genes = self.validate_genes(df)
        selected_tfs = self.validate_tfs(df, available_tfs)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        all_peaks = self._scan_all_combinations(df, selected_genes, selected_tfs)
        
        if all_peaks:
            result_df = pd.concat(all_peaks, ignore_index=True)
        else:
            logger.warning("No peaks detected for any gene-TF combination.")
            result_df = pd.DataFrame(
                columns=[
                    "gene", "tf", "signal_type", "peak_start", "peak_end",
                    "score", "region_idx", "peak_rank"
                ]
            )

        return result_df

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def __repr__(self) -> str:
        """String representation showing configuration."""
        return (
            f"DeepCISPeakScanner(genes={self.genes}, tfs={self.tfs}, "
            f"signal_type='{self.signal_type}', output_path={self.output_path}, "
            f"annotator_window_size={self.annotator_window_size}, "
            f"annotator_step_size={self.annotator_step_size}, "
            f"annotator_threshold_peak={self.annotator_threshold_peak}, "
            f"annotator_sigma={self.annotator_sigma}, "
            f"annotator_lambda_weight={self.annotator_lambda_weight})"
        )

def run(scanner_data: Union[pd.DataFrame, str, Path], output_path: Union[Path, str], genes: Optional[List[str]] = None,
        tfs: Optional[List[str]] = None, signal_types: List[str] = ["difference"], window_size: int = 250, step_size: Optional[int] = None,
        threshold_peak: float = 0.2, sigma: float = 50.0, lambda_weight: float = 2.0,) -> pd.DataFrame:
    all_results: List[pd.DataFrame] = []

    for signal_type in signal_types:
        logger.info("Running peak scan for signal_type=%s", signal_type)
        scanner = DeepCISPeakScanner(
            genes=_parse_list_arg(genes),
            tfs=_parse_list_arg(tfs),
            signal_type=cast(str, signal_type),
            output_path=output_path,
            annotator_window_size=window_size,
            annotator_step_size=step_size,
            annotator_threshold_peak=threshold_peak,
            annotator_sigma=sigma,
            annotator_lambda_weight=lambda_weight,
        )

        result_df = scanner.scan(scanner_data=scanner_data)
        all_results.append(result_df)
        logger.info("Detected %s peaks for signal_type=%s", len(result_df), signal_type)
    all_peaks = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    scanner._save_peaks_results(all_peaks, scanner_data, signal_types=signal_types, output_path=output_path)
    return all_peaks


def _parse_list_arg(values: Optional[List[str]]) -> Optional[List[str]]:
    """Parse list arguments from repeated args and comma-separated values."""
    if values is None:
        return None

    parsed: List[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed or None


def _build_parser() -> argparse.ArgumentParser:
    """Create command line parser for DeepCIS peak scanning."""
    parser = argparse.ArgumentParser(description=("Scan deepCIS predictions for TF binding peaks with configurable scanner and PeakAnnotator parameters."))

    parser.add_argument( "scanner_data", help="Path to deepCIS prediction CSV file.",)
    parser.add_argument( "--genes", "-g", nargs="+", default=None, help="Genes to scan (space-separated and/or comma-separated).",)
    parser.add_argument( "--tfs", "-t", nargs="+", default=None, help="TF columns to scan (space-separated and/or comma-separated).",)
    parser.add_argument("--signal-type", "-s", choices=["reference", "optimized", "difference", "all"], default="difference",
        help=( "Which signal to scan (default: difference). " "Use 'all' to run reference, optimized, and difference in one call."),
    )
    parser.add_argument( "--output-path", "-o", default="", help="Full output CSV path for results (default: next to input file with annotated_peaks in the name).",)

    parser.add_argument( "--annotator-window-size", type=int, default=250, help="PeakAnnotator window size in bp (default: 250).",)
    parser.add_argument( "--annotator-step-size", type=int, default=None, help=( "PeakAnnotator step size in bp. If omitted, inferred from window_start for each gene/TF signal."),)
    parser.add_argument( "--annotator-threshold-peak", type=float, default=0.2, help="PeakAnnotator detection threshold (default: 0.2).",)
    parser.add_argument( "--annotator-sigma", type=float, default=50.0, help="PeakAnnotator Gaussian sigma in bp (default: 50.0).",)
    parser.add_argument( "--annotator-lambda-weight", type=float, default=2.0, help="PeakAnnotator lambda weight (default: 2.0).",)
    parser.add_argument( "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Logging level (default: INFO).",)

    return parser


def main() -> None:
    """Run peak scanning from the command line."""
    parser = _build_parser()
    args = parser.parse_args()
    args.output_path = Path(args.output_path) if args.output_path else None

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    signal_types = ["reference", "optimized", "difference"] if args.signal_type == "all" else [args.signal_type]
    if args.output_path is None:
        base_name = Path(args.scanner_data).stem
        signal_type_str = "_".join(sorted(signal_types))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = Path(args.scanner_data).parent / "peak_annotations" / f"{base_name}_annotated_peaks_{signal_type_str}_{timestamp}.csv"

    all_peaks = run(
        scanner_data=args.scanner_data,
        output_path=args.output_path,
        genes=args.genes,
        tfs=args.tfs,
        signal_types=signal_types,
        window_size=args.annotator_window_size,
        step_size=args.annotator_step_size,
        threshold_peak=args.annotator_threshold_peak,
        sigma=args.annotator_sigma,
        lambda_weight=args.annotator_lambda_weight,
    )
    logger.info("Detected %s peaks in total across %s signal type(s)", len(all_peaks), len(signal_types))


if __name__ == "__main__":
    main()
