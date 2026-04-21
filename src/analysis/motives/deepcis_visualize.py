"""Visualization tools for deepCIS scanning results."""
import json
import traceback
import argparse
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from analysis.utils.io import print_status


PEAK_SIGNAL_TYPES = ("reference", "max_mutated", "difference")
DIFFERENCE_DIRECTIONS = ("max_mutated_minus_reference", "reference_minus_max_mutated")
PEAK_BACKGROUND_COLORS = {
    "reference": "#999999",
    "max_mutated": "#9999ff",
    "difference": "#ff9999",
}

PEAK_REQUIRED_COLUMNS = {
    "gene",
    "tf",
    "signal_type",
    "peak_start",
    "peak_end",
}


def _resolve_peak_signal_types(
    df: pd.DataFrame,
    peak_df: Optional[pd.DataFrame],
    highlight_peaks: bool,
    peak_signal_types: Optional[List[str]],
) -> List[str]:
    """Resolve which peak signal types are available for plotting."""
    if peak_signal_types is not None:
        return [signal_type for signal_type in peak_signal_types if signal_type in PEAK_SIGNAL_TYPES]

    if not highlight_peaks:
        return []

    if peak_df is not None:
        available_signal_types = [
            signal_type
            for signal_type in PEAK_SIGNAL_TYPES
            if signal_type in set(peak_df["signal_type"].astype(str).unique())
        ]
    else:
        available_signal_types = [
            signal_type
            for signal_type in PEAK_SIGNAL_TYPES
            if any(col.endswith(f"__{signal_type}__in_peak") for col in df.columns)
        ]

    return available_signal_types


def _select_random_subset(genes: List[str], tfs: List[str], limit: int = 3) -> Tuple[List[str], List[str]]:
    """Select up to ``limit`` random genes and TFs."""
    selected_genes = random.sample(genes, k=min(limit, len(genes)))
    selected_tfs = random.sample(tfs, k=min(limit, len(tfs)))
    return selected_genes, selected_tfs


def load_scan_results(
    scan_path: str,
) -> pd.DataFrame:
    """Load deepCIS scan results from a CSV file.

    Args:
        scan_path: Path to the CSV file containing scan results.

    Returns:
        DataFrame with scan results.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(scan_path):
        raise FileNotFoundError(f"Scan results file not found: {scan_path}")
    
    df = pd.read_csv(scan_path)
    return df


def load_peak_results(
    peak_path: str,
) -> pd.DataFrame:
    """Load peak annotation results from a CSV file.

    Args:
        peak_path: Path to the CSV file containing peak annotations from
            peak_scanner.py.

    Returns:
        DataFrame with peak annotations.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(peak_path):
        raise FileNotFoundError(f"Peak results file not found: {peak_path}")

    df = pd.read_csv(peak_path)
    missing_cols = PEAK_REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise KeyError(
            f"Peak results file missing required columns: {sorted(missing_cols)}"
        )
    return df


def get_tf_columns(df: pd.DataFrame) -> List[str]:
    """Extract all TF family columns from the DataFrame.

    TF columns are identified by excluding fixed metadata columns.
    This approach is robust to any TF naming scheme.

    Fixed columns (excluded):
        - gene
        - sequence_type
        - window_start
        - window_end
        - contains_padding

    Args:
        df: DataFrame with scan results.

    Returns:
        Sorted list of TF column names.
    """
    fixed_columns = {
        "gene",
        "sequence_type",
        "window_start",
        "window_end",
        "contains_padding",
        "window_id",
    }
    tf_cols = [
        col
        for col in df.columns
        if col not in fixed_columns and "__" not in col
    ]
    return sorted(tf_cols)
    
# Calculate window center positions: (start + end) / 2
def get_centers(data: pd.DataFrame) -> np.ndarray:
    # Adding 1 to adjust for the fact that the end index is not included, as well as conerting from 0-based to 1-based coordinates
    return ((data["window_start"] + data["window_end"] + 1) / 2).values


def extract_plot_data(
    df_subset: pd.DataFrame,
    tf_col: str,
    difference_direction: str = "max_mutated_minus_reference",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Extract and transform data from DataFrame to plottable coordinates.

    Separates reference and mutated sequences, calculates window center positions,
    and extracts the corresponding values for a given TF column.

    Args:
        df_subset: DataFrame filtered to contain only data for one gene and one TF.
        tf_col: Name of the TF column to extract values from.
        difference_direction: Direction for signed difference:
            "max_mutated_minus_reference" or "reference_minus_max_mutated".

    Returns:
        Tuple of (ref_x, ref_y, mut_x, mut_y, diff_x, diff_y, x_min, x_max) where:
        * ref_x, mut_x: Window center positions (in bp) for reference and mutated
        * ref_y, mut_y: Prediction scores for reference and mutated
                * diff_x, diff_y: Signed difference coordinates based on
                    ``difference_direction``
        * x_min, x_max: Global min/max genomic positions (window_start/window_end)
    """
    # Separate reference and mutated sequences
    ref_data = df_subset[df_subset["sequence_type"] == "reference"].sort_values(["window_start"])       #type: ignore
    mut_data = df_subset[df_subset["sequence_type"] == "max_mutated"].sort_values(["window_start"])     #type: ignore 

    ref_x = get_centers(ref_data) if len(ref_data) > 0 else np.array([])
    ref_y = ref_data[tf_col].values if len(ref_data) > 0 else np.array([])

    mut_x = get_centers(mut_data) if len(mut_data) > 0 else np.array([])
    mut_y = mut_data[tf_col].values if len(mut_data) > 0 else np.array([])

    # Compute signed difference from aligned windows.
    diff_data = ref_data[["window_start", "window_end", tf_col]].merge(
        mut_data[["window_start", "window_end", tf_col]],
        on=["window_start", "window_end"],
        how="inner",
        suffixes=("_ref", "_mut"),
    )
    diff_x = get_centers(diff_data) if len(diff_data) > 0 else np.array([])
    if len(diff_data) > 0:
        if difference_direction == "max_mutated_minus_reference":
            diff_y = diff_data[f"{tf_col}_mut"].to_numpy() - diff_data[f"{tf_col}_ref"].to_numpy()
        elif difference_direction == "reference_minus_max_mutated":
            diff_y = diff_data[f"{tf_col}_ref"].to_numpy() - diff_data[f"{tf_col}_mut"].to_numpy()
        else:
            raise ValueError(
                f"Invalid difference_direction '{difference_direction}'. "
                f"Expected one of {DIFFERENCE_DIRECTIONS}."
            )
    else:
        diff_y = np.array([])

    # Determine x-axis range (from min window_start to max window_end)
    x_min = df_subset["window_start"].min()
    x_max = df_subset["window_end"].max()

    return ref_x, ref_y, mut_x, mut_y, diff_x, diff_y, x_min, x_max


def _get_padding_regions(df_subset: pd.DataFrame) -> List[Tuple[float, float]]:
    """Extract and merge padding regions from the data.

    Args:
        df_subset: DataFrame with sequence data.

    Returns:
        List of (start, end) tuples for padded regions, sorted and deduplicated.
    """
    if "contains_padding" not in df_subset.columns:
        return []

    padding_regions = []

    # Extract padding regions from reference
    ref_padded = df_subset[
        (df_subset["sequence_type"] == "reference") &
        (df_subset["contains_padding"] == True)
    ]
    if len(ref_padded) > 0:
        padding_regions.extend(zip(ref_padded["window_start"], ref_padded["window_end"]))

    # Extract padding regions from mutated
    mut_padded = df_subset[
        (df_subset["sequence_type"] == "max_mutated") &
        (df_subset["contains_padding"] == True)
    ]
    if len(mut_padded) > 0:
        padding_regions.extend(zip(mut_padded["window_start"], mut_padded["window_end"]))

    # Merge overlapping regions
    return sorted(set(padding_regions)) if padding_regions else []


def _add_padding_background(ax: plt.Axes, padding_regions: List[Tuple[float, float]]) -> None:
    """Add light grey background for padded regions.

    Args:
        ax: Matplotlib axes to plot on.
        padding_regions: List of (start, end) tuples for padded regions.
    """
    for start, end in padding_regions:
        ax.axvspan(start, end, alpha=0.2, color="grey", zorder=0)


def _get_padding_edge_positions(df_subset: pd.DataFrame) -> List[float]:
    """Return center positions for first/last windows containing padding."""
    padding_regions = _get_padding_regions(df_subset)
    if not padding_regions:
        return []

    centers = sorted({(start + end + 1.0) / 2.0 for start, end in padding_regions})
    if len(centers) == 1:
        return [centers[0]]
    return [centers[0], centers[-1]]


def _add_vertical_markers(
    ax: plt.Axes,
    markers: List[Tuple[float, str]],
    color: str,
    linestyle: str,
    linewidth: float,
) -> bool:
    """Draw vertical markers and hide them from the automatic legend."""
    marker_drawn = False
    for x_pos, marker_name in markers:
        ax.axvline(
            x=x_pos,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.9,
            zorder=1,
            label="_nolegend_",
        )
        marker_drawn = True
    return marker_drawn


def _create_legend_proxy(label: str, color: str, linestyle: str, linewidth: float) -> Line2D:
    """Create a proxy artist for custom legend entries."""
    return Line2D([0], [0], color=color, linestyle=linestyle, linewidth=linewidth, label=label)


def _merge_intervals(regions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping or touching intervals."""
    if not regions:
        return []

    sorted_regions = sorted({(float(start), float(end)) for start, end in regions})
    merged: List[List[float]] = [[sorted_regions[0][0], sorted_regions[0][1]]]

    for start, end in sorted_regions[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def _get_peak_background_regions(
    df_subset: pd.DataFrame,
    tf_col: str,
    signal_types: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[float, float]]]:
    """Extract merged background regions for annotated peaks.

    Uses the window-centric merged output, so peak backgrounds are drawn as
    spans across the windows that overlap peaks rather than exact peak bounds.
    """
    if signal_types is None:
        signal_types = list(PEAK_SIGNAL_TYPES)

    background_regions: Dict[str, List[Tuple[float, float]]] = {}
    for signal_type in signal_types:
        in_peak_col = f"{tf_col}__{signal_type}__in_peak"
        if in_peak_col not in df_subset.columns:
            continue

        peak_windows = df_subset[df_subset[in_peak_col] == True]
        intervals = list(zip(peak_windows["window_start"], peak_windows["window_end"]))
        if intervals:
            background_regions[signal_type] = _merge_intervals(intervals)

    return background_regions


def _get_peak_background_regions_from_peaks(
    peak_df: pd.DataFrame,
    gene: str,
    tf_col: str,
    signal_types: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[float, float]]]:
    """Extract merged background regions from raw peak annotations.

    This uses the direct peak start/end coordinates emitted by peak_scanner.py
    instead of deriving highlighted windows from merged window annotations.
    """
    if signal_types is None:
        signal_types = [
            signal_type
            for signal_type in PEAK_SIGNAL_TYPES
            if signal_type in set(peak_df["signal_type"].astype(str).unique())
        ]

    background_regions: Dict[str, List[Tuple[float, float]]] = {}
    tf_peaks = peak_df[(peak_df["gene"] == gene) & (peak_df["tf"] == tf_col)]

    for signal_type in signal_types:
        signal_peaks = tf_peaks[tf_peaks["signal_type"] == signal_type]
        intervals = list(zip(signal_peaks["peak_start"] + 125, signal_peaks["peak_end"] + 125))
        if intervals:
            background_regions[signal_type] = _merge_intervals(intervals)

    return background_regions


def _add_peak_background(
    ax: plt.Axes,
    peak_regions: Dict[str, List[Tuple[float, float]]],
) -> None:
    """Add colored background spans for annotated peaks."""
    for signal_type, regions in peak_regions.items():
        color = PEAK_BACKGROUND_COLORS.get(signal_type, "#ff7f0e")
        for start, end in regions:
            ax.axvspan(start, end, alpha=0.12, color=color, zorder=1)


def _plot_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    label: str,
    color: str,
) -> None:
    """Plot a single line on the axes.

    Args:
        ax: Matplotlib axes to plot on.
        x: X-axis coordinates (window centers).
        y: Y-axis values (prediction scores).
        label: Label for the line (for legend).
        color: Line color (e.g., "black" or "blue").
        marker: Marker style (e.g., "o" for circles, "s" for squares).
    """
    if len(x) > 0:
        ax.plot(
            x,
            y,
            linestyle="-",
            color=color,
            linewidth=2,
            markersize=6,
            label=label,
            zorder=2,
        )


def _set_axis_properties(
    ax: plt.Axes,
    x_min: float,
    x_max: float,
    gene: str,
    tf_col: str,
    include_title: bool,
    show_difference: bool,
    show_padding_marker: bool,
    show_tss_tts_marker: bool,
) -> None:
    """Set axis limits, labels, title, and formatting.

    Args:
        ax: Matplotlib axes to configure.
        x_min: Minimum genomic position (from minimum window_start).
        x_max: Maximum genomic position (from maximum window_end).
        gene: Gene name (for title).
        tf_col: TF column name (for title).
        include_title: Whether to include a title.
    """
    # Set x-axis limits based on data range
    ax.set_xlim(x_min, x_max)
    if show_difference:
        ax.set_ylim(-1, 1)
    else:
        ax.set_ylim(0, 1)

    # Set labels
    ax.set_xlabel("Genomic Position (bp)")
    if show_difference:
        ax.set_ylabel("Binding Signal / Difference")
    else:
        ax.set_ylabel("Binding Prediction Score")

    # Set title
    if include_title:
        tf_name = tf_col.replace("tf_", "TF_")
        ax.set_title(f"{gene} - {tf_name}")

    # Additional formatting
    line_handles, line_labels = ax.get_legend_handles_labels()
    ordered_handles = []
    ordered_labels = []
    for handle, label in zip(line_handles, line_labels):
        if label and not label.startswith("_"):
            ordered_handles.append(handle)
            ordered_labels.append(label)

    if show_padding_marker:
        ordered_handles.append(_create_legend_proxy("Area containing Padding", "#7a7a7a", ":", 1.8))
        ordered_labels.append("Area containing Padding")

    if show_tss_tts_marker:
        ordered_handles.append(_create_legend_proxy("TSS/TTS", "#2f4f4f", "--", 1.4))
        ordered_labels.append("TSS/TTS")

    if ordered_handles:
        ax.legend(ordered_handles, ordered_labels, loc="best")
    ax.grid(True, alpha=0.3)


def _plot_gene_tf(
    df_subset: pd.DataFrame,
    gene: str,
    tf_col: str,
    ax: plt.Axes,
    include_title: bool = True,
    highlight_padding: bool = False,
    show_tss_tts: bool = True,
    tss_position: float = 1000.0,
    tts_position: float = 2020.0,
    highlight_peaks: bool = False,
    peak_signal_types: Optional[List[str]] = None,
    peak_df: Optional[pd.DataFrame] = None,
    show_difference: bool = True,
    difference_direction: str = "max_mutated_minus_reference",
) -> None:
    """Plot predictions for a single gene and TF family.

    Args:
        df_subset: DataFrame filtered to contain only data for this gene and TF.
        gene: Gene name (for title).
        tf_col: Name of the TF column to plot.
        ax: Matplotlib axes to plot on.
        include_title: Whether to include a title on the plot.
    """
    # Extract plot data using the transformation function
    ref_x, ref_y, mut_x, mut_y, diff_x, diff_y, x_min, x_max = extract_plot_data(
        df_subset,
        tf_col,
        difference_direction=difference_direction,
    )

    # Add optional background highlights before the lines.
    show_padding_marker = False
    if highlight_padding:
        padding_edges = _get_padding_edge_positions(df_subset)
        padding_markers: List[Tuple[float, str]] = []
        if len(padding_edges) >= 1:
            padding_markers.append((padding_edges[0], "Padding Start"))
        if len(padding_edges) >= 2:
            padding_markers.append((padding_edges[1], "Padding End"))
        show_padding_marker = _add_vertical_markers(
            ax,
            padding_markers,
            color="#7a7a7a",
            linestyle=":",
            linewidth=1.8,
        )

    show_tss_tts_marker = False
    if show_tss_tts:
        show_tss_tts_marker = _add_vertical_markers(
            ax,
            [(tss_position, "TSS"), (tts_position, "TTS")],
            color="#2f4f4f",
            linestyle="--",
            linewidth=1.4,
        )

    if highlight_peaks:
        if peak_df is not None:
            peak_regions = _get_peak_background_regions_from_peaks(
                peak_df,
                gene,
                tf_col,
                signal_types=peak_signal_types,
            )
        else:
            peak_regions = _get_peak_background_regions(
                df_subset,
                tf_col,
                signal_types=peak_signal_types,
            )
        _add_peak_background(ax, peak_regions)

    # Plot reference and mutated lines
    _plot_line(ax, ref_x, ref_y, "Reference", "black")
    _plot_line(ax, mut_x, mut_y, "Max Mutated", "blue")
    if show_difference:
        if difference_direction == "max_mutated_minus_reference":
            diff_label = "Difference (max_mutated - reference)"
        else:
            diff_label = "Difference (reference - max_mutated)"
        _plot_line(ax, diff_x, diff_y, diff_label, "red")

    # Set axis properties
    _set_axis_properties(
        ax,
        x_min,
        x_max,
        gene,
        tf_col,
        include_title,
        show_difference,
        show_padding_marker,
        show_tss_tts_marker,
    )


def _validate_and_set_defaults(
    df: pd.DataFrame,
    genes: Optional[List[str]],
    tfs: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    """Validate and set default values for genes and TFs.

    Args:
        df: DataFrame with scan results.
        genes: User-provided genes (may be None).
        tfs: User-provided TFs (may be None).

    Returns:
        Tuple of (validated_genes, validated_tfs).

    Raises:
        ValueError: If no valid genes or TFs are found.
    """
    # Set defaults
    if genes is None:
        genes = sorted(df["gene"].unique().tolist())
    if tfs is None:
        tfs = get_tf_columns(df)

    # Validate genes exist
    available_genes = set(df["gene"].unique())
    for gene in genes:
        if gene not in available_genes:
            raise ValueError(f"Gene '{gene}' not found in data.") # Available genes: {json.dumps(sorted(list(available_genes)), indent=2)}")

    # Validate TFs exist
    available_tfs = set(get_tf_columns(df))
    for tf in tfs:
        if tf not in available_tfs:
            raise ValueError(f"TF column '{tf}' not found in data. Available TFs: {json.dumps(sorted(list(available_tfs)), indent=2)}")

    return genes, tfs


def _save_gene_tf_plot(
    fig: plt.Figure,                    #type: ignore
    gene: str,
    tf_col: str,
    output_dir: str,
    output_format: str,
    signal_type: Optional[str] = None,
) -> None:
    """Save a plot figure to disk.

    Args:
        fig: Matplotlib figure to save.
        gene: Gene name (for filename).
        tf_col: TF column name (for filename).
        output_dir: Directory to save plot in.
        output_format: File format (e.g., "png", "pdf").
    """
    tf_name = tf_col.replace("tf_", "TF_")
    if signal_type is None:
        filename = f"{gene}_{tf_name}.{output_format}"
    else:
        filename = f"{gene}_{tf_name}_{signal_type}.{output_format}"
    filepath = os.path.join(output_dir, gene, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _process_gene_tf(
    gene_df: pd.DataFrame,
    gene: str,
    tf_col: str,
    output_dir: str,
    output_format: str,
    include_title: bool,
    highlight_padding: bool,
    show_tss_tts: bool,
    tss_position: float,
    tts_position: float,
    highlight_peaks: bool,
    peak_signal_types: Optional[List[str]],
    peak_df: Optional[pd.DataFrame],
    show_difference: bool,
    difference_direction: str,
    signal_type: Optional[str] = None,
) -> bool:
    """Create and save a single gene/TF plot.

    Args:
        gene_df: DataFrame filtered to this gene.
        gene: Gene name.
        tf_col: TF column name.
        output_dir: Directory to save plot.
        output_format: Output file format.
        include_title: Whether to include title.

    Returns:
        True if plot was created successfully, False otherwise (e.g., no data).
    """
    # Skip if no data
    if gene_df[tf_col].isna().all():
        return False

    if signal_type is not None and signal_type not in PEAK_SIGNAL_TYPES:
        raise ValueError(f"Invalid signal_type '{signal_type}'. Expected one of {PEAK_SIGNAL_TYPES}.")

    # Create figure and plot
    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        _plot_gene_tf(
            gene_df,
            gene,
            tf_col,
            ax,
            include_title=include_title,
            highlight_padding=highlight_padding,
            show_tss_tts=show_tss_tts,
            tss_position=tss_position,
            tts_position=tts_position,
            highlight_peaks=highlight_peaks,
            peak_signal_types=peak_signal_types,
            peak_df=peak_df,
            show_difference=show_difference,
            difference_direction=difference_direction,
        )
        _save_gene_tf_plot(fig, gene, tf_col, output_dir, output_format, signal_type=signal_type)
        return True
    except Exception as e:
        print_status(f"Error plotting {gene} / {tf_col}: {e}", "WARNING")
        plt.close(fig)
        return False


def _load_input_data(input: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Load input data from file path or DataFrame.

    Args:
        input: Path to CSV file or pandas DataFrame.

    Returns:
        Loaded DataFrame.

    Raises:
        ValueError: If input is invalid type or DataFrame is empty.
        FileNotFoundError: If CSV file does not exist.
    """
    if isinstance(input, str):
        return load_scan_results(input)
    elif isinstance(input, pd.DataFrame):
        if input.empty:
            raise ValueError("Input DataFrame is empty")
        return input
    else:
        raise ValueError(
            f"input must be a string (file path) or DataFrame, got {type(input)}"
        )


def _load_peak_input_data(peak_input: Union[str, pd.DataFrame, None]) -> Optional[pd.DataFrame]:
    """Load peak annotations from file or DataFrame when provided."""
    if peak_input is None:
        return None
    if isinstance(peak_input, str):
        return load_peak_results(peak_input)
    if isinstance(peak_input, pd.DataFrame):
        if peak_input.empty:
            raise ValueError("Peak input DataFrame is empty")
        missing_cols = PEAK_REQUIRED_COLUMNS - set(peak_input.columns)
        if missing_cols:
            raise KeyError(
                f"Peak input DataFrame missing required columns: {sorted(missing_cols)}"
            )
        return peak_input
    raise ValueError(
        f"peak_input must be a string (file path), DataFrame, or None, got {type(peak_input)}"
    )


def _resolve_output_directory(input: Union[str, pd.DataFrame], output_dir: Optional[str]) -> str:
    """Resolve the output directory path.

    Args:
        input: Original input path (string) or None if input was DataFrame.
        output_dir: User-provided output directory or None.

    Returns:
        Output directory path, or None if not determinable from input.
    """
    if output_dir is None:
        if isinstance(input, str):
            input_dir = os.path.dirname(input)
            output_dir = os.path.join(input_dir, "deepcis_scan_plots")
        else:
            output_dir = os.path.join(os.getcwd(), "deepcis_scan_plots")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def visualize_scan_results(
    input: Union[str, pd.DataFrame],
    genes: Optional[List[str]] = None,
    tfs: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    output_format: str = "png",
    include_title: bool = True,
    highlight_padding: bool = True,
    show_tss_tts: bool = True,
    tss_position: float = 1000.0,
    tts_position: float = 2020.0,
    highlight_peaks: bool = False,
    peak_signal_types: Optional[List[str]] = None,
    show_difference: bool = True,
    difference_direction: str = "max_mutated_minus_reference",
    random_subset: bool = False,
    *,
    peak_input: Union[str, pd.DataFrame, None] = None,
) -> None:
    """Main entry point for visualization.

    Can accept either a path to a CSV file or a pandas DataFrame directly.

    Args:
        input: Path to CSV file with scan results, or a DataFrame.
        peak_input: Optional path or DataFrame containing raw peak annotations
            from peak_scanner.py.
        genes: List of gene names to plot. If None, plots all.
        tfs: List of TF column names to plot. If None, plots all.
        output_dir: Directory to save plots. If None, defaults to "plots"
            next to the input file (for file input) or current working directory (for DataFrame input).
        output_format: File format for saving plots. Default: "png".
        include_title: Whether to include titles in plots. Default: True.
        highlight_padding: Whether to mark padding boundaries. Default: True.
        show_tss_tts: Whether to show TSS/TTS vertical marker lines. Default: True.
        tss_position: X-coordinate for TSS marker. Default: 1000.
        tts_position: X-coordinate for TTS marker. Default: 2020.
        show_difference: Whether to display the difference line. Default: True.
        difference_direction: Direction of difference line computation.
        random_subset: If True, sample up to 3 genes and 3 TFs from the
            compatible input before plotting.

    Raises:
        ValueError: If input is invalid type or DataFrame is empty.
    """
    # Load data
    df = _load_input_data(input)
    peak_df = _load_peak_input_data(peak_input)

    # Resolve output directory
    output_dir = _resolve_output_directory(input, output_dir)

    # Validate inputs and set defaults
    genes, tfs = _validate_and_set_defaults(df, genes, tfs)

    if random_subset:
        genes, tfs = _select_random_subset(genes, tfs)
        print_status(
            f"Random subset selected {len(genes)} genes and {len(tfs)} TFs",
            "INFO",
        )

    resolved_peak_signal_types = _resolve_peak_signal_types(
        df,
        peak_df,
        highlight_peaks,
        peak_signal_types,
    )
    separate_signal_images = len(resolved_peak_signal_types) > 1

    if separate_signal_images:
        print_status(
            f"Plotting {len(resolved_peak_signal_types)} signal types as separate images",
            "INFO",
        )

    # Create plots
    successful_plots = 0
    for gene in tqdm(genes, desc="Processing genes"):
        gene_df: pd.DataFrame = df.loc[df["gene"] == gene].copy()       #type:ignore

        for tf_col in tfs:
            if separate_signal_images:
                for signal_type in resolved_peak_signal_types:
                    if _process_gene_tf(
                        gene_df,
                        gene,
                        tf_col,
                        output_dir,
                        output_format,
                        include_title,
                        highlight_padding,
                        show_tss_tts,
                        tss_position,
                        tts_position,
                        highlight_peaks,
                        [signal_type],
                        peak_df,
                        show_difference,
                        difference_direction,
                        signal_type=signal_type,
                    ):
                        successful_plots += 1
            else:
                if _process_gene_tf(
                    gene_df,
                    gene,
                    tf_col,
                    output_dir,
                    output_format,
                    include_title,
                    highlight_padding,
                    show_tss_tts,
                    tss_position,
                    tts_position,
                    highlight_peaks,
                    resolved_peak_signal_types,
                    peak_df,
                    show_difference,
                    difference_direction,
                ):
                    successful_plots += 1

    print_status(
        f"Successfully saved {successful_plots} plots to {output_dir}",
        "SUCCESS",
    )


def parse_arguments(args=None):
    """Parse command-line arguments for deepCIS visualization.

    Args:
        args: List of argument strings to parse (for testing).
              If None, uses sys.argv.

    Returns:
        argparse.Namespace with parsed arguments.

    Raises:
        SystemExit: On invalid arguments.
    """
    parser = argparse.ArgumentParser(
        description="Visualize deepCIS sliding-window predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all genes and TFs
  python -m analysis.motives.deepcis_visualize \\
    --input data/deepcis_window_scan_results.csv \\
    --output plots

  # Visualize specific genes and TFs
  python -m analysis.motives.deepcis_visualize \\
    --input data/deepcis_window_scan_results.csv \\
    --genes AT1G02130 AT1G02150 \\
    --tfs tf_0 tf_1 tf_2 \\
    --output my_plots \\
    --format pdf

  # Save without titles
  python -m analysis.motives.deepcis_visualize \\
    --input data/deepcis_window_scan_results.csv \\
    --output plots \\
    --no-title

    # Place TSS/TTS markers at custom coordinates
    python -m analysis.motives.deepcis_visualize \
        --input data/deepcis_window_scan_results.csv \
        --tss-position 1000 \
        --tts-position 2020

    # Plot a random subset of compatible genes and TFs
    python -m analysis.motives.deepcis_visualize \
        --input data/deepcis_window_scan_results.csv \
        --random-subset

    # Highlight peaks from peak_scanner.py output
    python -m analysis.motives.deepcis_visualize \
        --input data/deepcis_window_scan_results.csv \
        --peaks data/peak_annotations/deepcis_predictions_difference.csv \
        --output plots \
        --highlight-peaks \
        --peak-signals reference difference
        """,
    )

    # Add arguments
    parser.add_argument("--input", type=str, required=True, metavar="PATH", help="Path to CSV file with deepCIS scan results",)
    parser.add_argument("--peaks", type=str, default=None, metavar="PATH", help="Optional CSV file with raw peak annotations from peak_scanner.py",)
    parser.add_argument("--output", type=str, default=None, metavar="DIR", help="Directory to save plots (default: 'plots' next to input file)",)
    parser.add_argument("--genes", type=str, nargs="+", default=None, metavar="GENE", help="Gene names to plot (default: all genes)",)
    parser.add_argument("--tfs", type=str, nargs="+", default=None, metavar="TF", help="TF columns to plot, e.g., tf_0 tf_1 (default: all TFs)",)
    parser.add_argument(
        "--random-subset",
        action="store_true",
        help="Randomly sample up to 3 compatible genes and 3 compatible TFs before plotting.",
    )
    parser.add_argument("--format", type=str, default="png", metavar="FORMAT", choices=["png", "pdf", "svg", "jpg", "jpeg"],
                        help="Output file format (default: png)",)
    parser.add_argument("--no-title", action="store_true", help="Do not include titles in plots",)

    padding_group = parser.add_mutually_exclusive_group()
    padding_group.add_argument("--highlight-padding", dest="highlight_padding", action="store_true", help="Show vertical boundary markers for windows affected by padding.",)
    padding_group.add_argument("--no-highlight-padding", dest="highlight_padding", action="store_false", help="Hide vertical boundary markers for windows affected by padding.",)

    tss_tts_group = parser.add_mutually_exclusive_group()
    tss_tts_group.add_argument("--show-tss-tts", dest="show_tss_tts", action="store_true", help="Show vertical lines for TSS and TTS markers (default).",)
    tss_tts_group.add_argument("--no-show-tss-tts", dest="show_tss_tts", action="store_false", help="Hide vertical lines for TSS and TTS markers.",)

    parser.add_argument("--tss-position", type=float, default=1000.0, metavar="BP", help="X-coordinate for TSS marker (default: 1000)",)
    parser.add_argument("--tts-position", type=float, default=2020.0, metavar="BP", help="X-coordinate for TTS marker (default: 2020)",)

    peak_group = parser.add_mutually_exclusive_group()
    peak_group.add_argument( "--highlight-peaks", dest="highlight_peaks", action="store_true", help="Highlight annotated peak windows in the background.",)
    peak_group.add_argument( "--no-highlight-peaks", dest="highlight_peaks", action="store_false", help="Do not highlight annotated peak windows in the background.",)

    parser.add_argument(
        "--peak-signals", type=str, nargs="+", default=None, choices=list(PEAK_SIGNAL_TYPES), metavar="SIGNAL",
        help=(
            "Peak signal types to highlight when --highlight-peaks is enabled "
            "(default: all available in the input)."
        ),
    )

    diff_group = parser.add_mutually_exclusive_group()
    diff_group.add_argument(
        "--show-difference",
        dest="show_difference",
        action="store_true",
        help="Show the red difference line (default).",
    )
    diff_group.add_argument(
        "--no-show-difference",
        dest="show_difference",
        action="store_false",
        help="Hide the red difference line and keep y-axis at [0, 1].",
    )

    parser.add_argument(
        "--difference-direction",
        type=str,
        default="max_mutated_minus_reference",
        choices=list(DIFFERENCE_DIRECTIONS),
        metavar="DIRECTION",
        help=(
            "Direction used for the signed difference line. "
            "Choices: max_mutated_minus_reference (default), "
            "reference_minus_max_mutated."
        ),
    )

    parser.set_defaults(highlight_padding=True, show_tss_tts=True, highlight_peaks=True, show_difference=True)

    # Parse
    parsed_args = parser.parse_args(args)

    # Validate
    if not os.path.exists(parsed_args.input):
        raise FileNotFoundError(f"Input file does not exist: {parsed_args.input}")

    return parsed_args


def run_visualization(args):
    """Run visualization with provided arguments.

    Args:
        args: argparse.Namespace with parsed arguments from parse_arguments().

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    try:
        visualize_scan_results(
            args.input,
            peak_input=args.peaks,
            genes=args.genes,
            tfs=args.tfs,
            output_dir=args.output,
            output_format=args.format,
            include_title=not args.no_title,
            highlight_padding=args.highlight_padding,
            show_tss_tts=args.show_tss_tts,
            tss_position=args.tss_position,
            tts_position=args.tts_position,
            highlight_peaks=args.highlight_peaks,
            peak_signal_types=args.peak_signals,
            show_difference=args.show_difference,
            difference_direction=args.difference_direction,
            random_subset=args.random_subset,
        )
        return 0
    except Exception as exc:
        print_status(f"Error during visualization: {exc}", "ERROR")
        traceback.print_exc()
        return 1


def main():
    """Command-line interface for deepCIS visualization."""
    args = parse_arguments()
    return run_visualization(args)


if __name__ == "__main__":
    import sys
    sys.exit(main())
