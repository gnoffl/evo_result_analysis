"""Visualization tools for deepCIS scanning results."""
import traceback
import argparse
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from analysis.utils.io import print_status


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
    fixed_columns = {"gene", "sequence_type", "window_start", "window_end", "contains_padding"}
    tf_cols = [col for col in df.columns if col not in fixed_columns]
    return sorted(tf_cols)
    
# Calculate window center positions: (start + end) / 2
def get_centers(data: pd.DataFrame) -> np.ndarray:
    # Adding 1 to adjust for the fact that the end index is not included, as well as conerting from 0-based to 1-based coordinates
    return ((data["window_start"] + data["window_end"] + 1) / 2).values


def extract_plot_data(
    df_subset: pd.DataFrame,
    tf_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Extract and transform data from DataFrame to plottable coordinates.

    Separates reference and mutated sequences, calculates window center positions,
    and extracts the corresponding values for a given TF column.

    Args:
        df_subset: DataFrame filtered to contain only data for one gene and one TF.
        tf_col: Name of the TF column to extract values from.

    Returns:
        Tuple of (ref_x, ref_y, mut_x, mut_y, x_min, x_max) where:
        * ref_x, mut_x: Window center positions (in bp) for reference and mutated
        * ref_y, mut_y: Prediction scores for reference and mutated
        * x_min, x_max: Global min/max genomic positions (window_start/window_end)
    """
    # Separate reference and mutated sequences
    ref_data = df_subset[df_subset["sequence_type"] == "reference"].sort_values(["window_start"])       #type: ignore
    mut_data = df_subset[df_subset["sequence_type"] == "max_mutated"].sort_values(["window_start"])     #type: ignore 

    ref_x = get_centers(ref_data) if len(ref_data) > 0 else np.array([])
    ref_y = ref_data[tf_col].values if len(ref_data) > 0 else np.array([])

    mut_x = get_centers(mut_data) if len(mut_data) > 0 else np.array([])
    mut_y = mut_data[tf_col].values if len(mut_data) > 0 else np.array([])

    # Determine x-axis range (from min window_start to max window_end)
    x_min = df_subset["window_start"].min()
    x_max = df_subset["window_end"].max()

    return ref_x, ref_y, mut_x, mut_y, x_min, x_max


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
    ax.set_ylim(0, 1)

    # Set labels
    ax.set_xlabel("Genomic Position (bp)")
    ax.set_ylabel("Binding Prediction Score")

    # Set title
    if include_title:
        tf_name = tf_col.replace("tf_", "TF_")
        ax.set_title(f"{gene} - {tf_name}")

    # Additional formatting
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)


def _plot_gene_tf(
    df_subset: pd.DataFrame,
    gene: str,
    tf_col: str,
    ax: plt.Axes,
    include_title: bool = True,
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
    ref_x, ref_y, mut_x, mut_y, x_min, x_max = extract_plot_data(df_subset, tf_col)

    # Add padding regions background
    padding_regions = _get_padding_regions(df_subset)
    _add_padding_background(ax, padding_regions)

    # Plot reference and mutated lines
    _plot_line(ax, ref_x, ref_y, "Reference", "black")
    _plot_line(ax, mut_x, mut_y, "Mutated", "blue")

    # Set axis properties
    _set_axis_properties(ax, x_min, x_max, gene, tf_col, include_title)


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
            raise ValueError(f"Gene '{gene}' not found in data. Available genes: {available_genes}")

    # Validate TFs exist
    available_tfs = set(get_tf_columns(df))
    for tf in tfs:
        if tf not in available_tfs:
            raise ValueError(f"TF column '{tf}' not found in data. Available TFs: {available_tfs}")

    return genes, tfs


def _save_gene_tf_plot(
    fig: plt.Figure,                    #type: ignore
    gene: str,
    tf_col: str,
    output_dir: str,
    output_format: str,
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
    filename = f"{gene}_{tf_name}.{output_format}"
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

    # Create figure and plot
    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        _plot_gene_tf(gene_df, gene, tf_col, ax, include_title=include_title)
        _save_gene_tf_plot(fig, gene, tf_col, output_dir, output_format)
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
) -> None:
    """Main entry point for visualization.

    Can accept either a path to a CSV file or a pandas DataFrame directly.

    Args:
        input: Path to CSV file with scan results, or a DataFrame.
        genes: List of gene names to plot. If None, plots all.
        tfs: List of TF column names to plot. If None, plots all.
        output_dir: Directory to save plots. If None, defaults to "plots"
            next to the input file (for file input) or current working directory (for DataFrame input).
        output_format: File format for saving plots. Default: "png".
        include_title: Whether to include titles in plots. Default: True.

    Raises:
        ValueError: If input is invalid type or DataFrame is empty.
    """
    # Load data
    df = _load_input_data(input)

    # Resolve output directory
    output_dir = _resolve_output_directory(input, output_dir)

    # Validate inputs and set defaults
    genes, tfs = _validate_and_set_defaults(df, genes, tfs)

    # Create plots
    successful_plots = 0
    for gene in tqdm(genes, desc="Processing genes"):
        gene_df = df[df["gene"] == gene]

        for tf_col in tfs:
            if _process_gene_tf(gene_df, gene, tf_col, output_dir, output_format, include_title):       #type: ignore
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
        """,
    )

    # Add arguments
    parser.add_argument("--input", type=str, required=True, metavar="PATH", help="Path to CSV file with deepCIS scan results",)
    parser.add_argument("--output", type=str, default=None, metavar="DIR", help="Directory to save plots (default: 'plots' next to input file)",)
    parser.add_argument("--genes", type=str, nargs="+", default=None, metavar="GENE", help="Gene names to plot (default: all genes)",)
    parser.add_argument("--tfs", type=str, nargs="+", default=None, metavar="TF", help="TF columns to plot, e.g., tf_0 tf_1 (default: all TFs)",)
    parser.add_argument("--format", type=str, default="png", metavar="FORMAT", choices=["png", "pdf", "svg", "jpg", "jpeg"],
                        help="Output file format (default: png)",)
    parser.add_argument("--no-title", action="store_true", help="Do not include titles in plots",)

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
            genes=args.genes,
            tfs=args.tfs,
            output_dir=args.output,
            output_format=args.format,
            include_title=not args.no_title,
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
