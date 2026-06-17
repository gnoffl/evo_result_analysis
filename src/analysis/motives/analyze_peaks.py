"""Utilities for checking WRKY peak overlap against target mutation regions.

This workflow loads annotated peaks from `peak_scanner.py` output and a run
directory containing gene subfolders with `parameters.json`. For each gene
folder, the mutable target region (`mutation_start`, `mutation_end`) is read
and compared against WRKY peaks for that gene.
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from analysis.utils.io import print_status  # noqa: E402


PEAK_GENE_COLUMN = "gene"
PEAK_TF_COLUMN = "tf"
PEAK_START_COLUMN = "peak_start"
PEAK_END_COLUMN = "peak_end"
PEAK_SCORE_COLUMN = "peak_area"
PARAMETERS_FILE_NAME = "parameters.json"

DIFF_CALC_COLUMN = "diff_calc"
PEAK_SUMMARY_SUFFIX = "peak_summary"
DIVERGING_COLORMAP = "RdBu_r"


def _format_analysis_parameters(parameters: Dict[str, object]) -> str:
    """Format analysis parameters for concise status logging."""
    return ", ".join(f"{key}={value}" for key, value in parameters.items())


def _run_logged_analysis(
    analysis_name: str,
    parameters: Dict[str, object],
    action: Callable[[], Any],
) -> Any:
    """Run an analysis step with start, success, and failure status output."""
    print_status(f"Starting {analysis_name} with parameters: {_format_analysis_parameters(parameters)}", "INFO")
    try:
        result = action()
    except Exception as exc:
        print_status(f"{analysis_name} failed: {exc}", "ERROR")
        raise

    print_status(f"{analysis_name} succeeded", "SUCCESS")
    return result


def _load_peaks(peaks_path: str, excluded_genes_path: Optional[str] = None) -> pd.DataFrame:
    """Load a DataFrame from a path or return a copy of the provided DataFrame."""
    path = Path(peaks_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    peaks_df = pd.read_csv(path)
    if excluded_genes_path:
        with open(excluded_genes_path, "r", encoding="utf-8") as handle:
            excluded_genes = {_normalize_gene_id(gene) for gene in json.load(handle)}
        peak_gene_prefix = peaks_df[PEAK_GENE_COLUMN].astype(str).str.rsplit("_", n=3).str[0]
        peaks_df = peaks_df[~peak_gene_prefix.isin(excluded_genes)]
    return peaks_df     #type: ignore


def _normalize_gene_id(gene_id: object) -> str:
    """Normalize gene identifiers for matching between sources."""
    return str(gene_id).strip()


def _coerce_to_int(value: object, field_name: str, gene_name: str) -> int:
    """Safely coerce a JSON value to int with clear field-specific errors."""
    try:
        return int(str(value))
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name!r} value in {gene_name}/{PARAMETERS_FILE_NAME}: {value!r}") from exc


def _parse_mutation_region(parameters: Dict[str, object], gene_name: str) -> Tuple[int, int]:
    """Parse mutation_start/mutation_end from one gene's parameters dictionary.

    `mutation_end` is stored as exclusive in the run data. The returned end
    position is inclusive to match peak interval semantics.
    """
    try:
        start_raw = parameters["mutation_start"]
        end_raw = parameters["mutation_end"]
    except KeyError as exc:
        raise KeyError(f"Missing {exc.args[0]!r} in {gene_name}/{PARAMETERS_FILE_NAME}") from exc

    start = _coerce_to_int(start_raw, "mutation_start", gene_name)
    end_exclusive = _coerce_to_int(end_raw, "mutation_end", gene_name)

    if end_exclusive <= start:
        raise ValueError(
            f"Invalid mutation range for {gene_name}: mutation_end ({end_exclusive}) must be > mutation_start ({start})"
        )

    end_inclusive = end_exclusive - 1
    return start, end_inclusive

def _filter_wrky_peaks(peaks_df: pd.DataFrame, tf_substring: str = "WRKY", peak_type: str = "reference") -> pd.DataFrame:
    """Return only WRKY peaks from the peak-scanner output."""
    missing = [column for column in [PEAK_GENE_COLUMN, PEAK_TF_COLUMN, PEAK_START_COLUMN, PEAK_END_COLUMN, PEAK_SCORE_COLUMN] if column not in peaks_df.columns]
    if missing:
        raise KeyError(f"Peak-scanner output is missing required columns: {missing}")

    signal_peaks = peaks_df[peaks_df["signal_type"] == peak_type] if "signal_type" in peaks_df.columns else peaks_df
    wrky_mask = (
        signal_peaks[PEAK_TF_COLUMN]
        .astype(str)
        .str.contains(tf_substring, case=False, na=False)
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    wrky_peaks = signal_peaks.loc[wrky_mask].copy()
    return wrky_peaks


def _load_run_targets(run_directory: Union[str, Path]) -> pd.DataFrame:
    """Load mutation target regions from all gene folders in a run directory."""
    run_path = Path(run_directory)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    if not run_path.is_dir():
        raise NotADirectoryError(f"Run directory is not a directory: {run_path}")

    rows = []
    for child in sorted(run_path.iterdir()):
        if not child.is_dir():
            continue

        params_path = child / PARAMETERS_FILE_NAME
        if not params_path.exists():
            continue

        with open(params_path, "r", encoding="utf-8") as handle:
            parameters = json.load(handle)

        target_start, target_end = _parse_mutation_region(parameters, gene_name=child.name)
        rows.append(
            {
                PEAK_GENE_COLUMN: _normalize_gene_id(child.name),
                "target_start": target_start,
                "target_end": target_end,
            }
        )

    return pd.DataFrame(rows)


def _calculate_peak_selection_key(
    row: Any,
    target_start: int,
    target_end: int,
    target_center: float,
) -> Tuple[int, float, int, int]:
    """Calculate the selection criteria tuple for ranking overlapping peaks.

    Selection criteria (in order of priority):
    1. Overlap length (maximized) — peaks with largest overlap preferred
    2. Center distance (minimized) — peaks closest to target center preferred
    3. Peak start position (minimized) — earliest peak preferred for deterministic tie-break
    4. Peak end position (minimized) — determines final ordering for deterministic tie-break

    Args:
        row: A NamedTuple from DataFrame.itertuples(index=False)
        target_start: Start position of target region (inclusive)
        target_end: End position of target region (inclusive)
        target_center: Calculated center of target region

    Returns:
        Tuple of (negative_overlap_length, center_distance, peak_start, peak_end)
        Negative overlap allows min() to find maximum; others in ascending order.
    """
    peak_start = int(getattr(row, PEAK_START_COLUMN))
    peak_end = int(getattr(row, PEAK_END_COLUMN)) + 250
    peak_center = (peak_start + peak_end) / 2.0

    # Calculate overlap length (inclusive intersection)
    overlap_length = min(peak_end, target_end) - max(peak_start, target_start) + 1

    # Calculate center distance (absolute difference between centers)
    center_distance = abs(peak_center - target_center)

    # Return tuple: negative overlap (to maximize via min()), then center distance, then positions
    return (
        -overlap_length,  # Negative so min() finds maximum overlap
        center_distance,  # Minimize distance between centers
        peak_start,       # Earliest peak start (deterministic)
        peak_end,         # Earliest peak end (deterministic)
    )


def _select_best_overlap(
    target_start: int,
    target_end: int,
    peaks_df: pd.DataFrame,
    verbose: bool = False
) -> Tuple[bool, Optional[pd.Series], int]:
    """Find the best overlapping peak for one target interval.

    Selection is based on genomic location only, with priority:
    1) largest overlap length with target interval
    2) smallest absolute distance between peak center and target center
    3) earliest peak start (deterministic tie-break)
    """
    overlap_mask = (
        (peaks_df[PEAK_START_COLUMN] <= target_end)
        & (peaks_df[PEAK_END_COLUMN] + 250 >= target_start)
    )
    overlapping = peaks_df.loc[overlap_mask].copy()
    if verbose:
        print(peaks_df[["gene", PEAK_START_COLUMN, PEAK_END_COLUMN]].head())
        print(target_start, target_end)
        print(overlapping[["gene", PEAK_START_COLUMN, PEAK_END_COLUMN]].head())
        print("----------------")

    if overlapping.empty:
        return False, None, 0

    target_center = (target_start + target_end) / 2.0

    best_overlap_row = min(
        overlapping.itertuples(index=False),
        key=lambda row: _calculate_peak_selection_key(row, target_start, target_end, target_center),
    )
    return True, pd.Series(best_overlap_row._asdict()), len(overlapping)


def derive_peak_locations(full_results_df: pd.DataFrame) -> pd.DataFrame:
    """Derive a mapping table from the full overlap results."""
    peak_locations = full_results_df.copy()
    peak_locations["gene_name"] = peak_locations[PEAK_GENE_COLUMN].str.split("_").str[1]
    peak_locations = peak_locations[["gene_name", "target_start", "target_end"]]
    return peak_locations


def analyze_wrky_peak_overlaps(
    peaks: pd.DataFrame,
    run_directory: Union[str, Path],
    output_folder: Path,
    tf_substring: str = "WRKY",
    peak_type: str = "reference",
    base_name: str = ""
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate run-directory target regions with overlapping WRKY peaks.

    Args:
        peaks: Annotated peak output as a DataFrame or CSV path.
        run_directory: Path containing gene folders with `parameters.json`.
        tf_substring: Substring used to keep WRKY-only peaks.
        output_folder: Path to the folder where output files will be saved.

    Returns:
        DataFrame with one row per gene folder target region plus overlap
        details from matching WRKY peaks.
    """
    peaks_df = _filter_wrky_peaks(peaks, tf_substring=tf_substring, peak_type=peak_type)
    targets_df = _load_run_targets(run_directory)

    if targets_df.empty:
        return pd.DataFrame(
            columns=[
                PEAK_GENE_COLUMN,
                "target_start",
                "target_end",
                "overlaps_peak",
                "overlap_count",
                "matched_peak_start",
                "matched_peak_end",
                "matched_peak_score",
                "matched_peak_tf",
            ]
        ), pd.DataFrame(columns=["gene_name", "target_start", "target_end"])

    peak_gene_norm = peaks_df[PEAK_GENE_COLUMN].astype(str).map(_normalize_gene_id)

    results = []
    for _, target_row in targets_df.iterrows():
        gene_id = target_row[PEAK_GENE_COLUMN]
        target_start = int(target_row["target_start"])
        target_end = int(target_row["target_end"])

        gene_peaks = peaks_df.loc[peak_gene_norm == gene_id].copy()

        overlaps_peak, best_overlap, overlap_count = _select_best_overlap(target_start, target_end, gene_peaks, verbose=False)

        result_row: Dict[str, object] = {
            PEAK_GENE_COLUMN: gene_id,
            "target_start": target_start,
            "target_end": target_end,
            "overlaps_peak": overlaps_peak,
            "overlap_count": overlap_count,
            "matched_peak_start": pd.NA,
            "matched_peak_end": pd.NA,
            "matched_peak_score": pd.NA,
            "matched_peak_tf": pd.NA,
        }

        if best_overlap is not None:
            result_row["matched_peak_start"] = int(best_overlap[PEAK_START_COLUMN])
            result_row["matched_peak_end"] = int(best_overlap[PEAK_END_COLUMN])
            result_row["matched_peak_score"] = float(best_overlap[PEAK_SCORE_COLUMN])
            result_row["matched_peak_tf"] = str(best_overlap[PEAK_TF_COLUMN])
        
        # else:
        #     print(gene_id, end=" ")

        results.append(result_row)

    result = pd.DataFrame(results)
    expected_peak_locations = derive_peak_locations(result)
    name = f"{base_name}_{peak_type}" if base_name else f"{peak_type}"
    output_path = output_folder / f"{name}.csv"
    mapping_path = output_folder / f"{name}_expected_peak_locations.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    expected_peak_locations.to_csv(mapping_path, index=False)
    return result, expected_peak_locations


def add_zeros(summary: pd.DataFrame) -> pd.DataFrame:
    # Enforce all combinations with 0 counts
    all_tfs = summary[PEAK_TF_COLUMN].unique()
    all_signal_types = summary["signal_type"].unique()
    
    full_index = pd.MultiIndex.from_product(
        [all_tfs, all_signal_types],
        names=[PEAK_TF_COLUMN, "signal_type"]
    )
    summary = (
        summary.set_index([PEAK_TF_COLUMN, "signal_type"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    return summary

def filter_overlapping_peaks(peaks_df: pd.DataFrame, expected_peak_locations: pd.DataFrame) -> pd.DataFrame:
    """Filter peaks to only those that overlap target regions in the mapping."""
    filtered_peaks = []
    for _, row in expected_peak_locations.iterrows():
        gene_name = row["gene_name"]
        target_start = row["target_start"]
        target_end = row["target_end"]

        gene_peaks = peaks_df[peaks_df[PEAK_GENE_COLUMN].astype(str).str.contains(gene_name, case=False, na=False)].copy()
        # only retain peaks whose middle point lies in the expected range
        peak_middle = (gene_peaks[PEAK_START_COLUMN] + gene_peaks[PEAK_END_COLUMN] + 250) / 2
        middle_mask = (peak_middle >= target_start) & (peak_middle <= target_end)
        filtered_peaks.append(gene_peaks.loc[middle_mask])

    if filtered_peaks:
        return pd.concat(filtered_peaks, ignore_index=True)
    else:
        return pd.DataFrame(columns=peaks_df.columns)

def summarize_peaks(
    peaks_df: pd.DataFrame,
    output_folder: Path,
    expected_peak_locations: pd.DataFrame,
    base_name: str = "",
) -> pd.DataFrame:
    """Summarize peak counts by TF in wide format, one column per signal type.

    Args:
        peaks_df: DataFrame of annotated peaks with signal_type and peak_area columns.
        output_folder: Directory where the summary CSV is written.
        expected_peak_locations: If non-empty, restrict peaks to those overlapping
            target regions.
        base_name: Prefix for the output filename.

    Returns:
        Wide-format DataFrame with one column per signal type, plus ``diff_calc``
        (diff_added - diff_removed), sorted by ``diff_calc`` descending.
    """
    if not expected_peak_locations.empty:
        peaks_df = filter_overlapping_peaks(peaks_df, expected_peak_locations)
    peaks_df = peaks_df.copy()
    peaks_df["added"] = np.where(peaks_df[PEAK_SCORE_COLUMN] > 0, "diff_added", "diff_removed")

    non_diff = peaks_df[~peaks_df["signal_type"].str.contains("difference", case=False, na=False)]
    diff = peaks_df[peaks_df["signal_type"].str.contains("difference", case=False, na=False)]

    summary_non_diff = (
        non_diff.groupby([PEAK_TF_COLUMN, "signal_type"])
        .agg(peak_count=(PEAK_SCORE_COLUMN, "count"))
        .reset_index()
    )
    diff_summary = (
        diff.groupby([PEAK_TF_COLUMN, "added"])
        .agg(peak_count=(PEAK_SCORE_COLUMN, "count"))
        .reset_index()
        .rename(columns={"added": "signal_type"})
    )

    long_summary = pd.concat([summary_non_diff, diff_summary], ignore_index=True)

    wide_summary = long_summary.pivot_table(
        index=PEAK_TF_COLUMN,
        columns="signal_type",
        values="peak_count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    wide_summary.columns.name = None

    max_mutated = wide_summary.get("max_mutated", pd.Series(0, index=wide_summary.index))
    reference = wide_summary.get("reference", pd.Series(0, index=wide_summary.index))
    wide_summary["diff_calc"] = max_mutated - reference

    wide_summary = wide_summary.sort_values(by="diff_calc", ascending=False).reset_index(drop=True)

    output_path = (
        output_folder / f"{base_name}_peak_summary.csv"
        if base_name
        else output_folder / "peak_summary.csv"
    )
    wide_summary.to_csv(output_path, index=False)
    return wide_summary


def save_figure(fig: Figure, filename: str, output_dir: Union[str, Path], fmt: str = "png") -> Path:
    """Save a figure to ``output_dir/filename.fmt``.

    Args:
        fig: Matplotlib figure to save.
        filename: File name without extension.
        output_dir: Directory to write into.
        fmt: Image format/extension (e.g. ``"png"``, ``"svg"``, ``"pdf"``).

    Returns:
        Path to the written figure.
    """
    path = Path(output_dir) / f"{filename}.{fmt}"
    fig.savefig(str(path), bbox_inches="tight", dpi=150)
    return path


def prepare_diff_calc_data(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Extract the per-TF net binding change and sort it for plotting.

    Args:
        summary_df: Wide-format peak summary containing ``tf`` and ``diff_calc``.

    Returns:
        A two-column DataFrame (``tf``, ``diff_calc``) sorted by ``diff_calc``
        descending, with a clean integer index.

    Raises:
        KeyError: If the required columns are missing.
    """
    missing = [
        column
        for column in (PEAK_TF_COLUMN, DIFF_CALC_COLUMN)
        if column not in summary_df.columns
    ]
    if missing:
        raise KeyError(f"Peak summary is missing required columns: {missing}")

    plot_df = summary_df[[PEAK_TF_COLUMN, DIFF_CALC_COLUMN]].copy()
    plot_df[DIFF_CALC_COLUMN] = plot_df[DIFF_CALC_COLUMN].astype(float)
    descending_order = np.argsort(plot_df[DIFF_CALC_COLUMN].to_numpy())[::-1]
    plot_df = plot_df.iloc[descending_order].reset_index(drop=True)
    return plot_df


def compute_symmetric_limit(diff_calc_values: "pd.Series | np.ndarray") -> float:
    """Compute a symmetric axis/color limit from net binding-change values.

    A symmetric limit (``±max(|diff_calc|)``) keeps the diverging colormap
    centered at zero. Passing a shared limit across runs makes per-run figures
    directly comparable.

    Args:
        diff_calc_values: Net binding-change values.

    Returns:
        A positive limit; ``1.0`` when no positive magnitude is present.
    """
    values = np.asarray(diff_calc_values, dtype=float)
    if values.size == 0:
        return 1.0
    limit = float(np.nanmax(np.abs(values)))
    return limit if limit > 0 else 1.0


def plot_diff_calc_barplot(plot_df: pd.DataFrame, value_limit: float, title: str = "") -> Figure:
    """Plot net binding change per TF family as a diverging horizontal bar chart.

    Bars are colored by signed magnitude on a symmetric diverging scale, so on a
    shared ``value_limit`` smaller changes appear paler across runs.

    Args:
        plot_df: Output of :func:`prepare_diff_calc_data` (already sorted).
        value_limit: Symmetric limit for the x-axis and color normalization.
        title: Optional plot title.

    Returns:
        The created figure.
    """
    norm = Normalize(vmin=-value_limit, vmax=value_limit)
    scalar_mappable = ScalarMappable(norm=norm, cmap=DIVERGING_COLORMAP)
    colors = scalar_mappable.to_rgba(plot_df[DIFF_CALC_COLUMN].to_numpy())

    height = max(4.0, 0.32 * len(plot_df))
    fig, ax = plt.subplots(figsize=(7.0, height))
    ax.barh(
        plot_df[PEAK_TF_COLUMN],
        plot_df[DIFF_CALC_COLUMN],
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.invert_yaxis()  # largest diff_calc on top
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-value_limit * 1.05, value_limit * 1.05)
    ax.set_xlabel("Net binding-site change (optimized − reference)")
    ax.set_ylabel("TF family")
    if title:
        ax.set_title(title)

    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02)
    colorbar.set_label(DIFF_CALC_COLUMN)

    fig.tight_layout()
    return fig


def plot_diff_calc_heatmap(plot_df: pd.DataFrame, value_limit: float, title: str = "") -> Figure:
    """Plot net binding change per TF family as a single-column heatmap.

    A one-column heatmap on a fixed, symmetric diverging scale stacks naturally
    into a multi-run grid for the later cross-run comparison.

    Args:
        plot_df: Output of :func:`prepare_diff_calc_data` (already sorted).
        value_limit: Symmetric limit for the color normalization.
        title: Optional plot title.

    Returns:
        The created figure.
    """
    heat_data = plot_df.set_index(PEAK_TF_COLUMN)[[DIFF_CALC_COLUMN]]

    height = max(4.0, 0.32 * len(plot_df))
    fig, ax = plt.subplots(figsize=(3.4, height))
    sns.heatmap(
        heat_data,
        cmap=DIVERGING_COLORMAP,
        center=0,
        vmin=-value_limit,
        vmax=value_limit,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": DIFF_CALC_COLUMN},
        ax=ax,
    )
    ax.set_xticklabels(["net change"])
    ax.set_xlabel("")
    ax.set_ylabel("TF family")
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def visualize_peak_summary(
    output_folder: Union[str, Path],
    summary_df: pd.DataFrame,
    base_name: str = "",
    fmt: str = "png",
    value_limit: Optional[float] = None,
) -> List[Path]:
    """Generate the diff_calc figures for a single run's peak summary.

    Auto-locates the summary CSV under ``output_folder`` and writes a diverging
    bar chart and a single-column heatmap of ``diff_calc`` next to it.

    Args:
        output_folder: Folder containing the summary CSV; figures are written here.
        summary_df: The peak-summary DataFrame.
        base_name: Prefix used for the summary CSV and figure file names.
        fmt: Output image format (e.g. ``"png"``, ``"svg"``, ``"pdf"``).
        value_limit: Optional shared symmetric color/axis limit for cross-run
            comparability; defaults to this run's ``max(|diff_calc|)``.

    Returns:
        Paths of the written figures.
    """
    sns.set_theme(style="whitegrid")
    output_folder = Path(output_folder)

    plot_df = prepare_diff_calc_data(summary_df)
    limit = (
        value_limit
        if value_limit is not None
        else compute_symmetric_limit(plot_df[DIFF_CALC_COLUMN])
    )

    title = base_name or "peak summary"
    bar_name = f"{base_name}_diff_calc_barplot" if base_name else "diff_calc_barplot"
    heat_name = f"{base_name}_diff_calc_heatmap" if base_name else "diff_calc_heatmap"

    bar_fig = plot_diff_calc_barplot(plot_df, limit, title=title)
    bar_path = save_figure(bar_fig, bar_name, output_folder, fmt)
    plt.close(bar_fig)

    heat_fig = plot_diff_calc_heatmap(plot_df, limit, title=title)
    heat_path = save_figure(heat_fig, heat_name, output_folder, fmt)
    plt.close(heat_fig)

    return [bar_path, heat_path]


def build_parser() -> ArgumentParser:
    """Build the command-line parser for manual overlap checks."""
    parser = ArgumentParser(description="Check whether WRKY peaks overlap mutation target regions.")
    parser.add_argument("annotated_peak_file", help="CSV file with annotated peak-scanner output")
    parser.add_argument(
        "--run-directory",
        default="",
        required=False,
        help="Run directory containing gene folders with parameters.json",
    )
    parser.add_argument("--excluded-genes-file", default=None, help="Optional json list of genes to exclude from analysis.")
    parser.add_argument("--peak_type", default="reference", help="Substring to filter signal type by. Default is 'reference' to match WRKY_tnt, but can be set to empty string to disable filtering.")
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Optional path to the folder to write the annotated overlap table. If not provided, saves as '<input_stem>_wrky_overlaps.csv' next to the input file.",
    )
    parser.add_argument(
        "--base_name",
        "-n",
        default="",
        help="Base name to use for output files. If not provided, uses the stem of the annotated_peak_file.",
    )
    parser.add_argument(
        "--summarize-peaks",
        action="store_true",
        help="If set, outputs a summary table of peak counts by gene and signal type.",
    )
    parser.add_argument(
        "--only_overlapping",
        action="store_true",
        help="If set, only outputs the subset of target regions that overlap WRKY peaks, instead of the full table with all target regions.",
    )
    parser.add_argument(
        "--expected-peak-locations",
        default=None,
        help="Optional path to load a simplified mapping of gene names to target regions for only the overlapping cases."
    )
    parser.add_argument(
        "--expected_overlap",
        action="store_true",
        help="If set, analyzes the expected overlap based on the original target regions and outputs a summary table of expected vs. observed overlaps.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, generates diff_calc figures (diverging bar chart and heatmap) for the run's peak summary.",
    )
    parser.add_argument(
        "--figure-format",
        default="png",
        help="Output format for figures generated by --visualize (e.g. png, svg, pdf). Default: png.",
    )
    parser.add_argument(
        "--value-limit",
        type=float,
        default=None,
        help="Optional shared symmetric color/axis limit for diff_calc figures, to make runs comparable. Default: this run's max(|diff_calc|).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="If set, runs all analyses (overlap check, peak summary, expected overlap analysis). Overrides individual flags if set."
    )
    return parser


def main(argv: Optional[list] = None) -> None:
    """Run the WRKY overlap check from the command line."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Determine output path: use --output if provided, otherwise save next to input file
    output_folder = Path(args.output_folder) if args.output_folder else Path(args.annotated_peak_file).parent

    run_overlap = args.all or args.expected_overlap
    run_summary = args.all or args.summarize_peaks or args.visualize  # summary is needed for visualization
    run_visualize = args.all or args.visualize

    print_status(
        "Selected analyses: "
        f"overlap={run_overlap}, "
        f"summary={run_summary}, "
        f"visualize={run_visualize}",
        "INFO",
    )

    if not (run_overlap or run_summary or run_visualize):
        print_status(
            "No analyses selected. Use --expected_overlap, --summarize-peaks and/or --visualize to run analyses, or --all to run all analyses.",
            "WARNING",
        )
        return
    
    # Only the overlap and summary analyses need the annotated peak table;
    # visualization reads the previously written summary CSV instead.
    peaks = (
        _load_peaks(args.annotated_peak_file, args.excluded_genes_file)
        if (run_overlap or run_summary)
        else pd.DataFrame()
    )
    expected_peak_locations = pd.DataFrame()

    if run_overlap:
        if not args.run_directory:
            raise ValueError("Run directory must be provided with --run-directory to perform overlap analysis.")
        overlap_parameters = {
            "annotated_peak_file": args.annotated_peak_file,
            "run_directory": args.run_directory,
            "excluded_genes_file": args.excluded_genes_file,
            "peak_type": args.peak_type,
            "output_folder": output_folder,
            "base_name": args.base_name,
        }
        _, expected_peak_locations = _run_logged_analysis(
            "WRKY overlap analysis",
            overlap_parameters,
            lambda: analyze_wrky_peak_overlaps(
                peaks=peaks,
                run_directory=args.run_directory,
                peak_type=args.peak_type,
                output_folder=output_folder,
                base_name=args.base_name,
            ),
        )

    if run_summary:
        if args.expected_peak_locations is not None and args.expected_peak_locations:
            expected_peak_locations = pd.read_csv(args.expected_peak_locations)
        elif args.only_overlapping and not run_overlap:
            raise ValueError(
                "Expected peak locations file must be provided with --expected-peak-locations when using --only_overlapping without running the overlap analysis."
            )

        summary_parameters = {
            "annotated_peak_file": args.annotated_peak_file,
            "excluded_genes_file": args.excluded_genes_file,
            "output_folder": output_folder,
            "expected_peak_locations": args.expected_peak_locations if args.expected_peak_locations is not None else ("<from overlap analysis>" if run_overlap else None),
            "only_overlapping": args.only_overlapping,
            "base_name": args.base_name,
        }
        peak_summary = _run_logged_analysis(
            "peak summary analysis",
            summary_parameters,
            lambda: summarize_peaks(peaks, output_folder, expected_peak_locations, base_name=args.base_name),
        )

    if run_visualize:
        visualize_parameters = {
            "output_folder": output_folder,
            "base_name": args.base_name,
            "figure_format": args.figure_format,
            "value_limit": args.value_limit,
        }
        _run_logged_analysis(
            "peak summary visualization",
            visualize_parameters,
            lambda: visualize_peak_summary(
                output_folder,
                summary_df=peak_summary,
                base_name=args.base_name,
                fmt=args.figure_format,
                value_limit=args.value_limit,
            ),
        )


if __name__ == "__main__":
    main()