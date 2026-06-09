"""Shared logic for the STARR-seq x deepCRE correlation workflows.

This module holds the transcription-factor-agnostic pieces of the correlation
pipeline so that the per-TF entry-point scripts
(``starrseq_deepcre_correlation_WRKY.py`` and
``starrseq_deepcre_correlation_bHLH.py``) only carry their TF-specific
orchestration. It covers:

- shared constants describing the 3020 bp extracted-sequence frame,
- linear-fit / bucketing helpers,
- all matplotlib plotting routines,
- the STARR-seq <-> deepCRE mapping and prediction pipeline.

The output root for plots is module state (:data:`CORRELATION_OUTPUT_ROOT`),
set once per run via :func:`set_output_root`. Importing this module has no
global matplotlib side effect; each entry-point script calls
:func:`configure_matplotlib` once at startup.
"""
import os
import json
from typing import Dict, List, Tuple, Union, cast
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tensorflow.keras.models import load_model  # type: ignore
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import matplotlib as mpl

from evolution.sequences import one_hot_encode

REF_SEQ_FILE = "reference_sequence.fa"
DEEPCRE_PATH = "/home/gernot/Code/PhD_Code/Evolution/models/Atha_S0X0.75dP7K25g_NC_003075.7_ssr_train_models_250705_211854.h5"
PARETO_PATH = os.path.join("saved_populations", "pareto_front.json")
INTRAGENIC = 500
EXTRAGENIC = 1000
MIN_PADDING = 20
IDEAL_PADDING_START = INTRAGENIC + EXTRAGENIC
IDEAL_PADDING_END = IDEAL_PADDING_START + MIN_PADDING
FULL_SEQUENCE_LENGTH = 2 * IDEAL_PADDING_START + MIN_PADDING
FULL_OVERLAP_LENGTH = 170
BUCKET_SIZE = 200
EDGE_POSITIONS = {0, IDEAL_PADDING_START, IDEAL_PADDING_END, FULL_SEQUENCE_LENGTH}
BUCKET_SUMMARY_FILE_NAME = "overlap_bucket_fit_parameters.csv"
POSITION_SERIES_COLORS = {
    "all": "#555555",
    "light": "#e6a817",
    "dark": "#2166ac",
}
OUTPUT_DPI = 600
ENABLE_BUCKETED_ANALYSIS = True
ADD_OVERALL_FIT_TO_BUCKETED_PLOTS = True
INDIVIDUAL_BUCKET_LABELS_TO_PLOT = ["800-999"]
BUCKET_COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
]

# Output root for all plots/CSVs. Set per-run via set_output_root() so importing
# this module never assumes a working directory.
CORRELATION_OUTPUT_ROOT: str = ""


def set_output_root(root: str) -> None:
    """Set the directory under which all correlation outputs are written.

    Args:
        root: Absolute path to the per-run output root (e.g.
            ``.../correlation/new``). Created lazily per analysis subfolder.
    """
    global CORRELATION_OUTPUT_ROOT
    CORRELATION_OUTPUT_ROOT = root


def configure_matplotlib() -> None:
    """Apply poster-friendly global matplotlib settings.

    Called once at the start of each entry-point script's ``main()`` so that
    importing this module has no global matplotlib side effect.
    """
    # Improve readability for poster graphics: increase font sizes globally
    mpl.rcParams.update(
        {
            "figure.titlesize": 22,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "legend.title_fontsize": 20,
            "savefig.dpi": OUTPUT_DPI,
        }
    )


def _fit_linear_model(x_values: pd.Series, y_values: pd.Series) -> Tuple[float, float, float, float]:
    if len(x_values) < 2 or x_values.nunique() < 2 or y_values.nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x_values, y_values, 1)
    correlation, p_value = spearmanr(x_values, y_values)
    return float(slope), float(intercept), cast(float, correlation), cast(float, p_value)


def _format_bucket_label(bucket_start: int) -> str:
    bucket_end = bucket_start + BUCKET_SIZE - 1
    return f"{bucket_start}-{bucket_end}"


def _darken_hex_color(hex_color: str, factor: float = 0.65) -> str:
    color = hex_color.lstrip("#")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    red = int(red * factor)
    green = int(green * factor)
    blue = int(blue * factor)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _sanitize_bucket_label(bucket_label: str) -> str:
    return bucket_label.replace("-", "_")


def _get_analysis_output_dir(analysis_name: str) -> str:
    if not CORRELATION_OUTPUT_ROOT:
        raise RuntimeError("call set_output_root() before plotting")
    output_dir = os.path.join(CORRELATION_OUTPUT_ROOT, analysis_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def add_length_corrected_overlap_buckets(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Correct overlap spans to meet FULL_OVERLAP_LENGTH and assign to buckets.

    For overlaps at edge positions that are shorter than FULL_OVERLAP_LENGTH (170 bp),
    adjusts the span to exactly 170 bp. Then assigns each overlap to a bucket based on
    its corrected start position.

    Args:
        prediction_df: DataFrame with columns 'overlap_start' and 'overlap_end'

    Returns:
        DataFrame with added columns:
        - group_overlap_start, group_overlap_end: length-corrected positions
        - group_overlap_length: corrected span length
        - overlap_bucket_start, overlap_bucket_end: bucket boundaries
        - overlap_bucket_label: formatted bucket label (e.g., "0-199")
        - overlap_bucket_order: numeric bucket start position for sorting
    """
    corrected_df = prediction_df.copy()
    corrected_df["group_overlap_start"] = corrected_df["overlap_start"]
    corrected_df["group_overlap_end"] = corrected_df["overlap_end"]

    overlap_span = corrected_df["overlap_end"] - corrected_df["overlap_start"]
    start_on_edge = corrected_df["overlap_start"].isin(EDGE_POSITIONS) & (overlap_span < FULL_OVERLAP_LENGTH)
    end_on_edge = corrected_df["overlap_end"].isin(EDGE_POSITIONS) & (overlap_span < FULL_OVERLAP_LENGTH)

    corrected_df.loc[start_on_edge, "group_overlap_start"] = corrected_df.loc[start_on_edge, "overlap_end"] - FULL_OVERLAP_LENGTH
    corrected_df.loc[end_on_edge, "group_overlap_end"] = corrected_df.loc[end_on_edge, "overlap_start"] + FULL_OVERLAP_LENGTH
    corrected_df["group_overlap_length"] = corrected_df["group_overlap_end"] - corrected_df["group_overlap_start"]
    corrected_df["overlap_bucket_start"] = (corrected_df["group_overlap_start"] // BUCKET_SIZE) * BUCKET_SIZE
    corrected_df["overlap_bucket_end"] = corrected_df["overlap_bucket_start"] + BUCKET_SIZE
    corrected_df["overlap_bucket_label"] = corrected_df["overlap_bucket_start"].astype(int).map(_format_bucket_label)
    corrected_df["overlap_bucket_order"] = corrected_df["overlap_bucket_start"].astype(int)
    return corrected_df


def _deduplicate_bucket_rows(
    bucket_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    include_differences: bool,
    delta: bool,
) -> pd.DataFrame:
    dedup_columns = [x_col, y_col]
    if include_differences and "differences" in bucket_df.columns:
        dedup_columns.append("differences")
    if "starr_reference" in bucket_df.columns:
        dedup_columns.append("starr_reference")
    if "gene" in bucket_df.columns:
        dedup_columns.append("gene")
    dedup = bucket_df[dedup_columns].drop_duplicates().dropna(subset=[x_col, y_col])    #type:ignore
    if delta:
        dedup = dedup[dedup["starr_reference"] == False]
    return dedup.dropna(subset=[x_col, y_col])


def _make_stats_row(
    analysis_name: str,
    bucket_label: str,
    bucket_start: int,
    bucket_end: int,
    bucket_size: int,
    count_points: int,
    count_unique_points: int,
    slope: float,
    intercept: float,
    correlation: float,
    p_value: float,
) -> Dict[str, Union[str, int, float]]:
    return {
        "analysis": analysis_name,
        "bucket_label": bucket_label,
        "bucket_start": bucket_start,
        "bucket_end": bucket_end,
        "bucket_size": bucket_size,
        "count_points": count_points,
        "count_unique_points": count_unique_points,
        "slope": slope,
        "intercept": intercept,
        "spearman_r": correlation,
        "spearman_p": p_value,
    }


def _plot_bucketed_correlation(
    prediction_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    save_name: str,
    title: str,
    x_label: str,
    y_label: str,
    analysis_name: str,
    include_differences: bool,
    use_buckets: bool = True,
    add_overall_fit_line: bool = True,
    delta: bool = False,
    subset_label: str = "",
) -> List[Dict[str, Union[str, int, float]]]:
    effective_analysis = f"{analysis_name}_{subset_label}" if subset_label else analysis_name
    effective_title = f"{title} ({subset_label})" if subset_label else title
    effective_save = save_name.replace(".png", f"_{subset_label}.png") if subset_label else save_name

    plt.clf()
    working_df = prediction_df.copy()
    if delta:
        working_df = working_df[working_df["starr_reference"] == False].copy()
    working_df = working_df.loc[working_df[x_col].notna() & working_df[y_col].notna(), :]

    summary_rows: List[Dict[str, Union[str, int, float]]] = []
    if working_df.empty:
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(effective_title)
        plt.savefig(os.path.join(_get_analysis_output_dir(effective_analysis), effective_save), bbox_inches="tight", dpi=OUTPUT_DPI)
        return summary_rows

    overall_dedup = _deduplicate_bucket_rows(working_df, x_col, y_col, include_differences, delta)
    overall_slope, overall_intercept, overall_corr, overall_p = _fit_linear_model(overall_dedup[x_col], overall_dedup[y_col])
    summary_rows.append(
        _make_stats_row(
            effective_analysis,
            "all",
            -1,
            -1,
            0,
            int(len(working_df)),
            int(len(overall_dedup)),
            overall_slope,
            overall_intercept,
            overall_corr,
            overall_p,
        )
    )

    if not use_buckets:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(overall_dedup[x_col], overall_dedup[y_col], color="tab:blue", alpha=0.85, label=f"all (n={len(working_df)})")
        if np.isfinite(overall_slope) and np.isfinite(overall_intercept):
            line_x = np.linspace(overall_dedup[x_col].min(), overall_dedup[x_col].max(), 100)
            ax.plot(line_x, overall_slope * line_x + overall_intercept, color="black", linewidth=2, label="overall fit")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(effective_title)
        ax.legend(title="Dataset", fontsize=14)
        fig.savefig(os.path.join(_get_analysis_output_dir(effective_analysis), effective_save), bbox_inches="tight", dpi=OUTPUT_DPI)
        plt.close(fig)
        return summary_rows

    working_df = working_df.loc[
        working_df["overlap_bucket_start"].notna() & working_df["overlap_bucket_label"].notna(), :
    ]

    bucket_values = sorted(working_df["overlap_bucket_start"].astype(int).unique().tolist())
    bucket_colors = BUCKET_COLOR_PALETTE

    fig, ax = plt.subplots(figsize=(8, 6))
    for bucket_index, bucket_start in enumerate(bucket_values):
        bucket_df = working_df[working_df["overlap_bucket_start"].astype(int) == bucket_start]
        dedup = _deduplicate_bucket_rows(bucket_df, x_col, y_col, include_differences, delta)
        bucket_label = _format_bucket_label(bucket_start)
        color = bucket_colors[bucket_index % len(bucket_colors)]

        ax.scatter(
            dedup[x_col],
            dedup[y_col],
            color=color,
            alpha=0.6,
            edgecolors="white",
            linewidths=0.3,
            label=f"{bucket_label} (n={len(bucket_df)})",
        )

        slope, intercept, correlation, p_value = _fit_linear_model(dedup[x_col], dedup[y_col])
        if np.isfinite(slope) and np.isfinite(intercept):
            line_x = np.linspace(dedup[x_col].min(), dedup[x_col].max(), 100)
            line_color = _darken_hex_color(color)
            ax.plot(line_x, slope * line_x + intercept, color=line_color, linewidth=2.8, zorder=4)

        summary_rows.append(
            _make_stats_row(
                effective_analysis,
                bucket_label,
                int(bucket_start),
                int(bucket_start + BUCKET_SIZE - 1),
                BUCKET_SIZE,
                int(len(bucket_df)),
                int(len(dedup)),
                slope,
                intercept,
                correlation,
                p_value,
            )
        )

    if add_overall_fit_line and np.isfinite(overall_slope) and np.isfinite(overall_intercept):
        line_x = np.linspace(overall_dedup[x_col].min(), overall_dedup[x_col].max(), 100)
        ax.plot(line_x, overall_slope * line_x + overall_intercept, color="black", linestyle="--", linewidth=2, label="overall fit")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(effective_title)
    ax.legend(title="Overlap bucket", fontsize=14, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    fig.savefig(os.path.join(_get_analysis_output_dir(effective_analysis), effective_save), bbox_inches="tight", dpi=OUTPUT_DPI)
    plt.close(fig)
    return summary_rows


def _plot_individual_bucket_correlation(
    prediction_df: pd.DataFrame,
    analysis_name: str,
    x_col: str,
    y_col: str,
    save_name: str,
    title: str,
    x_label: str,
    y_label: str,
    bucket_label: str,
    include_differences: bool,
    delta: bool = False,
    subset_label: str = "",
) -> None:
    effective_analysis = f"{analysis_name}_{subset_label}" if subset_label else analysis_name
    effective_title = f"{title} ({subset_label})" if subset_label else title
    effective_save = save_name.replace(".png", f"_{subset_label}.png") if subset_label else save_name

    working_df = prediction_df.copy()
    if delta:
        working_df = working_df[working_df["starr_reference"] == False].copy()
    working_df = working_df.loc[working_df["overlap_bucket_label"] == bucket_label, :]
    working_df = working_df.loc[working_df[x_col].notna() & working_df[y_col].notna(), :]

    if working_df.empty:
        print(f"No entries found for bucket {bucket_label} in {effective_save}; skipping plot.")
        return

    dedup = _deduplicate_bucket_rows(working_df, x_col, y_col, include_differences, delta)
    if dedup.empty:
        print(f"No plottable points found for bucket {bucket_label} in {effective_save}; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    genes = sorted(dedup["gene"].unique()) if "gene" in dedup.columns else [None]
    for i, gene in enumerate(genes):
        gene_df = dedup[dedup["gene"] == gene] if gene is not None else dedup
        color = BUCKET_COLOR_PALETTE[i % len(BUCKET_COLOR_PALETTE)]
        ax.scatter(
            gene_df[x_col],
            gene_df[y_col],
            s=70,
            color=color,
            alpha=0.65,
            edgecolors="white",
            linewidths=0.3,
            label=gene if gene is not None else f"{bucket_label} (n={len(working_df)})",
        )
    slope, intercept, _, _ = _fit_linear_model(dedup[x_col], dedup[y_col])
    if np.isfinite(slope) and np.isfinite(intercept):
        line_x = np.linspace(dedup[x_col].min(), dedup[x_col].max(), 100)
        ax.plot(line_x, slope * line_x + intercept, color=_darken_hex_color(BUCKET_COLOR_PALETTE[0]), linewidth=2.8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(effective_title)
    fig.savefig(os.path.join(_get_analysis_output_dir(effective_analysis), effective_save), bbox_inches="tight", dpi=OUTPUT_DPI)
    plt.close(fig)


def plot_individual_bucket_views(prediction_df: pd.DataFrame, bucket_label: str, subset_label: str = "") -> None:
    suffix = _sanitize_bucket_label(bucket_label)
    _plot_individual_bucket_correlation(
        prediction_df,
        "deepcre_starrseq",
        "prediction_mutated",
        "enrichment",
        f"deepcre_starrseq_correlation_bucket_{suffix}.png",
        "Correlation between deepCRE and STARR-seq",
        "deepCRE Prediction",
        "STARR-seq Enrichment",
        bucket_label,
        include_differences=True,
        delta=False,
        subset_label=subset_label,
    )
    _plot_individual_bucket_correlation(
        prediction_df,
        "mutation_starrseq",
        "differences",
        "enrichment",
        f"mutation_starrseq_correlation_bucket_{suffix}.png",
        "Correlation between number of mutations and STARR-seq enrichment",
        "Number of differences between reference and mutated sequence",
        "STARR-seq Enrichment",
        bucket_label,
        include_differences=False,
        delta=False,
        subset_label=subset_label,
    )
    _plot_individual_bucket_correlation(
        prediction_df,
        "mutation_deepcre",
        "differences",
        "prediction_mutated",
        f"mutation_deepcre_correlation_bucket_{suffix}.png",
        "Correlation between number of mutations and deepCRE predictions",
        "Number of differences between reference and mutated sequence",
        "deepCRE Prediction",
        bucket_label,
        include_differences=False,
        delta=False,
        subset_label=subset_label,
    )

def load_starrseq_data(starrseq_file):
    starrseq_data = []
    starr_fasta = Fasta(starrseq_file)
    for record in starr_fasta:
        full_name = record.name
        record_parts = full_name.split("_")
        tf = record_parts[0]
        location = record_parts[1]
        tf_id = record_parts[-1]
        if len(record_parts) == 4:
            reference = False
            binding_status = record_parts[2]
        elif record_parts[2] == "reference":
            reference = True
            binding_status = "binding"
        else:
            reference = False
            binding_status = "non_binding"
        chrom, start_end = location.split(":")
        start, end = map(int, start_end.split("-"))
        starrseq_data.append({
            "full_name": full_name,
            "tf": tf,
            "tf_id": tf_id,
            "chrom": chrom,
            "start": start,
            "end": end,
            "binding_status": binding_status,
            "reference": reference,
            "sequence": str(starr_fasta[full_name])
        })
    return starrseq_data


def reverse_complement(seq: str) -> str:
    complement = str.maketrans('ACGTacgt', 'TGCAtgca')
    return seq.translate(complement)[::-1]


def adjust_positions(start_pos: int, end_pos: int, additional_padding: int, reverse: bool, len_ref_seq: int) -> Tuple[int, int, int, int]:
    """adjusts the start and end position based on how they overlap wit the borders of the deepCRE extraction window

    Args:
        start_pos (int): found start of the start query
        end_pos (int): found start of the end query
        additional_padding (int): additional padding for short genes
        reverse (bool): indicates whether the gene is in forward or reverse orientation
        len_ref_seq (int): length of the reference sequence

    Returns:
        Tuple[int, int]: adjust values for start and end position of the insert
    """
    # calculate the real padding start and end based on the additional padding for short genes and the orientation of the gene
    shorter_additional_padding_half = additional_padding // 2
    longer_additional_padding_half = shorter_additional_padding_half + (additional_padding % 2)

    # the promoter site is always the "longer" side, so for the forward orientation
    real_padding_start = IDEAL_PADDING_START - longer_additional_padding_half if reverse else IDEAL_PADDING_START - shorter_additional_padding_half
    real_padding_end = IDEAL_PADDING_END + shorter_additional_padding_half if reverse else IDEAL_PADDING_END + longer_additional_padding_half

    # check whether match is before or after padding and adjust missing positions accordingly
    if start_pos == -1:
        if end_pos <= real_padding_start:
            start_pos = 0
        else:
            start_pos = real_padding_end
    # adjust not found end pos accoring to location of start pos
    if end_pos == -1:
        if start_pos < real_padding_start:
            end_pos = real_padding_start
        else:
            end_pos = len_ref_seq
    # add 50 to the end but limit to the end of padding / full seuquence based on position
    else:
        if start_pos < real_padding_start:
            end_pos = min(end_pos + 50, real_padding_start)
        else:
            end_pos = min(end_pos + 50, len_ref_seq)

    # return if the match is spanning the padding region
    if start_pos < real_padding_start and end_pos > real_padding_start:
        return -1, -1, -1, -1
    #overlap must cover the actual changed sequence of the starrseq entry
    overlap_length = end_pos - start_pos
    if overlap_length < 100:
        return -1, -1, -1, -1
    return start_pos, end_pos, real_padding_start, real_padding_end

def find_overlap_positions(ref_seq: str, start_query: str, end_query: str, additional_padding: int) -> Tuple[int, int, int, int, bool]:
    start_pos = ref_seq.find(start_query)
    end_pos = ref_seq.find(end_query)
    reverse = False
    # return if neither sequence is found
    if start_pos == -1 and end_pos == -1:
        start_query_reverse = reverse_complement(start_query)
        end_query_reverse = reverse_complement(end_query)
        start_pos = ref_seq.find(start_query_reverse)
        end_pos = ref_seq.find(end_query_reverse)
        if start_pos == -1 and end_pos == -1:
            return -1, -1, -1, -1, False
        reverse = True
        start_pos, end_pos = end_pos, start_pos
    
    start_pos, end_pos, real_padding_start, real_padding_end = adjust_positions(start_pos, end_pos, additional_padding, reverse, len(ref_seq))
    return start_pos, end_pos, real_padding_start, real_padding_end, reverse

def map_starrseq_to_deepcre(
    starrseq_data: List[Dict],
    gene_data: List[Dict],
    site_to_gene_ids: Dict[str, List[str]],
) -> List[Dict]:
    """Align each STARR-seq entry to its candidate gene reference windows.

    Args:
        starrseq_data: Parsed STARR-seq entries from :func:`load_starrseq_data`.
        gene_data: Gene window candidates; each dict carries ``gene``,
            ``ref_seq``, ``ref_fitness``, ``start`` and ``end``.
        site_to_gene_ids: Maps a site key
            (``<tf>_<chrom>:<start>-<end>``) to the list of candidate gene ids
            for that site. Built per-TF by the calling script.

    Returns:
        One mapping dict per successful (STARR-seq entry, gene) alignment.
    """
    mapping_results = []
    unmapped_starrseq_entries = []
    for starr_entry in starrseq_data:
        start_query = starr_entry["sequence"][:50]
        end_query = starr_entry["sequence"][-50:]
        search_key = starr_entry["tf"] + "_" + starr_entry["chrom"] + ":" + str(starr_entry["start"]) + "-" + str(starr_entry["end"])
        gene_ids = site_to_gene_ids.get(search_key, [])
        curr_gene_candidates = [gene for gene in gene_data if gene["gene"] in gene_ids]
        for gene_entry in curr_gene_candidates:
            additional_padding = gene_entry.get("additional_padding", 0)
            vals = find_overlap_positions(gene_entry["ref_seq"], start_query, end_query, additional_padding)
            start_pos, end_pos, real_padding_start, real_padding_end, reverse = vals
            if (start_pos, end_pos) == (-1, -1):
                continue
            # check for overlap of the sequence in the reference sequence of the gene with the starrseq entry
            if (starr_entry["start"] < gene_entry["end"] + EXTRAGENIC) and (starr_entry["end"] > gene_entry["start"] - EXTRAGENIC):
                mapping_results.append({
                    "starr_full_name": starr_entry["full_name"],
                    "gene": gene_entry["gene"],
                    "overlap_start": start_pos,
                    "overlap_end": end_pos,
                    "starr_binding_status": starr_entry["binding_status"],
                    "starr_reference": starr_entry["reference"],
                    "deepcre_ref_fitness": gene_entry["ref_fitness"],
                    "ref_seq": gene_entry["ref_seq"],
                    "starr_sequence": starr_entry["sequence"],
                    "reverse": reverse,
                    "real_padding_start": real_padding_start,
                    "real_padding_end": real_padding_end,
                })
        if not mapping_results or mapping_results[-1]["starr_full_name"] != starr_entry["full_name"]:
            unmapped_starrseq_entries.append(starr_entry)
    genes_mapped = {mapping["gene"] for mapping in mapping_results}
    genes_in_deepcis = {gene["gene"] for gene in gene_data}
    genes_not_mapped = genes_in_deepcis - genes_mapped
    if genes_not_mapped:
        print("unmapped genes: ", len(genes_not_mapped))
        print(json.dumps(sorted(list(genes_not_mapped)), indent=2))
    return mapping_results

def compare_sequences(seq_1: str, seq_2: str) -> int:
    if len(seq_1) != len(seq_2):
        return -1
    return sum(1 for a, b in zip(seq_1, seq_2) if a != b)


def get_starrseq_fragment(starr_seq: str, overlap_start: int, overlap_end: int, real_padding_start: int, real_padding_end: int) -> str:
    seq_length = overlap_end - overlap_start
    if overlap_start == 0 or overlap_start == real_padding_end:
        return starr_seq[-seq_length:]
    elif overlap_end == real_padding_start or overlap_end == 2 * (INTRAGENIC + EXTRAGENIC) + MIN_PADDING:
        return starr_seq[:seq_length]
    else:
        return starr_seq


def build_sequences(mapping_results: List[Dict], max_differences: int = 15) -> Tuple[np.ndarray, List[Dict]]:
    seqs = []
    meta_data = []
    for mapping in mapping_results:
        ref_seq = mapping["ref_seq"]
        starr_seq = mapping["starr_sequence"]
        overlap_start = mapping["overlap_start"]
        overlap_end = mapping["overlap_end"]
        real_padding_start = mapping.get("real_padding_start", IDEAL_PADDING_START)
        real_padding_end = mapping.get("real_padding_end", IDEAL_PADDING_END)
        if mapping["reverse"]:
            starr_seq = reverse_complement(starr_seq)
        starr_seq_fragment = get_starrseq_fragment(starr_seq, overlap_start, overlap_end, real_padding_start, real_padding_end)
        mutated_seq = ref_seq[:overlap_start] + starr_seq_fragment + ref_seq[overlap_end:]
        differences = compare_sequences(ref_seq, mutated_seq)
        if differences < 0 or differences > max_differences:
            print(f"Skipping {mapping['starr_full_name']}, ({mapping['gene']}) due to high number of differences ({differences}) between ref and mutated sequence.")
        else:
            mutated_seq = one_hot_encode(mutated_seq)
            seqs.append(mutated_seq)
            mapping["differences"] = differences
            meta_data.append(mapping)

    seqs = np.array(seqs)
    return seqs, meta_data


def make_deepcre_predictions(seqs: np.ndarray, meta_data: List[Dict]) -> pd.DataFrame:
    # Placeholder for actual deepCRE predictions
    model = load_model(DEEPCRE_PATH)
    predictions = model.predict(seqs)
    predictions = predictions.flatten().tolist()
    results_df = pd.DataFrame(meta_data)
    results_df["prediction_mutated"] = predictions
    return results_df

def merge_with_starrseq_results(predictions_df: pd.DataFrame, starr_seq_results: pd.DataFrame) -> pd.DataFrame:
    # add the "enrichment" column from starr_seq_results to predictions_df.
    # compatible columns are "starr_full_name" in predictions_df and "id" in starr_seq_results
    condition_cols = ["id", "enrichment"] + (["condition"] if "condition" in starr_seq_results.columns else [])
    rel_results_df = starr_seq_results[condition_cols]
    merged_df = pd.merge(predictions_df, rel_results_df, left_on="starr_full_name", right_on="id", how="left")
    merged_df.drop(columns=["id"], inplace=True)
    return merged_df

def plot_deepcre_starrseq_correlation(prediction_df: pd.DataFrame, colored: bool = False, delta: bool = False, subset_label: str = ""):
    prediction_col = "delta_prediction" if delta else "prediction_mutated"
    enrichment_col = "delta_enrichment" if delta else "enrichment"
    delta_description = "delta_" if delta else ""
    save_name = f"{delta_description}deepcre_starrseq_correlation.png"
    return _plot_bucketed_correlation(
        prediction_df,
        prediction_col,
        enrichment_col,
        save_name,
        f"Correlation between {delta_description}deepCRE and {delta_description}STARR-seq",
        f"{delta_description}deepCRE Prediction",
        f"{delta_description}STARR-seq Enrichment",
        f"{delta_description}deepcre_starrseq",
        include_differences=True,
        use_buckets=ENABLE_BUCKETED_ANALYSIS,
        add_overall_fit_line=ADD_OVERALL_FIT_TO_BUCKETED_PLOTS,
        delta=delta,
        subset_label=subset_label,
    )


def plot_mutation_starrseq_correlation(prediction_df: pd.DataFrame, delta: bool = False, subset_label: str = ""):
    enrichment_col = "delta_enrichment" if delta else "enrichment"
    delta_description = "delta_" if delta else ""
    save_name = f"{delta_description}mutation_starrseq_correlation.png"
    return _plot_bucketed_correlation(
        prediction_df,
        "differences",
        enrichment_col,
        save_name,
        f"Correlation between number of mutations and {delta_description}STARR-seq enrichment",
        "Number of differences between reference and mutated sequence",
        f"{delta_description}STARR-seq Enrichment",
        f"{delta_description}mutation_starrseq",
        include_differences=False,
        use_buckets=ENABLE_BUCKETED_ANALYSIS,
        add_overall_fit_line=ADD_OVERALL_FIT_TO_BUCKETED_PLOTS,
        delta=delta,
        subset_label=subset_label,
    )

def plot_mutation_deepcre_correlation(prediction_df: pd.DataFrame, delta: bool = False, subset_label: str = ""):
    prediction_col = "delta_prediction" if delta else "prediction_mutated"
    delta_description = "delta_" if delta else ""
    save_name = f"{delta_description}mutation_deepcre_correlation.png"
    return _plot_bucketed_correlation(
        prediction_df,
        "differences",
        prediction_col,
        save_name,
        f"Correlation between number of mutations and {delta_description}deepCRE predictions",
        "Number of differences between reference and mutated sequence",
        f"{delta_description}deepCRE Prediction",
        f"{delta_description}mutation_deepcre",
        include_differences=False,
        use_buckets=ENABLE_BUCKETED_ANALYSIS,
        add_overall_fit_line=ADD_OVERALL_FIT_TO_BUCKETED_PLOTS,
        delta=delta,
        subset_label=subset_label,
    )


def simply_plot_multi(
    series: List[Tuple[List, List, str, str]],
    x_axis_name: str,
    y_axis_name: str,
    title: str,
    output_name: str,
    subfolder: str,
    log: bool = False,
    rolling_window: int = 11,
) -> None:
    """Plot multiple series of position-correlation data on a single figure.

    Args:
        series: List of (x_vals, y_vals, label, color) tuples, one per subset.
        x_axis_name: X-axis label.
        y_axis_name: Y-axis label.
        title: Plot title.
        output_name: Output filename stem (without extension).
        subfolder: Subdirectory under CORRELATION_OUTPUT_ROOT.
        log: Whether to use a log scale for the y-axis.
        rolling_window: Window size for the rolling average line.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for x_vals, y_vals, label, color in series:
        x_arr = np.array(x_vals)
        y_arr = np.array(y_vals)
        ax.scatter(x_arr, y_arr, color=color, alpha=0.15, s=30, zorder=1)
        if len(x_arr) >= rolling_window:
            x_rolling = np.convolve(x_arr, np.ones(rolling_window) / rolling_window, mode="valid")
            y_rolling = np.convolve(y_arr, np.ones(rolling_window) / rolling_window, mode="valid")
            ax.plot(x_rolling, y_rolling, color=color, linestyle="-", linewidth=3, label=label, zorder=2)
        else:
            ax.plot(x_arr, y_arr, color=color, linestyle="-", linewidth=3, label=label, zorder=2)
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)
    ax.set_title(title)
    ax.legend()
    if log:
        ax.set_yscale("log")
    output_folder = os.path.join(CORRELATION_OUTPUT_ROOT, subfolder)
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, f"{output_name}.png"), bbox_inches="tight", dpi=OUTPUT_DPI)
    plt.close(fig)



def plot_correlation_over_positions_fixed_window(
    named_dfs: List[Tuple[pd.DataFrame, str, str]],
) -> None:
    """Plot Spearman correlation, slope, and p-value by fixed-width position windows.

    Args:
        named_dfs: List of (dataframe, label, color) tuples, one per subset to overlay.
    """
    corr_series, slope_series, pval_series = [], [], []
    for df, label, color in named_dfs:
        sorted_df = df.sort_values("group_overlap_start").reset_index(drop=True)
        slopes, correlations, p_values, x_pos = [], [], [], []
        for start in sorted_df["group_overlap_start"].unique():
            end = start + BUCKET_SIZE
            bucket_df = sorted_df[
                (sorted_df["group_overlap_start"] >= start) & (sorted_df["group_overlap_start"] < end)
            ]
            if len(bucket_df) > 10:
                slope, _, correlation, p_value = _fit_linear_model(
                    bucket_df["prediction_mutated"], bucket_df["enrichment"]
                )
                slopes.append(slope)
                correlations.append(correlation)
                p_values.append(p_value)
                x_pos.append((start + end) / 2)
        corr_series.append((x_pos, correlations, label, color))
        slope_series.append((x_pos, slopes, label, color))
        pval_series.append((x_pos, p_values, label, color))

    subfolder = "correlation_by_position_fixed_window"
    simply_plot_multi(corr_series, "Overlap start position", "Spearman correlation", "Correlation between deepCRE predictions and STARR-seq enrichment by overlap position", "correlation_by_position", subfolder)
    simply_plot_multi(slope_series, "Overlap start position", "Slope of linear fit", "Slope of linear fit between deepCRE predictions and STARR-seq enrichment by overlap position", "slope_by_position", subfolder)
    simply_plot_multi(pval_series, "Overlap start position", "P-value of Spearman correlation", "P-value of correlation between deepCRE predictions and STARR-seq enrichment by overlap position", "pvalue_by_position", subfolder, log=True)



def plot_correlation_over_positions_fixed_number_elements(
    named_dfs: List[Tuple[pd.DataFrame, str, str]],
) -> None:
    """Plot Spearman correlation, slope, and p-value using fixed-size rolling element windows.

    Args:
        named_dfs: List of (dataframe, label, color) tuples, one per subset to overlay.
    """
    corr_series, slope_series, pval_series = [], [], []
    for df, label, color in named_dfs:
        sorted_df = df.sort_values("group_overlap_start").reset_index(drop=True)
        slopes, correlations, p_values, x_pos = [], [], [], []
        for i in range(len(sorted_df) - BUCKET_SIZE):
            bucket_df = sorted_df.iloc[i: i + BUCKET_SIZE]
            if len(bucket_df) > 10:
                slope, _, correlation, p_value = _fit_linear_model(
                    bucket_df["prediction_mutated"], bucket_df["enrichment"]
                )
                slopes.append(slope)
                correlations.append(correlation)
                p_values.append(p_value)
                x_pos.append(bucket_df["group_overlap_start"].mean())
        corr_series.append((x_pos, correlations, label, color))
        slope_series.append((x_pos, slopes, label, color))
        pval_series.append((x_pos, p_values, label, color))

    subfolder = "correlation_by_position_fixed_number_elements"
    simply_plot_multi(corr_series, "Overlap start position", "Spearman correlation", "Correlation between deepCRE predictions and STARR-seq enrichment by overlap position", "correlation_by_position", subfolder)
    simply_plot_multi(slope_series, "Overlap start position", "Slope of linear fit", "Slope of linear fit between deepCRE predictions and STARR-seq enrichment by overlap position", "slope_by_position", subfolder)
    simply_plot_multi(pval_series, "Overlap start position", "P-value of Spearman correlation", "P-value of correlation between deepCRE predictions and STARR-seq enrichment by overlap position", "pvalue_by_position", subfolder, log=True)

def calculate_deltas(prediction_df: pd.DataFrame) -> pd.DataFrame:
    # calculate the delta between the deepCRE prediction for the mutated sequence and the deepCRE prediction for the reference sequence
    prediction_df["delta_prediction"] = prediction_df["prediction_mutated"] - prediction_df["deepcre_ref_fitness"]
    # calculate the delta between the STARR-seq enrichment for the mutated sequence and the STARR-seq enrichment for the reference sequence
    prediction_df["starr_seq_base"] = prediction_df.apply(lambda row: "_".join(row["starr_full_name"].split("_")[:2]), axis=1)
    references = prediction_df[prediction_df["starr_reference"] == True]
    references = references[["starr_seq_base", "enrichment"]].rename(columns={"enrichment": "reference_enrichment"})
    references = references.drop_duplicates(subset=["starr_seq_base"])
    prediction_df = pd.merge(prediction_df, references, on="starr_seq_base", how="left")
    prediction_df["delta_enrichment"] = prediction_df["enrichment"] - prediction_df["reference_enrichment"]
    return prediction_df


def save_bucket_statistics(stat_rows: List[Dict[str, Union[str, int, float]]]) -> None:
    stats_df = pd.DataFrame(stat_rows)
    expected_columns = [
        "analysis",
        "bucket_label",
        "bucket_start",
        "bucket_end",
        "bucket_size",
        "count_points",
        "count_unique_points",
        "slope",
        "intercept",
        "spearman_r",
        "spearman_p",
    ]
    if stats_df.empty:
        # Preserve stable file layout even when there are no rows.
        for analysis_name in ["deepcre_starrseq", "mutation_starrseq", "mutation_deepcre"]:
            output_file = os.path.join(_get_analysis_output_dir(analysis_name), BUCKET_SUMMARY_FILE_NAME)
            pd.DataFrame(columns=expected_columns).to_csv(output_file, index=False)
        return

    stats_df = stats_df.sort_values(["analysis", "bucket_start"]).reset_index(drop=True)
    for analysis_name, analysis_df in stats_df.groupby("analysis", sort=False):
        output_file = os.path.join(_get_analysis_output_dir(str(analysis_name)), BUCKET_SUMMARY_FILE_NAME)
        analysis_df.to_csv(output_file, index=False)


def run_correlation_analysis(enrichment_df: pd.DataFrame) -> None:
    """Generate every correlation plot and bucket-statistics CSV for one dataset.

    This is the transcription-factor-agnostic analysis stage shared by all
    entry-point scripts: it produces the position-series plots and, for each
    subset (overall, reference/synthetic, binding/non-binding, and light/dark
    when a ``condition`` column with both values is present), the three scatter
    correlation plots, the individual bucket views, and the per-bucket fit
    statistics CSV.

    The output root must already be configured via :func:`set_output_root`; the
    caller owns it so that a pooled analysis can redirect the outputs without
    this function knowing which TF(s) produced ``enrichment_df``.

    Args:
        enrichment_df: Analysis-ready dataframe as returned by the per-TF
            preparation step (predictions merged with enrichment, deltas
            computed, and length-corrected overlap buckets assigned).
    """
    has_conditions = (
        "condition" in enrichment_df.columns
        and len(enrichment_df["condition"].unique()) > 1
    )
    if has_conditions:
        position_series: List[Tuple[pd.DataFrame, str, str]] = [
            (enrichment_df, "all", POSITION_SERIES_COLORS["all"]),
            (enrichment_df[enrichment_df["condition"].str.lower() == "light"].copy(), "light", POSITION_SERIES_COLORS["light"]),
            (enrichment_df[enrichment_df["condition"].str.lower() == "dark"].copy(), "dark", POSITION_SERIES_COLORS["dark"]),
        ]   #type: ignore
    else:
        position_series = [(enrichment_df, "all", POSITION_SERIES_COLORS["all"])]
    plot_correlation_over_positions_fixed_window(position_series)
    plot_correlation_over_positions_fixed_number_elements(position_series)
    subsets = [
        ("", enrichment_df),
        ("reference", enrichment_df[enrichment_df["starr_reference"] == True]),
        ("synthetic", enrichment_df[enrichment_df["starr_reference"] == False]),
        ("binding", enrichment_df[enrichment_df["starr_binding_status"] == "binding"]),
        ("non_binding", enrichment_df[enrichment_df["starr_binding_status"] == "non_binding"]),
    ]
    if has_conditions:
        subsets += [
            ("light", enrichment_df[enrichment_df["condition"].str.lower() == "light"]),
            ("dark", enrichment_df[enrichment_df["condition"].str.lower() == "dark"]),
        ]
    for subset_label, subset_df in subsets:
        bucket_stats: List[Dict[str, Union[str, int, float]]] = []
        bucket_stats.extend(plot_deepcre_starrseq_correlation(subset_df, colored=True, delta=False, subset_label=subset_label))
        bucket_stats.extend(plot_mutation_starrseq_correlation(subset_df, delta=False, subset_label=subset_label))
        bucket_stats.extend(plot_mutation_deepcre_correlation(subset_df, delta=False, subset_label=subset_label))
        for bucket_label in INDIVIDUAL_BUCKET_LABELS_TO_PLOT:
            plot_individual_bucket_views(subset_df, bucket_label, subset_label=subset_label)
        save_bucket_statistics(bucket_stats)
