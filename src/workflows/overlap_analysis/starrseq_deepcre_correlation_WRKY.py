import os
import json
import pandas as pd
from pyfaidx import Fasta

from workflows.overlap_analysis import _common
from workflows.overlap_analysis._common import (
    INDIVIDUAL_BUCKET_LABELS_TO_PLOT,
    POSITION_SERIES_COLORS,
    add_length_corrected_overlap_buckets,
    build_sequences,
    calculate_deltas,
    load_starrseq_data,
    make_deepcre_predictions,
    merge_with_starrseq_results,
    plot_correlation_over_positions_fixed_number_elements,
    plot_correlation_over_positions_fixed_window,
    plot_deepcre_starrseq_correlation,
    plot_individual_bucket_views,
    plot_mutation_deepcre_correlation,
    plot_mutation_starrseq_correlation,
    save_bucket_statistics,
)
from typing import Dict, List, Tuple, Union

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

USE_NEW_DATA = True  # Set True to use new light+dark file; False to use old dark-only file

STARRSEQ_INPUT_FILE = os.path.join(DATA_DIR, "dCIS_WRKY_in_silico_mutated_GS2025d.fasta")
DEEP_CRE_RUN_FOLDER = "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/overlap/overlap_max_260326_201255_889602/all_positions_all"
_STARR_SEQ_OLD = os.path.join(DATA_DIR, "plantstarr-seq_main_dark_simon_gernot(1).csv")
_STARR_SEQ_NEW = os.path.join(DATA_DIR, "plantstarr-seq_main_light_and_dark_simon_gernot.csv")
STARR_SEQ_RESULTS = _STARR_SEQ_NEW if USE_NEW_DATA else _STARR_SEQ_OLD
MAPPING_FILE = os.path.join(DATA_DIR, "WRKY_dCIS_dCRE_overlaps.csv")
_DATA_VERSION = "new" if USE_NEW_DATA else "old"


def load_relevant_deepcre_window_candidates(run_folder: str):
    gene_folders = [os.path.join(run_folder, f) for f in os.listdir(run_folder) if os.path.isdir(os.path.join(run_folder, f))]
    gene_data = []
    for gene_folder in gene_folders:
        folder_name = os.path.basename(gene_folder)
        gene_name = folder_name.split("_")[1]
        start, end = folder_name.split("_")[2].split(":")[1].split("-")
        if start > end:
            start, end = end, start
        ref_seq_path = os.path.join(gene_folder, _common.REF_SEQ_FILE)
        ref_fasta = Fasta(ref_seq_path)
        ref_seq = str(ref_fasta["reference_sequence_full"])
        pareto_path = os.path.join(gene_folder, _common.PARETO_PATH)
        with open(pareto_path, "r") as f:
            pareto_data = json.load(f)
        ref_fitness = pareto_data[-1][1]
        gene_data.append({
            "gene": gene_name,
            "ref_fitness": ref_fitness,
            "folder_name": folder_name,
            "ref_seq": ref_seq,
            "start": int(start),
            "end": int(end)
        })
    return gene_data


def main():
    _common.configure_matplotlib()
    _common.set_output_root(os.path.join(BASE_DIR, "correlation", _DATA_VERSION))
    starrseq_data = load_starrseq_data(STARRSEQ_INPUT_FILE)
    gene_data = load_relevant_deepcre_window_candidates(DEEP_CRE_RUN_FOLDER)
    mapping_candidates = pd.read_csv(MAPPING_FILE)
    site_to_gene_ids = (
        mapping_candidates.groupby("deepCIS_segment")["Gene_ID"]
        .apply(list)
        .to_dict()
    )
    starr_seq_results = pd.read_csv(STARR_SEQ_RESULTS)
    mapping_results = _common.map_starrseq_to_deepcre(
        starrseq_data, gene_data, site_to_gene_ids,
    )
    seqs, meta_data = build_sequences(mapping_results)
    prediction_df = make_deepcre_predictions(seqs, meta_data)
    enrichment_df = merge_with_starrseq_results(prediction_df, starr_seq_results)
    enrichment_df = enrichment_df.dropna(subset=["enrichment"]).reset_index(drop=True)
    enrichment_df = calculate_deltas(enrichment_df)
    enrichment_df = add_length_corrected_overlap_buckets(enrichment_df)
    if "condition" in enrichment_df.columns and len(enrichment_df["condition"].unique()) > 1:
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
    if "condition" in enrichment_df.columns and len(enrichment_df["condition"].unique()) > 1:
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


if __name__ == "__main__":
    main()
