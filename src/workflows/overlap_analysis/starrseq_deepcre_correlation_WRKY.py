import os
import json
from typing import Dict, List

import pandas as pd
from pyfaidx import Fasta

from workflows.overlap_analysis import _common
from workflows.overlap_analysis._common import (
    add_length_corrected_overlap_buckets,
    build_sequences,
    calculate_deltas,
    load_starrseq_data,
    make_deepcre_predictions,
    merge_with_starrseq_results,
)

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


def prepare_wrky_enrichment_df() -> pd.DataFrame:
    """Load WRKY inputs and run the prediction pipeline into an analysis-ready df.

    Performs the TF-specific data joining: parse the WRKY STARR-seq variants and
    deepCRE reference windows, map STARR-seq fragments onto reference windows,
    build and score the mutated sequences (WRKY's default ``max_differences`` of
    15), merge in STARR-seq enrichment, and compute deltas and overlap buckets.

    Does not configure matplotlib or the output root; the caller owns those so
    the same dataframe can feed either the WRKY-only or the pooled analysis.

    Returns:
        Enrichment dataframe ready for :func:`_common.run_correlation_analysis`.
    """
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
    return enrichment_df


def main():
    _common.configure_matplotlib()
    _common.set_output_root(os.path.join(BASE_DIR, "correlation", _DATA_VERSION))
    _common.run_correlation_analysis(prepare_wrky_enrichment_df())


if __name__ == "__main__":
    main()
