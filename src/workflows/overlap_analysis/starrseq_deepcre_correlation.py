import os
import json
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tensorflow.keras.models import load_model  # type: ignore
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from evolution.sequences import one_hot_encode

STARRSEQ_INPUT_FILE = "src/workflows/overlap_analysis/dCIS_WRKY_in_silico_mutated_GS2025d.fasta"
DEEP_CRE_RUN_FOLDER = "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/overlap/overlap_max_260326_201255_889602/all_positions_all"
STARR_SEQ_RESULTS = "src/workflows/overlap_analysis/plantstarr-seq_main_dark_simon_gernot(1).csv"
MAPPING_FILE = "src/workflows/overlap_analysis/WRKY_dCIS_dCRE_overlaps.csv"
REF_SEQ_FILE = "reference_sequence.fa"
DEEPCRE_PATH = "/home/gernot/Code/PhD_Code/Evolution/models/Atha_S0X0.75dP7K25g_NC_003075.7_ssr_train_models_250705_211854.h5"
PARETO_PATH = os.path.join("saved_populations", "pareto_front.json")
INTRAGENIC = 500
EXTRAGENIC = 1000
PADDING = 20
PADDING_START = INTRAGENIC + EXTRAGENIC
PADDING_END = PADDING_START + PADDING

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


def load_relevant_deepcre_window_candidates(run_folder: str):
    gene_folders = [os.path.join(run_folder, f) for f in os.listdir(run_folder) if os.path.isdir(os.path.join(run_folder, f))]
    gene_data = []
    for gene_folder in gene_folders:
        folder_name = os.path.basename(gene_folder)
        gene_name = folder_name.split("_")[1]
        start, end = folder_name.split("_")[2].split(":")[1].split("-")
        if start > end:
            start, end = end, start
        ref_seq_path = os.path.join(gene_folder, REF_SEQ_FILE)
        ref_fasta = Fasta(ref_seq_path)
        ref_seq = str(ref_fasta["reference_sequence_full"])
        pareto_path = os.path.join(gene_folder, PARETO_PATH)
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

def reverse_complement(seq: str) -> str:
    complement = str.maketrans('ACGTacgt', 'TGCAtgca')
    return seq.translate(complement)[::-1]

def find_overlap_positions(ref_seq: str, start_query: str, end_query: str) -> Tuple[int, int, bool]:
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
            return -1, -1, False
        reverse = True
        start_pos, end_pos = end_pos, start_pos
    # check whether match is before or after padding and adjust missing positions accordingly
    if start_pos == -1:
        if end_pos <= PADDING_START:
            start_pos = 0
        else:
            start_pos = PADDING_END
    # adjust not found end pos accoring to location of start pos
    if end_pos == -1:
        if start_pos < PADDING_START:
            end_pos = PADDING_START
        else:
            end_pos = len(ref_seq)
    # add 50 to the end but limit to the end of padding / full seuquence based on position
    else:
        if start_pos < PADDING_START:
            end_pos = min(end_pos + 50, PADDING_START)
        else:
            end_pos = min(end_pos + 50, len(ref_seq))

    # return if the match is spanning the padding region
    if start_pos < PADDING_START and end_pos > PADDING_START:
        return -1, -1, False
    #overlap must cover the actual changed sequence of the starrseq entry
    overlap_length = end_pos - start_pos
    if overlap_length < 100:
        return -1, -1, False
    return start_pos, end_pos, reverse

def get_gene_candidates_for_starrseq_entry(starr_entry: Dict, mapping_candidates: pd.DataFrame, gene_data: List[Dict]) -> List[Dict]:
    search_key = starr_entry["tf"] + "_" + starr_entry["chrom"] + ":" + str(starr_entry["start"]) + "-" + str(starr_entry["end"])
    candidate = mapping_candidates[mapping_candidates["deepCIS_segment"] == search_key]
    gene_ids = candidate["Gene_ID"].tolist()
    curr_candidates = [gene for gene in gene_data if gene["gene"] in gene_ids]
    return curr_candidates

def map_starrseq_to_deepcre(starrseq_data: List[Dict], gene_data: List[Dict], mapping_candidates: pd.DataFrame) -> List[Dict]:
    mapping_results = []
    unmapped_starrseq_entries = []
    for starr_entry in starrseq_data:
        start_query = starr_entry["sequence"][:50]
        end_query = starr_entry["sequence"][-50:]
        curr_gene_candidates = get_gene_candidates_for_starrseq_entry(starr_entry, mapping_candidates, gene_data)
        for gene_entry in curr_gene_candidates:
            start_pos, end_pos, reverse = find_overlap_positions(gene_entry["ref_seq"], start_query, end_query)
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
                    "reverse": reverse
                })
        if mapping_results and mapping_results[-1]["starr_full_name"] != starr_entry["full_name"]:
            unmapped_starrseq_entries.append(starr_entry)
    genes_mapped = {mapping["gene"] for mapping in mapping_results}
    genes_in_deepcis = {gene["gene"] for gene in gene_data}
    genes_not_mapped = genes_in_deepcis - genes_mapped
    print("unmapped genes: ", len(genes_not_mapped))
    print(json.dumps(sorted(list(genes_not_mapped)), indent=2))
    return mapping_results

def compare_sequences(seq_1: str, seq_2: str) -> int:
    if len(seq_1) != len(seq_2):
        return max(len(seq_1), len(seq_2))
    return sum(1 for a, b in zip(seq_1, seq_2) if a != b)


def get_starrseq_fragment(starr_seq: str, overlap_start: int, overlap_end: int) -> str:
    seq_length = overlap_end - overlap_start
    if overlap_start == 0 or overlap_start == PADDING_END:
        return starr_seq[-seq_length:]
    elif overlap_end == PADDING_START or overlap_end == 2 * (INTRAGENIC + EXTRAGENIC) + PADDING:
        return starr_seq[:seq_length]
    else:
        return starr_seq


def build_sequences(mapping_results: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    seqs = []
    meta_data = []
    for mapping in mapping_results:
        ref_seq = mapping["ref_seq"]
        starr_seq = mapping["starr_sequence"]
        overlap_start = mapping["overlap_start"]
        overlap_end = mapping["overlap_end"]
        if mapping["reverse"]:
            starr_seq = reverse_complement(starr_seq)
        starr_seq_fragment = get_starrseq_fragment(starr_seq, overlap_start, overlap_end)
        mutated_seq = ref_seq[:overlap_start] + starr_seq_fragment + ref_seq[overlap_end:]
        differences = compare_sequences(ref_seq, mutated_seq)
        if differences <= 15:
            mutated_seq = one_hot_encode(mutated_seq)
            seqs.append(mutated_seq)
            mapping["differences"] = differences
            meta_data.append(mapping)
        else:
            print(f"Skipping {mapping['starr_full_name']}, ({mapping['gene']}) due to high number of differences ({differences}) between ref and mutated sequence.")

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
    rel_results_df = starr_seq_results[["id", "enrichment"]]
    merged_df = pd.merge(predictions_df, rel_results_df, left_on="starr_full_name", right_on="id", how="left")
    merged_df.drop(columns=["id"], inplace=True)
    return merged_df

def plot_deepcre_starrseq_correlation(prediction_df: pd.DataFrame, colored: bool = False, delta: bool = False):
    plt.clf()
    prediction_col = "delta_prediction" if delta else "prediction_mutated"
    enrichment_col = "delta_enrichment" if delta else "enrichment"
    delta_description = "delta_" if delta else ""
    dedup = prediction_df[[prediction_col, enrichment_col, "differences", "starr_reference"]].drop_duplicates().dropna()
    if delta:
        dedup = dedup[dedup["starr_reference"] == False]
    dedup = dedup[[prediction_col, enrichment_col, "differences"]]
    #color the points by the number of differences between the reference and mutated sequence
    if colored:
        plt.scatter(dedup[prediction_col], dedup[enrichment_col], c=dedup["differences"], cmap="viridis")
        #show legend for the colorbar
        cbar = plt.colorbar()
        cbar.set_label("Number of differences between reference and mutated sequence")
        save_name = f"{delta_description}deepcre_starrseq_correlation_colored.png"
    else:
        plt.scatter(dedup[prediction_col], dedup[enrichment_col])
        save_name = f"{delta_description}deepcre_starrseq_correlation.png"
    plt.xlabel(f"{delta_description}deepCRE Prediction (mutated)")
    plt.ylabel(f"{delta_description}STARR-seq Enrichment")
    plt.title(f"Correlation between {delta_description}deepCRE predictions and {delta_description}STARR-seq enrichment")
    # add linear regression line
    m, b = np.polyfit(dedup[prediction_col], dedup[enrichment_col], 1)
    plt.plot(dedup[prediction_col], m * dedup[prediction_col] + b, color="black")
    #print results of the regression
    print(f"Linear regression of deepCRE-STARR-seq correlation: y = {m:.4f}x + {b:.4f}")
    correlation, p_value = spearmanr(dedup[prediction_col], dedup[enrichment_col])
    print(f"Spearman correlation: {correlation}, p-value: {p_value}")
    plt.savefig(f"src/workflows/overlap_analysis/{save_name}", bbox_inches="tight")


def plot_mutation_starrseq_correlation(prediction_df: pd.DataFrame, delta: bool = False):
    plt.clf()
    enrichment_col = "delta_enrichment" if delta else "enrichment"
    delta_description = "delta_" if delta else ""
    dedup = prediction_df[["differences", enrichment_col, "starr_reference"]].drop_duplicates().dropna()
    if delta:
        dedup = dedup[dedup["starr_reference"] == False]
    dedup = dedup[["differences", enrichment_col]]
    plt.scatter(dedup["differences"], dedup[enrichment_col])
    plt.xlabel("Number of differences between reference and mutated sequence")
    plt.ylabel(f"{delta_description}STARR-seq Enrichment")
    plt.title(f"Correlation between number of mutations and {delta_description}STARR-seq enrichment")
    #add linear regression line
    m, b = np.polyfit(dedup["differences"], dedup[enrichment_col], 1)
    plt.plot(dedup["differences"], m * dedup["differences"] + b, color="black")
    #print results of the regression
    print(f"Linear regression of mutation-STARR-seq correlation: y = {m:.4f}x + {b:.4f}")
    correlation, p_value = spearmanr(dedup["differences"], dedup[enrichment_col])
    print(f"Spearman correlation: {correlation}, p-value: {p_value}")
    plt.savefig(f"src/workflows/overlap_analysis/{delta_description}mutation_starrseq_correlation.png", bbox_inches="tight")

def plot_mutation_deepcre_correlation(prediction_df: pd.DataFrame, delta: bool = False):
    plt.clf()
    prediction_col = "delta_prediction" if delta else "prediction_mutated"
    delta_description = "delta_" if delta else ""
    dedup = prediction_df[[ "differences", prediction_col, "starr_reference"]].drop_duplicates().dropna()
    if delta:
        dedup = dedup[dedup["starr_reference"] == False]
    dedup = dedup[[ "differences", prediction_col]]
    plt.scatter(dedup["differences"], dedup[prediction_col])
    plt.xlabel("Number of differences between reference and mutated sequence")
    plt.ylabel(f"{delta_description}deepCRE Prediction (mutated)")
    plt.title(f"Correlation between number of mutations and {delta_description}deepCRE predictions")
    #add linear regression line
    m, b = np.polyfit(dedup["differences"], dedup[prediction_col], 1)
    plt.plot(dedup["differences"], m * dedup["differences"] + b, color="black")
    #print results of the regression
    print(f"Linear regression of mutation-deepCRE correlation: y = {m:.4f}x + {b:.4f}")
    correlation, p_value = spearmanr(dedup["differences"], dedup[prediction_col])
    print(f"Spearman correlation: {correlation}, p-value: {p_value}")
    plt.savefig(f"src/workflows/overlap_analysis/{delta_description}mutation_deepcre_correlation.png", bbox_inches="tight")

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

def main():
    starrseq_data = load_starrseq_data(STARRSEQ_INPUT_FILE)
    gene_data = load_relevant_deepcre_window_candidates(DEEP_CRE_RUN_FOLDER)
    mapping_candidates = pd.read_csv(MAPPING_FILE)
    starr_seq_results = pd.read_csv(STARR_SEQ_RESULTS)
    mapping_results = map_starrseq_to_deepcre(starrseq_data, gene_data, mapping_candidates)
    seqs, meta_data = build_sequences(mapping_results)
    prediction_df = make_deepcre_predictions(seqs, meta_data)
    enrichment_df = merge_with_starrseq_results(prediction_df, starr_seq_results)
    enrichment_df = enrichment_df.dropna(subset=["enrichment"]).reset_index(drop=True)
    enrichment_df = calculate_deltas(enrichment_df)
    plot_deepcre_starrseq_correlation(enrichment_df, colored=True, delta=False)
    plot_mutation_starrseq_correlation(enrichment_df, delta=False)
    plot_mutation_deepcre_correlation(enrichment_df, delta=False)


if __name__ == "__main__":
    main()