import os
from typing import List, Tuple
from pyfaidx import Fasta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model          #type: ignore
from deepCRE.utils import one_hot_encode
from tqdm import tqdm


def plot_single_gene_scatter(ssr_predictions: np.ndarray, msr_predictions: np.ndarray, gene_name: str, output_path: str, mutations: List[int]):
    plt.clf()
    plt.figure(figsize=(8,6))
    plt.scatter(mutations, ssr_predictions, label="SSR Predictions", alpha=0.6)
    plt.scatter(mutations, msr_predictions, label="MSR Predictions", alpha=0.6)
    plt.xlabel("Number of Mutations")
    plt.ylabel("Predicted Expression Level")
    plt.title(f"SSR vs MSR Predictions for {gene_name}")
    plt.legend()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"{gene_name}_ssr_msr_comparison.png"), bbox_inches='tight')


def plot_single_gene_diff(ssr_predictions: np.ndarray, msr_predictions: np.ndarray, gene_name: str, output_path: str, mutations: List[int]):
    plt.clf()
    plt.figure(figsize=(8,6))
    differences = ssr_predictions - msr_predictions
    plt.scatter(mutations, differences, label="SSR - MSR", alpha=0.6, color='orange')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Number of Mutations")
    plt.ylabel("Prediction Difference (SSR - MSR)")
    plt.title(f"Prediction Differences for {gene_name}")
    plt.legend()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"{gene_name}_ssr_msr_difference.png"), bbox_inches='tight')


def plot_single_run(data: pd.DataFrame, output_path: str):
    gene_names = data["Base_Name"].unique()
    for gene_name in gene_names:
        gene_data = data[data["Base_Name"] == gene_name]
        ssr_predictions = gene_data["SSR_Predictions"].values
        msr_predictions = gene_data["MSR_Predictions"].values
        mutations = gene_data["Mutations"].values
        plot_single_gene_scatter(ssr_predictions, msr_predictions, gene_name, output_path, mutations)
        plot_single_gene_diff(ssr_predictions, msr_predictions, gene_name, output_path, mutations)


def plot_goi_pareto_comparison(genes_of_interest: List[str], output_path: str, runs_of_interest: List[str]):
    data_path = os.path.join(os.path.dirname(__file__), "data", "GOF_LOF")
    run_fastas = [d for d in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, d)) and d.endswith(".fa")]
    run_fastas = [os.path.join(data_path, d) for d in run_fastas if any(run in d for run in runs_of_interest)]
    model_ssr = load_model("/home/gernot/Code/PhD_Code/Evolution/models/Zmay_S0X0.75dP7K25g_NC_050105.1_ssr_train_models_250705_211854.h5")
    for run_fasta in run_fastas:
        msr_predictions, sequences, mutations, base_names = parse_fasta(run_fasta, genes_of_interest)
        ssr_predictions = model_ssr.predict(np.array([one_hot_encode(seq) for seq in sequences])).flatten()
        frame = pd.DataFrame({
            "Base_Name": base_names,
            "Mutations": mutations,
            "MSR_Predictions": msr_predictions, 
            "SSR_Predictions": list(ssr_predictions)
        })
        plot_single_run(data = frame, output_path=output_path)


def parse_fasta(fasta_path: str, genes: List[str]) -> Tuple[List[float], List[str], List[int], List[str]]:
    fasta = Fasta(fasta_path)
    sequences, predictions, mutations, base_names = [], [], [], []
    for record in fasta:
        seq = str(record)
        name = record.name
        if genes and not any(gene in name for gene in genes):
            continue
        name_parts = name.split('_')
        version_number_appended = name_parts[-2] != "mutations"
        pred = float(name_parts[-4]) if version_number_appended else float(name_parts[-3])
        mutation = float(name_parts[-2]) if version_number_appended else float(name_parts[-1])
        base_name = '_'.join(name_parts[:-4]) if version_number_appended else '_'.join(name_parts[:-3])
        mutations.append(mutation)
        sequences.append(seq)
        predictions.append(pred)
        base_names.append(base_name)
    return predictions, sequences, mutations, base_names


def compare_predictions(genes: List[str]):
    data_path = os.path.join(os.path.dirname(__file__), "data", "GOF_LOF")
    run_fastas = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, d)) and d.endswith(".fa")]
    msr_vals, seqs = [], []
    differences, relative_differences = [], []
    model_ssr = load_model("/home/gernot/Code/PhD_Code/Evolution/models/Zmay_S0X0.75dP7K25g_NC_050105.1_ssr_train_models_250705_211854.h5")

    print("starting to gather sequences...")
    for run_fasta in run_fastas:
        predictions, sequences, _, _ = parse_fasta(run_fasta, genes=genes)
        msr_vals.extend(predictions)
        seqs.extend(sequences)
    
    print("got all sequences!")
    print(len(seqs), len(msr_vals))
    chunk_size=512
    for i in tqdm(range(0, len(seqs), chunk_size)):
        curr_seqs = seqs[i:i+chunk_size]
        curr_seqs = [one_hot_encode(curr_seq) for curr_seq in curr_seqs]
        curr_seqs = np.array(curr_seqs)
        curr_msr_vals = np.array(msr_vals[i:i+chunk_size])
        ssr_predictions = model_ssr.predict(curr_seqs, verbose=0).flatten()
        difference = ssr_predictions - curr_msr_vals
        relative_difference = np.log(ssr_predictions / curr_msr_vals)
        differences.append(difference)
        relative_differences.append(relative_difference)
    difference = np.concatenate(differences)
    relative_difference = np.concatenate(relative_differences)
    print(f"Difference: min {difference.min()}, max {difference.max()}, mean {difference.mean()}")
    print(f"Log Ratio: min {relative_difference.min()}, max {relative_difference.max()}, mean {relative_difference.mean()}")
    

if __name__ == "__main__":
    # compare_predictions(["AT3G59900", "AT5G24800", "AT2G41230", "AT4G24020", "AT2G47060", "AT1G16540"])
    plot_goi_pareto_comparison(
        genes_of_interest=["AT3G59900", "AT5G24800", "AT2G41230", "AT4G24020", "AT2G47060", "AT1G16540"],
        output_path=os.path.join(os.path.dirname(__file__), "data", "GOF_LOF", "ssr_msr_comparisons"),
        runs_of_interest=["GOF_single"]
    )

