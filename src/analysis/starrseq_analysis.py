from typing import Callable, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import tqdm
import os
import json
import scipy.stats as stats
from pyfaidx import Fasta
from sklearn import linear_model
from tensorflow.keras.models import load_model  # type: ignore
from deepCRE.utils import one_hot_encode
from evolution.write_run_script import find_correct_WRKY_position, find_correct_bHLH_position


def reverse_complement(seq: str) -> str:
    complement = str.maketrans('ACGTacgt', 'TGCAtgca')
    return seq.translate(complement)[::-1]


def load_df():
    data = pd.read_csv("data/plantstarr-seq_main_dark_simon_gernot.csv")
    data["reference"] = data["id"].str.find("reference") > -1
    data["binding"] = data["id"].str.find("non_binding") <= -1
    data["base_sequence"] = data["id"].str.extract(r'_(.+?)_')[0]
    return data


def simplest_stats():
    data = load_df()
    data = data[["id", "binding", "enrichment"]]
    for tf in ["WRKY", "bHLH"]:
        print(f"\nGENE: {tf}")
        relevant_data = data[data["id"].str.startswith(tf)]
        binding_data = relevant_data[relevant_data["binding"]]["enrichment"]
        non_binding_data = relevant_data[~relevant_data["binding"]]["enrichment"]
        print(f"binding mean: {binding_data.mean()}")
        print(f"non-binding mean: {non_binding_data.mean()}")
        #mann whitney test
        stat, p = scipy.stats.mannwhitneyu(binding_data, non_binding_data, alternative="two-sided")
        print(f"Mann-Whitney U test for {tf}: U={stat}, p={p}")


def visualize_simple():
    data = load_df()
    data = data[["id", "binding", "enrichment"]]
    for tf in ["WRKY", "bHLH"]:
        relevant_data = data[data["id"].str.startswith(tf)]
        relevant_data["binding"] = relevant_data["binding"].apply(lambda x: "binding" if x else "non-binding")
        plt.figure(figsize=(6.666666, 4))
    
        # increase all fonts
        sns.set_context("notebook", font_scale=1.2)
        sns.boxplot(x="binding", y="enrichment", data=relevant_data)
        #fix order of the box plots
        plt.xticks([0, 1], ["Non-binding", "Binding"])
        # remove x-axis label
        plt.xlabel("")
        # add a line at 0
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        # for each boxplot add how many data points are represented by it
        box_pairs = [("Non-binding", "Binding")]
        for box_pair in box_pairs:
            non_binding_count = relevant_data[relevant_data["binding"] == "non-binding"].shape[0]
            binding_count = relevant_data[relevant_data["binding"] == "binding"].shape[0]
            plt.text(x=0.2, y=1.5, s=f"n={non_binding_count}", ha='center', va='bottom')
            plt.text(x=1.2, y=1.5, s=f"n={binding_count}", ha='center', va='bottom')

        # add bar connecting both box plots with a star, indicating significance
        plt.title(f"Enrichment for {tf} (Binding vs Non-binding)")
        plt.savefig(f"enrichment_{tf}.pdf", bbox_inches='tight')


def visualize_simple_2():
    data = load_df()
    data = data[["id", "binding", "enrichment"]]
    # Extract TF name from id
    data["TF"] = data["id"].str.extract(r"^(WRKY|bHLH)")
    # Keep only WRKY and bHLH
    data = data[data["TF"].isin(["WRKY", "bHLH"])]
    # Convert binding boolean → categorical labels
    data["binding"] = data["binding"].apply(lambda x: "Binding" if x else "Non-binding")

    # --- Plot ---
    plt.figure(figsize=(7, 5*7/8))
    sns.set_context("notebook", font_scale=1.1)  # slightly larger, nice for presentations

    ax = sns.boxplot(
        x="TF", y="enrichment", hue="binding", data=data,
        order=["WRKY", "bHLH"], hue_order=["Non-binding", "Binding"],
        gap=0.1
    )

    # Axis labels
    plt.xlabel("Transcription Factor")
    plt.ylabel("Enrichment")

    # Add a solid black line at y=0
    # plt.axhline(0, color="black", linestyle="-", linewidth=0.8)
    # plt.axhline(0, color=(0.5, 0.5, 0.5), linestyle='--', linewidth=0.8)

    # Add counts above each box
    box_positions = {}
    for i, tf in enumerate(["WRKY", "bHLH"]):
        for j, binding in enumerate(["Non-binding", "Binding"]):
            subset = data[(data["TF"] == tf) & (data["binding"] == binding)]
            n = subset.shape[0]

            # Calculate the x position of the box (Seaborn offsets hue groups)
            xpos = i + (j - 0.5) * 0.65  # 0.2 is approx hue offset
            ymean = subset["enrichment"].mean() if not subset.empty else 0
            ax.text(
                xpos,
                ymean + 1.5,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=12,
                color="black",
            )

    # Legend (explicit title)
    plt.legend(title="Binding status", loc="lower center", bbox_to_anchor=(0.5, 0.1), fontsize=11, title_fontsize=12)

    # Title
    plt.title("Enrichment (Binding vs Non-binding) for WRKY and bHLH")

    # Save & close
    plt.savefig("enrichment_WRKY_bHLH.pdf", bbox_inches="tight")
    plt.close()


def calc_averages():
    data = load_df()
    for tf in ["WRKY", "bHLH"]:
        relevant_data = data[data["id"].str.startswith(tf)]
        relevant_references = relevant_data[relevant_data["reference"]]
        non_references = relevant_data[~relevant_data["reference"]]
        binding = non_references[non_references["binding"]]
        non_binding = non_references[~non_references["binding"]]
        print(f"GENE: {tf}")
        print(f"average reference enrichment: {relevant_references['enrichment'].mean()}")
        print(f"average binding enrichment: {binding['enrichment'].mean()}")
        print(f"average non-binding enrichment: {non_binding['enrichment'].mean()}")


def describe_data():
    data = load_df()
    for tf in ["WRKY", "bHLH"]:
        print(f"\nGENE: {tf}")
        relevant_data = data[data["id"].str.startswith(tf)]
        print(len(relevant_data))
        # 1. How many variants per reference_id?
        variants_per_ref = relevant_data.groupby("base_sequence")["id"].nunique()
        print("Number of variants per base_sequence:")
        print(variants_per_ref.describe())
        print(variants_per_ref.value_counts().sort_index())

        # 2. How many binding vs non-binding variants per base_sequence?
        binding_counts = relevant_data.groupby(["base_sequence", "binding"])["id"].nunique().unstack(fill_value=0)
        print("\nBinding vs Non-binding per base_sequence:")
        print(binding_counts.describe())

        # 3. Cases with exactly 2 variants total
        two_variant_refs = binding_counts[binding_counts.sum(axis=1) == 2]
        print("\nbase_sequences with exactly 2 variants:")
        print(two_variant_refs.describe())

        # 4. Cases with 1 binding + 1 non-binding
        one_vs_one_refs = binding_counts[(binding_counts[True] == 1) & (binding_counts[False] == 1)]
        print("\nbase_sequences with exactly 1 binding and 1 non-binding variant:")
        print(one_vs_one_refs.describe())


def analyze_differences_permutation():
    data = load_df()
    columns = data.columns
    columns = ['id', 'base_sequence', 'binding', 'reference', 'enrichment']
    data = data[columns]
    for tf in ["WRKY", "bHLH"]:
        print(f"\nGENE: {tf}")
        relevant_data = data[data["id"].str.startswith(tf)]
        diffs = (
            relevant_data.groupby(["base_sequence", "binding"])["enrichment"]
            .mean()
            .unstack()  # creates columns "binding" and "non-binding"
            .dropna()
        )

        # calculate difference (binding - non-binding)
        diffs["diff"] = diffs[True] - diffs[False]

        # --- 2. Define permutation test function ---
        def paired_permutation_test(n_perm=10000):
            observed = diffs["diff"].abs().mean()
            null_dist = []

            for i in tqdm.tqdm(range(n_perm)):
                shuffled = (
                    relevant_data.groupby("base_sequence")
                    .apply(lambda g: g.assign(
                        binding=np.random.permutation(g["binding"].values) #type: ignore
                    ))
                    .reset_index(drop=True)
                )
                perm_diffs = (
                    shuffled.groupby(["base_sequence", "binding"])["enrichment"]
                    .mean()
                    .unstack()
                    .dropna()
                )
                perm_diffs["diff"] = perm_diffs[True] - perm_diffs[False]
                null_dist.append(perm_diffs["diff"].abs().mean())

            p_val = (np.sum(np.array(null_dist) >= observed) + 1) / (n_perm + 1)
            return observed, np.mean(null_dist), p_val

        # --- 3. Run test ---
        observed, p_value, null_dist = paired_permutation_test()

        print(f"Observed mean difference: {observed:.3f}")
        print(f"P-value (two-sided): {p_value:.4f}")


def analyze(exclude_references: bool = True):
    data = load_df()
    for tf in ["WRKY", "bHLH"]:
        print(f"\nGENE: {tf}")
        relevant_data = data[data["id"].str.startswith(tf)]
        relevant_data = relevant_data.drop(columns=["GC", "length", "min_bc", "min_ci", "min_co", "n_experiments"])
        relevant_references = relevant_data[relevant_data["reference"]]
        if exclude_references:
            relevant_data = relevant_data[~relevant_data["reference"]]
        relevant_references = relevant_references[["base_sequence", "enrichment"]]
        relevant_data = relevant_data.merge(relevant_references, on="base_sequence", suffixes=("", "_reference"), how="left")
        relevant_data["delta_enrichment"] = abs(relevant_data["enrichment"] - relevant_data["enrichment_reference"])
        binding_delta = relevant_data[relevant_data["binding"]]["delta_enrichment"]
        non_binding_delta = relevant_data[~relevant_data["binding"]]["delta_enrichment"]
        binding_normal = scipy.stats.normaltest(binding_delta)
        non_binding_normal = scipy.stats.normaltest(non_binding_delta)
        if binding_normal.pvalue < 0.05 and non_binding_normal.pvalue < 0.05:
            # use t-test
            print(f"performing t-test because both distributions are normal")
            t_stat, p_value = scipy.stats.ttest_ind(binding_delta, non_binding_delta)
        else:
            # use Mann-Whitney U test
            print(f"performing Mann-Whitney U test because at least one distribution is not normal")
            u_stat, p_value = scipy.stats.mannwhitneyu(binding_delta, non_binding_delta, alternative="two-sided", nan_policy='omit')
        average_binding = binding_delta.mean()
        average_non_binding = non_binding_delta.mean()
        print(f"Average absolute difference binding: {average_binding}")
        print(f"Average absolute difference non-binding: {average_non_binding}")
        print(f"Statistical test result: p-value={p_value}")


def visualize_differences():
    data = load_df()
    for tf, ylim in zip(["WRKY", "bHLH"], [650, 1200]):
        relevant_data = data[data["id"].str.startswith(tf)]
        relevant_data = relevant_data.drop(columns=["GC", "length", "min_bc", "min_ci", "min_co", "n_experiments"])
        relevant_data["base_sequence"] = relevant_data["id"].str.extract(r'_(.+?)_')[0]
        relevant_references = relevant_data[relevant_data["reference"]]
        relevant_references = relevant_references[["base_sequence", "enrichment"]]
        relevant_data = relevant_data.merge(relevant_references, on="base_sequence", suffixes=("", "_reference"), how="left")
        relevant_data["delta_enrichment"] = abs(relevant_data["enrichment"] - relevant_data["enrichment_reference"])
        binding_delta = relevant_data[relevant_data["binding"]]["delta_enrichment"]
        non_binding_delta = relevant_data[~relevant_data["binding"]]["delta_enrichment"]
        plt.clf()
        #create histograms comparing binding and non binding
        plt.subplot(1, 2, 1)
        plt.hist(binding_delta, bins=30, color="blue", alpha=0.7)
        plt.title("Binding")
        plt.subplot(1, 2, 2)
        plt.hist(non_binding_delta, bins=30, color="red", alpha=0.7)
        plt.title("Non-Binding")
        # make y axis scale the same for both
        plt.ylim(0, ylim)
        plt.savefig(f"{tf}_enrichment_differences_hist.png", bbox_inches='tight')
        # plt.subplot(1, 2, 1)
        # sns.swarmplot(data=binding_delta, color="blue")
        # plt.title("Binding")
        # plt.subplot(1, 2, 2)
        # sns.swarmplot(data=non_binding_delta, color="red")
        # plt.title("Non-Binding")
        # plt.suptitle(f"Enrichment differences for {tf}")
        # plt.savefig(f"{tf}_enrichment_differences_swarm.png", bbox_inches='tight')
        # plt.close()


def starrseq_sequences_for_deepCIS_predictions():
    data = load_df()
    ids = data["id"].tolist()
    WRKY_ids = [id for id in ids if id.startswith("WRKY_")]
    bHLH_ids = [id for id in ids if id.startswith("bHLH_")]
    source_sequences = Fasta("/home/gernot/ARCitect/ARCs/genRE/assays/Evolution/dataset/Starr_seq_tobias/dCIS_candidate_hubs/dCIS_candidate_hub_sequences_cleaned_no_duplicates.fa")
    submitted_sequences_bHLH = Fasta("data/dCIS_bHLH_in_silico_mutated_GS2025d.fasta")
    submitted_sequences_WRKY = Fasta("data/dCIS_WRKY_in_silico_mutated_GS2025d.fasta")
    bHLH_list = make_sequences(ids=bHLH_ids, source_sequences=source_sequences, submitted_sequences=submitted_sequences_bHLH)
    wrky_list = make_sequences(ids=WRKY_ids, source_sequences=source_sequences, submitted_sequences=submitted_sequences_WRKY)
    with open("data/dCIS_bHLH_extended_for_deepCIS.fasta", "w") as f:
        f.write("\n".join(bHLH_list))
    with open("data/dCIS_WRKY_extended_for_deepCIS.fasta", "w") as f:
        f.write("\n".join(wrky_list))

def make_sequences(ids: List[str], source_sequences: Fasta, submitted_sequences: Fasta):
    output_list = []
    for id in ids:
        source_id = id.split("_")[1]
        source_sequence = source_sequences[source_id][:].seq        #type: ignore
        submitted_sequence = submitted_sequences[id][:].seq         #type: ignore
        extended_sequence = source_sequence[:40] + submitted_sequence + source_sequence[-40:]
        if "reference" in id:
            if not source_sequence[40:-40] == submitted_sequence:
                print(f"bHLH MISMATCH for {id}:")
                print(f"source:    {source_sequence}")
                print(f"submitted: {submitted_sequence}")
                print("----")
                raise ValueError("sequences do not match")
            else:
                print("worked :)")
        output_list.append(f">{id}")
        output_list.append(extended_sequence)

def make_deepCIS_predictions():
    meta_data = pd.read_csv("/home/gernot/ARCitect/ARCs/genRE/assays/Evolution/dataset/Starr_seq_tobias/dCIS_candidate_hubs/dCIS_candidate_hubs_background.csv", index_col=False)
    trimmed_cols = meta_data.columns[4:-3].to_list()
    trimmed_cols = [col.split("_tnt")[0] for col in trimmed_cols]
    deepcis = load_model("/home/gernot/Code/PhD_Code/Evolution/models/deepCIS_model_chrom_1_model.h5")
    wrky_sequences = Fasta("data/dCIS_WRKY_extended_for_deepCIS.fasta")
    bHLH_sequences = Fasta("data/dCIS_bHLH_extended_for_deepCIS.fasta")
    wrky_ids = list(wrky_sequences.keys())
    bHLH_ids = list(bHLH_sequences.keys())
    tensor = []
    for id in wrky_ids:
        seq = wrky_sequences[id][:].seq  #type: ignore
        encoded = one_hot_encode(seq)
        tensor.append(encoded)
    tensor = np.array(tensor)
    wrky_pred = deepcis.predict(tensor)
    wrky_df = pd.DataFrame(wrky_pred, columns=trimmed_cols, index=wrky_ids)
    wrky_df["sequence"] = [str(wrky_sequences[id][:].seq) for id in wrky_ids]  #type: ignore
    wrky_df.to_csv("data/deepCIS_predictions_WRKY.csv")
    tensor = []
    for id in bHLH_ids:
        seq = bHLH_sequences[id][:].seq  #type: ignore
        encoded = one_hot_encode(seq)
        tensor.append(encoded)
    tensor = np.array(tensor)
    bHLH_pred = deepcis.predict(tensor)
    bHLH_df = pd.DataFrame(bHLH_pred, columns=trimmed_cols, index=bHLH_ids)
    bHLH_df["sequence"] = [str(bHLH_sequences[id][:].seq) for id in bHLH_ids]  #type: ignore
    bHLH_df.to_csv("data/deepCIS_predictions_bHLH.csv")
    return wrky_df, bHLH_df


def get_match_score(sequence: str, motif: str) -> int:
    score = 0
    for a, b in zip(sequence.lower(), motif.lower()):
        if a == b:
            score += 1
    return score

def get_better_match(sequence: str, forward_motif: str, reverse_motif: str) -> str:
    forward_score = get_match_score(sequence, forward_motif)
    reverse_score = get_match_score(sequence, reverse_motif)
    if forward_score >= reverse_score:
        return forward_motif
    else:
        return reverse_motif


def find_motifs():
    wrky_df, bhlh_df = make_deepCIS_predictions()
    wrky_target = "WRKY"
    wrky_length = 5
    forward_motif_wrky = "GTCAA"
    reverse_motif_wrky = "TTGAC"
    output_path_wrky = "data/deepCIS_predictions_WRKY_additional_info.csv"
    wrky_data = find_motif_epm(target=wrky_target, motif_length=wrky_length, data=wrky_df, forward_motif=forward_motif_wrky, reverse_motif=reverse_motif_wrky, output_path=output_path_wrky, position_finder=find_correct_WRKY_position)

    bhlh_target = "bHLH"
    bhlh_length = 6
    forward_motif_bHLH = reverse_motif_bHLH = "CACGTG"
    print(forward_motif_bHLH, reverse_motif_bHLH)
    output_path_bhlh = "data/deepCIS_predictions_bHLH_additional_info.csv"
    bhlh_data = find_motif_epm(target=bhlh_target, motif_length=bhlh_length, data=bhlh_df, forward_motif=forward_motif_bHLH, reverse_motif=reverse_motif_bHLH, output_path=output_path_bhlh, position_finder=find_correct_bHLH_position)
    return wrky_data, bhlh_data

def find_motif_epm(target: str, motif_length: int, data: pd.DataFrame, forward_motif: str, reverse_motif: str, output_path: str, position_finder: Callable):
    meta_data = pd.read_csv("/home/gernot/ARCitect/ARCs/genRE/assays/Evolution/dataset/Starr_seq_tobias/dCIS_candidate_hubs/dCIS_candidate_hubs_background.csv", index_col=False)
    meta_data = meta_data.loc[meta_data["target_column"] == target]
    ref_seqs = Fasta("/home/gernot/ARCitect/ARCs/genRE/assays/Evolution/dataset/Starr_seq_tobias/dCIS_candidate_hubs/dCIS_candidate_hub_sequences_cleaned_no_duplicates.fa")
    data["reference"] = ""
    data["start"] = -1
    data["end"] = -1
    data["reference_core_clean"] = ""
    data["orientation"] = ""
    for id in data.index:
        base_id = id.split("_")[1]
        if base_id not in meta_data["seq_id"].values:
            print(f"{base_id} not in meta data")
            continue
        matching_EPM = meta_data.loc[meta_data["seq_id"] == base_id, "label"].item()
        reference_sequence = ref_seqs[base_id][:].seq  #type: ignore
        sequence = data.loc[id, "sequence"]
        motif_start = position_finder(reference_sequence, matching_EPM)
        motif_end = motif_start + motif_length
        core = sequence[motif_start:motif_end]
        reference_core = reference_sequence[motif_start:motif_end]
        reference_clean_core = get_better_match(sequence=reference_core, forward_motif=forward_motif, reverse_motif=reverse_motif)
        extended_core = sequence[motif_start-1:motif_end+1]
        leading_flank = sequence[motif_start-1]
        trailing_flank = sequence[motif_end]
        data.loc[id, "reference"] = reference_sequence
        data.loc[id, "start"] = motif_start
        data.loc[id, "end"] = motif_end
        data.loc[id, "reference_core_clean"] = reference_clean_core
        data.loc[id, "orientation"] = "forward" if reference_clean_core == forward_motif else "reverse"
        # print(core, extended_core, leading_flank, trailing_flank)
        # print(core, reference_core, reference_clean_core, core == reference_core)

    # print(data.describe())
    data.to_csv(output_path)
    return data
    
def add_starrseq_results():
    results = pd.read_csv("data/plantstarr-seq_main_dark_simon_gernot.csv")
    wrky_data, bhlh_data = find_motifs()
    results = results.set_index("id")
    results = results[["enrichment"]]
    wrky_data = wrky_data.join(results, how="left")
    bhlh_data = bhlh_data.join(results, how="left")
    wrky_data.to_csv("data/deepCIS_predictions_WRKY_additional_info_with_starrseq.csv")
    bhlh_data.to_csv("data/deepCIS_predictions_bHLH_additional_info_with_starrseq.csv")
    return wrky_data, bhlh_data


def binding_boxplots(data: Dict[str, pd.DataFrame], name_thing: str):
     # Prepare data for plotting
    plot_data = []
    
    for name, df in data.items():
        if len(df) == 0:
            continue
            
        # Add binding data
        binding_subset = df[df["binding"]].copy()
        if len(binding_subset) > 0:
            binding_subset["category"] = name
            binding_subset["binding_status"] = "Binding"
            plot_data.append(binding_subset[["enrichment", "category", "binding_status"]])
        
        # Add non-binding data
        non_binding_subset = df[~df["binding"]].copy()
        if len(non_binding_subset) > 0:
            non_binding_subset["category"] = name
            non_binding_subset["binding_status"] = "Non-binding"
            plot_data.append(non_binding_subset[["enrichment", "category", "binding_status"]])
    
    if not plot_data:
        print("No data to plot")
        return
    
    # Combine all data
    combined_data = pd.concat(plot_data, ignore_index=True)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.set_context("notebook", font_scale=1.0)
    
    # Create boxplot with hue for binding status
    ax = sns.boxplot(
        data=combined_data,
        x="category", 
        y="enrichment", 
        hue="binding_status",
        hue_order=["Non-binding", "Binding"],
        palette=["blue", "orange"]
    )
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add horizontal line at y=0
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    # Customize labels and title
    plt.xlabel("Data Category")
    plt.ylabel("Enrichment")
    plt.title("Enrichment Distribution by Binding Status Across Categories")
    
    # Adjust legend
    plt.legend(title="Binding Status", loc="lower center")
    
    # Add sample sizes above each box
    for i, category in enumerate(combined_data["category"].unique()):
        for j, status in enumerate(["Non-binding", "Binding"]):
            subset = combined_data[
                (combined_data["category"] == category) & 
                (combined_data["binding_status"] == status)
            ]
            if len(subset) > 0:
                # Calculate x position (category index + hue offset)
                x_pos = i + (j - 0.5) * 0.4  # 0.4 is approximate hue spacing
                y_pos = subset["enrichment"].max() + 0.1
                ax.text(
                    x_pos, y_pos, f"n={len(subset)}", 
                    ha="center", va="bottom", fontsize=9
                )
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"binding_status_{name_thing}_boxplots.pdf", bbox_inches="tight")


def binding_vs_non_binding(data: pd.DataFrame, name: str):
    data = data.copy()
    data["core_mutated"] = data.apply(lambda x: apply_core(x, sequence_col="reference"), axis=1) != data.apply(apply_core, axis=1)
    data["flanks_mutated"] = data.apply(apply_flanking, axis=1) != data.apply(lambda row: apply_flanking(row, sequence_col="reference"), axis=1)
    full_data = data
    only_core_mutated_data = data[data["core_mutated"] & ~data["flanks_mutated"]]
    only_flanks_mutated_data = data[data["flanks_mutated"] & ~data["core_mutated"]]
    both_mutated_data = data[data["core_mutated"] & data["flanks_mutated"]]
    none_mutated_data = data[~data["core_mutated"] & ~data["flanks_mutated"]]
    flanks_mutated = data[data["flanks_mutated"]]
    core_mutated = data[data["core_mutated"]]
    diff_splits_dict = {
        "Full data": full_data,
        "Only core mutated": only_core_mutated_data,
        "Only flanks mutated": only_flanks_mutated_data,
        "Both mutated": both_mutated_data,
        "None mutated": none_mutated_data,
        "Flanks mutated": flanks_mutated,
        "Core mutated": core_mutated
    }
    binding_boxplots(diff_splits_dict, name_thing=name)
    for curr_name, curr_data in diff_splits_dict.items():
        print(f"\n--- {curr_name} ---")
        binding_data = curr_data[curr_data["binding"]]
        non_binding_data = curr_data[~curr_data["binding"]]
        binding_mean = binding_data["enrichment"].mean()
        non_binding_mean = non_binding_data["enrichment"].mean()
        print(f"Binding mean: {binding_mean}")
        print(f"Non-binding mean: {non_binding_mean}")
        if len(binding_data) == 0 and len(non_binding_data) == 0:
            print("No data")
            continue
        if len(binding_data) == 0:
            print("No binding data")
            continue
        if len(non_binding_data) == 0:
            print("No non-binding data")
            continue
        stat, p = scipy.stats.mannwhitneyu(binding_data["enrichment"], non_binding_data["enrichment"], alternative="two-sided")
        print(f"Mann-Whitney U test: U={stat}, p={p}")


def apply_flanking(row, sequence_col: str = "sequence"):
    return row[sequence_col][:row["start"]] + row[sequence_col][row["end"]:]

def apply_core(row, sequence_col: str = "sequence"):
    return row[sequence_col][row["start"]:row["end"]]


def apply_leading(row, sequence_col: str = "sequence"):
    if row["orientation"] == "forward":
        result = row[sequence_col][row["start"] - 1]
    else:
        result = reverse_complement(row[sequence_col][row["end"]])
    # print(row["orientation"], row["sequence"][row["start"] - 1:row["end"] + 1], result)
    return result

def apply_trailing(row, sequence_col: str = "sequence"):
    if row["orientation"] == "forward":
        result = row[sequence_col][row["end"]]
    else:
        result = reverse_complement(row[sequence_col][row["start"] - 1])
    # print(row["orientation"], row["sequence"][row["start"] - 1:row["end"] + 1], result)
    return result


def calculate_combined_frequencies(leading: pd.Series, trailing: pd.Series) -> pd.Series:
    # Calculate expected frequencies assuming independence
    expected_counts = {}
    total_n = leading.sum()  # Total number of samples

    relative_frequencies_leading = leading / total_n
    relative_frequencies_trailing = trailing / total_n
    
    # Generate all possible combinations
    for leading in relative_frequencies_leading.index:
        for trailing in relative_frequencies_trailing.index:
            combination = leading + trailing
            # Expected frequency = P(leading) * P(trailing) * total_sample_size
            expected_freq = relative_frequencies_leading[leading] * relative_frequencies_trailing[trailing] * total_n
            expected_counts[combination] = expected_freq
    
    # Convert to Series for easier handling
    expected_counts_series = pd.Series(expected_counts).sort_index()
    return expected_counts_series


def flank_combination_chi_squared(data: pd.DataFrame):
    observed_frequencies = data["both_flanks"].value_counts()
    observed_frequencies = observed_frequencies.sort_index()
    single_frequencies_leading = data["leading_flank"].value_counts().sort_index()
    single_frequencies_trailing = data["trailing_flank"].value_counts().sort_index()
    expected_frequencies = calculate_combined_frequencies(leading=single_frequencies_leading, trailing=single_frequencies_trailing)

    # print("Single frequencies leading flank:")
    # print(single_frequencies_leading)
    # print("Single frequencies trailing flank:")
    # print(single_frequencies_trailing)
    # print("Contingency table (2-flank combinations):")
    # print(observed_frequencies)
    # print("Expected frequencies (assuming independence):")
    # print(expected_frequencies)
    # print(expected_frequencies.values)
    print("observed_frequencies")
    print(observed_frequencies)
    print("Expected frequencies:")
    print(expected_frequencies)
    print("differences:")
    print((observed_frequencies - expected_frequencies).sort_values(ascending=False))
    chi2, p = scipy.stats.chisquare(f_obs=observed_frequencies.values, f_exp=expected_frequencies.values)
    print(f"Chi-squared test for independence of leading and trailing flanks: chi2={chi2}, p={p}")
    # print(expected_frequencies.sum(), observed_frequencies.sum())


def flank_combinations(data: pd.DataFrame):
    data = data.copy()
    data["leading_flank"] = data.apply(apply_leading, axis=1)
    data["trailing_flank"] = data.apply(apply_trailing, axis=1)
    data["both_flanks"] = data["leading_flank"] + data["trailing_flank"]
    combination_groups = data.groupby("both_flanks")["enrichment"].apply(list)
    combination_f_stat, combination_p_val = scipy.stats.f_oneway(*combination_groups)
    print(f"Combination flank ANOVA: F={combination_f_stat}, p={combination_p_val}")
    combination_means = combination_groups.apply(np.mean)
    print(combination_groups.apply(len))
    print(combination_means)
    flank_combination_chi_squared(data)


def leading_trailing_anova_enrichment(data: pd.DataFrame):
    data = data.copy()
    #add leading and trailing flank
    data["leading_flank"] = data.apply(apply_leading, axis=1)
    data["trailing_flank"] = data.apply(apply_trailing, axis=1)
    #split data frame base on leading flank
    leading_groups = data.groupby("leading_flank")["enrichment"].apply(list)
    trailing_groups = data.groupby("trailing_flank")["enrichment"].apply(list)
    #perform anova
    leading_f_stat, leading_p_val = scipy.stats.f_oneway(*leading_groups)
    trailing_f_stat, trailing_p_val = scipy.stats.f_oneway(*trailing_groups)
    print(f"Leading flank ANOVA: F={leading_f_stat}, p={leading_p_val}")
    print(f"Trailing flank ANOVA: F={trailing_f_stat}, p={trailing_p_val}")
    # calculate group averages
    leading_means = leading_groups.apply(np.mean)
    trailing_means = trailing_groups.apply(np.mean)
    # print group lengths and means
    print("Leading flank group sizes:")
    print(leading_groups.apply(len))
    print("Trailing flank group sizes:")
    print(trailing_groups.apply(len))
    print("Combination flank group sizes:")
    print("Leading flank means:")
    print(leading_means)
    print("Trailing flank means:")
    print(trailing_means)
    print("Combination flank means:")


def tf_linear_model(data: pd.DataFrame):
    data = data.copy()
    data["leading_flank"] = data.apply(apply_leading, axis=1)
    data["trailing_flank"] = data.apply(apply_trailing, axis=1)
    data["leading_flank_reference"] = data.apply(lambda row: apply_leading(row, sequence_col="reference"), axis=1)
    data["trailing_flank_reference"] = data.apply(lambda row: apply_trailing(row, sequence_col="reference"), axis=1)
    data["leading_flank_changed"] = data["leading_flank"] != data["leading_flank_reference"]
    tf_columns = list(data.columns)[:46]
    prediction_data = data[tf_columns]
    prediction_data = prediction_data.values
    enrichment = data["enrichment"].values
    model = linear_model.LinearRegression()
    model.fit(prediction_data, enrichment)
    coefficients = model.coef_
    intercept = model.intercept_
    fit = model.score(prediction_data, enrichment)
    print(f"Linear model fit: R^2={fit}, intercept={intercept}")
    coef_df = pd.DataFrame({"TF": tf_columns, "coefficient": coefficients})
    coef_df = coef_df.sort_values(by="coefficient", ascending=False)
    print(coef_df)

def starrseq_analysis_deep():
    wrky_data, bhlh_data = add_starrseq_results()
    wrky_data["binding"] = wrky_data.index.str.contains("non_binding") == False
    bhlh_data["binding"] = bhlh_data.index.str.contains("non_binding") == False
    
    print("\n--- WRKY ---")
    binding_vs_non_binding(wrky_data, name="WRKY")
    leading_trailing_anova_enrichment(wrky_data)
    flank_combinations(wrky_data)
    tf_linear_model(wrky_data)

    print("\n--- bHLH ---")
    binding_vs_non_binding(bhlh_data, name="bHLH")
    leading_trailing_anova_enrichment(bhlh_data)
    flank_combinations(bhlh_data)
    tf_linear_model(bhlh_data)

    print("\n--- Combined ---")
    tf_linear_model(pd.concat([wrky_data, bhlh_data]))


if __name__ == "__main__":
    # analyze(exclude_references=False)

    # visualize_differences()
    # analyze_differences_permutation()
    # simplest_stats()
    # visualize_simple()
    # visualize_simple_2()
    # starrseq_sequences_for_deepCIS_predictions()
    # make_deepCIS_predictions()
    # find_motifs()
    # add_starrseq_results()
    starrseq_analysis_deep()
    # visualize_simple_2()

# TODO: do changes in flanking sites influence which TFs bind?
#     TODO: More general: how do ALL positions influence which TFs bind?
# TODO: How do mutations in all positions influence enrichment?
#     TODO: which mutations in core are tolerated (exclude mutations that regenerate clean motif)
# TODO: does a fixed mutation in a flanking site influence the distribution of other positions --> maybe forces other flanking site, or something in the core?
