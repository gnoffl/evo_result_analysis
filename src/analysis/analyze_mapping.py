import os
import json
import argparse
import random
import string
from typing import Dict, Optional, Tuple
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tqdm


def get_epm_tfbs_mapping_old(input_path: str) -> pd.DataFrame:
    data = pd.read_csv(input_path, header=0)
    rel_cols = data[["epm", "tf_name"]]
    rel_cols = rel_cols.drop_duplicates()
    return rel_cols


def get_epm_tfbs_mapping_new(input_path: str) -> pd.DataFrame:
    print("!!!!THIS IS NOT PROPERLY IMPLEMENTED!!!! IMPLEMENT get_epm_tfbs_mapping_new!!!!")
    data = pd.read_csv(input_path, header=0)
    add_epm(data)
    rel_cols = data[["epm"]]
    rel_cols = rel_cols.drop_duplicates()
    rel_cols["tf_name"] = ["not_implemented" for _ in range(len(rel_cols))]
    return rel_cols


def add_epm(data: pd.DataFrame) -> None:
    if "epm" not in data.columns:
        data["epm"] = data["motif"].str.split(r"[RF]_(\d+)$", regex=True, expand=True)[0]


def get_epm_count_table(input_path: str, new: bool = True) -> pd.DataFrame:
    out_path = os.path.splitext(input_path)[0] + "_epm_counts.csv"
    # if os.path.exists(out_path):
    #     loaded_data = pd.read_csv(out_path, header=0)
    #     return loaded_data
    data = pd.read_csv(input_path, header=0)
    add_epm(data)
    if not new:
        data = data[data["type"] == "gene"]
    data_filtered = data[data["motif"].str.contains(r"p(\d)+m(\d)+F_(\d)+$", regex=True)]
    id_column = "sequence_id" if new else "loc"
    epm_counts = data_filtered[[id_column, "epm"]].groupby(id_column).value_counts()
    # create table with loc as rows, epm as columns, and counts as values
    epm_counts = epm_counts.unstack(fill_value=0)
    epm_counts = epm_counts.reset_index()
    epm_counts.columns.name = None  # remove the name of the columns index
    epm_counts = epm_counts.rename(columns={id_column: "sequence"})
    epm_counts.to_csv(out_path, index=False)
    return epm_counts


def get_min_max_dfs(input_df: pd.DataFrame, remembered_issues: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    add_original_sequence_and_number_mutations(input_df)
    max_mutations = input_df.groupby("original_sequence").agg({
        "number_mutations": "max"
    }).reset_index().set_index("original_sequence").rename(columns={"number_mutations": "max_number_mutations"})
    input_df = input_df.join(max_mutations, on="original_sequence", how="left")
    max_mutation_rows = input_df[input_df["number_mutations"] == input_df["max_number_mutations"]].drop_duplicates(subset=["original_sequence"])
    min_mutation_rows = input_df[input_df["number_mutations"] == 0]
    max_mutation_rows = max_mutation_rows.set_index("original_sequence")
    min_mutation_rows = min_mutation_rows.set_index("original_sequence")
    if not remembered_issues:
        raise ValueError("Remember that for some reason some sequences dont show up with 0 mutations, so they are missing from min_mutation_rows. When merging the data, remember to fillna those values to 0.")
    return min_mutation_rows, max_mutation_rows


def get_difference_number_epm(input_path: str, output_folder: str = ".", new: bool = True):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "initial_vs_final_per_original_sequence.csv")
    epm_counts = get_epm_count_table(input_path, new=new)
    # get the number of EPMs for each sequence
    epm_counts["number_epm"] = epm_counts.drop(columns=["sequence"]).sum(axis=1)
    min_mutation_rows, max_mutation_rows = get_min_max_dfs(epm_counts, remembered_issues=True)
    # max_mutation_rows = max_mutation_rows.set_index("original_sequence")
    # min_mutation_rows = min_mutation_rows.set_index("original_sequence")
    min_mutation_rows = min_mutation_rows[["number_epm"]].rename(columns={"number_epm": "number_epm_min_mutations"})
    max_mutation_rows = max_mutation_rows.rename(columns={"number_epm": "number_epm_max_mutations"})
    # missing_zeros = set(max_mutation_rows.index) - set(min_mutation_rows.index)
    # print(f"Missing zeros for {len(missing_zeros)} sequences: {missing_zeros}")
    merged = min_mutation_rows.join(max_mutation_rows, how="outer")
    merged["number_epm_min_mutations"].fillna(0, inplace=True)
    merged["number_epm_difference"] = merged["number_epm_max_mutations"] - merged["number_epm_min_mutations"]
    merged["log2_ratio"] = np.log2(merged["number_epm_max_mutations"] / merged["number_epm_min_mutations"])
    merged = merged.sort_values(by="log2_ratio", ascending=False)
    # add sum row
    merged.loc["sum"] = merged.sum(numeric_only=True)
    merged.at["sum", "log2_ratio"] = np.log2(merged.at["sum", "number_epm_max_mutations"] / merged.at["sum", "number_epm_min_mutations"])
    merged = merged.reset_index()
    merged = merged.rename(columns={"index": "original_sequence"})
    merged = merged.set_index("original_sequence")
    merged = merged[["number_epm_min_mutations", "number_epm_max_mutations", "number_epm_difference", "log2_ratio"]]
    # print the result
    merged.to_csv(output_path, index=True)


def statistics_epm_before_after(epm_counts: pd.DataFrame) -> Dict[str, float]:
    epm_cols = [col for col in epm_counts.columns if col.startswith("epm_")]
    min_mutation_rows, max_mutation_rows = get_min_max_dfs(epm_counts, remembered_issues=True)
    merged = min_mutation_rows.join(max_mutation_rows, how="outer", lsuffix="_min", rsuffix="_max")
    min_cols = [col for col in merged.columns if col.endswith("_min")]
    merged[min_cols] = merged[min_cols].fillna(0)
    significance = {}
    for col in epm_cols:
        min_col = col + "_min"
        max_col = col + "_max"
        #test normal distribution of columns
        min_normal = scipy.stats.normaltest(merged[min_col])
        max_normal = scipy.stats.normaltest(merged[max_col])
        if min_normal.pvalue > 0.05 and max_normal.pvalue > 0.05:
            difference = scipy.stats.ttest_rel(merged[min_col], merged[max_col])
        else:
            difference = scipy.stats.wilcoxon(merged[min_col], merged[max_col])
        significance[col] = difference.pvalue
    return significance


def add_tf_names(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Add TF names to the EPM counts DataFrame based on the mapping DataFrame.
    """
    df = df.merge(mapping, left_index=True, right_on="epm", how="left")
    df = df.reset_index()
    df = df.set_index("epm")
    return df


def compare_initial_and_final_distribution(input_path: str, mapping: pd.DataFrame, output_folder: str = ".", new: bool = True):
    """
    Compare the initial and final distributions of EPMs.
    """
    epm_counts = get_epm_count_table(input_path, new=new)
    significance = statistics_epm_before_after(epm_counts.copy())
    firsts, lasts = [], []
    # sequences are named as "<gene>_mutations_<number>"
    # for each gene get the first and last sequence
    genes = epm_counts["sequence"].str.split("_mutations_", expand=True)[0].unique()
    for gene in genes:
        first_seq = epm_counts[epm_counts["sequence"].str.startswith(gene)].iloc[0]
        last_seq = epm_counts[epm_counts["sequence"].str.startswith(gene)].iloc[-1]
        firsts.append(first_seq)
        lasts.append(last_seq)
    firsts = pd.DataFrame(firsts)
    lasts = pd.DataFrame(lasts)
    summed_firsts = firsts.drop(columns=["sequence"]).sum()
    summed_lasts = lasts.drop(columns=["sequence"]).sum()
    difference = summed_lasts - summed_firsts
    log_ratio = np.log2(summed_lasts / summed_firsts)
    df = pd.DataFrame({
        "initial": summed_firsts,
        "final": summed_lasts,
        "difference": difference,
        "log_ratio": log_ratio
    })
    df = df.reset_index()
    df = df.rename(columns={"index": "epm"})
    df["significance"] = df["epm"].map(significance)
    df = df.set_index("epm")
    df = add_tf_names(df, mapping)
    df = df.sort_values(by="log_ratio", ascending=False)
    # add sum row
    df.loc["sum"] = df.sum(numeric_only=True)
    # recalculate log ratio for sum row
    df.at["sum", "log_ratio"] = np.log2(df.at["sum", "final"] / df.at["sum", "initial"])
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "initial_vs_final_per_epm_counts.csv")
    df.to_csv(output_path, index=True)


def add_fitness(df: pd.DataFrame, data_path: str) -> None:
    """
    Add fitness scores to the EPM counts DataFrame.
    """
    df["fitness"] = 0.0
    df["diff_fitness"] = np.nan
    df["diff_fitness_normalized"] = np.nan
    current_gene = ""
    current_fitness_list = []
    previous_fitness = 0.0
    previous_mutations = 0
    for i, row in tqdm.tqdm(df.iterrows()):
        run, gene = row["sequence"].split("__")
        gene = gene.split("_mutations_")[0]
        if gene != current_gene:
            current_gene = gene
            pareto_front_path = os.path.join(data_path, run, gene, "saved_populations", f"{gene}_pareto_front.json")
            try:
                with open(pareto_front_path, "r") as f:
                    current_fitness_list = json.load(f)
            except FileNotFoundError:
                alternative_pareto_front_path = os.path.join(data_path, run, gene, "saved_populations", f"pareto_front.json")
                try:
                    with open(alternative_pareto_front_path, "r") as f:
                        current_fitness_list = json.load(f)
                except FileNotFoundError:
                    print(f"Could not find pareto front file for gene {gene} in run {run}. Setting fitness to NaN.")
                    current_fitness_list = []
            previous_fitness = 0.0
            previous_mutations = 0
        if not current_fitness_list:
            df.loc[i, "fitness"] = np.nan
            df.loc[i, "diff_fitness"] = np.nan
            df.loc[i, "diff_fitness_normalized"] = np.nan
            continue
                

        fitness = [fitness for seq, fitness, n_mutations in current_fitness_list if n_mutations == row["number_mutations"]][0]
        diff_mutations = row["number_mutations"] - previous_mutations
        df.loc[i, "fitness"] = fitness
        df.loc[i, "diff_fitness"] = fitness - previous_fitness if row["number_mutations"] > 0 else np.nan
        df.loc[i, "diff_fitness_normalized"] = df.loc[i, "diff_fitness"] / diff_mutations if row["number_mutations"] > 0 else np.nan
        previous_fitness = fitness
        previous_mutations = row["number_mutations"]


def find_diff_tf(epm_cols, diff_epms) -> Tuple[Optional[str], int]:
    difference_counter = 0
    tf = ""
    for col in epm_cols:
        if diff_epms[col] != 0:
            difference_counter += abs(diff_epms[col])
            tf = col
    if difference_counter == 1:
        return tf, difference_counter
    return None, difference_counter


def add_diff_tf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add difference in TFs to the EPM counts DataFrame.
    """
    df["diff_tf"] = pd.NA
    epm_cols = [col for col in df.columns if col.startswith("epm_")]
    previous_epms = df.iloc[0]
    previous_epms = previous_epms[epm_cols]
    for i in tqdm.tqdm(range(1, len(df))):
        current_row = df.iloc[i]
        if current_row["number_mutations"] == 0:
            previous_epms = current_row[epm_cols]
            continue
        current_epms = current_row[epm_cols]
        diff_epms = current_epms - previous_epms
        diff_tf, counter = find_diff_tf(epm_cols, diff_epms)
        if diff_tf is not None:
            df.loc[i, "diff_tf"] = diff_tf
        previous_epms = current_epms
    return df


def add_original_sequence_and_number_mutations(df: pd.DataFrame) -> None:
    df["original_sequence"] = df["sequence"].str.split("_mutations_", expand=True)[0]
    df["number_mutations"] = df["sequence"].str.split("_mutations_", expand=True)[1].str.split(":", expand=True)[0].str.split("_", expand=True)[0].astype(int)


def analyze_mutations_effect(input_path: str, output_folder: str, mapping: pd.DataFrame, data_path: str, epm_of_interest: str = "epm_ara_msr_max_corrected_p1m10", new: bool = True) -> pd.DataFrame:
    """
    Analyze the effect of mutations on EPMs.
    """
    output_path = os.path.join(output_folder, f"effect_per_mutation_epm.csv")
    epm_counts = get_epm_count_table(input_path, new=new)
    add_original_sequence_and_number_mutations(epm_counts)
    epm_counts = epm_counts.drop_duplicates(subset=["original_sequence", "number_mutations"])
    epm_counts = epm_counts.sort_values(by=["original_sequence", "number_mutations"])
    add_fitness(epm_counts, data_path=data_path)
    add_diff_tf(epm_counts)
    epm_counts = epm_counts.merge(mapping, left_on="diff_tf", right_on="epm", how="left")
    epm_counts = epm_counts.drop(columns=["diff_tf"])
    print(f"average mutation effect: {epm_counts['diff_fitness_normalized'].mean()}")

    epms, effects, pvals = [], [], []
    for epm_of_interest in tqdm.tqdm(epm_counts["epm"].unique()):
        of_interest = epm_counts[epm_counts["epm"] == epm_of_interest]
        other = epm_counts[epm_counts["epm"] != epm_of_interest]

        interest = of_interest["diff_fitness_normalized"]
        other = other["diff_fitness_normalized"]
        try:
            #see if samples are normal distributed
            interest_normal = scipy.stats.normaltest(interest)
            other_normal = scipy.stats.normaltest(other)
            if interest_normal.pvalue < 0.05 and other_normal.pvalue < 0.05:
                # use t-test
                print(f"performing t-test because both distributions are normal")
                t_stat, p_value = scipy.stats.ttest_ind(interest, other)
            else:
                # use Mann-Whitney U test
                print(f"performing Mann-Whitney U test because at least one distribution is not normal")
                u_stat, p_value = scipy.stats.mannwhitneyu(interest, other, alternative="two-sided", nan_policy='omit')
        except Exception:
            p_value = np.nan
        average_interest = interest.mean()
        average_other = other.mean()
        print(f"Average {epm_of_interest} diff_fitness_normalized: {average_interest}")
        print(f"Average other TFs diff_fitness_normalized: {average_other}")
        print(f"Statistical test result: p-value={p_value}")
        effects.append(average_interest)
        pvals.append(p_value)
        epms.append(epm_of_interest)

    epm_effect_df = pd.DataFrame({
        "epm": epms,
        "avg_effect": effects,
        "pval": pvals
    })

    #merge ratio and p_val onto epm_counts
    os.makedirs(output_folder, exist_ok=True)
    epm_effect_df.to_csv(output_path, index=False)

    return epm_counts


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze EPM mapping data")
    parser.add_argument("--input_old", "-io", type=str, required=False,
                        help="Path to input CSV file with EPM mapping data using the old moca blue pipeline", default="")
    parser.add_argument("--input", "-i", type=str, required=False,
                        help="Path to input CSV file with EPM mapping data using the new moca blue pipeline", default="")
    parser.add_argument("--output_folder", "-o", type=str, default=".", required=False,
                        help="Path to output folder for results")
    parser.add_argument("--data_path", "-d", type=str, default="data", required=False,
                        help="Path to data folder containing run directories")
    parser.add_argument("--difference_epm", action="store_true",
                        help="Calculate difference in number of EPMs between min and max mutations")
    parser.add_argument("--compare_distribution", action="store_true",
                        help="Compare initial and final EPM distributions")
    parser.add_argument("--mutations_effect", "-m", action="store_true",
                        help="Analyze the effect of mutations on EPMs")
    parser.add_argument("--all", action="store_true",
                        help="Run all analyses")
    args = parser.parse_args()
    if not args.input_old and not args.input:
        raise ValueError("At least one of --input_old or --input must be provided.")
    return args


def main():
    args = parse_args()

    if not args.compare_distribution and not args.difference_epm and not args.all and not args.mutations_effect:
        print("No analysis selected. Use --compare-distribution, --difference-epm, --mutations_effect, or --all.")
        return
    
    if args.compare_distribution or args.all:
        if args.input:
            mapping = get_epm_tfbs_mapping_new(args.input)
            print("Comparing initial and final distributions...")
            compare_initial_and_final_distribution(args.input, mapping=mapping, output_folder=args.output_folder, new=True)
        else:
            mapping = get_epm_tfbs_mapping_old(args.input_old)
            print("Comparing initial and final distributions...")
            compare_initial_and_final_distribution(args.input_old, mapping=mapping, output_folder=args.output_folder, new=False)
    
    if args.difference_epm or args.all:
        print("Calculating difference in number of EPMs...")
        if args.input:
            get_difference_number_epm(args.input, output_folder=args.output_folder, new=True)
        else:
            get_difference_number_epm(args.input_old, output_folder=args.output_folder, new=False)
        print("success")

    if args.mutations_effect or args.all:
        if args.input:
            mapping = get_epm_tfbs_mapping_new(args.input)
            print("Analyzing the effect of mutations on EPMs...")
            analyze_mutations_effect(args.input, output_folder=args.output_folder, mapping=mapping, new=True, data_path=args.data_path)
        else:
            mapping = get_epm_tfbs_mapping_old(args.input_old)
            print("Analyzing the effect of mutations on EPMs...")
            analyze_mutations_effect(args.input_old, output_folder=args.output_folder, mapping=mapping, new=False, data_path=args.data_path)



if __name__ == "__main__":
    main()
