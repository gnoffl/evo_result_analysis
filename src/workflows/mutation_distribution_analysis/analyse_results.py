import os
import pandas as pd


ARA_RANDOM_MUTATIONS_DIR = "src/workflows/mutation_distribution_analysis/mutated_sequences/ara_random_mutated_predictions.csv"
ZEA_RANDOM_MUTATIONS_DIR = "src/workflows/mutation_distribution_analysis/mutated_sequences/zea_random_mutated_predictions.csv"


def average_predications_after_random_mutations():
    ara_random_mutations_df = pd.read_csv(ARA_RANDOM_MUTATIONS_DIR)
    zea_random_mutations_df = pd.read_csv(ZEA_RANDOM_MUTATIONS_DIR)

    print("Average predictions after random mutations:")
    avg_ara = ara_random_mutations_df['prediction'].mean()
    avg_zea = zea_random_mutations_df['prediction'].mean()
    print(f"Arabidopsis: {avg_ara}")
    print(f"Maize: {avg_zea}")
    with open("src/workflows/mutation_distribution_analysis/average_predictions_after_random_mutations.txt", "w") as f:
        f.write("Average predictions after random mutations:\n")
        f.write(f"Arabidopsis: {avg_ara}\n")
        f.write(f"Maize: {avg_zea}\n")

def main():
    average_predications_after_random_mutations()

if __name__ == "__main__":
    main()