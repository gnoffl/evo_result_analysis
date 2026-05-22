import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model #type:ignore
from pyfaidx import Fasta
from deepCRE.utils import one_hot_encode

ZEA_MSR_PATH = "/home/gernot/Code/PhD_Code/Evolution/models/MSR_atha_bvul_bole_csat_cqui_ccum_dcar_gmax_osat_slyc_sbic_zmay_M0X0.75dP7K25f_A_thaliana_msr_train_models_250705_070641.h5"
ARA_MSR_PATH = "/home/gernot/Code/PhD_Code/Evolution/models/MSR_atha_bvul_bole_csat_cqui_ccum_dcar_gmax_osat_slyc_sbic_zmay_M0X0.75dP7K25f_Z_mays_msr_train_models_250705_070641.h5"
ZEA_MUTATED_PATH = "src/workflows/mutation_distribution_analysis/mutated_sequences/zea_random_mutated.fa"
ARA_MUTATED_PATH = "src/workflows/mutation_distribution_analysis/mutated_sequences/ara_random_mutated.fa"


def evaluate_sequences(model_path: str, fasta_path: str, output_path: str) -> None:
    sequences = Fasta(fasta_path)
    model = load_model(model_path)
    sequences_names = list(sequences.keys())
    sequences_encoded = np.array([one_hot_encode(str(sequences[name][:])) for name in sequences_names])
    predictions = model.predict(sequences_encoded)
    print(predictions.shape)
    results = pd.DataFrame({"sequence_name": sequences_names, "prediction": predictions.flatten()})
    results.to_csv(output_path, index=False)



if __name__ == "__main__":
    evaluate_sequences(ZEA_MSR_PATH, ZEA_MUTATED_PATH, "src/workflows/mutation_distribution_analysis/mutated_sequences/zea_random_mutated_predictions.csv")
    evaluate_sequences(ARA_MSR_PATH, ARA_MUTATED_PATH, "src/workflows/mutation_distribution_analysis/mutated_sequences/ara_random_mutated_predictions.csv")
