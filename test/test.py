import random
from typing import List
from unittest.mock import patch, mock_open
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pyfaidx import Fasta

from analysis.motives.deepcis_scanner import (
    TF_FAMILY_NAMES,
    load_deepcis_model,
)
from evolution.sequences import one_hot_encode

MODEL_PATH = "models/deepcis/deepCIS_model_chrom_1_model.h5"
SEQUENCE_LENGTH = 250
TRAILING_N = 50
N_SEQUENCES = 10
SEED = 0
PREDICTION_THRESHOLD = 0.5


def reverse_complement(seq: str) -> str:
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement.get(base, base) for base in reversed(seq))


def extract_ubi_gene():
    fasta = Fasta("/home/gernot/ARCitect/ARCs/genRE/assays/Gene_Data/dataset/genomes/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa")
    for record in fasta:
        print(record.name)
        if "5" in record.name:
            gene = record[84400792:84403652]
            gene = reverse_complement(str(gene))
            print(gene)
            break



def concept_mutation_conservation():
    real_numbers = [1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6, 7, 7, 8, 8, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 13,
                    13, 13, 13, 14, 15, 16, 16, 16, 17, 18, 19, 20, 21, 21, 21, 22, 22, 24, 24, 24, 24, 26, 28, 30, 31, 32,
                    33, 35, 36, 36, 40, 40, 42, ]
    real_numbers = np.array(sorted(real_numbers, reverse=True))
    ideal_curve = np.concatenate((np.arange(44, -1, -1), np.zeros(len(real_numbers) - 45)))
    indexes = np.arange(len(real_numbers))
    plt.clf()
    plt.plot(indexes, real_numbers, label='Found Distribution')
    plt.plot(indexes, ideal_curve, label='Ideal Curve')
    plt.legend()
    plt.savefig('mutation_conservation.png', bbox_inches='tight')
    print(real_numbers.sum())
    print(ideal_curve.sum())


def make_random_sequence(with_trailing_n: bool) -> str:
    """Build one random 250 bp DNA sequence.

    Args:
        with_trailing_n: If True, the last ``TRAILING_N`` bases are ``N`` and the
            preceding bases are random; if False, all bases are random.

    Returns:
        A DNA string of length ``SEQUENCE_LENGTH``.
    """
    random_length = SEQUENCE_LENGTH - TRAILING_N if with_trailing_n else SEQUENCE_LENGTH
    random_part = "".join(random.choice("ACGT") for _ in range(random_length))
    return random_part + "N" * (SEQUENCE_LENGTH - random_length)


def predict_sequences(model, sequences: List[str]) -> np.ndarray:
    """One-hot encode sequences and run them through deepCIS.

    Args:
        model: Loaded deepCIS Keras model.
        sequences: DNA strings, each of length ``SEQUENCE_LENGTH``.

    Returns:
        Array of shape ``(len(sequences), 46)`` with TF binding likelihoods.
    """
    encoded = np.stack([one_hot_encode(s) for s in sequences], axis=0)
    return np.asarray(model.predict(encoded, verbose=0), dtype=np.float32)


def predict_random_sequences() -> None:
    """Generate random sequences (with and without trailing N) and predict."""
    random.seed(SEED)
    sequences = [make_random_sequence(with_trailing_n=False) for _ in range(N_SEQUENCES)]
    sequences += [make_random_sequence(with_trailing_n=True) for _ in range(N_SEQUENCES)]
    kinds = ["random"] * N_SEQUENCES + ["random_with_50N"] * N_SEQUENCES

    model = load_deepcis_model(MODEL_PATH)
    predictions = predict_sequences(model, sequences)

    result = pd.DataFrame(predictions, columns=pd.Index(TF_FAMILY_NAMES))
    result.insert(0, "kind", np.array(kinds))

    is_predicted = result["ABI3VP1_tnt"] > PREDICTION_THRESHOLD
    counts = is_predicted.groupby(result["kind"]).sum()
    print(f"ABI3VP1_tnt predicted (> {PREDICTION_THRESHOLD}) per group, "
          f"out of {N_SEQUENCES} sequences each:")
    print(counts)


if __name__ == '__main__':
    # concept_mutation_conservation()
    # extract_ubi_gene()
    predict_random_sequences()