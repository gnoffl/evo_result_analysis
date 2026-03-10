from typing import List
from unittest.mock import patch, mock_open
from matplotlib import pyplot as plt
import numpy as np
from pyfaidx import Fasta


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


if __name__ == '__main__':
    # concept_mutation_conservation()
    extract_ubi_gene()