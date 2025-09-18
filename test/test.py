from typing import List
from unittest.mock import patch, mock_open
from matplotlib import pyplot as plt
import numpy as np


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
    concept_mutation_conservation()