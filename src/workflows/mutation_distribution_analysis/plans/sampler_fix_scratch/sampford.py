"""Sampford's method: exact pi = k*p, no numerical calibration needed."""
from itertools import permutations

import numpy as np


def exact_inclusion_successive(weights: np.ndarray, k: int) -> np.ndarray:
    """Exact inclusion probabilities of successive sampling, by enumeration."""
    inclusion = np.zeros(weights.size)
    for order in permutations(range(weights.size), k):
        remaining = weights.copy()
        probability = 1.0
        for item in order:
            probability *= remaining[item] / remaining.sum()
            remaining[item] = 0.0
        for item in order:
            inclusion[item] += probability
    return inclusion


def sampford_sample(
    target_pmf: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw k distinct indices with inclusion probabilities exactly k*target_pmf.

    Sampford (1967): draw the first unit with probability ``target_pmf``, the
    remaining k-1 *with replacement* with probability proportional to
    ``target_pmf / (1 - k * target_pmf)``, and accept only if all k are
    distinct. Requires ``k * target_pmf < 1`` everywhere.
    """
    inclusion_target = k * target_pmf
    if np.any(inclusion_target >= 1.0):
        raise ValueError("Sampford requires k * p < 1 for every unit.")
    tilted = target_pmf / (1.0 - inclusion_target)
    tilted = tilted / tilted.sum()
    n_indices = target_pmf.size
    while True:
        first = rng.choice(n_indices, p=target_pmf)
        rest = rng.choice(n_indices, size=k - 1, replace=True, p=tilted)
        candidate = np.concatenate(([first], rest))
        if np.unique(candidate).size == k:
            return candidate


pool_pmf = np.array([0.4, 0.2, 0.2, 0.2])
k = 2
rng = np.random.default_rng(0)
draws = 500_000
hits = np.zeros(4)
attempts_used = 0
for _ in range(draws):
    hits[sampford_sample(pool_pmf, k, rng)] += 1

print(f"target pi = k*p          : {np.round(k * pool_pmf, 4)}")
print(f"successive (current code): {np.round(exact_inclusion_successive(pool_pmf, k), 4)}")
print(f"Sampford, {draws} draws : {np.round(hits / draws, 4)}")
