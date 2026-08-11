"""To hit pi = k*p, the *input* weights must be pre-distorted, not the renormalisation."""
from itertools import permutations

import numpy as np
from scipy.optimize import brentq


def exact_inclusion(selection_weights: np.ndarray, k: int) -> np.ndarray:
    """Exact inclusion probabilities of successive sampling, by enumeration."""
    n = selection_weights.size
    inclusion = np.zeros(n)
    for order in permutations(range(n), k):
        remaining = selection_weights.copy()
        probability = 1.0
        for item in order:
            probability *= remaining[item] / remaining.sum()
            remaining[item] = 0.0
        for item in order:
            inclusion[item] += probability
    return inclusion


pool_pmf = np.array([0.4, 0.2, 0.2, 0.2])
k = 2
target = k * pool_pmf

print("what the code does now (selection weights = pool PMF):")
print(f"  input weights : {np.round(pool_pmf, 4)}")
print(f"  resulting pi  : {np.round(exact_inclusion(pool_pmf, k), 4)}")
print(f"  target pi     : {np.round(target, 4)}")

# Solve for the heavy weight that makes pi_heavy hit its target.
# By symmetry: weights = (a, b, b, b) with a + 3b = 1.
def heavy_inclusion_error(heavy_weight: float) -> float:
    light_weight = (1.0 - heavy_weight) / 3.0
    weights = np.array([heavy_weight, light_weight, light_weight, light_weight])
    return exact_inclusion(weights, k)[0] - target[0]


solved_heavy = brentq(heavy_inclusion_error, 0.4, 0.999999)
solved_light = (1.0 - solved_heavy) / 3.0
solved_weights = np.array(
    [solved_heavy, solved_light, solved_light, solved_light]
)
print("\nweights that DO yield pi = k*p under the same sequential procedure:")
print(f"  input weights : {np.round(solved_weights, 4)}")
print(f"  resulting pi  : {np.round(exact_inclusion(solved_weights, k), 4)}")
print(f"\nheavy position's selection weight must be raised "
      f"{solved_heavy / pool_pmf[0]:.2f}x above its pool share.")
