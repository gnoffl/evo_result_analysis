"""Exact inclusion probabilities under successive sampling, k=2, 4 positions."""
from itertools import permutations
import numpy as np

weights = np.array([0.4, 0.2, 0.2, 0.2])
k = 2
inclusion = np.zeros(4)
total_probability = 0.0
for order in permutations(range(4), k):
    remaining = weights.copy()
    probability = 1.0
    for item in order:
        probability *= remaining[item] / remaining.sum()
        remaining[item] = 0.0
    total_probability += probability
    for item in order:
        inclusion[item] += probability

print(f"sum over ordered draws = {total_probability:.6f}  (must be 1)")
print(f"sum of inclusion probs = {inclusion.sum():.6f}  (must be k = {k})")
for i in range(4):
    print(f"  item {i}: weight={weights[i]:.2f}  target k*p={k*weights[i]:.4f}"
          f"  exact pi={inclusion[i]:.4f}  ratio={inclusion[i]/(k*weights[i]):.4f}")

# Monte Carlo cross-check against numpy's own replace=False sampler.
rng = np.random.default_rng(0)
hits = np.zeros(4)
draws = 400_000
for _ in range(draws):
    hits[rng.choice(4, size=k, replace=False, p=weights)] += 1
print("numpy choice(replace=False) empirical:", np.round(hits / draws, 4))
