"""Feasibility and cost of Sampford sampling on the real ara pool."""
import json
import time

import numpy as np
import pandas as pd

path = ("src/workflows/mutation_distribution_analysis/mutation_pools/"
        "ara_msr_max_single_gen1999_mutation_pool.json")
data = json.load(open(path))
mutations = pd.DataFrame(data["mutations"])
gene_stats = pd.DataFrame(data["gene_stats"])
unique_positions, counts = np.unique(
    mutations["position"].to_numpy(), return_counts=True
)
pool_pmf = counts / counts.sum()
k = int(gene_stats["n_mutations"].median())

inclusion_target = k * pool_pmf
print(f"n positions = {unique_positions.size}, k = {k}")
print(f"max k*p = {inclusion_target.max():.4f}  -> Sampford feasible: "
      f"{inclusion_target.max() < 1.0}")

tilted = pool_pmf / (1.0 - inclusion_target)
tilted /= tilted.sum()
rng = np.random.default_rng(0)

attempts = 0
accepted = 0
hits = np.zeros(unique_positions.size)
start = time.time()
target_samples = 20_000
while accepted < target_samples:
    attempts += 1
    first = rng.choice(unique_positions.size, p=pool_pmf)
    rest = rng.choice(unique_positions.size, size=k - 1, replace=True, p=tilted)
    candidate = np.concatenate(([first], rest))
    if np.unique(candidate).size == k:
        hits[candidate] += 1
        accepted += 1
elapsed = time.time() - start

print(f"acceptance rate = {accepted / attempts:.3f} "
      f"({attempts / accepted:.1f} attempts per sample)")
print(f"{accepted} samples in {elapsed:.1f}s "
      f"({1000 * elapsed / accepted:.2f} ms per sample)")

empirical = hits / accepted
order = np.argsort(pool_pmf)[-5:]
print("\n5 heaviest positions: target pi vs Sampford empirical")
for i in order[::-1]:
    print(f"  pos {unique_positions[i]:5d}  target={inclusion_target[i]:.4f}"
          f"  sampford={empirical[i]:.4f}  ratio={empirical[i]/inclusion_target[i]:.3f}")
