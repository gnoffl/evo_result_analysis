"""Where Sampford's 2.1% acceptance rate comes from."""
import json

import numpy as np
import pandas as pd

path = ("src/workflows/mutation_distribution_analysis/mutation_pools/"
        "ara_msr_max_single_gen1999_mutation_pool.json")
data = json.load(open(path))
mutations = pd.DataFrame(data["mutations"])
gene_stats = pd.DataFrame(data["gene_stats"])
_, counts = np.unique(mutations["position"].to_numpy(), return_counts=True)
pool_pmf = counts / counts.sum()
n_positions = pool_pmf.size
k = int(gene_stats["n_mutations"].median())

tilted = pool_pmf / (1.0 - k * pool_pmf)
tilted /= tilted.sum()

# Poisson approximation to "all k draws distinct": exp(-expected collisions).
pairs_within_rest = k - 1
collisions_within_rest = (
    pairs_within_rest * (pairs_within_rest - 1) / 2 * np.sum(tilted**2)
)
collisions_with_first = pairs_within_rest * np.sum(pool_pmf * tilted)
expected_collisions = collisions_within_rest + collisions_with_first

uniform_pmf = np.full(n_positions, 1.0 / n_positions)
uniform_collisions = k * (k - 1) / 2 * np.sum(uniform_pmf**2)

print(f"n = {n_positions}, k = {k}")
print(f"effective pool size 1/sum(q^2) = {1 / np.sum(tilted**2):.0f} "
      f"(vs {n_positions} actual)")
print(f"\nexpected colliding pairs (real weights)  = {expected_collisions:.2f}"
      f"  -> acceptance ~ {np.exp(-expected_collisions):.4f}")
print(f"expected colliding pairs (uniform weights) = {uniform_collisions:.2f}"
      f"  -> acceptance ~ {np.exp(-uniform_collisions):.4f}")
print("measured acceptance rate                   = 0.021")

print("\nhow acceptance scales with k (real weights):")
for k_test in [5, 10, 20, 40, 60, 88, 120]:
    if (k_test * pool_pmf).max() >= 1.0:
        print(f"  k={k_test:4d}: infeasible (max k*p >= 1)")
        continue
    q = pool_pmf / (1.0 - k_test * pool_pmf)
    q /= q.sum()
    lam = ((k_test - 1) * (k_test - 2) / 2 * np.sum(q**2)
           + (k_test - 1) * np.sum(pool_pmf * q))
    print(f"  k={k_test:4d}: expected collisions={lam:5.2f}  "
          f"acceptance~{np.exp(-lam):.4f}  attempts~{np.exp(lam):6.1f}")
