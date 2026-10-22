"""The docstring's claim, tested the most direct way possible.

Draw many baselines, pool every drawn position into one big table, recompute
the empirical PMF. If blocking "preserves the empirical positional PMF", the
recovered PMF must equal the input PMF.
"""
import json
import numpy as np
import pandas as pd

# --- toy pool: one heavy position, three light ---
weights = np.array([0.4, 0.2, 0.2, 0.2])
k = 2
rng = np.random.default_rng(0)
replicates = 500_000
counts = np.zeros(4)
for _ in range(replicates):
    counts[rng.choice(4, size=k, replace=False, p=weights)] += 1

recovered = counts / counts.sum()
print("TOY POOL (k=2)")
print(f"  input PMF     : {np.round(weights, 4)}")
print(f"  recovered PMF : {np.round(recovered, 4)}")
print(f"  mean draws/replicate = {counts.sum() / replicates:.3f}  (= k, as it must be)")

# --- real ara pool ---
path = ("src/workflows/mutation_distribution_analysis/mutation_pools/"
        "ara_msr_max_single_gen1999_mutation_pool.json")
data = json.load(open(path))
mutations = pd.DataFrame(data["mutations"])
gene_stats = pd.DataFrame(data["gene_stats"])
unique_positions, pool_counts = np.unique(
    mutations["position"].to_numpy(), return_counts=True
)
input_pmf = pool_counts / pool_counts.sum()
draw_sizes = gene_stats["n_mutations"].to_numpy()

rng = np.random.default_rng(1)
hit_counts = np.zeros(unique_positions.size)
for draw_size in draw_sizes:
    for _ in range(40):
        drawn = rng.choice(
            unique_positions, size=int(draw_size), replace=False, p=input_pmf
        )
        hit_counts[np.searchsorted(unique_positions, drawn)] += 1
recovered_pmf = hit_counts / hit_counts.sum()

print("\nREAL ara POOL (k ~ 88, 40 replicates per gene)")
order = np.argsort(input_pmf)
for label, idx in [("5 heaviest", order[-5:]), ("5 lightest", order[:5])]:
    print(f"  {label} positions:")
    for i in idx[::-1]:
        print(f"    pos {unique_positions[i]:5d}  input={input_pmf[i]:.6f}"
              f"  recovered={recovered_pmf[i]:.6f}"
              f"  ratio={recovered_pmf[i] / input_pmf[i]:.3f}")
