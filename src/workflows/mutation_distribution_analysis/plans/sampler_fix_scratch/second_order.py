"""Do Sampford and pivotal give the SAME distance null? They share pi_i but
the distance statistic depends on joint (second-order) inclusion structure.
"""
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


def pivotal_sample(target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pivotal method with random pairing; O(n) duels, no rejection."""
    probabilities = target.astype(float).copy()
    undecided = list(np.flatnonzero((probabilities > 0) & (probabilities < 1)))
    while len(undecided) >= 2:
        a_slot = rng.integers(len(undecided))
        b_slot = rng.integers(len(undecided) - 1)
        if b_slot >= a_slot:
            b_slot += 1
        first, second = undecided[a_slot], undecided[b_slot]
        mass = probabilities[first] + probabilities[second]
        if mass < 1.0:
            if rng.random() < probabilities[first] / mass:
                probabilities[first], probabilities[second] = mass, 0.0
                dead_slot = b_slot
            else:
                probabilities[first], probabilities[second] = 0.0, mass
                dead_slot = a_slot
        else:
            if rng.random() < (1.0 - probabilities[second]) / (2.0 - mass):
                probabilities[first], probabilities[second] = 1.0, mass - 1.0
                dead_slot = a_slot
            else:
                probabilities[first], probabilities[second] = mass - 1.0, 1.0
                dead_slot = b_slot
        undecided[dead_slot] = undecided[-1]
        undecided.pop()
    return np.flatnonzero(probabilities > 0.5)


def sampford_sample(
    pmf: np.ndarray, tilted: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Sampford rejective sampler."""
    n = pmf.size
    while True:
        candidate = np.concatenate((
            [rng.choice(n, p=pmf)],
            rng.choice(n, size=k - 1, replace=True, p=tilted),
        ))
        if np.unique(candidate).size == k:
            return candidate


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
target = k * pool_pmf
tilted = pool_pmf / (1.0 - target)
tilted /= tilted.sum()

n_samples = 3000
results = {}
for name, sampler in [
    ("sampford", lambda rng: sampford_sample(pool_pmf, tilted, k, rng)),
    ("pivotal", lambda rng: pivotal_sample(target, rng)),
    ("successive (current)",
     lambda rng: rng.choice(pool_pmf.size, size=k, replace=False, p=pool_pmf)),
]:
    rng = np.random.default_rng(42)
    distances = []
    inclusion_hits = np.zeros(pool_pmf.size)
    start = time.time()
    for _ in range(n_samples):
        idx = sampler(rng)
        inclusion_hits[idx] += 1
        distances.append(np.diff(np.sort(unique_positions[idx])))
    elapsed = time.time() - start
    results[name] = (np.concatenate(distances), inclusion_hits / n_samples)
    print(f"{name:22s} {1000 * elapsed / n_samples:7.2f} ms/sample")

heaviest = int(np.argmax(pool_pmf))
print(f"\ninclusion prob of heaviest position (target {target[heaviest]:.4f}):")
for name, (_, inclusion) in results.items():
    print(f"  {name:22s} {inclusion[heaviest]:.4f}")

print(f"\ndistance null (k={k}, {n_samples} samples each):")
for name, (distances, _) in results.items():
    print(f"  {name:22s} mean={distances.mean():7.3f} "
          f"median={np.median(distances):6.1f} n={distances.size}")

sampford_distances = results["sampford"][0]
print("\nvs Sampford:")
for name in ["pivotal", "successive (current)"]:
    other = results[name][0]
    print(f"  {name:22s} KS={ks_2samp(sampford_distances, other).statistic:.4f} "
          f"W={wasserstein_distance(sampford_distances, other):.4f}")
