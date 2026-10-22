"""Is rejective conditional Poisson viable where Sampford collapsed?

Sampford's rejection is a birthday problem (collapses with skew).
Conditional Poisson's rejection is a size-matching problem: draw independent
Bernoullis, accept iff exactly k succeed. Acceptance ~ P(Poisson-binomial = k),
which depends only on sum(pi(1-pi)) -- so it degrades far more gracefully.
"""
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from workflows.mutation_distribution_analysis.baseline_sampler import (
    _filter_applicable,
)


def predicted_acceptance(inclusion_target: np.ndarray) -> float:
    """Normal approximation to P(Poisson-binomial == k)."""
    variance = np.sum(inclusion_target * (1.0 - inclusion_target))
    return 1.0 / np.sqrt(2.0 * np.pi * variance)


def measured_acceptance(
    inclusion_target: np.ndarray, k: int, rng: np.random.Generator, trials: int
) -> float:
    """Empirical acceptance of the rejective step."""
    hits = 0
    for _ in range(trials):
        if (rng.random(inclusion_target.size) < inclusion_target).sum() == k:
            hits += 1
    return hits / trials


def capped_target(pmf: np.ndarray, k: int) -> np.ndarray:
    """Proportional inclusion probabilities with iterative take-all capping."""
    target = k * pmf
    while target.max() > 1.0:
        take_all = target >= 1.0
        target[take_all] = 1.0
        free = ~take_all
        remaining = k - take_all.sum()
        target[free] = remaining * pmf[free] / pmf[free].sum()
    return target


rng = np.random.default_rng(0)
for tag in ["ara", "zea"]:
    path = ("src/workflows/mutation_distribution_analysis/mutation_pools/"
            f"{tag}_msr_max_single_gen1999_mutation_pool.json")
    data = json.load(open(path))
    mutations = pd.DataFrame(data["mutations"])
    gene_stats = pd.DataFrame(data["gene_stats"]).set_index("gene_id")
    references = data["references"]

    # Full unfiltered pool (call site 1).
    _, counts = np.unique(mutations["position"].to_numpy(), return_counts=True)
    full_pmf = counts / counts.sum()
    k_median = int(gene_stats["n_mutations"].median())
    target = capped_target(full_pmf, k_median)
    print(f"\n{tag} — UNFILTERED pool (mutation_distance_analysis), k={k_median}")
    print(f"  predicted acceptance = {predicted_acceptance(target):.4f}")

    # Per-gene applicable pools (call site 2): find the hardest genes.
    worst = []
    for gene_id, wildtype in references.items():
        applicable = _filter_applicable(mutations, wildtype)
        _, gene_counts = np.unique(
            applicable["position"].to_numpy(), return_counts=True
        )
        gene_pmf = gene_counts / gene_counts.sum()
        k = int(gene_stats.loc[gene_id, "n_mutations"])
        gene_target = capped_target(gene_pmf, k)
        worst.append((predicted_acceptance(gene_target), gene_id, gene_target, k))
    worst.sort()
    accepts = np.array([entry[0] for entry in worst])
    print(f"{tag} — APPLICABLE pools (baseline_sampler), {len(worst)} genes")
    print(f"  predicted acceptance: worst={accepts[0]:.4f} "
          f"median={np.median(accepts):.4f} best={accepts[-1]:.4f}")
    print(f"  worst-case attempts per sample = {1 / accepts[0]:.1f}")

    # Confirm the prediction on the single hardest gene.
    _, hardest_gene, hardest_target, hardest_k = worst[0]
    start = time.time()
    empirical = measured_acceptance(hardest_target, hardest_k, rng, 20_000)
    elapsed = time.time() - start
    print(f"  hardest gene ({hardest_gene[:28]}..., k={hardest_k}, "
          f"take-alls={int((hardest_target >= 1.0).sum())}):")
    print(f"    predicted={accepts[0]:.4f}  measured={empirical:.4f}")
    print(f"    -> {1000 * elapsed / 20_000 / max(empirical, 1e-9):.3f} ms "
          f"per accepted sample")
