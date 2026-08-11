"""Predicted Sampford acceptance rate on per-gene APPLICABLE pools."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from workflows.mutation_distribution_analysis.baseline_sampler import (
    _filter_applicable,
)


def predicted_acceptance(applicable_pmf: np.ndarray, k: int) -> float:
    """Poisson approximation to P(all k Sampford draws distinct)."""
    inclusion_target = k * applicable_pmf
    if inclusion_target.max() >= 1.0:
        return float("nan")
    tilted = applicable_pmf / (1.0 - inclusion_target)
    tilted /= tilted.sum()
    expected_collisions = (
        (k - 1) * (k - 2) / 2 * np.sum(tilted**2)
        + (k - 1) * np.sum(applicable_pmf * tilted)
    )
    return float(np.exp(-expected_collisions))


for tag in ["ara", "zea"]:
    path = ("src/workflows/mutation_distribution_analysis/mutation_pools/"
            f"{tag}_msr_max_single_gen1999_mutation_pool.json")
    data = json.load(open(path))
    mutations = pd.DataFrame(data["mutations"])
    gene_stats = pd.DataFrame(data["gene_stats"]).set_index("gene_id")
    references = data["references"]

    rates = []
    for gene_id, wildtype in references.items():
        applicable = _filter_applicable(mutations, wildtype)
        _, counts = np.unique(applicable["position"].to_numpy(), return_counts=True)
        k = int(gene_stats.loc[gene_id, "n_mutations"])
        rates.append(predicted_acceptance(counts / counts.sum(), k))

    rates = np.array(rates)
    finite = rates[np.isfinite(rates)]
    print(f"{tag}: predicted Sampford acceptance on applicable pools")
    print(f"  infeasible genes (nan): {np.isnan(rates).sum()}")
    print(f"  best={finite.max():.2e} median={np.median(finite):.2e} "
          f"worst={finite.min():.2e}")
    print(f"  median attempts/sample = {1 / np.median(finite):.0f}")
    print(f"  genes needing >10,000 attempts: {(finite < 1e-4).sum()}")
