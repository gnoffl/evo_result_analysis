"""Is Sampford feasible (max k*p < 1) on the per-gene APPLICABLE pools?"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from workflows.mutation_distribution_analysis.baseline_sampler import (
    _filter_applicable,
)

for tag in ["ara", "zea"]:
    path = ("src/workflows/mutation_distribution_analysis/mutation_pools/"
            f"{tag}_msr_max_single_gen1999_mutation_pool.json")
    data = json.load(open(path))
    mutations = pd.DataFrame(data["mutations"])
    gene_stats = pd.DataFrame(data["gene_stats"]).set_index("gene_id")
    references = data["references"]

    max_inclusion_targets = []
    for gene_id, wildtype in references.items():
        applicable = _filter_applicable(mutations, wildtype)
        _, counts = np.unique(applicable["position"].to_numpy(), return_counts=True)
        applicable_pmf = counts / counts.sum()
        k = int(gene_stats.loc[gene_id, "n_mutations"])
        max_inclusion_targets.append(k * applicable_pmf.max())

    max_inclusion_targets = np.array(max_inclusion_targets)
    print(f"{tag}: per-gene max(k * p_applicable) over {len(references)} genes")
    print(f"  min={max_inclusion_targets.min():.4f}  "
          f"median={np.median(max_inclusion_targets):.4f}  "
          f"max={max_inclusion_targets.max():.4f}")
    print(f"  genes with max k*p >= 1 (Sampford infeasible): "
          f"{(max_inclusion_targets >= 1.0).sum()}")
    print(f"  genes with max k*p >= 0.5: "
          f"{(max_inclusion_targets >= 0.5).sum()}")
