"""Facts needed to scope the Sampford fix at both call sites."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from workflows.mutation_distribution_analysis.baseline_sampler import (
    _filter_applicable,
)

path = ("src/workflows/mutation_distribution_analysis/mutation_pools/"
        "ara_msr_max_single_gen1999_mutation_pool.json")
data = json.load(open(path))
mutations = pd.DataFrame(data["mutations"])
gene_stats = pd.DataFrame(data["gene_stats"])
references = data["references"]

# --- how many distinct k values? (the tilt depends on k) ---
distinct_k = sorted(gene_stats["n_mutations"].unique())
print(f"distinct k values across {len(gene_stats)} genes: {len(distinct_k)}")
print(f"  range {distinct_k[0]}..{distinct_k[-1]}")

# --- how much does the applicable pool vary between genes? ---
print("\napplicable pool per gene (first 5 genes):")
full_unique = mutations['position'].nunique()
print(f"  full pool: {len(mutations)} rows, {full_unique} unique positions")
same_source_base_everywhere = True
for gene_id in list(references)[:5]:
    wildtype = references[gene_id]
    applicable = _filter_applicable(mutations, wildtype)
    n_source_per_position = applicable.groupby("position")["source_base"].nunique()
    if n_source_per_position.max() > 1:
        same_source_base_everywhere = False
    print(f"  {gene_id}: {len(applicable)} rows "
          f"({100 * len(applicable) / len(mutations):.1f}% of pool), "
          f"{applicable['position'].nunique()} unique positions, "
          f"max source_bases per position = {n_source_per_position.max()}")

# --- check the source_base claim across ALL genes ---
max_source_bases = 0
max_new_bases = 0
for gene_id, wildtype in references.items():
    applicable = _filter_applicable(mutations, wildtype)
    grouped = applicable.groupby("position")
    max_source_bases = max(max_source_bases,
                           int(grouped["source_base"].nunique().max()))
    max_new_bases = max(max_new_bases, int(grouped["new_base"].nunique().max()))
print(f"\nacross all {len(references)} genes:")
print(f"  max distinct source_base at one position (post-filter) = {max_source_bases}")
print(f"  max distinct new_base    at one position (post-filter) = {max_new_bases}")
