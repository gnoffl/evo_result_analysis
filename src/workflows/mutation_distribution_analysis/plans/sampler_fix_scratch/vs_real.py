import json, sys, numpy as np, pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
for tag in ["ara", "zea"]:
    p = f"src/workflows/mutation_distribution_analysis/mutation_pools/{tag}_msr_max_single_gen1999_mutation_pool.json"
    d = json.load(open(p))
    mut = pd.DataFrame(d["mutations"]); gs = pd.DataFrame(d["gene_stats"])
    up, cnt = np.unique(mut["position"].to_numpy(), return_counts=True)
    w = cnt / cnt.sum()
    real = np.concatenate([np.diff(np.sort(g["position"].to_numpy()))
                           for _, g in mut.groupby("gene_id") if len(g) > 1])
    rng = np.random.default_rng(7)
    null = np.concatenate([np.diff(np.sort(rng.choice(up, size=int(k), replace=False, p=w)))
                           for k in gs["n_mutations"] if k >= 2 for _ in range(20)])
    print(f"{tag}: weight max/mean={w.max()/w.mean():.2f}  max k*p={gs['n_mutations'].max()*w.max():.3f}")
    print(f"  real mean={real.mean():.2f} median={np.median(real):.1f} | "
          f"null mean={null.mean():.2f} median={np.median(null):.1f}")
    print(f"  real-vs-null KS={ks_2samp(real, null).statistic:.4f} "
          f"W={wasserstein_distance(real, null):.3f}")
