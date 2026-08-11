"""Quantify the successive-sampling vs pi-ps bias in the distance null."""

import json
import sys

import numpy as np
import pandas as pd

pool_path = sys.argv[1]
data = json.load(open(pool_path))
mutations = pd.DataFrame(data["mutations"])
gene_stats = pd.DataFrame(data["gene_stats"])

positions = mutations["position"].to_numpy()
unique_positions, counts = np.unique(positions, return_counts=True)
weights = counts / counts.sum()
draw_counts = gene_stats["n_mutations"].to_numpy()

print(f"pool rows                 : {len(positions)}")
print(f"unique positions          : {unique_positions.size}")
print(f"position span             : {unique_positions.min()}-{unique_positions.max()}")
print(f"k: min/median/mean/max    : {draw_counts.min()}/{np.median(draw_counts)}/"
      f"{draw_counts.mean():.1f}/{draw_counts.max()}")
print(f"weight max/mean ratio     : {weights.max() / weights.mean():.2f}")
for k in [int(np.median(draw_counts)), int(draw_counts.mean()), int(draw_counts.max())]:
    print(f"  k={k:4d}: max k*p = {k * weights.max():.4f}, "
          f"frac positions with k*p>0.5 = {(k * weights > 0.5).mean():.2e}")


def successive(k, rng):
    return rng.choice(unique_positions, size=k, replace=False, p=weights)


def pips_conditional_poisson(k, rng, n_iter=40):
    """Conditional-Poisson sampler with weights tuned so pi_i ~= k*p_i."""
    target = np.clip(k * weights, 1e-12, 1 - 1e-9)
    lam = target / (1 - target)  # start from odds
    # Simple fixed-point: rejective Poisson sampling conditioned on size k.
    for _ in range(n_iter):
        prob = lam / (1 + lam)
        # inclusion probs of conditional Poisson approximated by Hajek correction
        d = np.sum(prob * (1 - prob))
        approx = prob + prob * (1 - prob) * (k - prob.sum()) / max(d, 1e-12)
        approx = np.clip(approx, 1e-12, 1 - 1e-9)
        lam *= target / approx
    prob = lam / (1 + lam)
    # rejective sampling: draw Bernoulli until exactly k selected
    for _ in range(20000):
        sel = rng.random(prob.size) < prob
        if sel.sum() == k:
            return unique_positions[sel]
    return successive(k, rng)  # fallback


def inclusion_freq(sampler, k, reps, rng):
    hits = np.zeros(unique_positions.size)
    for _ in range(reps):
        drawn = sampler(k, rng)
        idx = np.searchsorted(unique_positions, drawn)
        hits[idx] += 1
    return hits / reps


def distances(sampler, reps, rng, ks):
    out = []
    for k in ks:
        if k < 2:
            continue
        for _ in range(reps):
            drawn = np.sort(sampler(k, rng))
            out.append(np.diff(drawn))
    return np.concatenate(out)


rng = np.random.default_rng(0)
k_test = int(np.median(draw_counts))
reps = 4000
freq_succ = inclusion_freq(successive, k_test, reps, rng)
target = k_test * weights
heavy = np.argsort(weights)[-10:]
print(f"\n--- inclusion probabilities, k={k_test}, {reps} reps ---")
print("10 heaviest positions: target k*p vs empirical successive inclusion")
for i in heavy[::-1]:
    print(f"  pos {unique_positions[i]:5d}  k*p={target[i]:.4f}  succ={freq_succ[i]:.4f}"
          f"  ratio={freq_succ[i] / target[i]:.3f}")

rng = np.random.default_rng(1)
ks = draw_counts
d_succ = distances(successive, 20, np.random.default_rng(2), ks)
d_pips = distances(pips_conditional_poisson, 20, np.random.default_rng(3), ks)
from scipy.stats import ks_2samp, wasserstein_distance
print("\n--- resulting inter-mutation distance null ---")
for name, arr in [("successive (current)", d_succ), ("pi-ps target", d_pips)]:
    print(f"  {name:22s} n={arr.size:8d} mean={arr.mean():8.2f} "
          f"median={np.median(arr):7.1f} p10={np.percentile(arr,10):6.1f}")
print(f"  KS = {ks_2samp(d_succ, d_pips).statistic:.4f}   "
      f"Wasserstein = {wasserstein_distance(d_succ, d_pips):.3f}")
