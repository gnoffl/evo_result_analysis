import numpy as np, sys
sys.path.insert(0, "src/workflows/mutation_distribution_analysis/plans/sampler_fix_scratch")
from cond_poisson_explained import exact_conditional_inclusion

target = np.array([0.8, 0.4, 0.4, 0.4])
k = 2
odds = target / (1 - target)
for iteration in range(9):
    working = odds / (1 + odds)
    achieved = exact_conditional_inclusion(working, k)
    print(f"iter {iteration}: odds_ratio={odds[0]/odds[1]:9.5f}  "
          f"pi_heavy={achieved[0]:.6f}  max_err={np.abs(achieved-target).max():.2e}")
    odds *= (target / (1 - target)) / (achieved / (1 - achieved))
