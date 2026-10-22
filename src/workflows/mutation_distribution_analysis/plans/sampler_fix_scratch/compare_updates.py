"""Compare the Chen-Dempster-Liu (1994) update with the odds-scale variant."""
import numpy as np, sys
sys.path.insert(0, "src/workflows/mutation_distribution_analysis/plans/sampler_fix_scratch")
from cond_poisson_explained import exact_conditional_inclusion

target = np.array([0.8, 0.4, 0.4, 0.4])
k = 2

def run(update_name):
    odds = target / (1 - target)
    errors = []
    for _ in range(30):
        achieved = exact_conditional_inclusion(odds / (1 + odds), k)
        errors.append(np.abs(achieved - target).max())
        if update_name == "cdl":            # w <- w * pi / pi_hat
            odds *= target / achieved
        else:                                # w <- w * odds(pi) / odds(pi_hat)
            odds *= (target / (1 - target)) / (achieved / (1 - achieved))
    return np.array(errors)

for name in ("cdl", "odds"):
    errors = run(name)
    ratios = errors[1:12] / errors[:11]
    print(f"{name:5s}: err[0..6] = {np.array2string(errors[:7], precision=2)}")
    print(f"       contraction ratios -> {np.round(ratios[-4:], 4)}")
    print(f"       iterations to reach 1e-15: "
          f"{int(np.argmax(errors < 1e-15)) if (errors < 1e-15).any() else '>30'}")
