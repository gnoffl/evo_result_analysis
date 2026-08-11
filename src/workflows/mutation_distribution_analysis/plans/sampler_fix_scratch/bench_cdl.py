"""Cost of CDL (11) calibration at realistic pool sizes, CDL vs the odds variant."""
import sys, time
import numpy as np
sys.path.insert(0, "/tmp/claude-1000/-home-gernot-Code-PhD-Code-evo-result-analysis/2093f5c2-06e8-4689-92e7-277694de9218/scratchpad")
from verify_cdl import log_leave_one_out

def calibrate(target, k, variant, tol=1e-10, max_iterations=500):
    odds = target / (1.0 - target)
    for iteration in range(1, max_iterations + 1):
        log_r_excl, log_r_full = log_leave_one_out(odds, k)
        achieved = np.exp(np.log(odds) + log_r_excl - log_r_full)
        error = np.abs(achieved - target).max()
        if error < tol:
            return odds, iteration, error
        if variant == "cdl":                      # w_j prop to pi_j / R(k-1,{j}^c)
            log_odds = np.log(target) - log_r_excl
        else:                                     # odds-ratio variant
            log_odds = (np.log(odds) + np.log(target / (1 - target))
                        - np.log(achieved / (1 - achieved)))
        odds = np.exp(log_odds - log_odds.mean())
    return odds, max_iterations, error

rng = np.random.default_rng(7)
for n, k in [(400, 88), (2100, 88), (2721, 88)]:
    weights = rng.random(n) ** 3 + 1e-3
    target = k * weights / weights.sum()
    if target.max() >= 1:
        target = np.minimum(target, 0.99)
        target *= k / target.sum()
    print(f"\nn={n}, k={k}, max target={target.max():.3f}")
    for variant in ("cdl", "odds"):
        start = time.perf_counter()
        _, iterations, error = calibrate(target, k, variant)
        elapsed = time.perf_counter() - start
        print(f"  {variant:4s}: {iterations:4d} iters, err={error:.1e}, "
              f"{elapsed:6.2f} s total, {1000*elapsed/iterations:5.1f} ms/iter")
