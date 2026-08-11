"""Verify CDL (11) literally, and the scaled prefix/suffix recursion for R."""
from itertools import combinations
import numpy as np

def brute_R(odds, k, exclude=None):
    idx = [i for i in range(odds.size) if i != exclude]
    return sum(float(np.prod(odds[list(c)])) for c in combinations(idx, k)) if k else 1.0

def log_leave_one_out(odds, k):
    """log R(k-1, {j}^c) for every j, and log R(k, S), via scaled prefix/suffix DP.

    Rows are scaled by their max; the scale of a term F[j-1][a]*B[j+1][k-1-a] does
    not depend on a, so it factors out of the convolution exactly.
    """
    n = odds.size
    forward = np.zeros((n + 1, k + 1)); forward[0, 0] = 1.0
    log_scale_forward = np.zeros(n + 1)
    for m in range(1, n + 1):
        forward[m, 0] = forward[m - 1, 0]   # e_0, in row m-1's scaled units
        forward[m, 1:] = forward[m - 1, 1:] + odds[m - 1] * forward[m - 1, :-1]
        row_max = forward[m].max()
        forward[m] /= row_max
        log_scale_forward[m] = log_scale_forward[m - 1] + np.log(row_max)
    backward = np.zeros((n + 2, k + 1)); backward[n + 1, 0] = 1.0
    log_scale_backward = np.zeros(n + 2)
    for m in range(n, 0, -1):
        backward[m, 0] = backward[m + 1, 0]
        backward[m, 1:] = backward[m + 1, 1:] + odds[m - 1] * backward[m + 1, :-1]
        row_max = backward[m].max()
        backward[m] /= row_max
        log_scale_backward[m] = log_scale_backward[m + 1] + np.log(row_max)
    log_r_excl = np.empty(n)
    for j in range(1, n + 1):
        total = float(np.dot(forward[j - 1, :k], backward[j + 1, :k][::-1]))
        log_r_excl[j - 1] = (log_scale_forward[j - 1] + log_scale_backward[j + 1]
                             + np.log(total))
    log_r_full = log_scale_forward[n] + np.log(forward[n, k])
    return log_r_excl, log_r_full

def calibrate_cdl(target, k, iterations=200, tol=1e-14):
    """CDL (1994) eq. (11): w_j proportional to pi_j / R(k-1, {j}^c), w_last pinned."""
    odds = target / (1.0 - target)
    for iteration in range(iterations):
        log_r_excl, log_r_full = log_leave_one_out(odds, k)
        achieved = np.exp(np.log(odds) + log_r_excl - log_r_full)
        error = np.abs(achieved - target).max()
        if error < tol:
            return odds, iteration, error
        log_odds = np.log(target) - log_r_excl
        log_odds += np.log(target[-1]) - log_odds[-1]      # pin w_last = pi_last
        odds = np.exp(log_odds - log_odds.mean())          # rescale: no effect on map
    return odds, iterations, error

# --- recursion vs brute force -------------------------------------------------
rng = np.random.default_rng(0)
for n, k in [(4, 2), (9, 3), (12, 5)]:
    odds = rng.random(n) * 6 + 0.05
    log_excl, log_full = log_leave_one_out(odds, k)
    ref_full = np.log(brute_R(odds, k))
    ref_excl = np.array([np.log(brute_R(odds, k - 1, exclude=j)) for j in range(n)])
    print(f"n={n:2d} k={k}: log R(k,S) err={abs(log_full-ref_full):.2e}  "
          f"leave-one-out max err={np.abs(log_excl-ref_excl).max():.2e}")

# --- CDL calibration ----------------------------------------------------------
print()
cases = [(np.array([0.8, 0.4, 0.4, 0.4]), 2)]
raw = rng.random(9) ** 2
cases.append((3 * raw / raw.sum(), 3))
raw = rng.random(40) ** 3
cases.append((10 * raw / raw.sum(), 10))
for target, k in cases:
    if target.max() >= 1:
        print(f"n={target.size} k={k}: infeasible, skipped"); continue
    odds, iters, err = calibrate_cdl(target, k)
    print(f"n={target.size:2d} k={k:2d} max_target={target.max():.3f}: "
          f"converged in {iters} iters, max err={err:.2e}")
