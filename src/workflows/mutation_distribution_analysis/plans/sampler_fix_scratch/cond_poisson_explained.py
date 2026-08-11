"""Conditional Poisson sampling, built up on a pool small enough to enumerate."""
from itertools import combinations

import numpy as np


def exact_conditional_inclusion(
    working: np.ndarray, k: int
) -> np.ndarray:
    """Exact inclusion probabilities of conditional Poisson, by enumeration.

    The design puts probability proportional to the product of member odds on
    each k-subset: P(S) ~ prod_{i in S} w_i / (1 - w_i).
    """
    odds = working / (1.0 - working)
    inclusion = np.zeros(working.size)
    total = 0.0
    for subset in combinations(range(working.size), k):
        weight = float(np.prod(odds[list(subset)]))
        total += weight
        for member in subset:
            inclusion[member] += weight
    return inclusion / total


def calibrate(target: np.ndarray, k: int, iterations: int = 60) -> np.ndarray:
    """Find working probabilities whose conditional inclusion equals ``target``.

    Fixed point on the odds scale (Tille 2006): scale each unit's working odds
    by the ratio of desired to achieved odds, and repeat.
    """
    odds = target / (1.0 - target)
    for _ in range(iterations):
        working = odds / (1.0 + odds)
        achieved = exact_conditional_inclusion(working, k)
        odds *= (target / (1.0 - target)) / (achieved / (1.0 - achieved))
    return odds / (1.0 + odds)


pool_pmf = np.array([0.4, 0.2, 0.2, 0.2])
k = 2
target = k * pool_pmf
print(f"target inclusion pi          : {np.round(target, 4)}")

naive = exact_conditional_inclusion(target, k)
print(f"\nfeed working = target (naive):")
print(f"  achieved inclusion         : {np.round(naive, 4)}")
print(f"  ratio to target            : {np.round(naive / target, 4)}")

calibrated = calibrate(target, k)
achieved = exact_conditional_inclusion(calibrated, k)
print(f"\nafter calibration:")
print(f"  working probabilities      : {np.round(calibrated, 4)}")
print(f"  achieved inclusion         : {np.round(achieved, 6)}")
print(f"  max abs error vs target    : {np.abs(achieved - target).max():.2e}")

# Does the calibration survive a harder, asymmetric case?
rng = np.random.default_rng(3)
raw = rng.random(9) ** 2
pmf = raw / raw.sum()
for k_test in [3, 5]:
    tgt = k_test * pmf
    if tgt.max() >= 1.0:
        print(f"\nn=9, k={k_test}: target infeasible (max k*p="
              f"{tgt.max():.3f}), needs take-all capping")
        continue
    cal = calibrate(tgt, k_test)
    got = exact_conditional_inclusion(cal, k_test)
    naive_got = exact_conditional_inclusion(tgt, k_test)
    print(f"\nn=9, k={k_test}, max k*p={tgt.max():.3f}")
    print(f"  naive max error      : {np.abs(naive_got - tgt).max():.2e}")
    print(f"  calibrated max error : {np.abs(got - tgt).max():.2e}")
