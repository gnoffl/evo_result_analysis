"""Pivotal method: exact first-order inclusion probabilities, no rejection."""
import numpy as np


def pivotal_sample(
    inclusion_target: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw a sample with inclusion probabilities exactly ``inclusion_target``.

    Deville & Tille (1998) pivotal method with random pairing. Repeatedly
    picks two undecided units and runs a mass-conserving "duel" that settles
    at least one of them to 0 or 1. The update is a martingale in each
    probability, so the final indicator has expectation equal to the target.

    Args:
        inclusion_target: Vector in [0, 1] summing to an integer ``k``.
        rng: NumPy generator.

    Returns:
        Sorted array of selected indices, of length ``round(sum(target))``.
    """
    probabilities = inclusion_target.astype(float).copy()
    undecided = [
        i for i in range(probabilities.size) if 0.0 < probabilities[i] < 1.0
    ]
    rng.shuffle(undecided)
    while len(undecided) >= 2:
        first = undecided.pop()
        second = undecided.pop()
        mass = probabilities[first] + probabilities[second]
        if mass < 1.0:
            if rng.random() < probabilities[first] / mass:
                probabilities[first], probabilities[second] = mass, 0.0
                winner = first
            else:
                probabilities[first], probabilities[second] = 0.0, mass
                winner = second
        else:
            survivor_probability = (1.0 - probabilities[second]) / (2.0 - mass)
            if rng.random() < survivor_probability:
                probabilities[first], probabilities[second] = 1.0, mass - 1.0
                winner = second
            else:
                probabilities[first], probabilities[second] = mass - 1.0, 1.0
                winner = first
        if 0.0 < probabilities[winner] < 1.0:
            undecided.append(winner)
            rng.shuffle(undecided)
    return np.flatnonzero(probabilities > 0.5)


# --- verify on the toy pool where the exact answer is known ---
pool_pmf = np.array([0.4, 0.2, 0.2, 0.2])
k = 2
target = k * pool_pmf
rng = np.random.default_rng(0)
draws = 400_000
hits = np.zeros(4)
for _ in range(draws):
    selected = pivotal_sample(target, rng)
    assert selected.size == k, f"got {selected.size} units, expected {k}"
    hits[selected] += 1
print("TOY POOL, k=2")
print(f"  target pi          : {np.round(target, 4)}")
print(f"  pivotal empirical  : {np.round(hits / draws, 4)}")
