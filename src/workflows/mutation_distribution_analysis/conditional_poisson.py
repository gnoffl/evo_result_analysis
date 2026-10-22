"""Maximum-entropy (conditional Bernoulli) sampling of distinct positions.

Draws ``k`` distinct positions from a weighted pool so that the **inclusion
probability** of position ``i`` — the probability that ``i`` ends up in the
drawn set — is exactly ``pi_i = k * p_i``, where ``p_i`` is position ``i``'s
share of the pool. This is the property a null model built by pooling real
mutation sets needs, and it is *not* what successive sampling (draw one,
remove it, renormalise, repeat) delivers: that scheme preserves the
*selection* probability of each individual draw and systematically flattens
the inclusion probabilities. See ``plans/sampler_inclusion_probability_fix.md``
for the measured size of that bias.

The design implemented here is the conditional Bernoulli / maximum-entropy
design of Chen, Dempster & Liu (1994):

1. flip one independent coin per position, position ``i``'s coin having odds
   ``w_i`` (so bias ``w_i / (1 + w_i)``);
2. condition on exactly ``k`` heads.

Conditioning gives every ``k``-subset ``S`` probability proportional to
``prod_{i in S} w_i``, which is the maximum-entropy design subject to the
size constraint. The working odds ``w`` are *not* the target inclusion
probabilities — conditioning moves the marginals — so they must be calibrated
by the inversion of Chen & Liu (1997) eq. (11), implemented in
:func:`calibrate_working_odds`.

Notation used throughout, following Chen & Liu (1997). ``n`` is the number of
positions in the pool and ``S = {1..n}`` the pool itself; ``k`` is the sample
size; ``pi`` (length ``n``, summing to ``k``) are the target inclusion
probabilities; ``w`` (length ``n``, positive) are the working odds; and

``R(k, C) = sum over all k-element subsets B of C of prod_{i in B} w_i``

is the total design weight of the ``k``-subsets drawable from a set ``C`` of
positions — the elementary symmetric polynomial of degree ``k`` in the odds of
``C``, with ``R(0, C) = 1`` and ``R(k, C) = 0`` for ``k > |C|``. The achieved
inclusion probability of position ``i`` under working odds ``w`` is then the
share of total design weight carried by the subsets containing it,

``pi_hat_i(w) = w_i * R(k - 1, S \\ {i}) / R(k, S)``

and calibration means inverting that relation.

Public API:

* :func:`inclusion_probabilities` — proportional target with take-all capping.
* :func:`log_normalising_constants` — all ``log R(k-1, S \\ {j})`` plus
  ``log R(k, S)``, by the Gail-Lubin-Rubinstein recursion.
* :func:`calibrate_working_odds` — the Chen-Dempster-Liu inversion.
* :func:`scale_odds_to_sample_size` — pin the odds' free global scale so the
  rejective draw accepts at a usable rate.
* :func:`sample_positions` — one rejective draw at given working odds.
* :class:`PositionSampler` — calibrate once per pool, draw many times.

References:
    Chen, S. X., Dempster, A. P. & Liu, J. S. (1994). Weighted finite
    population sampling to maximize entropy. *Biometrika* **81**(3), 457-469.

    Chen, S. X. & Liu, J. S. (1997). Statistical applications of the
    Poisson-binomial and conditional Bernoulli distributions. *Statistica
    Sinica* **7**(4), 875-892. Eq. (6) defines ``R``; eq. (11) p. 879 is the
    calibration; p. 877 states the maximum-entropy equivalence; p. 880 warns
    against the alternating recursion; Proposition 1(c) p. 881 is the
    prefix/suffix convolution used here.

    Gail, M. H., Lubin, J. H. & Rubinstein, L. V. (1981). Likelihood
    calculations for matched case-control studies and survival studies with
    tied death times. *Biometrika* **68**(3), 703-707. The non-alternating
    recursion for ``R``.
"""

from __future__ import annotations

import numpy as np

# Default stopping rule for the calibration: the largest absolute deviation
# between achieved and target inclusion probability that counts as converged.
DEFAULT_TOLERANCE = 1e-10

# Default cap on calibration iterations. No convergence proof is known for the
# Chen-Dempster-Liu inversion, so the cap exists to turn a hypothetical
# non-convergence into an exception rather than an infinite loop. The real
# pools converge in 8-9 iterations; the worst case measured during planning
# (n = 400, max pi = 0.847) took 95.
DEFAULT_MAX_ITERATIONS = 500

# Default cap on rejective-draw attempts. Acceptance is ~1/sqrt(2*pi*k),
# i.e. 4.3% at k = 88 (about 23 attempts), essentially independent of how
# skewed the weights are.
DEFAULT_MAX_ATTEMPTS = 10000


def inclusion_probabilities(
    counts: np.ndarray, sample_size: int
) -> np.ndarray:
    """Target inclusion probabilities proportional to ``counts``, with capping.

    Returns ``pi`` with ``sum(pi) == sample_size`` and every ``pi_i <= 1``.
    The unconstrained target is ``pi_i = sample_size * counts_i / sum(counts)``,
    which can exceed 1 for a sufficiently dominant position — an infeasible
    request, since an inclusion probability is a probability. Such positions
    are given ``pi_i = 1`` ("take-all"): they enter every sample. Removing them
    raises the proportional share of the remainder, which can push a second
    position over 1, so the capping is iterated to a fixed point. This matches
    R ``sampling::inclusionprobabilities()``.

    Capping rather than excluding the affected gene is a deliberate choice,
    argued in ``plans/sampler_inclusion_probability_fix.md`` §2.2: on the real
    pools it forces 1-2 positions of ~90 and displaces at most 0.18% of the
    inclusion mass, whereas dropping would cut on a continuous quantity that
    correlates with the outcome the baselines are built to measure.

    Positions with a zero count receive ``pi_i = 0`` and are never drawn.

    Args:
        counts: Non-negative pool weight per position (typically the number of
            pool rows at that position). At least one entry must be positive
            unless ``sample_size`` is zero, in which case an empty or all-zero
            pool is legal — nothing is being drawn from it.
        sample_size: Number of distinct positions to be drawn. Must not exceed
            the number of positions with a positive count.

    Returns:
        Array of the same length as ``counts`` holding the target inclusion
        probabilities, including the capped ``1.0`` entries and the ``0.0``
        entries of zero-count positions.

    Raises:
        ValueError: If ``counts`` has a negative entry, is all zero, or
            ``sample_size`` is negative or exceeds the number of positive
            counts.

    Example:
        >>> inclusion_probabilities(np.array([80.0, 10.0, 10.0]), 2)
        array([1.        , 0.5       , 0.5       ])
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1:
        raise ValueError("counts must be a 1-D array.")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative.")
    if sample_size < 0:
        raise ValueError(f"sample_size must be non-negative, got {sample_size}.")
    if sample_size == 0:
        return np.zeros_like(counts)
    if not np.any(counts > 0):
        raise ValueError("counts must contain at least one positive entry.")

    positive = counts > 0
    n_positive = int(positive.sum())
    if sample_size > n_positive:
        raise ValueError(
            f"Cannot draw {sample_size} distinct positions: only {n_positive} "
            "positions have a positive count."
        )

    target = np.zeros_like(counts)
    active = positive.copy()
    remaining_size = float(sample_size)

    while active.any() and remaining_size > 0:
        proportional = remaining_size * counts[active] / counts[active].sum()
        if proportional.max() < 1.0:
            target[active] = proportional
            return target
        active_indices = np.flatnonzero(active)
        newly_capped = active_indices[proportional >= 1.0]
        target[newly_capped] = 1.0
        active[newly_capped] = False
        remaining_size -= newly_capped.size

    return target


def log_normalising_constants(
    working_odds: np.ndarray, sample_size: int
) -> tuple[np.ndarray, float]:
    """Compute ``log R(k-1, S \\ {j})`` for every ``j``, and ``log R(k, S)``.

    Uses the Gail-Lubin-Rubinstein recursion
    ``R(j, C + {x}) = R(j, C) + w_x * R(j-1, C)``, which involves only sums and
    products of non-negative numbers, in a prefix/suffix pair of tables:

    ``forward[m][j] = R(j, {1..m})`` and ``backward[m][j] = R(j, {m..n})``.

    The leave-one-out constants follow from Chen & Liu's Proposition 1(c),

    ``R(k-1, S \\ {j}) = sum_{a=0..k-1} forward[j-1][a] * backward[j+1][k-1-a]``.

    Both tables are rescaled row by row (each row divided by its own maximum,
    the log of the divisor accumulated separately), because ``R(88, S)`` for a
    pool of 2721 positions is of order ``1e169`` and skewed odds make
    intermediates worse. The scale of a convolution term depends only on the
    row indices ``j-1`` and ``j+1``, never on the summation index ``a``, so it
    factors out of the sum exactly and the convolution stays in ordinary
    (scaled) arithmetic.

    Cost is ``O(n * k)`` in both time and memory: about 2 MB and 22 ms at
    ``n = 2721, k = 88``.

    Args:
        working_odds: Strictly positive odds ``w``, one per position.
        sample_size: Sample size ``k``. Must satisfy ``1 <= k < n`` so that
            every leave-one-out constant is non-zero.

    Returns:
        A tuple ``(log_r_excluding, log_r_full)`` where ``log_r_excluding[j]``
        is ``log R(k-1, S \\ {j})`` and ``log_r_full`` is ``log R(k, S)``.

    Raises:
        ValueError: If ``working_odds`` is not 1-D and strictly positive and
            finite, or if ``sample_size`` is outside ``[1, n - 1]``.
    """
    working_odds = np.asarray(working_odds, dtype=float)
    if working_odds.ndim != 1:
        raise ValueError("working_odds must be a 1-D array.")
    if not np.all(np.isfinite(working_odds)) or np.any(working_odds <= 0):
        raise ValueError("working_odds must be finite and strictly positive.")

    n_positions = working_odds.size
    if not 1 <= sample_size < n_positions:
        raise ValueError(
            f"sample_size must satisfy 1 <= k < n; got k={sample_size}, "
            f"n={n_positions}."
        )

    forward = np.zeros((n_positions + 1, sample_size + 1))
    forward[0, 0] = 1.0
    log_scale_forward = np.zeros(n_positions + 1)
    for row in range(1, n_positions + 1):
        # R(0, .) = 1 must be carried in the *previous row's scaled units*,
        # never reset to a literal 1.0 — resetting it silently corrupts the
        # whole table (it produced log R errors of order 1e0 during planning).
        forward[row, 0] = forward[row - 1, 0]
        forward[row, 1:] = (
            forward[row - 1, 1:] + working_odds[row - 1] * forward[row - 1, :-1]
        )
        row_maximum = forward[row].max()
        forward[row] /= row_maximum
        log_scale_forward[row] = log_scale_forward[row - 1] + np.log(row_maximum)

    backward = np.zeros((n_positions + 2, sample_size + 1))
    backward[n_positions + 1, 0] = 1.0
    log_scale_backward = np.zeros(n_positions + 2)
    for row in range(n_positions, 0, -1):
        backward[row, 0] = backward[row + 1, 0]
        backward[row, 1:] = (
            backward[row + 1, 1:] + working_odds[row - 1] * backward[row + 1, :-1]
        )
        row_maximum = backward[row].max()
        backward[row] /= row_maximum
        log_scale_backward[row] = (
            log_scale_backward[row + 1] + np.log(row_maximum)
        )

    log_r_excluding = np.empty(n_positions)
    for position in range(1, n_positions + 1):
        convolution = float(
            np.dot(
                forward[position - 1, :sample_size],
                backward[position + 1, :sample_size][::-1],
            )
        )
        log_r_excluding[position - 1] = (
            log_scale_forward[position - 1]
            + log_scale_backward[position + 1]
            + np.log(convolution)
        )

    log_r_full = float(
        log_scale_forward[n_positions] + np.log(forward[n_positions, sample_size])
    )
    return log_r_excluding, log_r_full


def calibrate_working_odds(
    target: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> np.ndarray:
    """Find working odds whose inclusion probabilities equal ``target``.

    Implements the Chen, Dempster & Liu (1994) inversion, reported as eq. (11)
    of Chen & Liu (1997) p. 879: ``w_j`` proportional to
    ``pi_j / R(k-1, S \\ {j})``, iterated to a fixed point. Substituting the
    identity ``pi_hat_j = w_j R(k-1, S \\ {j}) / R(k, S)`` shows what one step
    does — up to a factor common to all ``j`` it is
    ``w_j <- w_j * pi_j / pi_hat_j``, a multiplicative correction on the odds
    scale by the ratio of desired to achieved inclusion probability.

    The iteration is started from ``w = target / (1 - target)``, the Hájek
    point, which is where the marginals would already be right if conditioning
    did not move them.

    **Fixing the free scale.** ``w`` is only determined up to a global factor:
    scaling every ``w_i`` by ``c`` multiplies both numerator and denominator of
    ``pi_hat_i`` by ``c^k``. Chen & Liu pin the scale by holding ``w_n = pi_n``;
    this implementation instead mean-centres ``log w`` each iteration. **That is
    not a deviation from the method.** Every ``R(k-1, S \\ {j})`` is homogeneous
    of degree ``k-1`` in ``w``, so all ``n`` of them rescale by the same factor
    and eq. (11)'s ratio is unchanged — the map is provably insensitive to the
    scale of its input. Mean-centring keeps the recursion in numerical range
    regardless of which position happens to be last in the array, whereas
    pinning the last coordinate does not.

    Args:
        target: Target inclusion probabilities, every entry strictly inside
            ``(0, 1)`` and summing to an integer ``k`` with ``1 <= k < n``.
            Callers must strip zero-count and take-all-capped positions first
            (see :func:`inclusion_probabilities`).
        tolerance: Stop once ``max|pi_hat - target| < tolerance``.
        max_iterations: Maximum number of iterations before giving up.

    Returns:
        The calibrated working odds, mean-centred on the log scale. Only their
        ratios are meaningful.

    Raises:
        ValueError: If ``target`` is not 1-D, has an entry outside ``(0, 1)``,
            or does not sum to an integer ``k`` in ``[1, n - 1]``.
        RuntimeError: If ``max_iterations`` is reached without meeting
            ``tolerance``, or if the iterates leave the representable range.
            No convergence proof is known for this iteration, so convergence
            is a checked runtime property rather than an assumption; an
            unconverged result is never returned.
    """
    target = np.asarray(target, dtype=float)
    if target.ndim != 1:
        raise ValueError("target must be a 1-D array.")
    if np.any(target <= 0.0) or np.any(target >= 1.0):
        raise ValueError(
            "target inclusion probabilities must lie strictly inside (0, 1); "
            "strip zero-count and take-all-capped positions first."
        )

    target_sum = float(target.sum())
    sample_size = int(round(target_sum))
    if abs(target_sum - sample_size) > 1e-8:
        raise ValueError(
            f"target must sum to an integer sample size, got {target_sum}."
        )
    if not 1 <= sample_size < target.size:
        raise ValueError(
            f"target must sum to k with 1 <= k < n; got k={sample_size}, "
            f"n={target.size}."
        )

    log_target = np.log(target)
    working_odds = target / (1.0 - target)

    for _ in range(max_iterations):
        log_r_excluding, log_r_full = log_normalising_constants(
            working_odds, sample_size
        )
        achieved = np.exp(np.log(working_odds) + log_r_excluding - log_r_full)
        if np.abs(achieved - target).max() < tolerance:
            return working_odds

        log_working_odds = log_target - log_r_excluding
        log_working_odds -= log_working_odds.mean()
        if not np.all(np.isfinite(log_working_odds)):
            raise RuntimeError(
                "Calibration left the representable range: log working odds "
                "contain non-finite entries."
            )
        working_odds = np.exp(log_working_odds)

    final_error = float(np.abs(achieved - target).max())
    raise RuntimeError(
        f"Working-odds calibration did not converge in {max_iterations} "
        f"iterations: max|achieved - target| = {final_error:.3e}, "
        f"tolerance = {tolerance:.3e}."
    )


def scale_odds_to_sample_size(
    working_odds: np.ndarray, sample_size: int
) -> np.ndarray:
    """Rescale working odds so the coins produce ``sample_size`` heads on average.

    The conditional Bernoulli *design* is invariant to a global factor on the
    odds — scaling every ``w_i`` by ``c`` multiplies numerator and denominator
    of ``pi_hat_i`` by ``c^k`` — so :func:`calibrate_working_odds` is free to
    return whatever scale is numerically convenient. The **rejective draw is
    not** invariant: it flips unconditional coins with bias
    ``w_i / (1 + w_i)`` and keeps only the draws with exactly ``sample_size``
    heads, so its acceptance rate depends on how far the unconditional mean
    head count ``sum_i w_i / (1 + w_i)`` sits from ``sample_size``. At the real
    pool sizes, mean-centred odds put that mean thousands of standard
    deviations away and no draw is ever accepted.

    This function therefore finds the unique ``c > 0`` with

    ``sum_i (c * w_i) / (1 + c * w_i) = sample_size``

    and returns ``c * working_odds``. The left-hand side is continuous and
    strictly increasing in ``c``, running from 0 to ``n``, so for
    ``0 < sample_size < n`` the root exists, is unique, and is found here by
    bisection on ``log c``. Acceptance at the returned scale is the
    Poisson-binomial mode probability, ``~1/sqrt(2*pi*k)``.

    The sampled distribution is identical either way; only the cost of
    obtaining it changes.

    Args:
        working_odds: Strictly positive, finite odds.
        sample_size: Target mean (and conditioned) number of heads, with
            ``0 < sample_size < n``.

    Returns:
        The rescaled odds. Ratios — and hence the design — are unchanged.

    Raises:
        ValueError: If ``working_odds`` is not 1-D, finite and strictly
            positive, or ``sample_size`` is outside ``(0, n)``.
    """
    working_odds = np.asarray(working_odds, dtype=float)
    if working_odds.ndim != 1:
        raise ValueError("working_odds must be a 1-D array.")
    if not np.all(np.isfinite(working_odds)) or np.any(working_odds <= 0):
        raise ValueError("working_odds must be finite and strictly positive.")
    if not 0 < sample_size < working_odds.size:
        raise ValueError(
            f"sample_size must satisfy 0 < k < n; got k={sample_size}, "
            f"n={working_odds.size}."
        )

    log_odds = np.log(working_odds)

    def excess_heads(log_factor: float) -> float:
        """Mean head count at ``c = exp(log_factor)``, minus ``sample_size``."""
        shifted = log_odds + log_factor
        # Logistic evaluated stably: 1 / (1 + exp(-x)).
        return float(np.sum(1.0 / (1.0 + np.exp(-shifted)))) - sample_size

    lower, upper = -1.0, 1.0
    while excess_heads(lower) > 0.0:
        lower *= 2.0
    while excess_heads(upper) < 0.0:
        upper *= 2.0

    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if excess_heads(middle) < 0.0:
            lower = middle
        else:
            upper = middle
        if upper - lower < 1e-14 * max(1.0, abs(upper)):
            break

    return np.exp(log_odds + 0.5 * (lower + upper))


def sample_positions(
    working_odds: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> np.ndarray:
    """Draw ``sample_size`` distinct positions by the rejective method.

    Flips one Bernoulli(``w_i / (1 + w_i)``) coin per position and accepts the
    draw only if exactly ``sample_size`` came up heads. This *is* the
    definition of the conditional Bernoulli design — conditioning on the size —
    so its correctness needs no argument beyond that definition.

    Acceptance is ``P(|S| = k) ~ 1/sqrt(2*pi*k)``, which depends essentially
    only on ``k`` and not on how skewed the odds are: 4.3% at ``k = 88``, about
    23 attempts and 0.17 ms. A rejection-free sequential alternative (Chen &
    Liu 1997, Procedure 3) is documented in
    ``plans/sampler_inclusion_probability_fix.md`` §4.4 should this ever prove
    inadequate.

    That acceptance rate is only available at the right *scale* of the odds.
    Unlike the design itself, the rejective draw is not scale-invariant, so the
    odds are passed through :func:`scale_odds_to_sample_size` unless their mean
    head count already equals ``sample_size``. Rescaling does not change the
    distribution of the returned sample, only the number of attempts needed to
    obtain it.

    Args:
        working_odds: Calibrated odds from :func:`calibrate_working_odds`.
        sample_size: Number of positions to draw.
        rng: NumPy ``Generator`` supplying the coin flips.
        max_attempts: Give up after this many rejected draws.

    Returns:
        Ascending array of ``sample_size`` indices into ``working_odds``.

    Raises:
        ValueError: If ``working_odds`` is not strictly positive and finite, or
            ``sample_size`` is outside ``[0, n]``.
        RuntimeError: If no draw of the right size occurred within
            ``max_attempts``.
    """
    working_odds = np.asarray(working_odds, dtype=float)
    if working_odds.ndim != 1:
        raise ValueError("working_odds must be a 1-D array.")
    if not np.all(np.isfinite(working_odds)) or np.any(working_odds <= 0):
        raise ValueError("working_odds must be finite and strictly positive.")
    if not 0 <= sample_size <= working_odds.size:
        raise ValueError(
            f"sample_size must satisfy 0 <= k <= n; got k={sample_size}, "
            f"n={working_odds.size}."
        )

    if sample_size == 0:
        return np.empty(0, dtype=int)
    if sample_size == working_odds.size:
        return np.arange(working_odds.size)

    biases = working_odds / (1.0 + working_odds)
    if abs(float(biases.sum()) - sample_size) > 1e-6:
        working_odds = scale_odds_to_sample_size(working_odds, sample_size)
        biases = working_odds / (1.0 + working_odds)

    for _ in range(max_attempts):
        heads = rng.random(working_odds.size) < biases
        if int(heads.sum()) == sample_size:
            return np.flatnonzero(heads)

    raise RuntimeError(
        f"Rejective sampling failed to draw exactly {sample_size} positions "
        f"in {max_attempts} attempts."
    )


class PositionSampler:
    """Maximum-entropy sampler for one pool, calibrated once and reused.

    Calibration costs about 0.2 s at the real pool sizes while a draw costs
    about 0.2 ms, so the working odds are computed in ``__init__`` and
    amortised over the thousands of draws each call site makes from the same
    pool.

    Positions whose count is zero are excluded from the design. Positions whose
    proportional target exceeds 1 are taken with certainty (see
    :func:`inclusion_probabilities`): they are forced into every draw and the
    remaining positions are calibrated against a correspondingly reduced sample
    size, mirroring R ``sampling::UPmaxentropy``.

    Attributes:
        counts: The pool weights the sampler was built from.
        sample_size: Number of distinct positions returned by each draw.
        target: Target inclusion probabilities, one per entry of ``counts``.
        working_odds: Calibrated odds of the freely-sampled positions, aligned
            with ``free_indices``. Empty when every drawn position is forced.
        forced_indices: Indices taken with certainty (``target == 1``).
        free_indices: Indices participating in the rejective draw.
    """

    def __init__(
        self,
        counts: np.ndarray,
        sample_size: int,
        tolerance: float = DEFAULT_TOLERANCE,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        """Build and calibrate a sampler for one pool.

        Args:
            counts: Non-negative pool weight per position.
            sample_size: Number of distinct positions each draw returns.
            tolerance: Calibration stopping rule, see
                :func:`calibrate_working_odds`.
            max_iterations: Calibration iteration cap.

        Raises:
            ValueError: Propagated from :func:`inclusion_probabilities` for an
                invalid pool or sample size.
            RuntimeError: Propagated from :func:`calibrate_working_odds` if the
                calibration does not converge.
        """
        self.counts = np.asarray(counts, dtype=float)
        self.sample_size = int(sample_size)
        self.target = inclusion_probabilities(self.counts, self.sample_size)

        self.forced_indices = np.flatnonzero(self.target >= 1.0)
        self.free_indices = np.flatnonzero(
            (self.target > 0.0) & (self.target < 1.0)
        )
        self._free_sample_size = self.sample_size - self.forced_indices.size

        if self._free_sample_size > 0:
            calibrated = calibrate_working_odds(
                self.target[self.free_indices],
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            # Pin the free scale so the rejective draw accepts at the
            # Poisson-binomial mode rate; the design is unaffected.
            self.working_odds = scale_odds_to_sample_size(
                calibrated, self._free_sample_size
            )
        else:
            self.working_odds = np.empty(0)

    def draw(
        self,
        rng: np.random.Generator,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> np.ndarray:
        """Draw one sample of distinct positions.

        Args:
            rng: NumPy ``Generator`` supplying the coin flips.
            max_attempts: Rejective-draw attempt cap, see
                :func:`sample_positions`.

        Returns:
            Ascending array of ``sample_size`` indices into ``counts``. Each
            index ``i`` appears with probability ``target[i]``.

        Raises:
            RuntimeError: Propagated from :func:`sample_positions` if no draw
                of the right size occurred within ``max_attempts``.
        """
        if self._free_sample_size == 0:
            return np.sort(self.forced_indices)

        drawn = sample_positions(
            self.working_odds, self._free_sample_size, rng, max_attempts
        )
        return np.sort(
            np.concatenate([self.forced_indices, self.free_indices[drawn]])
        )
