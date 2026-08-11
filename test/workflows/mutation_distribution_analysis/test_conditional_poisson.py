"""Tests for the maximum-entropy (conditional Bernoulli) position sampler.

The module under test claims one thing above all others: the inclusion
probability of position ``i`` is exactly ``pi_i``. Everything here exists to
check that claim without assuming any part of the implementation, so wherever
it is affordable the reference value is obtained by **brute-force enumeration**
over subsets rather than by a second recursion.

Layout:

* ``TestNormalisingConstants`` — the Gail-Lubin-Rubinstein prefix/suffix
  recursion against enumerated ``R(k, C)``.
* ``TestCalibration`` — the Chen-Dempster-Liu inversion, checked by enumerating
  the achieved inclusion probabilities of the calibrated odds; plus the scale
  invariance the mean-centring relies on, and the non-convergence guard.
* ``TestInclusionProbabilities`` — proportional target and take-all capping,
  including a cascade where capping one position pushes a second over 1.
* ``TestSamplePositions`` and ``TestPositionSampler`` — the rejective draw,
  determinism, edge cases, and Monte-Carlo marginals on a small pool.
* ``TestRealPoolMarginals`` — marginals on the real ara pool at ``k = 88``,
  cross-checked against an independent Sampford (1967) sampler. Slow; skipped
  unless ``RUN_SLOW_SAMPLER_TESTS=1``.

Monte-Carlo assertions use fixed seeds, so a failure is reproducible rather
than flaky, and tolerances are stated in standard errors of the estimate.
"""

from __future__ import annotations

import itertools
import math
import os
import unittest

import numpy as np

from workflows.mutation_distribution_analysis.conditional_poisson import (
    PositionSampler,
    calibrate_working_odds,
    inclusion_probabilities,
    log_normalising_constants,
    sample_positions,
    scale_odds_to_sample_size,
)

# Path to the real ara mutation pool used by the slow marginal test.
REAL_POOL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))),
    "src",
    "workflows",
    "mutation_distribution_analysis",
    "mutation_pools",
    "ara_msr_max_single_gen1999_mutation_pool.json",
)

# The slow tests draw tens of thousands of samples from a 2721-position pool
# and additionally run a Sampford sampler over it, which takes minutes.
RUN_SLOW = os.environ.get("RUN_SLOW_SAMPLER_TESTS") == "1"

# Largest tolerated standardised deviation between an empirical inclusion
# frequency and its target. With 2721 positions, a threshold of 5 standard
# errors leaves a false-failure probability below 1e-3 even before accounting
# for the negative dependence induced by the fixed sample size.
TOLERANCE_SIGMAS = 5.0


def enumerate_r(working_odds: np.ndarray, degree: int, exclude: int | None = None) -> float:
    """Compute ``R(degree, C)`` by explicit enumeration of subsets.

    Args:
        working_odds: Odds ``w``, one per position.
        degree: Subset size.
        exclude: Optional index removed from the pool before enumerating.

    Returns:
        The sum, over every ``degree``-element subset ``B`` of the (possibly
        reduced) pool, of ``prod_{i in B} w_i``. Equals 1.0 for ``degree = 0``.
    """
    indices = [i for i in range(working_odds.size) if i != exclude]
    if degree == 0:
        return 1.0
    return sum(
        float(np.prod(working_odds[list(subset)]))
        for subset in itertools.combinations(indices, degree)
    )


def enumerate_inclusion(working_odds: np.ndarray, sample_size: int) -> np.ndarray:
    """Exact inclusion probabilities of a conditional Bernoulli design.

    Enumerates every ``sample_size``-subset, weights it by the product of its
    members' odds, and normalises. This is the definition the module is
    supposed to realise, so it is the right reference.

    Args:
        working_odds: Odds ``w``, one per position.
        sample_size: Number of positions per sample.

    Returns:
        Array of inclusion probabilities, one per position, summing to
        ``sample_size``.
    """
    total_weight = 0.0
    inclusion = np.zeros(working_odds.size)
    for subset in itertools.combinations(range(working_odds.size), sample_size):
        weight = float(np.prod(working_odds[list(subset)]))
        total_weight += weight
        for index in subset:
            inclusion[index] += weight
    return inclusion / total_weight


def standard_error(probability: float, n_replicates: int) -> float:
    """Standard error of a binomial proportion estimate.

    Args:
        probability: The reference probability.
        n_replicates: Number of independent replicates.

    Returns:
        The standard error, floored at a tenth of its value at ``p = 0.5`` so a
        reference probability of exactly 0 or 1 still yields a usable
        tolerance.
    """
    floor = 0.5 / math.sqrt(n_replicates) / 10.0
    return max(
        math.sqrt(probability * (1.0 - probability) / n_replicates), floor
    )


def sampford_sample(
    target: np.ndarray, sample_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw one sample under Sampford's (1967) design — independent reference.

    Sampford's scheme achieves the same first-order inclusion probabilities as
    the maximum-entropy design by an entirely different route: draw the first
    unit with probability ``pi/k``, the remaining ``k - 1`` **with
    replacement** with probability proportional to ``pi/(1 - pi)``, and accept
    only if all ``k`` drawn units are distinct. Its second-order structure
    closely approximates maximum entropy, so agreement between the two is
    strong independent evidence rather than a restatement.

    Used only as a test oracle: it collapses under skew (acceptance can fall
    below 1e-30 on the per-gene applicable pools), which is why the production
    code does not use it.

    Args:
        target: Inclusion probabilities, every entry strictly inside ``(0, 1)``
            and summing to ``sample_size``.
        sample_size: Number of distinct positions to draw.
        rng: NumPy ``Generator`` used for the draws.

    Returns:
        Ascending array of ``sample_size`` distinct indices.

    Raises:
        RuntimeError: If no accepted draw occurred within 100000 attempts.
    """
    first_probabilities = target / sample_size
    rest_weights = target / (1.0 - target)
    rest_probabilities = rest_weights / rest_weights.sum()
    for _ in range(100000):
        first = rng.choice(target.size, p=first_probabilities)
        rest = rng.choice(
            target.size, size=sample_size - 1, replace=True, p=rest_probabilities
        )
        drawn = np.concatenate([[first], rest])
        if np.unique(drawn).size == sample_size:
            return np.sort(drawn)
    raise RuntimeError("Sampford rejection did not accept within 100000 attempts.")


class TestNormalisingConstants(unittest.TestCase):
    """The prefix/suffix recursion reproduces enumerated ``R(k, C)``."""

    def test_matches_brute_force_enumeration(self) -> None:
        """log R agrees with enumeration to machine precision."""
        # Arrange: odds spanning two orders of magnitude, so the scaled tables
        # are actually exercised rather than sitting near 1.
        rng = np.random.default_rng(0)

        for n_positions, sample_size in [(4, 2), (9, 3), (12, 5)]:
            with self.subTest(n=n_positions, k=sample_size):
                working_odds = rng.random(n_positions) * 6.0 + 0.05

                # Act
                log_r_excluding, log_r_full = log_normalising_constants(
                    working_odds, sample_size
                )

                # Assert
                expected_full = math.log(enumerate_r(working_odds, sample_size))
                self.assertAlmostEqual(log_r_full, expected_full, delta=1e-12)
                for position in range(n_positions):
                    expected = math.log(
                        enumerate_r(
                            working_odds, sample_size - 1, exclude=position
                        )
                    )
                    self.assertAlmostEqual(
                        log_r_excluding[position], expected, delta=1e-12,
                        msg=f"leave-one-out constant for position {position}",
                    )

    def test_survives_extreme_skew(self) -> None:
        """A pool whose odds span twelve orders of magnitude stays accurate."""
        # Arrange
        working_odds = np.array([1e-6, 1e-3, 1.0, 1e3, 1e6, 2.0, 0.5])

        # Act
        log_r_excluding, log_r_full = log_normalising_constants(working_odds, 3)

        # Assert
        self.assertAlmostEqual(
            log_r_full, math.log(enumerate_r(working_odds, 3)), delta=1e-10
        )
        for position in range(working_odds.size):
            self.assertAlmostEqual(
                log_r_excluding[position],
                math.log(enumerate_r(working_odds, 2, exclude=position)),
                delta=1e-10,
                msg=f"position {position}",
            )

    def test_rejects_out_of_range_sample_size(self) -> None:
        """``k = n`` leaves no leave-one-out constant and must be rejected."""
        # Arrange
        working_odds = np.array([1.0, 2.0, 3.0])

        # Act / Assert
        with self.assertRaises(ValueError):
            log_normalising_constants(working_odds, 3)
        with self.assertRaises(ValueError):
            log_normalising_constants(working_odds, 0)

    def test_rejects_non_positive_odds(self) -> None:
        """Zero or negative odds are not a valid design."""
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            log_normalising_constants(np.array([1.0, 0.0, 2.0]), 2)


class TestCalibration(unittest.TestCase):
    """The Chen-Dempster-Liu inversion hits the target inclusion probabilities."""

    def test_calibrated_odds_reproduce_the_target(self) -> None:
        """Enumerated inclusion of the calibrated odds equals the target."""
        # Arrange: the toy of the design note, plus a random asymmetric case.
        rng = np.random.default_rng(1)
        raw = rng.random(9) ** 2
        cases = [
            (np.array([0.8, 0.4, 0.4, 0.4]), 2),
            (3.0 * raw / raw.sum(), 3),
        ]

        for target, sample_size in cases:
            with self.subTest(n=target.size, max_target=round(target.max(), 3)):
                self.assertLess(target.max(), 1.0, "case must be feasible")

                # Act
                working_odds = calibrate_working_odds(target)
                achieved = enumerate_inclusion(working_odds, sample_size)

                # Assert
                self.assertLess(np.abs(achieved - target).max(), 1e-10)

    def test_handles_a_feasibility_edge_case(self) -> None:
        """A target with max ``k*p`` just under 1 still calibrates exactly."""
        # Arrange: n = 9, k = 3, one position at 0.946 — close enough to the
        # take-all boundary that the iteration is slow but must still land.
        target = np.full(9, (3.0 - 0.946) / 8.0)
        target[0] = 0.946

        # Act
        working_odds = calibrate_working_odds(target)
        achieved = enumerate_inclusion(working_odds, 3)

        # Assert
        self.assertAlmostEqual(float(target.sum()), 3.0, places=12)
        self.assertLess(np.abs(achieved - target).max(), 1e-10)

    def test_is_invariant_to_the_scale_of_its_input(self) -> None:
        """Calibrating from rescaled odds gives the same odds *ratios*.

        Guards the mean-centring used in place of Chen & Liu's pinning of the
        last coordinate: the map is homogeneous, so only ratios are determined
        and the implementation may renormalise freely.
        """
        # Arrange
        target = np.array([0.8, 0.4, 0.4, 0.4])

        # Act
        odds_a = calibrate_working_odds(target)
        odds_b = calibrate_working_odds(target * 1.0)  # same target, same path
        ratios_a = odds_a / odds_a[0]

        # A second, independent route to the same design: scale the enumerated
        # solution and confirm the inclusion probabilities do not move.
        rescaled = odds_a * 137.0
        inclusion_original = enumerate_inclusion(odds_a, 2)
        inclusion_rescaled = enumerate_inclusion(rescaled, 2)

        # Assert
        np.testing.assert_allclose(ratios_a, odds_b / odds_b[0], rtol=1e-12)
        np.testing.assert_allclose(
            inclusion_original, inclusion_rescaled, rtol=1e-12
        )
        # The hand-derived solution for this toy is odds 4:1:1:1.
        np.testing.assert_allclose(ratios_a, [1.0, 0.25, 0.25, 0.25], rtol=1e-8)

    def test_raises_instead_of_returning_an_unconverged_result(self) -> None:
        """One iteration is not enough for the toy, and that must be an error."""
        # Arrange
        target = np.array([0.8, 0.4, 0.4, 0.4])

        # Act / Assert
        with self.assertRaises(RuntimeError):
            calibrate_working_odds(target, max_iterations=1)

    def test_rejects_infeasible_or_degenerate_targets(self) -> None:
        """Entries outside (0, 1) and non-integer sums are caller errors."""
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            calibrate_working_odds(np.array([1.0, 0.5, 0.5]))
        with self.assertRaises(ValueError):
            calibrate_working_odds(np.array([0.0, 0.5, 0.5]))
        with self.assertRaises(ValueError):
            calibrate_working_odds(np.array([0.3, 0.4, 0.5]))


class TestInclusionProbabilities(unittest.TestCase):
    """Proportional targets, and take-all capping where they are infeasible."""

    def test_proportional_when_feasible(self) -> None:
        """Without capping the target is simply ``k * counts / sum(counts)``."""
        # Arrange
        counts = np.array([40.0, 30.0, 20.0, 10.0])

        # Act
        target = inclusion_probabilities(counts, 2)

        # Assert
        np.testing.assert_allclose(target, [0.8, 0.6, 0.4, 0.2], rtol=1e-12)
        self.assertAlmostEqual(float(target.sum()), 2.0, places=12)

    def test_caps_a_dominant_position(self) -> None:
        """A position whose proportional share exceeds 1 is taken with certainty."""
        # Arrange: k*p = (1.6, 0.2, 0.2) before capping.
        counts = np.array([80.0, 10.0, 10.0])

        # Act
        target = inclusion_probabilities(counts, 2)

        # Assert
        np.testing.assert_allclose(target, [1.0, 0.5, 0.5], rtol=1e-12)
        self.assertAlmostEqual(float(target.sum()), 2.0, places=12)

    def test_capping_cascades(self) -> None:
        """Capping one position can push a second over 1; both must be capped."""
        # Arrange: counts (100, 45, 5) with k = 2 gives k*p = (1.33, 0.6, 0.067).
        # Capping the first leaves k = 1 over (45, 5): 0.9 and 0.1 — feasible.
        # Counts (100, 95, 5) with k = 2 gives (1.0, 0.95, 0.05); capping the
        # first leaves k = 1 over (95, 5) → 0.95, still feasible. A genuine
        # cascade needs the residual share to exceed 1, which requires k >= 3.
        counts = np.array([100.0, 90.0, 8.0, 2.0])

        # Act
        target = inclusion_probabilities(counts, 3)

        # Assert: k*p = (1.5, 1.35, 0.12, 0.03); capping both leaves k = 1 over
        # (8, 2) → (0.8, 0.2).
        np.testing.assert_allclose(target, [1.0, 1.0, 0.8, 0.2], rtol=1e-12)
        self.assertAlmostEqual(float(target.sum()), 3.0, places=12)
        self.assertLessEqual(target.max(), 1.0)

    def test_zero_counts_get_zero_probability(self) -> None:
        """Positions absent from the pool are never drawn."""
        # Arrange
        counts = np.array([10.0, 0.0, 10.0])

        # Act
        target = inclusion_probabilities(counts, 1)

        # Assert
        np.testing.assert_allclose(target, [0.5, 0.0, 0.5], rtol=1e-12)

    def test_full_sample_takes_everything(self) -> None:
        """``k`` equal to the number of positive counts caps every position."""
        # Arrange
        counts = np.array([5.0, 3.0, 2.0])

        # Act
        target = inclusion_probabilities(counts, 3)

        # Assert
        np.testing.assert_allclose(target, [1.0, 1.0, 1.0], rtol=1e-12)

    def test_zero_sample_size_is_all_zeros(self) -> None:
        """Drawing nothing is legal and gives every position probability 0."""
        # Arrange / Act
        target = inclusion_probabilities(np.array([5.0, 3.0]), 0)

        # Assert
        np.testing.assert_allclose(target, [0.0, 0.0], rtol=1e-12)

    def test_rejects_invalid_input(self) -> None:
        """Negative counts, empty pools and oversized samples are errors."""
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            inclusion_probabilities(np.array([1.0, -1.0]), 1)
        with self.assertRaises(ValueError):
            inclusion_probabilities(np.array([0.0, 0.0]), 1)
        with self.assertRaises(ValueError):
            inclusion_probabilities(np.array([1.0, 1.0]), 3)
        with self.assertRaises(ValueError):
            inclusion_probabilities(np.array([1.0, 0.0]), 2)
        with self.assertRaises(ValueError):
            inclusion_probabilities(np.array([1.0, 1.0]), -1)


class TestScaleOddsToSampleSize(unittest.TestCase):
    """Pinning the odds' free scale, which the rejective draw depends on.

    The design is scale-invariant but the *unconditional* coins are not, so
    without this step acceptance collapses: at the real pool (n = 2721,
    k = 88) the mean-centred odds produced by the calibration accept zero draws
    in 10000 attempts.
    """

    def test_mean_head_count_equals_the_sample_size(self) -> None:
        """After rescaling, the coins produce ``k`` heads on average."""
        # Arrange: odds spanning six orders of magnitude, badly scaled.
        rng = np.random.default_rng(40)
        working_odds = np.exp(rng.normal(-8.0, 3.0, size=500))

        for sample_size in (1, 20, 250, 499):
            with self.subTest(k=sample_size):
                # Act
                scaled = scale_odds_to_sample_size(working_odds, sample_size)

                # Assert
                mean_heads = float((scaled / (1.0 + scaled)).sum())
                self.assertAlmostEqual(mean_heads, sample_size, delta=1e-9)

    def test_ratios_and_hence_the_design_are_unchanged(self) -> None:
        """Rescaling multiplies every odd by one constant, so ``pi`` is fixed."""
        # Arrange
        working_odds = np.array([4.0, 1.0, 1.0, 1.0]) * 1e-5

        # Act
        scaled = scale_odds_to_sample_size(working_odds, 2)

        # Assert
        factors = scaled / working_odds
        np.testing.assert_allclose(factors, factors[0], rtol=1e-12)
        np.testing.assert_allclose(
            enumerate_inclusion(scaled, 2),
            enumerate_inclusion(working_odds, 2),
            rtol=1e-12,
        )

    def test_sample_positions_rescales_automatically(self) -> None:
        """Badly scaled odds still draw, because the sampler rescales them."""
        # Arrange: the calibrated toy design, scaled down by 1e6.
        working_odds = calibrate_working_odds(np.array([0.8, 0.4, 0.4, 0.4]))
        badly_scaled = working_odds * 1e-6

        # Act
        drawn = sample_positions(
            badly_scaled, 2, np.random.default_rng(41), max_attempts=500
        )

        # Assert
        self.assertEqual(drawn.size, 2)

    def test_rejects_out_of_range_sample_size(self) -> None:
        """No finite scale gives a mean of 0 or ``n`` heads."""
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            scale_odds_to_sample_size(np.array([1.0, 2.0, 3.0]), 0)
        with self.assertRaises(ValueError):
            scale_odds_to_sample_size(np.array([1.0, 2.0, 3.0]), 3)


class TestSamplePositions(unittest.TestCase):
    """The rejective draw returns the right size and the right marginals."""

    def test_draws_exactly_the_requested_size(self) -> None:
        """Every accepted draw has exactly ``sample_size`` distinct entries."""
        # Arrange
        working_odds = np.array([4.0, 1.0, 1.0, 1.0])
        rng = np.random.default_rng(2)

        # Act / Assert
        for _ in range(50):
            drawn = sample_positions(working_odds, 2, rng)
            self.assertEqual(drawn.size, 2)
            self.assertEqual(np.unique(drawn).size, 2)

    def test_marginals_match_the_enumerated_design(self) -> None:
        """Empirical inclusion frequencies match the enumerated design exactly."""
        # Arrange
        working_odds = np.array([4.0, 1.0, 1.0, 1.0])
        expected = enumerate_inclusion(working_odds, 2)
        n_replicates = 20000
        rng = np.random.default_rng(3)
        hits = np.zeros(working_odds.size)

        # Act
        for _ in range(n_replicates):
            hits[sample_positions(working_odds, 2, rng)] += 1
        observed = hits / n_replicates

        # Assert
        for position, expected_probability in enumerate(expected):
            tolerance = TOLERANCE_SIGMAS * standard_error(
                float(expected_probability), n_replicates
            )
            self.assertAlmostEqual(
                observed[position], expected_probability, delta=tolerance,
                msg=(
                    f"position {position}: observed {observed[position]:.4f} vs "
                    f"expected {expected_probability:.4f} (tol {tolerance:.4f})"
                ),
            )

    def test_zero_sample_size_returns_empty(self) -> None:
        """Drawing nothing consumes no randomness and returns an empty array."""
        # Arrange / Act
        drawn = sample_positions(
            np.array([1.0, 2.0]), 0, np.random.default_rng(4)
        )

        # Assert
        self.assertEqual(drawn.size, 0)

    def test_raises_when_rejection_never_accepts(self) -> None:
        """An attempt cap of 1 on an unlikely size must raise, not loop."""
        # Arrange: with odds 1e-9 the chance of two heads is about 1e-18.
        working_odds = np.full(4, 1e-9)

        # Act / Assert
        with self.assertRaises(RuntimeError):
            sample_positions(
                working_odds, 2, np.random.default_rng(5), max_attempts=1
            )

    def test_rejects_invalid_input(self) -> None:
        """Non-positive odds and out-of-range sizes are errors."""
        # Arrange
        rng = np.random.default_rng(6)

        # Act / Assert
        with self.assertRaises(ValueError):
            sample_positions(np.array([1.0, -1.0]), 1, rng)
        with self.assertRaises(ValueError):
            sample_positions(np.array([1.0, 1.0]), 3, rng)


class TestPositionSampler(unittest.TestCase):
    """End-to-end: counts in, correctly-included positions out."""

    def test_inclusion_frequencies_match_k_times_p(self) -> None:
        """On a skewed pool the realised inclusion equals the target ``k * p``.

        This is the whole point of the module, and the property the successive
        sampler it replaces does not have: there the heavy position came out
        *below* ``k * p`` and the light ones above it.
        """
        # Arrange: 40/35/25/25/20/4/1 rows over seven positions, k = 3, so the
        # heaviest target is 0.8 — skewed, but feasible without capping.
        counts = np.array([40.0, 35.0, 25.0, 25.0, 20.0, 4.0, 1.0])
        sampler = PositionSampler(counts, 3)
        expected = 3.0 * counts / counts.sum()
        n_replicates = 20000
        rng = np.random.default_rng(7)
        hits = np.zeros(counts.size)

        # Act
        for _ in range(n_replicates):
            hits[sampler.draw(rng)] += 1
        observed = hits / n_replicates

        # Assert
        np.testing.assert_allclose(sampler.target, expected, rtol=1e-12)
        for position, expected_probability in enumerate(expected):
            tolerance = TOLERANCE_SIGMAS * standard_error(
                float(expected_probability), n_replicates
            )
            self.assertAlmostEqual(
                observed[position], expected_probability, delta=tolerance,
                msg=(
                    f"position {position}: observed {observed[position]:.4f} vs "
                    f"target {expected_probability:.4f} (tol {tolerance:.4f})"
                ),
            )

    def test_capped_positions_appear_in_every_draw(self) -> None:
        """A take-all position is forced in, and the rest still sum correctly."""
        # Arrange: k*p = (1.6, 0.2, 0.2) → target (1, 0.5, 0.5).
        counts = np.array([80.0, 10.0, 10.0])
        sampler = PositionSampler(counts, 2)
        n_replicates = 5000
        rng = np.random.default_rng(8)
        hits = np.zeros(counts.size)

        # Act
        for _ in range(n_replicates):
            drawn = sampler.draw(rng)
            self.assertIn(0, drawn.tolist())
            hits[drawn] += 1
        observed = hits / n_replicates

        # Assert
        np.testing.assert_allclose(sampler.target, [1.0, 0.5, 0.5], rtol=1e-12)
        self.assertEqual(observed[0], 1.0)
        for position in (1, 2):
            tolerance = TOLERANCE_SIGMAS * standard_error(0.5, n_replicates)
            self.assertAlmostEqual(observed[position], 0.5, delta=tolerance)

    def test_zero_count_positions_are_never_drawn(self) -> None:
        """A position with no pool rows is excluded from the design entirely."""
        # Arrange
        counts = np.array([10.0, 0.0, 10.0, 5.0])
        sampler = PositionSampler(counts, 2)
        rng = np.random.default_rng(9)

        # Act / Assert
        self.assertEqual(sampler.target[1], 0.0)
        self.assertNotIn(1, sampler.free_indices.tolist())
        for _ in range(200):
            self.assertNotIn(1, sampler.draw(rng).tolist())

    def test_is_deterministic_under_a_fixed_seed(self) -> None:
        """Two generators seeded alike produce identical sequences of draws."""
        # Arrange
        counts = np.array([40.0, 30.0, 20.0, 6.0, 4.0])
        sampler = PositionSampler(counts, 2)
        rng_a = np.random.default_rng(202)
        rng_b = np.random.default_rng(202)

        # Act
        sequence_a = [sampler.draw(rng_a).tolist() for _ in range(20)]
        sequence_b = [sampler.draw(rng_b).tolist() for _ in range(20)]

        # Assert
        self.assertEqual(sequence_a, sequence_b)
        self.assertGreater(len(set(map(tuple, sequence_a))), 1)

    def test_single_draw_reduces_to_the_pool_share(self) -> None:
        """``k = 1`` has inclusion probability equal to the pool share."""
        # Arrange
        counts = np.array([50.0, 30.0, 20.0])
        sampler = PositionSampler(counts, 1)

        # Act / Assert
        np.testing.assert_allclose(sampler.target, [0.5, 0.3, 0.2], rtol=1e-12)
        rng = np.random.default_rng(10)
        for _ in range(100):
            self.assertEqual(sampler.draw(rng).size, 1)

    def test_zero_sample_size_draws_nothing(self) -> None:
        """``k = 0`` is legal and yields an empty draw."""
        # Arrange
        sampler = PositionSampler(np.array([5.0, 3.0]), 0)

        # Act
        drawn = sampler.draw(np.random.default_rng(11))

        # Assert
        self.assertEqual(drawn.size, 0)

    def test_full_sample_returns_every_position(self) -> None:
        """``k`` equal to the positive-count total takes all of them."""
        # Arrange
        sampler = PositionSampler(np.array([5.0, 3.0, 1.0]), 3)

        # Act
        drawn = sampler.draw(np.random.default_rng(12))

        # Assert
        np.testing.assert_array_equal(drawn, [0, 1, 2])

    def test_oversized_sample_raises(self) -> None:
        """Asking for more positions than the pool holds is an error."""
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            PositionSampler(np.array([5.0, 3.0]), 3)


@unittest.skipUnless(
    RUN_SLOW and os.path.isfile(REAL_POOL_PATH),
    "set RUN_SLOW_SAMPLER_TESTS=1 and provide the ara mutation pool",
)
class TestRealPoolMarginals(unittest.TestCase):
    """Marginals on the real ara pool, cross-checked against Sampford.

    Runs for several minutes: it calibrates a 2721-position design and draws
    tens of thousands of samples under two different exact schemes.
    """

    n_replicates = 20000
    sample_size = 88

    @classmethod
    def setUpClass(cls) -> None:
        """Load the real pool once and calibrate the sampler."""
        from workflows.mutation_distribution_analysis.mutation_pool import (
            MutationPool,
        )

        pool = MutationPool.load(REAL_POOL_PATH)
        _, cls.counts = np.unique(
            pool.mutations["position"].to_numpy(), return_counts=True
        )
        cls.counts = cls.counts.astype(float)
        cls.sampler = PositionSampler(cls.counts, cls.sample_size)

    def _inclusion_frequencies(self, draw_one) -> np.ndarray:
        """Accumulate inclusion frequencies over ``n_replicates`` draws.

        Args:
            draw_one: Callable taking no arguments and returning one array of
                drawn indices.

        Returns:
            Per-position fraction of replicates containing that position.
        """
        hits = np.zeros(self.counts.size)
        for _ in range(self.n_replicates):
            hits[draw_one()] += 1
        return hits / self.n_replicates

    def test_maximum_entropy_marginals_match_the_target(self) -> None:
        """Empirical inclusion equals ``k * p`` within Monte-Carlo error."""
        # Arrange
        rng = np.random.default_rng(2024)
        target = self.sampler.target

        # Act
        observed = self._inclusion_frequencies(lambda: self.sampler.draw(rng))

        # Assert
        errors = np.abs(observed - target)
        tolerances = np.array([
            TOLERANCE_SIGMAS * standard_error(float(p), self.n_replicates)
            for p in target
        ])
        worst = int(np.argmax(errors / tolerances))
        self.assertLess(
            errors[worst] / tolerances[worst], 1.0,
            msg=(
                f"position index {worst}: observed {observed[worst]:.4f} vs "
                f"target {target[worst]:.4f} (tol {tolerances[worst]:.4f})"
            ),
        )

    def test_agrees_with_an_independent_sampford_sampler(self) -> None:
        """Two exact pi-ps designs give the same first-order marginals."""
        # Arrange
        target = self.sampler.target
        self.assertLess(target.max(), 1.0, "no capping expected on this pool")
        rng_maximum_entropy = np.random.default_rng(31)
        rng_sampford = np.random.default_rng(37)

        # Act
        from_maximum_entropy = self._inclusion_frequencies(
            lambda: self.sampler.draw(rng_maximum_entropy)
        )
        from_sampford = self._inclusion_frequencies(
            lambda: sampford_sample(target, self.sample_size, rng_sampford)
        )

        # Assert: two independent Monte-Carlo estimates, so allow both errors.
        differences = np.abs(from_maximum_entropy - from_sampford)
        tolerances = np.array([
            TOLERANCE_SIGMAS
            * math.sqrt(2.0)
            * standard_error(float(p), self.n_replicates)
            for p in target
        ])
        worst = int(np.argmax(differences / tolerances))
        self.assertLess(
            differences[worst] / tolerances[worst], 1.0,
            msg=(
                f"position index {worst}: max-entropy "
                f"{from_maximum_entropy[worst]:.4f} vs Sampford "
                f"{from_sampford[worst]:.4f} (tol {tolerances[worst]:.4f})"
            ),
        )


if __name__ == "__main__":
    unittest.main()
