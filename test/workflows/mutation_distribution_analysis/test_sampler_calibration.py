"""**Historical record of a fixed bug — not a specification.**

Nothing in this module describes how the samplers behave today. It documents
the *successive-sampling* scheme that both call sites used until the switch to
the maximum-entropy design, and the bias that motivated replacing it. The
current contract lives in ``test_conditional_poisson.py``; the analysis lives in
``src/workflows/mutation_distribution_analysis/plans/sampler_inclusion_probability_fix.md``.

The removed implementations are reproduced verbatim below as
:func:`successive_sample_rows` and :func:`successive_sample_positions`, so this
file stays self-contained and cannot be mistaken for a test of production code.
Reading it as a description of ``baseline_sampler`` or
``mutation_distance_analysis`` would be a mistake.

What the record establishes:

1. **The empirical joint was reproduced.** A single uniform-over-rows draw
   reproduces the pool's ``(position, source_base, new_base)`` distribution.
   This part survives the fix — the substitution conditional is unchanged, and
   the new sampler relies on exactly this factorisation.

2. **The two old samplers agreed with each other.** They were the same
   algorithm written twice, so their agreement was never independent
   corroboration of correctness — only of consistency. Recorded because it was
   at the time mistaken for the former.

3. **Position blocking does *not* preserve the positional PMF for k > 1.** The
   old docstring claimed it "preserves the empirical positional PMF". That
   holds for the first draw only. Over ``k`` draws the scheme is successive
   sampling, whose per-position **inclusion** probability is not proportional
   to the pool share: heavy positions come out below ``k * p`` and light ones
   above it. The null was therefore *flatter* than the optimiser's positional
   prior, biasing the real-vs-random comparison toward making the real
   mutations look more positionally concentrated than they are.

4. **The fix caps what could not be requested.** A closing test shows the
   replacement handling the same skewed pool, where the proportional target
   exceeds 1 and take-all capping is the only feasible answer.

Monte-Carlo assertions run under fixed seeds, so a failure is reproducible rather
than flaky, and tolerances are stated in standard errors of the estimate.
"""

from __future__ import annotations

import itertools
import math
import unittest
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from workflows.mutation_distribution_analysis.conditional_poisson import (
    PositionSampler,
)
from workflows.mutation_distribution_analysis.mutation_pool import (
    MUTATION_COLUMNS,
)


def successive_sample_rows(
    applicable: pd.DataFrame, k: int, rng: np.random.Generator
) -> List[Tuple[int, str, str]]:
    """The removed ``baseline_sampler._sample_mutations_blocking``, verbatim.

    Draws one pool row uniformly, records its ``(position, source_base,
    new_base)``, removes every row sharing that position, and repeats.

    Args:
        applicable: Pool rows applicable to the target wildtype.
        k: Number of mutations to draw.
        rng: NumPy ``Generator`` used for index sampling.

    Returns:
        ``k`` tuples in draw order, with distinct positions.

    Raises:
        ValueError: If ``applicable`` holds fewer than ``k`` unique positions.
    """
    if k == 0:
        return []

    unique_position_count = applicable["position"].nunique()
    if unique_position_count < k:
        raise ValueError(
            f"Cannot draw {k} position-blocking mutations: only "
            f"{unique_position_count} unique applicable positions available."
        )

    working = applicable
    result: List[Tuple[int, str, str]] = []
    for _ in range(k):
        index = int(rng.integers(0, len(working)))
        row = working.iloc[index]
        position = int(row["position"])
        result.append((position, str(row["source_base"]), str(row["new_base"])))
        working = working.loc[working["position"] != position]
    return result


def successive_sample_positions(
    unique_positions: np.ndarray,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """The removed ``mutation_distance_analysis._sample_positions_blocking``.

    ``rng.choice(..., replace=False, p=...)`` is itself implemented as
    successive sampling, which is why this and :func:`successive_sample_rows`
    were the same algorithm rather than two independent ones.

    Args:
        unique_positions: Sorted distinct positions from the mutation pool.
        weights: Probability vector aligned with ``unique_positions``.
        k: Number of distinct positions to draw.
        rng: NumPy ``Generator`` used for the draw.

    Returns:
        ``k`` distinct positions.

    Raises:
        ValueError: If ``k`` exceeds the number of unique positions.
    """
    if k > unique_positions.size:
        raise ValueError(
            f"Cannot draw {k} position-blocking mutations: only "
            f"{unique_positions.size} unique applicable positions available."
        )
    return rng.choice(unique_positions, size=k, replace=False, p=weights)


# Replicate count for the Monte-Carlo estimates. Chosen so that the standard
# error of an inclusion frequency near 0.5 is about 0.005, keeping a 5-sigma
# tolerance comfortably tight while the per-draw pandas filtering stays fast.
N_REPLICATES = 10000

# Assertion width, in standard errors of the Monte-Carlo estimate.
TOLERANCE_SIGMAS = 5.0


def _mutations_frame(rows: Sequence[Tuple[int, str, str]]) -> pd.DataFrame:
    """Build a pool-shaped mutations frame from ``(position, src, new)`` rows.

    Args:
        rows: One tuple per pool row. Repeating a tuple raises that combination's
            multiplicity, which is exactly how the pool encodes its density.

    Returns:
        A DataFrame with the canonical ``MUTATION_COLUMNS``.
    """
    records = [
        {
            "gene_id": "g1",
            "position": position,
            "source_base": source_base,
            "new_base": new_base,
        }
        for position, source_base, new_base in rows
    ]
    return pd.DataFrame(records, columns=MUTATION_COLUMNS)


def _rows_from_counts(counts: Dict[int, int]) -> List[Tuple[int, str, str]]:
    """Expand ``{position: row_count}`` into pool rows with a fixed substitution.

    Args:
        counts: Row count per position.

    Returns:
        Pool rows, all carrying the same ``A -> T`` substitution so that only the
        positional structure varies.
    """
    rows: List[Tuple[int, str, str]] = []
    for position, count in counts.items():
        rows.extend([(position, "A", "T")] * count)
    return rows


def _exact_successive_inclusion(
    counts: Dict[int, int], k: int
) -> Dict[int, float]:
    """Exact per-position inclusion probabilities under successive sampling.

    Enumerates every ordered draw sequence of ``k`` distinct positions. The
    probability of a sequence is the product, over steps, of the drawn position's
    row count divided by the row count still available at that step — which is
    precisely uniform-over-rows with position blocking. Tractable only for tiny
    pools, which is why it is used as a reference rather than as an implementation.

    Args:
        counts: Row count per position.
        k: Number of distinct positions drawn.

    Returns:
        Mapping of position to the probability that it appears in the sample.

    Raises:
        ValueError: If ``k`` exceeds the number of positions.
    """
    positions = sorted(counts)
    if k > len(positions):
        raise ValueError(f"k={k} exceeds {len(positions)} available positions.")

    total = sum(counts.values())
    inclusion = {position: 0.0 for position in positions}
    for ordered_draw in itertools.permutations(positions, k):
        probability = 1.0
        remaining = total
        for position in ordered_draw:
            probability *= counts[position] / remaining
            remaining -= counts[position]
        for position in ordered_draw:
            inclusion[position] += probability
    return inclusion


def _standard_error(probability: float, n_replicates: int) -> float:
    """Standard error of a binomial proportion estimate.

    Args:
        probability: The true (or reference) probability.
        n_replicates: Number of independent replicates.

    Returns:
        The standard error, floored at the value for p = 0.5 divided by 10 so a
        reference probability of exactly 0 or 1 still yields a usable tolerance.
    """
    floor = 0.5 / math.sqrt(n_replicates) / 10.0
    return max(math.sqrt(probability * (1.0 - probability) / n_replicates), floor)


def _inclusion_from_row_sampler(
    counts: Dict[int, int], k: int, seed: int, n_replicates: int
) -> Dict[int, float]:
    """Empirical inclusion frequencies from ``successive_sample_rows``.

    Args:
        counts: Row count per position.
        k: Number of mutations drawn per replicate.
        seed: Seed for the NumPy generator.
        n_replicates: Number of replicates.

    Returns:
        Mapping of position to the fraction of replicates containing it.
    """
    applicable = _mutations_frame(_rows_from_counts(counts))
    rng = np.random.default_rng(seed)
    hits = {position: 0 for position in counts}
    for _ in range(n_replicates):
        drawn = successive_sample_rows(applicable, k, rng)
        for position, *_substitution in drawn:
            hits[position] += 1
    return {
        position: hit_count / n_replicates for position, hit_count in hits.items()
    }


def _inclusion_from_position_sampler(
    counts: Dict[int, int], k: int, seed: int, n_replicates: int
) -> Dict[int, float]:
    """Empirical inclusion frequencies from ``successive_sample_positions``.

    Args:
        counts: Row count per position.
        k: Number of positions drawn per replicate.
        seed: Seed for the NumPy generator.
        n_replicates: Number of replicates.

    Returns:
        Mapping of position to the fraction of replicates containing it.
    """
    positions = np.array(sorted(counts))
    weights = np.array([counts[position] for position in positions], dtype=float)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    hits = {int(position): 0 for position in positions}
    for _ in range(n_replicates):
        drawn = successive_sample_positions(positions, weights, k, rng)
        for position in drawn:
            hits[int(position)] += 1
    return {
        position: hit_count / n_replicates for position, hit_count in hits.items()
    }


# A deliberately skewed pool: one dominant position and two rare ones, which is
# the regime where successive sampling departs most from proportionality.
SKEWED_COUNTS: Dict[int, int] = {10: 80, 20: 10, 30: 10}


class TestEmpiricalJointIsReproduced(unittest.TestCase):
    """A single uniform-over-rows draw reproduces the pool's joint distribution."""

    def test_single_draw_matches_triple_frequencies(self) -> None:
        """P(position, source, new) equals that triple's share of pool rows."""
        # Arrange: a pool where three distinct triples have 5:3:2 multiplicity,
        # including two different substitutions at the same position.
        rows = (
            [(10, "A", "T")] * 5 + [(10, "A", "G")] * 3 + [(20, "C", "T")] * 2
        )
        applicable = _mutations_frame(rows)
        expected = {
            (10, "A", "T"): 0.5,
            (10, "A", "G"): 0.3,
            (20, "C", "T"): 0.2,
        }
        rng = np.random.default_rng(20240101)
        hits: Dict[Tuple[int, str, str], int] = {key: 0 for key in expected}

        # Act
        for _ in range(N_REPLICATES):
            (drawn,) = successive_sample_rows(applicable, 1, rng)
            hits[drawn] += 1

        # Assert
        for triple, expected_probability in expected.items():
            observed = hits[triple] / N_REPLICATES
            tolerance = TOLERANCE_SIGMAS * _standard_error(
                expected_probability, N_REPLICATES
            )
            self.assertAlmostEqual(
                observed,
                expected_probability,
                delta=tolerance,
                msg=(
                    f"triple {triple}: observed {observed:.4f} vs expected "
                    f"{expected_probability:.4f} (tol {tolerance:.4f})"
                ),
            )

    def test_first_draw_marginal_matches_positional_pmf(self) -> None:
        """With k=1 the positional marginal is exactly the pool share."""
        # Arrange
        expected = {
            position: count / sum(SKEWED_COUNTS.values())
            for position, count in SKEWED_COUNTS.items()
        }

        # Act
        observed = _inclusion_from_row_sampler(
            SKEWED_COUNTS, k=1, seed=7, n_replicates=N_REPLICATES
        )

        # Assert
        for position, expected_probability in expected.items():
            tolerance = TOLERANCE_SIGMAS * _standard_error(
                expected_probability, N_REPLICATES
            )
            self.assertAlmostEqual(
                observed[position],
                expected_probability,
                delta=tolerance,
                msg=f"position {position}",
            )


class TestSuccessiveSamplingMatchesClosedForm(unittest.TestCase):
    """The k>1 behaviour matches exact successive-sampling probabilities."""

    def test_row_sampler_matches_exact_inclusion(self) -> None:
        """Empirical inclusion equals the enumerated closed form for k=2."""
        # Arrange
        exact = _exact_successive_inclusion(SKEWED_COUNTS, k=2)

        # Act
        observed = _inclusion_from_row_sampler(
            SKEWED_COUNTS, k=2, seed=11, n_replicates=N_REPLICATES
        )

        # Assert
        for position, exact_probability in exact.items():
            tolerance = TOLERANCE_SIGMAS * _standard_error(
                exact_probability, N_REPLICATES
            )
            self.assertAlmostEqual(
                observed[position],
                exact_probability,
                delta=tolerance,
                msg=(
                    f"position {position}: observed {observed[position]:.4f} vs "
                    f"exact {exact_probability:.4f} (tol {tolerance:.4f})"
                ),
            )

    def test_exact_inclusion_sums_to_k(self) -> None:
        """Inclusion probabilities of a size-k sample sum to k."""
        # Arrange / Act
        exact = _exact_successive_inclusion(SKEWED_COUNTS, k=2)

        # Assert
        self.assertAlmostEqual(sum(exact.values()), 2.0, places=12)


class TestPositionBlockingDoesNotPreserveMarginalPmf(unittest.TestCase):
    """Document that inclusion is not proportional to the pool share for k>1.

    This contradicts ``successive_sample_rows``'s docstring claim that position
    blocking "preserves the empirical positional PMF". The claim is true only for
    k = 1 (covered above); these tests pin down what actually happens for k > 1.
    """

    def test_heavy_position_is_under_included(self) -> None:
        """The dominant position falls short of its proportional share k*p."""
        # Arrange
        total = sum(SKEWED_COUNTS.values())
        heavy_position = max(SKEWED_COUNTS, key=lambda key: SKEWED_COUNTS[key])
        proportional = 2 * SKEWED_COUNTS[heavy_position] / total

        # Act
        exact = _exact_successive_inclusion(SKEWED_COUNTS, k=2)

        # Assert
        self.assertLess(exact[heavy_position], proportional)
        # An inclusion probability is a probability; k*p here is not, which is the
        # structural reason proportionality cannot hold.
        self.assertGreater(proportional, 1.0)
        self.assertLessEqual(exact[heavy_position], 1.0)

    def test_light_positions_are_over_included(self) -> None:
        """Rare positions exceed their proportional share k*p."""
        # Arrange
        total = sum(SKEWED_COUNTS.values())
        heavy_position = max(SKEWED_COUNTS, key=lambda key: SKEWED_COUNTS[key])
        light_positions = [
            position for position in SKEWED_COUNTS if position != heavy_position
        ]

        # Act
        exact = _exact_successive_inclusion(SKEWED_COUNTS, k=2)

        # Assert
        for position in light_positions:
            proportional = 2 * SKEWED_COUNTS[position] / total
            self.assertGreater(
                exact[position],
                proportional,
                msg=(
                    f"position {position}: inclusion {exact[position]:.4f} "
                    f"should exceed proportional {proportional:.4f}"
                ),
            )

    def test_departure_vanishes_for_a_flat_pool(self) -> None:
        """With equal row counts, inclusion is exactly proportional."""
        # Arrange
        flat_counts = {10: 25, 20: 25, 30: 25, 40: 25}

        # Act
        exact = _exact_successive_inclusion(flat_counts, k=2)

        # Assert
        for position in flat_counts:
            self.assertAlmostEqual(exact[position], 2 / 4, places=12)


class TestTwoSamplersAreEquivalent(unittest.TestCase):
    """Cross-check the equivalence asserted by ``successive_sample_positions``."""

    def test_position_sampler_matches_exact_inclusion(self) -> None:
        """The weighted-without-replacement draw follows successive sampling."""
        # Arrange
        exact = _exact_successive_inclusion(SKEWED_COUNTS, k=2)

        # Act
        observed = _inclusion_from_position_sampler(
            SKEWED_COUNTS, k=2, seed=13, n_replicates=N_REPLICATES
        )

        # Assert
        for position, exact_probability in exact.items():
            tolerance = TOLERANCE_SIGMAS * _standard_error(
                exact_probability, N_REPLICATES
            )
            self.assertAlmostEqual(
                observed[position],
                exact_probability,
                delta=tolerance,
                msg=(
                    f"position {position}: observed {observed[position]:.4f} vs "
                    f"exact {exact_probability:.4f} (tol {tolerance:.4f})"
                ),
            )

    def test_both_samplers_agree_with_each_other(self) -> None:
        """The two independent implementations produce the same distribution."""
        # Arrange / Act
        from_rows = _inclusion_from_row_sampler(
            SKEWED_COUNTS, k=2, seed=17, n_replicates=N_REPLICATES
        )
        from_positions = _inclusion_from_position_sampler(
            SKEWED_COUNTS, k=2, seed=19, n_replicates=N_REPLICATES
        )

        # Assert: two independent Monte-Carlo estimates, so allow both errors.
        for position in SKEWED_COUNTS:
            tolerance = (
                TOLERANCE_SIGMAS
                * math.sqrt(2.0)
                * _standard_error(from_rows[position], N_REPLICATES)
            )
            self.assertAlmostEqual(
                from_rows[position],
                from_positions[position],
                delta=tolerance,
                msg=(
                    f"position {position}: rows {from_rows[position]:.4f} vs "
                    f"positions {from_positions[position]:.4f} "
                    f"(tol {tolerance:.4f})"
                ),
            )

    def test_agreement_holds_for_a_larger_k(self) -> None:
        """Equivalence is not an artefact of k=2."""
        # Arrange
        counts = {10: 40, 20: 30, 30: 20, 40: 6, 50: 4}
        exact = _exact_successive_inclusion(counts, k=3)

        # Act
        from_rows = _inclusion_from_row_sampler(
            counts, k=3, seed=23, n_replicates=N_REPLICATES
        )
        from_positions = _inclusion_from_position_sampler(
            counts, k=3, seed=29, n_replicates=N_REPLICATES
        )

        # Assert
        for position, exact_probability in exact.items():
            tolerance = TOLERANCE_SIGMAS * _standard_error(
                exact_probability, N_REPLICATES
            )
            self.assertAlmostEqual(
                from_rows[position], exact_probability, delta=tolerance,
                msg=f"row sampler, position {position}",
            )
            self.assertAlmostEqual(
                from_positions[position], exact_probability, delta=tolerance,
                msg=f"position sampler, position {position}",
            )


class TestReplacementOnTheSamePool(unittest.TestCase):
    """What the maximum-entropy sampler does with the pool that broke the old one.

    ``SKEWED_COUNTS`` at ``k = 2`` asks for ``k * p = 1.6`` at the heavy
    position — not a probability, so no design can deliver it. Successive
    sampling responded by silently landing at 0.700 and pushing the surplus
    onto the light positions. The replacement resolves it explicitly: the heavy
    position is taken with certainty and the residual sample size is
    re-proportioned over the rest.
    """

    def test_infeasible_target_is_capped_rather_than_absorbed(self) -> None:
        """The heavy position is taken always; the light ones split the rest."""
        # Arrange
        positions = sorted(SKEWED_COUNTS)
        counts = np.array([SKEWED_COUNTS[p] for p in positions], dtype=float)
        sampler = PositionSampler(counts, 2)
        rng = np.random.default_rng(4711)
        hits = np.zeros(counts.size)

        # Act
        for _ in range(N_REPLICATES):
            hits[sampler.draw(rng)] += 1
        observed = hits / N_REPLICATES

        # Assert
        np.testing.assert_allclose(sampler.target, [1.0, 0.5, 0.5], rtol=1e-12)
        self.assertEqual(observed[0], 1.0)
        for index in (1, 2):
            tolerance = TOLERANCE_SIGMAS * _standard_error(0.5, N_REPLICATES)
            self.assertAlmostEqual(observed[index], 0.5, delta=tolerance)

        # Under successive sampling the heavy position was included 0.978 of
        # the time and the light ones 0.511 each — the surplus above 1 that
        # capping assigns explicitly was instead absorbed by the light
        # positions, without anything in the code recording that it happened.
        old = _exact_successive_inclusion(SKEWED_COUNTS, k=2)
        self.assertLess(old[positions[0]], 1.0)
        for position in positions[1:]:
            self.assertGreater(old[position], 0.5)


if __name__ == "__main__":
    unittest.main()
