"""Unit tests for the Pareto-front mutation-position report."""

import unittest
from typing import List

from analysis.mutations.summarize_mutations import MutatedSequence, MutationsGene
from workflows.evo_alg_pooled_plots.example_gene_pareto.mutation_locations import (
    find_front_member,
    format_member_mutations,
)

REFERENCE_SEQUENCE = "ACGTACGTAC"
FINAL_GENERATION = 20000


def make_gene(sequences: List[MutatedSequence]) -> MutationsGene:
    """Build a ``MutationsGene`` holding one generation, bypassing file IO.

    Args:
        sequences: Front members to store under ``FINAL_GENERATION``.

    Returns:
        A ``MutationsGene`` instance with the given front.
    """
    gene = MutationsGene.__new__(MutationsGene)
    gene.reference_sequence = REFERENCE_SEQUENCE
    gene.generation_dict = {FINAL_GENERATION: sequences}
    return gene


def make_member(mutated_sequence: str, fitness: float) -> MutatedSequence:
    """Build a front member by diffing a sequence against the reference.

    Args:
        mutated_sequence: Sequence of the same length as ``REFERENCE_SEQUENCE``.
        fitness: Fitness of the member.

    Returns:
        The corresponding ``MutatedSequence``.
    """
    return MutatedSequence(
        reference_sequence=REFERENCE_SEQUENCE,
        mutated_sequence=mutated_sequence,
        fitness=fitness,
    )


class TestFindFrontMember(unittest.TestCase):
    """Tests for :func:`find_front_member`."""

    def test_returns_member_with_requested_mutation_count(self) -> None:
        # Arrange
        one_mutation = make_member("CCGTACGTAC", 0.1)
        two_mutations = make_member("CCGTACGTAA", 0.2)
        gene = make_gene([one_mutation, two_mutations])

        # Act
        member = find_front_member(gene, 2, FINAL_GENERATION)

        # Assert
        self.assertEqual(member, two_mutations)

    def test_raises_when_mutation_count_absent(self) -> None:
        # Arrange
        gene = make_gene([make_member("CCGTACGTAC", 0.1)])

        # Act / Assert
        with self.assertRaises(ValueError):
            find_front_member(gene, 5, FINAL_GENERATION)

    def test_raises_when_mutation_count_ambiguous(self) -> None:
        # Arrange
        gene = make_gene(
            [make_member("CCGTACGTAC", 0.1), make_member("ACGTACGTAA", 0.2)]
        )

        # Act / Assert
        with self.assertRaises(ValueError):
            find_front_member(gene, 1, FINAL_GENERATION)


class TestFormatMemberMutations(unittest.TestCase):
    """Tests for :func:`format_member_mutations`."""

    def test_lists_mutations_in_position_order(self) -> None:
        # Arrange
        member = make_member("CCGTACGTAA", 0.25)

        # Act
        report = format_member_mutations(member, 2)

        # Assert
        lines = report.splitlines()
        self.assertEqual(lines[0], "=== 2 mutations (fitness 0.2500) ===")
        self.assertEqual(lines[1], "  pos     0  A -> C")
        self.assertEqual(lines[2], "  pos     9  C -> A")


if __name__ == "__main__":
    unittest.main()
