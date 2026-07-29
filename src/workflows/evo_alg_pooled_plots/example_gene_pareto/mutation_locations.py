"""Report the mutation positions of selected Pareto-front members.

Companion to ``example_gene_pareto.py``: that script plots the front, this one
lists *where* the mutations of individual front members sit. The final Pareto
front is loaded with ``MutationsGene`` (which diffs every member against the
run's reference sequence) and searched by mutation count.

Positions come from ``MutatedSequence.mutations`` and are therefore 0-based
within the 3020 bp deepCRE extraction window.

``MutationsGene`` keys the untagged final ``pareto_front.json`` under
``final_generation``, so ``FINAL_GENERATION`` must match the run's
``number_of_generations``; requesting only that generation skips parsing the
per-generation front files.
"""

from pathlib import Path
from typing import List

from analysis.mutations.summarize_mutations import MutatedSequence, MutationsGene

# --- Hardcoded configuration (one-off script) --------------------------------

GENE_RUN_DIR = Path(
    "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/GOF_LOF/GOF/"
    "GOF_single_natural_260311_122228_299093/"
    "3_AT3G60640_gene:22415750-22417548_260312_000750_357934"
)

FINAL_GENERATION = 20000

# Pareto-front members to report, identified by their mutation count.
REQUESTED_MUTATION_COUNTS = (3, 4, 5, 6)


def find_front_member(
    gene: MutationsGene, mutation_count: int, generation: int
) -> MutatedSequence:
    """Find the single front member with a given mutation count.

    Args:
        gene: Loaded run data.
        mutation_count: Number of mutations the member must carry.
        generation: Generation to search in.

    Returns:
        The matching ``MutatedSequence``.

    Raises:
        ValueError: If no member or more than one member carries that mutation
            count (the Pareto front should hold exactly one per count).
    """
    matches = gene.search_mutation_count(mutation_count, generation=generation)[
        generation
    ]
    if len(matches) != 1:
        available = sorted(
            sequence.get_mutation_number()
            for sequence in gene.generation_dict[generation]
        )
        raise ValueError(
            f"Expected exactly one front member with {mutation_count} mutations, "
            f"found {len(matches)}. Mutation counts on the front: {available}."
        )
    return matches[0]


def format_member_mutations(member: MutatedSequence, mutation_count: int) -> str:
    """Render one front member's mutations as a report block.

    Args:
        member: The front member to report.
        mutation_count: Mutation count the member was selected by.

    Returns:
        A multi-line block: a header plus one line per mutation, ordered by
        position.
    """
    lines: List[str] = [
        f"=== {mutation_count} mutations (fitness {member.fitness:.4f}) ==="
    ]
    for position, reference_base, mutated_base in sorted(member.mutations):
        lines.append(f"  pos {position:5d}  {reference_base} -> {mutated_base}")
    return "\n".join(lines)


def main() -> None:
    """Print the mutation positions of the requested front members."""
    gene = MutationsGene(
        str(GENE_RUN_DIR),
        final_generation=FINAL_GENERATION,
        generation=FINAL_GENERATION,
    )
    print(f"Gene run: {GENE_RUN_DIR.name}")
    for mutation_count in REQUESTED_MUTATION_COUNTS:
        member = find_front_member(gene, mutation_count, FINAL_GENERATION)
        print(format_member_mutations(member, mutation_count))
        print()


if __name__ == "__main__":
    main()
