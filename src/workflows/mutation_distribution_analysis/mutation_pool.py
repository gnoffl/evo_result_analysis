"""Mutation pool extraction (Step 1 of mutation_distribution_analysis).

Extracts SNP mutations from a summarized-mutations JSON (produced by
``analysis.mutations.summarize_mutations``) into a flat, sample-friendly
artifact: long-format mutations table, per-gene statistics, and reference
sequence lookup. Persisted as a single JSON file.

The downstream PMF construction (Step 2) and baseline sampling (Step 3) consume
this artifact directly and never touch the raw ``MutationsGene`` objects again.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from analysis.mutations.summarize_mutations import (
    MutatedSequence,
    MutationsGene,
    load_mutations_from_json,
)
from analysis.utils.io import (
    print_section_header,
    print_status,
    print_subsection,
)


VALID_BASES = frozenset({"A", "C", "G", "T"})

MUTATION_COLUMNS = ["gene_id", "position", "source_base", "new_base"]
GENE_STATS_COLUMNS = [
    "gene_id",
    "n_mutations",
    "initial_fitness",
    "final_fitness",
]


def _select_best_sequence(
    gene_id: str, gene: MutationsGene, generation: int
) -> MutatedSequence:
    """Return the highest-fitness sequence for a gene at a given generation.

    Args:
        gene_id: Identifier of the gene (used for error messages only).
        gene: The gene whose ``generation_dict`` is consulted.
        generation: Generation key to look up in ``gene.generation_dict``.

    Returns:
        The sequence with maximum ``fitness`` for that generation.

    Raises:
        ValueError: If the gene has no sequences for the requested generation.
    """
    sequences = gene.generation_dict.get(generation)
    if not sequences:
        raise ValueError(
            f"Gene '{gene_id}' has no sequences for generation {generation}."
        )
    return max(sequences, key=lambda seq: seq.fitness)


def _filter_valid_mutations(
    mutations: List[Tuple[int, str, str]],
) -> Tuple[List[Tuple[int, str, str]], int]:
    """Partition mutations into ACGT-valid kept entries and a dropped count.

    Args:
        mutations: Iterable of ``(position, source_base, new_base)`` tuples.

    Returns:
        A pair ``(kept, dropped)`` where ``kept`` is the list of mutations
        whose source and new bases are both in ``{A, C, G, T}`` and
        ``dropped`` is the count of mutations excluded by that filter.
    """
    kept: List[Tuple[int, str, str]] = []
    dropped = 0
    for position, ref_base, mut_base in mutations:
        if ref_base in VALID_BASES and mut_base in VALID_BASES:
            kept.append((position, ref_base, mut_base))
        else:
            dropped += 1
    return kept, dropped


def _build_mutation_rows(
    gene_id: str, mutations: List[Tuple[int, str, str]]
) -> List[Dict[str, object]]:
    """Convert kept mutation tuples into long-format row dicts.

    Args:
        gene_id: Gene identifier to attach to every emitted row.
        mutations: List of ``(position, source_base, new_base)`` tuples.

    Returns:
        One dict per input tuple, matching ``MUTATION_COLUMNS``.
    """
    return [
        {
            "gene_id": gene_id,
            "position": position,
            "source_base": ref_base,
            "new_base": mut_base,
        }
        for position, ref_base, mut_base in mutations
    ]


def _build_gene_stats_row(
    gene_id: str, gene: MutationsGene, n_mutations: int, generation: int
) -> Dict[str, object]:
    """Build the per-gene statistics row for a single gene.

    Args:
        gene_id: Gene identifier.
        gene: Source gene; used for reference length and fitness lookup.
        n_mutations: Number of kept (ACGT-valid) mutations for this gene.
        generation: Generation passed to
            ``gene.get_init_and_optimal_fitness_generation``.

    Returns:
        A dict matching ``GENE_STATS_COLUMNS``.
    """
    initial_fitness, final_fitness = (
        gene.get_init_and_optimal_fitness_generation(generation)
    )
    return {
        "gene_id": gene_id,
        "n_mutations": n_mutations,
        "initial_fitness": initial_fitness,
        "final_fitness": final_fitness,
    }


@dataclass
class MutationPool:
    """Container for the pooled mutation data used by downstream PMF code.

    Attributes:
        mutations: Long-format DataFrame with one row per SNP across all genes.
            Columns: ``gene_id``, ``position``, ``source_base``, ``new_base``.
        gene_stats: One row per gene. Columns: ``gene_id``, ``n_mutations``,
            ``initial_fitness``, ``final_fitness``.
            ``initial_fitness`` is the fitness of the pareto-front sequence with
            the fewest mutations; ``final_fitness`` is the fitness of the
            sequence with the most mutations (highest fitness on the front).
        references: Mapping ``gene_id -> wildtype sequence``. Stored as a dict
            rather than a DataFrame because lookups are by gene id and the
            strings are large.
    """

    mutations: pd.DataFrame
    gene_stats: pd.DataFrame
    references: Dict[str, str]

    @classmethod
    def from_summarized_json(
        cls, path: str, generation: int = 1999
    ) -> "MutationPool":
        """Build a ``MutationPool`` from a summarized-mutations JSON file.

        For each gene, picks the highest-fitness sequence in the requested
        generation (which, by the pareto-front property, also has the most
        mutations) and extracts its SNPs. Mutations whose source or new base
        is not in ``{A, C, G, T}`` are dropped.

        Args:
            path: Path to a summarized-mutations JSON file produced by
                ``analysis.mutations.summarize_mutations``.
            generation: Generation to extract from each gene's
                ``generation_dict``. Defaults to ``1999`` (final generation).

        Returns:
            A populated ``MutationPool``.

        Raises:
            ValueError: If a gene has no sequences for the requested generation.
        """
        genes: Dict[str, MutationsGene] = load_mutations_from_json(path)

        mutation_rows: List[Dict[str, object]] = []
        gene_stats_rows: List[Dict[str, object]] = []
        references: Dict[str, str] = {}

        for gene_id, gene in genes.items():
            best_sequence = _select_best_sequence(gene_id, gene, generation)
            mutations, invalid_mutation_count = _filter_valid_mutations(best_sequence.mutations)

            if invalid_mutation_count:
                print_status(f"{gene_id}: dropped {invalid_mutation_count} non-ACGT mutation(s)", "WARNING",)

            mutation_rows.extend(_build_mutation_rows(gene_id, mutations))
            gene_stats_rows.append(_build_gene_stats_row(gene_id, gene, len(mutations), generation))
            references[gene_id] = gene.reference_sequence

        mutations_df = pd.DataFrame(mutation_rows, columns=MUTATION_COLUMNS)
        gene_stats_df = pd.DataFrame(gene_stats_rows, columns=GENE_STATS_COLUMNS)

        return cls(
            mutations=mutations_df,
            gene_stats=gene_stats_df,
            references=references,
        )

    @classmethod
    def load(cls, path: str) -> "MutationPool":
        """Load a previously saved ``MutationPool`` from a JSON file.

        Args:
            path: Path to a JSON file written by ``MutationPool.save``.

        Returns:
            The reconstructed ``MutationPool``.
        """
        with open(path, "r") as f:
            data = json.load(f)

        mutations = pd.DataFrame(
            data.get("mutations", []), columns=MUTATION_COLUMNS
        )
        gene_stats = pd.DataFrame(
            data.get("gene_stats", []), columns=GENE_STATS_COLUMNS
        )
        references = dict(data.get("references", {}))

        return cls(
            mutations=mutations,
            gene_stats=gene_stats,
            references=references,
        )

    def save(self, path: str) -> None:
        """Persist the pool to a JSON file.

        Args:
            path: Output file path. Parent directories must already exist.
        """
        payload = {
            "mutations": self.mutations.to_dict(orient="records"),
            "gene_stats": self.gene_stats.to_dict(orient="records"),
            "references": self.references,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the mutation-pool extraction entry point."""
    parser = argparse.ArgumentParser( description="Extract a MutationPool from a summarized-mutations JSON file and persist it for downstream PMF construction.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to a summarized-mutations JSON (all_mutated_sequences_*.json).",)
    parser.add_argument("--output", "-o", type=str, required=True, help="Path for the output mutation_pool JSON.",)
    parser.add_argument("--generation", "-g", type=int, default=1999, help="Generation to extract from each gene's generation dict.",)
    args = parser.parse_args()
    if not os.path.isfile(args.input):
        raise ValueError(f"Input file '{args.input}' does not exist or is not a file.")
    return args


def main() -> None:
    """CLI entry point: build a MutationPool and persist it to disk."""
    # print_section_header("MUTATION POOL EXTRACTION", "=")
    # print_status(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    args = parse_args()

    # print_subsection("Configuration")
    # print(f"  Input:      {args.input}")
    # print(f"  Output:     {args.output}")
    # print(f"  Generation: {args.generation}")

    try:
        # print_subsection("Building Mutation Pool")
        pool = MutationPool.from_summarized_json(
            args.input, generation=args.generation
        )
        # print_status(
        #     f"Collected {len(pool.mutations)} mutations across "
        #     f"{len(pool.gene_stats)} genes",
        #     "SUCCESS",
        # )

        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)

        # print_subsection("Saving Pool")
        # print_status(f"Writing pool to {args.output}")
        pool.save(args.output)
        # print_status("Saved successfully", "SUCCESS")

        # print_section_header("EXTRACTION COMPLETE", "=")
        # print_status(
        #     f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        # )
        # print(f"\nOUTPUT_FILE={args.output}\n")
    except Exception as e:
        # print_section_header("EXTRACTION FAILED", "=")
        # print_status(f"Error: {e}", "ERROR")
        raise


if __name__ == "__main__":
    main()
