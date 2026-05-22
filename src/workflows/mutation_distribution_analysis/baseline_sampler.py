"""Baseline sampler (Steps 2 & 3 of mutation_distribution_analysis).

Generates random "baseline" promoter sequences by sampling SNPs directly
from a pre-built :class:`MutationPool`. For a target gene with wildtype
``W``, only pool rows whose ``source_base`` matches ``W[position]`` are
biochemically applicable; uniform sampling over those rows reproduces the
empirical joint distribution ``P(position, source_base, new_base)`` while
position-blocking enforces "no two SNPs at the same site".

Public API:

* :func:`sample_baseline_sequence` — per-gene baseline.
* :func:`generate_baselines_fasta` — batch driver over an entire pool.
* :func:`main` — CLI entry point.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.utils.io import print_status
from workflows.mutation_distribution_analysis.mutation_pool import MutationPool


def _filter_applicable(
    mutations: pd.DataFrame, wildtype: str
) -> pd.DataFrame:
    """Return pool rows whose ``source_base`` matches the wildtype at that position.

    Args:
        mutations: Long-format mutation table from a ``MutationPool`` (columns
            ``gene_id``, ``position``, ``source_base``, ``new_base``).
        wildtype: Wildtype sequence of the target gene. Positions in
            ``mutations`` are interpreted as 0-based indices into this string.

    Returns:
        A DataFrame containing only the rows applicable to ``wildtype``,
        with the same columns as the input. Row order and multiplicity are
        preserved.
    """
    if mutations.empty:
        return mutations.copy()

    positions = mutations["position"].to_numpy()
    wt_array = np.array(list(wildtype))
    wt_at_positions = wt_array[positions]
    source_bases = mutations["source_base"].to_numpy()
    mask = wt_at_positions == source_bases
    return mutations.loc[mask]


def _sample_mutations_blocking(
    applicable: pd.DataFrame,
    k: int,
    rng: np.random.Generator,
) -> List[Tuple[int, str, str]]:
    """Sample ``k`` mutations uniformly from ``applicable`` with position blocking.

    After each draw, all rows sharing the drawn position are removed from
    the working set before the next draw. This guarantees no two sampled
    mutations land at the same site, while keeping each draw uniform over
    the rows of the (filtered) pool — which preserves the empirical
    positional PMF.

    Args:
        applicable: Pool rows applicable to the target wildtype. Must contain
            ``position``, ``source_base``, ``new_base`` columns.
        k: Number of mutations to draw.
        rng: NumPy ``Generator`` used for index sampling. Not consumed when
            ``k == 0``.

    Returns:
        A list of ``(position, source_base, new_base)`` tuples of length ``k``,
        in draw order. All positions are distinct.

    Raises:
        ValueError: If the number of unique positions in ``applicable`` is
            below ``k``.
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
        idx = int(rng.integers(0, len(working)))
        row = working.iloc[idx]
        position = int(row["position"])
        result.append(
            (position, str(row["source_base"]), str(row["new_base"]))
        )
        working = working.loc[working["position"] != position]
    return result


def _apply_mutations(
    wildtype: str, mutations: List[Tuple[int, str, str]]
) -> str:
    """Apply a list of SNPs to a wildtype sequence and return the result.

    Args:
        wildtype: Wildtype sequence.
        mutations: List of ``(position, source_base, new_base)`` tuples. Each
            ``source_base`` is asserted to match ``wildtype[position]`` as a
            defensive guard — by construction this always holds.

    Returns:
        A new string identical to ``wildtype`` except at the mutated positions.

    Raises:
        AssertionError: If a tuple's ``source_base`` does not match the
            wildtype base at its position.
    """
    chars = list(wildtype)
    for position, source_base, new_base in mutations:
        assert chars[position] == source_base, (
            f"source_base '{source_base}' does not match wildtype base "
            f"'{chars[position]}' at position {position}"
        )
        chars[position] = new_base
    return "".join(chars)


def sample_baseline_sequence(
    pool: MutationPool,
    gene_id: str,
    n_mutations: int,
    rng: np.random.Generator,
) -> str:
    """Draw a single baseline mutated sequence for the given gene.

    Args:
        pool: Pre-built mutation pool.
        gene_id: Identifier of the target gene; must be a key in
            ``pool.references``.
        n_mutations: Number of SNPs to introduce into the wildtype.
        rng: NumPy ``Generator`` used for sampling.

    Returns:
        The mutated baseline sequence as a string of the same length as
        the wildtype.

    Raises:
        KeyError: If ``gene_id`` is not present in ``pool.references``.
        ValueError: If the applicable pool has fewer unique positions than
            ``n_mutations``.
    """
    if gene_id not in pool.references:
        raise KeyError(f"Gene id '{gene_id}' not found in pool.references")

    wildtype = pool.references[gene_id]
    applicable = _filter_applicable(pool.mutations, wildtype)
    mutations = _sample_mutations_blocking(applicable, n_mutations, rng)
    return _apply_mutations(wildtype, mutations)


def generate_baselines_fasta(
    pool: MutationPool,
    output_path: str,
    n_per_gene: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Generate ``n_per_gene`` baseline sequences per gene and write a FASTA file.

    For each gene in ``pool.references``:

    * ``k`` is taken from ``pool.gene_stats`` (``n_mutations`` column).
    * The applicable subset of ``pool.mutations`` is computed once.
    * If the number of unique applicable positions is below ``k``, the gene
      is skipped with a warning and the loop continues.
    * Otherwise ``n_per_gene`` independent baselines are drawn and appended
      to the output FASTA. Each record uses one header line and one
      sequence line.

    Args:
        pool: Pre-built mutation pool.
        output_path: Destination FASTA file path. Overwritten if it exists.
        n_per_gene: Number of baselines to draw per gene.
        rng: NumPy ``Generator``. Defaults to ``np.random.default_rng()``.
    """
    if rng is None:
        rng = np.random.default_rng()

    gene_stats_by_id = pool.gene_stats.set_index("gene_id")

    with open(output_path, "w") as fasta:
        for gene_id, wildtype in pool.references.items():
            k = int(gene_stats_by_id.loc[gene_id, "n_mutations"])
            applicable = _filter_applicable(pool.mutations, wildtype)
            unique_positions = applicable["position"].nunique()

            if unique_positions < k:
                print_status(
                    f"Skipping {gene_id}: only {unique_positions} unique "
                    f"applicable positions, need {k}",
                    "WARNING",
                )
                continue

            for i in range(n_per_gene):
                mutations = _sample_mutations_blocking(applicable, k, rng)
                sequence = _apply_mutations(wildtype, mutations)
                fasta.write(f">{gene_id}_baseline_{i:03d}\n")
                fasta.write(f"{sequence}\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the baseline-sampler entry point."""
    parser = argparse.ArgumentParser(description=("Generate baseline mutated promoter sequences by sampling SNPs from a pre-built MutationPool."))
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to a MutationPool JSON (output of mutation_pool.py).",)
    parser.add_argument("--output", "-o", type=str, required=True, help="Path for the output FASTA file.",)
    parser.add_argument("--n-per-gene", "-n", type=int, default=1, help="Number of baseline sequences to draw per gene (default: 1).",)
    parser.add_argument("--seed", "-s", type=int, default=42, help="Seed for np.random.default_rng (default: 42).",)
    args = parser.parse_args()
    if not os.path.isfile(args.input):
        raise ValueError(
            f"Input file '{args.input}' does not exist or is not a file."
        )
    return args


def main() -> None:
    """CLI entry point: load a MutationPool and write baseline FASTA records."""
    args = parse_args()

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    pool = MutationPool.load(args.input)
    rng = np.random.default_rng(args.seed)
    generate_baselines_fasta(
        pool,
        args.output,
        n_per_gene=args.n_per_gene,
        rng=rng,
    )


if __name__ == "__main__":
    main()
