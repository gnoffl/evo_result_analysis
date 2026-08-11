"""Baseline sampler (Steps 2 & 3 of mutation_distribution_analysis).

Generates random "baseline" promoter sequences by sampling SNPs directly
from a pre-built :class:`MutationPool`. For a target gene with wildtype
``W``, only pool rows whose ``source_base`` matches ``W[position]`` are
biochemically applicable.

The draw is factorised into a position design and a substitution draw, which
the empirical joint ``P(position, source_base, new_base)`` permits exactly (see
:class:`GeneMutationSampler`). Positions are drawn under the maximum-entropy
design of
:mod:`workflows.mutation_distribution_analysis.conditional_poisson`, so a
position's probability of *ending up in* a drawn set of size ``k`` is exactly
``k`` times its share of the applicable rows, and no two SNPs land at the same
site. The substitution is then uniform over the applicable rows at the drawn
position.

The earlier scheme — draw a row uniformly, drop that position, repeat — is
successive sampling, whose *inclusion* probabilities are not proportional to
the pool share even though each individual draw is: heavy positions come out
under-included and light ones over-included, flattening the null. The bias is
larger here than in the distance analysis because ``_filter_applicable``
concentrates the weights (median ``max(k*p)`` of 0.23 in ara, 0.41 in zea). See
``plans/sampler_inclusion_probability_fix.md``.

Public API:

* :class:`GeneMutationSampler` — per-gene calibrated sampler.
* :func:`sample_baseline_sequence` — per-gene baseline.
* :func:`generate_baselines_fasta` — batch driver over an entire pool.
* :func:`main` — CLI entry point.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.utils.io import print_status
from tqdm import tqdm
from workflows.mutation_distribution_analysis.conditional_poisson import (
    PositionSampler,
)
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


@dataclass(frozen=True)
class GeneMutationSampler:
    """Draws position-blocked mutation sets for one gene's applicable pool.

    The joint distribution the pool encodes, ``P(position, source_base,
    new_base)``, factorises exactly:

    ``P(position = i) = n_i / N`` and
    ``P(new_base = b | position = i) = n_{i,b} / n_i``,

    where ``n_i`` is the number of applicable rows at position ``i``, ``N`` the
    total number of applicable rows, and ``n_{i,b}`` the number of those rows
    carrying substitution ``b``. The conditional does not involve the position
    design at all — it is a property of the rows at that position — so the
    positional marginal can be replaced by a correct one without touching it.
    That is what this class does: positions come from a
    :class:`~workflows.mutation_distribution_analysis.conditional_poisson.PositionSampler`
    with weights ``n_i``, so position ``i`` appears in a drawn set with
    probability exactly ``k * n_i / N``; the substitution is then drawn
    uniformly over the applicable rows at the chosen position.

    ``source_base`` needs no draw at all. ``_filter_applicable`` keeps only
    rows whose ``source_base`` equals the wildtype base at their position, so
    every applicable row at a given position carries the same ``source_base``
    by construction — verified here rather than assumed.

    Attributes:
        unique_positions: Ascending distinct positions of the applicable pool.
        group_starts: Index into the position-sorted row arrays at which each
            unique position's rows begin.
        group_sizes: Number of applicable rows at each unique position.
        source_bases: The single source base of each unique position.
        sorted_new_bases: ``new_base`` of every applicable row, ordered by
            position so that each unique position's candidates form a
            contiguous block.
        position_sampler: Calibrated maximum-entropy sampler over
            ``unique_positions``.
    """

    unique_positions: np.ndarray
    group_starts: np.ndarray
    group_sizes: np.ndarray
    source_bases: np.ndarray
    sorted_new_bases: np.ndarray
    position_sampler: PositionSampler

    def draw(self, rng: np.random.Generator) -> List[Tuple[int, str, str]]:
        """Draw one position-blocked mutation set.

        Args:
            rng: NumPy ``Generator`` used for the position draw and the
                substitution draws.

        Returns:
            A list of ``(position, source_base, new_base)`` tuples of length
            ``position_sampler.sample_size``, in ascending position order. All
            positions are distinct, and each ``source_base`` matches the
            wildtype base at its position.
        """
        drawn = self.position_sampler.draw(rng)
        if drawn.size == 0:
            return []

        offsets = rng.integers(0, self.group_sizes[drawn])
        row_indices = self.group_starts[drawn] + offsets
        return [
            (
                int(self.unique_positions[index]),
                str(self.source_bases[index]),
                str(self.sorted_new_bases[row]),
            )
            for index, row in zip(drawn, row_indices)
        ]


def _build_gene_sampler(
    applicable: pd.DataFrame, k: int
) -> GeneMutationSampler:
    """Group a gene's applicable pool by position and calibrate its sampler.

    Args:
        applicable: Pool rows applicable to the target wildtype, as returned by
            :func:`_filter_applicable`. Must contain ``position``,
            ``source_base`` and ``new_base`` columns.
        k: Number of distinct positions each draw should return.

    Returns:
        A :class:`GeneMutationSampler` for this gene. Calibration happens here,
        so building one costs about 0.3 s at the real pool sizes while each
        subsequent draw costs well under a millisecond.

    Raises:
        ValueError: If ``applicable`` holds fewer unique positions than ``k``,
            or if some position carries more than one distinct ``source_base``
            (which ``_filter_applicable`` makes impossible, so it would signal
            an upstream bug).
    """
    positions = applicable["position"].to_numpy()
    order = np.argsort(positions, kind="stable")
    sorted_positions = positions[order]
    sorted_source_bases = applicable["source_base"].to_numpy()[order]
    sorted_new_bases = applicable["new_base"].to_numpy()[order]

    unique_positions, group_starts, group_sizes = np.unique(
        sorted_positions, return_index=True, return_counts=True
    )
    if unique_positions.size < k:
        raise ValueError(
            f"Cannot draw {k} position-blocking mutations: only "
            f"{unique_positions.size} unique applicable positions available."
        )

    source_bases = sorted_source_bases[group_starts]
    if not np.all(sorted_source_bases == np.repeat(source_bases, group_sizes)):
        raise ValueError(
            "Applicable pool has more than one source_base at some position; "
            "_filter_applicable should have made that impossible."
        )

    return GeneMutationSampler(
        unique_positions=unique_positions,
        group_starts=group_starts,
        group_sizes=group_sizes,
        source_bases=source_bases,
        sorted_new_bases=sorted_new_bases,
        position_sampler=PositionSampler(group_sizes.astype(float), k),
    )


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

    Note:
        This builds and calibrates a :class:`GeneMutationSampler` on every
        call, which dominates the cost. Use :func:`generate_baselines_fasta`,
        or build the sampler yourself, when drawing repeatedly for one gene.
    """
    if gene_id not in pool.references:
        raise KeyError(f"Gene id '{gene_id}' not found in pool.references")

    wildtype = pool.references[gene_id]
    applicable = _filter_applicable(pool.mutations, wildtype)
    sampler = _build_gene_sampler(applicable, n_mutations)
    return _apply_mutations(wildtype, sampler.draw(rng))


def generate_baselines_fasta(
    pool: MutationPool,
    output_path: str,
    n_per_gene: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Generate ``n_per_gene`` baseline sequences per gene and write a FASTA file.

    For each gene in ``pool.references``:

    * ``k`` is taken from ``pool.gene_stats`` (``n_mutations`` column).
    * The applicable subset of ``pool.mutations`` is computed once, and one
      :class:`GeneMutationSampler` is built and calibrated from it.
    * If the number of unique applicable positions is below ``k``, the gene
      is skipped with a warning and the loop continues.
    * Otherwise ``n_per_gene`` independent baselines are drawn and appended
      to the output FASTA. Each record uses one header line and one
      sequence line.

    Both ``p`` and ``k`` are gene-specific, so the position design has to be
    calibrated per gene rather than once for the pool. That costs roughly
    0.3 s per gene, i.e. about five minutes for a thousand-gene pool.

    Args:
        pool: Pre-built mutation pool.
        output_path: Destination FASTA file path. Overwritten if it exists.
        n_per_gene: Number of baselines to draw per gene.
        rng: NumPy ``Generator``. Defaults to ``np.random.default_rng()``.
    """
    if rng is None:
        rng = np.random.default_rng()

    gene_stats_by_id = pool.gene_stats.set_index("gene_id")
    print_status(
        f"Calibrating a per-gene position design for "
        f"{len(pool.references)} genes; expect roughly "
        f"{0.3 * len(pool.references) / 60:.0f} minutes.",
        "INFO",
    )

    with open(output_path, "w") as fasta:
        for gene_id, wildtype in tqdm(pool.references.items(), desc="Generating baselines"):
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

            sampler = _build_gene_sampler(applicable, k)
            for i in range(n_per_gene):
                sequence = _apply_mutations(wildtype, sampler.draw(rng))
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
