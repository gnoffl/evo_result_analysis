"""Mutation-distance random baseline analysis.

Compares the inter-mutation distance distribution of the EA's *real*
mutations against a null distribution drawn from the EA's own positional
pool (uniform-over-rows + position-blocking, no wildtype filtering).

Public API:

* :func:`compute_real_distances` -- per-gene ``np.diff`` aggregated into a
  global :class:`~collections.Counter`.
* :func:`compute_random_distances` -- for each gene, draw ``n_per_gene``
  position-sets of size ``k = gene_stats.n_mutations`` from the full pool,
  diff each, sum into one ``Counter``.
* :func:`counter_to_array` -- expand a ``{distance: count}`` mapping into
  a flat array suitable for KS / Wasserstein tests.
* :func:`plot_overlay` -- one figure overlaying the real histogram (bars)
  with a rescaled random distribution (line).
* :func:`run_distance_analysis` -- end-to-end driver: load pool, compute
  both Counters, write two overlay plots and a stats file.
* :func:`main` -- CLI entry point.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from tqdm import tqdm

from analysis.mutations.analyze_mutations import (
    calculate_mutation_distances_single_gene,
)
from workflows.mutation_distribution_analysis.mutation_pool import MutationPool


def _sample_positions_blocking(
    unique_positions: np.ndarray,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``k`` distinct positions weighted-without-replacement from a PMF.

    Mathematically equivalent to
    :func:`workflows.mutation_distribution_analysis.baseline_sampler._sample_mutations_blocking`
    when only positions (not the ``(source_base, new_base)`` triple) are
    needed: uniform-over-rows with position blocking is identical to
    weighted-without-replacement over unique positions, with each unique
    position weighted by its row count in the pool.

    Args:
        unique_positions: Sorted 1-D array of distinct positions from the
            mutation pool.
        weights: Probability vector aligned with ``unique_positions``;
            must sum to 1.
        k: Number of distinct positions to draw.
        rng: NumPy ``Generator`` used for the draw.

    Returns:
        1-D array of ``k`` distinct positions sampled without replacement
        according to ``weights``.

    Raises:
        ValueError: If ``k`` exceeds the number of unique positions.
    """
    if k > unique_positions.size:
        raise ValueError(
            f"Cannot draw {k} position-blocking mutations: only "
            f"{unique_positions.size} unique applicable positions available."
        )
    return rng.choice(unique_positions, size=k, replace=False, p=weights)


def compute_real_distances(pool: MutationPool) -> Counter:
    """Aggregate inter-mutation distances of the real EA mutations.

    For each gene in ``pool.mutations``, sorts the mutation positions and
    computes consecutive differences (``np.diff``) via
    :func:`calculate_mutation_distances_single_gene`, then sums the
    per-gene counters into a single global one. Genes with fewer than two
    mutations contribute nothing.

    Args:
        pool: Mutation pool whose ``mutations`` DataFrame holds the real
            per-gene mutation positions.

    Returns:
        A ``Counter`` mapping each observed distance to its total
        occurrence count across all genes.
    """
    distances: Counter = Counter()
    for _, group in pool.mutations.groupby("gene_id"):
        positions = group["position"].tolist()
        if len(positions) < 2:
            continue
        distances += calculate_mutation_distances_single_gene(positions)
    return distances


def compute_random_distances(
    pool: MutationPool,
    n_per_gene: int,
    rng: np.random.Generator,
) -> Counter:
    """Aggregate inter-mutation distances of random draws from the pool.

    For every gene in ``pool.references``, ``k`` is read from
    ``pool.gene_stats`` (``n_mutations`` column). ``n_per_gene`` independent
    position-sets of size ``k`` are then drawn from the *full* unfiltered
    pool of positions via :func:`_sample_positions_blocking`
    (weighted-without-replacement, equivalent to uniform-over-rows with
    position blocking). The per-replicate ``np.diff`` of the sorted draws
    is accumulated and collapsed into a single ``Counter`` at the end.
    Genes with ``k < 2`` contribute nothing.

    Args:
        pool: Mutation pool providing both the sample space
            (``pool.mutations``) and the per-gene draw size
            (``pool.gene_stats``).
        n_per_gene: Number of independent replicates per gene.
        rng: NumPy ``Generator`` used for sampling. Driven sequentially
            across genes and replicates, so a fixed seed yields a fixed
            global distribution.

    Returns:
        A ``Counter`` mapping each observed distance to its total
        occurrence count across all genes and replicates.
    """
    pos_arr = pool.mutations["position"].to_numpy()
    unique_positions, counts = np.unique(pos_arr, return_counts=True)
    weights = counts / counts.sum()

    gene_stats_by_id = pool.gene_stats.set_index("gene_id")
    all_distances: list[np.ndarray] = []
    for gene_id in tqdm(pool.references.keys(), desc="Random per-gene draws"):
        k = int(gene_stats_by_id.loc[gene_id, "n_mutations"])
        if k < 2:
            continue
        for _ in range(n_per_gene):
            positions = _sample_positions_blocking(
                unique_positions, weights, k, rng
            )
            positions.sort()
            all_distances.append(np.diff(positions))

    if not all_distances:
        return Counter()
    flat = np.concatenate(all_distances)
    vals, cnts = np.unique(flat, return_counts=True)
    return Counter(dict(zip(vals.tolist(), cnts.tolist())))


def counter_to_array(counter: Counter) -> np.ndarray:
    """Expand a distance ``Counter`` into a flat 1-D numpy array.

    Each distance ``d`` is repeated ``counter[d]`` times. Order of the
    repeats is not meaningful for KS / Wasserstein, but distances are
    returned in ascending order to keep output deterministic.

    Args:
        counter: ``{distance: count}`` mapping.

    Returns:
        Flat array of length ``sum(counter.values())``. Empty input
        yields an empty ``int`` array.
    """
    if not counter:
        return np.array([], dtype=int)
    distances = np.array(sorted(counter.keys()))
    counts = np.array([counter[d] for d in distances])
    return np.repeat(distances, counts)


def plot_overlay(
    real: Counter,
    random_dist: Counter,
    name: str,
    output_dir: str,
    output_format: str = "png",
    max_distance: Optional[int] = None,
) -> None:
    """Plot real distances as bars overlaid with a rescaled random line.

    The random series is rescaled so its peak matches the real series'
    peak, mirroring the convention in
    :func:`analyze_mutations.plot_dist_hist`. When ``max_distance`` is
    given, both series are restricted to distances ``<= max_distance``
    and ``"_smaller"`` is appended to the output filename.

    Args:
        real: Real-mutation distance counter.
        random_dist: Random-draw distance counter.
        name: Identifier used in the output filename
            (``mutation_distances_{name}_overlay[_smaller].{format}``).
        output_dir: Destination directory.
        output_format: File extension (e.g. ``"png"``, ``"pdf"``).
        max_distance: Optional upper cutoff on the plotted distance axis.
    """
    if max_distance is not None:
        real = Counter({d: c for d, c in real.items() if d <= max_distance})
        random_dist = Counter(
            {d: c for d, c in random_dist.items() if d <= max_distance}
        )

    distances = sorted(real.keys())
    counts = [real[d] for d in distances]

    plt.clf()
    plt.figure(figsize=(12, 6))
    if counts:
        plt.bar(
            distances,
            counts,
            width=1.0,
            edgecolor="black",
            label="Real",
        )

    if random_dist and counts:
        max_count = max(counts)
        random_x = sorted(random_dist.keys())
        random_y = [random_dist[d] for d in random_x]
        max_rand = max(random_y)
        random_y_scaled = [y * (max_count / max_rand) for y in random_y]
        plt.plot(
            random_x,
            random_y_scaled,
            color="green",
            label="Random (rescaled)",
        )

    plt.xlabel("Mutation Distance")
    plt.ylabel("Frequency")
    plt.legend()
    if distances:
        plt.xlim(0, max(distances) + 1)

    suffix = "_smaller" if max_distance is not None else ""
    out_path = os.path.join(
        output_dir,
        f"mutation_distances_{name}_overlay{suffix}.{output_format}",
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_distance_analysis(
    pool_path: str,
    output_dir: str,
    name: str,
    n_per_gene: int = 10,
    seed: int = 42,
    output_format: str = "png",
) -> None:
    """Run the full distance-vs-random analysis for one mutation pool.

    Loads the pool, computes both distance distributions, writes two
    overlay plots (full range and ``<= 200`` zoom) and a stats file
    containing the KS statistic, KS p-value and Wasserstein distance.

    Args:
        pool_path: Path to a ``MutationPool`` JSON.
        output_dir: Destination directory; created if missing.
        name: Identifier used in all output filenames.
        n_per_gene: Random replicates per gene. Defaults to ``10``.
        seed: Seed for ``np.random.default_rng``. Defaults to ``42``.
        output_format: File extension for plots. Defaults to ``"png"``.
    """
    os.makedirs(output_dir, exist_ok=True)

    pool = MutationPool.load(pool_path)

    real_distances = compute_real_distances(pool)
    rng = np.random.default_rng(seed)
    random_distances = compute_random_distances(pool, n_per_gene, rng)

    plot_overlay(
        real_distances,
        random_distances,
        name,
        output_dir,
        output_format,
    )
    plot_overlay(
        real_distances,
        random_distances,
        name,
        output_dir,
        output_format,
        max_distance=200,
    )

    real_arr = counter_to_array(real_distances)
    random_arr = counter_to_array(random_distances)
    ks_result = ks_2samp(real_arr, random_arr)
    w_dist = wasserstein_distance(real_arr, random_arr)

    stats_path = os.path.join(
        output_dir, f"mutation_distances_{name}_stats.txt"
    )
    with open(stats_path, "w") as f:
        f.write(f"ks_statistic={float(ks_result.statistic)}\n")     #type:ignore
        f.write(f"ks_pvalue={float(ks_result.pvalue)}\n")           #type:ignore
        f.write(f"wasserstein_distance={float(w_dist)}\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the distance-analysis entry point.

    Returns:
        Parsed ``argparse.Namespace`` with: ``pool``, ``output_dir``,
        ``name``, ``n_per_gene``, ``seed``, ``format``.

    Raises:
        ValueError: If ``--pool`` does not point to an existing file or
            ``--format`` is not a supported image extension.
    """
    parser = argparse.ArgumentParser(description=( "Compare EA mutation-distance distributions against a random baseline drawn from the same positional pool."))
    parser.add_argument("--pool", type=str, required=True, help="Path to a MutationPool JSON (output of mutation_pool.py).",)
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output plots and the stats file.",)
    parser.add_argument("--name", type=str, required=True, help="Identifier used in all output filenames.",)
    parser.add_argument("--n-per-gene", type=int, default=10, help="Random replicates per gene (default: 10).",)
    parser.add_argument("--seed", type=int, default=42, help="Seed for np.random.default_rng (default: 42).",)
    parser.add_argument("--format", type=str, default="png", help="Output format for plots (default: png).",)

    args = parser.parse_args()
    if not os.path.isfile(args.pool):
        raise ValueError(
            f"Pool file '{args.pool}' does not exist or is not a file."
        )
    args.format = args.format.lower().strip(".").strip()
    if args.format not in {"png", "pdf", "jpg", "jpeg", "svg", "tiff"}:
        raise ValueError(
            f"Unsupported output format: {args.format}."
        )
    return args


def main() -> None:
    """CLI entry point: run the distance analysis for one pool."""
    args = parse_args()
    run_distance_analysis(
        pool_path=args.pool,
        output_dir=args.output_dir,
        name=args.name,
        n_per_gene=args.n_per_gene,
        seed=args.seed,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
