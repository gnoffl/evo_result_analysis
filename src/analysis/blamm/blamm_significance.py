"""Per-motif significance of TFBS gain and loss within one optimization run.

:mod:`analysis.blamm.blamm_tfbs_counts` reports, for every gene and motif, how
often the motif occurs in the unmodified reference sequence and in the optimized
sequence taken from the pareto front. That gives a per-gene difference

    d_{g,m} = count_mutated(g, m) - count_reference(g, m),

where ``g`` ranges over the genes analysed in the run and ``m`` over the motifs
of the scanned database. This module asks, per motif, whether the optimizer
changed that motif's occurrence count systematically across genes: genes are the
replicates, and each motif's vector ``(d_{g,m})_g`` is tested against zero with a
two-sided Wilcoxon signed-rank test. The p-values are Benjamini-Hochberg
corrected across all motifs of the run, so the reported q-values control the
false discovery rate over the whole motif database.

This is the within-run question only — significant enrichment or removal during a
single optimization run — the same test and the same star thresholds as the
intra-run cell annotations of the paper's TF-family heatmaps. Contrasts *between*
runs are not computed here.

Genes with no occurrence of a motif in either sequence are absent from
``motif_counts_per_gene.csv``; they contribute ``d = 0`` and are filled in from
the gene list of ``selected_entries.csv``. Zero differences carry no rank
information for the Wilcoxon test, so this only affects the reported gene count,
never the p-value.

Example:
    python -m analysis.blamm.blamm_significance \\
        --output-folder /path/to/run__mutations_MAX
"""

import argparse
import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from analysis.utils.io import print_status
from analysis.utils.statistics import (
    benjamini_hochberg_qvalues,
    qvalue_to_stars,
    wilcoxon_pvalue_vs_zero,
)

#: File written by :mod:`analysis.blamm.blamm_tfbs_counts` holding one row per
#: ``(gene, motif)`` pair with at least one occurrence in either sequence.
PER_GENE_COUNTS_FILE: str = "motif_counts_per_gene.csv"

#: File written by :mod:`analysis.blamm.blamm_tfbs_counts` documenting the pareto
#: entry chosen per gene; its ``gene`` column is the full list of analysed genes.
SELECTED_ENTRIES_FILE: str = "selected_entries.csv"

#: File written by this module.
SIGNIFICANCE_FILE: str = "motif_significance.csv"

#: Columns identifying a motif, carried over from the per-gene table.
MOTIF_KEY_COLUMNS: List[str] = ["motif_id", "motif_name", "source_db"]

#: Columns of :data:`SIGNIFICANCE_FILE`, declared so the file carries a header
#: even when there is nothing to test.
SIGNIFICANCE_COLUMNS: List[str] = MOTIF_KEY_COLUMNS + [
    "n_genes",
    "n_genes_gained",
    "n_genes_lost",
    "n_genes_unchanged",
    "median_diff",
    "mean_diff",
    "direction",
    "p_intra",
    "q_intra",
    "significance_stars",
]

#: Labels of the ``direction`` column, keyed by the sign of ``mean_diff``.
DIRECTION_ENRICHED: str = "enriched"
DIRECTION_REMOVED: str = "removed"
DIRECTION_AMBIGUOUS: str = ""


def build_diff_matrix(
    per_gene_counts: pd.DataFrame, genes: Sequence[str]
) -> pd.DataFrame:
    """Build the genes x motifs matrix of per-gene occurrence differences.

    Args:
        per_gene_counts: Table from
            :func:`analysis.blamm.blamm_tfbs_counts.count_motif_hits`, with the
            columns ``gene``, ``motif_id`` and ``diff``.
        genes: Every gene analysed in the run, including genes without any motif
            occurrence. Defines the row index and therefore the replicate count.

    Returns:
        DataFrame indexed by gene (in the order of *genes*) with one column per
        motif; ``(gene, motif)`` pairs absent from *per_gene_counts* are 0.

    Raises:
        ValueError: If *genes* is empty or contains duplicates, if
            *per_gene_counts* holds a gene not listed in *genes*, or if it holds
            a duplicated ``(gene, motif_id)`` pair.
    """
    if len(genes) == 0:
        raise ValueError("genes must not be empty")
    if len(set(genes)) != len(genes):
        raise ValueError("genes must not contain duplicates")

    unknown_genes = set(per_gene_counts["gene"]) - set(genes)
    if unknown_genes:
        raise ValueError(
            f"per_gene_counts holds genes missing from the gene list: "
            f"{sorted(unknown_genes)}"
        )
    duplicated = per_gene_counts.duplicated(subset=["gene", "motif_id"]).any()
    if duplicated:
        raise ValueError("per_gene_counts holds duplicated (gene, motif_id) pairs")

    matrix = per_gene_counts.pivot(index="gene", columns="motif_id", values="diff")
    return matrix.reindex(index=list(genes)).fillna(0.0)


def _direction_labels(mean_diff: pd.Series) -> pd.Series:
    """Label each motif ``enriched`` / ``removed`` by the sign of its mean diff.

    The median is a poor direction indicator for these counts: a motif changed in
    a minority of genes still has a median of zero, so the direction is read off
    the mean instead — the net occurrence change per gene, matching
    ``diff_per_gene`` of the aggregate count table.

    Args:
        mean_diff: Mean per-gene occurrence difference per motif.

    Returns:
        Series of direction labels; empty where the net change is exactly zero.
    """
    return pd.Series(
        np.where(
            mean_diff > 0,
            DIRECTION_ENRICHED,
            np.where(mean_diff < 0, DIRECTION_REMOVED, DIRECTION_AMBIGUOUS),
        ),
        index=mean_diff.index,
    )


def motif_significance(
    per_gene_counts: pd.DataFrame, genes: Sequence[str]
) -> pd.DataFrame:
    """Test every motif's per-gene occurrence difference against zero.

    Args:
        per_gene_counts: Table from
            :func:`analysis.blamm.blamm_tfbs_counts.count_motif_hits`.
        genes: Every gene analysed in the run (see :func:`build_diff_matrix`).

    Returns:
        One row per motif with the columns of :data:`SIGNIFICANCE_COLUMNS`,
        sorted by ``q_intra`` ascending and then by ``motif_id``. ``p_intra`` and
        ``q_intra`` are NaN for motifs whose occurrence count is unchanged in
        every gene, since the signed-rank test is undefined there.

    Raises:
        ValueError: Propagated from :func:`build_diff_matrix`.
    """
    if per_gene_counts.empty:
        return pd.DataFrame(columns=pd.Index(SIGNIFICANCE_COLUMNS))

    matrix = build_diff_matrix(per_gene_counts, genes)
    motif_metadata = (
        per_gene_counts[MOTIF_KEY_COLUMNS]
        .drop_duplicates(subset=["motif_id"])
        .set_index("motif_id")
    )

    rows = []
    for motif_id in matrix.columns:
        differences = matrix[motif_id].to_numpy()
        rows.append(
            {
                "motif_id": motif_id,
                "motif_name": motif_metadata.at[motif_id, "motif_name"],
                "source_db": motif_metadata.at[motif_id, "source_db"],
                "n_genes": len(matrix.index),
                "n_genes_gained": int(np.count_nonzero(differences > 0)),
                "n_genes_lost": int(np.count_nonzero(differences < 0)),
                "n_genes_unchanged": int(np.count_nonzero(differences == 0)),
                "median_diff": float(np.median(differences)),
                "mean_diff": float(np.mean(differences)),
                "p_intra": wilcoxon_pvalue_vs_zero(differences),
            }
        )

    result = pd.DataFrame(rows)
    result["direction"] = _direction_labels(result["mean_diff"])
    result["q_intra"] = benjamini_hochberg_qvalues(result["p_intra"])
    result["significance_stars"] = result["q_intra"].map(qvalue_to_stars)
    return (
        result.reindex(columns=SIGNIFICANCE_COLUMNS)
        .sort_values(by=["q_intra", "motif_id"])
        .reset_index(drop=True)
    )


def analyse_output_folder(output_folder: str) -> pd.DataFrame:
    """Add per-motif significance to an existing blamm count analysis.

    Reads the per-gene counts and the analysed gene list from *output_folder* and
    writes :data:`SIGNIFICANCE_FILE` next to them. No blamm scan is repeated, so
    this can annotate results produced by an earlier run of
    :mod:`analysis.blamm.blamm_tfbs_counts`.

    Args:
        output_folder: Folder holding :data:`PER_GENE_COUNTS_FILE` and
            :data:`SELECTED_ENTRIES_FILE`.

    Returns:
        The significance table that was written.

    Raises:
        FileNotFoundError: If either input file is missing.
        ValueError: If the two files disagree about the analysed genes.
    """
    per_gene_path = os.path.join(output_folder, PER_GENE_COUNTS_FILE)
    selections_path = os.path.join(output_folder, SELECTED_ENTRIES_FILE)
    for path in (per_gene_path, selections_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing required input: {path}")

    per_gene_counts = pd.read_csv(per_gene_path)
    genes = pd.read_csv(selections_path)["gene"].tolist()
    significance = motif_significance(per_gene_counts, genes)
    significance.to_csv(
        os.path.join(output_folder, SIGNIFICANCE_FILE), index=False
    )
    return significance


def summarise_significance(significance: pd.DataFrame) -> str:
    """Describe how many motifs reached significance in each direction.

    Args:
        significance: Table from :func:`motif_significance`.

    Returns:
        Single-line human-readable summary.
    """
    significant = significance[significance["significance_stars"] != ""]
    enriched = int((significant["direction"] == DIRECTION_ENRICHED).sum())
    removed = int((significant["direction"] == DIRECTION_REMOVED).sum())
    return (
        f"{len(significant)}/{len(significance)} motifs significant "
        f"(q < 0.05): {enriched} enriched, {removed} removed, "
        f"{len(significant) - enriched - removed} with zero net change"
    )


def parse_arguments(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse and validate command line arguments.

    Args:
        args: Argument list to parse; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Test, per motif, whether an optimization run changed the motif's "
            "occurrence count across genes (Wilcoxon signed-rank vs 0, "
            "BH-corrected across motifs)."
        )
    )
    parser.add_argument(
        "--output-folder",
        "-o",
        nargs="+",
        required=True,
        help=(
            "One or more folders written by analysis.blamm.blamm_tfbs_counts; "
            f"each receives a {SIGNIFICANCE_FILE}."
        ),
    )
    parsed = parser.parse_args(args)
    for folder in parsed.output_folder:
        if not os.path.isdir(folder):
            parser.error(f"Output folder does not exist: {folder}")
    return parsed


def main(args: Optional[Sequence[str]] = None) -> int:
    """Entry point for command line execution.

    Args:
        args: Argument list to parse; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    parsed = parse_arguments(args)
    for folder in parsed.output_folder:
        try:
            significance = analyse_output_folder(folder)
        except (FileNotFoundError, ValueError, KeyError) as error:
            print_status(f"{folder}: {error}", "ERROR")
            return 1
        print_status(f"{folder}: {summarise_significance(significance)}", "SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
