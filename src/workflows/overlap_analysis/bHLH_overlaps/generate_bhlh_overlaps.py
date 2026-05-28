"""Generate the bHLH dCIS-to-deepCRE overlap mapping CSV.

Each unique bHLH binding site in the dCIS in-silico mutagenesis FASTA is
intersected with every gene's deepCRE extraction windows (1500 bp promoter +
1500 bp terminator, separated by central padding for a 3020 bp total). A row
is emitted for every non-empty (site, gene-window) overlap, with positions
mapped into the 3020 bp extracted-sequence frame as 0-based half-open
intervals.

The output (default ``data/bHLH_dCIS_dCRE_overlaps.csv``) parallels
``WRKY_dCIS_dCRE_overlaps.csv`` and feeds the bHLH STARR-seq x deepCRE
correlation script as a candidate-gene filter.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import pandas as pd
from pyfaidx import Fasta

from evolution.extract_sequences import (
    find_genes,
    find_start_end,
    genomic_to_relative_position,
)


_BASE_DIR = os.path.dirname(__file__)
DEFAULT_FASTA_PATH = os.path.normpath(
    os.path.join(_BASE_DIR, "..", "data", "dCIS_bHLH_in_silico_mutated_GS2025d.fasta")
)
DEFAULT_GTF_PATH = (
    "/home/gernot/ARCitect/ARCs/genRE/assays/Gene_Data/dataset/annotations/Arabidopsis_thaliana.TAIR10.52.gtf"
)
DEFAULT_OUTPUT_PATH = os.path.normpath(
    os.path.join(_BASE_DIR, "..", "data", "bHLH_dCIS_dCRE_overlaps.csv")
)
DEFAULT_INTRAGENIC = 500
DEFAULT_EXTRAGENIC = 1000
GENE_ID_SUFFIX = "_gene"
OUTPUT_COLUMNS: List[str] = [
    "site_id",
    "gene_id",
    "strand",
    "region",
    "start",
    "end",
    "additional_padding",
]


@dataclass(frozen=True)
class Site:
    """A unique dCIS binding site parsed from a FASTA header.

    Genomic coordinates are 0-based and inclusive on both ends, matching the
    convention used by the existing WRKY mapping file.

    Attributes:
        site_id: Concatenation ``<tf>_<chrom>:<start>-<end>`` (e.g.
            ``"bHLH_1:14811615-14811864"``); used as the join key downstream.
        tf: Transcription factor prefix from the header (e.g. ``"bHLH"``).
        chrom: Chromosome name as it appears in the FASTA header (e.g. ``"1"``).
        start: Leftmost included genomic base (0-based).
        end: Rightmost included genomic base (0-based, inclusive).
    """

    site_id: str
    tf: str
    chrom: str
    start: int
    end: int


def parse_fasta_sites(fasta_path: str) -> List[Site]:
    """Parse the dCIS FASTA into deduplicated unique genomic sites.

    Headers look like ``<tf>_<chrom>:<start>-<end>_<variant>_<id>``. Multiple
    variants share the same ``<tf>_<chrom>:<start>-<end>`` prefix; this
    function collapses them to one :class:`Site` per unique prefix, preserving
    first-seen order.

    Args:
        fasta_path: Path to the dCIS in-silico mutagenesis FASTA file.

    Returns:
        Unique sites in first-seen order.

    Raises:
        ValueError: If a FASTA header is missing the expected
            ``<tf>_<chrom>:<start>-<end>...`` structure.
    """
    fasta = Fasta(fasta_path)
    seen: set = set()
    sites: List[Site] = []
    for record in fasta:
        record_parts = record.name.split("_")
        if len(record_parts) < 3 or ":" not in record_parts[1]:
            raise ValueError(f"Unexpected FASTA header: {record.name!r}")
        tf = record_parts[0]
        location = record_parts[1]
        site_id = f"{tf}_{location}"
        if site_id in seen:
            continue
        seen.add(site_id)
        chrom, start_end = location.split(":")
        start_str, end_str = start_end.split("-")
        sites.append(
            Site(
                site_id=site_id,
                tf=tf,
                chrom=chrom,
                start=int(start_str),
                end=int(end_str),
            )
        )
    return sites


def _strip_gene_suffix(gene_id: str) -> str:
    """Strip the trailing ``_gene`` suffix added by ``find_genes``."""
    if gene_id.endswith(GENE_ID_SUFFIX):
        return gene_id[: -len(GENE_ID_SUFFIX)]
    return gene_id


def overlaps_for_site(
    site: Site,
    genes_df: pd.DataFrame,
    intragenic: int = DEFAULT_INTRAGENIC,
    extragenic: int = DEFAULT_EXTRAGENIC,
) -> List[Dict[str, Union[str, int]]]:
    """Find all gene-window overlaps for a single site.

    For each gene on the same chromosome as the site, computes the promoter
    and terminator extraction windows via :func:`find_start_end` and
    intersects each window separately with the site. Every non-empty overlap
    becomes one output row with positions mapped into the 3020 bp extracted
    sequence via :func:`genomic_to_relative_position`.

    Args:
        site: The dCIS site to query.
        genes_df: Gene table as returned by :func:`find_genes` (columns
            ``chromosome``, ``start``, ``end``, ``strand``, ``gene_id``).
        intragenic: Intragenic window width passed to ``find_start_end``.
        extragenic: Extragenic window width passed to ``find_start_end``.

    Returns:
        One dict per (site, gene-window) overlap with keys matching
        :data:`OUTPUT_COLUMNS`.
    """
    rows: List[Dict[str, Union[str, int]]] = []
    chrom_genes = genes_df[genes_df["chromosome"] == site.chrom]
    for gene in chrom_genes.itertuples(index=False):
        (
            prom_start, prom_end, term_start, term_end, additional_padding,
        ) = find_start_end(int(gene.start), int(gene.end), intragenic, extragenic, gene.strand)

        windows = ((prom_start, prom_end), (term_start, term_end),)
        for window_start, window_end in windows:
            # The site uses inclusive-inclusive 0-based coords; the window
            # uses half-open. Pull both into inclusive-inclusive so that
            # min/max do the right thing.
            overlap_first_base = max(site.start, window_start)
            overlap_last_base = min(site.end, window_end - 1)
            if overlap_first_base > overlap_last_base:
                continue
            additional_parameters = {
                "prom_start": prom_start,
                "prom_end": prom_end,
                "term_start": term_start,
                "term_end": term_end,
                "strand": gene.strand,
                "additional_padding": additional_padding,
            }
            mapped_first = genomic_to_relative_position(overlap_first_base, **additional_parameters)
            mapped_last = genomic_to_relative_position(overlap_last_base, **additional_parameters)
            if mapped_first is None or mapped_last is None:
                continue
            rel_first, region = mapped_first
            rel_last, _ = mapped_last
            # min/max keeps the half-open interval correctly ordered on both
            # strands; +1 converts the inclusive last position to half-open.
            extracted_seq_start = min(rel_first, rel_last)
            extracted_seq_end = max(rel_first, rel_last) + 1
            rows.append(
                {
                    "site_id": site.site_id,
                    "gene_id": _strip_gene_suffix(str(gene.gene_id)),
                    "strand": gene.strand,
                    "region": region,
                    "start": extracted_seq_start,
                    "end": extracted_seq_end,
                    "additional_padding": additional_padding,
                }
            )
    return rows


def build_mapping(
    fasta_path: str,
    gtf_path: str,
    intragenic: int = DEFAULT_INTRAGENIC,
    extragenic: int = DEFAULT_EXTRAGENIC,
) -> Tuple[pd.DataFrame, List[Site]]:
    """Build the full bHLH overlap mapping DataFrame.

    Args:
        fasta_path: Path to the dCIS in-silico mutagenesis FASTA file.
        gtf_path: Path to the genome annotation (GTF/GFF).
        intragenic: Intragenic window width passed to ``find_start_end``.
        extragenic: Extragenic window width passed to ``find_start_end``.

    Returns:
        DataFrame with columns :data:`OUTPUT_COLUMNS`, one row per
        (site, gene-window) overlap.
    """
    sites = parse_fasta_sites(fasta_path)
    genes_df = find_genes(
        annotation_path=gtf_path,
        gene_name_attribute="gene_id",
        feature_type_filter=["gene"],
        genes_of_interest=[],
    )
    all_rows: List[Dict[str, Union[str, int]]] = []
    # Pre-slice per chromosome so each site loops over a smaller table.
    per_chrom_genes: Dict[str, pd.DataFrame] = {
        chrom: chrom_df for chrom, chrom_df in genes_df.groupby("chromosome", sort=False)
    }
    for site in sites:
        chrom_genes = per_chrom_genes.get(site.chrom)
        if chrom_genes is None:
            continue
        all_rows.extend(
            overlaps_for_site(site, chrom_genes, intragenic=intragenic, extragenic=extragenic)
        )
    return pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS), sites


def summarize(df: pd.DataFrame, sites: Sequence[Site]) -> str:
    """Build a multi-line summary string for stdout reporting.

    Args:
        df: The full mapping DataFrame returned by :func:`build_mapping`.
        sites: The unique sites parsed from the input FASTA.

    Returns:
        Multi-line summary string.
    """
    unique_sites = len(sites)
    sites_with_overlap = int(df["site_id"].nunique()) if not df.empty else 0
    rows = len(df)
    short_gene_rows = int((df["additional_padding"] > 0).sum()) if not df.empty else 0
    return (
        f"unique sites: {unique_sites}\n"
        f"sites with >=1 overlap: {sites_with_overlap}\n"
        f"rows emitted: {rows}\n"
        f"short-gene rows (additional_padding>0): {short_gene_rows}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", default=DEFAULT_FASTA_PATH, help="Path to the dCIS in-silico mutagenesis FASTA",)
    parser.add_argument("--gtf", default=DEFAULT_GTF_PATH, help="Path to the genome annotation (GTF/GFF)",)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output CSV path",)
    parser.add_argument("--intragenic", type=int, default=DEFAULT_INTRAGENIC, help="Intragenic window width (default 500)",)
    parser.add_argument("--extragenic", type=int, default=DEFAULT_EXTRAGENIC, help="Extragenic window width (default 1000)",)
    return parser.parse_args()


def main() -> None:
    """CLI entry point: build the mapping, write it to CSV, print a summary."""
    args = _parse_args()
    df, sites = build_mapping(
        fasta_path=args.fasta,
        gtf_path=args.gtf,
        intragenic=args.intragenic,
        extragenic=args.extragenic,
    )
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(summarize(df, sites))


if __name__ == "__main__":
    main()
