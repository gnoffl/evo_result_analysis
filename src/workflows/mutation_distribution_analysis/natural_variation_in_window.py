"""How much natural variation falls into the plasmid test-sequence window?

The enhancer-library plasmid (pPSntF_enhLib_35Spr) carries a 170 bp test
sequence at -220..-51 relative to the TSS. In deepCRE extraction coordinates
(3020 bp: 1000 bp extragenic + 500 bp intragenic promoter, 20 bp padding,
1500 bp terminator) that is 0-based index 780..949, i.e. 1-based VCF POS
781..950.

This script counts, per gene, how many of the re-extracted natural SNPs fall
into that window, and compares the SNP density inside it against the rest of
the promoter half.

Usage:
    conda run -n deepCREshap python -m \
        workflows.mutation_distribution_analysis.natural_variation_in_window
"""

from __future__ import annotations

import glob
import os

import pandas as pd

VCF_DIRECTORIES = {
    "GOF": "/home/gernot/Code/PhD_Code/Evolution/data/Arabidopsis_GOF_reextracted_vcfs",
    "LOF": "/home/gernot/Code/PhD_Code/Evolution/data/Arabidopsis_LOF_reextracted_vcfs",
}

# 1-based, inclusive, in deepCRE extraction coordinates.
WINDOW_START = 781
WINDOW_END = 950
WINDOW_LENGTH = WINDOW_END - WINDOW_START + 1

# The promoter half of the deepCRE input, 1-based inclusive.
PROMOTER_START = 1
PROMOTER_END = 1500
PROMOTER_LENGTH = PROMOTER_END - PROMOTER_START + 1


def read_vcf_positions(vcf_path: str) -> pd.DataFrame:
    """Read the CHROM/POS columns of one re-extracted VCF.

    Args:
        vcf_path: Path to a VCF written by ``extract_sequences.py``.

    Returns:
        DataFrame with columns ``gene_id`` and ``position`` (1-based position
        in the 3020 bp deepCRE construct).
    """
    table = pd.read_csv(
        vcf_path,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 1],
        names=["gene_id", "position"],
    )
    return table


def load_variation(vcf_directory: str) -> pd.DataFrame:
    """Load all SNP positions from a directory of re-extracted VCFs.

    Args:
        vcf_directory: Directory containing ``*.vcf`` files, one per gene.

    Returns:
        Concatenated DataFrame with columns ``gene_id`` and ``position``.

    Raises:
        ValueError: If the directory contains no VCF files.
    """
    vcf_paths = sorted(glob.glob(os.path.join(vcf_directory, "*.vcf")))
    if not vcf_paths:
        raise ValueError(f"No VCF files found in '{vcf_directory}'.")
    return pd.concat(
        [read_vcf_positions(path) for path in vcf_paths], ignore_index=True
    )


def summarize(dataset_name: str, variation: pd.DataFrame) -> None:
    """Print a describe-style summary of window occupancy for one dataset.

    Args:
        dataset_name: Label used in the printed header (e.g. ``"GOF"``).
        variation: DataFrame with ``gene_id`` and ``position`` columns.
    """
    in_window = variation["position"].between(WINDOW_START, WINDOW_END)
    in_promoter = variation["position"].between(PROMOTER_START, PROMOTER_END)

    genes = variation["gene_id"].unique()
    counts_in_window = (
        variation[in_window]
        .groupby("gene_id")
        .size()
        .reindex(genes, fill_value=0)
        .sort_values(ascending=False)
    )

    n_window = int(in_window.sum())
    n_promoter = int(in_promoter.sum())
    n_promoter_outside = n_promoter - n_window

    density_window = n_window / (len(genes) * WINDOW_LENGTH)
    density_promoter_outside = n_promoter_outside / (
        len(genes) * (PROMOTER_LENGTH - WINDOW_LENGTH)
    )

    print(f"\n=== {dataset_name} ===")
    print(f"genes:                       {len(genes)}")
    print(f"SNPs total (3020 bp):        {len(variation)}")
    print(f"SNPs in promoter (1-1500):   {n_promoter}")
    print(f"SNPs in window ({WINDOW_START}-{WINDOW_END}):  {n_window}")
    print(f"genes with 0 in window:      {int((counts_in_window == 0).sum())}")
    print(f"\nper-gene SNP count in the {WINDOW_LENGTH} bp window:")
    print(counts_in_window.describe().to_string())
    print("\nSNP density (per bp per gene):")
    print(f"  window:               {density_window:.5f}")
    print(f"  promoter outside win: {density_promoter_outside:.5f}")
    if density_promoter_outside > 0:
        ratio = density_window / density_promoter_outside
        print(f"  ratio window/outside: {ratio:.3f}")
    else:
        print("  ratio window/outside: undefined (no SNPs outside the window)")


def main() -> None:
    """Summarize window occupancy for every configured VCF directory."""
    for dataset_name, vcf_directory in VCF_DIRECTORIES.items():
        summarize(dataset_name, load_variation(vcf_directory))


if __name__ == "__main__":
    main()
