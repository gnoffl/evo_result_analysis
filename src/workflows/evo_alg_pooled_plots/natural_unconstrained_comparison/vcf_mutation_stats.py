"""Statistics on mutation counts in natural variation VCF files.

Counts unique mutation locations and total mutations per gene,
then reports averages and std for GOF, LOF, and pooled.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

GOF_DIR = Path("/home/gernot/Code/PhD_Code/Evolution/data/Arabidopsis_GOF_reextracted_vcfs")
LOF_DIR = Path("/home/gernot/Code/PhD_Code/Evolution/data/Arabidopsis_LOF_reextracted_vcfs")
OUTPUT_DIR = Path(__file__).parent / "vcf_mutation_stats"


def count_mutations_in_vcf(vcf_path: Path) -> Tuple[int, int]:
    """Count unique locations and total mutations in a VCF file.

    Args:
        vcf_path: Path to VCF file.

    Returns:
        Tuple of (unique_locations, total_mutations).
    """
    positions = []
    for line in vcf_path.read_text().splitlines():
        if line.startswith("#"):
            continue
        pos = int(line.split("\t")[1])
        positions.append(pos)
    unique_locations = len(set(positions))
    total_mutations = len(positions)
    return unique_locations, total_mutations


def collect_stats(vcf_dir: Path) -> Tuple[List[int], List[int]]:
    """Collect location and mutation counts from all VCFs in a directory.

    Args:
        vcf_dir: Directory containing VCF files.

    Returns:
        Tuple of (all_unique_locations, all_total_mutations) as lists over genes.
    """
    all_unique_locations = []
    all_total_mutations = []
    for vcf_file in sorted(vcf_dir.glob("*.vcf")):
        unique_locations, total_mutations = count_mutations_in_vcf(vcf_file)
        all_unique_locations.append(unique_locations)
        all_total_mutations.append(total_mutations)
    return all_unique_locations, all_total_mutations


def build_summary_row(
    label: str, unique_locations: List[int], total_mutations: List[int]
) -> dict:
    """Build a summary statistics dict for one group of genes.

    Args:
        label: Group label (e.g. 'GOF', 'LOF', 'Pooled').
        unique_locations: Unique location counts per gene.
        total_mutations: Total mutation counts per gene.

    Returns:
        Dict with summary statistics for the group.
    """
    locs = np.array(unique_locations)
    muts = np.array(total_mutations)
    return {
        "group": label,
        "n_genes": len(locs),
        "unique_locations_mean": round(locs.mean(), 1),
        "unique_locations_std": round(locs.std(), 1),
        "unique_locations_min": int(locs.min()),
        "unique_locations_max": int(locs.max()),
        "total_mutations_mean": round(muts.mean(), 1),
        "total_mutations_std": round(muts.std(), 1),
        "total_mutations_min": int(muts.min()),
        "total_mutations_max": int(muts.max()),
    }


if __name__ == "__main__":
    gof_locs, gof_muts = collect_stats(GOF_DIR)
    lof_locs, lof_muts = collect_stats(LOF_DIR)
    pooled_locs = gof_locs + lof_locs
    pooled_muts = gof_muts + lof_muts

    rows = [
        build_summary_row("GOF", gof_locs, gof_muts),
        build_summary_row("LOF", lof_locs, lof_muts),
        build_summary_row("Pooled", pooled_locs, pooled_muts),
    ]

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "vcf_mutation_stats.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
