import json
import os
import random
from typing import Dict, List, Optional

import pandas as pd
from evolution.extract_sequences import CENTRAL_PADDING, extract_string, find_genes
from pyfaidx import Fasta


from workflows.overlap_analysis.starrseq_deepcre_correlation_WRKY import wrky_mapping_results
from workflows.overlap_analysis.starrseq_deepcre_correlation_bHLH import bhlh_mapping_results

DATA = "/home/gernot/Code/PhD_Code/Evolution/data/starrseq_v2"
ARC = "/home/gernot/ARCitect/ARCs/genRE/assays/Gene_Data/dataset"
NTAB_BUILD = "/home/gernot/Code/PhD_Code/PhD_Tools/pseudo_chromosomes/nicotiana_build"

# hand-written gene lists; absent for now, so the flowering blocks are skipped
FLOWERING_GENES_ARA_PATH = DATA + "/ara_flowering_genes.json"
FLOWERING_GENES_NTAB_PATH = DATA + "/ntab_flowering_genes.json"
GOF_GENES_ARA_PATH = "/home/gernot/Code/PhD_Code/Evolution/data/Arabidopsis_GOF_extracted_genes.fa"
LOF_GENES_ARA_PATH = "/home/gernot/Code/PhD_Code/Evolution/data/Arabidopsis_LOF_extracted_genes.fa"
TARGET_GENE_COUNT_NTAB = 50
TARGET_GENE_COUNT_ARA = 50
ARA_GENOME = ARC + "/genomes/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"
NATB_GENOME = NTAB_BUILD + "/ntab_pseudo.fa"
ARA_ANNOTATION = ARC + "/annotations/Arabidopsis_thaliana.TAIR10.52.gtf"
NTAB_ANNOTATION = NTAB_BUILD + "/ntab_pseudo.gtf"
ARA_VCF = ARC + "/1001AraVCFs_div_panel/vcf_downloads/combined/all_snps_combined.vcf"
GENE_NAME_ATTRIBUTE = "gene_id"
FEATURE_TYPE_FILTER = ["gene"]
SEED = 20260921
INTRAGENIC = 500
EXTRAGENIC = 1000

ARA_FLOWERING_FASTA = DATA + "/ara_flowering_extracted_genes.fa"
NTAB_FLOWERING_FASTA = DATA + "/ntab_flowering_extracted_genes.fa"
GENE_METADATA_PATH = DATA + "/candidate_gene_metadata.csv"


def draw_random_genes(
    annotation_path: str, exclude_genes: List[str], n_genes: int, seed: int
) -> List[str]:
    """Randomly draw genes from an annotation, excluding a given set.

    Args:
        annotation_path: Genome annotation file to draw candidate genes from.
        exclude_genes: Genes already selected elsewhere; not drawn again.
        n_genes: Number of genes to draw.
        seed: Seed of the draw.

    Returns:
        The drawn gene ids; empty if ``n_genes <= 0``.

    Raises:
        ValueError: If the annotation does not hold enough further genes.
    """
    if n_genes <= 0:
        return []
    # find_genes appends "_<feature type>" to every id (e.g. "AT1G01010_gene")
    suffix = "_" + FEATURE_TYPE_FILTER[0]
    all_genes = find_genes(
        annotation_path, GENE_NAME_ATTRIBUTE, FEATURE_TYPE_FILTER, genes_of_interest=[]
    )["gene_id"].str.removesuffix(suffix).unique()
    candidates = [gene_id for gene_id in all_genes if gene_id not in exclude_genes]
    if len(candidates) < n_genes:
        raise ValueError(
            "{} holds {} spare genes, need {}".format(
                annotation_path, len(candidates), n_genes
            )
        )
    return random.Random(seed).sample(candidates, n_genes)


def extract_genes(
    genome_path: str,
    annotation_path: str,
    gene_ids: List[str],
    output_path: str,
    vcf_paths: Optional[List[str]] = None,
) -> None:
    """Extract the flanking-region frames for a gene list and write them to FASTA.

    Args:
        genome_path: Genome FASTA the sequence is read from.
        annotation_path: Annotation the gene coordinates are read from.
        gene_ids: Genes to extract.
        output_path: FASTA path the frames are written to.
        vcf_paths: VCF files restricting the frame to natural variants; empty
            for no restriction.
    """
    fasta = Fasta(genome_path, as_raw=True, sequence_always_upper=True, read_ahead=10000)
    gene_df = find_genes(annotation_path, GENE_NAME_ATTRIBUTE, FEATURE_TYPE_FILTER, gene_ids)
    extract_string(
        fasta, gene_df, INTRAGENIC, EXTRAGENIC, output_path, CENTRAL_PADDING, vcf_paths or []
    )

def gene_ids_from_fasta(fasta_path: str) -> List[str]:
    """Read the gene ids out of an already-extracted frame FASTA."""
    fasta = Fasta(fasta_path)
    return [name.split("_gene:")[0].split("_", 1)[1] for name in fasta.keys()]


def add_to_metadata(
    metadata: pd.DataFrame, gene_ids: List[str], species: str, reason: str
) -> pd.DataFrame:
    """Append one group of genes to the gene-tracking table."""
    new_rows = pd.DataFrame({"gene_id": gene_ids, "species": species, "reason": reason})
    return pd.concat([metadata, new_rows], ignore_index=True)


def extract_all_candidate_genes() -> List[Dict]:
    metadata = pd.DataFrame(columns=["gene_id", "species", "reason"])
    old_fragment_mappings = pd.DataFrame(wrky_mapping_results() + bhlh_mapping_results())

    # Extract ara genes
    flowering_genes_ara = []
    if os.path.isfile(FLOWERING_GENES_ARA_PATH):
        with open(FLOWERING_GENES_ARA_PATH) as f:
            flowering_genes_ara = json.load(f)
    else:
        print("warning: no {}, extracting without flowering genes".format(
            FLOWERING_GENES_ARA_PATH
        ))
    metadata = add_to_metadata(metadata, flowering_genes_ara, "arabidopsis", "flowering")

    gof_genes = gene_ids_from_fasta(GOF_GENES_ARA_PATH)
    metadata = add_to_metadata(metadata, gof_genes, "arabidopsis", "GOF")
    lof_genes = gene_ids_from_fasta(LOF_GENES_ARA_PATH)
    metadata = add_to_metadata(metadata, lof_genes, "arabidopsis", "LOF")

    # dict.fromkeys removes duplicates while preserving order
    chosen_genes_ara = list(dict.fromkeys(flowering_genes_ara + gof_genes + lof_genes))
    random_genes_ara = draw_random_genes(
        ARA_ANNOTATION, chosen_genes_ara, TARGET_GENE_COUNT_ARA - len(chosen_genes_ara), SEED
    )
    metadata = add_to_metadata(metadata, random_genes_ara, "arabidopsis", "random")

    all_genes_ara = chosen_genes_ara + random_genes_ara
    extract_genes(
        ARA_GENOME, ARA_ANNOTATION, all_genes_ara, ARA_FLOWERING_FASTA, [ARA_VCF]
    )

    # Extract ntab genes
    flowering_genes_ntab = []
    if os.path.isfile(FLOWERING_GENES_NTAB_PATH):
        with open(FLOWERING_GENES_NTAB_PATH) as f:
            flowering_genes_ntab = json.load(f)
    else:
        print("warning: no {}, extracting without flowering genes".format(
            FLOWERING_GENES_NTAB_PATH
        ))
    metadata = add_to_metadata(metadata, flowering_genes_ntab, "ntab", "flowering")

    random_genes_ntab = draw_random_genes(
        NTAB_ANNOTATION,
        flowering_genes_ntab,
        TARGET_GENE_COUNT_NTAB - len(flowering_genes_ntab),
        SEED,
    )
    metadata = add_to_metadata(metadata, random_genes_ntab, "ntab", "random")

    all_genes_ntab = flowering_genes_ntab + random_genes_ntab
    extract_genes(NATB_GENOME, NTAB_ANNOTATION, all_genes_ntab, NTAB_FLOWERING_FASTA)

    if not metadata.empty:
        metadata.to_csv(GENE_METADATA_PATH, index=False)
    return old_fragment_mappings


def main():
    extract_all_candidate_genes()



if __name__ == "__main__":
    main()