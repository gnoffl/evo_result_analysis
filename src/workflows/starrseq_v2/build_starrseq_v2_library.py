import json
import os
import random
from typing import List, Optional, Tuple

import pandas as pd
from evolution.extract_sequences import CENTRAL_PADDING, extract_string, find_genes
from pyfaidx import Fasta


from workflows.overlap_analysis.starrseq_deepcre_correlation_WRKY import wrky_mapping_results
from workflows.overlap_analysis.starrseq_deepcre_correlation_bHLH import bhlh_mapping_results

DATA = "/home/gernot/Code/PhD_Code/Evolution/data/starrseq_v2"
ARC = "/home/gernot/ARCitect/ARCs/genRE/assays/Gene_Data/dataset"

# FLOR-ID list and its N. tabacum orthologs, both built in flowering_data/
FLOWERING_GENES_ARA_PATH = os.path.join(
    os.path.dirname(__file__), "flowering_data", "ara_flowering_genes_florid.json"
)
FLOWERING_GENES_NTAB_PATH = os.path.join(
    os.path.dirname(__file__), "flowering_data", "ntab_flowering_genes.json"
)
GOF_GENES_ARA_PATH = "/home/gernot/Code/PhD_Code/Evolution/data/Arabidopsis_GOF_extracted_genes.fa"
LOF_GENES_ARA_PATH = "/home/gernot/Code/PhD_Code/Evolution/data/Arabidopsis_LOF_extracted_genes.fa"
TARGET_GENE_COUNT_NTAB = 500
TARGET_GENE_COUNT_ARA = 500
ARA_GENOME = ARC + "/genomes/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"
NATB_GENOME = ARC + "/genomes/nicotiana_tabacum.fa"
ARA_ANNOTATION = ARC + "/annotations/Arabidopsis_thaliana.TAIR10.52.gtf"
NTAB_ANNOTATION = ARC + "/annotations/hlx-Nicotiana_tabacum-GCF_000715135.1-4097.agat.gtf"
ARA_VCF = ARC + "/1001AraVCFs_div_panel/vcf_downloads/combined/all_snps_combined.vcf"
GENE_NAME_ATTRIBUTE = "gene_id"
FEATURE_TYPE_FILTER = ["gene"]
SEED = 20260921
ARM_SUBSET_FRACTION = 0.2
ARM_SUBSET_MIN_GENES = 50
FULL_GRID_SUBSET_FRACTION = 0.2
FULL_GRID_SUBSET_MIN_GENES = 20
# off-target windows avoid the central 1500-1519 region of the 3020 bp frame
STARRSEQ_WINDOW_LENGTH = 170
RANDOM_WINDOW_STARTS = list(range(0, 1330)) + list(range(1520, 2850))
INTRAGENIC = 500
EXTRAGENIC = 1000
# insert site of the extracted construct, verified against construct_1_extracted.fa
INSERT_START = 780
INSERT_END = 950
INSERT_LENGTH = INSERT_END - INSERT_START
BACKGROUND_PATH = "/home/gernot/Code/PhD_Code/evo_result_analysis/src/workflows/overlap_analysis/correct_construct/construct_1_extracted.fa"

ARA_FLOWERING_GOF_FASTA = DATA + "/ara_flowering_GOF_LOF_extracted_genes.fa"
NTAB_FLOWERING_FASTA = DATA + "/ntab_flowering_extracted_genes.fa"
STARRSEQ_V1_FASTA = DATA + "/starrseq_v1_extracted_genes.fa"
GENE_METADATA_PATH = DATA + "/candidate_gene_metadata.csv"
ARA_FLOWERING_CONSTRUCT_PATH = DATA + "/ara_flowering_GOF_constructs.fa"
NTAB_FLOWERING_CONSTRUCT_PATH = DATA + "/ntab_flowering_GOF_constructs.fa"
ARA_FLOWERING_OFF_TARGET_PATH = DATA + "/ara_flowering_GOF_off_target_constructs.fa"
NTAB_FLOWERING_OFF_TARGET_PATH = DATA + "/ntab_flowering_GOF_off_target_constructs.fa"
STARRSEQ_V1_OFF_TARGET_PATH = DATA + "/starrseq_v1_off_target_constructs.fa"
STARRSEQ_MAPPING_PATH = DATA + "/starrseq_v1_mapping.csv"


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


def draw_subset_genes(
    eligible: pd.DataFrame, fraction: float, min_genes: int, seed: int
) -> List[str]:
    """Draw a fixed-size gene sample per species from the eligible rows.

    Args:
        eligible: Rows the draw runs on; columns gene_id and species.
        fraction: Share of each species' unique genes to draw.
        min_genes: Lower bound on the per-species sample size.
        seed: Seed of the draw.

    Returns:
        The drawn gene ids across all species.
    """
    rng = random.Random(seed)
    drawn_genes = []
    for _, rows in eligible.groupby("species"):
        genes = rows["gene_id"].unique()
        if len(genes) <= min_genes:
            raise ValueError(
                f"species {rows['species'].iloc[0]} has only {len(genes)} eligible genes, need at least {min_genes}"
            )
        subset_size = min(len(genes), max(min_genes, round(fraction * len(genes))))
        mask = [True] * subset_size + [False] * (len(genes) - subset_size)
        rng.shuffle(mask)
        drawn_genes.extend(gene for gene, drawn in zip(genes, mask) if drawn)
    return drawn_genes


def select_arm_subset(metadata: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Draw the arm subset per species over unique genes (starrseq_v1 excluded).

    Args:
        metadata: Gene-tracking table with columns gene_id, species, reason.
        seed: Seed of the draw.

    Returns:
        ``metadata`` with an added boolean column ``arm_subset``; every row of a
        drawn gene is flagged.
    """
    eligible = metadata[metadata["reason"] != "starrseq_v1"]
    arm_genes = draw_subset_genes(eligible, ARM_SUBSET_FRACTION, ARM_SUBSET_MIN_GENES, seed)
    metadata["arm_subset"] = metadata["gene_id"].isin(arm_genes)
    return metadata


def select_full_grid_subset(metadata: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Draw the full-grid subset per species out of the arm subset.

    Args:
        metadata: Gene-tracking table already carrying ``arm_subset``.
        seed: Seed of the draw.

    Returns:
        ``metadata`` with an added boolean column ``full_grid_subset``.
    """
    eligible = metadata[metadata["arm_subset"]]
    grid_genes = draw_subset_genes(
        eligible, FULL_GRID_SUBSET_FRACTION, FULL_GRID_SUBSET_MIN_GENES, seed
    )
    metadata["full_grid_subset"] = metadata["gene_id"].isin(grid_genes)
    return metadata


def add_v1_windows(
    metadata: pd.DataFrame, starrseq_v1_mapping: pd.DataFrame
) -> pd.DataFrame:
    """Attach the STARR-seq v1 mutated windows as alternative_start/_end columns.

    Args:
        metadata: Gene-tracking table.
        starrseq_v1_mapping: v1 fragment-to-gene mapping with gene, overlap_start
            and overlap_end (offsets into the gene's reference window).

    Returns:
        ``metadata`` with one row per (gene row, mutated window); genes without a
        v1 window keep a single row with NaN in both columns.
    """
    windows = starrseq_v1_mapping[["gene", "overlap_start", "overlap_end"]].drop_duplicates()
    windows = windows.rename(columns={
        "gene": "gene_id",
        "overlap_start": "alternative_start",
        "overlap_end": "alternative_end",
    })
    windows["reason"] = "starrseq_v1"
    return metadata.merge(windows, on=["gene_id", "reason"], how="left")


def add_random_windows(metadata: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Fill alternative_start/_end of the arm rows that carry no v1 window.

    One window per gene, drawn uniformly over the two allowed start ranges.

    Args:
        metadata: Gene-tracking table carrying arm_subset and the alternative
            window columns.
        seed: Seed of the draw.

    Returns:
        ``metadata`` with the missing arm windows filled in.
    """
    missing = metadata["arm_subset"] & metadata["alternative_start"].isna()
    rng = random.Random(seed)
    starts = {}
    for gene_id in metadata.loc[missing, "gene_id"].unique():
        offset = rng.randrange(len(RANDOM_WINDOW_STARTS))
        starts[gene_id] = RANDOM_WINDOW_STARTS[offset]
    metadata.loc[missing, "alternative_start"] = metadata.loc[missing, "gene_id"].map(starts)
    metadata.loc[missing, "alternative_end"] = (
        metadata.loc[missing, "alternative_start"] + STARRSEQ_WINDOW_LENGTH
    )
    # nullable int: rows outside the arm subset keep a missing window
    metadata[["alternative_start", "alternative_end"]] = (
        metadata[["alternative_start", "alternative_end"]].astype("Int64")
    )
    return metadata


def extract_ntab(metadata):
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
    return metadata

def extract_ara(metadata):
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
    excluded_genes_ara = list(dict.fromkeys(metadata["gene_id"].to_list()))
    random_genes_ara = draw_random_genes(
        ARA_ANNOTATION, excluded_genes_ara, TARGET_GENE_COUNT_ARA - len(chosen_genes_ara), SEED
    )
    metadata = add_to_metadata(metadata, random_genes_ara, "arabidopsis", "random")

    all_genes_ara = chosen_genes_ara + random_genes_ara
    extract_genes(
        ARA_GENOME, ARA_ANNOTATION, all_genes_ara, ARA_FLOWERING_GOF_FASTA, [ARA_VCF]
    )
    return metadata


def extract_starrseq_v1(metadata):
    starrseq_v1_mapping = pd.DataFrame(wrky_mapping_results() + bhlh_mapping_results())
    starrseq_v1_genes = list(dict.fromkeys(starrseq_v1_mapping["gene"]))
    metadata = add_to_metadata(metadata, starrseq_v1_genes, "arabidopsis", "starrseq_v1")
    extract_genes(
        ARA_GENOME, ARA_ANNOTATION, starrseq_v1_genes, STARRSEQ_V1_FASTA, [ARA_VCF]
    )
    return metadata, starrseq_v1_mapping

def extract_all_candidate_genes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.DataFrame(columns=["gene_id", "species", "reason"])
    metadata, starrseq_v1_mapping = extract_starrseq_v1(metadata)
    metadata = extract_ara(metadata)
    metadata = extract_ntab(metadata)

    return metadata, starrseq_v1_mapping


def splice_correct_windows(gene_fasta_path: str, output_path: str) -> None:
    """Splice each gene frame's correct window into the construct backbone.

    Args:
        gene_fasta_path: Extracted gene frames of 3020 bp.
        output_path: FASTA written, one construct per gene frame.
    """
    background = str(Fasta(BACKGROUND_PATH)[0])
    placeholder = "N" * INSERT_LENGTH
    frames = Fasta(gene_fasta_path)
    with open(output_path, "w") as output_file:
        for gene_id in frames.keys():
            insert = str(frames[gene_id])[INSERT_START:INSERT_END]
            output_file.write(f">{gene_id}\n{background.replace(placeholder, insert)}\n")


def assign_directions(metadata: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add a ``direction`` column of "maximize"/"minimize".

    GOF -> max, LOF -> min, rest is random, but equally distributed over arm and full grid subsets per species.

    Args:
        metadata: Gene-tracking table carrying reason, arm_subset and
            full_grid_subset.
        seed: Seed of the draw.
    """
    metadata["direction"] = pd.NA
    metadata.loc[metadata["reason"] == "GOF", "direction"] = "maximize"
    metadata.loc[metadata["reason"] == "LOF", "direction"] = "minimize"

    rng = random.Random(seed)
    free = metadata["direction"].isna()
    strata = [
        free & metadata["full_grid_subset"],
        free & metadata["arm_subset"] & ~metadata["full_grid_subset"],
        free & ~metadata["arm_subset"],
    ]
    species_strata = [
        stratum & (metadata["species"] == species)
        for stratum in strata
        for species in metadata["species"].unique()
    ]
    for stratum in species_strata:
        genes = metadata.loc[stratum, "gene_id"].unique()
        labels = (["maximize", "minimize"] * len(genes))[: len(genes)]
        rng.shuffle(labels)
        metadata.loc[stratum, "direction"] = metadata.loc[stratum, "gene_id"].map(
            dict(zip(genes, labels))
        )
    return metadata


def flag_full_length_windows(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add ``full_length_window``, True where the alternative window spans 170 bp."""
    full_length = (
        metadata["alternative_end"] - metadata["alternative_start"]
    ) == STARRSEQ_WINDOW_LENGTH
    metadata["full_length_window"] = full_length.fillna(False)
    return metadata


def map_genes_to_frames(frame_keys: List[str], gene_ids: List[str]) -> dict:
    """Map each gene id to its frame header.

    Headers are ``<chromosome>_<gene_id>_gene:<start>-<end>``; the chromosome
    can itself contain underscores, so the gene id is found by suffix match.

    Args:
        frame_keys: Headers of an extracted frame FASTA.
        gene_ids: Gene ids to look for.

    Returns:
        gene id -> header, for the genes present in the FASTA.
    """
    wanted = set(gene_ids)
    mapping = {}
    for key in frame_keys:
        parts = key.split("_gene:")[0].split("_")
        for start in range(len(parts)):
            candidate = "_".join(parts[start:])
            if candidate in wanted:
                mapping[candidate] = key
                break
    return mapping


def splice_off_target_windows(
    metadata: pd.DataFrame,
    gene_fasta_path: str,
    output_path: str,
    reason: Optional[str] = None,
) -> None:
    """Splice each alternative window of the metadata into the construct backbone.

    Only rows flagged ``full_length_window`` are used.

    Args:
        metadata: Gene-tracking table carrying alternative_start/alternative_end
            and full_length_window.
        gene_fasta_path: Extracted gene frames of 3020 bp.
        output_path: FASTA written, one construct per usable window.
        reason: Restricts the rows to this ``reason``; None uses all of them.
    """
    background = str(Fasta(BACKGROUND_PATH)[0])
    placeholder = "N" * INSERT_LENGTH
    frames = Fasta(gene_fasta_path)
    frames_by_gene = map_genes_to_frames(list(frames.keys()), metadata["gene_id"].to_list())
    usable = metadata["full_length_window"]
    if reason is not None:
        usable = usable & (metadata["reason"] == reason)
    # a gene annotated in both directions carries the same window twice, which
    # would write the same record name twice
    rows = metadata[usable].drop_duplicates(
        subset=["gene_id", "alternative_start", "alternative_end"]
    )
    with open(output_path, "w") as output_file:
        for _, row in rows.iterrows():
            key = frames_by_gene.get(row["gene_id"])
            if key is None:
                continue
            start, end = int(row["alternative_start"]), int(row["alternative_end"])
            insert = str(frames[key])[start:end]
            output_file.write(
                f">{key}_off_{start}\n{background.replace(placeholder, insert)}\n"
            )


def main():
    metadata, starrseq_v1_mapping = extract_all_candidate_genes()
    metadata = select_arm_subset(metadata, SEED)
    metadata = select_full_grid_subset(metadata, SEED)
    metadata = add_v1_windows(metadata, starrseq_v1_mapping)
    metadata = add_random_windows(metadata, SEED)
    metadata = assign_directions(metadata, SEED)
    metadata = flag_full_length_windows(metadata)
    splice_correct_windows(ARA_FLOWERING_GOF_FASTA, ARA_FLOWERING_CONSTRUCT_PATH)
    splice_correct_windows(NTAB_FLOWERING_FASTA, NTAB_FLOWERING_CONSTRUCT_PATH)
    splice_off_target_windows(metadata, ARA_FLOWERING_GOF_FASTA, ARA_FLOWERING_OFF_TARGET_PATH)
    splice_off_target_windows(metadata, NTAB_FLOWERING_FASTA, NTAB_FLOWERING_OFF_TARGET_PATH)
    splice_off_target_windows(
        metadata, STARRSEQ_V1_FASTA, STARRSEQ_V1_OFF_TARGET_PATH, reason="starrseq_v1"
    )
    metadata.to_csv(GENE_METADATA_PATH, index=False)
    starrseq_v1_mapping.to_csv(STARRSEQ_MAPPING_PATH, index=False)
    print(metadata.iloc[:20])
    print(metadata.iloc[-20:])
    # TODO: final length check, N counts, duplicates, nuclease sites, etc


if __name__ == "__main__":
    main()
