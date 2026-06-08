"""bHLH STARR-seq x deepCRE correlation, mirroring the WRKY analysis.

This is the bHLH sibling of ``starrseq_deepcre_correlation_WRKY.py``. It reuses
all TF-agnostic logic from :mod:`_common` and only carries the bHLH-specific
inputs and orchestration:

- Reference gene windows are produced fresh by invoking
  ``evolution/extract_sequences.py`` on first run (see :func:`ensure_refs_fasta`)
  instead of being read from a precomputed deepCRE run folder.
- The mapping CSV uses ``site_id``/``gene_id`` columns (see
  ``bHLH_overlaps/generate_bhlh_overlaps.py``).
- The difference filter tolerates one extra mutation (16 vs. WRKY's 15) to
  account for the 6 nt bHLH core motif vs. WRKY's 5 nt motif.

Only the new light+dark STARR-seq CSV is consumed; there is no ``USE_NEW_DATA``
toggle and no ``correlation_bHLH/old/`` output path.
"""
import os
import sys
import json
import subprocess
from typing import Dict, List

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tensorflow.keras.models import load_model  # type: ignore

from evolution.sequences import one_hot_encode

from workflows.overlap_analysis import _common

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

STARRSEQ_INPUT_FILE = os.path.join(DATA_DIR, "dCIS_bHLH_in_silico_mutated_GS2025d.fasta")
MAPPING_FILE = os.path.join(DATA_DIR, "bHLH_dCIS_dCRE_overlaps.csv")
REFS_FASTA_PATH = os.path.join(DATA_DIR, "bHLH_reference_sequences.fa")
GENES_JSON_PATH = os.path.join(DATA_DIR, "bHLH_genes_of_interest.json")
STARR_SEQ_RESULTS = os.path.join(DATA_DIR, "plantstarr-seq_main_light_and_dark_simon_gernot.csv")
GENOME_FASTA_PATH = "/home/gernot/ARCitect/ARCs/genRE/assays/Gene_Data/dataset/genomes/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"
GTF_PATH = "/home/gernot/ARCitect/ARCs/genRE/assays/Gene_Data/dataset/annotations/Arabidopsis_thaliana.TAIR10.52.gtf"
CORRELATION_OUTPUT_ROOT = os.path.join(BASE_DIR, "correlation_bHLH", "new")

# bHLH core motif is 6 nt (vs. WRKY's 5 nt), so one extra mutation is tolerated.
MAX_DIFFERENCES = 16


def ensure_refs_fasta(
    refs_path: str,
    genes_json_path: str,
    mapping_csv_path: str,
    genome_fasta: str,
    gtf: str,
) -> None:
    """Extract the bHLH reference-window FASTA on first run; skip if it exists.

    The bHLH pipeline has no precomputed deepCRE run folder, so the per-gene
    3020 bp reference windows are produced by invoking
    ``evolution.extract_sequences`` as a subprocess. The list of genes is taken
    from the mapping CSV's ``gene_id`` column (written to ``genes_json_path`` if
    it does not already exist). The extraction defaults (``--intragenic 500``,
    ``--extragenic 1000``) match the rest of the pipeline.

    Args:
        refs_path: Output FASTA path for the extracted reference windows.
        genes_json_path: Path to the genes-of-interest JSON (created if absent).
        mapping_csv_path: bHLH overlap mapping CSV providing the gene ids.
        genome_fasta: Path to the genome FASTA passed to ``extract_sequences``.
        gtf: Path to the genome annotation passed to ``extract_sequences``.

    Raises:
        subprocess.CalledProcessError: If the extraction subprocess fails.
    """
    if os.path.exists(refs_path):
        return  # idempotent: assume the file is good if present
    if not os.path.exists(genes_json_path):
        ids = sorted(set(pd.read_csv(mapping_csv_path)["gene_id"]))
        with open(genes_json_path, "w") as fh:
            json.dump(ids, fh)
    subprocess.run(
        [
            sys.executable, "-m", "evolution.extract_sequences",
            "-f", genome_fasta,
            "-a", gtf,
            "-o", refs_path,
            "--genes_of_interest", genes_json_path,
        ],
        check=True,
    )


def load_bhlh_window_candidates(refs_fasta_path: str, model_path: str) -> List[Dict]:
    """Load bHLH reference windows and their deepCRE reference fitness.

    Parses each FASTA header ``{chrom}_{gene_id}_gene:{start}-{end}`` using the
    same logic as the WRKY folder-name loader, then computes ``ref_fitness`` for
    every reference sequence in a single batched ``model.predict`` pass (the
    bHLH equivalent of WRKY's ``pareto_front.json`` reference fitness).

    Args:
        refs_fasta_path: FASTA of extracted 3020 bp reference windows.
        model_path: Path to the deepCRE model used to score references.

    Returns:
        One gene-window dict per FASTA record, with keys ``gene``, ``ref_seq``,
        ``start``, ``end``, ``folder_name`` and ``ref_fitness``.
    """
    refs_fasta = Fasta(refs_fasta_path)
    gene_data: List[Dict] = []
    for record in refs_fasta:
        header = record.name
        gene_name = header.split("_")[1]
        start, end = header.split("_")[2].split(":")[1].split("-")
        if start > end:
            start, end = end, start
        gene_data.append({
            "gene": gene_name,
            "ref_fitness": None,
            "folder_name": header,
            "ref_seq": str(record),
            "start": int(start),
            "end": int(end),
        })

    # Batch ref_fitness in a single forward pass over all reference sequences.
    model = load_model(model_path)
    encoded = np.array([one_hot_encode(gene["ref_seq"]) for gene in gene_data])
    predictions = model.predict(encoded).flatten().tolist()
    for gene, ref_fitness in zip(gene_data, predictions):
        gene["ref_fitness"] = ref_fitness
    return gene_data


def build_site_to_gene_ids(mapping_candidates: pd.DataFrame) -> Dict[str, List[str]]:
    """Map each ``site_id`` to its unique candidate ``gene_id`` list.

    Short genes can produce duplicate ``(site_id, gene_id)`` rows in the mapping
    CSV (one per overlapping window); ``.unique()`` collapses them so each
    (site, gene) pair is aligned only once downstream.

    Args:
        mapping_candidates: The bHLH overlap mapping table with ``site_id`` and
            ``gene_id`` columns.

    Returns:
        Mapping from site key to its deduplicated list of candidate gene ids.
    """
    return (
        mapping_candidates.groupby("site_id")["gene_id"]
        .apply(lambda site_genes: list(site_genes.unique()))
        .to_dict()
    )


def prepare_bhlh_enrichment_df() -> pd.DataFrame:
    """Load bHLH inputs and run the prediction pipeline into an analysis-ready df.

    Performs the TF-specific data joining: ensure the reference-window FASTA is
    extracted, parse the bHLH STARR-seq variants and reference windows, map
    STARR-seq fragments onto reference windows, build and score the mutated
    sequences (bHLH's ``MAX_DIFFERENCES`` of 16, to allow for the 6 nt core
    motif), merge in STARR-seq enrichment, and compute deltas and overlap
    buckets.

    Does not configure matplotlib or the output root; the caller owns those so
    the same dataframe can feed either the bHLH-only or the pooled analysis.

    Returns:
        Enrichment dataframe ready for :func:`_common.run_correlation_analysis`.
    """
    ensure_refs_fasta(REFS_FASTA_PATH, GENES_JSON_PATH, MAPPING_FILE, GENOME_FASTA_PATH, GTF_PATH)
    starrseq_data = _common.load_starrseq_data(STARRSEQ_INPUT_FILE)
    gene_data = load_bhlh_window_candidates(REFS_FASTA_PATH, _common.DEEPCRE_PATH)
    mapping_candidates = pd.read_csv(MAPPING_FILE)
    site_to_gene_ids = build_site_to_gene_ids(mapping_candidates)
    starr_seq_results = pd.read_csv(STARR_SEQ_RESULTS)
    mapping_results = _common.map_starrseq_to_deepcre(starrseq_data, gene_data, site_to_gene_ids)
    seqs, meta_data = _common.build_sequences(mapping_results, max_differences=MAX_DIFFERENCES)
    prediction_df = _common.make_deepcre_predictions(seqs, meta_data)
    enrichment_df = _common.merge_with_starrseq_results(prediction_df, starr_seq_results)
    enrichment_df = enrichment_df.dropna(subset=["enrichment"]).reset_index(drop=True)
    enrichment_df = _common.calculate_deltas(enrichment_df)
    enrichment_df = _common.add_length_corrected_overlap_buckets(enrichment_df)
    return enrichment_df


def main():
    _common.configure_matplotlib()
    _common.set_output_root(CORRELATION_OUTPUT_ROOT)
    _common.run_correlation_analysis(prepare_bhlh_enrichment_df())


if __name__ == "__main__":
    main()
