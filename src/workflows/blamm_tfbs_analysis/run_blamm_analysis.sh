#!/bin/bash
# Run the blamm TFBS count analysis for all evolutionary runs of the paper,
# at maximum mutation count and at exactly 5 mutations.
#
# Outputs land in one folder per (run, mutation count) next to this script:
#   <run_name>__mutations_MAX/
#   <run_name>__mutations_5/
# Each folder holds motif_counts_per_gene.csv, motif_counts_aggregate.csv,
# motif_significance.csv, selected_entries.csv, skipped_genes.csv,
# run_parameters.json, console.log and the blamm working directory.
#
# motif_significance.csv tests, per motif, whether the run changed that motif's
# occurrence count across genes (Wilcoxon signed-rank vs 0, BH-corrected across
# motifs). It can be recomputed for existing output folders without re-scanning:
#   conda run -n deepCREshap python -m analysis.blamm.blamm_significance \
#       -o <output_folder> [<output_folder> ...]
#
# Usage: bash run_blamm_analysis.sh

set -euo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="deepCREshap"

MOTIF_DB_DIR="/home/gernot/Code/PhD_Code/moca-pipeline/data/motif_databases.12.27/motif_databases"
MEME_FILES=(
    "${MOTIF_DB_DIR}/JASPAR/JASPAR2024_CORE_plants_non-redundant_v2.meme"
    "${MOTIF_DB_DIR}/ARABD/ArabidopsisDAPv1.meme"
    "${MOTIF_DB_DIR}/ARABD/ArabidopsisPBM_20140210.meme"
)
MOTIF_DIR="${WORKFLOW_DIR}/motifs"
MOTIFS_JASPAR="${MOTIF_DIR}/plant_tf_motifs.jaspar"
MOTIF_METADATA="${MOTIF_DIR}/plant_tf_motifs_metadata.csv"

P_VALUE="0.0001"

# blamm manages its own thread pool; the blamm documentation recommends
# disabling multithreading inside the BLAS library itself.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

RUN_FOLDERS=(
    "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/GOF_LOF/GOF/GOF_single_mutation_251009_121226_109368"
    "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/GOF_LOF/LOF/LOF_single_mutation_251020_180028_564570"
    "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/paper_runs/single_mutation/arabidopsis_toolkit_msr_max_single_260224_155640_550482"
    "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/paper_runs/single_mutation/arabidopsis_toolkit_msr_min_single_260224_115131_937839"
)
MUTATION_COUNTS=("MAX" "5")

echo "Preparing motif database"
mkdir -p "${MOTIF_DIR}"
conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.blamm.meme_to_jaspar \
    --meme-files "${MEME_FILES[@]}" \
    --output-jaspar "${MOTIFS_JASPAR}" \
    --output-metadata "${MOTIF_METADATA}"

for run_folder in "${RUN_FOLDERS[@]}"; do
    run_name="$(basename "${run_folder}")"
    for mutation_count in "${MUTATION_COUNTS[@]}"; do
        output_folder="${WORKFLOW_DIR}/${run_name}__mutations_${mutation_count}"
        mkdir -p "${output_folder}"
        echo "=== ${run_name} | mutations=${mutation_count} ==="
        conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.blamm.blamm_tfbs_counts \
            --run-folder "${run_folder}" \
            --motifs "${MOTIFS_JASPAR}" \
            --motif-metadata "${MOTIF_METADATA}" \
            --output "${output_folder}" \
            --mutation-count "${mutation_count}" \
            --p-value "${P_VALUE}" \
            2>&1 | tee "${output_folder}/console.log"
    done
done

echo "All analyses complete."
