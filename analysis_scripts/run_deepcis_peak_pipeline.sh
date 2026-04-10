#!/bin/bash

# Script to run DeepCIS scan -> peak calling -> random-subset visualization
# Usage:
#   ./analysis_scripts/run_deepcis_peak_pipeline.sh <results_folder> <analysis_name> <output_folder> [options]

set -euo pipefail

# Color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 <results_folder> <analysis_name> <output_folder> [options]"
    echo ""
    echo "Required Arguments:"
    echo "  results_folder                  Folder containing gene directories with evolution results"
    echo "  analysis_name                   Name to identify this analysis run"
    echo "  output_folder                   Root output folder for all generated artifacts"
    echo ""
    echo "Optional Arguments (deepcis_scanner):"
    echo "  --model <path>                  Path to deepCIS model (.h5 or SavedModel directory)"
    echo "  --deepcis-window-size <int>     Sliding window size in bp"
    echo "  --deepcis-step <int>            Step size between windows in bp"
    echo "  --deepcis-batch-size <int>      Batch size for model inference"
    echo "  --overwrite                     Re-run deepcis_scanner even if output exists"
    echo ""
    echo "Optional Arguments (peak_scanner):"
    echo "  --signal-type <type>            Peak signal type: reference|max_mutated|difference|all (default: all)"
    echo "  --annotator-window-size <int>   PeakAnnotator window size in bp"
    echo "  --annotator-step-size <int>     PeakAnnotator step size in bp"
    echo "  --annotator-threshold-peak <f>  PeakAnnotator detection threshold"
    echo "  --annotator-sigma <f>           PeakAnnotator Gaussian sigma in bp"
    echo "  --annotator-lambda-weight <f>   PeakAnnotator lambda weight"
    echo "  --log-level <level>             Peak scanner log level (default: INFO)"
    echo ""
    echo "Optional Arguments (filters shared by peak + visualization):"
    echo "  --genes \"g1 g2\"               Genes to include (space- or comma-separated string)"
    echo "  --tfs \"tf_0 tf_1\"             TFs to include (space- or comma-separated string)"
    echo ""
    echo "Optional Arguments (deepcis_visualize):"
    echo "  --viz-format <fmt>              Plot format: png|pdf|svg|jpg|jpeg (default: png)"
    echo "  --no-highlight-padding          Disable padding background in plots"
    echo "  --no-highlight-peaks            Disable peak background overlays in plots"
    echo ""
    echo "Behavior:" 
    echo "  1) Runs deepcis_scanner"
    echo "  2) Runs peak_scanner (all signal types by default)"
    echo "  3) Clears visualization folder and runs deepcis_visualize with --random-subset"
    echo ""
    echo "Output layout inside <output_folder>:"
    echo "  deepcis_scan/deepcis_window_scan_<analysis_name>.csv"
    echo "  deepcis_scan/<analysis_name>_annotated_peaks_<signal_types>_<timestamp>.csv"
    echo "  deepcis_scan/deepcis_scan_plots/<gene>/*.png"
    exit 1
}

parse_list_to_array() {
    # Converts comma/space separated list into bash array by name reference.
    local raw="$1"
    local -n out_ref=$2
    local normalized

    normalized=$(echo "$raw" | tr ',' ' ')
    # shellcheck disable=SC2206
    out_ref=($normalized)
}

if [ $# -lt 3 ]; then
    usage
fi

RESULTS_FOLDER="$1"
ANALYSIS_NAME="$2"
OUTPUT_FOLDER="$3"
shift 3

MODEL_PATH="models/deepcis/deepCIS_model_chrom_1_model.h5"
DEEPCIS_WINDOW_SIZE=""
DEEPCIS_STEP_SIZE=""
DEEPCIS_BATCH_SIZE=""
OVERWRITE_FLAG=""

SIGNAL_TYPE="all"
ANNOTATOR_WINDOW_SIZE=""
ANNOTATOR_STEP_SIZE=""
ANNOTATOR_THRESHOLD_PEAK=""
ANNOTATOR_SIGMA=""
ANNOTATOR_LAMBDA_WEIGHT=""
LOG_LEVEL="INFO"

VIZ_FORMAT="png"
HIGHLIGHT_PADDING=true
HIGHLIGHT_PEAKS=true

GENES_RAW=""
TFS_RAW=""
declare -a GENES_ARR=()
declare -a TFS_ARR=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --deepcis-window-size)
            DEEPCIS_WINDOW_SIZE="$2"
            shift 2
            ;;
        --deepcis-step)
            DEEPCIS_STEP_SIZE="$2"
            shift 2
            ;;
        --deepcis-batch-size)
            DEEPCIS_BATCH_SIZE="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE_FLAG="--overwrite"
            shift
            ;;
        --signal-type)
            SIGNAL_TYPE="$2"
            shift 2
            ;;
        --annotator-window-size)
            ANNOTATOR_WINDOW_SIZE="$2"
            shift 2
            ;;
        --annotator-step-size)
            ANNOTATOR_STEP_SIZE="$2"
            shift 2
            ;;
        --annotator-threshold-peak)
            ANNOTATOR_THRESHOLD_PEAK="$2"
            shift 2
            ;;
        --annotator-sigma)
            ANNOTATOR_SIGMA="$2"
            shift 2
            ;;
        --annotator-lambda-weight)
            ANNOTATOR_LAMBDA_WEIGHT="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --genes)
            GENES_RAW="$2"
            shift 2
            ;;
        --tfs)
            TFS_RAW="$2"
            shift 2
            ;;
        --viz-format)
            VIZ_FORMAT="$2"
            shift 2
            ;;
        --no-highlight-padding)
            HIGHLIGHT_PADDING=false
            shift
            ;;
        --no-highlight-peaks)
            HIGHLIGHT_PEAKS=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

if [ ! -d "$RESULTS_FOLDER" ]; then
    echo -e "${RED}Error: Results folder '$RESULTS_FOLDER' does not exist${NC}"
    exit 1
fi

# Validate signal type upfront to fail fast.
case "$SIGNAL_TYPE" in
    reference|max_mutated|difference|all)
        ;;
    *)
        echo -e "${RED}Error: --signal-type must be one of reference|max_mutated|difference|all${NC}"
        exit 1
        ;;
esac

# Prepare output folder first so realpath always works.
mkdir -p "$OUTPUT_FOLDER"

RESULTS_FOLDER=$(realpath "$RESULTS_FOLDER")
OUTPUT_FOLDER=$(realpath "$OUTPUT_FOLDER")

# Ensure module execution works from repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resolve model path: absolute paths stay absolute, relative paths are repo-root relative.
if [[ "$MODEL_PATH" = /* ]]; then
    MODEL_PATH_RESOLVED="$MODEL_PATH"
else
    MODEL_PATH_RESOLVED="$REPO_ROOT/$MODEL_PATH"
fi

if [ ! -e "$MODEL_PATH_RESOLVED" ]; then
    echo -e "${RED}Error: Model path '$MODEL_PATH_RESOLVED' does not exist${NC}"
    exit 1
fi

MODEL_PATH=$(realpath "$MODEL_PATH_RESOLVED")

if [ -n "$GENES_RAW" ]; then
    parse_list_to_array "$GENES_RAW" GENES_ARR
fi
if [ -n "$TFS_RAW" ]; then
    parse_list_to_array "$TFS_RAW" TFS_ARR
fi

cd "$REPO_ROOT"

SCAN_DIR="$OUTPUT_FOLDER/deepcis_scan"
SCAN_BASENAME="deepcis_window_scan_${ANALYSIS_NAME}"
SCAN_CSV="$SCAN_DIR/${SCAN_BASENAME}.csv"
PEAK_DIR="$SCAN_DIR"
if [ "$SIGNAL_TYPE" = "all" ]; then
    PEAK_SIGNAL_LABEL="reference_max_mutated_difference"
else
    PEAK_SIGNAL_LABEL="$SIGNAL_TYPE"
fi
PEAK_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
PEAK_OUTPUT_FILE="$PEAK_DIR/${ANALYSIS_NAME}_annotated_peaks_${PEAK_SIGNAL_LABEL}_${PEAK_TIMESTAMP}.csv"
VIS_DIR="$SCAN_DIR/deepcis_scan_plots"

echo "=========================================="
echo "Starting DeepCIS Peak Pipeline"
echo "=========================================="
echo "Results Folder: $RESULTS_FOLDER"
echo "Analysis Name: $ANALYSIS_NAME"
echo "Output Folder: $OUTPUT_FOLDER"
echo "Repo Root: $REPO_ROOT"
echo ""
echo "Resolved Outputs:"
echo "  Scan CSV: $SCAN_CSV"
echo "  Peak Dir: $PEAK_DIR"
echo "  Peak File: $PEAK_OUTPUT_FILE"
echo "  Viz Dir:  $VIS_DIR"
echo ""
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

mkdir -p "$PEAK_DIR"

# Step 1: DeepCIS scan

echo -e "${BLUE}[STEP 1/3] Running deepcis_scanner.py...${NC}"
declare -a SCAN_CMD=(
    python -m src.analysis.motives.deepcis_scanner
    --run-folder "$RESULTS_FOLDER"
    --output "$OUTPUT_FOLDER"
    --name "$ANALYSIS_NAME"
    --model "$MODEL_PATH"
)
if [ -n "$DEEPCIS_WINDOW_SIZE" ]; then
    SCAN_CMD+=(--window-size "$DEEPCIS_WINDOW_SIZE")
fi
if [ -n "$DEEPCIS_STEP_SIZE" ]; then
    SCAN_CMD+=(--step "$DEEPCIS_STEP_SIZE")
fi
if [ -n "$DEEPCIS_BATCH_SIZE" ]; then
    SCAN_CMD+=(--batch-size "$DEEPCIS_BATCH_SIZE")
fi
if [ -n "$OVERWRITE_FLAG" ]; then
    SCAN_CMD+=("$OVERWRITE_FLAG")
fi

"${SCAN_CMD[@]}"

echo -e "${GREEN}✓ deepcis_scanner.py completed successfully${NC}"

if [ ! -f "$SCAN_CSV" ]; then
    echo -e "${RED}Error: Expected scan output not found: $SCAN_CSV${NC}"
    exit 1
fi
echo ""

# Step 2: Peak calling

echo -e "${BLUE}[STEP 2/3] Running peak_scanner.py...${NC}"
declare -a PEAK_CMD=(
    python -m src.analysis.motives.peak_scanner
    "$SCAN_CSV"
    --output-path "$PEAK_OUTPUT_FILE"
    --signal-type "$SIGNAL_TYPE"
    --log-level "$LOG_LEVEL"
)
if [ -n "$ANNOTATOR_WINDOW_SIZE" ]; then
    PEAK_CMD+=(--annotator-window-size "$ANNOTATOR_WINDOW_SIZE")
fi
if [ -n "$ANNOTATOR_STEP_SIZE" ]; then
    PEAK_CMD+=(--annotator-step-size "$ANNOTATOR_STEP_SIZE")
fi
if [ -n "$ANNOTATOR_THRESHOLD_PEAK" ]; then
    PEAK_CMD+=(--annotator-threshold-peak "$ANNOTATOR_THRESHOLD_PEAK")
fi
if [ -n "$ANNOTATOR_SIGMA" ]; then
    PEAK_CMD+=(--annotator-sigma "$ANNOTATOR_SIGMA")
fi
if [ -n "$ANNOTATOR_LAMBDA_WEIGHT" ]; then
    PEAK_CMD+=(--annotator-lambda-weight "$ANNOTATOR_LAMBDA_WEIGHT")
fi
if [ ${#GENES_ARR[@]} -gt 0 ]; then
    PEAK_CMD+=(--genes "${GENES_ARR[@]}")
fi
if [ ${#TFS_ARR[@]} -gt 0 ]; then
    PEAK_CMD+=(--tfs "${TFS_ARR[@]}")
fi

"${PEAK_CMD[@]}"

echo -e "${GREEN}✓ peak_scanner.py completed successfully${NC}"

if [ ! -f "$PEAK_OUTPUT_FILE" ]; then
    echo -e "${RED}Error: Expected peak output not found: $PEAK_OUTPUT_FILE${NC}"
    exit 1
fi

echo "Using peak file: $PEAK_OUTPUT_FILE"
echo ""

# Step 3: Visualization (random subset)

echo -e "${BLUE}[STEP 3/3] Running deepcis_visualize.py with --random-subset...${NC}"

# Clean visualization output to ensure only latest plots are present.
if [ -d "$VIS_DIR" ]; then
    rm -rf "$VIS_DIR"
fi
mkdir -p "$VIS_DIR"

declare -a VIZ_CMD=(
    python -m src.analysis.motives.deepcis_visualize
    --input "$SCAN_CSV"
    --peaks "$PEAK_OUTPUT_FILE"
    --output "$VIS_DIR"
    --random-subset
    --format "$VIZ_FORMAT"
)

if [ "$HIGHLIGHT_PADDING" = false ]; then
    VIZ_CMD+=(--no-highlight-padding)
fi
if [ "$HIGHLIGHT_PEAKS" = false ]; then
    VIZ_CMD+=(--no-highlight-peaks)
fi
if [ ${#GENES_ARR[@]} -gt 0 ]; then
    VIZ_CMD+=(--genes "${GENES_ARR[@]}")
fi
if [ ${#TFS_ARR[@]} -gt 0 ]; then
    VIZ_CMD+=(--tfs "${TFS_ARR[@]}")
fi

"${VIZ_CMD[@]}"

echo -e "${GREEN}✓ deepcis_visualize.py completed successfully${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}Pipeline completed successfully${NC}"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Artifacts:"
echo "  Scan:          $SCAN_CSV"
echo "  Peak file:     $PEAK_OUTPUT_FILE"
echo "  Visualizations:$VIS_DIR"
echo "=========================================="
