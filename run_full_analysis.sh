#!/bin/bash

# Script to run complete evolutionary analysis pipeline
# Usage: ./run_full_analysis.sh <results_folder> <analysis_name> <output_folder> [options]
#
# This script runs three analysis scripts in sequence:
# 1. simple_result_stats.py - Basic statistics and visualizations
# 2. summarize_mutations.py - Mutation summarization
# 3. analyze_mutations.py - Detailed mutation analysis

# Color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print usage
usage() {
    echo "Usage: $0 <results_folder> <analysis_name> <output_folder> [options]"
    echo ""
    echo "Required Arguments:"
    echo "  results_folder  - Folder containing gene directories with evolution results"
    echo "  analysis_name   - Name to identify this analysis run"
    echo "  output_folder   - Output folder for results"
    echo ""
    echo "Optional Arguments (Step 2: summarize_mutations):"
    echo "  -f, --final-generation <int>    Final generation to consider (default: 1999)"
    echo "  -g, --generation <int>          Specific generation to filter (default: None - all generations)"
    echo ""
    echo "Optional Arguments (Step 3: analyze_mutations):"
    echo "  -w, --window-size <int>         Window size for rolling mean plots (default: 31)"
    echo "  -m, --mutable-positions <int>       Number of mutable positions (default: 3000)"
    echo ""
    echo "Optional Arguments (All steps):"
    echo "  -o, --output-format <format>        Output format for plots (default: png)"
    echo "  -l, --last-generation <int>         Last generation for simple_result_stats (default: 1999)"
    echo ""
    echo "Example:"
    echo "  $0 ./results/GOF_run1 GOF_analysis ./outputs"
    echo "  $0 ./results/GOF_run1 GOF_analysis ./outputs -f 1500 -g 1000 -w 51"
    exit 1
}

# Check minimum required arguments
if [ $# -lt 3 ]; then
    usage
fi

RESULTS_FOLDER="$1"
ANALYSIS_NAME="$2"
OUTPUT_FOLDER="$3"
shift 3  # Remove first 3 arguments, remaining are optional

# Default values for optional parameters
FINAL_GENERATION=1999
GENERATION=""
WINDOW_SIZE=31
MUTABLE_POSITIONS=3000
OUTPUT_FORMAT="png"
LAST_GENERATION=1999

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--final-generation)
            FINAL_GENERATION="$2"
            shift 2
            ;;
        -g|--generation)
            GENERATION="$2"
            shift 2
            ;;
        -w|--window-size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        -m|--mutable-positions)
            MUTABLE_POSITIONS="$2"
            shift 2
            ;;
        -o|--output-format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        -l|--last-generation)
            LAST_GENERATION="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Verify results folder exists
if [ ! -d "$RESULTS_FOLDER" ]; then
    echo -e "${RED}Error: Results folder '$RESULTS_FOLDER' does not exist${NC}"
    exit 1
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Get absolute paths
RESULTS_FOLDER=$(realpath "$RESULTS_FOLDER")
OUTPUT_FOLDER=$(realpath "$OUTPUT_FOLDER")

echo "=========================================="
echo "Starting Full Analysis Pipeline"
echo "=========================================="
echo "Results Folder: $RESULTS_FOLDER"
echo "Analysis Name: $ANALYSIS_NAME"
echo "Output Folder: $OUTPUT_FOLDER"
echo ""
echo "Parameters:"
echo "  Final Generation: $FINAL_GENERATION"
if [ -n "$GENERATION" ]; then
    echo "  Specific Generation: $GENERATION"
fi
echo "  Last Generation (stats): $LAST_GENERATION"
echo "  Window Size: $WINDOW_SIZE"
echo "  Mutable Positions: $MUTABLE_POSITIONS"
echo "  Output Format: $OUTPUT_FORMAT"
echo ""
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Track overall success
OVERALL_SUCCESS=true

# Step 1: Simple Result Stats
echo -e "${BLUE}[STEP 1/3] Running simple_result_stats.py...${NC}"
if python -m src.analysis.simple_result_stats \
    --results_folder "$RESULTS_FOLDER" \
    --name "$ANALYSIS_NAME" \
    --output_folder "$OUTPUT_FOLDER" \
    --output_format "$OUTPUT_FORMAT" \
    --last_generation $LAST_GENERATION \
    --all; then
    echo -e "${GREEN}✓ simple_result_stats.py completed successfully${NC}"
else
    echo -e "${RED}✗ simple_result_stats.py failed${NC}"
    OVERALL_SUCCESS=false
fi
echo ""

# Step 2: Summarize Mutations
echo -e "${BLUE}[STEP 2/3] Running summarize_mutations.py...${NC}"

# Build command with optional generation parameter
SUMMARIZE_CMD="python -m src.analysis.summarize_mutations \
    --results_folder \"$RESULTS_FOLDER\" \
    --name \"$ANALYSIS_NAME\" \
    --output_folder \"$OUTPUT_FOLDER\" \
    --final_generation $FINAL_GENERATION"

if [ -n "$GENERATION" ]; then
    SUMMARIZE_CMD="$SUMMARIZE_CMD --generation $GENERATION"
fi

# Run and capture output to extract the mutation file path
SUMMARIZE_OUTPUT=$(eval $SUMMARIZE_CMD 2>&1)
SUMMARIZE_EXIT_CODE=$?

echo "$SUMMARIZE_OUTPUT"

if [ $SUMMARIZE_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ summarize_mutations.py completed successfully${NC}"
    # Extract the output file path from the output
    MUTATION_DATA=$(echo "$SUMMARIZE_OUTPUT" | grep "OUTPUT_FILE=" | cut -d'=' -f2)
    if [ -z "$MUTATION_DATA" ]; then
        echo -e "${RED}✗ Could not extract mutation data file path from output${NC}"
        OVERALL_SUCCESS=false
    fi
else
    echo -e "${RED}✗ summarize_mutations.py failed${NC}"
    OVERALL_SUCCESS=false
fi
echo ""

# Step 3: Analyze Mutations
if [ -z "$MUTATION_DATA" ] || [ ! -f "$MUTATION_DATA" ]; then
    echo -e "${RED}✗ Mutation data file not found or not set: $MUTATION_DATA${NC}"
    echo -e "${RED}  Cannot proceed with analyze_mutations.py${NC}"
    OVERALL_SUCCESS=false
else
    echo -e "${BLUE}[STEP 3/3] Running analyze_mutations.py...${NC}"
    if python -m src.analysis.analyze_mutations \
        --mutation_data "$MUTATION_DATA" \
        --name "$ANALYSIS_NAME" \
        --output_folder "$OUTPUT_FOLDER" \
        --window_size $WINDOW_SIZE \
        --mutable_positions $MUTABLE_POSITIONS \
        --output_format "$OUTPUT_FORMAT" \
        --all; then
        echo -e "${GREEN}✓ analyze_mutations.py completed successfully${NC}"
    else
        echo -e "${RED}✗ analyze_mutations.py failed${NC}"
        OVERALL_SUCCESS=false
    fi
fi
echo ""

# Final summary
echo "=========================================="
echo "Analysis Pipeline Complete"
echo "=========================================="
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
if [ "$OVERALL_SUCCESS" = true ]; then
    echo -e "${GREEN}✓ All steps completed successfully${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Some steps failed - check output above${NC}"
    exit 1
fi
