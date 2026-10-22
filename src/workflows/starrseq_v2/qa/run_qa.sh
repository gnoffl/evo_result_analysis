#!/usr/bin/env bash
# The QA run of 2026-10-22: pools the three contributor sheets with Gernot's own
# selection output and writes candidates.fa / metadata.csv into qa/output/.
#
# Sheet order matters: the first occurrence of a duplicated sequence becomes the
# ordered representative, so library_upload.csv comes first to keep our own ids.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
QA_DIR="$REPO_ROOT/src/workflows/starrseq_v2/qa"

cd "$REPO_ROOT"
conda run -n deepCREshap python -m src.workflows.starrseq_v2.qa.check_upload_sheets \
    "$QA_DIR/input/configs" \
    "$QA_DIR/output" \
    "$REPO_ROOT/src/workflows/starrseq_v2/selection/output/library_upload.csv" \
    "$QA_DIR/input/all_constructs.csv" \
    "$QA_DIR/input/autoTRAIT_batch_constructs_phase_Sep26.ods" \
    "$QA_DIR/input/top500_selection.ods"
