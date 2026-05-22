# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`evo_result_analysis` is a Python package for analyzing results from evolutionary algorithm runs that optimize DNA sequences (promoters) for transcription factor binding. The analysis pipeline covers:

- Parsing Pareto front outputs (JSON files with sequence/fitness/mutation tuples)
- Predicting transcription factor binding via deepCIS (TensorFlow sliding-window predictions)
- Peak detection and TF motif annotation
- Mutation effect statistics
- STAR-seq enrichment analysis
- Publication figure generation

## Setup & Installation

```bash
pip install -e .   # editable install (recommended for development)
```

Python ≥ 3.8 required. Key dependencies: numpy, pandas, scipy, matplotlib, seaborn, tensorflow, pyfaidx, sklearn.

All necessary dependencies are installed in the conda environment "deepCREshap".

## Running Tests

```bash
pytest test/                          # all tests
pytest test/analysis/motives/         # specific subdirectory
pytest test/analysis/motives/test_analyze_mapping.py  # single file
```

No pytest config file — uses default discovery. Tests use `unittest.mock` (patch, mock_open) plus temporary directories for integration tests.

## Architecture

### Package layout (`src/`)

**`analysis/`** — core domain analysis, organized by subdomain:

- `motives/` — TF binding analysis: `deepcis_scanner.py` (TF predictions), `peak_annotation.py` (4-stage peak detection algorithm), `analyze_mapping.py` (EPM→TF mapping, fitness aggregation), `analyze_peaks.py`, `deepcis_annotation.py`, `deepcis_visualize.py`
- `mutations/` — mutation effect analysis: `analyze_mutations.py`, `summarize_mutations.py`, `genomic_annotation.py`
- `overview/` — Pareto front statistics, cross-method comparison
- `starrseq/` — STAR-seq enrichment with Mann-Whitney U tests
- `utils/` — `io.py` (formatted console output), `sequence_processing.py` (exon/intron/UTR extraction)

**`workflows/`** — higher-level pipelines and figure generation:

- `paper_plots/` — publication-ready figures
- `overlap_analysis/` — STAR-seq × deepCRE correlation
- `hoffie/` — protein design/plasmid workflows
- `figure_composition.py` — assembles multi-panel SVG figures
- `candidate_selection.py`, `update_excel_sheet.py`

### Data flow

1. **Input**: Pareto front JSON files from evolutionary runs
2. **Sequence analysis**: deepCIS sliding-window predictions → peak detection → TF family mapping
3. **Statistics**: mutation counts, fitness distributions, binding enrichment (t-test or Mann-Whitney U depending on normality)
4. **Output**: CSV files, JSON annotations, PNG/SVG figures

### Key design patterns

- `GeneRunData` dataclass holds gene metadata passed through pipelines
- `PeakAnnotator` class encapsulates the 4-part peak detection algorithm (region detection → smoothing → peak selection → formatting)
- Most modules follow functional pipelines over pandas DataFrames
- Dual format support: old vs. new pipeline output formats controlled by feature flags

### External dependencies

Imports from `evolution`, `deepCRE`, and `evolution.sequences` are other research packages not included in this repo. These must be available in the Python environment.

### Known incomplete implementations

`get_epm_tfbs_mapping_new()` in `analyze_mapping.py` is marked as NOT PROPERLY IMPLEMENTED — avoid relying on it.
