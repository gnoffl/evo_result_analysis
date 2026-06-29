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
conda activate deepCREshap   # required; contains tensorflow, pyfaidx, scikit-learn
pip install -e .              # editable install (recommended for development)
```

Python ≥ 3.8 required. Core dependencies (pyproject.toml): numpy, pandas, scipy, matplotlib, seaborn, statsmodels, tqdm. Heavy ML/bio deps (conda env only): tensorflow, pyfaidx, scikit-learn.

All necessary dependencies are installed in the conda environment "deepCREshap".

## Execution Environment

**ALWAYS** run Python and pytest via the `deepCREshap` conda environment, never bare `python` or `pytest`:

```bash
conda run -n deepCREshap python script.py
conda run -n deepCREshap python -m pytest test/
```

This applies to every execution: tests, scripts, one-off commands, anything.

## Running Tests

```bash
conda run -n deepCREshap python -m pytest test/                          # all tests
conda run -n deepCREshap python -m pytest test/analysis/motives/         # specific subdirectory
conda run -n deepCREshap python -m pytest test/analysis/motives/test_analyze_mapping.py  # single file
```

No pytest config file — uses default discovery. Tests use `unittest.mock` (patch, mock_open) plus temporary directories for integration tests.

## Test discipline

**Editing existing files:** Before returning to the user, rerun the tests for every edited file. If no test file exists for an edited module, explicitly flag this to the user before finishing.

**Creating new files:** Tests are mandatory. Write the test file alongside the new source file in the same session — never create a new module without a corresponding test file.

## Analysis Scripts

Shell pipeline entry points in `analysis_scripts/`:

- `run_deepcis_peak_pipeline.sh` — end-to-end: deepCIS scan → peak calling → visualization (usage: `<results_folder> <analysis_name> <output_folder> [options]`)
- `run_full_analysis.sh` — full evolutionary analysis: stats → mutation summarization → mutation analysis (same argument pattern)

## Architecture

### Package layout (`src/`)

**`analysis/`** — core domain analysis, organized by subdomain:

- `motives/` — TF binding analysis: `deepcis_scanner.py` (TF predictions), `peak_annotation.py` (4-stage peak detection algorithm), `analyze_mapping.py` (EPM→TF mapping, fitness aggregation), `analyze_peaks.py`, `deepcis_annotation.py`, `deepcis_visualize.py`
- `mutations/` — mutation effect analysis: `analyze_mutations.py`, `summarize_mutations.py`, `genomic_annotation.py`
- `overview/` — Pareto front statistics, cross-method comparison
- `starrseq/` — STAR-seq enrichment with Mann-Whitney U tests
- `utils/` — `io.py` (formatted console output for progress tracking in long-running workflows only; not for general use in scripts), `sequence_processing.py` (exon/intron/UTR extraction)

**`workflows/`** — higher-level pipelines and figure generation:

- `paper_plots/` — publication-ready figures
- `overlap_analysis/` — STAR-seq × deepCRE correlation
- `hoffie/` — protein design/plasmid workflows
- `evo_alg_pooled_plots/` — cross-run pooled evolution plots
- `mutation_distance_analysis/` — pairwise sequence distance analysis
- `mutation_distribution_analysis/` — per-position mutation frequency
- `deepCRE_TPM_correlation/` — deepCRE score vs. TPM expression correlation
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

## Version Control

Images (`*.png`, `*.svg`) and generated data files (`*.csv`) produced by scripts in this repo are gitignored — do not commit them. When adding a new workflow that writes such outputs, add the appropriate globs to `.gitignore`.

`data/` holds genome FASTA files (gitignored). `models/` holds deepCIS `.h5` model files (gitignored). Both must be populated manually before running analyses.
