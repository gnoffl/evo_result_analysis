# Copilot Instructions: evo_result_analysis

## Project Overview

**evo_result_analysis** is a Python package for analyzing evolutionary optimization of plant DNA sequences for transcription factor (TF) binding. The codebase uses deepCIS neural network predictions to identify how in-silico mutations affect TF binding, with particular focus on *Arabidopsis* GOF/LOF candidate validation.

**Key Domain**: Bioinformatics | Evolutionary Computing | Gene Regulatory Analysis

## Architecture

```
GeneRunData (data container)
           ↓
deepcis_scanner (TensorFlow NN predictions: signal per bp)
           ↓
peak_annotation (4-part algorithm: detect regions → preprocess → select peaks → format)
           ↓
peak_scanner (batch orchestration across genes/TFs)
           ↓
deepcis_annotation (map mutations to binding effects)
           ↓
workflows (candidate selection, Excel output, figures)
```

### Key Modules

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `src/analysis/motives/peak_annotation.py` | Peak detection in TF signals | `PeakAnnotator` (4-part algorithm with cached parameters) |
| `src/analysis/motives/peak_scanner.py` | Batch peak detection | `DeepCISPeakScanner` (orchestrates genes × TFs) |
| `src/analysis/motives/deepcis_scanner.py` | NN prediction caching | `DeepCISScanner` (TensorFlow model loader) |
| `src/analysis/mutations/` | Mutation tracking | `GeneRunData`, mutation summarization |
| `src/workflows/candidate_selection.py` | Filter & validate candidates | GOF/LOF target selection |

## Development Setup

### Prerequisites
- Python 3.8+
- TensorFlow (deep learning model inference)
- Dependencies: numpy, pandas, scipy, matplotlib, seaborn

### Installation

```bash
# Development install (editable mode)
pip install -e .

# Run full analysis pipeline
./run_full_analysis.sh [gene_file] [option_flags]

# Run mutations-only pipeline
./run_mutations_analysis.sh [gene_file]
```

## Code Conventions

### Patterns & Style

1. **Heavy OOP with Type Hints**
   - All classes use type annotations (`typing` module)
   - Dataclasses for data containers
   - Parameter caching in `__init__` (see `PeakAnnotator._window_size_elements`)

2. **Pandas-Centric Data Flow**
   - DataFrames as primary data structure
   - Column names are stable and documented
   - Often include `window_start`, `signal`, `score` columns

3. **Constants & Parameters**
   ```python
   DEFAULT_WINDOW_SIZE = 250      # bp (base pairs)
   DEFAULT_STEP_SIZE = 10         # bp between measurements
   DEFAULT_SIGMA = 20.0           # bp (Gaussian smoothing)
   ```

4. **Configuration Objects**
   - Prefer passing config to `__init__` over method parameters
   - Cache derived values (e.g., `window_size_elements = window_size // step_size`)
   - Example: `PeakAnnotator(window_size=250, step_size=10, df=signal_df)`

5. **Testing & Validation**
   - Ad-hoc testing via `test/test.py`
   - Code should be written with testability in mind
   - Full unittests should be placed in matching files in `test/`

### Recent Refactoring Pattern

The codebase underwent OOP refactoring in 2026. When working on similar classes:
- **Extract repeated parameters** → class attributes (reduces method signatures)
- **Break large methods** → 3-5 focused helper methods (improves readability)
- **Infer parameters from data** → pass DataFrame to `__init__` for auto-setup
- See: [OOP_REFACTORING_SUMMARY.md](OOP_REFACTORING_SUMMARY.md)

## Common Tasks

### Add a New Analysis Workflow

1. Create workflow file in `src/workflows/`
2. Accept `GeneRunData` objects or DataFrames
3. Use existing scanners/annotators for standard operations
4. Export results as CSV or pickle for downstream use

### Modify Peak Detection Parameters

Edit `PeakAnnotator.__init__()`:
```python
annotator = PeakAnnotator(
    window_size=250,      # bp detection window
    step_size=10,         # spacing between measurements
    threshold_peak=0.1,   # moving average threshold
    sigma=20.0,           # Gaussian smoothing
    lambda_weight=1.0,    # peak sharpness vs mass balance
    df=signal_df          # (optional) infer step_size from data
)
peaks = annotator.detect_peaks(signal_df)
```

### Debug Peak Detection

```python
# Create annotator with diagnostic caching
annotator = PeakAnnotator(df=your_data)

# Inspect cached derived values
print(f"window_size_elements: {annotator._window_size_elements}")
print(f"sigma_elements: {annotator._sigma_elements}")
print(f"actual_step_size: {annotator._actual_step_size}")

# Access individual algorithm parts
regions = annotator._detect_signal_regions(signal)
deriv, mass = annotator._preprocess_signal_for_peaks(signal)
```

## Testing

```bash
# Run test utilities (ad-hoc testing)
python test/test.py

# Check syntax without running
python -m py_compile src/analysis/motives/peak_annotation.py
```

> **Note**: Full unit test suite is in development. Tests currently use direct imports and manual validation.

## Documentation Reference

- [PEAK_DETECTION_OOP_GUIDE.md](PEAK_DETECTION_OOP_GUIDE.md) — Algorithm details & class structure
- [OOP_REFACTORING_SUMMARY.md](OOP_REFACTORING_SUMMARY.md) — Recent refactoring history and rationale

## Assistant Tips

### When Refactoring Classes

1. **Identify repeated parameters** across methods
2. **Move to `__init__`** and cache (especially derived/computed values)
3. **Break down large methods** (>50 lines) into 3-5 focused helpers
4. **Preserve method signatures** for external callers (add new params as optional)
5. **Always validate inputs** early in `__init__` or the first called method

### When Working with DataFrames

- Column presence is critical (validate early)
- Assume `window_start` is uniformly spaced if present
- Use `astype(np.float64)` for signal arrays (numerical stability)
- Return empty DataFrames with correct column structure on no-data edge cases

### When Adding Parameters

- Use `Optional[Type]` for parameters you want to infer/default
- Document default behavior clearly
- Consider inferring from DataFrame if a DataFrame is already being passed
- Cache derived values (avoid recomputing in loops)

## Project Structure at a Glance

```
evo_result_analysis/
├── .github/
│   └── copilot-instructions.md   ← You are here
├── src/
│   ├── analysis/
│   │   ├── motives/              ← TF binding analysis
│   │   ├── mutations/            ← Mutation tracking
│   │   ├── overview/             ← Summary statistics
│   │   └── utils/                ← Shared utilities
│   └── workflows/                ← High-level analysis pipelines
├── test/                         ← Test utilities
├── data/                         ← FASTA files, CSVs, predictions
├── pyproject.toml                ← Project metadata & dependencies
├── README.md                     ← User documentation
└── run_full_analysis.sh          ← Main pipeline script
```

## Quick Links

- **Installation**: See pyproject.toml and `pip install -e .`
- **Main Analysis**: `./run_full_analysis.sh`
- **Peak Detection**: `src/analysis/motives/peak_annotation.py` → `PeakAnnotator` class
- **Batch Processing**: `src/analysis/motives/peak_scanner.py` → `DeepCISPeakScanner` class
- **Candidate Filtering**: `src/workflows/candidate_selection.py`

---

*Last updated: 2026-03-24*  
*For questions about architecture or code patterns, consult the [OOP_REFACTORING_SUMMARY.md](OOP_REFACTORING_SUMMARY.md) and [PEAK_DETECTION_OOP_GUIDE.md](PEAK_DETECTION_OOP_GUIDE.md) documents.*
