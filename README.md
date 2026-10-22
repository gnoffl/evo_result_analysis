# evo_result_analysis

A package for evolutionary result analysis.

## Installation

From the project root, run:

```bash
pip install .
```

## Usage

Import modules from the package in your Python code:

```python
from evo_result_analysis.analysis import analyze_mapping
```

To visualize a random subset of compatible genes and TFs, use:

```bash
python -m analysis.motives.deepcis_visualize \
  --input data/deepcis_window_scan_results.csv \
  --random-subset
```

## Project Structure

- `src/analysis/` — Analysis modules
- `test/` — Test scripts

## Development

To install in editable mode (for development):

```bash
pip install -e .
```
