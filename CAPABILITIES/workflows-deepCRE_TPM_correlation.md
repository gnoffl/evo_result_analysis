# src/workflows/deepCRE_TPM_correlation

Correlates genome-wide deepCRE predictions (precomputed cross-prediction CSV, first two model columns: an Arabidopsis SSR and an MSR model) with Arabidopsis leaf RNA-seq `logMaxTPM`. Fits a linear regression + Pearson correlation for all genes and for "low" (pred 0-0.4) / "high" (pred 0.6-1.0) subsets, writes `correlation_analysis_summary.csv` and one scatter PNG per model into the folder. Inputs and outputs are repo-relative paths, so it must be run from the repo root.

## correlate.py  [script]
Run `python src/workflows/deepCRE_TPM_correlation/correlate.py` from repo root; no CLI args. Not imported elsewhere in `src/` (only by its test); `create_plots(ax=...)` was added for publication composition.

- `load_data() -> (tpm_data, pred_data)` — reads the hardcoded TPM CSV and prediction CSV, both indexed by `gene_id`.
- `merge_data(tpm_data, pred_data) -> pd.DataFrame` — inner join of `logMaxTPM` with prediction columns.
- `analyze_model(merged: pd.DataFrame, model_col: str) -> dict` — `scipy.linregress` + `pearsonr` of `model_col` vs `logMaxTPM` (NaN rows dropped); returns `data, slope, intercept, r_value, p_value, corr_coef, corr_pval`.
- `plot_single_dataset(results: Dict, model_col: str, ax) -> None` — blue scatter + red fit line for one entry of `results`.
- `create_plots(results: dict, model_cols: List[str], ax=None) -> None` — one figure per model, drawing `_low`/`_high` partner columns on the same axes; with `ax`, draws only the first and saves nothing.
- `analyze_subset(results, model_col: str, subset_name: str, filter_low: bool = False) -> dict` — subset `low` (0-0.4) or `high` (0.6-1.0) predictions, always also requires `logMaxTPM > 0.2`, then `analyze_model`. `filter_low` is unused. If < 2 genes it returns a dict without `data`, which makes `save_summary`/`create_plots` raise KeyError.
- `save_summary(results, model_cols: List[str]) -> None` — writes the summary CSV (hardcoded path).
