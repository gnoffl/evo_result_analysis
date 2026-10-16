# src/analysis/utils/

Small shared helpers: console progress formatting (`io.py`), the paper's per-feature significance recipe (`statistics.py`: Wilcoxon vs 0 → BH q-values → stars), and an empty stub (`sequence_processing.py`). Despite the module names, there are no data-loading helpers here.

## io.py  [library]
Formatted console output. Meant ONLY for progress tracking in long-running workflows; don't use it as default output in ordinary scripts.

- `print_section_header(title: str, symbol: str = "=") -> None` — 80-character banner with a centred title.
- `print_subsection(title: str) -> None` — 80-character `─` rules around an indented title.
- `print_status(message: str, status: str = "INFO") -> None` — prints `[HH:MM:SS] <symbol> message`; status is INFO/SUCCESS/ERROR/WARNING.

## statistics.py  [library]
Shared significance recipe, so the star thresholds are identical everywhere; used by blamm_significance and the evo_alg_pooled_plots/tf_comparison workflows. **Use this instead of writing new star/Wilcoxon/BH helpers.**

- `SIGNIFICANCE_STAR_THRESHOLDS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]` — q-value → stars, strict `<`.
- `wilcoxon_pvalue_vs_zero(values: np.ndarray) -> float` — two-sided Wilcoxon signed-rank p-value vs 0 (scipy default zero_method="wilcox"); NaN if all values are zero.
- `benjamini_hochberg_qvalues(pvalues: pd.Series) -> pd.Series` — BH q-values (statsmodels `fdr_bh`); NaN p-values are left out of the correction and stay NaN.
- `qvalue_to_stars(qvalue: float) -> str` — stars per the thresholds above; "" for NaN or not significant.

## sequence_processing.py  [one-off]
**[broken]** Empty stub: an argparse parser with no arguments and a `main()` that does nothing. Despite its docstring and CLAUDE.md, it contains NO exon/intron/UTR extraction code.
