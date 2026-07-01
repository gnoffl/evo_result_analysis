# Refactor `compare_TFs.py` into a reusable toolbox + thin per-analysis scripts

## Context

`compare_TFs.py` was written as a one-off to compare which transcription factors (TFs)
are introduced/removed across four evolutionary runs (Arabidopsis max, GOF | Arabidopsis
min, LOF) and render an annotated heatmap. We now want to run *similar* analyses for other
contrasts — different models, different starting genes — not just minimization vs.
maximization.

The current script bakes the max/min axis into three places (column reordering in
`build_matrix`, the `mean(max) − mean(min)` row score in `order_tfs`, the `n_max_runs`
separator and the hardcoded `ARA_MAX_RUN`/`ARA_MIN_RUN` paired test). It also writes all
outputs loosely into the source directory, and conflates "what to compute" with "what to
display" inside one `main()`.

**Goal:** lift the already-well-factored functions into two shared library modules
(calculation + visualization), de-bake the max/min assumptions so the functions compose,
and express each concrete comparison as a small thin script that picks what it needs. The
existing figure must be reproducible (small visual cleanups allowed). We explicitly
**reject** a config-driven `STUDIES` dict — the toolbox + thin-script model keeps each
analysis simple and readable.

## Target layout

```
src/workflows/evo_alg_pooled_plots/tf_comparison/
  tf_comparison_calc.py     # loading + all statistics (zero plotting)
  tf_comparison_plot.py     # plot_heatmap + star helpers (zero stats/IO)
  minmax_comparison.py      # thin script: reproduces today's min-vs-max figure
  summary.md                # update to describe the toolbox + thin-script model
  minmax_comparison/        # NEW output subfolder for this analysis (gitignored)
  # (future siblings: model_comparison.py, startgene_comparison.py, each + its subfolder)
```
`compare_TFs.py` is removed once its functions have moved and `minmax_comparison.py`
reproduces its output. Output globs `evo_alg_pooled_plots/**/*.png` / `**/*.csv` already
cover the new subfolders — no `.gitignore` change needed.

## `tf_comparison_calc.py` — public API

Move, keeping behavior, the loading/stats functions and their private helpers
(`_wilcoxon_pvalue`, `_bh_qvalues`, `_diff_series_to_matrix`, `_tf_stats`) plus the
data-schema column-name constants (`TF_COLUMN`, `DIFF_COLUMN`, `REF_COLUMN`,
`MUTATED_COLUMN`, `ANNOTATED_PEAKS_GLOB`, `SIGNAL_TYPE_COLUMN`, `GENE_COLUMN`).

Changes that de-bake max/min:

- `build_matrix(runs, normalization="per_gene")` — **stop reordering**. `runs` becomes
  `List[Tuple[run_directory, display_label]]` (drop the direction element) and columns are
  built in the order given. Manual column ordering for free; grouping is the caller's job.
- Replace `order_tfs(matrix, directions)` with two composable orderings:
  - `order_tfs_by_group_contrast(matrix, left_columns, right_columns)` → rows sorted by
    `mean(left) − mean(right)` descending (today's max-minus-min behavior; missing values
    treated as 0, as now).
  - `order_tfs_by_mean(matrix)` → rows sorted by overall row mean descending (the flat /
    no-grouping case).
- `paired_tf_significance(run_a_directory, run_b_directory)` — unchanged logic, but rename
  the per-run output columns from `*_max`/`*_min` to neutral `*_a`/`*_b`
  (`median_diff_a`, `n_nonzero_a`, `p_a`, `q_a`, and `_b`). The contrast columns
  (`median_D`, `n_nonzero_D`, `p_contrast`, `q_contrast`, `n_genes`) keep their names.
- `single_run_tf_significance(run_dir)` — unchanged.
- `count_genes`, `load_run_diffs`, `load_per_gene_diffs` — moved unchanged.
- **New** `top_bottom_tfs(matrix, n)` — extract the top-N/bottom-N row slice currently
  inlined in `main()` (lines 447–452) into a reusable, tested function.

## `tf_comparison_plot.py` — public API

Move `q_to_stars`, `STAR_THRESHOLDS`, `COLORMAP`, and `plot_heatmap`, with one change:

- `plot_heatmap(matrix, annotate, cbar_label, row_stars=None, cell_stars=None,
  separator_after_column=None)` — replace `n_max_runs` with
  `separator_after_column: Optional[int]`. Draw the divider `axvline` only when
  `0 < separator_after_column < n_runs`; `None` ⇒ no separator (flat case). All other
  behavior (symmetric diverging scale, row-label stars, cell-annotation stars) unchanged.

`row_stars`/`cell_stars` are unchanged and remain the **"compute everything, display a
subset"** mechanism: a thin script computes whatever stats it wants, writes all CSVs, then
passes only the chosen q-value columns as stars.

## `minmax_comparison.py` — thin script (reproduces today's figure)

Self-contained orchestration, importing from the two libs:

1. Define the 4 runs in display order with labels, the `left_columns` (max labels),
   `right_columns` (min labels), the ara paired pair, and
   `OUTPUT_DIR = Path(__file__).parent / "minmax_comparison"` (`makedirs(exist_ok=True)`).
2. `paired_tf_significance(ara_max_dir, ara_min_dir)` → save CSV; build `row_stars` from
   `q_contrast`; build the ara `cell_stars` from `q_a`/`q_b` (matches today's reuse).
3. `single_run_tf_significance` for GOF and LOF → save per-run CSVs → their `q_intra`
   stars into `cell_stars`.
4. For each normalization (`per_gene`, `log_fold_change`):
   `build_matrix` → `order_tfs_by_group_contrast(left_columns, right_columns)` →
   `top_bottom_tfs(n=5)` → `plot_heatmap(separator_after_column=len(left_columns))` →
   save PNG into the subfolder.

A future `model_comparison.py` differs only by: different runs/labels, possibly
`order_tfs_by_mean` + `separator_after_column=None` for the flat case, and whichever
paired/single tests make sense.

## Tests

Per project rule (one test file per source module; `unittest`; Arrange-Act-Assert;
`tempfile` + synthetic-run writers; imports as `from workflows.evo_alg_pooled_plots...`):

- `test/workflows/evo_alg_pooled_plots/tf_comparison/test_tf_comparison_calc.py` — migrate
  the calc cases from the existing `test_compare_TFs.py` (`CountGenesTest`,
  `BuildMatrixTest`, `OrderTfsTest`, `LoadPerGeneDiffsTest`, `PairedTfSignificanceTest`,
  `SingleRunTfSignificanceTest`, `TopBottomNTfsSlicingTest`). Update `BuildMatrixTest` to
  the order-preserving contract; split `OrderTfsTest` into the two new ordering functions
  and **add** an `order_tfs_by_mean` case; point `TopBottomNTfsSlicingTest` at the new
  `top_bottom_tfs` function; update `PairedTfSignificanceTest` only if it touches renamed
  columns (it currently asserts only contrast columns, so likely unaffected).
- `test_tf_comparison_plot.py` — migrate `QToStarsTest` and `PlotHeatmapTest`; replace the
  `n_max_runs` argument with `separator_after_column` and add a case asserting no divider
  is drawn when it is `None`.
- `test_minmax_comparison.py` — integration smoke test: build two synthetic runs in a temp
  dir (reuse the `_write_run`/`_write_annotated_peaks` helper pattern), point the script's
  output dir at a temp dir, run its orchestration, assert the expected CSV + PNG files are
  created.
- Remove the obsolete `test_compare_TFs.py` after its cases are migrated (its target module
  no longer exists). **Flag to user**: this is the one existing test file being replaced;
  net coverage increases.

## Verification

```bash
conda run -n deepCREshap python -m pytest test/workflows/evo_alg_pooled_plots/tf_comparison/
```
All migrated + new unit tests must pass. Confirm no lint/diagnostic warnings on the new
files.

End-to-end (depends on real run data under the hardcoded ARCitect paths, so the user runs
this smoke check):
`conda run -n deepCREshap python -m src.workflows.evo_alg_pooled_plots.tf_comparison.minmax_comparison`
and visually compare the new `minmax_comparison/compare_TFs_*_top5.png` to the existing
`compare_TFs_*_top5.png` — they should match (modulo any agreed minor cleanups).
