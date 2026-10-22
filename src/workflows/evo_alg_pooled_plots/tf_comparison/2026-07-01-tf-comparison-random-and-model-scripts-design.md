# Two new TF-comparison scripts: random-start run + 2×2 gene-source × model

## Status

Design for review. Not yet implemented. Run directories will be supplied at
implementation time (paths are hardcoded module constants, as in
`minmax_comparison.py`).

## Goal

Add two thin orchestration scripts to the existing TF-comparison toolbox
(`src/workflows/evo_alg_pooled_plots/tf_comparison/`, see `summary.md`), plus the
minimal new calculation functions they need:

1. **`random_start_comparison.py`** — the single-run analogue of
   `minmax_comparison.py`: one maximization run whose starting sequences are
   *random* (not natural plant genes).
2. **`model_comparison.py`** — a four-run 2×2 analysis: natural genes from two
   species (Arabidopsis "ara", *Zea mays* "zea") each optimized under two
   deepCRE fitness models (ara model, zea model). All four runs **maximize**
   expression.

Both follow the established pattern: a thin script picks runs, groupings and
tests, calls the `tf_comparison_calc` + `tf_comparison_plot` toolbox, and writes
its outputs into a sibling subfolder. The scripts add no plotting or statistics
of their own beyond orchestration; two new *calc* functions are added to the
toolbox for the 2×2's CSV-only statistics.

## Background (unchanged from `summary.md`)

A run optimizes promoter sequences; per gene we get a per-TF
`diff = peak_count(optimized) − peak_count(reference)` (positive = TF binding
introduced, negative = removed). **Genes are the replicates.** Per-gene diffs are
recovered from each run's `deepcis_scan/*_annotated_peaks_*.csv`; the aggregated
`diff_calc` (from `*_peak_summary.csv`) drives the descriptive heatmap. Tests are
Wilcoxon signed-rank (paired/one-sample) with Benjamini–Hochberg FDR correction
across TFs. Stars: `*` q<0.05, `**` q<0.01, `***` q<0.001.

## Confirmed decisions

- **2×2 pairing structure** (confirmed by user): the two ara-gene runs share the
  *same* ara genes; the two zea-gene runs share the *same* zea genes; ara and zea
  gene sets are disjoint. So the model swap is **paired by gene within each gene
  source**, and the gene-source axis is unpaired.
- **2×2 figure grouping = by model** (user choice): left group = both ara-model
  runs, right group = both zea-model runs; divider after column 2; rows ordered
  by `mean(ara-model) − mean(zea-model)`. Headline: *does the predictor model
  drive the TF strategy?*
- **Row-label (inter-run) stars are dropped from the figure** (user choice). The
  inter-run significance lives in CSVs only; the user reads it there as needed.
- **Statistics A and C are computed as CSV-only outputs** (user choice), i.e. not
  displayed on the figure:
  - **A — pooled model main effect** (`pooled_model_tf_significance`)
  - **C — interaction / species-specificity** (`interaction_model_tf_significance`)

## Assumptions to confirm at review

1. **Cell stars (intra-run) stay on the figure.** Only the *row-label* stars are
   dropped. Each cell still shows whether that TF's binding changed vs wild-type
   within that run (`q_intra`-style). If the user wants a completely star-free
   figure, drop `cell_stars` too.
2. **Script names** `random_start_comparison.py` and `model_comparison.py`
   (`model_comparison.py` matches the sibling name anticipated in
   `REFACTOR_PLAN.md`).
3. **References cancel within a gene.** For the model contrast `D = diff(ara-model)
   − diff(zea-model)` computed per gene, the wild-type reference peak counts are
   assumed identical across the two models of a gene source (the deepCIS
   reference scan is model-independent — the "model" is the evolution *fitness*
   predictor, not the TF annotator). This is the same assumption `summary.md` §3.4
   relies on for ara min/max. If references differ, `D` still equals
   `optimized_a − optimized_b` only when references match; otherwise the reference
   delta leaks in. **Verify** on the real data that the two runs of a gene source
   share identical reference peak counts per (gene, TF).
4. **Exactly one `*_annotated_peaks_*.csv` per run.** `load_per_gene_diffs` raises
   `FileNotFoundError` unless there is exactly one such file in `deepcis_scan/`.
   At least one candidate random-start directory currently holds two (stale
   timestamps). At implementation time we point at a clean directory or remove the
   stale file first. (We never delete test files; this is real run data, so the
   user decides which CSV is canonical.)

## Script 1 — `random_start_comparison.py`

Pure reuse of the existing toolbox; **no new functions.**

Pipeline (per normalization in `{per_gene, fold_change}`, matching minmax):

1. `matrix = build_matrix([(RUN_DIR, LABEL)], normalization)` → single column.
2. `matrix = order_tfs_by_mean(matrix)` (flat, no left/right grouping).
3. Optional `matrix = top_bottom_tfs(matrix, TOP_BOTTOM_N)` (default `None` = all).
4. `significance = single_run_tf_significance(RUN_DIR)` → save CSV; build
   `cell_stars = {LABEL: {tf: q_to_stars(q_intra)}}`.
5. `plot_heatmap(matrix, annotate=True, cbar_label=..., cell_stars=cell_stars,
   row_stars=None, separator_after_column=None)` → save PNG.

Module constants: `RUN_DIR`, `LABEL` (e.g. `"random max"`), `OUTPUT_DIR =
Path(__file__).parent / "random_start_comparison"`, `OUTPUT_BASENAME`,
`SIGNIFICANCE_BASENAME`, `NORMALIZATIONS` (reused shape), `TOP_BOTTOM_N`.

Outputs in `random_start_comparison/`:

- `<sig_basename>.csv` — `single_run_tf_significance` table.
- `<basename>_per_gene.png`, `<basename>_log_fold_change.png`.

Question it answers: *starting from random sequence, which TF families does the
optimizer systematically introduce/remove to raise predicted expression?*

## Script 2 — `model_comparison.py`

### Runs and columns (grouped by model)

Four `(run_directory, label)` runs in display order:

| order | label (proposed) | gene source | model |
| ------- | ------------------ | ------------- | ------- |
| 1 | `araG/araM` | ara | ara |
| 2 | `zeaG/araM` | zea | ara |
| 3 | `araG/zeaM` | ara | zea |
| 4 | `zeaG/zeaM` | zea | zea |

`LEFT_COLUMNS = ["araG/araM", "zeaG/araM"]` (ara model),
`RIGHT_COLUMNS = ["araG/zeaM", "zeaG/zeaM"]` (zea model),
`separator_after_column = 2`. Labels are tweakable at review.

### Figure

For each normalization:
`build_matrix(RUNS, normalization)` →
`order_tfs_by_group_contrast(matrix, LEFT_COLUMNS, RIGHT_COLUMNS)` →
optional `top_bottom_tfs` →
`plot_heatmap(annotate=True, cell_stars=cell_stars, row_stars=None,
separator_after_column=2)`.
No row-label stars. Outputs `model_comparison/model_comparison_per_gene.png` and
`..._log_fold_change.png`.

### Statistics (all CSV)

Two gene-source pairs, with a consistent A/B convention **A = ara model, B = zea
model** so the contrast `D = diff_a − diff_b` means the same thing in both:

- `ARA_GENE_PAIR = (araG/araM_dir, araG/zeaM_dir)`
- `ZEA_GENE_PAIR = (zeaG/araM_dir, zeaG/zeaM_dir)`

1. **Two stratified paired model contrasts** — existing `paired_tf_significance`:
   - `paired_tf_significance(*ARA_GENE_PAIR)` → CSV
     `model_comparison_significance_ara_genes.csv`. `q_contrast` = model effect on
     ara genes.
   - `paired_tf_significance(*ZEA_GENE_PAIR)` → CSV
     `model_comparison_significance_zea_genes.csv`. `q_contrast` = model effect on
     zea genes.
   - **Cell stars for all four runs come from these tables' `q_a`/`q_b`** (each
     pair covers its two runs' intra-run tests over that gene source's genes — the
     same reuse minmax does for the ara pair). No separate `single_run_tf_significance`
     calls are needed. Mapping:
     - `araG/araM` → ara pair `q_a`; `araG/zeaM` → ara pair `q_b`
     - `zeaG/araM` → zea pair `q_a`; `zeaG/zeaM` → zea pair `q_b`

2. **A — pooled model main effect** — NEW `pooled_model_tf_significance` → CSV
   `model_comparison_significance_pooled_model.csv`.

3. **C — interaction / species-specificity** — NEW `interaction_model_tf_significance`
   → CSV `model_comparison_significance_interaction.csv`.

## New calc functions (`tf_comparison_calc.py`)

### Shared private helper

Extract the pair-contrast construction currently inlined in
`paired_tf_significance` so all three functions share it:

```python
def _pair_contrast_matrix(
    run_a_directory: str, run_b_directory: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Return (a_mat, b_mat, contrast_mat, shared_genes, all_tfs) for a run pair.

    ``a_mat``/``b_mat`` are genes × TFs matrices (absent (gene,TF) = 0) over the
    genes shared by both runs and the union of their TFs; ``contrast_mat =
    a_mat - b_mat``.
    """
```

Refactor `paired_tf_significance` to call this helper (behavior unchanged; its
existing tests must still pass). This keeps the three functions DRY and avoids a
third copy of the intersect/unstack logic.

### A — `pooled_model_tf_significance`

```python
def pooled_model_tf_significance(
    model_pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """Pooled per-TF model main effect across several paired gene sources.

    Each element of ``model_pairs`` is ``(model_a_run_dir, model_b_run_dir)`` for
    one gene source (the two runs share that source's genes). For every TF, the
    per-gene contrast ``D = diff_a − diff_b`` is computed within each pair and the
    D vectors from all pairs are concatenated (gene sets are disjoint across
    pairs), then tested against 0 with a two-sided Wilcoxon signed-rank test.
    p-values are BH-corrected across TFs.

    Returns one row per TF with columns
    ``tf, n_genes, median_D, n_nonzero_D, p_model, q_model``, sorted by
    ``q_model`` ascending.
    """
```

Implementation notes:

- Build each pair's `contrast_mat` via `_pair_contrast_matrix`.
- Global TF set = union across all pairs; reindex each `contrast_mat`'s columns to
  it (fill 0) so pooled columns align.
- Concatenate contrast matrices row-wise (`pd.concat`, axis=0). Gene indices are
  disjoint; if any collision risk, prefix with a pair id.
- Per TF: `_tf_stats(pooled[tf].to_numpy())` → `(median_D, n_nonzero_D, p_model)`;
  `n_genes` = total pooled gene count.
- `q_model = _bh_qvalues(p_model)`.

Rationale: each `D` is a within-gene difference, so per-gene wild-type baselines
cancel and genes (nested in species) act as blocks. Pooling tests the model main
effect with maximum power. **Caveat (documented in docstring):** a TF whose model
effect flips sign between species washes out here — that interaction is what
function C surfaces.

### C — `interaction_model_tf_significance`

```python
def interaction_model_tf_significance(
    group_a_pair: Tuple[str, str],
    group_b_pair: Tuple[str, str],
) -> pd.DataFrame:
    """Per-TF test of whether the model effect DIFFERS between two gene groups.

    Each ``*_pair`` is ``(model_a_run_dir, model_b_run_dir)`` for one gene group.
    For every TF, the per-gene contrast ``D = diff_a − diff_b`` is formed within
    each group and the two D distributions (group A genes vs group B genes) are
    compared with a two-sided Mann–Whitney U test (unpaired: the groups have
    different genes). p-values are BH-corrected across TFs.

    Returns one row per TF with columns
    ``tf, n_a, n_b, median_D_a, median_D_b, u_stat, p_interaction,
    q_interaction``, sorted by ``q_interaction`` ascending.
    """
```

Implementation notes:

- `contrast_mat_a`, `contrast_mat_b` via `_pair_contrast_matrix` for each group.
- Union TFs across both groups; a TF absent in a group contributes that group's
  all-zero D vector (length = that group's gene count), consistent with the fill-0
  convention elsewhere.
- Per TF: `scipy.stats.mannwhitneyu(d_a, d_b, alternative="two-sided")`; on
  `ValueError` (e.g. both vectors identical/degenerate) record `u_stat=nan,
  p=nan`. Add `from scipy.stats import mannwhitneyu` to the module imports.
- `q_interaction = _bh_qvalues(p_interaction)`.

Interpretation: a significant TF means the model swap does *different* things to
that TF depending on whether the genes are ara or zea (species-dependent model
behavior). This is the unpaired interaction axis, distinct from A's "is there a
model effect at all".

## Files touched / created

- **Edit** `src/workflows/evo_alg_pooled_plots/tf_comparison/tf_comparison_calc.py`
  — add `_pair_contrast_matrix`, `pooled_model_tf_significance`,
  `interaction_model_tf_significance`; refactor `paired_tf_significance` onto the
  helper; add the `mannwhitneyu` import.
- **New** `src/workflows/evo_alg_pooled_plots/tf_comparison/random_start_comparison.py`
- **New** `src/workflows/evo_alg_pooled_plots/tf_comparison/model_comparison.py`
- **Edit** `summary.md` — document the two new scripts, the two new calc
  functions, the "group by model / no row stars / A+C CSV-only" decisions, and
  the new outputs.
- **New** `test/.../test_random_start_comparison.py` — integration smoke test.
- **New** `test/.../test_model_comparison.py` — integration smoke test.
- **Edit** `test/.../test_tf_comparison_calc.py` — add cases for the two new
  functions and the helper; confirm `paired_tf_significance` cases still pass
  after the refactor.
- No `.gitignore` change expected — `evo_alg_pooled_plots/**/*.png` and `**/*.csv`
  already cover the new subfolders (verify).

## Testing plan (`unittest`, Arrange-Act-Assert, mocked/synthetic data)

Reuse the existing synthetic-run writer pattern (`_write_run` /
`_write_annotated_peaks` + `tempfile`) from the current tests.

New calc unit tests (`test_tf_comparison_calc.py`):

- `PooledModelTfSignificanceTest`: two synthetic pairs with known per-gene D;
  assert `median_D`, `n_nonzero_D`, `n_genes` (= pooled count), column set, sort
  order, and that BH q's are monotone in p. Include a TF present in only one gene
  source (zeros in the other) to lock the fill-0 behavior. Include a sign-flip TF
  and assert the pooled effect attenuates (documents the caveat).
- `InteractionModelTfSignificanceTest`: two synthetic groups where one TF has
  clearly different D distributions and another has matched distributions; assert
  the differing TF gets the smaller `p_interaction`, check `n_a`/`n_b`/`median_D_a`
  /`median_D_b`, and the NaN path for a degenerate (all-equal) TF.
- `_pair_contrast_matrix` (via a focused test or through the above): shared-gene
  intersection, TF union, fill-0.

Integration smoke tests (one file per script), following
`test_minmax_comparison.py`: build synthetic run dirs in a temp dir, point the
script's output dir at a temp dir, run `main(...)` with injected paths, assert the
expected CSV + PNG files are created and are non-empty.

## Verification

```bash
conda run -n deepCREshap python -m pytest test/workflows/evo_alg_pooled_plots/tf_comparison/
```

All unit + integration tests pass; no lint/diagnostic warnings on new/edited
files.

End-to-end (real data, hardcoded paths — run by the user as a smoke check once

paths are wired in):

```bash
conda run -n deepCREshap python -m src.workflows.evo_alg_pooled_plots.tf_comparison.random_start_comparison
conda run -n deepCREshap python -m src.workflows.evo_alg_pooled_plots.tf_comparison.model_comparison
```

## Open questions for the user

1. Confirm the two assumptions above (cell stars stay; script names).
2. Column labels for the 2×2 (`araG/araM` etc.) — keep or prefer a different
   short form?
3. Should either script default to a `top_bottom_n` slice (minmax used 5) or show
   all TFs?
