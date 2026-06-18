# TF comparison across evolutionary runs — goal, decisions, implementation

This folder contains a single, one-off analysis script, `compare_TFs.py`, plus
its outputs. This document explains *what* it answers, *why* it is built the way
it is, and *how* to read the figures it produces. It is written so that, coming
back to it later, the whole analysis is self-explanatory.

## 1. Background and goal

We run an evolutionary algorithm that mutates gene **promoter sequences** to
either **maximize** ("max") or **minimize** ("min") a predicted gene-expression
fitness. For every gene we therefore have a **reference** (wild-type) promoter
and an **optimized** ("mutated") promoter.

A separate algorithm (deepCIS sliding-window scan → peak detection) counts, for
each promoter, how often each **transcription factor (TF)** family is predicted
to bind. Per gene we thus get a peak count per TF for the reference and for the
optimized sequence. The per-TF net change is

```python
diff = peak_count(optimized) − peak_count(reference)
```

(positive = TF binding **introduced** by optimization, negative = **removed**).

**Biological question:** which TF families does the optimizer systematically
*introduce* to raise expression and *remove* to lower it (and vice versa for
repressors)? And **is that change statistically significant**, or just noise?

The script answers this with two complementary outputs:

1. A **heatmap** giving a descriptive overview of `diff` across several runs.
2. **Two layers of per-TF significance tests** (with multiple-testing correction):
   an inter-run paired contrast for the two Arabidopsis runs, and an independent
   intra-run test for every run (including GOF/LOF).

## 2. The runs

`RUNS` lists four runs (directory, direction, label):

| label   | direction | dataset                              | ~#genes |
|---------|-----------|--------------------------------------|---------|
| ara max | max       | `ara_msr_max_single`                 | ~999    |
| GOF     | max       | `GOF/GOF_single`                     | ~100    |
| ara min | min       | `ara_msr_min_single`                 | ~999    |
| LOF     | min       | `LOF/LOF_single`                     | ~70     |

All four appear in the heatmap. The **inter-run paired contrast** uses only the
two ara runs (see §3.2). The **intra-run test** is computed for all four runs
independently.

## 3. Key decisions and why we made them

### 3.1 Genes are the replicates — not aggregated totals

The original peak summary (`*_peak_summary.csv`) collapses every gene into a
**single total count per TF per run**. A single count pair (reference total vs
optimized total) has **no measure of variability**, so no honest p-value can be
derived from it without pretending every peak is an independent event (Poisson),
which ignores gene-to-gene structure. The **genes are the natural replicates**,
so the test must operate on **per-gene** counts. These per-gene counts are
recovered from each run's `deepcis_scan/*_annotated_peaks_*.csv` (one row per
detected peak) — no need to re-run the deepCIS predictions.

### 3.2 Paired inter-run test on the two ara runs only (not pooled with GOF/LOF)

We considered pooling both max runs (ara max + GOF) against both min runs
(ara min + LOF) and doing an unpaired comparison. We rejected that because:

- **Batch effects:** the ara runs and the GOF/LOF runs were run with *different
  parameters*, so pooling them mixes incompatible batches.
- **Imbalance:** GOF/LOF are much smaller (~100 / ~70 genes) than the ara runs
  (~999 each).
- **A cleaner option exists:** `ara_msr_max_single` and `ara_msr_min_single`
  contain **exactly the same genes** (verified: same 999 genes, matched on the
  core gene id). The same gene optimized up vs down is a **paired** observation,
  which is more powerful and removes per-gene baseline variance.

So the **inter-run** significance analysis is **ara max vs ara min, paired by
gene**. GOF/LOF cannot participate in an inter-run paired comparison because they
have different gene sets and different run parameters.

### 3.3 Intra-run tests for all four runs

Even though GOF/LOF cannot be paired with another run, the question "does this
TF's binding change significantly vs reference within this run?" is still
answerable independently. For each run we test `diff = max_mutated − reference`
vs 0 per TF using a Wilcoxon signed-rank test over that run's own genes, then
BH-correct across TFs. This gives GOF and LOF their own significance layer and
makes the heatmap cell annotations meaningful for every column.

For the ara runs, the intra-run q-values (`q_max`, `q_min`) are already produced
as a by-product of the paired analysis (`paired_tf_significance`) and are reused
directly — there is no redundant computation.

### 3.4 Three two-sided tests per TF (paired ara analysis)

For each TF we compute, over the shared genes (a gene/TF combination absent in a
run contributes `diff = 0`), three **Wilcoxon signed-rank** tests:

| test       | per-gene quantity tested        | answers                                                                              |
|------------|---------------------------------|---------                                                                             |
| `contrast` | `D = diff_max − diff_min` vs 0  | does this TF behave **differently** between maximization and minimization? (primary) |
| `max`      | `diff_max` vs 0                 | is this TF's binding changed when **maximizing**?                                    |
| `min`      | `diff_min` vs 0                 | is this TF's binding changed when **minimizing**?                                    |

- **Wilcoxon signed-rank** (not a t-test): the per-gene diffs are integer counts,
  heavily tied at zero and not normally distributed — a non-parametric paired
  test is the honest choice.
- **Two-sided**, deliberately: TFs can be **activators or repressors**, so we do
  *not* expect a single direction. Some TFs are enriched during maximization and
  removed during minimization; others the reverse. A one-sided test would miss
  the repressors.
- The **`contrast` test is primary** because it matches the heatmap's max−min row
  ordering and directly captures "differentiates up- from down-optimization".
  When the reference scan is identical across the two runs (it is, same genes),
  `D` reduces to `optimized_max − optimized_min`.
- Tests that are undefined (e.g. a TF whose diff is zero for every gene) are
  recorded as `NaN` rather than a fake p-value.

### 3.5 Benjamini–Hochberg FDR correction

We test ~30–60 TFs at once. At α = 0.05 we would expect several "significant"
hits by chance alone. **Benjamini–Hochberg** controls the **False Discovery
Rate** — the expected *fraction of false positives among the TFs we call
significant*. "FDR 0.05" means ≈5% of the flagged TFs are expected to be
spurious. We chose BH (not Bonferroni, which controls the much stricter
probability of *any* false positive and would kill power) because this is an
exploratory screen. Each p-value column is BH-corrected independently into a
q-value. For the paired ara analysis: `q_contrast`, `q_max`, `q_min`. For each
intra-run analysis: `q_intra`.

### 3.6 Use a library, keep the script simple

BH is applied via `statsmodels.stats.multitest.multipletests(method="fdr_bh")`
rather than a hand-rolled implementation. The script is intentionally a flat set
of functions — no classes, no CLI — because it is a single-purpose analysis.

## 4. Implementation (`compare_TFs.py`)

### Private helpers

- **`_diff_series_to_matrix(diff, genes, tfs)`** — unstacks a
  `(core_gene, tf)`-indexed diff Series into a genes × TFs DataFrame, filling
  missing combinations with 0. Used by both significance functions.
- **`_tf_stats(arr)`** — returns `(median, n_nonzero, wilcoxon_p)` for one
  per-gene diff vector. Used in both significance functions (four call sites total).

### Public data-flow functions

1. **`load_per_gene_diffs(run_dir)`** — reads the run's single
   `deepcis_scan/*_annotated_peaks_*.csv`, keeps the `reference` and
   `max_mutated` rows (the `difference` signal is ignored), collapses each full
   gene id to its **core id** (first two underscore fields, e.g.
   `1_AT1G01150_gene:..._<timestamp>` → `1_AT1G01150`; this is what makes the two
   runs matchable, since full ids differ only by timestamp), counts peaks per
   `(core_gene, tf, signal_type)`, and returns `diff = max_mutated − reference`
   per `(core_gene, tf)`. Note: `max_mutated` is the *optimized* sequence in both
   directions — it is legacy naming, so in the min run it is the *minimized*
   sequence.
2. **`paired_tf_significance(max_run_dir, min_run_dir)`** — intersects the genes,
   builds the three per-gene vectors per TF via `_diff_series_to_matrix` and
   `_tf_stats`, runs the three Wilcoxon tests (NaN on failure), and BH-corrects
   each p-column. Returns one row per TF, sorted by `q_contrast`.
3. **`single_run_tf_significance(run_dir)`** — loads per-gene diffs for one run,
   builds the per-TF diff matrix via `_diff_series_to_matrix`, tests each TF's
   diff vs 0 with `_tf_stats`, and BH-corrects. Returns one row per TF with
   `tf, n_genes, median_diff, n_nonzero, p_intra, q_intra`, sorted by `q_intra`.
4. **`build_matrix` / `order_tfs`** — the descriptive side: assemble the TF × run
   matrix of normalized `diff_calc` and order rows by `mean(max) − mean(min)`.
5. **`plot_heatmap`** — draws the heatmap with two optional significance overlays:
   - `row_stars`: inter-run significance stars appended to TF row labels.
   - `cell_stars`: intra-run significance stars appended to each cell's numeric
     annotation (passed as a `{run_label: {tf: stars}}` dict; when provided, the
     heatmap uses a string annotation matrix so each cell shows e.g. `"0.12*  "`).
6. **`main()`** — runs both significance analyses, writes the CSVs, builds both
   star maps, and saves the heatmaps. All outputs land **next to the script**.

### Outputs in this folder

- **`compare_TFs_significance.csv`** — paired ara inter-run analysis; one row per
  TF, columns: `tf, n_genes, median_D, n_nonzero_D, p_contrast, median_diff_max,
  n_nonzero_max, p_max, median_diff_min, n_nonzero_min, p_min, q_contrast,
  q_max, q_min`. Sorted by `q_contrast`.
- **`compare_TFs_significance_GOF.csv`** and **`compare_TFs_significance_LOF.csv`**
  — intra-run analysis for GOF and LOF respectively; one row per TF, columns:
  `tf, n_genes, median_diff, n_nonzero, p_intra, q_intra`. Sorted by `q_intra`.
- **`compare_TFs_per_gene.png`** and **`compare_TFs_log_fold_change.png`** — the
  two heatmaps (see §5).

## 5. How to read the visualization

Two heatmaps are produced, differing only in how `diff` is normalized:

- **`compare_TFs_per_gene.png`** — cell value = `diff_calc / number_of_genes`
  (average net peak change per gene). This is the main, directly interpretable
  view.
- **`compare_TFs_log_fold_change.png`** — cell value =
  `log2(optimized_total / reference_total)`. A relative view; complementary to
  the per-gene one.

In both:

- **Rows = TF families, columns = runs.** A thick black vertical line separates
  the max runs (left) from the min runs (right).
- **Color** is a symmetric diverging scale (`RdBu_r`) centered at 0: **red =
  binding introduced** (positive diff), **blue = binding removed** (negative
  diff), white = no change. The scale limit is symmetric (±max|value|) so red and
  blue are comparable.
- **Row order:** TFs are sorted by `mean(max runs) − mean(min runs)` descending.
  TFs **introduced when maximizing and removed when minimizing** rise to the top;
  the bottom rows are their mirror image. With this ordering, the max and min
  columns should read as **near mirror images** (red ↔ blue) for a TF that
  behaves consistently — that visual mirror *is* the biological signal.
- **Row label stars** are appended to a TF's row label based on the primary
  `q_contrast` (ara max vs ara min paired contrast): `*` q < 0.05, `**` q < 0.01,
  `***` q < 0.001. A starred row label means the TF behaves significantly
  differently between maximization and minimization after FDR correction.
- **Cell stars** are appended to each cell's numeric value based on the intra-run
  `q_intra` for that run: `*` q < 0.05, `**` q < 0.01, `***` q < 0.001. A
  starred cell means the TF's binding is significantly changed vs reference within
  that run after FDR correction.

**Caveats to remember:**

- Row label stars come *only* from the paired ara contrast. GOF/LOF row labels
  are never starred because no inter-run paired comparison exists for them.
- Cell stars are independent per column and BH-corrected within each run
  separately — they do not correct across runs.

## 6. Reproducing

From the repo root, in the `deepCREshap` conda env:

```bash
python src/workflows/evo_alg_pooled_plots/tf_comparison/compare_TFs.py
```

Unit tests:

```bash
pytest test/workflows/evo_alg_pooled_plots/tf_comparison/test_compare_TFs.py
```
