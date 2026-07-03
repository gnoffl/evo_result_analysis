# TF comparison across evolutionary runs — goal, decisions, implementation

This folder contains a small **toolbox** of shared functions plus **thin
per-analysis scripts** that compose them. The toolbox is split into a
calculation module (`tf_comparison_calc.py`, loading + statistics, zero plotting)
and a visualization module (`tf_comparison_plot.py`, the heatmap + star helpers,
zero stats/IO). Each concrete comparison is its own short script that picks the
runs, groupings, and tests it needs and writes its outputs into a sibling
subfolder. The first such script is `minmax_comparison.py`, which reproduces the
original minimization-vs-maximization figure into `minmax_comparison/`. Future
siblings (e.g. `model_comparison.py`, `startgene_comparison.py`) differ only in
their run list, ordering, and which tests apply.

This document explains *what* the analysis answers, *why* it is built the way it
is, and *how* to read the figures it produces. It is written so that, coming back
to it later, the whole analysis is self-explanatory.

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

`minmax_comparison.py` lists four runs (directory, label) in display order, with
the maximization runs grouped on the left and the minimization runs on the right:

| label   | group | dataset                              | ~#genes |
|---------|-------|--------------------------------------|---------|
| ara max | left  | `ara_msr_max_single`                 | ~999    |
| GOF     | left  | `GOF/GOF_single`                     | ~100    |
| ara min | right | `ara_msr_min_single`                 | ~999    |
| LOF     | right | `LOF/LOF_single`                     | ~70     |

The left/right grouping is expressed by the `LEFT_COLUMNS` / `RIGHT_COLUMNS`
label lists, not baked into the loader — `build_matrix` keeps columns in the
order given. All four appear in the heatmap. The **inter-run paired contrast**
uses only the two ara runs (see §3.2). The **intra-run test** is computed for all
four runs independently.

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

For the ara runs, the intra-run q-values (`q_a`, `q_b` — the neutral per-run
columns of the paired table) are already produced as a by-product of the paired
analysis (`paired_tf_significance`) and are reused directly — there is no
redundant computation.

### 3.4 Three two-sided tests per TF (paired ara analysis)

For each TF we compute, over the shared genes (a gene/TF combination absent in a
run contributes `diff = 0`), three **Wilcoxon signed-rank** tests:

The two runs are passed as a neutral A/B pair, so the table columns are named
`*_a` / `*_b`; in `minmax_comparison.py`, A is the max run and B is the min run.

| test       | per-gene quantity tested    | answers                                                                              |
|------------|-----------------------------|---------                                                                             |
| `contrast` | `D = diff_a − diff_b` vs 0  | does this TF behave **differently** between the two runs (max vs min)? (primary)     |
| `a`        | `diff_a` vs 0               | is this TF's binding changed in run A (when **maximizing**)?                         |
| `b`        | `diff_b` vs 0               | is this TF's binding changed in run B (when **minimizing**)?                         |

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
  `D` reduces to `optimized_a − optimized_b`.
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
q-value. For the paired analysis: `q_contrast`, `q_a`, `q_b`. For each intra-run
analysis: `q_intra`.

### 3.6 Use a library; toolbox + thin scripts, not a config dict

BH is applied via `statsmodels.stats.multitest.multipletests(method="fdr_bh")`
rather than a hand-rolled implementation. The functions are kept as a flat
toolbox — no classes — and each concrete comparison is a small script that calls
them. We explicitly **rejected** a config-driven `STUDIES` dict: a thin script
per analysis stays simpler and more readable, and the toolbox functions are
de-baked of any max/min assumption so they compose for other contrasts (different
models, different start genes).

## 4. Implementation

### `tf_comparison_calc.py` — loading + statistics

Private helpers:

- **`_diff_series_to_matrix(diff, genes, tfs)`** — unstacks a
  `(core_gene, tf)`-indexed diff Series into a genes × TFs DataFrame, filling
  missing combinations with 0. Used by both significance functions.
- **`_tf_stats(arr)`** — returns `(median, n_nonzero, wilcoxon_p)` for one
  per-gene diff vector. Used in both significance functions (four call sites total).
- **`_wilcoxon_pvalue`**, **`_bh_qvalues`** — the test and the BH correction.
- **`_pair_contrast_matrix(run_a_directory, run_b_directory)`** — builds the
  shared-gene × union-TF matrices `a_mat`, `b_mat` and their difference
  `contrast_mat = a_mat − b_mat` for a run pair (fill-0 for absent combinations).
  `paired_tf_significance`, `pooled_model_tf_significance`, and
  `interaction_model_tf_significance` all share this one copy of the
  intersect/unstack/fill-0 logic.

Public functions:

1. **`load_per_gene_diffs(run_dir, n_core_fields=2)`** — reads the run's single
   `deepcis_scan/*_annotated_peaks_*.csv`, keeps the `reference` and
   `max_mutated` rows (the `difference` signal is ignored), collapses each full
   sequence id to its **core id** (first `n_core_fields` underscore fields, e.g.
   `1_AT1G01150_gene:..._<timestamp>` → `1_AT1G01150`; this is what makes the two
   runs matchable, since full ids differ only by timestamp), counts peaks per
   `(core_gene, tf, signal_type)`, and returns `diff = max_mutated − reference`
   per `(core_gene, tf)`. Note: `max_mutated` is the *optimized* sequence in both
   directions — it is legacy naming, so in the min run it is the *minimized*
   sequence. **`n_core_fields`** defaults to 2 (natural-gene ids like
   `1_AT1G01150`). Random-start sequences are named
   `random_sequence_<index>_<timestamp>` whose first two fields are always
   `random_sequence`; they need `n_core_fields=3` (core `random_sequence_000`),
   otherwise **all sequences collapse into one replicate** and every intra-run
   Wilcoxon test degenerates to n=1 → `p = 1`. `single_run_tf_significance` takes
   the same argument and `random_start_comparison.py` passes 3.
2. **`paired_tf_significance(run_a_directory, run_b_directory, label_a="a",
   label_b="b")`** — intersects the genes, builds the three per-gene vectors per
   TF via `_diff_series_to_matrix` and `_tf_stats`, runs the three Wilcoxon tests
   (NaN on failure), and BH-corrects each p-column. The contrast columns are
   `median_D, n_nonzero_D, p_contrast, q_contrast, n_genes`; the per-run columns
   are `median_diff_<label_a>, n_nonzero_<label_a>, p_<label_a>, q_<label_a>` and
   their `<label_b>` counterparts. `label_a`/`label_b` default to the neutral
   `a`/`b` (so `minmax_comparison` keeps `q_a`/`q_b`); a caller passes meaningful
   names (e.g. `ara_model`/`zea_model`) to make its CSV self-describing. Sorted by
   `q_contrast`.
3. **`single_run_tf_significance(run_dir)`** — loads per-gene diffs for one run,
   builds the per-TF diff matrix via `_diff_series_to_matrix`, tests each TF's
   diff vs 0 with `_tf_stats`, and BH-corrects. Returns one row per TF with
   `tf, n_genes, median_diff, n_nonzero, p_intra, q_intra`, sorted by `q_intra`.
4. **`build_matrix(runs, normalization)`** — assembles the TF × run matrix of
   normalized `diff_calc`. `runs` is a list of `(directory, label)` tuples;
   **columns are kept in the order given** (no max/min reordering — grouping is
   the caller's job).
5. **`order_tfs_by_group_contrast(matrix, left_columns, right_columns)`** — orders
   rows by `mean(left) − mean(right)` descending (the max-minus-min behavior).
   **`order_tfs_by_mean(matrix)`** orders by overall row mean descending (the flat
   case, for analyses with no left/right grouping).
6. **`top_bottom_tfs(matrix, n)`** — keeps the top-N and bottom-N rows of an
   already-ordered matrix (order preserved, overlaps de-duplicated).
7. **`pooled_model_tf_significance(model_pairs)`** — pooled per-TF **model main
   effect** across several paired gene sources. Each pair is
   `(model_a_run_dir, model_b_run_dir)` sharing that source's genes; the per-gene
   contrast `D = diff_a − diff_b` is formed within each pair (via
   `_pair_contrast_matrix`), the D vectors from all pairs are concatenated over
   the disjoint gene sets, and tested against 0 with a two-sided Wilcoxon test,
   BH-corrected across TFs. Columns: `tf, n_genes, median_D, n_nonzero_D,
   p_model, q_model`, sorted by `q_model`. Caveat: a TF whose model effect flips
   sign between gene sources washes out here (that interaction is what function 8
   surfaces).
8. **`interaction_model_tf_significance(group_a_pair, group_b_pair, label_a="a",
   label_b="b")`** — per-TF test of whether the model effect **differs between
   two gene groups**. The per-group contrast `D = diff_a − diff_b` is formed
   within each pair, and the two D distributions (group A genes vs group B genes)
   are compared with a two-sided **Mann–Whitney U** test (unpaired: the groups
   have different genes), BH-corrected across TFs. Columns: `tf, n_<label_a>,
   n_<label_b>, median_D_<label_a>, median_D_<label_b>, u_stat, p_interaction,
   q_interaction`, sorted by `q_interaction`; `label_a`/`label_b` default to
   `a`/`b` and `model_comparison` passes `ara_genes`/`zea_genes`. A degenerate TF
   (test undefined) records NaN.

### `tf_comparison_plot.py` — visualization

- **`q_to_stars`**, **`STAR_THRESHOLDS`**, **`COLORMAP`** — the star mapping and
  shared style constants.
- **`plot_heatmap(matrix, annotate, cbar_label, row_stars, cell_stars,
  separator_after_column)`** — draws the heatmap with two optional significance
  overlays:
  - `row_stars`: inter-run significance stars appended to TF row labels.
  - `cell_stars`: intra-run significance stars appended to each cell's numeric
     annotation (passed as a `{run_label: {tf: stars}}` dict; when provided, the
     heatmap uses a string annotation matrix so each cell shows e.g. `"0.12*  "`).
  - `separator_after_column`: draws a thick vertical divider after that column
     index (only when `0 < value < n_runs`); `None` draws no separator (the flat
     case). `row_stars`/`cell_stars` are the **"compute everything, display a
     subset"** mechanism — a script computes whatever stats it wants, writes all
     CSVs, then passes only the chosen q-value columns as stars.

### `minmax_comparison.py` — thin orchestration script

Defines the four runs in display order, the `LEFT_COLUMNS` / `RIGHT_COLUMNS`
groupings, the paired ara pair, the GOF/LOF single runs, and `OUTPUT_DIR =
<this folder>/minmax_comparison`. `main()` runs the paired analysis (CSV +
`row_stars` from `q_contrast` + ara `cell_stars` from `q_a`/`q_b`), the GOF/LOF
intra-run analyses (CSVs + `q_intra` cell stars), then for each normalization
builds the matrix → `order_tfs_by_group_contrast` → `top_bottom_tfs(n=5)` →
`plot_heatmap(separator_after_column=len(LEFT_COLUMNS))` and saves the PNG. Its
arguments default to the module constants but can be overridden (the integration
test passes synthetic runs and a temp output dir).

### Outputs (in `minmax_comparison/`)

- **`compare_TFs_significance.csv`** — paired ara inter-run analysis; one row per
  TF, columns: `tf, n_genes, median_D, n_nonzero_D, p_contrast, median_diff_a,
  n_nonzero_a, p_a, median_diff_b, n_nonzero_b, p_b, q_contrast, q_a, q_b`.
  Sorted by `q_contrast`.
- **`compare_TFs_significance_GOF.csv`** and **`compare_TFs_significance_LOF.csv`**
  — intra-run analysis for GOF and LOF respectively; one row per TF, columns:
  `tf, n_genes, median_diff, n_nonzero, p_intra, q_intra`. Sorted by `q_intra`.
- **`compare_TFs_per_gene_top5.png`** and **`compare_TFs_log_fold_change_top5.png`**
  — the two heatmaps (top/bottom 5 TFs; see §5).

## 5. How to read the visualization

Two heatmaps are produced, differing only in how `diff` is normalized:

- **`compare_TFs_per_gene_top5.png`** — cell value = `diff_calc / number_of_genes`
  (average net peak change per gene). This is the main, directly interpretable
  view.
- **`compare_TFs_log_fold_change_top5.png`** — cell value =
  `log2(optimized_total / reference_total)`. A relative view; complementary to
  the per-gene one.

In both:

- **Rows = TF families, columns = runs.** A thick black vertical line separates
  the max runs (left) from the min runs (right).
- **Color** is a symmetric diverging scale (`RdBu_r`) centered at 0: **red =
  binding introduced** (positive diff), **blue = binding removed** (negative
  diff), white = no change. The scale limit is symmetric (±max|value|) so red and
  blue are comparable.
- **Row order:** TFs are sorted by `mean(left columns) − mean(right columns)`
  descending (`order_tfs_by_group_contrast`, i.e. max-minus-min here).
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

## 6. Sibling scripts

### `random_start_comparison.py` — single random-start maximization run

The single-run analogue of `minmax_comparison.py`: one maximization run whose
starting sequences are **random** (not natural plant genes). It answers *starting
from random sequence, which TF families does the optimizer systematically
introduce/remove to raise predicted expression?*

Pure reuse of the toolbox — no new functions. Per normalization:
`build_matrix([(RUN_DIR, LABEL)])` → `order_tfs_by_mean` (flat, no left/right
grouping) → optional `top_bottom_tfs` → `plot_heatmap(cell_stars=…,
row_stars=None, separator_after_column=None)`. The single intra-run significance
layer comes from `single_run_tf_significance(RUN_DIR, CORE_ID_FIELDS)` (`q_intra`
→ cell stars); there are no row-label stars because there is only one run.

**`CORE_ID_FIELDS = 3`** (not the default 2): random-start sequences are named
`random_sequence_<index>_<timestamp>`, so 3 fields are needed to keep each of the
~104 sequences a distinct replicate. With the default 2 they would all collapse
to `random_sequence` and every intra-run test would degenerate to n=1 → `p = 1`.

Outputs in `random_start_comparison/`: `random_start_comparison_significance.csv`
plus `random_start_comparison_per_gene.png` and `…_log_fold_change.png`.

### `model_comparison.py` — 2×2 gene-source × fitness-model, grouped by model

A four-run 2×2 analysis: natural genes from two species (Arabidopsis "ara", *Zea
mays* "zea") each optimized under two deepCRE fitness models (ara model, zea
model). **All four runs maximize.** Headline question: *does the predictor model
drive the TF strategy?*

**Pairing structure:** the two ara-gene runs share the same ara genes; the two
zea-gene runs share the same zea genes; ara and zea gene sets are disjoint. So
the model swap is **paired by gene within each gene source**, and the gene-source
axis is **unpaired**. A consistent **A = ara model, B = zea model** convention is
used so `D = diff_a − diff_b` means the same thing in both sources.

**Figure — grouped by model:** left group = both ara-model runs (`araG/araM`,
`zeaG/araM`), right group = both zea-model runs (`araG/zeaM`, `zeaG/zeaM`),
divider after column 2; rows ordered by `order_tfs_by_group_contrast` =
`mean(ara-model) − mean(zea-model)`. **Row-label (inter-run) stars are dropped**
by choice; only **cell (intra-run) stars** stay. Those cell stars are reused from
the two stratified paired tables' per-model q-columns (one paired table per gene
source covers its two runs' intra-run tests), so no separate
`single_run_tf_significance` calls are needed. To make the CSVs self-describing,
`model_comparison` passes meaningful side-labels (`label_a="ara_model"`,
`label_b="zea_model"`) so the columns read `q_ara_model` / `q_zea_model` rather
than the toolbox-default `q_a` / `q_b`:

- `araG/araM` → ara pair `q_ara_model`; `araG/zeaM` → ara pair `q_zea_model`
- `zeaG/araM` → zea pair `q_ara_model`; `zeaG/zeaM` → zea pair `q_zea_model`

**Statistics (all CSV-only, not shown on the figure):**

- Two **stratified paired model contrasts** (`paired_tf_significance` per gene
  source): `model_comparison_significance_ara_genes.csv` (model effect on ara
  genes) and `…_zea_genes.csv` (model effect on zea genes). `q_contrast` = the
  model effect within that gene source; `q_ara_model` / `q_zea_model` = each
  model's intra-run change vs wild-type.
- **A — pooled model main effect** (`pooled_model_tf_significance` over both
  pairs): `model_comparison_significance_pooled_model.csv`. Tests the overall
  model effect with maximum power by blocking on gene.
- **C — interaction / species-specificity** (`interaction_model_tf_significance`
  of the two pairs, labelled `ara_genes` / `zea_genes`):
  `model_comparison_significance_interaction.csv`. Columns `median_D_ara_genes` /
  `median_D_zea_genes` give the model contrast within each gene source; the
  Mann–Whitney `q_interaction` tests whether the model swap does *different*
  things depending on gene source.

Figure outputs in `model_comparison/`: `model_comparison_per_gene.png` and
`model_comparison_log_fold_change.png`.

**Reference-cancellation assumption (verified):** `D = diff(ara-model) −
diff(zea-model)` per gene assumes the wild-type reference peak counts are
identical across the two models of a gene source (the deepCIS reference scan is
model-independent — the "model" is the evolution *fitness* predictor, not the TF
annotator). This mirrors §3.4, so `D` reduces to `optimized_a − optimized_b`.
Verified on the real runs (2026-07-02): within each gene source the two runs
share exactly the same genes (105 ara / 100 zea) with **zero** reference-count
mismatches across all `(gene, TF)` pairs, so no reference delta leaks into `D`.

## 7. Reproducing

From the repo root, in the `deepCREshap` conda env:

```bash
conda run -n deepCREshap python -m src.workflows.evo_alg_pooled_plots.tf_comparison.minmax_comparison
conda run -n deepCREshap python -m src.workflows.evo_alg_pooled_plots.tf_comparison.random_start_comparison
conda run -n deepCREshap python -m src.workflows.evo_alg_pooled_plots.tf_comparison.model_comparison
```

Unit tests:

```bash
conda run -n deepCREshap python -m pytest test/workflows/evo_alg_pooled_plots/tf_comparison/
```
