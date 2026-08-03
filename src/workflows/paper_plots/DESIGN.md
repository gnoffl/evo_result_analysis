# Publication figure layer — design

**Date:** 2026-07-13 (design), last reviewed against the code 2026-08-03
**Status:** implemented — the ax-injection refactor is complete (see
"Implementation status"), and the composition scripts for figures 1–4 and 6 are in
`compose_plots_mpl.py` / `fig1_miniatures.py`. Per-figure sections below are ordered
by the date they were written (§3 fig2, §3b–3f fig4, §3g fig3, §3h fig6,
§3i fig1), not by figure number.

## Problem

Publication figures combine "standardized" repeatable plots (`src/analysis/`, e.g.
`motives/`, `overview/`, `mutations/`, `starrseq/`) with more specialized, less
repeatable analyses (`src/workflows/`, e.g. `evo_alg_pooled_plots/tf_comparison`).
Three recurring pain points:

1. **Non-uniform figure sizes** — each plot function hardcodes its own `figsize`
   (`(6.67, 4)`, `(7, 5)`, `(12, 6)`, `(8, 5.333)`, `plt.figure()` with no size, …).
2. **Varying text sizes** — literal `fontsize=12/13/15/17` scattered per call, plus
   inconsistent `sns.set_context("notebook", 1.2)` vs `("poster", 0.8)` vs
   `("notebook", 1.0)`.
3. **Publication-only details** — some annotations (significance bars, insets,
   shared legends, arrows) are wanted only in the publication iteration, not in
   every routine analysis run.

### Root cause

The plot functions in `src/analysis/` use the pyplot **state-machine** API and each
independently bakes in size and font choices. On top of that,
`workflows/figure_composition.py` composes figures by **scaling each saved SVG to
fit an arbitrary panel rectangle** — which means even identical source text renders
at different apparent sizes on the page. **Scale-to-fit composition is fundamentally
incompatible with uniform typography**; text size cannot be fixed downstream of it.

Since all source plots can be re-rendered (data + `deepCREshap` conda env are
available), the correct fix is to fix size and style **at render time in one
controlled context**, not at composition time by rescaling.

## Approach (chosen)

Render each multi-panel publication figure as **one** matplotlib figure: a single
`plt.figure()` with a `gridspec`/`subfigures` layout, where panels are drawn by the
existing `src/analysis` / `src/workflows` plot functions called with an `ax` handed
to them, under **one shared stylesheet**. Publication-only extras are added inline
in the per-figure composition script.

Benefits: uniform text by construction (one rcParams, nothing rescaled); uniform
panel geometry (set in the gridspec in physical units); publication-only details
live in the composition script and never leak into routine runs; vector-native
output with no SVG surgery.

`workflows/figure_composition.py` is **demoted** to a fallback for genuinely frozen
raster panels only. Nothing in the current figures needs it.

## Components

All new code lives in `src/workflows/paper_plots/`.

### 1. `style.py` — shared style module (single source of truth)

- **`publication_style()`** — context manager activating the publication stylesheet
  (rcParams: font family, base font size, axis/tick/legend/label sizes, line widths,
  vector `savefig` defaults). Applied as `with publication_style():` so it never
  pollutes global rcParams state. Target format: generic vector (≈180 mm
  double-column, ~7–8 pt text, sans-serif, vector PDF/SVG at high DPI); exact
  numbers tunable later once a journal is fixed.
- **Width constants + helpers** — `SINGLE_COLUMN_MM` (85 mm), `DOUBLE_COLUMN_MM`
  (180 mm), `mm_to_inch()` and `figure_size_inches(width_mm, height_mm)` (the form
  every composition actually calls), so figure/panel sizes are declared in real
  physical units.
- **`panel_label(ax, "A")`** — one consistent way to stamp bold panel letters
  (replaces the scale-dependent letters `figure_composition.py` overlays).
- **`save_publication_figure(fig, path)`** — enforces vector output + DPI in one
  place.

### 2. Backward-compatible ax-injection in the plot functions used by the paper

For each `src/analysis` / `src/workflows` plot function a paper figure needs, add an
optional `ax` parameter following this pattern:

```python
def plot_xxx(..., ax: plt.Axes | None = None) -> None:
    own_figure = ax is None
    if own_figure:
        sns.set_context(...)          # standalone-only styling
        fig, ax = plt.subplots(figsize=(...))
    ax.set_xlabel("Position")          # draw WITHOUT inline font sizes
    # ... draw with ax.* methods, not plt.* state-machine calls ...
    if own_figure:
        ax.xaxis.label.set_fontsize(13)   # re-apply today's explicit sizes
        ax.tick_params(labelsize=12)      # (standalone only)
        fig.savefig(...); plt.close(fig)
```

Rules:

- Convert `plt.*` state-machine calls to `ax.*` methods.
- Move `figsize` and `sns.set_context(...)` **into** the `if own_figure:` branch —
  they only ever made sense there.
- Split inline `fontsize=` literals: draw without a size in the body; re-apply the
  old sizes in the standalone-only tail.

Guarantee: **`ax` provided → old hardcoded style ignored, publication stylesheet
governs; standalone (`ax=None`) → pixel-identical to today.** Existing pipelines
keep working unchanged.

**Scope:** refactor **every** visualization function in `src/analysis/`, then every
visualization function in `src/workflows/` **except `src/workflows/hoffie/`** (left
untouched). Functions handled case-by-case depending on their current shape:

- *Pure `plt.*` state-machine* (e.g. `starrseq_analysis.py`, `analyze_mutations.py`,
  most of `simple_result_stats.py`, `compare_methods.py`): full pattern above.
- *Already return a `Figure`* (e.g. `analyze_peaks.plot_diff_calc_*`,
  `compare_runs.plot_*`, `binding_vs_nonbinding.plot_*`): add `ax`; when given, draw
  onto it and skip figure creation; when `None`, keep returning the `Figure`.
- *Already accept `ax`* (e.g. `correlate.plot_single_dataset`,
  `analyze_in_true_background` inner draw helpers): only ensure inline font sizes are
  guarded so the stylesheet governs when driven from a publication script.

Inventory (files, ~45 functions) — checklist tracked in the task list:

- `src/analysis/`: `motives/analyze_peaks.py`, `motives/deepcis_visualize.py`,
  `mutations/analyze_mutations.py`, `overview/compare_analysis_results.py`,
  `overview/compare_methods.py`, `overview/simple_result_stats.py`,
  `starrseq/starrseq_analysis.py`
- `src/workflows/` (excl. `hoffie/`): `candidate_selection.py`,
  `compare_reruns/compare_runs.py`, `deepCRE_TPM_correlation/correlate.py`,
  `evo_alg_pooled_plots/natural_unconstrained_comparison/*.py`,
  `evo_alg_pooled_plots/tf_comparison/*.py`,
  `mutation_distance_analysis/mutation_distance_analysis.py`,
  `mutation_distribution_analysis/plot_optimization_vs_random.py`,
  `overlap_analysis/binding_vs_nonbinding/binding_vs_nonbinding.py`,
  `overlap_analysis/_common.py`,
  `overlap_analysis/correct_construct/analyze_in_true_background.py`

### 3. Per-figure composition scripts (`src/workflows/paper_plots/`)

Simple scripts, one function per figure (in the spirit of the current
`compose_plots.py`): activate the style, build a `gridspec`, call each plot function
with its `ax`, add publication-only extras inline (significance bars, insets, shared
legends, arrows), stamp panel labels via `panel_label`, save via
`save_publication_figure`. Kept deliberately simple — reusable logic lives in
`style.py` and the plot functions, not here.

Implemented in `compose_plots_mpl.py` (render-time matplotlib composition), one
function per figure with hardcoded input paths so the data backing each published
panel is recorded in version control. Sibling to `compose_plots_svg.py` (the
SVG-scaling fallback), never a replacement for it. Output goes to
`figures/mpl_compositions/` as `.svg`.

- **`fig2()`** — two single-mutation Arabidopsis runs (max top row, min bottom
  row), three data columns: scatter (start vs. final fitness coloured by mutations
  at half max, magma), average Pareto front, histogram of mutations at half max.
  Publication-level styling:
  - Column layout is `scatter | thin shared-colorbar column | pareto | hist`, so
    the three data columns stay equal width (a per-panel colorbar would otherwise
    shrink only the scatter cell). Figure uses `layout="constrained"` (tight_layout
    is incompatible with the row-spanning colorbar axes).
  - Both scatter panels share one colour normalisation (`vmin=0`,
    `vmax=global max mutations`) and one shared colorbar spanning both rows.
  - Pareto and histogram columns each share x/y limits across the two runs (via
    `sync_axis_limits`) for direct run-to-run comparison; scatter y is independent
    (max→~1, min→~0 by objective).
  - Neutral grey (`0.35`) Pareto markers + histogram bars (thin white bar edges);
    Pareto spread drawn as a filled ±1 std band, not error bars.
  - X-tick numbers kept on every panel; the x-axis *label* is shown only on the
    bottom row. Left-margin row labels ("maximization"/"minimization"); panel
    letters A–F.
  - Figure height 100 mm (double-column width). Pareto x-axis cropped to
    `_PARETO_X_MAX` (45): the fronts saturate long before the 90-mutation
    expansion limit, so the flat tail is dropped to fill the panels.
  - Pareto data scanned from each run's raw results folder; hist/scatter load the
    cached `stats_<name>.json`.

New reusable primitives in `style.py`:

- **`sync_axis_limits(axes, sync_x, sync_y)`** — sets a group of axes to the union
  of their limits, so same-type panels are comparable across runs.
- Added `scatter.edgecolors="none"` and outward ticks to `PUBLICATION_RC`.

Backward-compatible params added to `simple_result_stats.py` (defaults preserve
standalone behaviour; the composition script passes the publication values):

- `draw_visualize_start_vs_max_fitness_by_mutations`: `add_colorbar`, `vmin`,
  `vmax`, and now returns the scatter mappable (for a shared colorbar). Its default
  colormap was changed `viridis`→`magma` **globally** (intentional deviation from
  the standalone-preservation guarantee below).
- `show_average_pareto_front`: `color`, `error_style` (`"bars"` default / `"band"`).
- `hist_half_max_mutations`: `color`.

Test note: `sync_axis_limits` is unit-tested (`test_style.py`); the new
`simple_result_stats` params have dedicated unit tests in
`test_simple_result_stats.py` (`add_colorbar` axes count, `vmin`/`vmax` on the
returned mappable, `error_style` band vs bars + invalid-value guard, `color`
applied to Pareto markers and histogram bars).

### 3b. Figure 4 scatter overlays — mean-start line + logit-linear fit

**Date added:** 2026-07-24

Figure 4's four start-vs-final-fitness scatter panels (A/C/E/G, drawn by
`draw_visualize_start_vs_max_fitness_by_mutations`) each carry two overlays,
added as backward-compatible params (defaults off, so standalone output is
unchanged):

- **`mean_start_line`** — a dashed medium-grey vertical line at the mean start
  fitness of the plotted points, drawn behind the scatter (`zorder=0`). It marks
  the average starting point of the runs so the reader can see, at a glance,
  where the "typical" gene begins on the x-axis.
- **`fit_line`** — a best-fit curve of final vs. start fitness.

#### Why not a linear (or polynomial) fit?

The response variable here, `final_fitness`, is a **deepCRE model output bounded
to the open interval (0, 1)**. An ordinary straight-line (or polynomial) fit is
unbounded: fitted or extrapolated values readily fall below 0 or above 1, which
are impossible fitness values, and the straight line cannot represent the
*saturation* that dominates this data (final fitness piles up against 1 for the
maximization/GOF runs and against 0 for the minimization/LOF runs). A `1/x`-style
curve (the first idea considered) saturates but is not naturally confined to
(0, 1) and has no principled link to a bounded response.

#### The logit-linear fit (`fit_logit_linear`)

The model is a straight line on the **logit (log-odds) scale**, mapped back
through the **logistic (sigmoid)** function so the fitted curve lives in (0, 1)
by construction:

```
ŷ(x) = 1 / (1 + exp( -(slope * x + intercept) ))     equivalently
logit(ŷ) = slope * x + intercept,   logit(y) = log( y / (1 - y) )
```

This is a sigmoid: monotonic, asymptoting to 0 as `x → -∞` and to 1 as
`x → +∞`, and never leaving (0, 1). `slope > 0` gives an S-curve rising with
start fitness (the GOF/maximization shape); the LOF/minimization panels get the
mirror behaviour.

**How the parameters are estimated (updated 2026-07-29).** The two parameters are
fit by **nonlinear least squares on the original (0, 1) scale**
(`scipy.optimize.curve_fit` against the sigmoid directly), i.e. minimising
`Σ(y - ŷ)²`.

This replaces the original implementation, which transformed the responses to
logit space and fit a straight line there with `numpy.polyfit`. That approach was
a *different estimator*, not a computational shortcut to the same one, and it had
a criterion mismatch: it minimised squared error in **logit** space while the
returned R² was measured in **original** space, so the parameters it returned
were not the ones maximising the number reported next to them. The weighting is
badly skewed too — a 0.98 → 0.99 shift is a logit change of ≈0.7 while a
0.50 → 0.51 shift is only ≈0.04, so near-boundary points had roughly 17× the
influence of mid-range ones on the fitted line. Fitting on the original scale
makes the optimised objective and the reported R² the same quantity.

**Clipping is now confined to the initial guess.** Because residuals are
evaluated *through* the sigmoid, a response at exactly 0 or 1 is an ordinary data
point (the curve approaches it asymptotically, residual stays finite) and needs
no clipping — which also removes it as an extreme leverage point. The clip
survives only to make the optimiser's starting point computable: `p0` comes from
the cheap closed-form logit-space `polyfit`, whose transform is infinite at the
endpoints. The parameter is therefore named **`initial_guess_epsilon`** (default
`1e-6`) rather than `clip_epsilon`, and it provably **does not affect the fitted
parameters** — only where the search starts (unit-tested by fitting the same data
with `1e-2` and `1e-9` and asserting identical results).

**Convergence fallback.** `curve_fit` is iterative and can fail. On
`RuntimeError` the function falls back to the closed-form logit-space estimate
(the old behaviour), so it always returns usable parameters — but it emits a
`RuntimeWarning` saying so, and noting that the returned R² is then no longer the
optimised criterion. This is deliberately a warning rather than a silent fallback
or a `print`, so a degraded fit cannot slip into a published figure unnoticed and
can be escalated to an error with `-W error::RuntimeWarning` if desired. With a
good `p0` on a well-behaved 2-parameter sigmoid this path is not expected in
practice. The
returned parameter covariance is discarded, so the accompanying `OptimizeWarning`
about unestimable covariance (raised on perfect/degenerate fits) is suppressed.

**Reported fit quality.** `fit_logit_linear` returns `(slope, intercept,
r_squared)`, where R² is `1 - SS_res/SS_tot` between the observed `y` and the
fitted `ŷ`, on the original (0, 1) scale.

**Measured on the paper data (2026-08-03)**, using the same point selection as the
panels (genes carrying `num_mutations_half_max_effect`), under the current
original-scale estimator. No run hit the convergence fallback.

| panel | run | n | slope | intercept | R² | final-fitness SD |
| --- | --- | --- | --- | --- | --- | --- |
| A | GOF (unconstrained) | 105 | 4.53 | 8.77 | 0.063 | 1.4e-04 |
| C | GOF (natural) | 104 | 8.20 | −1.46 | 0.857 | 2.19e-01 |
| E | LOF (unconstrained) | 68 | 1.03 | −10.92 | 0.027 | 5.9e-05 |
| G | LOF (natural) | 68 | 6.51 | −5.98 | 0.470 | 3.24e-01 |

This supersedes the previously recorded GOF-natural ≈ 0.85 / LOF-natural ≈ 0.46,
which had been measured under the old logit-space estimator. Notably the two
constrained R² values barely moved (0.8567 and 0.4696): the estimator change was
made because the optimised criterion and the reported number must be the same
quantity, **not** because it improved the fit, and on this data it does not.

⚠️ **The two unconstrained panels' R² is not interpretable as fit quality.** Those
runs saturate almost regardless of start fitness (final-fitness SD ≈ 1e-4), so
`SS_tot` is negligible and R² is a ratio of two near-zero quantities — 0.063 and
0.027 mean "there is essentially no variance to explain", not "the sigmoid fits
badly". The fit still renders as the intended near-flat line at ≈1 / ≈0, faithfully
showing that the unconstrained runs hit the fitness ceiling/floor within the
mutation budget. Only the constrained panels' R² should be quoted in the caption.
Related: those panels' slope/intercept are correspondingly weakly determined (an
almost-flat response pins down little), so they should not be interpreted either.

**Caveats (honest limitations).**

- This is **least squares**, *not* a maximum-likelihood model for a bounded
  response. It assumes roughly homoscedastic Gaussian noise in `y`, which cannot
  hold exactly for a (0, 1)-bounded response (variance must shrink toward the
  boundaries). It remains a *descriptive trend line* for the figure, and R² a
  descriptive goodness-of-fit. If inference on the slope is ever needed (p-values,
  CIs), **beta regression** (`statsmodels.BetaModel` with a logit link) is the
  principled tool — it models that heteroscedasticity structurally. Not adopted
  here because these are deterministic model outputs rather than noisy
  proportions, and because it yields only a pseudo-R², not comparable to the R²
  the other panels report.
- **Only the response is transformed, not the predictor.** Start fitness is also
  bounded in (0, 1), but boundedness is a *prediction* constraint, not a data
  constraint: regression conditions on `x` and never predicts it, so no
  out-of-range `x` can be produced. A logit-logit form
  (`logit(final) = slope * logit(start) + intercept`, a log-odds power law) would
  be *more interpretable* — `slope = 1, intercept = 0` is exactly "no change",
  `slope < 1` means weak starters gain proportionally more — but that is a
  question of functional form, deliberately kept separate from the estimator fix
  above and not adopted.
- The curve is drawn only across the observed range of start fitness
  (`min`…`max`), not extrapolated, so the panel shows the trend where there is
  data.

#### Implementation & tests

`fit_logit_linear` lives in `simple_result_stats.py` (reusable, unit-tested);
the scatter function gains `mean_start_line`, `fit_line`,
`mean_start_line_color`, `fit_line_color`. `compose_plots_mpl.py`'s
`_populate_fig4` passes them for all four panels
(`_MEAN_START_LINE_COLOR = "0.5"`, `_FIT_LINE_COLOR = "black"`).

Tests (`test_simple_result_stats.py`): `fit_logit_linear` recovers known sigmoid
parameters from noiseless data (R² = 1), stays within (0, 1) even when
extrapolated, keeps exact-0/1 responses finite, drops non-finite input points,
and rejects fewer than two points. Three tests cover the 2026-07-29 estimator
change specifically: `initial_guess_epsilon` of `1e-2` vs `1e-9` yields identical
parameters (proving the clip no longer touches the fit), the returned parameters
give a strictly lower original-scale RSS than the logit-space `polyfit` they
replace, and a patched `curve_fit` raising `RuntimeError` falls back to the
closed-form estimate *and* emits the `RuntimeWarning` (asserted via
`assertWarns`). The scatter function draws the mean line at the correct x,
draws a bounded fit curve, and draws neither overlay by default.

### 3c. Figure 4 bottom row — region panels I and J

**Date added:** 2026-07-29

Figure 4 splits top/bottom first (`outer_grid`, 2×1, height ratios 4.0 : 2.4) and
gives each half its own sub-grid, so the two halves have independent column
geometry and spacing (see §3e). `run_grid` is 4×3 (scatter | colorbar strip |
histogram) for A–H; `bottom_grid` is 2×2: panels I and J side by side in row 0
(the taller row, whose extra height carries the legends drawn below those axes),
panel K spanning both columns in row 1 (see §3d). Figure height 280 mm.

- **Panel I** — GOF possible-mutation region breakdown, constrained (VCF) vs
  unconstrained, per genomic region, log y-axis
  (`region_mutation_breakdown.plot_region_breakdown`). LOF is passed as empty
  frames because `plot_region_breakdown` keeps only `group == "GOF"` rows.
- **Panel J** — share of the mutations the unconstrained runs *actually*
  introduced that sit on natural-variation (VCF) positions, per region, GOF vs
  LOF (`actual_mutation_region_breakdown.plot_pooled_allowance_bars`).

Shared bottom-row styling:

- Colours sampled from the same **magma** map as the scatter colorbar, so the
  bottom row reads as part of the figure. Panel I takes the two far-apart samples
  (`#3b0f70` constrained / `#f9795d` unconstrained) because its bars are read
  against each other; panel J takes the intermediate pair (`#ca3e72` GOF /
  `#fec68a` LOF) so the two hue meanings stay distinguishable side by side.
- Legends are moved out of the axes by `_move_legend_below`: frameless, untitled,
  one row, anchored at `(0.5, -0.24)` just under the x-label. Inside-the-axes
  legends overlapped the bars at panel size, and the entry labels are
  self-explanatory without a title.
- Panel I's y-label drops "(log scale)" and panel J's is shortened to
  "Mutations at natural positions (%)" — both overridden in the composition only,
  so the standalone plots keep their fuller labels.

#### Why panel J is pooled, not a per-gene mean

The first version plotted the **mean over genes of each gene's
`percent_allowed`**, with SD whiskers. Two problems, both visible in the render:

1. **Negative whiskers.** Per-gene percentages are strongly right-skewed — in the
   GOF terminator the median gene has 2 mutations in that region, so its only
   possible values are 0, 50, 100 %, and 73 % of genes sit at exactly 0. Mean ± SD
   over that shape (mean 9.9, SD 22.9) reaches well below zero, which is not a
   possible percentage.
2. **Upward bias.** The unweighted mean lets a gene with 1-of-1 allowed count as
   100 %, inflating the GOF promoter to 7.1 % against a pooled 4.1 %.

Panel J therefore plots the **pooled percentage** — `Σ allowed / Σ total × 100`
over the genes of that cell, the statistic `build_summary_dataframe` already
recommended for exactly this reason.

#### Why a cluster bootstrap and not a Wilson interval

A pooled percentage is a ratio of sums, so there is no per-observation spread to
take an SD of. The obvious interval is binomial (Wilson), but that assumes the
numerator is a sum of **independent** Bernoulli trials, and mutations are
**clustered within genes**: positions in one gene share its VCF density, region
lengths and optimization trajectory. The effective sample size is the number of
genes (30–104), not the number of mutations (thousands), so a Wilson interval on
the mutation count comes out **too narrow** (anti-conservative).

`pooled_percent_with_bootstrap_ci` instead resamples **whole genes** with
replacement (percentile cluster bootstrap, `B = 10 000`, fixed seed), recomputing
the full ratio for each resample. Within-gene dependence is preserved by never
splitting a gene; every resampled value is itself a valid percentage, so the
interval cannot leave [0, 100] — the negative-whisker problem is fixed
structurally, not by clipping the axis. Genes with zero mutations are kept in the
resampling pool (they contribute nothing to either sum but are part of the gene
sample); resamples whose denominator comes out 0 are discarded before the
percentiles are taken. Considered and not adopted: BCa (better coverage, ~3× the
code, invisible at this bar size), beta-binomial/GLMM and cluster-robust SEs
(more machinery than a bar chart needs; the route to take if a significance test
is ever wanted).

#### Values on the published panel

| group | region | genes w/ mutations | mutations | pooled % | 95 % CI |
|---|---|---|---|---|---|
| GOF | promoter | 96 / 104 | 587 | 4.09 | 2.46–5.84 |
| GOF | 5'-UTR | 104 / 104 | 5507 | 3.72 | 3.08–4.38 |
| GOF | 3'-UTR | 104 / 104 | 1126 | 3.46 | 2.45–4.55 |
| GOF | terminator | 30 / 104 | 90 | 8.89 | 4.55–15.00 |
| LOF | promoter | 68 / 68 | 1063 | 4.80 | 3.12–6.64 |
| LOF | 5'-UTR | 68 / 68 | 1739 | 4.66 | 3.77–5.59 |
| LOF | 3'-UTR | 68 / 68 | 1852 | 4.05 | 3.00–5.15 |
| LOF | terminator | 68 / 68 | 1466 | 6.07 | 4.29–8.21 |

Two caveats for the caption. The tall GOF terminator bar rests on 30 genes and 90
mutations and has by far the widest CI — the bootstrap is reporting honestly that
this cell is weakly determined. And **all eight intervals overlap**, so the panel
supports "roughly 3–9 % of introduced mutations coincide with natural variation,
everywhere" and *not* any claim that a region or gene set differs from another; a
gene-level permutation test would be needed for that. Pooling also weights genes
by mutation count, so the per-gene spread is not visible in the panel.

#### Implementation & tests

New in `actual_mutation_region_breakdown.py`:
`pooled_percent_with_bootstrap_ci` (estimator + interval),
`build_pooled_ci_dataframe` (one row per group×region with `n_genes`,
`n_genes_total`, mutation totals, pooled percent and CI bounds; per-cell seed
offset `seed + cell_index`), and `plot_pooled_allowance_bars` (dodged `ax.bar`
with asymmetric `yerr`, `ylim(bottom=0)`, same `palette`/`ax`/`show_legend`/
`show_title` contract as the sibling plotters). `main()` additionally writes
`actual_mutation_allowance_pooled_ci.csv`. The interim mean±SD bar function was
removed; the original per-gene boxplot is unchanged.

In `compose_plots_mpl.py`: `_actual_mutation_allowance_dataframe()` (scans both
unconstrained runs against their VCF dirs) and `_move_legend_below(ax)`.

Tests: `test_actual_mutation_region_breakdown.py` covers the estimator's
ratio-of-sums behaviour, interval bracketing and [0, 100] bounds, zero-variation
and all-zero/empty cells, seed reproducibility, both validation errors, the
per-cell frame's gene counts and totals, and the plot's bar heights, tick order,
zero-based y-axis and ax-injection. `test_compose_plots_mpl.py` covers
`_actual_mutation_allowance_dataframe`'s run/VCF pairing and
`_move_legend_below`'s untitled, frameless, below-axes legend.

### 3d. Figure 4 panel K — example-gene deepCIS binding track

**Date added:** 2026-07-29

One full-width panel (`bottom_grid[1, :]`) showing what an optimization does to
a single gene's predicted TF binding: the deepCIS window scan of
`AT3G60640` (run dir `3_AT3G60640_gene:22415750-22417548_260312_000750_357934` in
the constrained GOF run `GOF_single_natural_260311_122228_299093`), reference vs
its **5-mutation** Pareto-front member, for TF family `LOBAS2_tnt`.

Data source — the precomputed scan CSV, not a re-run of the model:

```
.../deepdive_simons_gene/deepcis_scan/deepcis_window_scan_deepdive_simons_gene_mut5.csv
```

produced by

```
analysis_scripts/run_deepcis_peak_pipeline.sh \
  <GOF_single_natural_260311_122228_299093 run dir> deepdive_simons_gene \
  <analysis out dir> --genes AT3G60640 --mutation-count 5
```

Design decisions:

- **No peak highlighting.** The panel's claim is where the *introduced mutations*
  sit relative to the binding signal, not which windows were peak-called, so
  `highlight_peaks=False`.
- **No difference curve** (`show_difference=False`). It would force the y-range to
  (-1, 1) and halve the vertical resolution of a short full-width panel; the two
  curves already show the change.
- **Mutation positions are read from the same front member that was scanned**, not
  hardcoded: `_fig4_introduced_mutation_positions()` loads the final Pareto front
  via `MutationsGene`, picks the 5-mutation member with
  `mutation_locations.find_front_member`, and shifts each 0-based
  `MutatedSequence` position by +1 into the 1-based frame the scan's window
  centers use (`deepcis_visualize.get_centers`). The panel therefore cannot drift
  from the sequence it plots. Positions (1-based): 804, 937, 957, 1149, 1194 —
  three in the promoter, two in the 5'-UTR.
- **Line restyling** (`_restyle_fig4_deepcis_lines`). Both sequences predict an
  identical signal over most of the window and the optimized curve is drawn last,
  so at the standalone weight (solid, lw 2) the reference was invisible outside the
  few windows the mutations touch — it read as "reference is zero in the 3'-UTR".
  The reference is thinned to 1.2 and the optimized to 1.0 **dashed**, so both are
  visible where they coincide.
- **Legend below the axes**, via `_legend_below_keeping_entries` rather than
  `_move_legend_below`: the vertical markers (TSS/TTS, introduced mutations) exist
  only as proxy handles, which a fresh `ax.legend()` would drop. Handles of curves
  still on the axes are re-taken from the live artists, so the restyled optimized
  curve gets a dashed swatch. Grid off, matching every other panel.
- **Panel letter above the left spine** (`x=0.0, y=1.04`) instead of in the left
  margin like every other panel: at full width the margin holds only the tall
  y-label, and a letter placed beside it collided with it.

New in `deepcis_visualize.py` (backward compatible): `_plot_gene_tf` gains
`mutation_positions` / `mutation_marker_label` (default None → nothing drawn, no
legend entry), drawn as thin `MUTATION_MARKER_COLOR` (`#d62728`) axvlines behind
the curves via the existing `_add_vertical_markers`, with a matching legend proxy
threaded through `_set_axis_properties(mutation_marker_label=...)`.

Tests: `test_deepcis_visualize.py` covers one marker per position with its legend
entry, and no markers / no legend entry for the default and for an empty list.
`test_compose_plots_mpl.py` covers the restyle, the proxy-preserving legend move,
the 0→1-based position shift, and the track drawing both curves with markers,
`ylim (0, 1)` and a `ValueError` when the gene is absent from the scan CSV.

### 3e. Figure 4 geometry — nested grids to recover horizontal space

**Date added:** 2026-07-29

The first version used one flat 6×3 gridspec for the whole figure. Two sources of
wasted horizontal space:

1. **The colorbar column bled into the bottom row.** Panels I and J sat in columns
   0 and 2 of the *same* grid as the run rows, so the thin colorbar strip and both
   its gaps ate into their width even though nothing in the bottom row uses a
   colorbar.
2. **Symmetric, oversized column gaps.** With the default `wspace`, the gap
   between a scatter and the colorbar was as wide as the gap between the colorbar
   and the histogram — but the scatter side has no decorations to accommodate,
   only the colorbar label and the histogram y-label do.

Fix: split top/bottom first (`outer_grid`, 2×1) and give each half its own
sub-grid via `SubplotSpec.subgridspec`, then set `wspace=0.03` on both. Under
`layout="constrained"` `wspace` is *extra* padding on top of the space reserved
for tick and axis labels, so a small value collapses the empty
scatter-to-colorbar gap while the colorbar-label / y-label gap stays as wide as
its decorations need. The bottom half, free of the colorbar column, gives I and J
a true half-width each.

The run-row identifiers ("GOF", "LOF (natural)", …) moved from
`text(-0.42, 0.5, transform=transAxes)` to an `annotate` with a **point-based offset**
(`_FIG4_ROW_LABEL_OFFSET_POINTS = 42`) from the left spine: an axes-fraction
offset scales with panel width, so it drifted when the panels got wider.

### 3f. Figure 4 panels B/D/F/H — cropped x, broken y

**Date added:** 2026-07-29

Both axes of the four half-max histograms were dominated by a handful of genes,
leaving the panels nearly empty. The distribution of `num_mutations_half_max_effect`:

| panel | n genes | max | genes > 30 | tallest bar |
|---|---|---|---|---|
| B GOF | 105 | 18 | 0 | 60 |
| D GOF (natural) | 104 | 48 | 1 | 27 |
| F LOF | 68 | 13 | 0 | 12 |
| H LOF (natural) | 68 | 37 | 3 | 6 |

**x — overflow bin at 30** (`hist_half_max_mutations(max_mutations_shown=30)`, a
new backward-compatible param; `None` keeps the full data range). Only 4 of 345
genes exceed 30 mutations, but they stretched the axis to 50. They are **pooled
into a final bin tick-labelled `≥30`** rather than dropped, so the truncation is
visible in the panel itself and no gene silently disappears — the pooled bin is
plainly there in D and H.

**y — broken axis** (`style.broken_y_axes`). The four panels must share one count
axis to be comparable, and panel B's spike of 60 genes at 2 mutations set that
axis alone: the tallest bar anywhere else is 27. The histogram cell is therefore
split into a pair of stacked, x-sharing axes — lower `(0, 29)`, upper `(55, 63)`,
height ratio 0.7 : 4 — and the *same* histogram is drawn on both, each showing
only its slice of the range. **No bar in any panel has a count between 28 and 54**,
so the gap hides no data. The upper strip keeps its `60` tick in all four panels
(that is what makes them comparable) and drops its axis labels; the panel letter
sits on the upper strip.

Panel B's spike is the one bar that actually crosses the cut, so the gap between
the axes splits it in two. Bridging the gap with a patch in the bar's colour (so
the bar reads as continuous) was tried and **rejected**: the gap belongs to the
axis break, and a bar drawn straight through it obscures that the axis was cut at
all. The bar stays split.

The *width* of that gap was still too large to read the two halves as one value.
`hspace=0` on the sub-gridspec (`_FIG4_HIST_BREAK_HSPACE`) was already set, but
under `layout="constrained"` gridspec `hspace` is only extra space on top of the
layout engine's own `h_pad` (default ≈0.042 in on each axes, so ≈2 mm of
whitespace across the cut) — plus the space reserved for the slanted marks
themselves. Two fixes, both needed:

- `broken_y_axes` calls `set_in_layout(False)` on the break marks: they straddle
  the cut deliberately and must not be padded around.
- `_populate_fig4` shrinks the layout engine's `h_pad` to
  `_FIG4_LAYOUT_H_PAD_INCHES` (0.01 in). `h_pad` is figure-wide, so this tightens
  every row slightly; the run rows keep their separation from their tick labels
  and axis labels, so only the break visibly changes. A per-cell pad is not
  available — `SubFigure` in matplotlib 3.7 has no own layout engine.

`broken_y_axes` is generic and lives in `style.py`:

- Validates that the two ranges are disjoint and ordered — overlapping ranges
  would draw the same bars twice.
- Sets `set_autoscaley_on(False)` on both axes, or the `ax.hist` call that follows
  would autoscale the limits back to the full range and undo the break.
- Draws the slanted break marks **only where a vertical spine is actually
  visible**, so they do not float in mid-air on the right, where the publication
  stylesheet hides the spine.

Consequence for the composition: `sync_axis_limits(hist_axes)` now syncs **x
only** — y is set explicitly by the break, and x is identical by construction
anyway (same overflow bin).

Why not the alternatives considered: a **log y-axis** keeps everything on one
axes but makes bar heights non-proportional; **per-objective y sharing** would
give F/H a tighter range but breaks GOF-vs-LOF comparison; **percent of genes**
fixes the unequal-n comparability problem (105 vs 68) but not the emptiness
(B 57 % vs H 9 %). Note the unequal-n caveat still stands for the raw counts as
plotted.

### 3g. Figure 3 — mutation signatures + the significant-TF contrast

**Date added:** 2026-08-03 (documenting code committed 2026-07-2x, `9b50961`)

`fig3()` / `_populate_fig3` compose seven panels at double-column width, 250 mm
high, from the two **maximization** single-mutation runs (`ara_msr_max_single`,
`zea_msr_max_single`, both read from their `all_mutated_sequences_*_gen1999.json`)
plus a four-run TF contrast.

Layout is **one flat 5×4 grid** (deliberately not the nested-halves construction
figure 4 needed, §3e — here no column strip has to be kept out of another row):

| row | contents |
|---|---|
| 0 | A (ara) and B (zea) rolling-mean net nucleotide change, two columns each |
| 1 | thin (0.2) full-width strip holding the shared A/C/G/T legend |
| 2 | C (ara) and D (zea) total net nucleotide change bars, one column each |
| 3 | E — ara mutation-distance difference, left two columns |
| 4 | F — zea mutation-distance difference, left two columns |
| 2–4 | G — significant-TF heatmap, right two columns, with a `width_ratios=[1, 0.045]` sub-grid splitting off its colour-bar column |

Design decisions:

- **One shared A/C/G/T legend** in the row-1 strip; the per-panel legends that
  `make_line_plot_rolling_window` and `plot_net_nucleotide_change` create are
  removed. A/B and C/D use the same nucleotide colours (`COLORS`), so one key
  serves four panels. `plot_sum=False` on the line panels — the per-nucleotide
  traces are the signal, the sum only compresses them.
- **Colour is reserved for the heatmap.** The distance panels' bars are repainted a
  uniform neutral grey (`_color_bars_neutral`, `_NEUTRAL_COLOR` at
  `_BAR_ALPHA`) and their legend dropped entirely: the sign of the
  real-minus-random difference is already read off the zero line, so the original
  colour carried no information. The net-change bars (C/D) keep their A/C/G/T
  colours because there the colour *is* the category.
- **Shared limits within a panel type** via `sync_axis_limits`: y for A/B and for
  C/D (x is position/nucleotide, identical by construction), both x and y for E/F.
  Duplicated axis labels are dropped — y-label on A and C only, x-label on F only.
- **Species titles in italics** (`fontstyle="italic"`) since they are binomials.
- **Distance panels cropped to `_DISTANCE_MAX = 30`**: the real-vs-random
  difference is concentrated at short inter-mutation distances and the long tail is
  flat. The random baseline is `_DISTANCE_REPLICATES_PER_GENE = 10` replicates per
  gene under a fixed seed (`_DISTANCE_SEED = 42`), built in memory by
  `_distance_difference_proportions` from a `MutationPool.from_summarized_json`, so
  no cached pool file is required.

#### Panel G — which TFs, in which order, with which stars

`_draw_significant_tf_heatmap` mirrors `minmax_comparison.py` so the panel matches
that standalone figure's data, ordering and stars:

- **Rows:** the per-gene-normalized four-run diff matrix
  (`build_matrix(..., normalization="per_gene")`) restricted to the TFs whose
  **paired ara max-vs-min contrast** reaches the *** tier
  (`q_contrast < _THREE_STAR_ALPHA = 0.001`). Because every surviving TF is *** by
  construction, the redundant row-label stars are dropped — the selection *is* the
  claim, and repeating it per row would suggest a per-row test result.
- **Columns:** `ara max`, `GOF` (left, maximization) then `ara min`, `LOF`
  (right, minimization), ordered by `order_tfs_by_group_contrast` so max-favoured
  TFs sit above min-favoured ones.
- **Cell stars are intra-run, not the selection test.** The ara pair reuses the
  `q_a`/`q_b` columns from the paired analysis; GOF and LOF have no paired partner
  and each get an independent `single_run_tf_significance` test
  (`_significant_tf_cell_stars`). Rendering fix: seaborn's `va="center"` leaves
  asterisk glyphs sitting high in their text box, so each annotation is nudged down
  by `_STAR_VERTICAL_NUDGE` (0.12 cell heights).
- **Two soft grey dividers** (`0.45`, lw 1.0 — the default black/2.0 reads harsh):
  vertical between the max and min run groups, horizontal where the ordering score
  (max-group mean minus min-group mean) changes sign. The horizontal one is drawn
  only when both blocks are non-empty.
- Every TF label is forced (`set_yticks`/`set_yticklabels`); seaborn's "auto"
  locator would otherwise label every other row in this short axes.
- **Panel letter re-anchored in figure coordinates.** G's long TF row labels push
  its axes far right, so an axes-fraction letter landed right of B's. After a
  `fig.canvas.draw()` (needed for the constrained layout to resolve positions) the
  letter is moved to B's letter x at G's own top edge, so the right-column letters
  line up.

**Caveat for the caption:** panel G shows only *** TFs from the **ara** max-vs-min
contrast; GOF/LOF columns are displayed for those same TFs but did not take part in
the selection, so the panel is not a symmetric four-run screen.

**Stale code comments (not fixed here):** the constant block above `_ARA_MAX_DIR`
and the docstring of `_significant_tf_cell_stars` still call the heatmap "panel E"
from an earlier layout; it is panel G.

### 3h. Figure 6 — STARR-seq × deepCRE positional correlation

**Date added:** 2026-08-03 (documenting code committed 2026-07-2x, `40b3959` /
`f01c417` / `4e0a253`)

`fig6()` / `_populate_fig6` compose four panels at double-column width, 170 mm
high, over the STARR-seq overlap analysis:

- **A** — Spearman correlation between deepCRE prediction and STARR-seq enrichment
  as a function of overlap start position; **B** — the p-value of that correlation
  on a log y-axis. Both over the **pooled WRKY + bHLH** dataset, overlaying the
  `all` / `light` / `dark` STARR-seq conditions (`_fig6_position_series`,
  `POSITION_SERIES_COLORS`).
- **C** — the **WRKY-only** point-level scatter with the peak-correlation window
  highlighted on top of the full dataset and coloured by binding status.
- **D** — a boxplot of the enrichment of *exactly* those highlight-window points,
  binding vs non-binding.

Grid is 4×4: row 0 A|B (two columns each), row 1 a thin legend strip (0.15), row 2
C (three columns) | D (one column), row 3 a second legend strip (0.22).
`_FIG6_WIDTH_RATIOS = [1, 1, 0.6, 1.4]` — columns 0+1 and 2+3 sum equal so A and B
stay the same width, while the last column is widened (third narrowed to
compensate) so single-column panel D's two x-tick labels no longer touch.

Design decisions:

- **The pool is built per TF, then concatenated.** `_build_fig6_dataframes` runs
  `prepare_wrky_enrichment_df()` and `prepare_bhlh_enrichment_df()` independently
  (own reference windows and mapping), so genes shared between the two TF sets are
  never double-counted, then row-concatenates for A/B. The WRKY frame is reused for
  C/D rather than recomputed. This runs the deepCRE prediction pipelines and is the
  slow part of the figure — there is **no cached point-level dataframe**, so fig6
  is rebuilt from scratch on every call.
- **Two legend strips, not in-panel legends.** All four auto-created legends are
  removed; the condition key goes under A/B and the binding-status + fit-line key
  under C/D. The bottom key is assembled by hand — the overlay plot draws its two
  fit lines *unlabelled*, so their handles are appended with `_FIT_ALL_LABEL` /
  `_FIT_HIGHLIGHT_LABEL`, and `markerscale=2.2` compensates for the shrunken
  publication marker sizes.
- **Publication marker/line weights.** The standalone plots use `s=30`/`lw=3` and
  35/55, far too heavy at panel size: `_POSITION_SCATTER_SIZE = 4.0` with
  `alpha 0.12`, `_POSITION_LINE_WIDTH = 1.4`, overlay markers 6.0 (background) /
  12.0 (highlight).
- **The highlight-window fit line is recoloured black and pushed to
  `zorder = 0.5`** (behind every scatter point, which sit at `zorder ≥ 1`). The
  standalone dark red both clashed with the binding-status colours and occluded the
  data; recolouring happens *before* the legend handles are read so the proxy
  matches.
- **A dashed vertical connector at `HIGHLIGHT_WINDOW_CENTER`** on A and B ties the
  top row to the window C/D zoom into. Kept black rather than the panel-C highlight
  colour so the figure is not overloaded with hues.
- **Panels C and D share `BINDING_STATUS_COLORS`** and, more importantly, share the
  *point set*: `plot_overlay_highlight_correlation` returns the deduplicated
  highlight points and those are handed straight to `_draw_binding_status_boxplot`,
  so D cannot drift from C. `saturation=1.0` keeps the box fills identical to C's
  point colours (seaborn otherwise desaturates to 0.75). Absent groups are dropped
  from the order rather than drawn empty.
- Standalone titles are stripped from A/B/C; axis labels and panel letters carry
  the meaning in a multi-panel figure. A/B share the overlap-position x-axis via
  `sync_axis_limits(..., sync_y=False)` (y is a correlation vs. a log p-value).

#### Panel D's significance annotation

`_annotate_binding_status_significance` runs a **two-sided Mann-Whitney U** on the
binding vs non-binding `enrichment` of the highlight-window points, draws a
significance bar labelled by `q_to_stars(p)` (or `ns`), and prints U, p, the
rank-biserial effect size and both group sizes to the console so the numbers are
available for the caption. Mann-Whitney rather than a t-test to match
`binding_vs_nonbinding.py` and to assume nothing about the normality of enrichment
values. The test is skipped (with a printed note) when either group is empty.

⚠️ Two honesty notes: the label is a raw p-value passed through a function named
`q_to_stars`, so the stars are **not** multiplicity-corrected — a single planned
comparison, but the caption should say "uncorrected". And `n=` labels sit under each
box, so the reader can see the group sizes the test rests on. The bar is drawn
*before* the sample-size labels because it expands the y-limits.

### 3i. Figure 1 — graphical-abstract miniatures

**Date added:** 2026-08-03 (documenting `fig1_miniatures.py`, commits `5b51eda` /
`49a6255`)

Figure 1 is a graphical abstract, designed separately in
**`FIG1_GRAPHICAL_ABSTRACT_PLAN.md`** (scientific briefing + implementation plan) —
that document, not this one, is the source of truth for its content.

`fig1_miniatures.py` renders the small inset illustrations it needs as **standalone
SVGs** under `figures/mpl_compositions/fig1_miniatures/`, rather than as panels of
one composed figure: each is placed by hand in the abstract's layout. Every
miniature reuses the exact data source and plotting building block of the full
figure it is a reduction of, so the insets cannot disagree with the panels they
advertise:

- `pareto_front_mini` — the example gene's final Pareto front
  (`example_gene_pareto.load_final_front`), single `Purples` shade at 0.85.
- `deepcis_scan_mini` — the same scan CSV, gene and TF as figure 4 panel K
  (importing the `_FIG4_DEEPCIS_*` values' twins), with the mutation and TSS/TTS
  markers off; keeps a Reference/Optimized legend since the two curves are
  otherwise indistinguishable, with `_DEEPCIS_MINI_Y_MAX = 1.3` giving it headroom.
- `fig6c_mini` — figure 6 panel C's highlight-window points only, without the
  full-data background or the binding-status split; recoloured to a `magma` shade
  (0.2) rather than the binding highlight colour, which would imply a split that
  isn't shown.
- `fig3g_mini` — figure 3 panel G's heatmap reduced to the top/bottom
  `_TF_HEATMAP_MINI_N_EXTREME = 3` TFs by group contrast, stars removed, `_tnt`
  suffix stripped from TF names (a model-naming artefact, not part of the family
  name), colour bar relabelled "introduction frequency". It imports
  `_ARA_MAX_DIR`, `_FIG3_TF_*` and `_THREE_STAR_ALPHA` from `compose_plots_mpl.py`
  so the selection rule stays shared with the full panel.

Panels are 55 × 36 mm (`_MINI_SIZE_MM`; the heatmap 55 × 40 mm) and saved with
`bbox_inches="tight"`, SVG only — they are placed as vector insets, so no raster
fallback is wanted. Tests: `test_fig1_miniatures.py` (6 tests) covers each
miniature's data restriction, legend presence/absence, marker suppression, colour
shade, label rewriting, and the missing-gene error.

### 4. `figure_composition.py`

Unchanged in behavior; documented as a fallback for frozen raster panels only.

## Data flow

```text
publication figure script
  → with publication_style():
      fig = plt.figure(figsize=mm_to_inch(...))
      gs  = fig.add_gridspec(...)
      plot_a(..., ax=fig.add_subplot(gs[0, 0]))   # standardized analysis func
      plot_b(..., ax=fig.add_subplot(gs[0, 1]))   # specialized workflow func
      # inline publication-only annotations
      panel_label(ax_a, "A"); panel_label(ax_b, "B")
      save_publication_figure(fig, "figures/figN.pdf")
```

## Testing

- `style.py` helpers are unit-testable: `publication_style()` applies and restores
  rcParams; `mm_to_inch()` conversion; `panel_label()` adds the expected text;
  `save_publication_figure()` writes vector output (mock `savefig`).
- Refactored plot functions: extend/add tests asserting that passing an `ax` draws
  onto it and does **not** call `savefig` (mock it), and that `ax=None` preserves
  current create-and-save behavior.
- Per-figure composition scripts are one-off and exercised via their helper calls;
  the helpers that carry real logic (data assembly, position shifts, statistics,
  legend surgery) are unit-tested with synthetic frames and mocked pipelines, while
  the `_populate_figN` layout functions themselves are not (see the test-gap note
  under "Implementation status").

## Resolved decisions

- Stylesheet defaults (generic vector, ≈180 mm double-column, ~7–8 pt) kept as-is
  for now; user may tune later.
- `figure_composition.py` is demoted, **not deleted** (kept for possible later use).
- Scope: all of `src/analysis/` then all of `src/workflows/` except `hoffie/`.

## Implementation status

### Ax-injection refactor (2026-07-14) — COMPLETE

Refactor complete; full test suite green at that point (834 passed, hoffie
excluded). All edited files lint-clean. (2026-07-13 checkpoint: interrupted at 819
by an account session limit; resumed and finished 2026-07-14.)

The suite has grown with the figure work since: **985 tests collected**
repo-wide as of 2026-08-03, of which `test/workflows/paper_plots/` contributes 35
(`test_style.py` 16, `test_compose_plots_mpl.py` 13, `test_fig1_miniatures.py` 6) —
all passing. The 834 figure above is a historical checkpoint, not the current count.

### Composition scripts

| figure | entry point | status |
| --- | --- | --- |
| 1 (graphical abstract) | `fig1_miniatures.py` (insets only) | insets implemented; composition per `FIG1_GRAPHICAL_ABSTRACT_PLAN.md` |
| 2 | `compose_plots_mpl.fig2` | implemented (§3) |
| 3 | `compose_plots_mpl.fig3` | implemented (§3g) |
| 4 | `compose_plots_mpl.fig4` | implemented (§3b–3f) |
| 6 | `compose_plots_mpl.fig6` | implemented (§3h) |

There is no figure 5 composition in this module. `python -m
workflows.paper_plots.compose_plots_mpl` runs `fig2`, `fig3`, `fig4`, `fig6` in
order.

**Known test gaps** (flagged rather than silently accepted): `test_compose_plots_mpl.py`
covers the figure-4 helpers and `_draw_binding_status_boxplot`, but the figure-3
helpers (`_distance_difference_proportions`, `_significant_tf_cell_stars`,
`_draw_significant_tf_heatmap`, `_color_bars_neutral`) and the figure-6 helpers
`_build_fig6_dataframes`, `_fig6_position_series` and
`_annotate_binding_status_significance` have no unit tests. The last one is the
notable one — it computes a published p-value.

**Done + tested + lint-clean:**

- `style.py` foundation (+ `test_style.py`).
- `src/analysis/`: `mutations/analyze_mutations.py`, `overview/simple_result_stats.py`,
  `overview/compare_methods.py`, `overview/compare_analysis_results.py`,
  `starrseq/starrseq_analysis.py`, `motives/analyze_peaks.py`
  (`deepcis_visualize.py` already took `ax` — no change needed).
- `src/workflows/`: `candidate_selection.py`, `compare_reruns/compare_runs.py`,
  `deepCRE_TPM_correlation/correlate.py`, `mutation_distance_analysis/…`,
  `mutation_distribution_analysis/plot_optimization_vs_random.py`,
  `evo_alg_pooled_plots/tf_comparison/tf_comparison_plot.py`,
  `evo_alg_pooled_plots/natural_unconstrained_comparison/*.py`,
  `overlap_analysis/correct_construct/analyze_in_true_background.py`.

**Left as-is by design (no single reusable axes to inject):**

- `overlap_analysis/binding_vs_nonbinding/binding_vs_nonbinding.py` — both plot
  functions are multi-panel (one subplot per TF family).
- `evo_alg_pooled_plots/tf_comparison/{minmax,model,random_start}_comparison.py`
  — all plotting is inline inside a `main()` entrypoint; would need a plotting
  function extracted before an `ax` makes sense.
- `analyze_in_true_background.plot_prediction_vs_enrichment` — multi-panel grid.

**Completed in the 2026-07-14 resume:**

- `overlap_analysis/_common.py` — ax-injection on leaf drawers
  (`_plot_bucketed_correlation`, `_plot_individual_bucket_correlation`,
  `_plot_overlay_correlation`, `simply_plot_multi`) threaded through the 1:1
  public wrappers (`plot_deepcre_starrseq_correlation`,
  `plot_mutation_starrseq_correlation`, `plot_mutation_deepcre_correlation`,
  `plot_overlay_highlight_correlation`); orchestrators
  (`plot_individual_bucket_views`, `plot_correlation_over_positions_*`) left as-is.
  Tests: `test/workflows/overlap_analysis/test_common_ax_injection.py`.
- `starrseq_analysis.binding_boxplots` — was missed in the first pass (produces
  the `binding_status_*_boxplots.pdf` figures); now ax-injected + tested.
- Added `_with_ax` / contract tests for `candidate_selection.draw_line_plot`,
  `compare_runs.{plot_paired_scatter,plot_mutation_overlap}` (return-contract
  only, per that module's no-render-test convention),
  `mutation_distance_analysis.{plot_overlay,plot_difference}`.
- Created `test/workflows/deepCRE_TPM_correlation/test_correlate.py`.

**Standalone-preservation check:** the inline `fontsize`/`set_context`/`figsize`
in every refactored function are re-applied only inside the `if own_figure:`
branch (verified by code inspection), and the suite exercises the standalone
(`ax=None`) paths. This was a code-level + test check, not a literal pixel diff.

## Notes

- Git operations are handled by the user; this repo's convention is to never
  commit/add/push from the agent and to use `rm` not `git rm`.
- Generated images/CSVs are gitignored; new output globs go in `.gitignore`.
