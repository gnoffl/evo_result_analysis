# Publication figure layer — design

**Date:** 2026-07-13
**Status:** approved, pending implementation plan

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
- **Width constants + helper** — `SINGLE_COLUMN_MM`, `DOUBLE_COLUMN_MM`,
  `mm_to_inch()`, so figure/panel sizes are declared in real physical units.
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
fitted `ŷ`, on the original (0, 1) scale. ⚠️ The previously recorded values
(GOF-natural R² ≈ 0.85, LOF-natural R² ≈ 0.46) were measured under the old
logit-space estimator and are **stale** — R² should now be equal or better, since
it is the quantity being optimised, but the numbers need re-measuring on the
paper data. The two *unconstrained* runs (GOF/LOF `single`) saturate almost
regardless of start fitness (final-fitness std ≈ 1e-4), so their fit still
renders as a near-flat line at ≈1 / ≈0 — faithfully showing that the
unconstrained runs hit the fitness ceiling/floor within the mutation budget.

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

Figure 4's gridspec is 5 rows: four run rows (A–H) plus a bottom row holding two
half-width region panels, `grid[4, 0]` and `grid[4, 2]` (the thin middle column
stays the shared colorbar strip). Figure height 235 mm.

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
- Per-figure composition scripts are one-off and exercised via their helper calls.

## Resolved decisions

- Stylesheet defaults (generic vector, ≈180 mm double-column, ~7–8 pt) kept as-is
  for now; user may tune later.
- `figure_composition.py` is demoted, **not deleted** (kept for possible later use).
- Scope: all of `src/analysis/` then all of `src/workflows/` except `hoffie/`.

## Implementation status (2026-07-14) — COMPLETE

Refactor complete; full test suite green (834 passed, hoffie excluded). All
edited files lint-clean. (2026-07-13 checkpoint: interrupted at 819 by an
account session limit; resumed and finished 2026-07-14.)

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
