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
