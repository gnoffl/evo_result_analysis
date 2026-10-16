# src/workflows/paper_plots/

Publication-figure layer. `style.py` is the shared stylesheet + layout helpers (mm sizing, panel letters, axis syncing, broken y-axes, SVG+PNG saving); `compose_plots_mpl.py` (main figs 2/3/4/6) and `supplementary_plots_mpl.py` (S1–S3) each build one matplotlib figure per paper figure under `publication_style()` by calling existing plot functions with an injected `ax`, from hardcoded ARC dataset paths. Outputs go to `src/workflows/paper_plots/figures/{mpl_compositions,supplements}/*.svg` (+ `.png` twin). `compose_plots_svg.py` is the older approach (stitching pre-rendered SVGs). Design notes: `DESIGN.md`, `FIG1_GRAPHICAL_ABSTRACT_PLAN.md`.

## style.py  [library]
Single source of truth for publication typography/geometry. Imported by compose_plots_mpl, supplementary_plots_mpl, fig1_miniatures and `evo_alg_pooled_plots/example_gene_pareto/example_gene_pareto.py`.

- `SINGLE_COLUMN_MM = 85.0`, `DOUBLE_COLUMN_MM = 180.0` — journal column widths (mm).
- `PUBLICATION_RC` — rcParams dict: sans-serif (Arial/Helvetica/DejaVu Sans); font.size/axes title/label 8 pt, tick/legend labels 7 pt, figure title 9 pt; axes lw 0.6, grid lw 0.4, lines lw 1.0, markersize 3; top/right spines off; ticks out; scatter edgecolors none; savefig dpi 600, bbox tight; editable text in vector output (pdf/ps fonttype 42, svg fonttype none). No colour palette is defined here.
- `mm_to_inch(millimetres: float) -> float` — mm → inches.
- `figure_size_inches(width_mm: float, height_mm: float) -> tuple[float, float]` — mm dims → matplotlib `figsize`.
- `publication_style(extra_rc: Optional[dict] = None)` — context manager; applies matplotlib `"default"` style + `PUBLICATION_RC` (+ `extra_rc`), restores on exit. Create figures inside the `with`.
- `panel_label(ax: Axes, label: str, *, x: float = -0.08, y: float = 1.02, fontsize: Optional[float] = None, fontweight: str = "bold") -> Text` — bold panel letter in axes-fraction coords (ha=right, va=bottom).
- `save_publication_figure(fig: Figure, path: str, *, dpi: int = 600, transparent: bool = False) -> None` — saves `path` and a sibling `.png`. **Only accepts `.svg`** (raises ValueError otherwise, although the docstring/module example mention `.pdf`).
- `sync_axis_limits(axes: Sequence[Axes], *, sync_x: bool = True, sync_y: bool = True) -> None` — set common x/y limits = union of current limits; no-op for <2 axes.
- `broken_y_axes(fig: Figure, subplot_spec: SubplotSpec, lower_ylim: tuple[float, float], upper_ylim: tuple[float, float], *, height_ratios=(1.0, 4.0), hspace=0.08, break_mark_size=5.0) -> tuple[Axes, Axes]` — splits a grid cell into (upper, lower) axes with shared x, fixed (non-autoscaling) y-limits, hidden facing spines and diagonal break marks; caller draws the same data on both. Raises if ranges overlap.

## compose_plots_mpl.py  [script]
Main paper figures 2, 3, 4, 6 composed in one matplotlib figure each (double-column width) with hardcoded data paths. Run: `conda run -n deepCREshap python src/workflows/paper_plots/compose_plots_mpl.py` (edit `__main__` to choose figure). Imports plot/calc functions from `analysis.overview.simple_result_stats`, `analysis.mutations.analyze_mutations`, `analysis.motives.deepcis_visualize._plot_gene_tf`, `workflows.evo_alg_pooled_plots.{tf_comparison,example_gene_pareto,natural_unconstrained_comparison}`, `workflows.mutation_distance_analysis`, `workflows.mutation_distribution_analysis.mutation_pool`, `workflows.overlap_analysis`, and `style`.

- `OUTPUT_DIR` — `"src/workflows/paper_plots/figures/mpl_compositions"`.
- Private constants reused elsewhere (imported by fig1_miniatures): `_ARA_MAX_DIR`, `_ARA_MIN_DIR`, `_GOF_DIR`, `_LOF_DIR` (Evo_run_analysis dataset dirs for ara msr max/min single, GOF_single, LOF_single), `_FIG3_TF_RUNS` (list of (dir, label) for the 4-run TF matrix), `_FIG3_TF_LEFT_COLUMNS`/`_FIG3_TF_RIGHT_COLUMNS` (max vs min run labels), `_THREE_STAR_ALPHA = 0.001`. Also `_FIG4_DEEPCIS_*` (example gene AT3G60640 / LOBAS2_tnt scan CSV) and `_FIG4_CONDITION_COLORS` / `_FIG4_GROUP_COLORS` (magma-sampled hex colours).
- `fig2() -> None` — 2×3 panels (ara max/min single): start-vs-final fitness scatter (shared magma colorbar), average Pareto front (x cropped at 45), half-max-mutation histogram; saves `fig2_composed.svg`.
- `fig3() -> None` — net nucleotide change bars (ara, zea), rolling-mean positional net change lines, real-minus-random inter-mutation distance difference (≤30 bp), and a 4-run TF Δ-binding heatmap restricted to *** (q<0.001) ara max-vs-min TFs; saves `fig3_composed.svg`.
- `fig4() -> None` — 4 GOF/LOF run rows (scatter + broken-y half-max histogram with ≥30 overflow bin), GOF region-breakdown boxes, natural-position allowance dot+bootstrap CI panel, and full-width deepCIS reference-vs-optimized track of an example gene with introduced mutations marked; saves `fig4_composed.svg`.
- `fig6() -> None` — STARR-seq × deepCRE: Spearman r and p-value vs overlap start position (pooled WRKY+bHLH; all/light/dark), and WRKY scatter with the highlighted best window. Slow: reruns both deepCRE prediction pipelines (no cache). Saves `fig6_composed.svg`.
- Private ax-injectable helpers (not public API, but the reusable bits): `_draw_significant_tf_heatmap(ax, colorbar_ax)`, `_draw_fig4_deepcis_track(ax)`, `_color_bars_neutral(ax)` (all bars grey 0.35, alpha 0.7), `_move_legend_below(ax)` (rebuild legend as one frameless row below axes), `_legend_below_keeping_entries(ax)` (same, but keeps proxy handles), `_populate_fig{2,3,4,6}(fig, ...)` (draw a whole figure into a given Figure, used by tests/previews).

## supplementary_plots_mpl.py  [script]
Supplementary figures S1–S3, same construction as compose_plots_mpl. Run: `conda run -n deepCREshap python src/workflows/paper_plots/supplementary_plots_mpl.py` (renders all). Imports from `workflows.adversarial.{plot,reevaluate}`, `workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark_{calc,plot}`, `workflows.evo_alg_pooled_plots.tf_comparison.{mut5_comparison,tf_comparison_calc,tf_comparison_plot}`, and `style`.

- `OUTPUT_DIR` — `"src/workflows/paper_plots/figures/supplements"`.
- `SUPPLEMENTARY_FIGURES` — `[fig_s1, fig_s2, fig_s3]`.
- `fig_s1() -> None` — transposed 4-run TF Δ-binding-peaks heatmap at the 5-mutation budget, TFs with paired ara max-vs-min q<0.05, intra-run stars in cells, contrast stars on tick labels; `figS1_tf_comparison_mut5.svg`.
- `fig_s2() -> None` — 3 log-log CPU runtime-scaling panels (GA population size, GA generations, greedy max mutations), points by regime + power-law fit with R² label; `figS2_runtime_scaling.svg`.
- `fig_s3() -> None` — adversarial re-evaluation, 2 rows (GOF, LOF) × 4 panels (pooled normalized + min/median/max-gap example genes), reading CSVs from `src/workflows/adversarial/outputs/`; `figS3_adversarial_reevaluation.svg`.
- Private ax-injectable helpers: `_draw_mut5_tf_heatmap(ax, colorbar_ax)`, `_draw_runtime_scaling_panel(ax, runtimes, algorithm, swept_parameter, title)`, `_set_sweep_ticks(ax, swept_values)` (one integer-labelled tick per swept level on a log x-axis), `_label_fit_lines(ax, panel_runtimes, fits)`, `_draw_adversarial_row(axes_row, prepared, is_top_row, is_bottom_row)`, `_short_tf_name(tf_name)` (strip `_tnt` suffix).
- Magma-sampled colours used: regime/adversarial dark `#641a80`, light `#f9795d` (private `_REGIME_COLORS`, `_ADVERSARIAL_*_COLOR`).

## fig1_miniatures.py  [script]
Tiny standalone inset panels for the fig 1 graphical abstract, saved as SVG to `OUTPUT_DIR` (`.../mpl_compositions/fig1_miniatures`). Imports private constants from compose_plots_mpl and data/plot functions from `example_gene_pareto`, `overlap_analysis`, `tf_comparison`, `deepcis_visualize._plot_gene_tf`. `main()` currently renders only `pareto_front_mini` (others commented out).

- `OUTPUT_DIR` — miniature output folder (relative to repo root).
- `pareto_front_mini() -> None` — example gene's final Pareto front over rejected population candidates (mutation count vs deepCRE prediction).
- `deepcis_scan_mini() -> None` — fig 4 example-gene deepCIS reference-vs-optimized track without mutation/TSS/TTS markers.
- `fig6c_mini() -> None` — WRKY points in the best-fit overlap window + linear fit (reruns WRKY deepCRE pipeline).
- `fig3g_mini() -> None` — fig 3G heatmap reduced to top/bottom 3 TFs, ara max/min columns only, no stars.

## compose_plots_svg.py  [one-off]
Older composition of figs 2/3/4/5 by stitching pre-rendered SVG files via `workflows.figure_composition_svg.compose_figures`, hardcoded paths. Largely superseded by compose_plots_mpl. **[broken]** `fig_5_starrseq` (the one enabled in `__main__`) passes `.png` files to an SVG-only composer (`ET.parse` will fail); docstring run path `src/workflows/compose_example_figure.py` is stale.
