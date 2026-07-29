# Plan: Figure 1 — Graphical Abstract

## Context

The paper (composed in `compose_plots_mpl.py`) currently has Fig 2, 3, 4, 6 but **no
Figure 1 / graphical abstract**. The reader entry point is missing: there is no single
glanceable panel that states the whole story before the detailed figures.

The full narrative arc, established from the existing figures, is:

1. **Input** — a plant gene's ~3020 bp regulatory window (promoter / 5'-UTR / 3'-UTR /
   terminator), for *Arabidopsis* and *Zea mays*.
2. **Model** — a deepCRE deep-learning model predicts expression from sequence.
3. **Optimize** — an evolutionary algorithm mutates the sequence to **maximize (GOF, ↑)**
   or **minimize (LOF, ↓)** predicted expression; a Pareto front trades fitness against
   mutation count (Fig 2/4).
4. **Mechanism** — the optimizer systematically introduces/removes specific TF families
   (WRKY, bHLH …), with mirror-image GOF/LOF behaviour, and its edits partly coincide with
   naturally variable positions (Fig 3, natural-vs-unconstrained analysis).
5. **Validation** — model predictions agree with experimental **STARR-seq** enrichment and
   **RNA-seq TPM** (Fig 6, TPM correlation).

**Goal:** add `fig1()` — a **conceptual schematic**, **balanced across the full pipeline**,
composed **with code** under the existing `style.py` stylesheet so it matches every other
figure, mixing pre-rendered conceptual image assets (illustrations Claude/an image tool
renders) with matplotlib-drawn elements and a few small real-data insets.

## Recommended design

A **left-to-right horizontal ribbon of 4–5 conceptual stages** (landscape, reads in one
glance), with connecting arrows and a small data inset embedded at the two stages where
real data is cheap and meaningful. Balanced weighting: each stage gets comparable width.

```
 ┌ INPUT ────┐   ┌ MODEL ────┐   ┌ OPTIMIZE ────────┐   ┌ MECHANISM ─────┐   ┌ VALIDATION ──┐
 │ plant icon│──▶│ deepCRE   │──▶│ evolve loop icon  │──▶│ WRKY / bHLH    │──▶│ STARR-seq ✓  │
 │ + promoter│   │ model box │   │ GOF ↑ / LOF ↓     │   │ motif logos    │   │ mini scatter │
 │ region bar│   │ seq→expr  │   │ [mini Pareto ↑↓]  │   │ [tiny TF strip]│   │ + TPM label  │
 └───────────┘   └───────────┘   └───────────────────┘   └────────────────┘   └──────────────┘
```

(Five labelled blocks; MODEL is narrow. If 5 reads too dense, MODEL folds into OPTIMIZE as
"deepCRE-guided evolution" — decide during implementation from the rendered draft.)

**Element inventory** (which are image assets vs. mpl):

| Element | How it's produced |
|---|---|
| Plant silhouettes (*Arabidopsis* + maize) | pre-rendered image asset |
| Promoter region track (colored promoter/5'-UTR/3'-UTR/terminator bar + labels) | **mpl** (`broken_barh`, precise + on-brand colors) |
| deepCRE model box / neural-net icon | pre-rendered image asset |
| Evolution loop icon (mutate→select circular arrow) | pre-rendered image asset |
| Mini Pareto inset (GOF rising + LOF falling front) | **mpl, real data** — reuse `show_average_pareto_front(..., ax=)` |
| WRKY / bHLH motif logos | pre-rendered image assets (or real sequence logos if a PWM + logomaker is available — check during impl) |
| Tiny diverging TF strip (optional) | **mpl, real data** — reuse `plot_heatmap` trimmed to a few top TFs |
| STARR-seq validation mini scatter (+ ✓) | **mpl** — small representative scatter (see cost note) |
| Arrows + stage labels ("optimize", "GOF ↑", "LOF ↓", "validate") | **mpl** overlay axes |

## Implementation

All edits in `compose_plots_mpl.py`, following the exact pattern of `fig4()`/`fig6()` and
their `_populate_*` companions.

1. **Add `fig1()` + `_populate_fig1(fig)`** near the top of the figure functions.
   - `fig1()` mirrors `fig6()`: open `with publication_style():`, create
     `plt.figure(figure_size_inches(DOUBLE_COLUMN_MM, ~110.0))` (landscape; height tunable),
     call `_populate_fig1(fig)`, `save_publication_figure(fig, .../fig1_composed.svg)`.
   - Use **explicit placement** (`fig.add_axes([left, bottom, w, h])`) per stage rather than
     a rigid GridSpec — a schematic needs free-form positioning. One extra full-figure
     overlay axes (`fig.add_axes([0,0,1,1])`, `axis("off")`, high z-order) hosts the
     connecting arrows and stage captions in figure-fraction coordinates.

2. **Add small helpers (unit-tested):**
   - `_place_image(ax, image_path)` — `ax.imshow(matplotlib.image.imread(path)); ax.axis("off")`
     (add `import matplotlib.image as mpimg` at top). This is the conceptual-asset embed.
   - `_draw_promoter_region_track(ax)` — `broken_barh` of the region layout (promoter 1–1000,
     5'-UTR 1001–1500, N-gap, 3'-UTR 1521–2020, terminator 2021–3020; borders already encoded
     in `region_mutation_breakdown.py`) with region labels; reuse region colors used elsewhere.
   - `_arrow(overlay_ax, xy_from, xy_to, label=None)` — `FancyArrowPatch`/`annotate` in
     figure-fraction coords.

3. **Reuse existing data functions for insets (no reimplementation):**
   - Mini Pareto: `show_average_pareto_front` (already `ax`-aware, `analysis.overview.simple_result_stats`)
     — plot a GOF run (rises) and a LOF run (falls) on one small axes to show the ↑/↓ fan.
     Strip labels/ticks down to a schematic sparkline.
   - Optional TF strip: `build_matrix` + `plot_heatmap` (`tf_comparison`), trimmed to a handful
     of rows (e.g. WRKY, bHLH), reusing the fig3 significance machinery already imported.
   - Reuse run paths already hardcoded in `_FIG4_RUNS` / `_FIG3_RUNS` so the abstract is backed
     by the same data as the detail figures.

4. **Conceptual image assets** — store as figure **inputs** in a new committed dir
   `assets/fig1/` (NOT under `figures/`, which is fully gitignored by `.gitignore:14`). List
   required assets in a short `assets/fig1/README.md` manifest so they can be regenerated.
   Reference them by relative path from the repo root, consistent with `OUTPUT_DIR`.

5. **Wire `__main__`** — uncomment/add `fig1()` in the `if __name__ == "__main__":` block
   (currently only `fig4()` is active).

6. **Update `DESIGN.md`** — add a short "Figure 1 / graphical abstract" subsection documenting
   the conceptual-schematic approach and the image-asset embedding convention (`_place_image`,
   `assets/fig1/`), so the mixed illustration+mpl pattern is recorded like the other figures.

## Open decisions / notes to resolve during implementation

- **Validation inset cost:** `fig6`'s real STARR-seq dataframe rebuild runs the full deepCRE
  TF pipelines (slow, TensorFlow). For a graphical-abstract sparkline, prefer a
  **lightweight representative scatter** (a cached/subsampled df or a stylized real subset)
  rather than rebuilding the pipeline inside `fig1()`. Flag to user if the true df is wanted.
- **Motif logos:** prefer real sequence logos if a PWM/count matrix + `logomaker` is already
  available in the env; otherwise use conceptual logo image assets. Check during impl.
- **Asset generation:** the image assets themselves (plant, model box, evolve loop, logos,
  STARR-seq icon) are produced separately (Claude image render / BioRender / Illustrator);
  the code just embeds them. This plan delivers the composition code + a manifest of what to draw.

## Testing

Per project test discipline and the existing `test_compose_plots_mpl.py` convention (only
lightweight helpers are unit-tested; `figN`/`_populate_*` are not, because they need data/models):

- Add tests in `test/workflows/paper_plots/test_compose_plots_mpl.py` for the new pure helpers:
  - `_place_image`: mock `mpimg.imread` to return a tiny array; assert an image was added to the
    axes and `ax.axis("off")` took effect (Arrange-Act-Assert, `unittest`, `matplotlib.use("Agg")`).
  - `_draw_promoter_region_track`: assert the expected number of region patches and that region
    labels are present.
  - `_arrow`: assert a patch/annotation is added to the overlay axes.
- Run the edited-file tests before returning:
  `conda run -n deepCREshap python -m pytest test/workflows/paper_plots/test_compose_plots_mpl.py`

## End-to-end verification

1. Create placeholder assets in `assets/fig1/` (or final ones).
2. Run the composition:
   `conda run -n deepCREshap python src/workflows/paper_plots/compose_plots_mpl.py`
   (with `fig1()` active in `__main__`).
3. Inspect `figures/mpl_compositions/fig1_composed.png` — confirm the 5 stages, arrows, mini
   Pareto ↑/↓ fan, and validation inset render at double-column width with uniform typography
   matching Fig 2/3/4/6.
4. Iterate on layout/heights from the rendered draft.
