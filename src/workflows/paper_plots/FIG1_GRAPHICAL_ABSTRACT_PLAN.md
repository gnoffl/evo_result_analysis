# Plan: Figure 1 — Graphical Abstract

> **Part 1 (below) is a self-contained scientific briefing** meant to be handed to a
> collaborator or a generative agent that has never seen this project. It carries the
> biology, the computational stack, the actual datasets, the measured numbers, and the
> honest limits of each claim, so a design discussion can start without further context.
> **Part 2** is the implementation plan for the matplotlib composition.
>
> Facts marked **[verify]** are inferences from the code/configs that the author should
> confirm before they appear in a published caption.

---

# Part 1 — Scientific briefing

## 1.1 One-paragraph summary of the paper

Deep-learning models can predict a plant gene's expression level from its surrounding
DNA sequence alone. This work turns such a model around and uses it as a **fitness
function for a genetic algorithm** that rewrites a gene's own regulatory sequence to
either **raise (gain of function, GOF)** or **lower (loss of function, LOF)** the
predicted expression, while simultaneously keeping the number of introduced point
mutations as small as possible. Because the search is multi-objective, every gene yields
a **Pareto front**: the best achievable predicted expression for 1, 2, 3, … mutations.
The paper then asks *what the optimizer actually did* — where in the regulatory window
the mutations land, which nucleotides they favour, which **transcription-factor (TF)
family binding sites** they create or destroy, and how much of that overlaps with
variation that exists in nature. Finally it confronts the model with two experimental
readouts (**plantSTARR-seq enrichment** and **RNA-seq expression**) to bound how far the
in-silico result can be trusted.

The take-home message the graphical abstract must convey: **a handful of single-base
changes, chosen by model-guided evolution, is enough to move predicted expression from
one extreme to the other — and those changes are not random, they are systematic edits
to TF binding-site content.**

## 1.2 Biological background

**What regulates a plant gene.** Whether a gene is transcribed, and how strongly, is
largely set by non-coding DNA flanking it: the **promoter** immediately upstream of the
**transcription start site (TSS)**, the **5′-UTR**, and, on the other end, the **3′-UTR**
and the **terminator** downstream of the **transcription termination site (TTS)**. These
regions contain short (~6–15 bp) **cis-regulatory elements** — binding sites for
**transcription factors (TFs)**, proteins that dock onto specific DNA words and recruit
or block the transcription machinery. A gene's expression level is, to first order, a
readout of which TFs can bind its flanking regions and where.

**TF families.** TFs are grouped into families by the DNA-binding domain they share, and
family members recognise similar motifs. Two families matter repeatedly in this paper:

- **WRKY** — binds the W-box (core `TTGAC(C/T)`); central to defence and stress
  responses; can act as activator or repressor.
- **bHLH** (basic helix-loop-helix) — binds G-box-like `CANNTG` motifs; huge family,
  involved in light signalling, hormone responses, development.

The paper's TF analysis covers **46 families** in total (the output layer of the deepCIS
model; full list in `src/analysis/motives/deepcis_scanner.py:TF_FAMILY_NAMES`). Names in
the code carry the suffix of the assay the motif came from (`WRKY_tnt`, `bHLH_tnt`,
`ARF_ecoli`, …) — `_tnt` / `_ecoli` are **assay tags, not biology**, and should never
appear in a figure label.

**Why engineer regulatory sequence at all.** Changing a promoter changes *how much* of a
protein a plant makes, without changing the protein itself. That is the lever for crop
traits (stress tolerance, yield, nutrient content) and the reason "design a promoter
that gives expression level X" is a real biotech goal. Coding-sequence edits change what
a protein does; regulatory edits change the dose.

**Natural variation.** Within a species, individuals differ at millions of positions
(SNPs, catalogued in **VCF** files). A mutation that already exists in some accession is
far more plausible — and far easier to obtain by breeding rather than transformation —
than an arbitrary base swap. This paper therefore runs each optimization **twice**: once
free to use any substitution ("unconstrained"), once restricted to the substitutions
observed in natural populations at that exact position ("natural" / VCF-constrained).

## 1.3 The computational stack (three separate models — do not conflate them)

### deepCRE — sequence → expression (the fitness function)

- A convolutional neural network that predicts gene expression from flanking DNA.
  Published as **Peleke et al. 2024, Nature Communications**
  (`s41467-024-47744-0`); this project uses an in-house reimplementation
  (`deepCRE_reimplemented_Fritz`).
- Input: one **3020 bp "extraction window"** per gene, assembled as
  ```
  0                    1000  1500  1520                    2020            3020
  |---- promoter -------|TSS|-5'UTR-| NNN |-3'UTR-|TTS|--- terminator -----|
       1000 bp extragenic      500 bp      20 N     500 bp     1000 bp extragenic
                              intragenic   pad     intragenic
  ```
  i.e. **1500 bp around the TSS + 20 `N` spacer + 1500 bp around the TTS**. The gene body
  in between is *not* seen by the model. Region borders used in the analysis:
  promoter 1–1000, 5′-UTR 1001–1500, spacer 1501–1520, 3′-UTR 1521–2020,
  terminator 2021–3020 (`region_mutation_breakdown.py`). Short genes (< 1000 bp) get
  extra central padding, which shifts these borders per gene.
- Output: a single number in **(0, 1)** — the model is a **binary high-vs-low expression
  classifier, and the score is the probability of the "high" class** **[verify]**. This
  is why every fitness axis in the paper is bounded by 0 and 1, and why saturation at the
  ceiling/floor is a recurring feature.
- Two model flavours are used: **SSR** (single-species, trained on one genome) and
  **MSR** (multi-species, trained across 12 species: *A. thaliana*, beet, cabbage,
  *Camelina*, quinoa, cucumber, carrot, soybean, rice, tomato, sorghum, maize). The
  paper's main runs use the **MSR** model; the MSR model is also the better one against
  measured expression (§1.6).

### deepCIS — sequence → TF binding (the mechanism readout)

- A separate network predicting, for a **250 bp window**, the binding probability of each
  of **46 TF families**.
- Applied as a **sliding-window scan** across the 3020 bp sequence with **step 10 bp**,
  giving a per-family binding *signal along the sequence* for the reference and for the
  optimized variant.
- A **4-stage peak caller** (`PeakAnnotator`: region detection by moving average →
  Gaussian smoothing (σ = 20 bp) + derivatives → greedy non-overlapping peak selection →
  coordinate mapping) turns each signal into discrete **predicted binding sites**.
- The per-gene, per-family quantity used for statistics is
  `diff = peaks(optimized) − peaks(reference)`: **positive = the optimizer introduced
  binding, negative = it destroyed binding.**

### The genetic algorithm — the optimizer

- Implemented with **DEAP** (`Evolution/src/evolution/genetic_algorithm.py`),
  **multi-objective NSGA-II** selection, hall of fame = a **Pareto front archive**.
- **Two objectives:** (1) the deepCRE score — maximised for GOF (`weights = [1.0, -1.0]`)
  or minimised for LOF (`weights = [-1.0, -1.0]`); (2) the **number of point mutations**
  relative to the reference — always minimised.
- Only **single-base substitutions** — no indels, no rearrangements. Mutations are
  confined to the 3020 bp window; `N` positions are never mutated. A hard cap of
  **90 mutations** (`max_number_mutations`, ≈3 % of 3020 bp) is enforced by randomly
  reverting excess mutations.
- Two mutation regimes: **"single"** (population initialised with one mutation each, the
  operator flips exactly one base — the paper's main runs) and **"multi"** (mutations
  distributed across the sequence from the start). A **greedy** hill-climbing variant
  exists as a baseline.
- Run sizes: main paper runs = population 100, **2000 generations**, ~1000 genes each;
  GOF/LOF runs = population 200, **20 000 generations**, 104 GOF / 68 LOF genes.
- **The Pareto front is the central data object.** By project convention
  `front[0]` is the optimized endpoint and `front[-1]` is the reference — index, never
  `max`/`min`.

### The three models in one sentence

deepCRE *scores* a sequence, the genetic algorithm *rewrites* it, deepCIS *explains*
what the rewrite did to TF binding. **plantSTARR-seq and RNA-seq are the only
experimental data in the paper** — everything else is prediction.

## 1.4 The datasets / runs

| Run label | Species | Model | Objective | Genes | Constraint |
|---|---|---|---|---|---|
| `ara_msr_max_single` | *A. thaliana* | MSR | maximize | ~999 | none |
| `ara_msr_min_single` | *A. thaliana* | MSR | minimize | ~999 | none |
| `zea_msr_max_single` | *Z. mays* | MSR | maximize | ~1000 | none |
| `zea_msr_min_single` | *Z. mays* | MSR | minimize | ~1000 | none |
| `GOF_single` | *A. thaliana* | MSR | maximize | 104 | none |
| `GOF_single_natural` | *A. thaliana* | MSR | maximize | 104 | VCF (natural SNPs only) |
| `LOF_single` | *A. thaliana* | MSR | minimize | 68 | none |
| `LOF_single_natural` | *A. thaliana* | MSR | minimize | 68 | VCF (natural SNPs only) |

The ara max / ara min runs contain **exactly the same 999 genes**, which is what makes
the max-vs-min TF comparison a *paired* test. The GOF and LOF gene sets are different
genes from each other and from the ara runs; how those two sets were chosen is **[verify
— not recorded in this repo]**. Note the naming trap: `*_single_mutation_*` directories
are the **unconstrained** runs, `*_single_natural_*` the **VCF-constrained** ones.
Legacy naming also calls the optimized sequence `max_mutated` even in minimization runs.

## 1.5 What was found (the results a graphical abstract may draw on)

**Optimization works, and cheaply.** Predicted expression is driven to the bound: the
unconstrained runs reach a median final fitness of **1.0000** (GOF) and **0.0000** (LOF),
with a final-fitness standard deviation of ~1e-4 — i.e. they hit the ceiling/floor within
the mutation budget almost regardless of where the gene started. The Pareto fronts
saturate well before the 90-mutation cap (they are cropped at 45 mutations in Fig 2
because the tail is flat).

**Natural variation is a real but surmountable constraint.** Restricting to natural SNPs
weakens the optimization in the expected direction for both objectives, highly
significantly (Wilcoxon signed-rank, one-sided, direction fixed a priori): GOF median
1.0000 → **0.9695** (p = 4.3e-19), LOF median 0.0000 → **0.2664** (p = 3.8e-13), with
matched-pairs rank-biserial **r = ±1.00** — *every single gene* moved in the expected
direction. The headline framing: even using only mutations that already exist in nature,
the optimizer still pushes predicted expression to ~0.97 / ~0.27.

**The optimizer edits TF binding-site content systematically.** Per-gene `diff` values
per TF family, tested with Wilcoxon signed-rank + **Benjamini–Hochberg FDR** across
families, show families that are consistently *introduced* when maximizing and *removed*
when minimizing, and families with the mirror-image behaviour (consistent with activators
vs repressors). The max-vs-min paired contrast on the ara runs is the primary test; each
run also gets an independent intra-run test. This mirror-image GOF/LOF structure is the
most visually compelling mechanistic result.

**Where the mutations go.** Mutations are counted per genomic region (promoter, 5′-UTR,
3′-UTR, terminator), with region length differences (1000/500/500/1000 bp) kept in mind.
Their *spacing* is compared against a null model that draws positions from the
optimizer's own empirical positional distribution — so the question is specifically
"is the *spacing* special, given the positional preference", not "is the position
special". Net nucleotide change (times a base was introduced minus removed, summed over
genes) gives a compositional signature of the edits.

**Overlap with natural variation.** Of the mutations the *unconstrained* runs actually
introduced, only a small fraction sit at positions where natural variation exists:
**pooled ≈ 3.5–9 %** across regions and both objectives (e.g. GOF promoter 4.09 %
[95 % CI 2.46–5.84], LOF terminator 6.07 % [4.29–8.21]). Intervals come from a
**gene-level cluster bootstrap** (10 000 resamples of whole genes), because mutations are
clustered within genes and a binomial interval would be anti-conservative. **All eight
intervals overlap** → the supportable statement is *"roughly 3–9 % of introduced
mutations coincide with natural variation, everywhere"*, and **not** that any region or
gene set differs from another. (The tall GOF-terminator bar rests on only 30 genes /
90 mutations.)

**Single-gene illustration.** `AT3G60640` (constrained GOF run): its **5-mutation**
Pareto-front member versus the reference, scanned for the `LOBAS2` family. The five
mutations sit at 1-based positions **804, 937, 957, 1149, 1194** — three in the promoter,
two in the 5′-UTR — and the binding signal changes only in the windows those mutations
touch. This is the concrete "5 bases are enough" visual.

## 1.6 How far the experimental validation goes — read this before claiming "validated"

Two independent experimental confrontations, with genuinely different verdicts. **The
graphical abstract must not overstate this.**

**(a) deepCRE vs measured RNA-seq expression — supportive.** Predictions for 23 090
*A. thaliana* genes against `logMaxTPM` from leaf RNA-seq: **MSR model Pearson
r = 0.545**, SSR model r = 0.356 (both p ≈ 0). Within the high- or low-expression subsets
alone the correlation collapses (r ≈ 0.08–0.19), i.e. the model separates high from low
but is weak at ranking *within* a class — consistent with it being a binary classifier.

**(b) plantSTARR-seq enrichment vs deepCRE — mixed, and the negative half matters.**
plantSTARR-seq measures the regulatory activity of many short (**170 bp**) DNA fragments
in parallel as an RNA/DNA **enrichment** ratio, here in two conditions (**light** and
**dark**). Fragments from an **in-silico mutagenesis library** for WRKY and bHLH sites
were mapped into their gene's 3020 bp window and spliced in, then scored by deepCRE.

- *Positional correlation — supportive.* Spearman correlation between prediction and
  enrichment depends strongly on **where in the window** the fragment sits, peaking in a
  200 bp window centred at position **≈928** (i.e. ~70 bp upstream of the TSS at 1000):
  **ρ = 0.479** pooled (p = 2.6e-50), **0.546** light, **0.415** dark. So in the region
  that matters most for a promoter, model and experiment agree moderately.
- *Binding vs non-binding — null.* Sequences that **do** contain a predicted TF site were
  compared against matched controls that do not, in both a plasmid (vector) and a native
  genomic background, absolute and within-locus (delta) framings, Mann–Whitney U.
  Absolute: no difference (p 0.21–0.95). Delta: p-values 0.010–0.038 but effect sizes
  **|r| < 0.08** with **inconsistent sign** across TFs and backgrounds, median deltas
  1e-4–1e-3 against a prediction range 0.2–0.95. **Conclusion in the source analysis:
  deepCRE does not meaningfully distinguish binding from non-binding sequences.** The
  small p-values are a sample-size artefact.

**Honest framing for the abstract.** The defensible claim is *"model predictions track
experimental readouts at the level of whole regulatory windows (RNA-seq r = 0.55;
STARR-seq ρ ≈ 0.48 in the promoter-proximal window), while single-site binding-status
contrasts are not resolved."* A ✓/"validated" stamp on the validation stage of the ribbon
would overstate the STARR-seq result. Prefer a neutral "compared against experiment"
framing, or show the correlation panel and let it speak. Also note: nothing in this paper
is an experimental test of a **designed** sequence — the optimized sequences were never
synthesised. This is a computational design study with a model-validation appendix.

## 1.7 Vocabulary and conventions for anyone drafting the figure

**Use:** regulatory window / extraction window (not "the promoter" for the whole 3020 bp);
point mutation or single-base substitution (never "edit" as in CRISPR, never indel);
predicted expression / deepCRE score / fitness (never "measured expression");
TF binding site or predicted binding site; GOF = gain of function = maximization,
LOF = loss of function = minimization; Pareto front (never "trade-off curve" alone);
"natural" / VCF-constrained vs unconstrained.

**Avoid:** anything implying CRISPR, gene editing tooling, or transgene insertion; "the
algorithm designs proteins" (it designs regulatory DNA); "validated in planta";
protein-level or phenotype-level outcomes (no bigger/greener plant as an *outcome* of the
optimization — the optimization output is a *sequence and a predicted score*);
naming individual TF proteins (the resolution here is *families*).

**Scientific correctness checklist for illustrated panels.**

- The promoter is **upstream (5′) of the TSS**; the terminator is **downstream (3′) of the
  TTS**; the gene body between them is **not part of the model input** — if the illustration
  shows a gene, the two 1500 bp windows should visibly flank it with the body greyed out.
- Mutations are **sparse**: a handful to a few dozen over 3020 bp. Do not draw a densely
  peppered sequence.
- TF proteins bind as **dimers or complexes at short motifs**; drawing one blob per base
  is wrong.
- Two species only: ***Arabidopsis thaliana*** (small rosette, thin bolting stem, small
  white four-petal flowers) and ***Zea mays*** (maize; broad strap leaves, tassel).
  Species names always italic.
- The fitness axis is bounded **0 to 1** and GOF/LOF saturate at the bounds — any curve
  drawn should approach an asymptote, not shoot past it.
- Arrows in the pipeline run **left to right**; the GOF/LOF split is the only place where
  a fan of two directions (↑ / ↓) is appropriate.

**Colour conventions already used in the paper figures** (a graphical abstract should
match, not invent): the **magma** colormap is the figure family's identity — mutation
counts use magma; Fig 4's region panels sample it (`#3b0f70` constrained / `#f9795d`
unconstrained; `#ca3e72` GOF / `#fec68a` LOF); introduced-mutation markers are
`#d62728`; Pareto/neutral elements are grey `0.35`; there is one shared A/C/G/T
nucleotide palette (`analyze_mutations.COLORS`).

## 1.8 Open items to settle with the author before finalising

- **Fig 5 is missing** from `compose_plots_mpl.py` (only 2, 3, 4, 6 exist) — what does it
  show, and does the abstract need it?
- The **paper's title and one-sentence claim** are not recorded anywhere in this repo;
  the abstract's headline text should come from them.
- Target **journal / graphical-abstract format** (aspect ratio, max size, whether text is
  allowed) — the current plan assumes double-column landscape.
- Whether to depict **both species** or simplify to *Arabidopsis* (only the ara runs carry
  the GOF/LOF and validation analyses; maize appears in Fig 3's mutation signatures).
- Whether the **greedy** and **multi-mutation** regimes are part of the paper's story or
  supplementary.

---

# Part 2 — Implementation plan

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
5. **Validation** — model predictions are confronted with experimental **plantSTARR-seq**
   enrichment and **RNA-seq TPM** (Fig 6, TPM correlation). ⚠️ The agreement is
   **partial** — moderate at the level of whole regulatory windows, absent for
   single-site binding-status contrasts. See §1.6 before wording or stamping this stage.

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
| STARR-seq validation mini scatter (no ✓ — see §1.6) | **mpl** — small representative scatter (see cost note) |
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
