# Positional span of the 5-mutation Pareto-front sequences

## Question

For each gene, where do the mutations of the **5-mutation** Pareto-front entry
sit inside the reference window, and how far apart are they? Concretely: the
position of the first mutation, the position of the last mutation, and the span
between them.

## Method

For every gene of a run we load the summarized-mutations JSON
(`all_mutated_sequences_*.json`) and take the Pareto-front entry carrying
exactly **5** mutations (if several existed, the highest-fitness one; in
practice every gene had exactly one). Let the mutated positions of that entry be
$p_1 \le p_2 \le \dots \le p_5$, 0-based indices into the 3020 bp reference
window. We record

- `first_position` $= p_1$
- `last_position` $= p_5$
- `span` $= p_5 - p_1$

and summarize each run with `DataFrame.describe()`.

Runs covered (all `*_single`, i.e. unconstrained single-nucleotide mutation):
`ara_msr_max`, `ara_msr_min`, `zea_msr_max`, `zea_msr_min` (paper runs,
generation 1999) and `GOF`, `LOF` (generation 19999).

## Results

Per run (median, and mean ± sd, in bp):

| run | n genes | span median | span mean ± sd | first median | last median |
|---|---|---|---|---|---|
| ara_msr_max | 999 | 326 | 405 ± 276 | 1060 | 1421 |
| ara_msr_min | 999 | 459 | 546 ± 327 | 1056 | 1654 |
| zea_msr_max | 1000 | 240 | 291 ± 264 | 1113 | 1386 |
| zea_msr_min | 1000 | 460 | 551 ± 362 | 1114 | 1741 |
| GOF | 105 | 295 | 375 ± 239 | 1089 | 1415 |
| LOF | 68 | 366 | 443 ± 277 | 1056 | 1451 |

All experiments pooled (one row per gene, n = 4171):

| statistic | first_position | last_position | span |
|---|---|---|---|
| count | 4171 | 4171 | 4171 |
| mean | 1124.7 | 1571.0 | 446.2 |
| std | 202.7 | 328.8 | 325.8 |
| min | 162 | 668 | 5 |
| 25% | 1012 | 1334 | 212 |
| **50%** | **1086** | **1451** | **355** |
| 75% | 1186 | 1841 | 651 |
| max | 2396 | 2993 | 2094 |

**Pooled median span = 355 bp.**

The pooling is per gene, so the four 1000-gene runs dominate and GOF/LOF
contribute ~4 % of the rows. Weighting each run equally instead (median of the
six per-run medians) gives 346 bp, so the choice barely matters here.

## Interpretation

- The **first** mutation sits at ~1050–1150 in every run and is comparatively
  tightly distributed (sd 155–241), while the **last** mutation varies far more
  (sd 235–379). The span differences between runs are therefore driven almost
  entirely by how far downstream the last mutation reaches, not by where the
  block starts.
- **Minimization runs spread wider than maximization runs** in both species
  (median 459/460 vs 326/240 bp), and LOF is wider than GOF (366 vs 295 bp) —
  the same direction in the constrained-gene-set runs.

**Caveat:** the span is a 2-point statistic over 5 mutations and is therefore
sensitive to a single outlying mutation; it says nothing about how the three
interior mutations are distributed. The per-gene CSV holds everything needed for
a more robust spread measure.

## Files

- `mutation_span.py` — the analysis (run with
  `conda run -n deepCREshap python -m workflows.evo_alg_pooled_plots.mutation_range.mutation_span`)
- `results/mutation_span_per_gene_5mut.csv` — one row per gene
- `results/mutation_span_describe_5mut.csv` — per-run `describe()`
- `results/mutation_span_describe_5mut_pooled.csv` — pooled `describe()`
