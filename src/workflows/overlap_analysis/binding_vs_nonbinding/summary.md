# Binding vs non-binding deepCRE predictions

## Question

Does deepCRE assign different expression-activity predictions to STAR-seq
construct sequences labelled **`binding`** (contain a predicted TF site) versus
**`non_binding`** (controls), separately for each TF family (WRKY, bHLH)? In
other words: does deepCRE "see" the presence/absence of a TF binding site?

## What we implemented (`binding_vs_nonbinding.py`)

The analysis is run in **two independent backgrounds**, kept strictly separate
because they answer the same question in different sequence contexts:

1. **Vector (construct) background** — the insert embedded in the experimental
   plasmid with two barcodes, scored by deepCRE (cached in
   `correct_construct/predictions_cache.csv`). Barcodes are averaged to one
   value per sequence.
2. **Genomic (native) background** — the insert mapped onto its native deepCRE
   reference window and scored there (built by the per-TF correlation
   pipelines, `prediction_mutated`).

For each background we run two framings:

### Absolute analysis

Compare the raw deepCRE prediction of `binding` vs `non_binding` sequences per
TF, using a **Mann-Whitney U** test. Reference sequences (the native,
unmutated sequences) are genuine binding sequences and are kept in the binding
group.

### Delta analysis

**Why:** each locus has its own baseline predicted activity (across loci,
predictions span ~0.2–0.95). That between-locus spread dominates the variance
and masks the small contribution of the binding-site mutations themselves. To
isolate the mutation effect we compute, per variant,

```python
delta = prediction(variant) − prediction(reference of the same locus)
```

This is a within-locus design that removes the per-locus baseline. The genomic
side reuses the precomputed `delta_prediction = prediction_mutated −
deepcre_ref_fitness`; the vector side subtracts the single per-locus reference.

**Reference handling:** references are the *normaliser* in the delta analysis,
so their own delta is structurally 0. Including them in the binding group would
inject a spike of zeros (a strong bias), so references are **excluded** from
both delta groups. (They are legitimately kept in the absolute analysis, where
their prediction is a real observation.)

Delta tests per TF: Mann-Whitney U (binding-delta vs non_binding-delta) plus a
Wilcoxon signed-rank test of each group against 0.

### Effect sizes (important)

Every test reports not just a p-value but a **common-language effect size**
`prob_binding_gt_non_binding` (= U / (n₁·n₂); 0.5 = no effect) and
**rank-biserial `r`** (−1…1). With ~700–2000 observations per group, a
vanishingly small distributional difference produces a small p-value, so the
p-value alone is misleading — the effect size is what tells you whether the
difference matters. Plots annotate `p` and `r` together.

## Outputs (`plots/`)

Per background: `*_boxplot.png` / `*_stats.csv` (absolute) and
`*_delta_boxplot.png` / `*_delta_stats.csv` (delta). PNGs and CSVs are
gitignored.

## Results

| Background | TF | analysis | p | effect size `r` |
| --- | --- | --- | --- | --- |
| vector | WRKY | absolute | 0.79 | +0.01 |
| vector | bHLH | absolute | 0.95 | +0.00 |
| genomic | WRKY | absolute | 0.45 | −0.02 |
| genomic | bHLH | absolute | 0.21 | +0.03 |
| vector | WRKY | delta | 0.017 | −0.07 |
| vector | bHLH | delta | 0.019 | −0.06 |
| genomic | WRKY | delta | 0.038 | +0.06 |
| genomic | bHLH | delta | 0.010 | −0.08 |

**Interpretation.** The absolute analysis shows no difference. The delta
analysis returns "significant" p-values, but the effect sizes are tiny
(all `|r| < 0.08`, i.e. P(binding > non_binding) within ~3–4% of the 0.5
coin-flip) and the **direction is inconsistent** across TFs and backgrounds.
Median deltas are on the order of 1e-4 to 1e-3, against a prediction range of
~0.2–0.95. The small delta p-values are an artefact of large sample size, not a
meaningful biological signal.

**Conclusion:** deepCRE does **not** meaningfully distinguish binding from
non_binding sequences, in either the vector or the genomic background — the
mutations that flip binding status change its predicted expression by
essentially nothing. The delta framing did not rescue a hidden signal; it only
made a microscopic, inconsistent one cross an arbitrary significance threshold —
which is exactly why the effect-size columns were added.

## Running

```bash
conda run -n deepCREshap python \
  src/workflows/overlap_analysis/binding_vs_nonbinding/binding_vs_nonbinding.py
conda run -n deepCREshap python -m pytest \
  test/workflows/overlap_analysis/binding_vs_nonbinding/test_binding_vs_nonbinding.py
```
