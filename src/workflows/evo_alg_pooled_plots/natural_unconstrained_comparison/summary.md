# Natural vs unconstrained optimization — final fitness comparison

## Question

The same genes were optimized twice: once allowing **all mutations**
(unconstrained) and once allowing only **naturally occurring mutations**
(natural). Restricting to natural mutations should weaken the optimization.
Does the final fitness differ per gene between the two conditions?

## Method

Each gene contributes one final-fitness value per condition, so the two values
are **paired**. We use a **Wilcoxon signed-rank test** on the per-gene paired
differences (`natural − unconstrained`), run separately for the maximization
(GOF) and minimization (LOF) gene sets. Genes are matched across runs by their
core id (first two underscore fields, e.g. `5_AT5G13640`), since the full ids
carry differing timestamp suffixes.

The hypothesis that natural is weaker is **directional and stated a priori**, so
the test is **one-sided**, with the side chosen per gene set. (One-sided is only
legitimate because the direction was fixed before seeing the data.)

Effect size is the **matched-pairs rank-biserial correlation** (−1 to 1; sign
follows the typical difference).

"Weaker" means:
- **maximization (GOF):** natural reaches a *lower* final fitness → difference < 0
  (one-sided alternative `less`)
- **minimization (LOF):** natural reaches a *higher* final fitness → difference > 0
  (one-sided alternative `greater`)

## Results

| Gene set | n pairs | median unconstrained | median natural | p-value (one-sided) | rank-biserial r |
|---|---|---|---|---|---|
| Maximization (GOF) | 104 | 1.0000 | 0.9695 | 4.3 × 10⁻¹⁹ | −1.00 |
| Minimization (LOF) | 68 | 0.0000 | 0.2664 | 3.8 × 10⁻¹³ | +1.00 |

## Interpretation

The natural-only constraint weakens optimization in the expected direction for
both gene sets, and the effect is highly significant.

The striking part is the effect size: rank-biserial `r = ±1.0` means the
constraint hurt **every single gene** — no gene did as well or better under the
natural-only constraint. The Wilcoxon statistic is 0 in both cases (all paired
differences share one sign), which is the most decisive outcome a paired test
can produce.

## Pairing note

GOF has 105 unconstrained genes but only 104 in the natural run (one gene is
absent from the natural run), so maximization is tested on the 104 shared genes.
LOF pairs all 68 genes.

## Figures

One paired box plot per gene set (box per condition, a faint line per gene
linking its two values, significance bracket with the one-sided p-value). The
y-axis is **log** for minimization (its final-fitness values span ~6 orders of
magnitude, so a linear axis collapses the unconstrained run into a flat line at
zero) and **linear** for maximization (its values cluster near 1.0, under one
order of magnitude, where log just compresses everything against the top). The
log axis is safe because no final fitness is exactly zero.

## Natural mutation availability (VCF statistics)

To contextualise how many mutations the natural run could draw from, we counted
unique positions and total mutations per gene in the source VCF files
(`vcf_mutation_stats.py`).

| Group | n genes | Unique locations (mean ± std) | Total mutations (mean ± std) |
|---|---|---|---|
| GOF | 104 | 139.2 ± 70.6 | 143.6 ± 75.6 |
| LOF | 68 | 151.6 ± 72.2 | 156.1 ± 77.5 |
| Pooled | 172 | 144.1 ± 71.5 | 148.6 ± 76.6 |

"Unique locations" counts distinct sequence positions with at least one
variant; "total mutations" counts all VCF records (multiple ALT alleles at the
same position count separately). The difference between the two is small,
indicating that most positions carry only one alternative allele.

## Solution-space statistical tests

To quantify whether the constrained run has a significantly smaller solution
space than the unconstrained run, and whether GOF and LOF differ in solution
space, we run four tests on the per-gene position and mutation counts:

| Test | Groups compared | Method |
|---|---|---|
| Paired (constrained vs unconstrained) | GOF Constrained vs GOF Unconstrained | Wilcoxon signed-rank |
| Paired (constrained vs unconstrained) | LOF Constrained vs LOF Unconstrained | Wilcoxon signed-rank |
| Unpaired (GOF vs LOF) | GOF Constrained vs LOF Constrained | Mann-Whitney U (two-sided) |
| Unpaired (GOF vs LOF) | GOF Unconstrained vs LOF Unconstrained | Mann-Whitney U (two-sided) |

Tests are run separately for the `positions` and `mutations` metrics. Results
are printed to stdout and saved to `statistical_tests.csv`.

## Natural allowance of the mutations actually introduced

The VCF statistics above say how much natural variation was *available*.
`actual_mutation_region_breakdown.py` asks the converse question: of the
mutations the **unconstrained** runs actually introduced, how many landed on a
position that natural variation also varies — i.e. how many a constrained run
could have made too?

Per gene, the most-mutated individual of the final pareto front (`front[0]`) is
diffed against the reference; each mutation's 1-based position is checked against
that gene's VCF (any alternative base counts) and binned by genomic region. Only
the unconstrained runs (`*_single_mutation_*`) are read — the natural runs are
VCF-constrained by construction and would trivially yield 100 %.

Reported per gene set and region as a **pooled percentage** (`Σ allowed / Σ
total`), with a **gene-level cluster bootstrap** 95 % CI (10 000 resamples of
whole genes, fixed seed):

| Group | Region | Genes w/ mutations | Mutations | Pooled % | 95 % CI |
|---|---|---|---|---|---|
| GOF | promoter | 96 / 104 | 587 | 4.09 | 2.46–5.84 |
| GOF | 5'-UTR | 104 / 104 | 5507 | 3.72 | 3.08–4.38 |
| GOF | 3'-UTR | 104 / 104 | 1126 | 3.46 | 2.45–4.55 |
| GOF | terminator | 30 / 104 | 90 | 8.89 | 4.55–15.00 |
| LOF | promoter | 68 / 68 | 1063 | 4.80 | 3.12–6.64 |
| LOF | 5'-UTR | 68 / 68 | 1739 | 4.66 | 3.77–5.59 |
| LOF | 3'-UTR | 68 / 68 | 1852 | 4.05 | 3.00–5.15 |
| LOF | terminator | 68 / 68 | 1466 | 6.07 | 4.29–8.21 |

**Interpretation.** Only about 3–9 % of the mutations the optimizer chose sit at
positions where nature also varies, in every region and both gene sets. Given
that the unconstrained runs have ~10³ possible positions per region against ~10¹–10²
VCF positions (panel I / the region breakdown), this is roughly what random
choice would give — the optimizer shows no preference for naturally variable
positions. All eight intervals overlap, so no region or gene set can be claimed
to differ; a gene-level permutation test would be required for that. The GOF
terminator value rests on only 30 genes and 90 mutations, hence its wide interval.

**Why pooled + bootstrap, not a per-gene mean ± SD.** Per-gene percentages are
strongly right-skewed (many genes have 1–4 mutations in a region, so their
percentage can only be 0/50/100 and most are 0): the unweighted mean is biased
upward (7.1 % vs 4.1 % pooled for the GOF promoter) and mean ± SD extends below
0 %, an impossible value. A binomial (Wilson) interval on the pooled count would
be too narrow, because mutations are clustered within genes — the effective
sample size is the number of genes, not of mutations. Resampling whole genes
respects that clustering and keeps every endpoint inside [0, 100]. See
`paper_plots/DESIGN.md` §3c for the full rationale; this analysis is panel J of
figure 4.

## Files

- `natural_unconstrained_comparison_significance.csv` — summary table (above)
- `natural_unconstrained_comparison_maximization_paired.csv` — per-gene paired
  final fitness for GOF
- `natural_unconstrained_comparison_minimization_paired.csv` — per-gene paired
  final fitness for LOF
- `natural_unconstrained_comparison_maximization.png` — paired box plot (linear)
- `natural_unconstrained_comparison_minimization.png` — paired box plot (log)
- `vcf_mutation_stats.py` — counts unique locations and total mutations per
  gene from the source VCF files
- `statistical_tests.csv` — four statistical test results (Wilcoxon and
  Mann-Whitney U) for both positions and mutations metrics
- `actual_mutation_region_breakdown.py` — per-region share of actually introduced
  mutations sitting at natural-VCF positions; writes
  `actual_mutation_allowance_per_gene.csv`,
  `actual_mutation_allowance_summary.csv`,
  `actual_mutation_allowance_pooled_ci.csv` (pooled percent + bootstrap CI),
  `actual_mutation_allowance.png` (per-gene boxplot) and
  `pooled_allowance_bars.png` (pooled bars, the figure-4 panel J version)
