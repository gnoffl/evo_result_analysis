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
