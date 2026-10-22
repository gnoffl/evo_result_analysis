# Fixing the positional null: inclusion-probability bias in the baseline samplers

**Status:** analysis complete, implementation not started.
**Date:** 2026-08-05, revised 2026-08-06 (calibration mechanics + prior art).
**Context:** paper in preparation, not yet submitted — fix before submission,
accepting that figures must be redrawn and numbers recomputed.
**Implementation plan:** `conditional_poisson_implementation_plan.md` (same folder).

---

## 1. The problem

Two samplers draw `k` distinct positions from a mutation pool, weighted by
pooled positional usage, to build the "at random" null:

- `src/workflows/mutation_distribution_analysis/baseline_sampler.py`
  → `_sample_mutations_blocking` (lines 58–106)
- `src/workflows/mutation_distance_analysis/mutation_distance_analysis.py`
  → `_sample_positions_blocking` (lines 42–77)

Both implement **successive sampling**: draw one row/position, remove it,
renormalise, repeat. `rng.choice(..., replace=False, p=...)` is implemented
this way internally, so the two are the *same* algorithm written twice — their
agreement (verified in `test/workflows/mutation_distribution_analysis/test_sampler_calibration.py`)
is not independent corroboration.

### The docstring is wrong

`baseline_sampler.py:68-69` claims position blocking "preserves the empirical
positional PMF". False. It preserves the **selection probability** at each
individual draw; it does not preserve the **inclusion probability** — the
chance a position ends up in the final set of `k`.

Key identity: for *any* without-replacement design, `sum_i pi_i = k` (take
expectations of `|S| = sum_i 1{i in S}`). "Proportional to the PMF" therefore
forces `pi_i = k * p_i` — the constant is not a modelling choice, it is pinned
by the identity. That is the benchmark the docstring commits itself to.

The causal claim in the docstring is also inverted: keeping each draw uniform
over remaining rows is precisely what *prevents* `pi ∝ p`, not what causes it.

### Mechanism

A heavy position often wins draw 1; when it does it is spent and gets no
further chances. Light positions almost always survive to draws 2..k and
collect the renormalisation windfall each time a heavy competitor is removed.
Since `sum pi_i = k` is conserved, mass migrates from peaks to valleys — the
null is **flatter** than the PMF it was calibrated against.

Exact toy verification (weights `(0.4, 0.2, 0.2, 0.2)`, `k=2`):

| position | weight | target `k*p` | exact `pi` | ratio |
| --- | --- | --- | --- | --- |
| heavy | 0.40 | 0.800 | **0.700** | 0.875 |
| light | 0.20 | 0.400 | **0.433** | 1.083 |

Most direct demonstration (no theory): pool 500 000 sampled sets and recompute
the empirical PMF. Input `(0.40, 0.20, 0.20, 0.20)` → recovered
`(0.350, 0.217, 0.216, 0.217)`. Mean draws/replicate = 2.000 exactly, i.e.
nothing is lost, mass is *moved*.

### Why `p` should be read as an inclusion probability

`p` was built by pooling the real mutation sets, each of which already
contained `k` *distinct* positions. So `p_i` is a statement about how often
position `i` **ended up in** a real set. Matching it requires `pi ∝ p`.
Self-consistency check: setting `pi_i^(g) = k_g * p_i` per gene gives
`sum_g pi_i^(g) = (sum_g k_g) * p_i ∝ p_i`, so the pooled footprint of the
null matches the pooled footprint of the data. Target is coherent.

---

## 2. Measured consequences

Bias magnitude scales with `max(k*p)`.

### Call site 1 — `mutation_distance_analysis` (unfiltered pool)

| | ara | zea |
| --- | --- | --- |
| unique positions | 2721 | ~2700 |
| genes | 999 | 1000 |
| k (min / median / max) | 44 / 88 / 90 | 79 / 90 / 90 |
| weight max/mean | 4.9x | 5.2x |
| **max k*p** | **0.162** | **0.169** |

Heaviest-position inclusion ratios 0.949–0.986 (consistently low, as predicted).

Distance null, successive vs a correct pi-ps design:
**KS = 0.0066, Wasserstein = 0.28 bp** (mean distance 25.73 vs 25.52).

Reported effect, real vs null:
**KS = 0.056 (ara) / 0.078 (zea), Wasserstein = 3.6 / 4.9 bp.**

→ The bias is **~6–12% of the reported effect, in the same direction** (it
flatters the result). The qualitative claim survives; the effect size is
slightly optimistic.

**Caveat on direction:** "flatter null → longer distances" is not a theorem.
It holds only because the heavy positions are spatially clustered (top-10
weights sit at positions 1114–1163). With dispersed hotspots the sign could
flip. This is a property of the data, not of the mathematics.

### Call site 2 — `baseline_sampler` (per-gene applicable pool) — WORSE, UNQUANTIFIED

`_filter_applicable` keeps only rows whose `source_base` matches the gene's
wildtype: ~22 000 of 83 652 rows (~26%), ~2100 of 2721 positions. Weights are
therefore far more concentrated:

| per-gene `max(k * p_applicable)` | median | max | genes with `k*p >= 1` |
| --- | --- | --- | --- |
| ara | 0.229 | 1.088 | **3** |
| zea | 0.414 | 1.155 | **6** |

- 9 genes have an **infeasible target** (`k*p > 1`, requesting a probability
  above 1). Needs take-all capping regardless of sampler choice.
- 25 ara / 120 zea genes exceed `k*p = 0.5`, so the bias here is substantially
  larger than at call site 1.
- **The downstream effect of this has NOT been quantified.** These baselines
  feed the optimised-vs-random comparison
  (`evaluate_sequences.py` / `plot_optimization_vs_random.py`). That number is
  still needed before submission.

### 2.1 Why those particular genes are infeasible

Infeasibility is `k * n_i / N > 1`, i.e. one position holding more than `1/k`
of the gene's applicable rows — at `k = 90`, more than **1.11%**. Three factors
combine, and all 9 genes fail identically.

**A hotspot at position 1001, and it is overwhelmingly `T`.** Unfiltered,
position 1001 carries 160 rows in zea (pool max is 1129 with 168) and 109 in
ara. Of those, **117 (73%) in zea and 65 (60%) in ara have
`source_base = T`.** In a 3020-bp reference with the 20-N spacer at 1500–1519,
position 1001 sits at the start of the transcribed block — one base past the
TSS under deepCRE's standard layout. That reading is inferred from the spacer
position and has **not** been checked against the deepCRE source.

**`_filter_applicable` concentrates weight onto it instead of diluting it.**
Pool-wide the filter discards ~75% of rows (median 22 111 of 89 711 survive in
zea). At position 1001, for a gene whose wildtype is `T` there, it discards
almost nothing. The filter therefore roughly triples that one position's share
while shrinking every other. The split by wildtype base at 1001 is total (zea):

| base at 1001 | genes | median `max(k*p)` | infeasible | 1001 is the heaviest position |
| --- | --- | --- | --- | --- |
| A | 204 | 0.274 | 0 | 0% |
| C | 202 | 0.273 | 0 | 0% |
| G | 163 | 0.275 | 0 | 0% |
| **T** | **431** | **0.473** | **6** | **100%** |

**Among the `T` genes, the failures have small applicable pools.** The worst
zea gene has 9120 applicable rows against a median of 22 111. Correlation of
`max(k*p)` with applicable-pool size is **-0.677**; with `k` it is **0.003**,
because `k` is saturated at 88–90 for nearly every gene and so sets the
threshold without discriminating between genes. Numerator fixed at 117,
denominator halved: `117/9120 * 90 = 1.155`.

### 2.2 Decision: cap, do not drop the genes

Take-all capping is kept and the 9 genes stay in the analysis.

**Capping is nearly free here.** It forces **1 or 2 positions** of `k = 88–90`,
and the displaced inclusion mass — the gap between the impossible proportional
target and the capped one — is at most **0.18% of `k`**:

| | k | `max(k*p)` | positions capped | displaced / k |
| --- | --- | --- | --- | --- |
| worst ara | 90 | 1.088 | 1 | 0.098% |
| worst zea | 90 | 1.155 | 2 | 0.179% |

The target asks for a position to be included *more than certainly*; capping
grants *certainly*. The residual redistributes over ~1340 other positions,
scaling each by 1.002.

**Dropping is a threshold artefact.** Nothing changes qualitatively at
`k*p = 1`; it is where the arithmetic breaks, not where the gene stops being
representative. No gene sits in `[0.95, 1.0)`, but **8 zea genes sit in
`[0.9, 1.0)`** and would be kept untouched while a gene at 1.02 is deleted.

**Dropping also selects on the outcome.** Across all genes, Spearman
correlation of `max(k*p)` with fitness gain is **+0.656 (ara, p = 3.4e-124)**
and **+0.343 (zea, p = 6.3e-29)**; with initial fitness, **-0.657 / -0.357**.
The 9 genes are ~0.9 sd above the mean in gain (ara 0.867 vs 0.482 overall;
zea 0.872 vs 0.597). So the drop rule removes low-initial-fitness,
high-headroom genes. **Be honest about the magnitude:** 9 genes of ~2000 at
0.9 sd shifts a cross-gene mean by roughly 0.004 sd, so this is negligible in
size — it removes the only argument dropping could have had rather than
supplying an argument against it. The decision rests on the two points above.

**Rejected third option: reduce `k` for those genes.** Worst of the three —
matching the optimised sequence's mutation count is the one thing the design
must hold fixed.

**When dropping *would* be right:** if capping cascaded far enough that the
null stopped being random (say 30 of 90 positions forced, leaving the baseline
largely deterministic). At 1–2 of 90 this is not close. Re-check if the pool is
ever built differently or `k` pushed higher.

**Suggested sensitivity check** when §5.2 is run: report the
optimised-vs-random comparison with and without the 9 genes. Costs nothing once
the pipeline is running and forecloses the reviewer question.

---

## 3. Options considered

All three candidate designs achieve exact `pi_i = k * p_i`.

| design | exact `pi` | robustness | speed | max-entropy |
| --- | --- | --- | --- | --- |
| **Sampford (1967)** | yes | **fails at call site 2** | 2.34 ms | approximates it |
| **pivotal (Deville & Tillé 1998)** | yes | always feasible | 7.4 ms | **no** |
| **conditional Poisson** | yes | always feasible | **0.16–0.19 ms** | **yes** |

### Sampford — rejected

Draw position 1 with prob `p`, the remaining `k-1` **with replacement** with
prob `∝ p/(1 - k*p)`, accept iff all `k` distinct. Exact; verified on the toy
(target `(0.8,0.4,0.4,0.4)` → measured `(0.799,0.400,0.400,0.401)`).

Rejection is a **birthday problem**, so it collapses under skew:

- unfiltered pool, k=88: acceptance 2.1% (45 attempts, 2.34 ms/sample).
  Decomposition — uniform weights would give 24.5% (1.41 expected colliding
  pairs); real weights give 2.2% (3.80 pairs) because skew shrinks the
  effective pool to ~1006 of 2721 slots.
- scaling in k is brutal: k=40 → 2 attempts; k=88 → 45; k=120 → 1357.
- **applicable pools: median 83 (ara) / 256 (zea) attempts, 31 / 61 genes need
  >10 000 attempts, worst-case acceptance 1e-31.** Not slow — impossible.

### Pivotal — rejected

Exact and always feasible, but **not maximum-entropy**, so the joint structure
of the null is an artefact of the pairing rather than a modelling statement.
Also 3x slower. Verified working (toy: target `(0.8,0.4,0.4,0.4)` → measured
`(0.8011,0.4002,0.3991,0.3995)`).

**Trap if ever revisited:** must use *random* pairing. The Local Pivotal Method
(`lpm1`/`lpm2` in R `BalancedSampling`) deliberately duels spatial neighbours
to impose spatial balance — that would systematically distort inter-mutation
distances.

### Why the design choice matters at all (second-order sensitivity)

Inter-mutation distance depends on the **joint** arrangement of positions, so
second-order inclusion structure `pi_ij` feeds into the *location* of the null,
not just its variance. Unusual: in ordinary survey sampling only `pi_i` matters
for unbiasedness.

Measured on the unfiltered ara pool, k=88, 3000 samples each:

| comparison | KS | Wasserstein |
| --- | --- | --- |
| pivotal vs Sampford (both correct) | 0.0027 | 0.14 |
| successive (current) vs Sampford | 0.0076 | 0.45 |

→ Design choice among *correct* designs contributes ~1/3 of what fixing the
first-order bias does (~5% of the reported effect). Numerically it won't change
the conclusion; it does determine whether the null can be **defined** or only
**described procedurally**.

---

## 4. Decision: conditional Poisson (maximum entropy) at both call sites

Wins on all three axes simultaneously — principle, robustness, and speed.

### 4.1 The design in two steps

1. **Poisson sampling.** One independent coin per position, bias `pi_i`; heads
   are the sample. Inclusion probability is exactly `pi_i` by construction.
   Flaw: `|S|` is random, sd `= sqrt(sum pi(1-pi))` ≈ 9.2 for k=88.
2. **Condition on the size.** Discard draws without exactly `k` heads.
   Acceptance `= P(|S| = k) ≈ 1/sqrt(2*pi*k)` — depends essentially only on
   `k`, hence **skew-immune**. Predicted 0.0442 vs measured 0.0428 on the
   hardest gene.

Conditioning independent coins on a fixed total gives every `k`-subset a
probability proportional to the product of its members' **odds**. Writing
`w_i` for the coin bias of position `i` and `S` for a set of exactly `k`
positions:

```text
P(S)  ∝  prod_{i in S}  w_i / (1 - w_i)
```

Derivation, so this is not an assumption: the probability that the coins land
heads exactly on `S` is `prod_{i in S} w_i * prod_{i not in S} (1 - w_i)`,
which factors as `[prod_{all i} (1 - w_i)] * prod_{i in S} w_i/(1 - w_i)`. The
bracket is the same constant for every `S`, so it vanishes into the `∝`, and
conditioning on `|S| = k` only restricts and renormalises — which leaves
relative weights untouched.

This is the **conditional Bernoulli** model of Chen, Dempster & Liu, and it is
the maximum-entropy design: independent coins impose no structure between
positions, and conditioning adds exactly one constraint. Chen & Liu (1997),
p. 877: *"the maximum entropy model in (3) is just a conditional Bernoulli
model with the `w_i` proportional to `p_i/(1-p_i)`."*

Methods sentence this licenses: *"positions were drawn under the
maximum-entropy (conditional Bernoulli) design with inclusion probabilities
proportional to pooled positional usage, conditioned on exactly k distinct
positions."*

### 4.2 The trap: conditioning moves the marginals

Feeding the target in directly as the coin biases (`w = pi`) does **not** give
inclusion probabilities `pi`. Exact enumeration of all 6 subsets of the toy
pool, target `pi = (0.8, 0.4, 0.4, 0.4)`, `k = 2`:

| | heavy | light | light | light |
| --- | --- | --- | --- | --- |
| target `pi` | 0.800 | 0.400 | 0.400 | 0.400 |
| achieved | **0.857** | 0.381 | 0.381 | 0.381 |
| ratio | **1.071** | 0.952 | 0.952 | 0.952 |

The heavy position comes out **too high** — the *opposite* sign to the
successive-sampling bias, which pushed it down. Reason: when the heavy coin is
tails the other three must supply 2 of 3 heads (unlikely at 0.4 each); when it
is heads they need only 1 of 3. The conditioning event therefore favours
subsets containing it.

**General lesson: the sign of these biases is not intuitable — compute it.**

This is not a new observation. `w = pi` is the **Hájek model**, and its failure
is stated outright in Chen & Liu (1997), p. 890: *"Although the Hájek model is
essentially a conditional Bernoulli model, it can not guarantee that the
marginal inclusion probability of each sample unit equals to its pre-specified
value, violating a condition required by all PPS sampling. To correct it, an
inversion scheme as illustrated in (11) needs to be employed in order to find
a set of proper `p_i`'s that give rise to the pre-specified marginal
probability `pi_i`."*

### 4.3 Calibration mechanics

Notation, all of it needed below. `n` is the number of positions in the pool,
`k` the sample size, `S = {1..n}` the whole pool. `pi_i` is the **target**
inclusion probability of position `i` (`sum_i pi_i = k`, all `pi_i < 1`).
`w_i` is position `i`'s **working odds** — the free parameter of the design,
equal to `bias/(1 - bias)` for its coin. `C \ {j}` means the pool with
position `j` removed. Following Chen & Liu's notation,

```text
R(k, C)  =  sum over all k-element subsets B of C  of  prod_{i in B} w_i
```

i.e. the total design weight of all `k`-subsets drawable from `C` (the
elementary symmetric polynomial of degree `k` in the odds). `R(0, C) = 1`, and
`R(k, C) = 0` for `k > |C|`.

**The equation being solved.** The achieved inclusion probability of position
`i` is the share of total design weight carried by subsets containing it:

```text
pi_hat_i(w)  =  w_i * R(k - 1, S \ {i})  /  R(k, S)
```

Numerator: every subset containing `i` is `i` itself (weight `w_i`) times a
`(k-1)`-subset of the rest. Denominator: all `k`-subsets. Calibration is
inverting this — find `w` with `pi_hat(w) = pi`. That is `n` coupled nonlinear
equations, because raising one `w_i` inflates `R(k, S)` and so pushes every
other position's inclusion *down*. No closed form; iterate.

**The published iteration.** Chen, Dempster & Liu (1994), reported as equation
(11) of Chen & Liu (1997) p. 879:

```text
w_j^(t+1)  =  pi_j * R(k-1, S\{n}) / R(k-1, S\{j})   evaluated at w = w^(t),
                                                     for j = 1..n-1
w_n^(t+1)  =  w_n^(t)  =  pi_n
```

In words: **`w_j` proportional to `pi_j / R(k-1, S\{j})`**, with the constant
of proportionality pinned by holding the last coordinate fixed. Substituting
the `pi_hat` identity above shows what it is doing — up to an overall factor
common to all `j`, it is

```text
w_j  <-  w_j * pi_j / pi_hat_j
```

a multiplicative correction on the odds scale by the ratio of desired to
achieved inclusion probability. Under-included positions get their odds
raised, over-included lowered.

**Two properties that matter for the implementation.**

- **`w` is only determined up to a global scale.** Scaling all `w_i` by `c`
  multiplies numerator and denominator of `pi_hat_i` by `c^k`, so `pi_hat` is
  unchanged. Chen & Liu exploit this by pinning `w_n = pi_n`. Consequence: the
  iteration is *completely insensitive* to the scale of its input, since
  `R(k-1, S\{j})` for different `j` all have degree `k-1` and so rescale
  identically — the ratio in (11) is scale-free. **We may therefore renormalise
  `w` freely between iterations for numerical comfort, with no effect on the
  iterates.** Verified: hand-derivation of the toy gives `w = (4,1,1,1)`, the
  solver gives `(2.846, 0.712, ...)`; both are odds ratio 4:1, i.e. the same
  design.
- **Working odds need not correspond to biases summing to `k`.** Only the
  inclusion probabilities sum to `k`.

**Convergence, measured.** No convergence proof is known to us for this or any
of the competing iterations (§4.5), so the implementation must verify its own
error rather than trust an iteration count. Measured behaviour, with `R`
computed by the recursion below and the stopping rule
`max_i |pi_hat_i - pi_i| < 1e-10`:

| case | max target `pi` | iterations | ms/iter | total |
| --- | --- | --- | --- | --- |
| toy, n=4, k=2 | 0.800 | 74 | — | — |
| asymmetric, n=9, k=3 | 0.717 | 68 | — | — |
| n=400, k=88 (high skew) | 0.847 | 95 | 3.1 | 0.30 s |
| n=2100, k=88 (call site 2 size) | 0.166 | 9 | 17.3 | 0.16 s |
| n=2721, k=88 (call site 1 size) | 0.132 | 8 | 22.0 | 0.18 s |

Iteration count is driven by skew, not by `n`: at the real pools' skew it
converges in under ten iterations. Contraction on the toy is ~2/3 per
iteration, monotone.

**Recursion for `R`.** Brute-force enumeration is impossible for
`C(2721, 88)`. Two published recursions exist, and the choice between them is
a numerical-stability decision:

- **Gail, Lubin & Rubinstein (1981)** — `R(k,C) = R(k, C\{j}) + w_j R(k-1, C\{j})`.
  Additions and multiplications of non-negative numbers only. **Use this one.**
- **Chen, Dempster & Liu (1994), "Method 1"** — Newton's identities,
  `R(k,C) = (1/k) sum_{i=1..k} (-1)^(i+1) T(i,C) R(k-i,C)` with power sums
  `T(i,C) = sum_{j in C} w_j^i`. Cheaper, because all `n` leave-one-out values
  follow from `T(i, S\{j}) = T(i,S) - w_j^i`. But it is an **alternating**
  series, and Chen & Liu (1997) p. 880 explicitly warn that consecutive terms
  can agree to several significant figures with opposite signs. Rejected on
  those grounds: this is exactly the cancellation the heavy positions would
  suffer from.

All `n` leave-one-out values `R(k-1, S\{j})` are obtained in `O(n*k)` total by
a **prefix/suffix pair** of Gail-Lubin-Rubinstein tables — which is Chen &
Liu's Proposition 1(c), `sum_{i=0..k} R(i,C) R(k-i, C^c) = R(k,S)`:

```text
forward[m][j]   =  R(j, {1..m})
backward[m][j]  =  R(j, {m..n})
R(k-1, S\{j})   =  sum_{a=0..k-1} forward[j-1][a] * backward[j+1][k-1-a]
```

**Overflow control.** `R(88, S)` for 2721 positions is order `1e169` and skew
makes intermediates worse. Scale each DP row by its own maximum and carry the
log of the accumulated divisor. The scale of a term
`forward[j-1][a] * backward[j+1][k-1-a]` depends only on the row indices
`j-1` and `j+1`, **not on `a`**, so it factors out of the convolution exactly
and the sum stays in scaled arithmetic:

```text
log R(k-1, S\{j})  =  log_scale_forward[j-1] + log_scale_backward[j+1]
                      + log( sum_a forward_scaled * backward_scaled )
log pi_hat_j       =  log w_j + log R(k-1, S\{j}) - log R(k, S)
```

Verified against brute-force enumeration: max error in `log R` is `2.7e-15`
over `(n,k)` in `{(4,2), (9,3), (12,5)}` with random odds spanning two orders
of magnitude. **Caution learned the hard way:** the `R(0, ·) = 1` entry must be
carried in the *current row's scaled units*, not reset to a literal `1.0` —
doing the latter silently corrupts the whole table (it produced errors of order
`1e0` in `log R`, and a calibration that did not converge).

### 4.4 Drawing the sample

Two exact options once `w` is known.

- **Rejective** (the definition): flip Bernoulli(`w_i/(1+w_i)`) across all
  positions, accept iff exactly `k` heads. ~23 attempts, ~0.17 ms/sample.
  Correctness is definitional, which makes it the easiest to defend and test.
- **Sequential, rejection-free** (Chen & Liu 1997 Procedure 3, "ID-checking
  sampling"; implemented as `UPMEqfromw` + `UPMEsfromq` in R `sampling`):
  convert `w` into a table `q[i, z]` of conditional selection probabilities,
  then walk the positions once, including position `i` with probability
  `q[i, remaining]`. Exact, `O(n*k)`, zero rejections.

**Choice: rejective**, with the sequential method recorded as the fallback if
acceptance ever degrades. At `k <= 90` acceptance is `~1/sqrt(2*pi*k) >= 4%`,
so rejection costs nothing and buys a sampler whose correctness needs no
argument beyond the definition of conditioning.

### 4.5 Prior art — three variants of the same inversion exist

Worth recording, because they are not interchangeable and only one is being
cited:

| variant | update | source |
| --- | --- | --- |
| **multiplicative on odds** (**chosen**) | `w_j <- w_j * pi_j / pi_hat_j` | Chen, Dempster & Liu (1994); Chen & Liu (1997) eq. (11) |
| additive on the `pi` scale | `w <- w + pi - pi_hat` | R `sampling::UPMEpiktildefrompik`, attributed to Deville (2000) / Tillé (2006); docs call it "Newton's method" |
| multiplicative on inclusion *odds* | `w_j <- w_j * [pi_j/(1-pi_j)] / [pi_hat_j/(1-pi_hat_j)]` | **not found in the literature** |

All three are fixed points of the same equation and agree at the solution. The
third converged ~3x faster per iteration in our tests (contraction 2/9 vs 2/3
on the toy, 4 iterations vs 8 at n=2721) but has no citation, and at 0.2 s per
calibration the speed is irrelevant. **We use CDL (11).**

Still open, and we should not claim otherwise: we have found **no proof of
existence/uniqueness** of `w` for a feasible target, and **no convergence
proof** for any of the three iterations. The scale non-uniqueness is used
operationally by CDL (pinning `w_n`) rather than proved. Tillé (2006) §5.6 is
the likely place for the theory and has not been consulted.

---

## 5. Sources

- **Chen, S. X. & Liu, J. S. (1997).** Statistical applications of the
  Poisson-binomial and conditional Bernoulli distributions. *Statistica Sinica*
  **7**(4), 875–892. <https://www3.stat.sinica.edu.tw/statistica/oldpdf/A7n44.pdf>
  — the accessible primary source. Eq. (6) defines `R(k,C)`; eq. (11) (p. 879)
  is the calibration; p. 877 states the maximum-entropy equivalence; p. 880
  warns about the alternating recursion; p. 890 names the Hájek failure;
  Proposition 1 (p. 881) gives the identities used above.
- **Chen, S. X., Dempster, A. P. & Liu, J. S. (1994).** Weighted finite
  population sampling to maximize entropy. *Biometrika* **81**(3), 457–469 —
  origin of both the design and eq. (11). Cite alongside the 1997 paper, which
  is what we actually read.
- **Gail, M. H., Lubin, J. H. & Rubinstein, L. V. (1981).** Likelihood
  calculations for matched case-control studies and survival studies with
  tied death times. *Biometrika* **68**(3), 703–707 — the non-alternating
  recursion for `R`.
- **Tillé, Y. (2006).** *Sampling Algorithms*, Springer, ch. 5–6, esp. §5.6 —
  standard modern treatment; **not yet consulted**, and the place to look for
  existence/uniqueness and convergence.
- **R `sampling` package** (Tillé & Matei), functions `UPmaxentropy`,
  `UPMEpiktildefrompik`, `UPMEqfromw`, `UPMEpikfromq`, `UPMEsfromq`,
  `inclusionprobabilities`.
  <https://cran.r-project.org/web/packages/sampling/> — reference
  implementation; source read from the CRAN GitHub mirror. Uses the additive
  variant and the sequential draw. **Not installed** (R itself is, at
  `/usr/bin/R`). Intended as a **test-time cross-check, not a runtime
  dependency** — pulling R/rpy2 into a TensorFlow conda env is a bad trade.
- Sampford (1967), *Biometrika* — rejected alternative. Deville & Tillé (1998)
  — pivotal, rejected. Hájek (1964, 1981) — successive-sampling asymptotics and
  the model that `w = pi` corresponds to.
- Aires (2000), *Methodol. Comput. Appl. Probab.* **2**, 457–469; Bondesson et
  al. (2006), *Scand. J. Statist.* **33**, 699–720 — exact-inclusion-probability
  algorithms and comparisons.
- Keywords: `inclusion probability` vs `selection probability`,
  `conditional Bernoulli`, `conditional Poisson sampling`,
  `maximum entropy sampling`, `pi-ps sampling`, `Horvitz-Thompson`,
  `Poisson-binomial`.
- Same trap in ML vocabulary: `Gumbel top-k trick`, `weighted reservoir
  sampling` (Efraimidis–Spirakis), and Kool, van Hoof & Welling, "Estimating
  Gradients for Discrete Random Variables by Sampling without Replacement"
  (ICLR 2020).
- Python `samplics` (PyPI 0.4.11) resolves but is **not** verified to implement
  max-entropy sampling — recollection is Brewer / Hanurav-Vijayan / Murthy /
  systematic. Would need checking before relying on it. Nothing
  survey-sampling related is currently in `deepCREshap`.

---

## 6. Verified scratch scripts

In `plans/sampler_fix_scratch/`. All were run against the real pools in
`mutation_pools/`; none are production code.

| script | what it establishes |
| --- | --- |
| `toy.py` | exact `pi` under successive sampling; numpy `choice` matches |
| `recover_pmf.py` | pooled null PMF ≠ input PMF (the direct demonstration) |
| `pre_distort.py` | to hit `pi = k*p` the *input* weights must be pre-distorted |
| `check_bias.py` | bias magnitude on the real ara pool vs a pi-ps design |
| `vs_real.py` | reported real-vs-null effect sizes for both pools |
| `sampford.py`, `sampford_real.py` | Sampford exactness + cost |
| `why_rejection.py` | acceptance-rate decomposition and k-scaling |
| `callsites.py`, `feasibility.py`, `accept_baseline.py` | call-site differences, infeasible genes, Sampford collapse |
| `pivotal.py`, `second_order.py` | pivotal exactness; second-order comparison |
| `condpoisson_viability.py` | conditional Poisson acceptance across all genes |
| `cond_poisson_explained.py` | calibration converges to machine precision (enumeration) |
| `trace.py` | per-iteration convergence trace of the odds-scale variant on the toy |
| `verify_cdl.py` | scaled prefix/suffix recursion vs brute-force `R`; CDL (11) end to end |
| `compare_updates.py` | CDL vs odds-variant contraction rates (2/3 vs 2/9 on the toy) |
| `bench_cdl.py` | iteration counts and ms/iter at n=400 / 2100 / 2721, k=88 |
