# Implementation plan: conditional Bernoulli (maximum-entropy) position sampling

**Status:** implemented 2026-08-06, except §5.2 (see §8).
**Date:** 2026-08-06.
**Rationale and prior art:** `sampler_inclusion_probability_fix.md` (same
folder). Read §4 of that document first; this file assumes it.

**Algorithm committed to:** the conditional Bernoulli / maximum-entropy design
of **Chen, Dempster & Liu (1994)**, with working odds obtained by their
inversion, equation (11) of **Chen & Liu (1997)**, and the normalising constant
`R(k, C)` computed by the non-alternating recursion of **Gail, Lubin &
Rubinstein (1981)**. Every component is citable; no novel method is introduced.

---

## 1. Scope

Replace the successive-sampling position draw at both call sites so that the
inclusion probability of position `i` is exactly `pi_i = k * p_i` (with
take-all capping where that is infeasible), and record the resulting change to
every published number that depends on it.

**Call site 1** — `src/workflows/mutation_distance_analysis/mutation_distance_analysis.py`,
`_sample_positions_blocking` (lines 42–77), used by `compute_random_distances`.

**Call site 2** — `src/workflows/mutation_distribution_analysis/baseline_sampler.py`,
`_sample_mutations_blocking` (lines 58–106), used by
`generate_baselines_fasta`.

Out of scope: the peak-detection and TF-mapping pipelines, and any change to
how the mutation pool itself is built.

---

## 2. New module

`src/workflows/mutation_distribution_analysis/conditional_poisson.py`, with a
matching `test/workflows/mutation_distribution_analysis/test_conditional_poisson.py`
written in the same session. Both call sites import from here; the module has no
dependency on either.

Notation used throughout, matching Chen & Liu (1997): `n` positions,
sample size `k`, target inclusion probabilities `pi` (length `n`, summing to
`k`), working odds `w` (length `n`), and
`R(k, C) = sum over k-subsets B of C of prod_{i in B} w_i`.

### 2.1 `inclusion_probabilities(counts, sample_size)`

Proportional target with **iterative take-all capping**: set
`pi = sample_size * counts / counts.sum()`; while any `pi_i >= 1`, fix those to
`1`, remove them, decrement `sample_size` by how many were fixed, and
re-proportion the remainder; repeat until feasible. Matches R
`sampling::inclusionprobabilities()`.

Required for the 9 genes with `max(k*p) > 1` at call site 2. Returns the full
length-`n` vector including the capped `1.0` entries; downstream code must
strip them before calibration (§2.3) and force them into every sample.

Guards: `counts` non-negative and not all zero; `0 <= sample_size <= n_positive`
where `n_positive` is the number of non-zero counts; raise `ValueError`
otherwise.

### 2.2 `log_normalising_constants(working_odds, sample_size)`

Returns `(log_r_excluding, log_r_full)` where
`log_r_excluding[j] = log R(k-1, S\{j})` for every `j`, and
`log_r_full = log R(k, S)`.

Method: two Gail-Lubin-Rubinstein tables,
`forward[m][j] = R(j, {1..m})` and `backward[m][j] = R(j, {m..n})`, each built
by `R(j, C + {x}) = R(j, C) + w_x * R(j-1, C)`. Combine by Chen & Liu's
Proposition 1(c):

```text
R(k-1, S\{j}) = sum_{a=0..k-1} forward[j-1][a] * backward[j+1][k-1-a]
```

**Numerical requirements, both non-optional:**

- Scale each table row by its own maximum and accumulate the log of the
  divisor. The scale of a convolution term depends only on the row indices
  `j-1` and `j+1`, never on `a`, so it factors out of the sum exactly and no
  log-space summation is needed inside the convolution.
- The `R(0, ·) = 1` entry must be carried in the **current row's scaled
  units** — i.e. copied from the previous row — not reset to a literal `1.0`.
  Resetting it silently corrupts the entire table. This mistake was made and
  caught during planning; it produced `log R` errors of order `1e0` and a
  calibration that failed to converge, with no exception raised.

Cost: `O(n*k)` time, `O(n*k)` memory (~2 MB at `n=2721, k=88`), ~22 ms per
call at that size.

### 2.3 `calibrate_working_odds(target, tolerance=1e-10, max_iterations=500)`

Chen, Dempster & Liu (1994) inversion, eq. (11) of Chen & Liu (1997). Per
iteration:

1. `log_r_excluding, log_r_full = log_normalising_constants(w, k)` with
   `k = round(target.sum())`
2. `log achieved = log w + log_r_excluding - log_r_full`
3. if `max|achieved - target| < tolerance`: return `w`
4. `log w <- log target - log_r_excluding`, then subtract the mean of `log w`

Step 4 is eq. (11): `w_j` proportional to `pi_j / R(k-1, S\{j})`. CDL pin the
constant by holding `w_n = pi_n`; we pin it by mean-centring `log w` instead.
**This is not a deviation from the method** — the map is provably insensitive to
the scale of its input, because every `R(k-1, S\{j})` has the same degree
`k-1` and so rescales identically, leaving eq. (11)'s ratio unchanged.
Mean-centring keeps the recursion in range regardless of which position happens
to be last. Document this in the docstring, with the reason.

Initialise `w = target / (1 - target)` (the Hájek starting point).

**Raise `RuntimeError` if `max_iterations` is reached without meeting
`tolerance`.** No convergence proof is known for this iteration
(`sampler_inclusion_probability_fix.md` §4.5), so the guarantee has to be a
checked runtime property rather than an assumption. Never return an
unconverged result.

Preconditions: all `target_i` strictly in `(0, 1)` — callers strip capped and
zero entries first. Raise `ValueError` if violated.

Expected cost at the real pools: 8–9 iterations, ~0.2 s. Worst observed
(`n=400`, `max pi = 0.847`): 95 iterations, 0.30 s.

### 2.4 `sample_positions(working_odds, sample_size, rng)`

Rejective draw: Bernoulli(`w/(1+w)`) across all positions, accept iff exactly
`sample_size` heads. Return the accepted index array.

Acceptance is `~1/sqrt(2*pi*k)` — 4.3% at `k=88`, ~23 attempts, ~0.17 ms. Cap
attempts (say 10 000) and raise rather than spin forever. Rejection-free
sequential alternative recorded in `sampler_inclusion_probability_fix.md` §4.4
if this ever proves inadequate; it is not expected to.

### 2.5 `PositionSampler` — the only stateful piece

Small class holding `(target, working_odds, sample_size)` for one pool so
calibration is paid once and amortised over draws, with a `draw(rng)` method.
Justified over free functions only because both call sites draw thousands of
samples from the same calibrated weights; keep `__init__` to validation plus
one `calibrate_working_odds` call.

---

## 3. Call-site changes

### 3.1 `mutation_distance_analysis.compute_random_distances`

The pool is shared across genes; only `k` varies (42 distinct values across 999
genes). Cache one `PositionSampler` **per distinct `k`** — calibration is
`k`-dependent, so this is neither a single precomputation nor a per-draw cost.
42 calibrations × ~0.2 s ≈ 10 s per species.

Delete `_sample_positions_blocking` and the "mathematically equivalent" note at
lines 50–55, which is true but describes the wrong equivalence.

### 3.2 `baseline_sampler.generate_baselines_fasta`

Per-gene, since `_filter_applicable` makes both `p` and `k` gene-specific.
`applicable` is already computed once per gene at line 202, so the sampler is
constructed there. ~999 genes × ~0.3 s ≈ 5 min per species — acceptable for a
one-off regeneration; flag it in the console output since this is a
long-running workflow (`io.py` formatted output is appropriate here).

**Factorise the draw.** The current sampler draws uniformly over *rows*, which
implies `P(position = i) = n_i/N` and
`P(new_base = b | position = i) = n_{i,b}/n_i`. That joint factorises exactly,
and the conditional does not depend on the position design (which uses only
counts). So:

1. draw positions with `PositionSampler`, weights proportional to `n_i`;
2. set `source_base = wildtype[position]` — **no sampling needed.** Verified
   across all 999 ara genes that post-filter every position has exactly one
   distinct `source_base` (max = 1), because `_filter_applicable` defines it
   that way. Up to 3 distinct `new_base` values per position;
3. draw `new_base` uniformly among applicable rows at that position.

`_apply_mutations`'s assertion at line 129 then holds by construction.

Handle capped positions (`pi_i = 1`) by forcing them into every sample and
calibrating on the remainder with `k` decremented, mirroring
`sampling::UPmaxentropy`.

### 3.3 Docstring corrections

- `baseline_sampler.py:68-69` — the "preserves the empirical positional PMF"
  claim. Replace with the selection- vs inclusion-probability distinction and a
  pointer to the new module.
- `mutation_distance_analysis.py:50-55` — the "mathematically equivalent" note.

---

## 4. Tests

`test/workflows/mutation_distribution_analysis/test_conditional_poisson.py`,
unittest, Arrange-Act-Assert. The existing
`test_sampler_calibration.py` documents the *old* behaviour — keep it as a
record of the bug, but re-label it so it is not read as a specification.

1. **Recursion vs enumeration.** `log_normalising_constants` against
   brute-force `itertools.combinations` over `(n,k)` in
   `{(4,2), (9,3), (12,5)}` with odds spanning two orders of magnitude. Assert
   max `log R` error `< 1e-12`. Achieved `2.7e-15` in the planning scratch.
2. **Calibration exactness.** `calibrate_working_odds` then exact enumerated
   inclusion probabilities; assert `max|achieved - target| < 1e-10`. Include
   the `n=9, k=3, max k*p = 0.946` feasibility-edge case.
3. **Scale invariance.** Calibrating the same target twice from inputs
   differing by a constant factor gives working odds with identical *ratios*.
   Guards the mean-centring deviation in §2.3.
4. **Non-convergence raises.** `max_iterations=1` on a case needing more must
   raise `RuntimeError`, not return silently.
5. **Take-all capping.** `inclusion_probabilities` on the 9 known-infeasible
   genes: output sums to `k`, no entry exceeds 1, capped entries are exactly
   1.0. Include a synthetic cascade where capping one position pushes a second
   over 1.
6. **Sampler marginals on a real pool.** `sample_positions` on the unfiltered
   ara pool, `k=88`; assert empirical inclusion frequencies match `pi` within
   Monte-Carlo error. Cross-check against **Sampford** on the same pool —
   Sampford closely approximates the maximum-entropy design, so agreement is
   strong independent evidence. Mark slow; keep the replicate count low enough
   to stay in the normal test run, or skip by default with an env flag.
7. **Determinism** under a fixed `numpy` `default_rng` seed.
8. **Edges.** `k = 0`, `k = 1`, `k = n` (every `pi_i = 1`), `k > n` raises,
   zero-count positions excluded rather than assigned `pi = 0` inside the
   recursion.
9. **Factorisation.** In `baseline_sampler`, that `source_base` is uniquely
   determined post-filter for every gene, and that `new_base` frequencies match
   the applicable-row frequencies.

**Optional cross-check, not a runtime dependency:** R
`sampling::UPmaxentropy` / `UPMEpiktildefrompik` on a small pool, comparing
working odds up to scale. R is at `/usr/bin/R` but the `sampling` package is not
installed. Worth doing once by hand and recording the numbers in a test
docstring rather than wiring rpy2 into `deepCREshap`.

---

## 5. Regeneration and reporting

Both are required before submission; the second is the larger unknown.

1. **Figure 3 distance panels.** Regenerate; report old-vs-new KS and
   Wasserstein for real-vs-null. Expected shift is ~6–12% of the reported
   effect, reducing it (`sampler_inclusion_probability_fix.md` §2).
2. **Optimised-vs-random comparison** (`evaluate_sequences.py`,
   `plot_optimization_vs_random.py`). **This number does not exist yet.** Call
   site 2 has median `max(k*p)` of 0.229 (ara) / 0.414 (zea) — far more skewed
   than call site 1 — so the bias is larger and its effect on this comparison
   is unquantified. Quantify it *before* deciding how to present the result.

Keep the pre-fix figures and numbers until both are reproduced, so the
old-vs-new comparison can be stated rather than asserted.

---

## 6. Order of work

1. `conditional_poisson.py` §2.1–2.4 with tests 1–5, 7–8. Stop and review:
   this is where every correctness risk lives.
2. `PositionSampler` §2.5, test 6 on the real pool.
3. Rewire call site 1 (§3.1), regenerate figure 3 distance panels, report
   old-vs-new.
4. Rewire call site 2 (§3.2), including the factorisation and test 9.
5. Regenerate the optimised-vs-random comparison and report it (§5.2).
6. Docstring corrections (§3.3).

Step 1 is self-contained and verifiable against enumeration; steps 3 and 5 are
where the publication numbers change and each needs a decision from Gernot
before anything is redrawn for the paper.

---

## 7. Known risks

- **No convergence or existence proof** for the CDL inversion is known to us.
  Mitigated by the runtime assertion (§2.3), not eliminated. Tillé (2006) §5.6
  is the place to look if a reviewer asks.
- **Runtime at call site 2** is ~5 min/species. Acceptable once; annoying if the
  baselines get regenerated repeatedly. Cache calibrated odds to disk keyed by
  gene if that becomes a problem — not worth doing up front.
- **The direction of the correction is data-dependent.** "Flatter null → longer
  distances" holds for these pools only because the heavy positions are
  spatially clustered (top-10 weights at positions 1114–1163). Do not state it
  as a general property in the paper.

---

## 8. Implementation record (2026-08-06)

### 8.1 One correction to the plan: the odds' scale is not free at draw time

§2.3 mean-centres `log w` and §2.4 draws rejectively from the result. Those two
are incompatible as written. The **design** is invariant to a global factor on
`w` — that is what licenses the mean-centring — but the **rejective draw is
not**: it flips *unconditional* coins with bias `w_i/(1+w_i)` and keeps only
the draws with exactly `k` heads, so its acceptance rate depends on how far
`sum_i w_i/(1+w_i)` sits from `k`. At the real pool the mean-centred odds
accepted **zero** draws in 10 000 attempts.

Fix: a new `scale_odds_to_sample_size(w, k)` finds the unique `c > 0` with
`sum_i (c*w_i)/(1 + c*w_i) = k` by bisection on `log c` (the left side is
continuous and strictly increasing from 0 to `n`, so the root exists and is
unique for `0 < k < n`) and returns `c*w`. `PositionSampler` applies it once
after calibration; `sample_positions` applies it defensively when handed odds
at the wrong scale. **The sampled distribution is unchanged** — only the number
of attempts needed to obtain it. Acceptance is then the Poisson-binomial mode
probability as §2.4 predicted.

### 8.2 Deviations of substance, all deliberate

- **`_sample_mutations_blocking` was deleted too**, not only
  `_sample_positions_blocking`. §3.2 rewires `generate_baselines_fasta` but
  `sample_baseline_sequence` — public API — used the same biased draw. Leaving
  it would keep a biased entry point reachable.
- **The old samplers live on in `test_sampler_calibration.py`** as
  `successive_sample_rows` / `successive_sample_positions`. §3.1 deletes them
  from `src` while §4 keeps the test as a record of the bug; inlining them into
  the test satisfies both. The module docstring now opens with "Historical
  record of a fixed bug — not a specification", and a closing test shows the
  replacement capping the same pool.
- `TestSamplePositionsBlocking` in `test_mutation_distance_analysis.py` was
  removed; its coverage moved to `test_conditional_poisson.py`.
- `inclusion_probabilities` accepts an empty or all-zero pool when
  `sample_size == 0`, so `k = 0` genes need no special case at the call sites.
- The README's sampling description and decision table were stale in the same
  way the docstrings were; both were corrected.

### 8.3 Verified

Every prediction in §2 and §4 reproduced. Calibration 0.21 s and draw 0.20 ms
at `n = 2721, k = 88`; 42 distinct `k` values at call site 1, 7.4 s total; call
site 2 median `max(k*p)` 0.229, max 1.088, 3 infeasible ara genes, ~5.8 min per
species. Marginals on the real pool match `k*p` (worst standardised deviation
3.74 over 2721 positions at 5000 replicates) and agree with an independent
Sampford sampler at 20 000 replicates. Tests 1–9 all implemented and passing;
the real-pool pair is gated behind `RUN_SLOW_SAMPLER_TESTS=1` (65 s).

### 8.4 §5.1 — distance null, old vs new

`n_per_gene = 10`, seed 42, both pools. Real-vs-null, and the two nulls against
each other:

| | ara | zea |
| --- | --- | --- |
| mean distance, real | 22.890 | 23.139 |
| mean distance, old null | 25.806 | 25.393 |
| mean distance, new null | 25.581 | 25.175 |
| real vs old: KS / Wasserstein | 0.0573 / 3.685 | 0.0785 / 4.900 |
| **real vs new: KS / Wasserstein** | **0.0516 / 3.471** | **0.0712 / 4.647** |
| old null vs new null: KS / Wasserstein | 0.0061 / 0.232 | 0.0074 / 0.322 |

The effect shrinks by 10.0% (ara) / 9.3% (zea) on KS and 5.8% / 5.2% on
Wasserstein, in the predicted direction — §2 of the fix note estimated 6–12%.
Both remain overwhelmingly significant (`p < 1e-170`). **Figures not
regenerated**; only the statistics were recomputed.

### 8.5 Not done

**§5.2, the optimised-vs-random comparison.** It needs the baselines
regenerated (~5 min/species) and then MSR-model inference over both FASTAs via
`evaluate_sequences.py`, whose model paths point outside this repo. Left for a
deliberate run rather than done in passing, as §6 asks.
