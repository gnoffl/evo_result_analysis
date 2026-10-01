# Where the greedy algorithm becomes faster than the genetic algorithm (90 mutations)

Question: at a mutation budget of 90, at which population size / generation count
does the genetic algorithm (GA) take longer than the greedy algorithm?

All numbers are **CPU only** — the GPU cells of the benchmark shared devices between
15 concurrent tasks, so their wall times are contended upper bounds.

Source: `benchmarks_mixed_hardware/runtime_summary.csv` and `benchmarks_mixed_hardware/scaling_fits.csv`.

## Scope of the question

A mutation budget of 90 constrains **only the greedy side**. The greedy algorithm's
mutation cap *is* its number of optimization steps, hence its only cost driver. The
GA's runtime is independent of the cap — that is exactly what the `maxmut2999`
control cell shows — so the GA side is a function of population size `p` and
generation count `g` alone.

Notation used below: `t` is the optimization wall time in seconds, `x` the swept
parameter, and `t = c * x ** k` the fitted power law with coefficient `c` and
exponent `k` (`k = 1` is linear scaling).

## Step 1 — greedy runtime at a cap of 90

90 lies inside the measured range of caps (5, 10, 20, 40, 100), so the runtime is
interpolated locally in log-log space between the measured medians at 40 and 100
rather than taken from the global fit, whose R² is only 0.94 in the natural regime.

| regime | t(cap 40) | t(cap 100) | local exponent | **t(cap 90)** |
|---|---|---|---|---|
| natural | 21.4 s | 35.5 s | 0.55 | **≈ 33 s** |
| unconstrained | 273.5 s | 753.1 s | 1.11 | **≈ 670 s** |

## Step 2 — GA configuration reaching that runtime

Inverting the quotable power laws of `scaling_fits.csv`, with the parameter that is
not swept held at its sweep pivot (population sweep runs at `g = 1000`, generation
sweep at `p = 100`):

| regime | swept | other held at | fit | crossover |
|---|---|---|---|---|
| natural | generations | p = 100 | c = 1.478, k = 0.922, R² = 0.999 | **≈ 30 generations** |
| natural | population | g = 1000 | c = 22.36, k = 0.789, R² = 0.978 | **≈ 2 individuals** |
| unconstrained | generations | p = 100 | c = 0.403, k = 0.894, R² = 0.985 | **≈ 4000 generations** |
| unconstrained | population | g = 1000 | R² = 0.656 | *not quotable, see below* |

## Reading

- **Natural regime: the greedy algorithm essentially always wins.** The GA passes
  33 s already at ~30 generations (p = 100), or at ~2 individuals (g = 1000) — both
  far below any configuration that would actually be run. Expressed as evaluation
  budgets the two routes agree to within a factor of ~2 (≈3000 vs ≈1700
  evaluations), so the estimate is self-consistent.
- **Unconstrained regime: the GA stays ahead over the useful range.** At p = 100 it
  only reaches 670 s at ~4000 generations, an evaluation budget of ~4·10⁵.
- **The unconstrained population crossover cannot be quoted.** That curve is not
  monotonic (the runtime *falls* from p = 25 to p = 50) and its power law has
  R² = 0.66, below the `POWER_LAW_R_SQUARED_FLOOR` of 0.95; it is the curve
  supplementary figure S2 panel A refuses to fit. Inverting it anyway gives ~1700
  individuals, while extrapolating the local 100 → 200 slope gives ~700. The two
  disagree by 2.4x and both lie far outside the measured range (max 200). All that
  can be said is "well beyond 200, order 10³". Pinning this down needs benchmark
  cells above p = 200 in the unconstrained regime.

## Caveat

The two mutation regimes differ by a large factor — the GA is ~4x slower per
evaluation under the natural constraint, the greedy algorithm ~20x slower — so these
crossovers are regime-specific and must not be averaged into one number.
