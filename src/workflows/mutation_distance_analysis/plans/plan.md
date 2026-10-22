# Mutation-Distance Random Baseline — Implementation Plan

## Context

Inter-mutation distance distributions for the **real** EA mutations are already computed in `src/analysis/mutations/analyze_mutations.py:421` (`calculate_mutation_distances`): `np.diff(sorted(positions))` per gene, aggregated into a `Counter` across all ~1000 genes.

We want a **null distribution** of distances drawn from the EA's own empirical mutation pool, to answer: *"are the EA's mutation positions spaced in a characteristic way, or does the spacing look the same as random draws from the EA's own positional preferences?"*

The existing `get_random_mutation_distributions()` (`analyze_mutations.py:433`) only does **uniform** sampling (90 positions from 3000, 10k replicates). That ignores the EA's strong positional preferences — we want the random baseline to honour them.

This task is **distance-only**: we ignore `source_base`/`new_base` content and the gene-specific wildtype (per the user's decision that sequence content is irrelevant for this analysis). The random sampler draws positions directly from the unfiltered empirical positional distribution embedded in `pool.mutations`.

The currently-empty file at `src/workflows/mutation_distance_analysis/mutation_distance_analysis.py` is the target.

---

## Approach

Reuse the existing `MutationPool` artefact and the position-blocking sampler from `baseline_sampler.py`. The pool already contains everything needed:

- `pool.mutations[['gene_id', 'position']]` — real per-gene mutation positions ➜ real distances.
- `pool.mutations[['position', 'source_base', 'new_base']]` as the **full sample space** for random draws (no wildtype filtering).
- `pool.gene_stats['n_mutations']` — the per-gene `k` to use for random draws.

No second input JSON is needed — the pool *is* the real data.

### Per-pool workflow

1. **Real distances.** Group `pool.mutations` by `gene_id`, sort positions, `np.diff`, sum into one `Counter`. (Reuse `calculate_mutation_distances_single_gene` at `analyze_mutations.py:378` to avoid re-implementing the per-gene step.)
2. **Random distances.** For each `gene_id` in `pool.references`:
   - `k = int(pool.gene_stats_by_id.loc[gene_id, "n_mutations"])`.
   - Repeat `n_per_gene` times (default **10**):
     - Call `_sample_mutations_blocking(pool.mutations, k, rng)` — **no `_filter_applicable` call**; the unfiltered pool is the sample space.
     - Take the `position` column from the returned tuples, sort, `np.diff`, accumulate into a single global `Counter`.
3. **Overlay plot.** Two figures per pool — full range and `≤200 bp` zoom — with the real histogram (bars) and the rescaled random distribution (line) on the same axes. Mirror the style of `plot_dist_hist` (`analyze_mutations.py:483`): random series is rescaled so its max equals the real series' max, matching the existing rescaling convention.
4. **Quantitative comparison.** Expand each `Counter` into a flat 1-D numpy array (each distance `d` repeated `count(d)` times) and compute:
   - `scipy.stats.ks_2samp(real_arr, random_arr)` → KS statistic + p-value.
   - `scipy.stats.wasserstein_distance(real_arr, random_arr)` → earth-mover distance.
   Write all three numbers to a plain `*_stats.txt`.

### Why no wildtype filtering

`baseline_sampler.py` filters `pool.mutations` to rows where `source_base == wildtype[position]` because it produces **applied sequences** and substitutions must be biochemically valid. Distances depend only on positions, and the user has stated that sequence identity is irrelevant for this analysis — so we skip filtering and treat the entire pool as one positional empirical distribution. This is the simpler and stricter "follow the empirical distribution" interpretation, and avoids reintroducing a per-gene dependency that would partially conflate the random sampler with the gene assignment.

### Helper reuse (no modification)

- `MutationPool.load` — `src/workflows/mutation_distribution_analysis/mutation_pool.py:212`.
- `baseline_sampler._sample_mutations_blocking` — `src/workflows/mutation_distribution_analysis/baseline_sampler.py:58`. Import the private symbol from the sibling workflow; algorithm is exactly what we need (uniform-over-rows + position blocking). No need to call `_filter_applicable` or `_apply_mutations`.
- `analyze_mutations.calculate_mutation_distances_single_gene` — `src/analysis/mutations/analyze_mutations.py:378`. Used for both the real path and the random path (per-replicate diff+Counter).
- `analysis.utils.io.print_status`, `print_subsection` — CLI logging consistent with the sibling workflow.

### Public API (signatures only)

```python
def compute_real_distances(pool: MutationPool) -> Counter:
    """np.diff on sorted positions per gene_id; aggregate into one Counter."""

def compute_random_distances(
    pool: MutationPool,
    n_per_gene: int,
    rng: np.random.Generator,
) -> Counter:
    """For each gene, draw n_per_gene position-sets of size k = gene_stats.n_mutations
    from the full pool (position-blocking), diff each, sum into one Counter."""

def counter_to_array(counter: Counter) -> np.ndarray:
    """Expand {d: count} into a flat array [d, d, ..., d] (count copies) for KS/W."""

def plot_overlay(
    real: Counter,
    random_dist: Counter,
    name: str,
    output_dir: str,
    output_format: str = "png",
    max_distance: Optional[int] = None,
) -> None:
    """One figure: bars = real, rescaled line = random. If max_distance is given,
    restricts both series to <= max_distance and suffixes the filename."""

def run_distance_analysis(
    pool_path: str,
    output_dir: str,
    name: str,
    n_per_gene: int = 10,
    seed: int = 42,
    output_format: str = "png",
) -> None:
    """End-to-end: load pool, compute both Counters, write 2 plots + 1 stats.txt."""

def main() -> None:
    """CLI entry point."""
```

### CLI

```bash
python -m workflows.mutation_distance_analysis.mutation_distance_analysis \
    --pool       ./src/workflows/mutation_distribution_analysis/mutation_pools/ara_msr_max_single_gen1999_mutation_pool.json \
    --output-dir ./src/workflows/mutation_distance_analysis/results/ \
    --name       ara_msr_max_single \
    --n-per-gene 10 \
    --seed       42 \
    --format     png
```

Argparse layout follows `baseline_sampler.py:220`. Required: `--pool`, `--output-dir`, `--name`. Defaults: `--n-per-gene 10`, `--seed 42`, `--format png`. Both species are run by invoking the CLI twice (mirrors `evaluate_sequences.py`'s style — that script hard-codes two species in `__main__`); no `--all` flag in this PR.

### Outputs per invocation (in `--output-dir`)

| File | Content |
| --- | --- |
| `mutation_distances_{name}_overlay.{format}` | Full-range bars (real) + rescaled line (random) |
| `mutation_distances_{name}_overlay_smaller.{format}` | Same overlay restricted to distance ≤ 200 |
| `mutation_distances_{name}_stats.txt` | KS statistic, KS p-value, Wasserstein distance (3 lines) |

---

## Files to create / modify

| File | Action | Purpose |
| --- | --- | --- |
| `src/workflows/mutation_distance_analysis/mutation_distance_analysis.py` | fill in (file exists empty) | helpers + driver + CLI |
| `src/workflows/mutation_distance_analysis/__init__.py` | create if missing | package marker |
| `test/workflows/mutation_distance_analysis/__init__.py` | create | test package marker |
| `test/workflows/mutation_distance_analysis/test_mutation_distance_analysis.py` | create | unit tests for the four helpers |

No modifications to `baseline_sampler.py` or `mutation_pool.py`.

---

## Tests (`unittest`, AAA, mocks where useful)

- `compute_real_distances`: a 2-gene synthetic pool with known positions → assert exact Counter contents; verify ordering insensitivity (mutations in any order).
- `compute_random_distances`:
  - Deterministic under a fixed seed (two runs equal).
  - Total mass equals `sum_g (k_g - 1) * n_per_gene` (each replicate contributes `k-1` distances).
  - All distances ≥ 1 (position-blocking guarantees uniqueness).
- `counter_to_array`: hand-built Counter `{2: 3, 5: 1}` → array `[2, 2, 2, 5]` (order of repeats is irrelevant for KS/Wasserstein, but assert sorted contents).
- `run_distance_analysis`: monkeypatch `MutationPool.load` to return a tiny synthetic pool, run end-to-end into a `TemporaryDirectory`, assert the 3 expected files exist and the stats file parses as 3 numeric lines.

The smoke check on the two real pools is run manually by the user (per `MEMORY.md` smoke-tests note).

---

## Verification

1. **Run both species** by hand:

   ```bash
   python -m workflows.mutation_distance_analysis.mutation_distance_analysis \
       --pool ./src/workflows/mutation_distribution_analysis/mutation_pools/ara_msr_max_single_gen1999_mutation_pool.json \
       --output-dir ./src/workflows/mutation_distance_analysis/results/ \
       --name ara_msr_max_single --n-per-gene 10 --seed 42

   python -m workflows.mutation_distance_analysis.mutation_distance_analysis \
       --pool ./src/workflows/mutation_distribution_analysis/mutation_pools/zea_msr_max_single_gen1999_mutation_pool.json \
       --output-dir ./src/workflows/mutation_distance_analysis/results/ \
       --name zea_msr_max_single --n-per-gene 10 --seed 42
   ```

2. **Visual checks** on the overlays:
   - Both series start at distance 1 and tail off.
   - Random line broadly resembles real bars (since the random sampler honours the EA's positional preferences); any systematic difference in tail or short-distance bins is the signal of interest.
3. **Stats file** contains finite numbers; KS statistic in `[0, 1]`; Wasserstein ≥ 0.
4. **Determinism**: re-running with the same `--seed` produces byte-identical stats files and visually-identical plots.
5. **Sanity arithmetic**: total counts in the random `Counter` equal `sum(pool.gene_stats['n_mutations'] - 1) * n_per_gene`.
6. **Unit tests**:

   ```bash
   pytest test/workflows/mutation_distance_analysis/ -v
   ```

Once green, the analysis is ready to feed into the comparison narrative alongside the existing `mutation_distances_*.png` real-only figures from `analyze_mutations.plot_mutation_distances`.
