# Speed Up `compute_random_distances`

## Context

`mutation_distance_analysis.py` runs end-to-end in unacceptable time. Profiling the call graph for `compute_random_distances` shows the cost is **almost entirely in `_sample_mutations_blocking`**, which is called ~10,000 times (≈1,000 genes × `n_per_gene=10` replicates) and each call runs a `k ≈ 90`-step inner loop whose body is a boolean DataFrame filter:

```python
# baseline_sampler.py:105
working = working.loc[working["position"] != position]
```

For 10,000 calls × ~90 iterations that's **~900,000 pandas filter operations** on a multi-thousand-row DataFrame. Pandas overhead per `.loc[mask]` is ~50–100 μs even before the O(n) work, so this single line dominates the runtime by orders of magnitude.

The fix is a one-function rewrite of the random sampler in numpy. The algorithmic distribution is preserved exactly — only the per-call cost changes.

## Why a fast equivalent exists

The current sampler does "uniform-over-rows + position-blocking". Mathematically that is *sequential weighted sampling without replacement over unique positions*, with each position weighted by its row count in `pool.mutations`:

- P(position p drawn first) = `count(p) / sum(counts)`
- After p is drawn, all rows with position p are excluded; remaining draws are uniform over the remaining rows ⇒ weighted-without-replacement renormalization.

`np.random.Generator.choice(unique_positions, size=k, replace=False, p=weights)` implements exactly this distribution in C, with zero Python overhead per inner iteration.

We **don't need** `source_base` / `new_base` for this analysis (per the existing plan: "this task is distance-only"), so collapsing the pool from N rows to ~n_unique_positions weighted entries is lossless.

## Approach

All changes are confined to `src/workflows/mutation_distance_analysis/mutation_distance_analysis.py`. **No modifications to `baseline_sampler.py` or `mutation_pool.py`** (those are kept intact because the sibling workflow needs the row-level `(pos, src, new)` triples).

### 1. Pre-compute the positional PMF once

In `compute_random_distances`, before the gene loop, collapse `pool.mutations["position"]` into a unique-position array + normalized weight vector:

```python
pos_arr = pool.mutations["position"].to_numpy()
unique_positions, counts = np.unique(pos_arr, return_counts=True)
weights = counts / counts.sum()
n_unique = unique_positions.size
```

This runs once (not per gene).

### 2. Replace the per-call `_sample_mutations_blocking` with a numpy call

Add a small private helper local to this module:

```python
def _sample_positions_blocking(
    unique_positions: np.ndarray,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw k distinct positions ~ weighted-without-replacement from the PMF."""
    return rng.choice(unique_positions, size=k, replace=False, p=weights)
```

Call it once per `(gene, replicate)` instead of stepping through `_sample_mutations_blocking`'s pandas loop. The import of `_sample_mutations_blocking` from `baseline_sampler` is removed from this file.

### 3. Replace incremental `Counter +=` with a single bincount at the end

The current code does `distances += calculate_mutation_distances_single_gene(...)` inside the inner loop (Python-level Counter merge, ~10,000 times). Replace with a flat numpy accumulator and a single `np.unique` at the end:

```python
all_distances: list[np.ndarray] = []
for gene_id in tqdm(pool.references.keys(), desc="Random per-gene draws"):
    k = int(gene_stats_by_id.loc[gene_id, "n_mutations"])
    if k < 2:
        continue
    for _ in range(n_per_gene):
        pos = _sample_positions_blocking(unique_positions, weights, k, rng)
        pos.sort()
        all_distances.append(np.diff(pos))

flat = (
    np.concatenate(all_distances)
    if all_distances
    else np.array([], dtype=int)
)
vals, cnts = np.unique(flat, return_counts=True)
return Counter(dict(zip(vals.tolist(), cnts.tolist())))
```

Note: `calculate_mutation_distances_single_gene` (`analyze_mutations.py:378`) is no longer used in the random path — it's kept in the real path (`compute_real_distances`) where it's called only once per gene.

### Optional (only if still too slow after the above)

If 10,000 numpy calls still aren't fast enough, replace the per-replicate `rng.choice` with **batched weighted sampling without replacement via the Efraimidis–Spirakis / Gumbel-top-k trick** — draws all `n_per_gene` replicates of one gene in a single vectorized op:

```python
# all n_per_gene replicates for one gene, drawn at once
keys = -np.log(rng.uniform(size=(n_per_gene, n_unique))) / weights
idx = np.argpartition(keys, k - 1, axis=1)[:, :k]
pos_batch = unique_positions[idx]      # shape (n_per_gene, k)
pos_batch.sort(axis=1)
diffs = np.diff(pos_batch, axis=1)     # shape (n_per_gene, k-1)
all_distances.append(diffs.ravel())
```

This is mathematically equivalent (Efraimidis–Spirakis is the standard weighted-without-replacement algorithm) and drops the outer replicate loop. Recommend skipping this on the first pass — the `rng.choice` change alone should be enough.

## Critical files

| File | Change |
| --- | --- |
| `src/workflows/mutation_distance_analysis/mutation_distance_analysis.py` | Rewrite `compute_random_distances`; add local helper `_sample_positions_blocking`; drop import of `_sample_mutations_blocking` and (in the random path) `calculate_mutation_distances_single_gene` |
| `test/workflows/mutation_distance_analysis/test_mutation_distance_analysis.py` | Tests for `compute_random_distances` keep invariant assertions (total mass = `sum(k_g - 1) * n_per_gene`, all distances ≥ 1, determinism under fixed seed). Any test that asserts specific sample positions from the old algorithm needs to be re-keyed or relaxed to invariant-only — the new sampler is statistically equivalent but produces different specific samples from a given seed. |

`baseline_sampler.py`, `mutation_pool.py`, and `compute_real_distances` are **untouched**.

## What changes for the user

- **Behavior:** statistically identical null distribution (same PMF, same algorithm class). KS / Wasserstein stats will shift by an amount within Monte-Carlo noise of `n_per_gene=10` but the qualitative comparison is unchanged.
- **Determinism:** the new implementation is still deterministic under a fixed `--seed`, but a re-run with the same seed under the *new* code will not match a previous run under the *old* code (different sampling primitives consume RNG differently). Re-running the new code twice with the same seed gives byte-identical outputs, matching the existing verification step.
- **Output files:** unchanged (same 2 plots + 1 stats file per pool, same filenames).
- **Estimated speedup:** ~50–500× on the random-sampling phase, which is the dominant phase. Multi-minute runs should drop to seconds.

## Verification

1. **Run both species** as in the original plan and confirm wall-clock improvement:

   ```bash
   time python -m workflows.mutation_distance_analysis.mutation_distance_analysis \
       --pool ./src/workflows/mutation_distribution_analysis/mutation_pools/ara_msr_max_single_gen1999_mutation_pool.json \
       --output-dir ./src/workflows/mutation_distance_analysis/results/ \
       --name ara_msr_max_single --n-per-gene 10 --seed 42
   ```

2. **Sanity arithmetic (invariant, not byte-equality):**
   `sum(random_counter.values()) == sum(pool.gene_stats["n_mutations"] - 1) * n_per_gene` (excluding genes with `k < 2`).
3. **Determinism:** two runs with the same `--seed` produce byte-identical stats files and visually-identical plots.
4. **Distributional sanity:** overlay plots still show the random line tracking the real bars (since the sampler still honours the empirical positional PMF — only the implementation changed).
5. **Unit tests** (in the conda env `deepCREshap`):

   ```bash
   pytest test/workflows/mutation_distance_analysis/ -v
   ```

6. **Cross-check against the old implementation** on a small synthetic pool (a few hundred mutations, a few genes, `n_per_gene=1000`): both samplers' resulting distance histograms should agree to within Monte-Carlo noise. Optional one-off sanity check; not a permanent test.
