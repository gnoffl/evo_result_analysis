# Baseline Sampler — Implementation Plan (Steps 2 & 3 of mutation_distribution_analysis)

## Context

The conceptual plan (`src/workflows/mutation_distribution_analysis/plans/plan.md`) lays out a four-step pipeline. Step 1 (extracting the SNP pool) is already implemented in `src/workflows/mutation_distribution_analysis/mutation_pool.py` and produces `MutationPool` objects with three flat tables:

- `mutations`: long-format `DataFrame[gene_id, position, source_base, new_base]` (~75k rows)
- `gene_stats`: `DataFrame[gene_id, n_mutations, initial_fitness, final_fitness]`
- `references`: `dict[gene_id, wildtype_sequence]`

The original `plan.md` proposed Step 2 = explicit positional + conditional PMFs, Step 3 = sample positions then bases. We are **superseding that** with a simpler approach that uses the pool directly as a sample space:

> For a target gene with wildtype `W`, only mutations in the pool whose `source_base` matches `W[position]` are biochemically applicable. Uniform sampling over those rows reproduces the empirical joint distribution (position × source_base × new_base) implicitly — no PMFs needed. Position-blocking after each draw enforces "no two SNPs at the same site".

This document plans the single new module that delivers both filtering (Step 2) and sampling (Step 3), plus a CLI driver that batches across all genes and writes a FASTA file of baselines.

## Approach

One new module at `src/workflows/mutation_distribution_analysis/baseline_sampler.py` with three layers:

1. **Pure helpers** — filtering, position-blocking sampler, mutation application.
2. **Per-gene API** — `sample_baseline_sequence(pool, gene_id, n_mutations, rng) -> str`.
3. **Batch driver + CLI** — generates `n_per_gene` baselines for every gene in the pool and writes them to a FASTA file.

The existing empty `mutation_distribution.py` is left untouched (can be removed later).

### Algorithm (per gene)

Given target `gene_id`, wildtype `W = pool.references[gene_id]`, count `k`:

1. **Filter applicable mutations** — keep `pool.mutations` rows where `W[row.position] == row.source_base`. Vectorised: build a numpy array of `W[position]` over the position column, compare to `source_base` column, boolean-index the DataFrame. No self-exclusion (rows whose `gene_id == target` are kept).
2. **Iterative position-blocking sample**, `k` times:
   - Uniformly sample one row from the working applicable set.
   - Append `(position, source_base, new_base)` to the result.
   - Drop **all** rows at that `position` from the working set.
3. **Apply mutations** — `wildtype` copied into a `list[str]`; for each `(pos, src, new)` overwrite index `pos` with `new`. Sanity-assert `wildtype[pos] == src` (always true by construction; assertion guards regressions). Return `"".join(...)`.

Uniform-over-rows is the correct weighting: it preserves the empirical (position, source_base, new_base) frequencies — a position that appears 50× in the filtered set is 50× more likely to be sampled than a position that appears once, matching the empirical positional PMF directly.

### Batch driver

For each `gene_id` in `pool.references`:

- `k = pool.gene_stats.loc[gene_id, 'n_mutations']` (default; could be overridden by future flags).
- Compute the applicable set once for this gene (shared across all `n_per_gene` draws — each draw operates on a fresh copy of the applicable DataFrame).
- If the **number of unique positions** in the applicable set is `< k`: print a warning naming the gene and the shortfall, skip the gene entirely, continue.
- Otherwise: draw `n_per_gene` independent baselines; write each as a FASTA record with header `>{gene_id}_baseline_{i:03d}` (zero-padded to width 3, supports up to 999 per gene). One line per sequence — no line wrapping. Records appended in `gene_id` iteration order.

### Public API (signatures only)

```python
def sample_baseline_sequence(
    pool: MutationPool,
    gene_id: str,
    n_mutations: int,
    rng: np.random.Generator,
) -> str
    # Raises KeyError if gene_id not in pool.references.
    # Raises ValueError if applicable unique positions < n_mutations.

def generate_baselines_fasta(
    pool: MutationPool,
    output_path: str,
    n_per_gene: int = 1,
    rng: np.random.Generator | None = None,
) -> None
    # Default rng: np.random.default_rng() if None.
    # Genes with insufficient applicable pool are warned + skipped.

def main() -> None
    # CLI entry point.
```

### Private helpers (separately tested)

```python
def _filter_applicable(
    mutations: pd.DataFrame, wildtype: str
) -> pd.DataFrame

def _sample_mutations_blocking(
    applicable: pd.DataFrame, k: int, rng: np.random.Generator
) -> list[tuple[int, str, str]]

def _apply_mutations(
    wildtype: str, mutations: list[tuple[int, str, str]]
) -> str
```

### CLI

```bash
python -m workflows.mutation_distribution_analysis.baseline_sampler \
    --input ./mutation_pools/ara_msr_max_single_gen1999_mutation_pool.json \
    --output ./baselines_ara_msr_max_single_gen1999.fasta \
    --n-per-gene 1 \
    --seed 42
```

Argparse layout mirrors `mutation_pool.py`:

- `--input` / `-i` (required): path to a `MutationPool` JSON.
- `--output` / `-o` (required): path to the FASTA output.
- `--n-per-gene` / `-n` (int, default `1`): baselines per gene.
- `--seed` / `-s` (int, default `42`): RNG seed; `np.random.default_rng(seed)` is passed to `generate_baselines_fasta`.

`parse_args` validates the input path exists, mirrors the pattern in `mutation_pool.py:254`. `main` creates output directories, calls `MutationPool.load`, then `generate_baselines_fasta`.

## Files to create / modify

| File | Action | Purpose |
| --- | --- | --- |
| `src/workflows/mutation_distribution_analysis/baseline_sampler.py` | create | helpers + per-gene API + batch driver + CLI |
| `test/workflows/mutation_distribution_analysis/test_baseline_sampler.py` | create | unit tests (unittest, AAA, mocks) |
| `src/workflows/mutation_distribution_analysis/mutation_distribution.py` | leave empty | unused; do not delete in this PR |

## Existing code reused (no modification)

- `MutationPool` (load + DataFrames + references dict) — `src/workflows/mutation_distribution_analysis/mutation_pool.py:142`
- `analysis.utils.io.print_status`, `print_section_header`, `print_subsection` — for CLI/warning output consistent with `mutation_pool.py`

## Tests

All in `test/workflows/mutation_distribution_analysis/test_baseline_sampler.py`. Style follows `test_mutation_pool.py`: `unittest.TestCase`, AAA, small synthetic DataFrames built inline, no real I/O except via `tempfile.TemporaryDirectory`.

Helper for the tests: a `_make_pool(mutations_records, refs, gene_stats_records)` builder that returns a `MutationPool` with the supplied tables — avoids re-running `from_summarized_json` in each test.

**`_filter_applicable`**

- `test_keeps_only_matching_source_base` — pool of 4 rows, wildtype agrees with 2 of the source_bases; result has those 2 only.
- `test_returns_empty_when_no_matches` — no positions agree; result is an empty DataFrame with the right columns.
- `test_handles_duplicate_positions` — pool has 3 rows at the same position, all matching; all 3 are retained (multiplicity preserved).

**`_sample_mutations_blocking`**

- `test_returns_k_mutations` — seeded RNG, k=3 on a 20-row applicable set; result has length 3.
- `test_no_duplicate_positions` — repeat the above and assert all positions in the result are unique.
- `test_deterministic_under_seed` — two calls with the same seed produce identical results.
- `test_raises_when_unique_positions_below_k` — applicable set has only 2 unique positions but k=3; raises `ValueError`.
- `test_zero_k_returns_empty_list` — k=0 returns `[]` without consuming the RNG state.

**`_apply_mutations`**

- `test_replaces_specified_positions_only` — wildtype `"AAAA"`, mutations `[(1, "A", "T"), (3, "A", "G")]` → `"ATAG"`.
- `test_returns_wildtype_when_no_mutations` — empty list → wildtype unchanged.
- `test_asserts_source_base_matches_wildtype` — mismatching source_base raises (defensive check).

**`sample_baseline_sequence`**

- `test_end_to_end_on_small_pool` — 3-base wildtype, hand-picked pool, seeded RNG; assert the returned string differs from wildtype at exactly k positions and every change matches an entry in the pool.
- `test_raises_when_gene_id_missing` — `KeyError` containing the gene id.
- `test_raises_when_pool_too_thin_for_k`.

**`generate_baselines_fasta`**

- `test_writes_expected_number_of_records` — pool with 2 genes, `n_per_gene=3` → 6 records in the FASTA.
- `test_header_format_zero_padded` — headers are exactly `>{gene_id}_baseline_000`, `_001`, ... (assert against the literal strings).
- `test_skips_gene_when_pool_too_thin` — one gene has only 1 applicable unique position but k=5; the FASTA contains only the other gene's records; a warning was issued (capture via `mock.patch` on `print_status` and assert it was called with a string containing the skipped gene id).
- `test_deterministic_under_seed` — two runs with identical seed produce byte-identical FASTA files.
- `test_each_baseline_differs_from_wildtype_at_k_positions` — parse the written FASTA, compare against `pool.references`, count diffs.

**CLI** (`main`)

- `test_cli_invokes_generate_baselines_fasta` — `mock.patch` `MutationPool.load` and `generate_baselines_fasta`; invoke `main()` with a constructed `sys.argv`; assert call args match (input path, output path, `n_per_gene`, seeded RNG).

Ensure the test output directory used by `tempfile.TemporaryDirectory` is cleaned automatically; no persistent fixtures are written.

## Verification

End-to-end smoke run against the existing pool:

```bash
python -m workflows.mutation_distribution_analysis.baseline_sampler \
    --input  ./src/workflows/mutation_distribution_analysis/mutation_pools/ara_msr_max_single_gen1999_mutation_pool.json \
    --output ./baselines_ara_msr_max_single_gen1999.fasta \
    --n-per-gene 1 \
    --seed 42
```

Manual checks to perform on the produced FASTA:

- Record count equals `len(pool.references) - (number of skipped genes)`. Skipped count is announced by `print_status` lines on stderr.
- Headers are unique and follow `>{gene_id}_baseline_000`.
- For a hand-picked gene (e.g. `5_AT5G38895_...`): parse its baseline, diff against `pool.references[gene_id]`, confirm exactly `gene_stats.loc[gene_id, 'n_mutations']` positions changed and every change agrees with at least one row in `pool.mutations` (matching `position` and `source_base` matches the wildtype base).
- All output sequences have the same length as the corresponding wildtype.
- Re-running with the same seed produces a byte-identical FASTA.

Run the unit suite:

```bash
pytest test/workflows/mutation_distribution_analysis/test_baseline_sampler.py -v
```

Once green and the smoke check passes, the four-step pipeline is functionally complete; Step 4 (validation against the empirical distribution) becomes a thin consumer of the FASTA + pool and can be planned separately.
