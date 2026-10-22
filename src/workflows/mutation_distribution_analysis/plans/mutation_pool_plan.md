# Mutation Pool — Implementation Plan (Step 1 of mutation_distribution_analysis)

## Context

The conceptual plan in `plan.md` lays out a four-step pipeline for characterizing SNP mutations from the evolutionary optimization algorithm and generating biologically realistic random baselines. This document plans **Step 1 only**: extracting all mutations from the summarized-mutations JSON into a single sample-friendly artifact that the downstream PMF construction (Step 2) and baseline sampling (Step 3) will consume.

After this step is done, downstream code never has to touch the raw `MutationsGene` objects again — it works only with the persisted pool. This keeps a clean boundary between "extract & filter the raw data" and "estimate distributions / sample".

## Approach

A single new module at `src/workflows/mutation_distribution_analysis/mutation_pool.py` containing a `MutationPool` class plus a small CLI entry point. The existing empty `mutation_distribution.py` is left untouched — it will host Step 2 (PMF construction).

### `MutationPool` structure

Three components held in memory as separate tables and persisted together in one JSON file:

1. **`mutations`** — a `pd.DataFrame` with columns `gene_id, position, source_base, new_base`. One row per SNP across all genes (~75k rows total). This is the long-format table the PMF code will `groupby` over in Step 2.
2. **`gene_stats`** — a `pd.DataFrame` with columns `gene_id, n_mutations, initial_fitness, final_fitness, reference_length`. One row per gene. Drives the mutations-per-gene count PMF and lets you sanity-check selection. Initial fitness it the lowest fitness related to the 0 mutation reference sequence, final fitness is the highest fitness related to the pareto front element with the most mutations.
3. **`references`** — a `dict[str, str]` mapping `gene_id → wildtype sequence`. Needed later for looking up `source_base` at sampled positions of a target gene and for applying baseline mutations. Stays a dict (not a DataFrame) because lookups are by gene id and the strings are large.

Keeping the three tables flat and separate (rather than grouping by gene) makes the downstream PMF code simple: a single `.groupby(['position', 'source_base'])['new_base'].value_counts()` builds the conditional substitution distribution directly off `mutations`.

### JSON schema

One JSON file with three top-level keys:

```json
{
  "mutations": [
    {"gene_id": "5_AT5G38895_...", "position": 672, "source_base": "C", "new_base": "A"},
    {"gene_id": "5_AT5G38895_...", "position": 726, "source_base": "T", "new_base": "G"}
  ],
  "gene_stats": [
    {"gene_id": "5_AT5G38895_...", "n_mutations": 80, "fitness": 0.9999993, "reference_length": 3000}
  ],
  "references": {
    "5_AT5G38895_...": "TTGTCT..."
  }
}
```

The DataFrames are serialized as records (list of dicts) via `df.to_dict(orient='records')` and reloaded via `pd.DataFrame(data['mutations'])`. Verbose but human-readable.

### Class API

```python
@dataclass
class MutationPool:
    mutations: pd.DataFrame      # columns: gene_id, position, source_base, new_base
    gene_stats: pd.DataFrame     # columns: gene_id, n_mutations, fitness, reference_length
    references: dict[str, str]   # gene_id -> wildtype sequence

    @classmethod
    def from_summarized_json(cls, path: str, generation: int = 1999) -> "MutationPool"
    @classmethod
    def load(cls, path: str) -> "MutationPool"
    def save(self, path: str) -> None
```

No custom iterator / accessor helpers — Step 2 code can use pandas directly on the exposed DataFrames.

### Construction logic (`from_summarized_json`)

1. Call existing `load_mutations_from_json` from `src/analysis/mutations/summarize_mutations.py:263` to load `Dict[str, MutationsGene]`.
2. For each gene:
   - Grab the list at `gene.generation_dict[generation]` (default: 1999).
   - Select `max(sequences, key=lambda s: s.fitness)` — the highest-fitness pareto tip. By the pareto-front property, this is also the sequence with the most mutations, so no separate tie-breaking is needed.
   - Take its `.mutations` list of `(pos, ref_base, mut_base)` tuples directly (no re-parsing).
   - **Filter**: drop any mutation where `ref_base ∉ {A,C,G,T}` or `mut_base ∉ {A,C,G,T}` (guards against N-stretch positions visible in some reference sequences).
   - Log per-gene drop count if non-zero via `print_status`.
3. Assemble the three components and return.

### Files to create / modify

| File | Action | Purpose |
| --- | --- | --- |
| `src/workflows/mutation_distribution_analysis/mutation_pool.py` | create | `MutationPool` class + `main()` CLI |
| `test/workflows/mutation_distribution_analysis/__init__.py` | create | test package marker |
| `test/workflows/mutation_distribution_analysis/test_mutation_pool.py` | create | unit tests |
| `src/workflows/mutation_distribution_analysis/mutation_distribution.py` | leave empty | reserved for Step 2 |

### Existing code reused (no modification)

- `load_mutations_from_json` — `src/analysis/mutations/summarize_mutations.py:263`
- `MutationsGene` / `MutatedSequence` — same file, lines 98 / 13 (for typing only)
- `analysis.utils.io.print_section_header`, `print_subsection`, `print_status` — for CLI output consistent with `summarize_mutations.py`

### CLI

Mirrors the pattern in `summarize_mutations.py`:

```bash
python -m workflows.mutation_distribution_analysis.mutation_pool \
    --input  <path-to-all_mutated_sequences_*.json> \
    --output <path-for-mutation_pool.json> \
    --generation 1999
```

## Tests

Following the project's pytest / AAA / mock-IO conventions:

- `test_from_summarized_json_selects_highest_fitness` — three pareto-front sequences with different fitnesses → the highest-fitness one is picked.
- `test_filters_non_acgt_mutations` — a synthetic mutation with `N` as `source_base` is dropped; A↔C kept.
- `test_save_load_roundtrip` — write to tmp, reload, equals original.
- `test_gene_stats_derived_correctly` — `n_mutations` matches len(mutations); `reference_length` matches len(reference_sequence).
- `test_mutations_dataframe_shape_and_columns` — `pool.mutations` has the expected columns and total row count equals the sum of per-gene mutation counts.

Tests use `tmp_path` for any file IO; ensure the test output dir is gitignored.

## Verification

End-to-end smoke run on the real data:

```bash
python -m workflows.mutation_distribution_analysis.mutation_pool \
    --input  /home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/ara_msr_max_single/all_mutated_sequences_ara_msr_max_single_gen1999.json \
    --output ./mutation_pool_ara_msr_max_single_gen1999.json \
    --generation 1999
```

Expected outcomes to check by eye:

- `gene_stats` has ~1000 rows (one per gene).
- `mutations` has ≈ 60k–90k rows (matches the conceptual plan's projection).
- Every `source_base` / `new_base` in `mutations` is one of A/C/G/T.
- For one hand-picked gene (e.g. `5_AT5G38895_...`), the rows in `mutations` for that `gene_id` match the highest-fitness pareto-front entry in the source JSON (fitness `0.9999993...`, first sequence in that gene's gen-1999 list).
- Every value in `gene_stats.reference_length` is 3000.
- Reloading via `MutationPool.load()` reproduces the saved object exactly (DataFrame equality on `mutations` and `gene_stats`, dict equality on `references`).

Once this step is verified, Step 2 (PMF construction) and Step 3 (baseline sampling) can be planned and built as thin consumers of `MutationPool`.
