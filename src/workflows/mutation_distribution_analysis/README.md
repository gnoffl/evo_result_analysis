# Mutation Distribution Analysis

Generate **random baseline promoter sequences** drawn from the evolutionary
algorithm's own empirical mutation distribution, to serve
as a null model against the SNPs introduced by the evolutionary optimisation
algorithm. The question this workflow answers is:

> *Are the SNPs the optimiser picks doing something specific to fitness, or
> would any randomly drawn SNP from the same empirical distribution look
> similar to the MSR model?*

A baseline sequence has the **same number of SNPs** as the optimised pareto-tip
sequence and those SNPs are drawn from the **same empirical (position,
source_base, new_base) distribution** that the optimiser itself produced —
just without any fitness-directed selection.

---

## Conceptual approach

Each mutation is treated as an atomic three-tuple
`(position, source_base, new_base)`. Pooling all such tuples across every
optimised gene gives an empirical joint distribution. Baselines are drawn from
that distribution; comparing their MSR-model predictions to the optimised
sequences isolates *what the algorithm contributes beyond drawing from its
own mutation prior*.

### The pool-as-sample-space shortcut

The original plan in `plans/plan.md` proposed building explicit PMFs
(`P(position)`, `P(new_base | position, source_base)`, fallbacks for unseen
combinations, etc.). That plan was **superseded during implementation** by a
much simpler equivalent (`plans/baseline_sampler_plan.md`):

* For a target gene with wildtype `W`, filter the pool to rows where
  `source_base == W[position]` — i.e. the subset of pool entries that are
  consistent with this gene's wildtype.
* **Uniformly sample rows** from that subset.

Uniform-over-rows reproduces the empirical joint `(position, source_base,
new_base)` density implicitly: a `(pos, src, new)` combination that occurs
50× in the filtered pool is drawn 50× more often than one that occurs once.
No explicit PMF, no smoothing decisions, no fallback rules for sparse cells.

After each draw all rows sharing the chosen position are removed from the
working set ("position blocking"), guaranteeing no two SNPs land at the same
site in one baseline.

---

## Pipeline (3 stages, one file each)

```flowchart
summarized-mutations JSON               (input from upstream analysis)
        │
        ▼  mutation_pool.py            (Step 1: extract pool)
mutation_pools/<run>_mutation_pool.json
        │
        ▼  baseline_sampler.py         (Steps 2 + 3: filter + sample)
mutated_sequences/<run>_random_mutated.fa
        │
        ▼  evaluate_sequences.py       (Step 4: score with MSR model)
mutated_sequences/<run>_random_mutated_predictions.csv
```

### Step 1 — `mutation_pool.py`

Flattens upstream `MutationsGene` objects into a small, sample-friendly
artifact. For each gene it picks the **highest-fitness sequence on the
pareto front at the requested generation** (default `1999`); by the pareto
property this is also the sequence with the most mutations, so no separate
tie-break is needed. Non-ACGT mutations (caused by the `N` padding stretch
that sits in the centre of every reference sequence) are dropped with a
per-gene warning.

The persisted `MutationPool` JSON has three top-level keys:

| Key | Shape | Purpose |
| --- | --- | --- |
| `mutations` | long-format records `gene_id, position, source_base, new_base` | the sample space for Step 2/3 |
| `gene_stats` | one row per gene: `n_mutations`, `initial_fitness`, `final_fitness` | drives the per-gene SNP count `k`; lets you sanity-check selection |
| `references` | `gene_id -> wildtype sequence` | needed to filter applicable rows and to apply mutations |

Keeping the three tables flat and separate (rather than nested per gene)
keeps downstream code to one `groupby` away from any aggregate it needs.

### Step 2 + 3 — `baseline_sampler.py`

A single module with three layers:

1. **Pure helpers** (`_filter_applicable`, `_sample_mutations_blocking`,
   `_apply_mutations`) — small, independently testable, vectorised where
   it matters.
2. **Per-gene API** — `sample_baseline_sequence(pool, gene_id, n_mutations, rng)`
   returns one baseline string.
3. **Batch driver** — `generate_baselines_fasta` iterates every gene in the
   pool, takes `k = gene_stats.n_mutations` for that gene, draws
   `n_per_gene` baselines and appends them to a single FASTA. Genes whose
   applicable subset has fewer unique positions than `k` are skipped with a
   warning rather than aborting the run.

FASTA headers are `>{gene_id}_baseline_{i:03d}` — zero-padded so up to 999
baselines per gene sort lexicographically.

### Step 4 — `evaluate_sequences.py`

Loads the per-species MSR `.h5` model, one-hot encodes the FASTA, and writes
a two-column CSV (`sequence_name, prediction`). Currently a small script
hard-wired to the two species (Arabidopsis + maize) and the two
corresponding MSR models — this is the entry point that will feed into the
downstream comparison plots.

---

## Artifacts produced

Both committed alongside the code so reruns aren't required for downstream
analysis:

* `mutation_pools/{ara,zea}_msr_max_single_gen1999_mutation_pool.json` —
  the extracted pools (~16–18 MB each, ~75k mutations across ~1000 genes).
* `mutated_sequences/{ara,zea}_random_mutated.fa` — sampled baselines
  (multiple per gene in the current files; headers `_baseline_000`,
  `_baseline_001`, ...).
* `mutated_sequences/{ara,zea}_random_mutated_predictions.csv` — MSR
  predictions over the baselines.

---

## CLI

```bash
# Step 1
python -m workflows.mutation_distribution_analysis.mutation_pool \
    --input  /path/to/all_mutated_sequences_<run>_gen1999.json \
    --output ./mutation_pools/<run>_mutation_pool.json \
    --generation 1999

# Step 2 + 3
python -m workflows.mutation_distribution_analysis.baseline_sampler \
    --input  ./mutation_pools/<run>_mutation_pool.json \
    --output ./mutated_sequences/<run>_random_mutated.fa \
    --n-per-gene 1 \
    --seed 42

# Step 4 — currently invoked directly (hard-wired paths)
python src/workflows/mutation_distribution_analysis/evaluate_sequences.py
```

Both upstream stages accept a `--seed`/`--generation` flag where the run
should be reproducible; `evaluate_sequences.py` doesn't currently accept
arguments.

---

## Key design decisions (cumulative)

| Decision | Choice | Rationale |
| --- | --- | --- |
| Mutation type | SNPs only | Algorithm only introduces SNPs; indels excluded |
| Sequence per gene | Highest-fitness pareto tip at gen 1999 | Also the most-mutated sequence → consistent target for the "what does the algorithm do" question |
| Pool scope | Pooled across all ~1000 genes | ~75k mutations give reliable empirical estimates |
| Source-base conditioning | Always | Sticks closely to what the evolutionary algorithm actually does (sampling `(position, source_base, new_base)` triples) and is more specific than conditioning on position alone |
| Sampling implementation | **Uniform over filtered rows + position blocking** (replaces the PMF approach in `plans/plan.md`) | Implicitly preserves the empirical joint; no smoothing or fallback edge cases |
| Position uniqueness | Without replacement (position-blocking) | No two SNPs at the same site, mirroring the algorithm's behaviour |
| Insufficient applicable pool | Skip gene with warning, continue | Single gene shouldn't abort a 1000-gene run |
| Pool persistence format | Plain JSON (three top-level keys) | Human-readable, reloads cleanly to the original DataFrames |
| Pareto-front fitness recorded | `initial_fitness` (fewest mutations) + `final_fitness` (most mutations) | Captures the front's span without persisting the whole front |

---

## What's still open

* **Step 4 (validation) from `plans/plan.md`** — sanity checks that the
  sampler is well-calibrated against the empirical pool (sampled position
  histogram, transition/transversion ratio, count distribution) have not
  been implemented yet.
* `evaluate_sequences.py` is a quick script — its paths are hard-coded,
  there is no CLI, and it is not under tests. Promote to a CLI module if
  it grows beyond the current two-species comparison.
* The downstream comparison of baseline predictions vs. optimised-sequence
  predictions (the original motivation) lives outside this folder and is
  the next consumer of the CSV outputs.

---

## File map

```file_view
mutation_distribution_analysis/
├── README.md                       (this file)
├── mutation_pool.py                Step 1 — pool extraction
├── baseline_sampler.py             Steps 2+3 — filtering + sampling
├── evaluate_sequences.py           Step 4 — MSR scoring (script form)
├── mutation_pools/                 persisted pools (ara, zea)
├── mutated_sequences/              sampled FASTAs + prediction CSVs
└── plans/
    ├── plan.md                     original conceptual plan (PMF approach)
    ├── mutation_pool_plan.md       Step 1 implementation plan
    └── baseline_sampler_plan.md    Steps 2+3 implementation plan
                                    (supersedes the PMF approach in plan.md)
```

Unit tests for steps 1 and 2/3 live at
`test/workflows/mutation_distribution_analysis/`.
