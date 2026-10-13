# STARR-seq v2 candidate selection

Three stages from finished evolution runs to an ordered oligo library.

1. **Pool** every final Pareto front into one overview table (`pool_front_table.py`).
2. **Choose criteria** on that table — Gernot's call, not implemented here.
3. **Build the library**: select, add references, deduplicate, screen restriction
   sites, assign IDs, write the table and the FASTA (`build_library.py`).

Status: **step 1 implemented** (`pool_front_table.py`, cross-checked by
`verify_pool_front_table_untested.py`); steps 2 and 3 are plan only. Every
number below was measured on the actual tree, not estimated.

## Measured facts

| fact | value |
| --- | --- |
| run tree | `Evolution_runs/dataset/starrseq_v2/full_run/<config>_<ts>/<record>_<ts>/` — **flat, no group directories** |
| analysis tree | `Evo_run_analysis/dataset/starrseq_v2/full_run/<config>/stats_<config>.json` |
| configs / runs | 68 / 3282 |
| front entries | **180 353** — this is the row count of the pooled table |
| every run has `parameters.json` + `saved_populations/pareto_front.json` | 3282 / 3282 |
| every front entry is 170 bp | 180 353 / 180 353 |
| every front contains a 0-mutation entry | 3282 / 3282 |
| mutation counts are integers, never above the cap of 60 | confirmed |
| generations | 1000 (3180 runs), 300 (102 runs — the natural-mutation arm) |
| analysis dir name == config name minus timestamp | 68 / 68, bijective |
| stats keys == run folder names | 3282 unique, 0 missing, 0 conflicting |
| the half-max count is always present in its own front | 3282 / 3282 |
| every record's gene id resolves against the metadata | 2151 records, 0 unresolved |

### Three things that break the obvious implementation

**1. Fronts have gaps — 1360 of 3282 runs (41%).** (An earlier draft said
1363; that count could not be reproduced. 1360 = fronts whose distinct mutation
counts are fewer than `max + 1`.) A front is *not* mutation
counts `0 … max`. Gaps cluster at the top of the range (counts 50–59 are each
missing from 125–173 runs; low counts are almost never missing), which is just
the front thinning out near the cap. Consequences: never index a front by
mutation count, never assume `front[i]` has `i` mutations, and do not validate
contiguity. Front lengths run from **1** (four runs where nothing but the
reference survived) to 62, median 56–59 depending on group.

**2. Eight fronts hold two entries at the same mutation count** — in every case
two *different sequences* with bitwise-identical fitness. Genuine
non-dominated ties, not corruption. So mutation count does **not** key a front
either.

Therefore the row key is **`(run_dir, front_index)`**, and `n_mutations` is an
ordinary column. A `tied_at_count` flag marks the 16 affected rows.

**3. 852 front entries contain `N`**, inherited from the genome in the natural
backgrounds. They cannot be ordered and must be dropped in step 3, so step 1
records `n_count` per row while it has the sequence in hand.

## Step 1 — pool all fronts

`pool_front_table.py`. Hardcoded paths at the top, no CLI, matching
`build_starrseq_v2_library.py`.

### Inputs

Absolute paths, as module-level constants:

```python
RUNS     = "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/starrseq_v2/full_run"
ANALYSIS = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/starrseq_v2/full_run"
METADATA = "/home/gernot/Code/PhD_Code/Evolution/data/starrseq_v2/candidate_gene_metadata.csv"
```

`full_run` is its own tree, so the `probe` pilot and the older `M0M0.75` /
`M0X0.75` round are excluded simply by not being under it.

Run with
`conda run -n deepCREshap python -m workflows.starrseq_v2.selection.pool_front_table`,
then check it with
`conda run -n deepCREshap python -m workflows.starrseq_v2.selection.verify_pool_front_table_untested`.

Config folders can also hold `problem_configs.json` / `problem_configs_errors.json`
for runs that never started (8 today: all-`N` mutation windows and one missing
VCF). Those genes simply have no run in that config; pooling skips the files.

`run_dir` is `"<config_dir>/<run_dir>"` relative to `RUNS`, both including
their timestamps. `record_name` is `parameters.json:sequence_name` — the run
directory is that name plus a three-field timestamp, but read the field rather
than stripping the suffix.

### Factors come from the config folder name

Inverted from the previous draft, and this is deliberate. The config name is
written deterministically by `write_run_script.py:1486` as
`"_".join(factor levels + [direction])`, and all 68 parse unambiguously by
consuming known levels left to right. `parameters.json` becomes the
**cross-check**, not the source.

The reason is `location`. In `parameters.json`, `mutation_start/end` is
`(780, 950)` for construct+correct, construct+offsite *and* natural+correct —
three of the four cases are identical. Only natural+offsite differs (38
distinct windows). So the parameter file cannot separate construct+correct
from construct+offsite at all; that needs `reference_sequence_initial`, and
natural+correct from natural+offsite only by a window value that could in
principle be 780 (it never is in this data, but nothing forbids it —
`RANDOM_WINDOW_STARTS` includes 780). Parsing the config name is exact; the
cross-checks below then catch any disagreement.

Cross-checks against `parameters.json`, each raising on mismatch:

| factor | check |
| --- | --- |
| `models` | `len(models) == 1` ⟺ `single` |
| `model_sp` | model paths all under `/ntab/` ⟺ `ntab_model`, `/Ara_ssr/` ⟺ `ara_model` |
| `background`, `gene_sp`, `location` | `reference_sequence_initial` equals `REFERENCE_PATHS[(background, gene_sp, location)]` |
| `mutations` | `vcf_path is None` ⟺ `unconstrained` |
| `direction` | `weights == [1,-1]` ⟺ `maximize`, `[-1,-1]` ⟺ `minimize` |

The seven distinct `reference_sequence_initial` values in the data were checked
against `evolution.write_run_script.REFERENCE_PATHS` and all seven match, so
the dict can be inverted directly. (The `probe` runs carry a stale
`ntab_flowering_GOF_constructs.fa` that is *not* in the dict — another reason
the probe stays out of scope.)

`direction` is additionally checked against the gene's metadata directions —
see "Joining the gene metadata" for why this is a membership test, not equality.

### Group

There are no group directories; the group is derived from the factors:
`v1_genes` → `starrseq_v1`; otherwise count the factors differing from
`BASELINE` — 0 → `baseline`, 1 → `arms`, more → `grid`. Measured:
baseline 4 configs, arms 18, grid 44, starrseq_v1 2.

### Source position of the 170 bp

Frame offsets into the 3020 bp gene frame. **Three cases, and the metadata is
not consulted for any of them:**

| case | source window | runs |
| --- | --- | --- |
| `location == correct` (either background) | `(780, 950)` | 1778 |
| `construct` + `offsite` | `_off_<start>` parsed from `record_name`; end = start + 170 | 1348 |
| `natural` + `offsite` | `parameters.json` `mutation_start` / `mutation_end` | 156 |

Verified: all 1348 construct+offsite records carry an `_off_<start>` suffix and
no other record does; all 156 natural+offsite runs have `mutation_start != 780`.

Two traps this avoids, both measured:

**`parameters.json` is not the source.** `mutation_start/end` says where the
optimiser could mutate *in the reference it was handed*, which is a different
coordinate system per background. In a construct it is the plasmid insert site
`(780, 950)` whatever the insert's origin. It coincides with the source window
for `correct` and is flatly wrong for construct+offsite — reading it would
mislabel all 1348 of those rows as `(780, 950)`.

**The metadata is not the source either.** 73 genes carry more than one
distinct `alternative_start`, and **146 offsite runs sit on such a gene**. A
`gene_id` lookup cannot tell which of that gene's windows a given run used, so
it would silently pick one at random. The record name and the parameter file
both name the window exactly.

Cross-check: for construct+offsite, the parsed start must appear among that
gene's `full_length_window` rows in the metadata (0 mismatches today).

### Joining the gene metadata

The metadata has 2171 rows for 2065 genes — **101 genes carry up to 3 rows**,
one per `reason`. A plain `merge` on `gene_id` would duplicate front rows two-
and threefold and silently inflate the table. Collapse to one row per gene
first:

| column | rule |
| --- | --- |
| `species` | identical within a gene (verified: 0 genes disagree) — take the single value |
| `arm_subset`, `full_grid_subset` | **not joined.** Whether a gene also has arm / grid runs is in the table itself: `front_table.groupby("gene_id")["group"].agg(set)`. Verified identical to the flags for all non-v1 genes (191 / 38 genes, 0 disagreements); for v1 genes the flags are misleading, because the v1 group has no arms or grid |
| `reason` | **aggregate**: sorted unique values joined with `|`, e.g. `GOF|flowering` |
| `direction` | **not joined.** Taken from `weights` in `parameters.json` |
| `alternative_start/end` | **not joined** — see the source-position rule above |

`direction` cannot come from the metadata: 13 genes carry both `maximize` and
`minimize` under different reasons, and nothing in a run's output records which
reason it was launched under. `weights` is authoritative. The cross-check is
therefore the weaker but correct one: the run's direction must be **among** the
gene's metadata directions.

### Columns

*Key* — `run_dir` (relative to `full_run`; unique per run), `front_index`.

*Identity* — `group`, `config_name`, `record_name`, `gene_id`.

*Factors* — `models`, `model_sp`, `background`, `location`, `mutations`,
`gene_sp`, `direction`.

*Gene metadata* (collapsed to one row per gene first) — `species`, `reason`.

*Source position* — `source_start`, `source_end`.

*Varying parameters* — `number_of_generations`, `n_models`, `model_paths`
(the model file **basenames**, sorted, joined with `|` — full paths are long
and carry no information the basename lacks), `has_vcf`.
(`checkpoint_generations` varies too, but only as a function of
`number_of_generations`, so it is dropped.)

*Front entry* — `n_mutations` (int, from `round()`; verified already integral),
`fitness`, `n_count` (count of `N` in the sequence).

*Per-front scalars, taken from the analysis output, not recomputed* —
`start_fitness`, `final_fitness`, `max_mutations_reached`,
`num_mutations_half_max_effect`.

*Flags* — `is_reference` (`n_mutations == 0`), `is_half_max`
(`n_mutations == num_mutations_half_max_effect`), `is_max_mutations`,
`tied_at_count`.

*Derived* — `fitness_delta_vs_reference`, signed so that **positive always
means the optimisation worked**: `fitness - start_fitness` for `maximize`,
`start_fitness - fitness` for `minimize`. (`start_fitness` from the stats file
is the 0-mutation fitness, so this is 0 on every reference row.)

Measured constant across all 3282 runs, so written once to
`fixed_parameters.json` rather than repeated 180k times: `population_size`,
`mutation_rate`, `mutation_probability`, `crossover_probability`,
`tournament_size`, `size_of_hall_of_fame`, `initial_number_mutations`,
`initial_mutation_strategy`, `mutation_strategy`, `evaluate_population`,
`eval_pop_kwargs`, `max_number_mutations`, `max_N`, `limit_mutations`,
`maximize_greedy`, `sort_before_crossover`, `verbose`. They are written from
the first run; the verification script checks the constancy. If it fails, that
parameter moves into the table.

### Validation

Kept **out of** `pool_front_table.py` so the module stays short to review. All
checks live in `verify_pool_front_table_untested.py`, which re-derives everything
from the raw sources, then compares the written tables entry by entry. It has
no unit tests (hence the name); it prints counts and every failed check. All
checks pass today.

- every run has `parameters.json` and `pareto_front.json` *(3282/3282 today)*
- every sequence is 170 bp *(180 353/180 353)*
- every front has a 0-mutation entry *(3282/3282)*
- mutation counts are integers within `0 … max_number_mutations` *(holds)*
- config-name factors match every `parameters.json` cross-check above
- `direction` is among the gene's metadata directions
- every offsite source window appears among that gene's metadata windows *(0 mismatches)*
- every `gene_id` resolves *(2151/2151)*
- the metadata collapses to one row per gene without losing a distinct `species`
- every arms / grid run's gene is in `arm_subset` / `full_grid_subset`, and every
  subset gene has a run there or a `problem_configs.json` entry
- a stats entry exists for every run *(3282/3282)*, and its start fitness, max
  count and final fitness agree with the front
- the half-max count exists in its own front *(3282/3282)*
- no two runs share `(config_name, record_name)`
- run folder == `sequence_name` + timestamp; `base_folder` == config folder
- the 0-mutation entry equals both `reference[mutation window]` and the natural
  gene frame at `[source_start, source_end)` — this checks the source window
  against the sequence itself
- runs present + `problem_configs.json` == exactly the records
  `write_run_script.sequences_per_direction` yields for that config
- every written value equals the raw front entry (read with `float_precision="round_trip"`)

Reported but **not** errors: gap count per front, fronts shorter than some
threshold, duplicate-count ties, entries containing `N`.

### Outputs

Into `src/workflows/starrseq_v2/selection/output/` (add to `.gitignore`):

| file | content |
| --- | --- |
| `front_table.csv.gz` | the pooled table, 180 353 rows |
| `run_overview.csv` | one row per run, 3282 rows, per-run columns only |
| `fixed_parameters.json` | the parameters verified constant |

Gzipped CSV, not parquet: `deepCREshap` has no `pyarrow`, and `pandas` reads
`.csv.gz` natively.

**No sequences in the table.** Criteria are metadata and fitness; step 3
re-reads only the fronts it selected, keyed by `run_dir` and `front_index`.
`n_count` is carried so that `N`-containing entries can be excluded at
criteria time rather than being a surprise in step 3.

### Tests

`test/workflows/starrseq_v2/selection/test_pool_front_table.py`, against a
synthetic tree in a temporary directory. Must cover, because the real data
contains every one of them:

- a front with gaps, a front with a duplicate mutation count, a front of length 1
- a sequence containing `N`
- all three source-window cases, including a construct+offsite record whose
  gene has **two** metadata windows (the 146-run trap)
- a gene with three metadata rows, asserting the table does not fan out
- group derivation for all four groups
- the sign of `fitness_delta_vs_reference` for both directions

The cross-checks are not unit-tested; they live in the verification script.

Reuse `gene_id_from_record` from `evolution.write_run_script` rather than
re-implementing the suffix match; it already strips the `_off_<start>` suffix
via the `_gene:` split.

## Step 2 — selection criteria

Gernot's. Only the interface is fixed here, so step 3 stays criteria-agnostic:

step 2 produces `selected_keys.csv` with `run_dir`, `front_index`,
`selection_reason`.

Keying on `front_index` rather than `n_mutations` is what makes the eight tied
fronts unambiguous.

**Read the table with `float_precision="round_trip"`.**
`pd.read_csv("front_table.csv.gz", float_precision="round_trip")`. The default
parser is not exact: it read **87 596 of the 180 353** fitness values back one
ULP (±1 unit in the last bit) off the value that was written. Any criterion that compares fitness exactly (ties,
`==` against the stats file) silently goes wrong without it.

**7 runs sit on a starrseq_v1 window inside an `ara_genes` config.**
`build_starrseq_v2_library.py:497` splices the off-target constructs without
the `reason` filter that the v1 call (line 499) uses, so AT1G25560 (@440),
AT1G22810 (@2734) and AT1G13260 (@554) were run at their v1 window **in addition
to** their random window, in the construct+offsite `ara_genes` configs. The data
is correct (source window and direction are right), but these genes have two
offsite runs where every other gene has one. Two of the 7 repeat a v1-group
condition exactly. Decide whether to exclude them before counting runs per gene.
Find them with `run_overview` rows where `location == "offsite"`, `gene_sp == "ara_genes"`,
and the source window is a `starrseq_v1` metadata window.

## Step 3 — build the library

`build_library.py`. Takes `selected_keys.csv` + the run tree.

Read any step-1 table with `float_precision="round_trip"` (see step 2), and
take fitness values from the fronts themselves when exactness matters.

1. **Fetch sequences** by `(run_dir, front_index)`.
2. **Add references** — for every selected `run_dir`, include its 0-mutation
   entry if not already selected.
3. **Drop sequences containing `N`** (852 candidates across the whole pool).
4. **Deduplicate** on the exact 170 bp string, keeping one by a deterministic
   sort and recording every collapsed origin in an `origins` column.
   Magnitude, measured on the references alone: 3282 reference entries are only
   **1980 distinct sequences**. Note that **167 reference sequences are shared
   by more than one gene id** — expected for an allotetraploid like
   N. tabacum (homeologs) and for overlapping annotations. Identical sequences
   are indistinguishable in the assay, so merging them is right, but the
   `origins` column must preserve every contributing gene.
5. **Screen restriction sites** — drop, never repair; repairing would move the
   sequence off the front it was selected from. Screen each sequence **with the
   cloning prefix and suffix attached**, then emit the bare 170 bp.
6. **Assign IDs** — `SQ2_000001…`, numbered after a deterministic sort on
   `(gene_id, config_name, n_mutations, front_index)`, so a rerun reproduces
   them.
7. **Final checks** — every sequence 170 bp and `ACGT`-only, IDs unique,
   sequences unique, total within the order budget.

Outputs: `library_table.csv` (ID + step-1 metadata + sequence + `origins` +
`selection_reason`) and `library.fasta` (ID → 170 bp).

### Step 3 needs four inputs that do not exist yet

None of these are in the repo — the `starrseq_v2/README.md` "Open items"
section records them as never filled in. Steps 1 and 2 do not depend on them.

- **the restriction enzyme recognition sequences** to screen against (as a
  FASTA, matching what `get_restriction_regexes` expects)
- **the cloning prefix and suffix** that flank the 170 bp insert in the ordered
  oligo. Needed because a recognition site can straddle the junction between
  the insert and its flanks: a sequence clean on its own can still carry a site
  once the adapters are attached, so screening the bare 170 bp would pass
  sequences that fail at the bench.
- **the order budget** — how many sequences the library may contain
- **which controls go in**, if any
