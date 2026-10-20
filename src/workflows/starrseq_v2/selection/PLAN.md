# STARR-seq v2 candidate selection

Four stages from finished evolution runs to an ordered oligo library.

1. **Pool** every final Pareto front into one overview table (`pool_front_table.py`).
2. **Select** anchors, then the globally best low-mutation entries, then an
   axis-uniform sample from each front (`select_candidates.py`).
3. **Build my sheet**: fetch sequences, number, point at references, write the
   table and the per-entry upload sheet (`build_library.py`).
4. **Clean all contributors' sheets together**: drop failing rows with their
   reasons, deduplicate, write the full table, per-sheet removal lists and the
   FASTA (`../qa/check_upload_sheets.py`).

Status: **all four steps implemented** — `pool_front_table.py` (cross-checked by
`verify_pool_front_table_untested.py`), `select_candidates.py`,
`build_library.py` (cross-checked by `verify_build_library_untested.py`) and
`check_upload_sheets.py` (unit-tested only, not yet run on real sheets). Every
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

`select_candidates.py`, producing `selected_keys.csv` with `run_dir`,
`front_index`, `selection_reason`. Step 3 stays criteria-agnostic.

### The criterion

Selection runs in **three stages**, each masked out of the next, so a slot goes to
the most specific claim on it.

1. **Anchors.** Every run contributes its reference (0 mutations), its half-max
   entry and its max-mutation entry.
2. **Best picks.** `TARGET_BEST = 5_600` entries ranked *globally across all runs*,
   half by fitness gain at a fixed low mutation count, half by fitness gain per
   mutation. See below.
3. **Uniform fill.** The rest of `TARGET` is filled by drawing one further entry
   per run in **rounds**, so the runs stay within one entry of each other on this
   stage until a front runs dry. `TARGET = 57_000`.

An entry an earlier stage already took **gains the later stage's reason but does
not consume its budget**, so each stage adds exactly its quota of new entries and
the reason field records every claim. This is why `selection_reason` is a `|`-joined
list and why a label count is not an entry count (measured below).

### Best picks — the two rankings

Both rank across all runs at once; neither caps how much one run may contribute.

**Ranking A, fixed counts.** For each count in
`BEST_MUTATION_COUNTS = (1, 2, 3, 5, 7, 10, 15)`, take the entries with exactly
that many mutations and rank them by `fitness_delta_vs_reference`, descending.
Half of `TARGET_BEST` is split equally over the seven counts (400 each). Reason
`best_at_<count>`.

Several counts are used because on a Pareto front fitness rises monotonically with
mutation count, so a single threshold would only ever return the entries at the
threshold. Ranking separately at each count asks the comparable question seven
times.

**Ranking B, gain per mutation.** `fitness_delta_vs_reference / n_mutations` over
every entry with at least one mutation (references would divide by zero and have no
gain by definition), descending. The other half of `TARGET_BEST`. Reason
`best_efficiency`. It runs after A, so A's picks no longer consume its budget.

Ranking B exists because A can only ever return entries at seven counts; B fills in
the counts between them and reaches runs that were strong but never top-400 at any
single count. Measured, B's picks spread over counts 1–29 rather than clustering.

Ties in either ranking go to the lower `run_dir`, then the lower `front_index`, so
the rankings are deterministic and no RNG is involved.

**`fitness_delta_vs_reference` is the only legitimate score here** — `fitness`
itself is not comparable across runs, because `direction` is mixed and the delta is
already sign-corrected so that positive means the optimisation worked.

**Known and accepted: the rankings are not headroom-corrected.** Fitness lives in
[0, 1] (measured 0.0087–0.9982) and `start_fitness` spans 0.024–0.994, so a run
whose reference already sits near the target end of the scale cannot post a large
delta however well it optimised. The rankings therefore favour runs with room to
move: at count 3 the top 200 entries have a median `start_fitness` of 0.323 against
0.150 over all entries at that count. This selects for the largest *measurable*
effect in the assay, which is the intent, but it is not a ranking of "how well the
evolution worked".

**A quota can fail even when its ranking is not empty**, because only *new* entries
count against it. If every run's entry at some target count is already that run's
half-max anchor, the count contributes nothing and `select_candidates` raises rather
than silently rebalancing. On the real data no count comes close to saturating.

`TARGET` counts **selected entries, not distinct sequences.** Steps 1 and 2 never
load sequences, so step 2 cannot deduplicate; step 4 does. The offset is stable
because the duplicates are almost all references, which are selected at any
target — measured 1405 duplicates at 57 449 entries and 1403 at 57 000. So set
`TARGET` to the wanted oligo count plus ~1400.

Where one entry is several anchors at once, the reasons are joined with `|`
(`half_max|max` on 14 runs, `reference|half_max|max` on the 4 length-1 fronts).
Where two entries share a mutation count, the **lower front index** wins — fronts
run from the optimised end to the reference, so the lower index is the more
optimised entry. This makes the tie-break deterministic; no RNG is involved.

### Sampling uniformly on the count axis, not over the entries

Each draw is `U(1, max_mutations_reached - 1)`, continuous, snapped to the
nearest entry not yet selected; ties again go to the lower front index. Counts 0
and `max` are excluded from the range because both are already anchors.

Sampling uniformly over the *entries* would give each entry probability
`1/|front|`, so the expected number of draws per unit of count axis would follow
the local entry density — and the density falls off near the cap, where the gaps
are (counts 50–59 are each missing from 125–173 runs). Snapping instead gives an
entry probability proportional to its catchment, half the gap to each neighbour,
so an entry alone in a sparse stretch inherits the whole gap.

Measured on the 1129 runs that reached 60 mutations, the draws land within a few
percent of the axis-uniform prediction in every 6-count bin, including the
truncated top bin (predicted 1314 draws, observed 1292). **At this aggregate
level the correction barely changes anything** — entry-uniform sampling would
have put ~1330 draws in that bin. It matters per run, on the individual fronts
with large gaps, which is what the unit test covers.

`SEED = 0`; the selection reproduces exactly.

**Known and accepted: a count can be drawn twice on a tied front.** `available`
in `draw_one` is per *entry*, so taking one of the two entries at a shared count
leaves its twin drawable; and where a tie sits at the max or half-max count, the
anchor takes the lower front index and the twin stays in the pool outright. Only
the eight tied fronts can be affected, and the two entries are different
sequences with bitwise-identical fitness, so both are legitimate library
members. Measured on the current selection: **0 runs** have two selected entries
at one count — but that is chance, not a guarantee (expected ≈0.8 occurrences),
so a different `SEED` or `TARGET` may produce one. Making availability per
mutation count instead of per entry would close it at a cost of ≤8 entries.

### Short fronts

A run that runs out of entries simply stops being drawn from, and the rounds
carry on over the others — so the ~100 short fronts hand their shortfall to the
long ones automatically and the target is still met. The runs are visited in a
**random order**, reshuffled once per selection, so the partial final round does
not always fall on the same runs.

The order of visits is the only thing separating a run that gets 18 entries from
one that gets 17; everything else about the two is identical.

### N

Entries containing `N` are excluded from both the anchors and the sampling pool.
All 852 of them sit in **20 runs where every entry contains `N`**, so this drops
those 20 runs whole and no run is ever left missing a single anchor.

### Result

At `TARGET = 57_000` and `TARGET_BEST = 5_600`: **57 000 entries from 3262 runs** —
9764 anchors (3262 × 3 anchor slots, collapsed onto 9764 distinct entries), 5600
best picks and 41 636 from the uniform fill.

**Label counts are not entry counts.** Because an already-selected entry gains a
later reason for free, the labels over-count: `best_efficiency` appears 5462 times
but added only 2800 entries, and the seven `best_at_<count>` labels sum to 2979 for
2800 added. 6046 distinct entries carry at least one `best_` label; 2696 entries
carry more than one reason of any kind. The commonest combinations are
`best_at_<count>|best_efficiency` (2232 entries in total — ranking B re-labels most
of ranking A) and `half_max|best_efficiency` (267).

**The best picks are concentrated, by design.** A run strong enough to lead at
several counts enters once per count, and ranking B is uncapped, so:

| | entries | runs | genes | configs | max per run |
| --- | --- | --- | --- | --- | --- |
| ranking A (`best_at_*`) | 2979 | 748 | 551 | 50 | 7 |
| ranking B (`best_efficiency`) | 5462 | 740 | 539 | 51 | **29** |
| any `best_` label | 6046 | 834 | 609 | 51 | 29 |

609 of the library's 1881 genes carry a best pick. The extreme case is
`AT4G15880` (delta 0.894 at its best entry), which ranking B labels at **every**
count from 1 to 29 — a complete mutation ladder on the single strongest responder.

**Best picks are additive, not compensated.** The uniform fill does not know a run
already won best picks, so those runs end up larger: entries per run now run
**1 to 44** (median 16), against the flat 17–18 of the anchors-plus-fill design,
and 247 runs hold more than 25 entries. The old guarantee that "the order of visits
is the only thing separating a run that gets 18 entries from one that gets 17" now
holds only *within* the fill stage.

A per-run cap on ranking B is the lever if this is too concentrated: capping it at
one entry per run was measured at 1400 runs / 961 genes for the same budget, at the
cost of a lower median gain per mutation (0.031 against 0.047) and the loss of the
ladders.

Deduplicated on the exact 170 bp string, that is **55 597 distinct sequences** —
1403 lost, 2.5%. The loss is almost entirely in the references (3262 → 1969);
the mutated entries are effectively unique (random 47 236 → 47 157, max 3244 →
3244, half_max 3258 → 3240). The reference collapse is the known one: identical
natural windows shared across runs and, for 167 of them, across gene ids, which
is why step 3 must keep every contributing origin.

That left ~4.4k of what was then a ~60k budget for controls and for whatever
the restriction screen drops. The budget is now **100k oligos, shared with the
other contributors' sheets** (step 4), so `TARGET` is set against what remains
after their contribution.

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

`build_library.py`. Takes `selected_keys.csv` + `front_table.csv.gz` + the run
tree, and writes my raw upload sheet — **one row per selected entry**, not yet
screened or deduplicated. Both happen in step 4, pooled with the other
contributors' sheets, so there is exactly one screen and one deduplication.

Read any step-1 table with `float_precision="round_trip"` (see step 2).

1. **Fetch sequences** by `(run_dir, front_index)`, reading each front once.
   No separate "add references" step: every run's reference is an anchor in
   step 2, so it is already selected. Asserted, not assumed.
2. **Number the entries** `1…N` after a deterministic sort on
   `(config_name, gene_id, run_dir, front_index)`, so one experimental arm is
   contiguous. The trailing `(run_dir, front_index)` keys the front table, which
   is what makes the sort total.
3. **Reference pointers** — each entry's `reference_id` is the identifier of its
   run's reference. Reference rows point at themselves.

`N`-containing entries need no handling here: step 2 already excludes them, and
all 852 sit in 20 runs where every entry contains `N`, so those runs are gone.

### Numbering before deduplication

The number is assigned **before** deduplication, so every selected entry stays
individually addressable in its own run context. The same 170 bp arising from a
maximize run on one gene and a minimize run on another is one oligo but two
rows, and both contexts are wanted for analysis.

The consequence has to be stated because it is silent: **several identifiers
map onto one ordered oligo and therefore one measurement.** Join assay results
on step 4's `representative_id`, never on `sequence_number`, or duplicates are
counted twice.

This also removes the need for an `origins` column. One row per contributing
gene already records every origin of a collapsed sequence — relevant because
167 reference sequences are shared across more than one gene id, expected for
an allotetraploid like N. tabacum (homeologs) and for overlapping annotations.

### The identifier

The upload template's `id` is five `_`-separated fields,
`GENOME_CHR:START-END_SETUP_AUTHOR_UNIQUE`. Underscores inside a field are
replaced with `-` and `|` with `+`, because chromosome names and config names
are full of underscores:

```
TAIR10_4:12674106-12670896_AT4G24540.config-a.f12.m37.random_GS_000123
Ntab-TN90_NW-015791515.1-Nicotiana-tabacum:14248-7986_NW-015791515.1-000002.….f03.m08.half-max_GS_000456
```

`SETUP` is `<gene_id>.<config_name>.f<front_index>.m<n_mutations>.<reason>`,
both numbers zero-padded to two digits (counts run 0–60). `UNIQUE` is the
6-digit zero-padded `sequence_number`.

`f` and `m` are both needed: a front is not indexed by mutation count, and on
the eight tied fronts two entries share a count, so only the front index
separates them.

**The coordinates are the gene's genomic span, taken verbatim from the record
name, not the 170 bp window's coordinates.** The window's frame offsets are
carried in `source_start`/`source_end` instead. This is simply the cheaper
route — the record name already holds chromosome and span, so nothing has to be
derived and nothing can be derived wrongly.

(An earlier draft of this section claimed the existing frame→genomic converters
were broken for genes under 1000 bp. **They are not.** `additional_padding`
compensates exactly, fixing the TSS at offset 1000 and the TTS at 2020 for every
gene; the hardcoded 1520 in those converters is an affine anchor, not a block
boundary. See the vault note `extraction-frame-landmark-invariance`.)

The span is kept in the record name's own order — reversed for minus-strand
genes, which is how `extract_sequences.py:399-402` encodes the strand.

Genome names verified against the files the extraction used:
`Arabidopsis_thaliana.TAIR10.dna.toplevel.fa` → `TAIR10`;
`hlx-Nicotiana_tabacum-GCF_000715135.1-4097.gff` and the FASTA header
(`… Ntab-TN90 Ntab-TN90_scaffold10 …`) → `Ntab-TN90`.

### Outputs

Into `output/` (already gitignored):

| file | content |
| --- | --- |
| `library_table.csv` | one row per selected entry — my run context for analysis, joined to step 4's full table on `identifier` = `id` |
| `library_upload.csv` | the five template columns (`Number, id, reference, product, path_to_config`), one row per entry — my input to step 4 |
| `configs_to_copy.txt` | the 68 config source paths to copy into `configs_gernot/` |

The template's `reference` column expects a literal 170 bp sequence, which
would repeat every reference on every row. Every contributor puts the
reference's **identifier** there instead — a deliberate deviation the recipient
has to be told about.

`path_to_config` is `configs_gernot/<config_name>.json`. The config folder name
in the run tree equals the config file's basename, so the mapping is direct.
`configs/starrseq_v2/` holds 138 files but only 68 distinct configs: `bins/` is
a re-partition of the same 68 into six execution batches, and all 68 are
**byte-identical** to their namesakes in `arms/baseline/grid/starrseq_v1`, so
flattening them into one folder is lossless.

### Validation

`verify_build_library_untested.py` re-reads every selected entry from its
`pareto_front.json` and re-derives the numbering and the reference pointers
independently, then compares them against the written table and sheet.

## Step 4 — pooled cleaning of all contributors' sheets

`qa/check_upload_sheets.py <configs_root> <out_dir> <sheet>...`. Sheets are
.csv or .ods (first sheet read). Every
contributor delivers a sheet in the five template columns, with identifiers in
`reference`. All sheets are pooled first, so the screen and the deduplication
run once over everything that is ordered.

1. **Load and renumber.** `source_file` records the input sheet. `Number` is
   rewritten to `1…N` over the pooled rows, in the order the sheets are given.
2. **Row-level drops.** Nothing aborts the run; a failing row is dropped and
   every reason it fails is recorded, `|`-joined, in `drop_reasons`:
   `empty_id` and `empty_reference` (empty, whitespace, or any pandas default NA value),
   `whitespace_in_id`, `whitespace_in_reference`, `duplicate_id` (every row carrying that id), `not_acgt`, `wrong_length`
   (≠ 170), `restriction_site`, `reference_missing` (the reference id is in no
   sheet), `unresolved_config` (`path_to_config` is not a file under
   `configs_root`).
3. **Reference cascade.** A row whose reference was dropped is dropped as
   `reference_dropped`, repeated until nothing changes, so chained references
   are followed. Without an orderable baseline none of its variants can be
   interpreted, even where a mutation destroyed the site. For my sheet this is
   exactly the earlier "a run whose reference carries a site loses all of its
   entries", since every entry points at its run's reference. A row that fails
   a check of its own is listed under that reason only.
4. **Deduplicate** the kept rows on the exact 170 bp. The representative is the
   kept row with the lowest `Number` — the first occurrence in sheet order.
   Dropped rows never represent. Duplicates are **not** removed: they stay in
   the table with `representative_id` pointing at the oligo that measures them.
5. **Final checks**, raising on failure: `Number` is `1…N` without gaps or
   duplicates, representative sequences are unique, and every kept row's
   representative and reference are kept.

`reference` is never rewritten. A row's baseline measurement is two lookups:
`reference` → that row's `representative_id`. The cascade guarantees it
resolves for every kept row.

### The restriction screen

`GGTCT(C|$)|GAGAC(C|$)` — BsaI and its reverse complement. The `$` alternative
covers a site the cloning suffix would complete by supplying the final `C`;
the two patterns being reverse complements of each other is what makes the
single trailing-edge case sufficient, so the bare 170 bp is screened and no
second pass with the flanks attached is needed. Drop, never repair: repairing
would move the sequence off the front it was selected from.

### Outputs

Into `<out_dir>`:

| file | content |
| --- | --- |
| `library_full_table.csv` | every input row: the five template columns, `source_file`, `drop_reasons` (empty = kept), `representative_id` (empty = dropped) |
| `removed_<sheet>.csv` | per input sheet, its dropped rows with their reasons — to hand back to the author. Written even when empty. Duplicates are not listed. |
| `library.fasta` | the representatives only, `id` → 170 bp — what is ordered |

There is no separate representatives-only sheet: the FASTA headers are ids, and
joining on `id` against the full table gives everything else. The run prints
the drop counts per reason and sheet, and the oligo count against the 100k
budget; the count is only reported, never enforced.

### Result

Measured under the earlier single-contributor pipeline, whose screen and
deduplication rules are the ones step 4 now applies: from 57 000 selected
entries, 1775 dropped by the screen (91 runs lost with their reference, costing
1522 entries; 253 for a site of their own), leaving **55 225 entries from 3171
runs over 1853 genes**, which deduplicate to **53 874 distinct sequences**. Not
yet re-measured on the pooled sheets.

### Still open

- **which controls go in**, if any.
- the other contributors' sheets — the pooled oligo count against the 100k
  budget is known only once they are in.
