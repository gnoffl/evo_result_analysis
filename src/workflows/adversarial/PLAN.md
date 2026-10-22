# Adversarial re-evaluation of evolved sequences across MSR models

## Question

The evolutionary algorithm optimized promoter sequences against **one** MSR model
(`..._A_thaliana_...`). Are the introduced changes model-specific artifacts, or do
other models from the *same* training run agree that the sequence improved?

Concretely: re-score every sequence on a run's Pareto front with all sibling MSR
models and compare their predictions against the fitness the optimization model
reported.

## Data

Two runs, both under
`/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/GOF_LOF/`:

| run | genes | objective (`weights`) |
|---|---|---|
| `GOF/GOF_single_mutation_251009_121226_109368` | 105 | `[1.0, -1.0]` (maximize prediction) |
| `LOF/LOF_single_mutation_251020_180028_564570` | 68 | `[-1.0, -1.0]` (minimize prediction) |

Per gene folder:

- `parameters.json` — `models[*].path` (the optimization model),
  `mutation_start` / `mutation_end` (0 / 3020 in both runs), `max_number_mutations`
  (90 in both runs), `sequence_name` (used as `gene_id`), `reference_sequence`.
- `reference_sequence.fa` — records `reference_sequence_full` (3020 bp) and
  `reference_sequence_mutation_window_0_3020`.
- `saved_populations/pareto_front.json` — list of `[sequence, fitness, mutation_count]`.

Models: `/home/gernot/Code/PhD_Code/Evolution/models/MSR_models_M0X0.75` holds 12
`.h5` MSR models from training run `250705_070641`, one per species
(`A_thaliana` … `Z_mays`). Every gene in both runs used the `A_thaliana` model, so
there are **11 "other" models** per gene.

### Facts established from the data (drive the decisions below)

- Pareto-front sequences are already **full length** (3020 bp) and the mutation
  window spans the whole sequence, so sequence reconstruction is a no-op for these
  two runs. It is still implemented, guarded by a length check, so the script works
  on runs with a narrower window.
- **LOF fronts are clean**: exactly 91 entries per gene, one per mutation count 0–90.
- **GOF fronts are not**: 231 054 entries across 105 genes (median 1911, max 8526 per
  gene) covering only 47–91 *distinct* mutation counts each. The extras are exact
  float ties at saturation — e.g. one gene has 221 distinct sequences all at
  mutation count 56 with fitness `0.999999702`. Only one representative per mutation
  count is kept.
- GOF fronts have **gaps** in mutation count (65 of 91 counts present in one gene),
  handled by expansion (see below).

## Decisions

1. **Ties → one representative per (gene, mutation count).** Reuse
   `deduplicate_pareto_front` from `src/analysis/overview/simple_result_stats.py`:
   it sorts by mutation count ascending and keeps the first entry per count, which
   is exactly the representative rule. Reduces ~237k entries to ~13k sequences.
   Within one mutation count the tied fitness values are identical in both runs, so the
   choice of representative is immaterial there; a warning is emitted if a future run
   violates that, because the representative would then be an arbitrary pick among
   genuinely different individuals.
2. **All 12 models are predicted, including the optimization model.** The
   optimization model is recorded in an `optimization_model` column rather than
   encoded in column names, so the CSV header is constant even if a run mixes
   optimization models. `prediction_<optimization_model>` must reproduce
   `original_fitness` to float precision — a free end-to-end validation of
   reconstruction and one-hot encoding. `reevaluate.py` reports the maximum absolute
   difference. `plot.py` masks that column out of the "other models" band, so the
   figure is unaffected. Extra inference cost is 1/12.
3. **Normalization for pooling** (per gene). Let `f_ref(m)` be the optimization
   model's stored fitness at mutation count `m`, `a = min_m f_ref(m)`,
   `b = max_m f_ref(m)`, and `p_X(m)` the prediction of model `X`. Then

   ```
   norm_X(m) = (p_X(m) - a) / (b - a)
   ```

   `f_ref` maps exactly onto `[0, 1]`; the other models are shifted by the *same*
   `a` and are therefore not bounded to `[0, 1]`. Keeping the shared baseline (rather
   than each model's own `p_X(0)`) is deliberate: it leaves visible the case where a
   model scores the *unmutated* reference sequence differently from the optimization
   model.
4. **Front expansion happens in `plot.py`, not in `reevaluate.py`.** The CSV holds
   only sequences that were actually evaluated, with honest gaps in `mutation_count`;
   filling the grid is a pooling decision. Per gene, reindex onto `0 … max_number_mutations`
   and forward-fill **whole rows** — sequence and all 12 predictions from the largest
   count below the gap. This is the row-level analogue of `expand_pareto_front`, which
   fills a gap with `("", fitness(m-1), m)`: the front is monotone, so a gap means the
   algorithm found nothing strictly better at `m`, and carrying the sequence forward is
   the same argument that justifies carrying the fitness forward. `expand_pareto_front`
   cannot be called directly because it discards everything but
   `(sequence, fitness, count)`; a unit test therefore asserts the expanded
   `original_fitness` series equals `[item[1] for item in expand_pareto_front(front, max_number_mutations)]`
   on real front data, pinning this implementation to the existing one.

   After expansion every gene contributes at every mutation count, so the pooled band
   is comparable across the whole x range. Filled rows are marked with an
   `is_expanded` column and `plot.py` reports what fraction of the plotted grid was
   carried forward, and from which mutation count on the majority of genes are being
   carried forward — a band in that region reflects forward-filled values rather than
   independent measurements and must not be read as consensus.

## Deliverables

```
src/workflows/adversarial/
  PLAN.md
  __init__.py
  reevaluate.py     # models folder + run folders -> one CSV per run
  plot.py           # CSV -> pooled figure + example-gene figures
  outputs/<run_basename>__<models_folder_basename>/input.csv
                                                  /adversarial_predictions.csv
                                                  /mean_normalized_gap.csv
                                                  /pooled.pdf
                                                  /examples.pdf
test/workflows/adversarial/
  __init__.py
  test_reevaluate.py
  test_plot.py
```

### `reevaluate.py`

CLI (all defaulted, so a bare invocation reproduces the analysis):

```
--models-folder  /home/gernot/Code/PhD_Code/Evolution/models/MSR_models_M0X0.75
--run-folders    .../GOF_LOF/GOF/GOF_single_mutation_251009_121226_109368
                 .../GOF_LOF/LOF/LOF_single_mutation_251020_180028_564570
--output-root    src/workflows/adversarial/outputs
```

Output path derived from **both** input folders,
`<output-root>/<run_basename>__<models_folder_basename>/`, so re-evaluating the same run
against a different set of models cannot overwrite an existing result.

Alongside the predictions, `input.csv` records what produced the folder, as `key,value`
rows: `timestamp`, `run_folder`, `models_folder`, `number_of_models`,
`number_of_genes`, `number_of_sequences`, `optimization_models`, `analysis_commit`
(this repository's HEAD, or `unknown`), and one `model_sha256:<model_name>` row per
model. The checksums are the point of the record: a model file can be replaced in place
while keeping its name, in which case the folder name alone would not identify what was
actually used.

Steps:

1. Glob `*.h5` in the models folder → `{stem: path}`; load each once via
   `evolution.load_models.get_model_loader("tensorflow")`.
2. Per gene folder: read `parameters.json`, `reference_sequence.fa`
   (`reference_sequence_full`), `saved_populations/pareto_front.json`.
3. `deduplicate_pareto_front` → one entry per mutation count.
4. Reconstruct the full sequence: use as-is when
   `len(sequence) == len(reference_sequence)`, otherwise splice into
   `reference[mutation_start:mutation_end]`; raise when the length matches neither.
5. One-hot encode with `evolution.sequences.one_hot_encode` (≈91 × 3020 × 4 per gene,
   so per-gene batching is memory-trivial) and predict with all 12 models.
6. Append rows; write CSV once per run folder. Report
   `max |prediction_<optimization_model> - original_fitness|`.

CSV columns:

```
gene_id, sequence, mutation_count, max_number_mutations, optimization_model,
original_fitness, prediction_<stem_1>, ..., prediction_<stem_12>
```

`gene_id` is the bare gene identifier (`AT1G01720`), reduced by `clean_gene_id` from
the run's `sequence_name` (`1_AT1G01720_gene:267992-269819`). The cleaning is
idempotent and `plot.py` applies it when reading, so tables written before it was
introduced still plot with clean identifiers without being regenerated.

Cost: ~13k sequences × 12 models ≈ 1.6×10^5 forward passes, ~1300 `predict` calls.
Minutes on GPU.

### `plot.py`

CLI: `--csv` (defaults to the two paths above), `--example-genes` (optional
explicit gene ids), `--output-format` (`pdf`),
`--min-fitness-range` (default 0).

#### Known limitation of the normalization

The normalization divides by `b - a`, the range the optimization model achieved for
that gene. In the GOF run **42 of 105 genes have `b - a < 0.1`** and 10 have
`b - a < 0.01`, because those genes were already near-saturated before optimization
(median fitness at 0 mutations is 0.816, q75 = 0.957). For them, a small absolute
disagreement of another model becomes an enormous normalized value — the raw min-max
band reached -25. Consequences, both handled rather than hidden:

- The pooled figure's y axis is limited to the interquartile band plus the
  optimization model's line, and states in the plot whenever the min-max band runs off
  axis and how far. Without this, the informative part of the figure occupied a few
  pixels.
- `--min-fitness-range` drops genes below a chosen range. It defaults to 0, so nothing
  is filtered unless asked for. Worth using: the largest-gap example gene selected for
  GOF (gap 1.08) is a near-saturated gene whose raw predictions all agree closely (they
  span 0.87-1.0) — its large gap is an artifact of the small denominator, not real model
  disagreement.

The LOF run is not affected in practice (median `b - a` = 0.888, only 3 of 68 genes
below 0.1).

- Expand each gene's front onto `0 … max_number_mutations` (decision 4).
- Normalize per gene (decision 3).
- **`pooled.pdf`** — x = mutation count, y = normalized prediction.
  Bold line: normalized `original_fitness`, median over genes.
  Band over the 11 other models pooled across models × genes: median line,
  interquartile range as a solid band, min–max as a lighter band.
- **`examples.pdf`** — 3 genes, **raw** (unnormalized) predictions, same band
  structure. Genes are taken at evenly spaced ranks of the mean normalized gap, so for
  three genes that is the gene with the **smallest, median and largest** gap;
  overridable with `--example-genes`. A random draw was tried instead and reverted:
  showing the full range of behaviour is more informative than a typical case.
- **`mean_normalized_gap.csv`** — the gap of every gene, so genes can be picked
  deliberately after looking at the distribution.

#### The mean normalized gap

For one gene, let

- `m` be the mutation count, running over the expanded grid `0 ... M` with `M = 90`;
- `f(m)` the optimization model's Pareto front fitness at `m` (gaps forward-filled);
- `a = min_m f(m)` and `b = max_m f(m)`, so `nu(x) = (x - a) / (b - a)` is the gene's
  normalization and `nu(f(m))` spans exactly `[0, 1]`;
- `S` the set of the 11 models not used for the optimization, and `p_s(m)` the
  prediction of model `s` in `S` on the sequence at mutation count `m`;
- `mu(m) = median over s in S of nu(p_s(m))` the other models' normalized median.

Then the per-mutation-count gap and the gene's value are

```
d(m) = | mu(m) - nu(f(m)) |
mean_normalized_gap = (1 / (M + 1)) * sum over m = 0 ... M of d(m)
```

That is: normalize, take the median **over models** at each mutation count, take the
absolute difference to the optimization model's own normalized curve, then average
**over mutation counts**. Since `nu` is affine and increasing, normalizing before or
after the median gives the same value.

The name states the construction rather than an interpretation, and larger means worse:
0 means the other models' median traces the front exactly, 1 means the median sits, on
average over the front, a full achieved range `b - a` away from where the optimization
model claims to be. It was originally called the "agreement score", which was renamed
because larger values indicate *less* agreement.

Two properties to keep in mind:

- An earlier version used `d(M)` alone. It was discarded because the optimization model
  saturates long before mutation count 90 and the other models catch up eventually, so
  `d(M)` was ~0 for almost every gene and did not separate a model that tracks the front
  from one that lags badly. Averaging over the whole front captures the lag, which is
  where the models actually differ.
- The average runs over the expanded grid, so a gene whose front stops early
  contributes a constant `d` over its forward-filled tail — for a GOF gene ending at
  mutation count 67, 23 of 91 grid points repeat `d(67)`. Genes with short fronts
  therefore weight their late-front gap more heavily.

### Tests

`unittest`, mocking model loading and using a temporary run folder:

- `test_reevaluate.py` — model discovery and optimization-model exclusion bookkeeping;
  reconstruction for full-length and windowed sequences, plus the length-mismatch
  error; one-representative-per-mutation-count reduction; CSV schema.
- `test_plot.py` — front expansion against `expand_pareto_front`; the normalization
  formula (`f_ref` → exactly `[0, 1]`, others unbounded); quartile/min-max band
  computation; example-gene selection.

### Housekeeping

Add to `.gitignore`:

```
src/workflows/adversarial/**/*.csv
src/workflows/adversarial/**/*.pdf
src/workflows/adversarial/**/*.png
```
