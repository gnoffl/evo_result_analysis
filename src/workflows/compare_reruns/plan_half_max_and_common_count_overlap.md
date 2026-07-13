# Plan: half-max-point comparison, common-count overlap, and position-only matching for `compare_reruns`

## Context

`compare_runs.py` compares two evolutionary-algorithm runs of the same genes at a
single point: the reference (`front[-1]`) vs. the maximally-mutated endpoint
(`front[0]`). It reports per-gene mutation-set overlap (`shared`/`a_only`/`b_only`,
mutation = `(position, base)`) and fitness/mutation-count deltas, plus pooled
summaries and paired Wilcoxon tests.

Two problems / needs for the manuscript:

1. **Overlap is computed at different mutation counts per run** (each run's own
   `front[0]`), so two runs with different max-mutation counts can never reach 100%
   overlap. Fix: compute the overlap at a **common mutation count**.
2. We want to also compare solutions at the **half-max mutation point** (the solution
   reaching half of a run's fitness journey), reusing existing half-max logic.
3. We want an optional **position-only** mutation match (same position, ignore the
   substituted base).

Everything is exposed via CLI flags; kept deliberately minimal for a fast, checkable
change during manuscript writing.

## Settled decisions

- **New CLI flags**
  - `--comparison-point {endpoint, half_max}` (default `endpoint`).
  - `--position-only` (`store_true`, default off → current `(position, base)` match).
- **Overlap is always computed at a common mutation count** (both fronts sampled at
  the same count), for both modes:
  - `endpoint`: common count = the **largest mutation count present on both fronts**
    (`max(counts_A & counts_B)`). Pareto fronts drop dominated points, so intermediate
    counts have gaps and the two endpoint counts rarely coincide — using
    `min(max_A, max_B)` would skip ~20% of genes. The rounded count sets always share
    0, so this is well defined and never misses.
  - `half_max`: common count = `min(half_max_A, half_max_B)` via
    `calculate_half_max_mutations`. Half-max counts sit in the dense part of the
    front, so a missing count is rare; when it happens the gene is skipped.
- **Reported fitness / n_mutations differ by mode:**
  - `endpoint`: `after_A/after_B` and `n_mutations_A/n_mutations_B` come from the
    **true endpoint** `front[0]` (unchanged). Only the overlap uses the common count.
  - `half_max`: `after_A/after_B` = fitness of each front **at the common count**;
    `n_mutations_A/n_mutations_B` = the common count (so `delta_n_mutations = 0`).
    This honors the decision to *not* compare the half-max counts themselves.
- **Missing common count** (`half_max` only — `endpoint` cannot miss by construction):
  **skip the gene and record it** (gene key + reason). Written to a file and the count
  printed.
- **Outputs go into a per-configuration subfolder of `--output-dir`**, named for the
  comparison point and the match mode: `{comparison_point}` + `"_position_only"` when
  `--position-only` (e.g. `endpoint/`, `half_max/`, `endpoint_position_only/`). Within
  the subfolder the files have fixed names (`per_gene.csv`, `summary.txt`,
  `excluded_genes.txt`, and the figures), so `--output-dir` itself identifies the
  comparison (one `--output-dir` per run pair) and configs never overwrite each other.
  There is no `--name` flag.
- **Do not** relocate helpers to avoid `pyfaidx` — import directly.

## Reused existing code (import directly, do not reimplement)

- `calculate_half_max_mutations(front)` — `src/analysis/overview/simple_result_stats.py:42`
- `get_data_at_mutation_count(front, target_count)` —
  `src/workflows/candidate_selection.py:60` (exact-count `bisect` lookup, raises
  `ValueError` if absent). Import as
  `from workflows.candidate_selection import get_data_at_mutation_count`
  (accepting the `pyfaidx` import it triggers) and
  `from analysis.overview.simple_result_stats import calculate_half_max_mutations`.

## Changes to `compare_runs.py`

Additive except the `build_per_gene_table` return signature and one new CSV column.

1. **Imports:** add the two functions above.

2. **New exception** `GeneComparisonSkipped(Exception)` — carries a reason string;
   distinct from the existing `ValueError` (which still hard-aborts on reference
   mismatch, i.e. incomparable runs).

3. **`mutation_set(reference_sequence, optimized_sequence, position_only=False)`** —
   when `position_only`, return `set[int]` of positions; else the current
   `set[tuple[int, str]]`. Downstream `&`/`-` set algebra unchanged.

4. **New helper `resolve_comparison_members(front_a, front_b, comparison_point)`**
   returning `(reported_a, reported_b, overlap_member_a, overlap_member_b, overlap_count)`,
   raising `GeneComparisonSkipped` if a lookup misses:
   - `endpoint`: `reported_* = front_*[0]`;
     `overlap_count = max({round(m[2]) for m in front_a} & {round(m[2]) for m in front_b})`
     (largest shared count; always defined because both fronts include count 0);
     `overlap_member_* = get_data_at_mutation_count(front_*, overlap_count)`.
   - `half_max`:
     `overlap_count = min(calculate_half_max_mutations(front_a), calculate_half_max_mutations(front_b))`;
     `overlap_member_* = get_data_at_mutation_count(front_*, overlap_count)`;
     `reported_* = overlap_member_*`.
   Wrap the `get_data_at_mutation_count` calls; on `ValueError` raise
   `GeneComparisonSkipped(f"common count {overlap_count} absent on a front")`.

5. **`compare_single_gene(..., comparison_point="endpoint", position_only=False)`** —
   keep the before-validation as-is; get members via `resolve_comparison_members`;
   build record from `reported_*` (fitness, n_mutations, deltas) and compute the
   mutation sets from `overlap_member_*` with `position_only`. Add a new
   `overlap_mutation_count` field (the common count used) for transparency.

6. **`PER_GENE_COLUMNS`** — add `"overlap_mutation_count"` (after `delta_n_mutations`).

7. **`build_per_gene_table(..., comparison_point, position_only)`** — return
   `(per_gene, overlap_counts, excluded_genes)` where `excluded_genes` is
   `list[tuple[str, str]]`; catch `GeneComparisonSkipped` per gene and record instead
   of aborting; `ValueError` still propagates. Update the single call site.

8. **`compute_summary` / `format_summary`** — thread `comparison_point` and
   `position_only` so headers are honest: report the comparison point, the match mode
   ("position + base" vs "position only"), the mean overlap-count used, and the number
   of skipped genes.

9. **Plots** — `render_all_figures` / `plot_mutation_overlap` take `position_only`
   (and a point label) to correct axis/title wording. Keep all 3 figures in both modes.
   Note: in `half_max`, `n_mutations_A == n_mutations_B`, so the mutation-count scatter
   collapses to the `y=x` line (expected, harmless).

10. **`run_comparison(..., comparison_point="endpoint", position_only=False)`** — build
    the per-config subfolder (`{comparison_point}` + optional `_position_only`) inside
    `--output-dir`; write fixed-name outputs there (`per_gene.csv`, `summary.txt`, the
    figures); pass flags through; when `excluded_genes` is non-empty, write
    `excluded_genes.txt` and print the count. No `--name` parameter.

11. **CLI** (`parse_args` / `main`) — drop `--name`; add `--comparison-point` and
    `--position-only`; forward to `run_comparison`.

## Tests (`test/workflows/compare_reruns/test_compare_runs.py`)

Update and extend (unittest; mandatory per project discipline):

- `mutation_set`: position-only cases (same position, different base → shared in
  position-only, distinct in position+base).
- `compare_single_gene`:
  - endpoint: overlap now at `min(max_A, max_B)` — assert `after`/`n_mutations` still
    from `front[0]` while `shared/a_only/b_only` and `overlap_mutation_count` reflect
    the common count. Hand-build fronts where A and B have different max counts.
  - half_max: `after`/`n_mutations` at the common count, `delta_n_mutations == 0`,
    overlap at the common count.
  - `GeneComparisonSkipped` when the common count is absent on one front.
- `build_per_gene_table`: update call sites for the 3-tuple return; add a test that a
  skipped gene appears in `excluded_genes` and not in `per_gene`; assert the new
  `overlap_mutation_count` column.
- `compute_summary` / `format_summary`: assert point/match labels and skipped count
  appear.

## Verification

Run in the `deepCREshap` conda env:

- Unit tests:
  `conda run -n deepCREshap python -m pytest test/workflows/compare_reruns/ -q`
- Endpoint behavior check against real runs: run the CLI with defaults; confirm
  `after_A/after_B` and `n_mutations_A/n_mutations_B` match the previous (committed
  example) values, while the shared-mutation counts/fraction are now **>=** the old
  values (overlap sampled at the common count). Note: committed example CSVs will no
  longer match byte-for-byte (overlap columns change + new column) — that is expected
  and the examples can be regenerated by the user.
- Half-max smoke test: `--comparison-point half_max` — confirm CSV/plots/summary
  produced, `delta_n_mutations == 0`, `overlap_mutation_count` populated, and any
  skipped genes listed in the config subfolder's `excluded_genes.txt`.
- Position-only check: rerun with `--position-only` for both points; confirm shared
  counts are `>=` the position+base run and the config subfolder carries the
  `_position_only` suffix.

## Housekeeping

- Confirm the repo `.gitignore` policy for generated outputs; add a glob for
  `*_excluded_genes.txt` if these should be ignored alongside the CSV/PNG outputs.
