# Plan: Flexible mutation-count / gene selection for the TFBS annotation pipeline

## Context

`src/analysis/motives/` runs a 4-stage pipeline that finds transcription-factor
binding sites (TFBS) in DNA sequences **before** and **after** evolutionary
optimization. The "after" sequence is currently hardcoded to the Pareto-front
entry with the **highest mutation count** (`_get_max_mutation_entry`,
`deepcis_scanner.py:203`, used at line 349), labeled `max_mutated`.

For the manuscript we need quick, targeted checks:

1. Select the "after" sequence at a **specific mutation count**, not only the max.
2. Restrict the (expensive) scan to a **subset of genes** from the start.
3. **Control which TFs** appear in the final line plots.

Design decisions from the grilling session:

- **One selection per run** — a single target count (or "max" default).
- **Exact match only** — if a gene's Pareto front has no entry at exactly the
  requested count, skip that gene and log it.
- **Rename** the after-label `max_mutated` → `optimized` across all modules for
  unambiguous intermediate outputs. (Accepted cost: existing scan CSVs become
  incompatible and must be re-scanned.)
- **Encode the mutation count in output filenames** (via the analysis name in
  the shell), not necessarily in the dataframe.
- **Gene subset enforced at the scan stage** (plus existing downstream filters).
- **TF control** is already satisfied by `deepcis_visualize --tfs`; the only fix
  is to stop `--random-subset` from silently overriding explicit selections.

The pipeline is driven by `analysis_scripts/run_deepcis_peak_pipeline.sh` (NOT
`run_full_analysis.sh`, which runs the unrelated overview+mutations pipeline —
confirmed with the user).

## Changes

### 1. `deepcis_scanner.py` — count-based selection + gene subset

- Add `_get_entry_by_mutation_count(pareto_front, target) -> ParetoEntry` that
  returns the entry where `round(entry[2]) == target`, and raises `ValueError`
  when none exists. Keep `_get_max_mutation_entry` for the default path.
- `scan_single_gene_folder(...)`: add `mutation_count: Optional[int] = None`.
  When `None`, use `_get_max_mutation_entry` (current behavior); otherwise use
  `_get_entry_by_mutation_count`. Rename the label at line 354 and its docstring
  from `"max_mutated"` to `"optimized"`.
- `scan_all_genes(...)`: add `mutation_count: Optional[int] = None` and
  `genes: Optional[List[str]] = None`. Filter `gene_folders` (line 423) by
  basename when `genes` is given, and log any requested gene not found. The
  existing `try/except` around `scan_single_gene_folder` (line 444) already
  turns the `ValueError` from a missing exact count into a "Skipping gene" log —
  reuse it (no new skip machinery).
- CLI (`parse_arguments`): add `--mutation-count` (int, default `None`) and
  `--genes` (nargs="+", default `None`); thread both into the `scan_all_genes`
  call. Filename encoding is handled by the shell via `--name` (below), so the
  scanner keeps using `--name` verbatim.

### 2. Rename `max_mutated` → `optimized` (coordinated, mechanical)

Producer and consumers must agree, so rename every occurrence of the signal-type
value together (grep-verified sites):

- `peak_scanner.py`: `valid_types` (L84), signal filter (L234), `validate` set
  (L290, L293), `--signal-type` choices + help (L555-556), `all` expansion (L581).
- `analyze_peaks.py`: `diff_calc` source column (L418) and comment (L420) — the
  `optimized - reference` net-change computation.
- `deepcis_visualize.py`: `PEAK_SIGNAL_TYPES` (L18), `DIFFERENCE_DIRECTIONS` and
  the `max_mutated_minus_reference` direction strings (L19, 162, 173, 202-204,
  497, 562-565, 794, 1022-1028), color-map key (L22), signal filters (L185, L245)
  → `optimized` / `optimized_minus_reference` / `reference_minus_optimized`.
- `deepcis_scanner.py`: label + docstring (L328, L354).
- `annotate_individual_mutation.py` (L79): update for consistency even though the
  module raises `NotImplementedError` (low priority).

### 3. `run_deepcis_peak_pipeline.sh` — wire it all together

- Add `--mutation-count` option (var `MUTATION_COUNT`, default empty).
- When set, suffix the analysis name **once** so every downstream filename
  carries it: `ANALYSIS_NAME="${ANALYSIS_NAME}_mut${MUTATION_COUNT}"` (leave the
  name unchanged when unset, preserving current filenames / backward compat).
  This flows automatically into `SCAN_BASENAME` (L235), peak output (L244), and
  viz dir, since all derive from `ANALYSIS_NAME`.
- Step 1 `SCAN_CMD` (L270): forward `--genes "${GENES_ARR[@]}"` (currently only
  peak_scanner/viz get it) and `--mutation-count "$MUTATION_COUNT"` when set.
- Rename in the shell: signal-type validation (L193) and `PEAK_SIGNAL_LABEL` for
  the `all` case (L239 `reference_max_mutated_difference` → `reference_optimized_difference`).
- Step 4 (L370): only append `--random-subset` when **both** `GENES_ARR` and
  `TFS_ARR` are empty, so explicit `--genes`/`--tfs` are plotted in full.
- Update `usage()`/help text for the new options.

### 4. Tests (mandatory — all edited modules have test files)

Update the renamed literals and add coverage in:
`test/analysis/motives/test_deepcis_scanner.py`,
`test_peak_scanner.py`, `test_analyze_peaks.py`, `test_deepcis_visualize.py`,
`test_deepcis_annotation.py`.

New scanner unit tests (mock model/IO, per project test discipline):

- `_get_entry_by_mutation_count` returns the exact-count entry.
- Missing exact count raises `ValueError` → `scan_all_genes` skips that gene and
  logs it; other genes still processed.
- `genes=` filter restricts scanned folders; unknown requested gene is logged.
- `mutation_count=None` reproduces the old max-mutation behavior and now emits
  the `optimized` label.

## Verification (end-to-end, manuscript-oriented)

1. Run the full tests for edited modules:
   `conda run -n deepCREshap python -m pytest test/analysis/motives/`
2. Real quick check on a real run folder, single gene + specific count:
   `bash analysis_scripts/run_deepcis_peak_pipeline.sh <run> myrun <out> \
     --mutation-count 30 --genes "GENE_A" --tfs "WRKY bHLH"`
   Confirm:
   - Scan CSV is named `deepcis_window_scan_myrun_mut30.csv` and contains
     `sequence_type` values `reference` and `optimized` (no `max_mutated`).
   - Only `GENE_A` is present (scan stage actually filtered).
   - A gene lacking exactly 30 mutations is reported as skipped in the log.
   - Final line plots show `GENE_A` with `WRKY`/`bHLH` only (no random subset).
3. Backward-compat check: run without `--mutation-count`/`--genes` and confirm
   filenames/behavior match the previous max-mutation run (except the
   `max_mutated`→`optimized` label, which is the intended, documented change).
