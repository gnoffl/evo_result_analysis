# Plan: bHLH analog of `starrseq_deepcre_correlation_WRKY.py` via shared `_common.py`

## Context

The STARR-seq × deepCRE correlation analysis at
`src/workflows/overlap_analysis/starrseq_deepcre_correlation_WRKY.py` produces
all the bucketed scatter/fit/delta plots and the bucket-fit-parameter CSVs for
the WRKY transcription factor. The user wants the **same outputs** for bHLH,
using the bHLH dCIS in-silico mutated FASTA and the freshly-generated
`bHLH_dCIS_dCRE_overlaps.csv` (whose positions are calculated correctly, unlike
the WRKY mapping). Two things make this not a pure copy-paste:

1. **No precomputed deepCRE run folder exists for bHLH.** WRKY's
   `load_relevant_deepcre_window_candidates` walks
   `…/all_positions_all/<gene>/` to pick up each gene's 3020 bp reference
   sequence and a `ref_fitness` from `pareto_front.json`. For bHLH the
   reference sequences are produced fresh by invoking
   `evolution/extract_sequences.py` via its CLI, and `ref_fitness` is a raw
   `model.predict` on each reference (confirmed identical to
   `pareto_data[-1][1]`).
2. **The bHLH mapping CSV's columns differ** (`site_id`/`gene_id`/`strand`/
   `region`/`start`/`end`/`additional_padding`) from WRKY's
   (`deepCIS_segment`/`Gene_ID`/…), and per-(site, gene) rows can appear twice
   on short genes.

The user explicitly wants maximum reuse of the existing WRKY logic and has
chosen the **B1 approach**: extract shared utilities/plotting into a new
`_common.py`, slim the WRKY script down to its TF-specific orchestration, and
add a sibling bHLH script. Same outputs, deltas included, no behavioral
changes to the shared code during the move.

## Pre-step: produce the bHLH reference-sequences FASTA (automated in the bHLH script)

The bHLH script invokes `evolution/extract_sequences.py` via its CLI as a
subprocess on first run and **skips on subsequent runs when the output FASTA
already exists**. No manual invocation. Required because the bHLH pipeline
has no equivalent of WRKY's `DEEP_CRE_RUN_FOLDER`.

New helper in `starrseq_deepcre_correlation_bHLH.py`:

```python
def ensure_refs_fasta(
    refs_path: str,
    genes_json_path: str,
    mapping_csv_path: str,
    genome_fasta: str,
    gtf: str,
) -> None:
    if os.path.exists(refs_path):
        return  # idempotent: assume the file is good if present
    if not os.path.exists(genes_json_path):
        ids = sorted(set(pd.read_csv(mapping_csv_path)["gene_id"]))
        with open(genes_json_path, "w") as fh:
            json.dump(ids, fh)
    subprocess.run(
        [
            sys.executable, "-m", "evolution.extract_sequences",
            "-f", genome_fasta,
            "-a", gtf,
            "-o", refs_path,
            "--genes_of_interest", genes_json_path,
        ],
        check=True,
    )
```

Called from `main()` before `load_bhlh_window_candidates`. `extract_sequences.py`
defaults (`--intragenic 500`, `--extragenic 1000`) already match the rest of
the pipeline.

Header format produced (verified in `extract_sequences.py:397/400`):
`>{chrom}_{gene_id}_gene:{start}-{end}` — identical to the WRKY folder-name
format the existing loader already parses (`+` strand: `start<end`; `-`
strand: `start>end`).

Note: `extract_sequences.py:442`'s `args.overwrite == "false"` compares a bool
to a string, so the script always overwrites when invoked. The existence
check at the top of `ensure_refs_fasta` is what actually prevents accidental
re-extraction across runs.

## Refactor 1 — extract shared logic into `_common.py`

**New file:** `src/workflows/overlap_analysis/_common.py`. Move-only. No
function bodies edited except for the four small signature changes listed
below.

Move these from `starrseq_deepcre_correlation_WRKY.py`:

- **Constants** (`:28–66`): `INTRAGENIC`, `EXTRAGENIC`, `PADDING`,
  `PADDING_START`, `PADDING_END`, `FULL_SEQUENCE_LENGTH`,
  `FULL_OVERLAP_LENGTH`, `BUCKET_SIZE`, `EDGE_POSITIONS`,
  `BUCKET_SUMMARY_FILE_NAME`, `OUTPUT_DPI`, `ENABLE_BUCKETED_ANALYSIS`,
  `ADD_OVERALL_FIT_TO_BUCKETED_PLOTS`, `INDIVIDUAL_BUCKET_LABELS_TO_PLOT`,
  `BUCKET_COLOR_PALETTE`, `POSITION_SERIES_COLORS`, `PARETO_PATH`,
  `REF_SEQ_FILE`, `DEEPCRE_PATH`. The `mpl.rcParams.update(...)` block
  (`:69–80`) is **not** kept at module top — it moves into
  `configure_matplotlib()` (see signature changes below) so importing
  `_common` has no global matplotlib side effect.
- **Helpers** (`:83–198`): `_fit_linear_model`, `_format_bucket_label`,
  `_darken_hex_color`, `_sanitize_bucket_label`, `_get_analysis_output_dir`,
  `add_length_corrected_overlap_buckets`, `_deduplicate_bucket_rows`,
  `_make_stats_row`.
- **Plotting** (`:201–810`): `_plot_bucketed_correlation`,
  `_plot_individual_bucket_correlation`, `plot_individual_bucket_views`,
  `plot_deepcre_starrseq_correlation`, `plot_mutation_starrseq_correlation`,
  `plot_mutation_deepcre_correlation`, `simply_plot_multi`,
  `plot_correlation_over_positions_fixed_window`,
  `plot_correlation_over_positions_fixed_number_elements`.
- **Pipeline** (`:430–850`): `load_starrseq_data` (TF-agnostic — verified
  for bHLH headers), `reverse_complement`, `find_overlap_positions`,
  `compare_sequences`, `get_starrseq_fragment`, `build_sequences`,
  `make_deepcre_predictions`, `merge_with_starrseq_results`,
  `calculate_deltas`, `save_bucket_statistics`.

Four **small signature changes** to `_common.py` (the only behavioral edits in
the refactor, each isolated and trivially verifiable):

1. **`map_starrseq_to_deepcre(starrseq_data, gene_data, site_to_gene_ids)`** —
   takes a precomputed `Dict[str, List[str]]` instead of a `mapping_candidates`
   DataFrame + the TF-specific `get_gene_candidates_for_starrseq_entry`. The
   per-TF candidate lookup logic moves out. Body change: replace line 549's
   call with `gene_ids = site_to_gene_ids.get(search_key, [])` followed by the
   existing `[gene for gene in gene_data if gene["gene"] in gene_ids]` filter.
2. **`_common.CORRELATION_OUTPUT_ROOT: str = ""`** (module-mutable) with a
   small setter and a guard in `_get_analysis_output_dir`:

   ```python
   CORRELATION_OUTPUT_ROOT: str = ""

   def set_output_root(root: str) -> None:
       global CORRELATION_OUTPUT_ROOT
       CORRELATION_OUTPUT_ROOT = root

   def _get_analysis_output_dir(analysis_name: str) -> str:
       if not CORRELATION_OUTPUT_ROOT:
           raise RuntimeError("call set_output_root() before plotting")
       ...
   ```

   This converts a silent "writes PNGs under cwd" failure mode into an
   immediate crash and avoids threading `output_root` through ~14 plotting
   signatures (which would inflate the diff and break the byte-stable
   bucket-CSV verification).
3. **`configure_matplotlib() -> None`** — wraps the `mpl.rcParams.update(...)`
   block from the original WRKY script top. Each TF script's `main()` calls
   `_common.configure_matplotlib()` once at startup. Stops `_common` from
   mutating global matplotlib state at import time.
4. **`build_sequences(mapping_results, max_differences=15)`** — adds a
   default-15 keyword arg around the existing `differences <= 15` check
   (`:606`). WRKY uses the default; the bHLH script passes
   `max_differences=16` (the bHLH core motif is 6 nt, so one extra mutation
   is tolerated vs. WRKY's 5 nt motif). The existing
   `print(f"Skipping … due to high number of differences ({differences})…")`
   warning at `:612` is preserved for both, so any variant still exceeding
   the threshold is logged before being dropped.

**Not moved** (stays in WRKY's file): `load_relevant_deepcre_window_candidates`
(folder-walking, WRKY-only), `get_gene_candidates_for_starrseq_entry` (deleted
— replaced by the dict-build in `main`).

**Not moved** (stays as TF-script concern): `USE_NEW_DATA` and the
`_DATA_VERSION` / output-root construction. Each TF script owns its own.

## Refactor 2 — trim `starrseq_deepcre_correlation_WRKY.py`

After the move it contains only WRKY-specific things:

- `BASE_DIR`, `DATA_DIR`, `USE_NEW_DATA`, `STARRSEQ_INPUT_FILE`,
  `DEEP_CRE_RUN_FOLDER`, `_STARR_SEQ_OLD`, `_STARR_SEQ_NEW`,
  `STARR_SEQ_RESULTS`, `MAPPING_FILE`, `_DATA_VERSION`.
- `load_relevant_deepcre_window_candidates(run_folder)` unchanged.
- `main()` near-identical to before, with three diffs:
  1. `_common.configure_matplotlib()` at the very top (replaces the
     module-level `mpl.rcParams.update(...)` block).
  2. `_common.set_output_root(os.path.join(BASE_DIR, "correlation", _DATA_VERSION))`
     right after.
  3. Replace `mapping_results = map_starrseq_to_deepcre(starrseq_data, gene_data, mapping_candidates)` with:

     ```python
     mapping_candidates = pd.read_csv(MAPPING_FILE)
     site_to_gene_ids = (
         mapping_candidates.groupby("deepCIS_segment")["Gene_ID"]
         .apply(list)
         .to_dict()
     )
     mapping_results = _common.map_starrseq_to_deepcre(
         starrseq_data, gene_data, site_to_gene_ids,
     )
     ```

All other names in `main()` resolve through `from . import _common` /
`from ._common import ...`.

## New file — `src/workflows/overlap_analysis/starrseq_deepcre_correlation_bHLH.py`

```text
BASE_DIR / DATA_DIR
STARRSEQ_INPUT_FILE     = data/dCIS_bHLH_in_silico_mutated_GS2025d.fasta
MAPPING_FILE            = data/bHLH_dCIS_dCRE_overlaps.csv
REFS_FASTA_PATH         = data/bHLH_reference_sequences.fa   # auto-produced by ensure_refs_fasta()
GENES_JSON_PATH         = data/bHLH_genes_of_interest.json   # auto-produced by ensure_refs_fasta()
STARR_SEQ_RESULTS       = data/plantstarr-seq_main_light_and_dark_simon_gernot.csv  # shared with WRKY
GENOME_FASTA_PATH       = /home/gernot/ARCitect/.../Arabidopsis_thaliana.TAIR10.dna.toplevel.fa
GTF_PATH                = /home/gernot/ARCitect/.../Arabidopsis_thaliana.TAIR10.52.gtf
CORRELATION_OUTPUT_ROOT = os.path.join(BASE_DIR, "correlation_bHLH", "new")
```

The bHLH script intentionally **omits the WRKY `USE_NEW_DATA` toggle** — only
the new light+dark STARR-seq CSV is consumed. (The old CSV also contains
bHLH rows but isn't used; no `correlation_bHLH/old/` path is generated.)

**New function** `load_bhlh_window_candidates(refs_fasta_path, model_path) -> List[Dict]`:

1. Open `refs_fasta_path` with `pyfaidx.Fasta`.
2. For each record: parse header `{chrom}_{gene_id}_gene:{start}-{end}`
   using the same logic WRKY's loader uses (`split("_")[1]` →
   `gene_id`, `split("_")[2].split(":")[1].split("-")` → `start, end`;
   swap if `start > end` for minus-strand entries). Build a `gene_data`
   entry `{"gene": gene_id, "ref_seq": str(record), "start": int(start),
   "end": int(end), "folder_name": <header>, "ref_fitness": <filled below>}`.
3. **Batch ref_fitness** in one pass: `np.array([one_hot_encode(g["ref_seq"]) for g in gene_data])`
   → `model.predict(...)` → assign `ref_fitness` per gene. Single forward
   pass; ~863 sequences × 3020 bp ≈ trivial.

**`main()`** mirrors WRKY's exactly except for the loader, the dict build,
and the bHLH-specific `max_differences=16`:

```python
_common.configure_matplotlib()
_common.set_output_root(CORRELATION_OUTPUT_ROOT)
ensure_refs_fasta(REFS_FASTA_PATH, GENES_JSON_PATH, MAPPING_FILE,
                  GENOME_FASTA_PATH, GTF_PATH)
starrseq_data      = _common.load_starrseq_data(STARRSEQ_INPUT_FILE)
gene_data          = load_bhlh_window_candidates(REFS_FASTA_PATH, _common.DEEPCRE_PATH)
mapping_candidates = pd.read_csv(MAPPING_FILE)
site_to_gene_ids = (
    mapping_candidates.groupby("site_id")["gene_id"]
    .apply(lambda s: list(s.unique()))
    .to_dict()
)
starr_seq_results = pd.read_csv(STARR_SEQ_RESULTS)
mapping_results   = _common.map_starrseq_to_deepcre(starrseq_data, gene_data, site_to_gene_ids)
seqs, meta_data   = _common.build_sequences(mapping_results, max_differences=16)
# … rest is byte-for-byte identical to WRKY's main from
# `make_deepcre_predictions` onward.
```

**Notes on bHLH-specific behavior:**

- The `differences <= N` filter in `build_sequences` uses `N=15` for WRKY
  (5 nt core motif) and `N=16` for bHLH (6 nt core motif, so one additional
  mutation is tolerated). `build_sequences` is parameterized to accept
  `max_differences`; WRKY uses the default 15, the bHLH script passes 16.
  The existing skip-print warning at `:612` is preserved for both, so any
  variant that still exceeds the threshold is logged before being dropped.
- Short genes where the same site overlaps both the promoter and terminator
  window are handled implicitly during alignment: site sequences are 170 bp
  and `find_overlap_positions` rejects matches shorter than 100 bp
  (`:532`), so at most one of the two windows can produce a valid alignment
  for a given site. `.unique()` at `site_to_gene_ids` dict-build time
  collapses the duplicate (site, gene) rows in the mapping CSV (32 such
  pairs) so the alignment is attempted only once per pair, and the right
  window wins naturally. The mapping CSV's `region` column is therefore not
  threaded downstream — it would be redundant with the alignment-derived
  `start_pos` (promoter if `< PADDING_START`, terminator if
  `>= PADDING_END`).

## Tests

**Update:** `test/workflows/test_overlap_analysis.py` (the existing WRKY
tests).

- Change the top-level import from
  `from workflows.overlap_analysis import starrseq_deepcre_correlation_WRKY as overlap_analysis`
  to `from workflows.overlap_analysis import _common as overlap_analysis`.
  Both functions exercised by the file
  (`add_length_corrected_overlap_buckets` and `save_bucket_statistics`)
  move into `_common.py` body-identical — so no other test edits should be
  needed beyond the import.
- Add a new test for `_common.map_starrseq_to_deepcre` exercising the new
  `site_to_gene_ids` dict signature with a tiny in-memory dict and a pair
  of mock `gene_data` entries — locks in the refactored interface before
  the bHLH script depends on it.
- Add a new test for `_common.build_sequences(max_differences=N)`
  parameterization: a sequence with exactly `N` differences is kept, one
  with `N+1` is dropped. Run with both `N=15` and `N=16` to cover both
  TFs.

**Add:** `test/workflows/overlap_analysis/test_correlation_bhlh.py`
(unittest, Arrange-Act-Assert, mocked filesystem & model). Cover:

- `ensure_refs_fasta` short-circuits when `refs_path` already exists
  (subprocess is **not** called) and runs the patched
  `subprocess.run(["python", "-m", "evolution.extract_sequences", ...])`
  when it doesn't, writing the genes JSON on the way.
- `load_bhlh_window_candidates` parses a mock FASTA with one `+`-strand and
  one `-`-strand record, returns the expected `(gene, start, end, ref_seq)`
  with `start <= end` after the swap, and populates `ref_fitness` from a
  patched `load_model`/`predict`.
- bHLH `site_to_gene_ids` build from a mock mapping CSV containing one
  duplicated `(site_id, gene_id)` row dedups to a single entry.

The remaining WRKY pipeline behavior is verified via the output-diff step
in §Verification, not via additional unit tests.

## Verification

1. **Activate `deepCREshap`**.
2. **Snapshot WRKY's current outputs** before any code change:
   `cp -r src/workflows/overlap_analysis/correlation/new /tmp/wrky_pre_refactor`.
3. **Do the refactor**, then **re-run WRKY**:
   `python -m src.workflows.overlap_analysis.starrseq_deepcre_correlation_WRKY`.
4. **Diff WRKY outputs**:
   `diff -r src/workflows/overlap_analysis/correlation/new /tmp/wrky_pre_refactor`.
   The bucket-summary CSVs (`overlap_bucket_fit_parameters.csv`) are
   byte-stable and must match exactly. PNG bytes may differ (matplotlib
   timestamps); inspecting one or two visually is sufficient.
5. **Run all tests**:
   `python -m pytest test/workflows/test_overlap_analysis.py test/workflows/overlap_analysis/`.
6. **Run the bHLH script** end-to-end (extracts refs on first call via
   `ensure_refs_fasta`):
   `python -m src.workflows.overlap_analysis.starrseq_deepcre_correlation_bHLH`.
   Inspect the printed summary; visually compare one bucket plot between
   WRKY and bHLH to confirm structural analogy.
7. Hand off to the user for the end-to-end smoke check.

## Critical files

- **Modify**: `src/workflows/overlap_analysis/starrseq_deepcre_correlation_WRKY.py`
  (gut to TF-specific shell; ~600→~80 lines).
- **Modify**: `test/workflows/test_overlap_analysis.py` (update import to
  `_common`; add the two new interface tests above).
- **New**: `src/workflows/overlap_analysis/_common.py` (moved logic + the
  four signature changes above).
- **New**: `src/workflows/overlap_analysis/starrseq_deepcre_correlation_bHLH.py`.
- **New**: `test/workflows/overlap_analysis/test_correlation_bhlh.py`.
- **Generated at first run** (no commit needed):
  `src/workflows/overlap_analysis/data/bHLH_genes_of_interest.json` and
  `src/workflows/overlap_analysis/data/bHLH_reference_sequences.fa`, both
  written by `ensure_refs_fasta()` inside the bHLH script.
- **Reused untouched**: `evolution/extract_sequences.py` (CLI),
  `src/workflows/overlap_analysis/bHLH_overlaps/generate_bhlh_overlaps.py`
  (already produces the mapping CSV; no changes needed).

## Decisions confirmed with the user

- **`pareto_data[-1][1]` ≡ `model.predict` on the reference** — confirmed
  identical. WRKY and bHLH `delta_prediction` are on the same scale; no
  spot-check needed.
- **`data/short_genes.json`** is safe to ignore for bHLH (no hidden
  dependency).
- **Output directory**: `correlation_bHLH/new/` (sibling of
  `correlation/new/`).
- **`USE_NEW_DATA` for bHLH**: omitted. The old light-dark CSV also
  contains bHLH rows, but the bHLH script only consumes the new CSV. No
  toggle, no `correlation_bHLH/old/` path.
- **`mpl.rcParams` side effect**: wrapped in `_common.configure_matplotlib()`.
  Importing `_common` has no global matplotlib side effect; each TF
  script's `main()` calls the configuration function once at startup.
- **bHLH difference filter**: 16 (vs. WRKY's 15), reflecting the 6 nt vs.
  5 nt core motif length difference. The existing skip-print warning is
  preserved.
- **Refs FASTA extraction**: invoked automatically by the bHLH script via
  `ensure_refs_fasta()`, with an existence check so the slow extraction
  runs only on first use.

## Risks & mitigations

| Risk | Catch |
| --- | --- |
| Plotting fn called before `set_output_root()` → writes PNGs to cwd. | `_get_analysis_output_dir` guard raises `RuntimeError`. |
| Move-only refactor introduces a behavioral diff in WRKY. | Verification §2 + §4 (byte-stable bucket-summary CSV diff). |
| Forgetting `_common.configure_matplotlib()` in a TF script's `main()` → poster-size fonts not applied; first PNG looks visibly different. | Visual inspection of one PNG in step 4; can be made noisy by adding a `len(mpl.rcParams)`-based assertion if recurring. |
