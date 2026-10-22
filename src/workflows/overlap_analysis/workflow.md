# Overlap Analysis Workflow

This folder validates evolutionary-algorithm / deepCIS outputs by asking a single
question for each transcription factor (TF):

> Do **deepCRE binding predictions** for in-silico mutated TF binding sites agree
> with the **experimentally measured STARR-seq enrichment** of the same sequences?

It does this for **WRKY**, for **bHLH**, and for the two **pooled together**.
Every plot and statistics file under `correlation*/` is an answer to that question
for one subset of the data.

This document explains, for *every* file in the folder, **how it was produced** and
**how information flows** from the raw inputs through the code to the outputs.
Implementation minutiae are deliberately omitted; the goal is the data-flow map.

---

## 1. Provenance at a glance

Each file is in exactly one of three categories:

- **External input** — created outside this repository (by me, by a collaborator,
  or a public reference). These are the true sources of truth.
- **Generated** — written by code in this folder (or by an `evolution`/`deepCRE`
  tool the code invokes). Reproducible by re-running the relevant script.
- **Code / docs** — the scripts and design notes themselves.

| File | Category | Produced by |
| --- | --- | --- |
| `data/dCIS_WRKY_in_silico_mutated_GS2025d.fasta` | External input | My in-silico mutagenesis library (WRKY) |
| `data/dCIS_bHLH_in_silico_mutated_GS2025d.fasta` | External input | My in-silico mutagenesis library (bHLH) |
| `data/plantstarr-seq_main_light_and_dark_simon_gernot.csv` | External input | Experimental STARR-seq results (light + dark) |
| `data/plantstarr-seq_main_dark_simon_gernot(1).csv` | External input | Experimental STARR-seq results (dark only, earlier export) |
| `data/WRKY_dCIS_dCRE_overlaps.csv` | External input | Simon's WRKY overlap file (**coords untrustworthy**) |
| `data/wrky_overlaps_reference_mapping.csv` | External input | **Unused / legacy** (not read by any code here) |
| `data/short_genes.json` | External input | **Unused / legacy** (not read by any code here) |
| `data/*.fasta.fai` | Generated | `pyfaidx` index, written on first `Fasta()` open |
| `data/bHLH_dCIS_dCRE_overlaps.csv` | Generated | `bHLH_overlaps/generate_bhlh_overlaps.py` |
| `data/bHLH_genes_of_interest.json` | Generated | `starrseq_deepcre_correlation_bHLH.py` (`ensure_refs_fasta`) |
| `data/bHLH_reference_sequences.fa` (+ `.fai`) | Generated | `ensure_refs_fasta` → `evolution.extract_sequences` |
| `correlation/{old,new}/…` | Generated | `starrseq_deepcre_correlation_WRKY.py` |
| `correlation_bHLH/new/…` | Generated | `starrseq_deepcre_correlation_bHLH.py` |
| `correlation_combined/new/…` | Generated | `starrseq_deepcre_correlation_combined.py` |
| `_common.py` | Code | shared, TF-agnostic library |
| `starrseq_deepcre_correlation_WRKY.py` | Code | WRKY entry point |
| `starrseq_deepcre_correlation_bHLH.py` | Code | bHLH entry point |
| `starrseq_deepcre_correlation_combined.py` | Code | pooled entry point |
| `bHLH_overlaps/generate_bhlh_overlaps.py` | Code | bHLH overlap-mapping generator |
| `bHLH_plan.md`, `bHLH_analysis_plan.md` | Docs | design specs (below) |
| `workflow.md` | Docs | this file |

**External paths referenced by the code but living outside this repo:**

| Path (in code) | What it is |
| --- | --- |
| `DEEPCRE_PATH` (`_common.py`) | Pre-trained deepCRE Keras model (`Atha_S0X0.75…ssr_train….h5`) |
| `DEEP_CRE_RUN_FOLDER` (WRKY script) | An `evolution` run folder; per-gene `reference_sequence.fa` + `saved_populations/pareto_front.json` |
| `GENOME_FASTA_PATH` (bHLH script) | `Arabidopsis_thaliana.TAIR10.dna.toplevel.fa` |
| `GTF_PATH` (bHLH script / generator) | `Arabidopsis_thaliana.TAIR10.52.gtf` (32,833 gene features) |

---

## 2. The shared coordinate frame

Everything hinges on one idea: deepCRE represents each gene as a single
**3020 bp "extracted window"**:

```text
 0                       1500     1520(+padding)                       3020
 |---- promoter (1500) ----|-- N --|------- terminator (1500) ---------|
       INTRAGENIC 500 in  +              INTRAGENIC 500 in  +
       EXTRAGENIC 1000 out             EXTRAGENIC 1000 out
```

(`MIN_PADDING = 20` central N's; short genes get extra central `additional_padding`
so the total stays 3020.) These constants live once in `_common.py`
(`INTRAGENIC=500`, `EXTRAGENIC=1000`, `MIN_PADDING=20`,
`IDEAL_PADDING_START = INTRAGENIC + EXTRAGENIC = 1500`,
`IDEAL_PADDING_END = IDEAL_PADDING_START + MIN_PADDING = 1520`,
`FULL_SEQUENCE_LENGTH=3020`, `FULL_OVERLAP_LENGTH=170`, `BUCKET_SIZE=200`).
The `IDEAL_*` names signal that these positions only hold exactly for genes that are
longer than 1000 bp; short genes shift them — see `adjust_positions` below.
A STARR-seq fragment is 170 bp; an
"overlap" is the region of a gene's 3020 bp window that a fragment maps into, and
overlaps are grouped into 200 bp **buckets** by their start position so that
correlations can be examined as a function of where in the gene the fragment sits.

---

## 3. The code

### 3.1 `_common.py` — the TF-agnostic engine

This module holds every piece of the pipeline that does **not** depend on which TF
is being analysed. The three entry-point scripts only supply TF-specific inputs and
then hand a finished dataframe to `run_correlation_analysis`. Logical sections:

**Constants & setup.** The 3020 bp-frame constants above, plotting style
(`configure_matplotlib`, poster fonts, 600 dpi), and the output root. The output
root is module state set per run via `set_output_root(...)`; `_get_analysis_output_dir`
raises if it was never set, so a misconfigured run crashes instead of silently
writing PNGs to the working directory.

**Input parsing.**

- `load_starrseq_data(fasta)` — reads an in-silico-mutated FASTA and parses each
  header `<tf>_<chrom>:<start>-<end>_<status>_<id>` into a record (TF, genomic
  coords, `binding`/`non_binding`, whether it is a `reference` variant, sequence).
  TF-agnostic: works for both WRKY and bHLH headers.

**Mapping STARR-seq fragments onto gene windows.**

- `adjust_positions(start_pos, end_pos, additional_padding, reverse, len_ref_seq)`
  — computes the per-gene `real_padding_start` and `real_padding_end` by adjusting
  `IDEAL_PADDING_START/END` for the gene's `additional_padding` (the extra central
  N's that short genes receive). The extra bp is split asymmetrically if the
  additional padding is an uneven number: the promoter side always gets the smaller
  fragment of the padding (the odd bp). For forward-strand genes the promoter is on the left
  (positions 0…`real_padding_start`), so the left boundary shifts inward by the
  smaller half; for reverse-strand genes the promoter is on the right, so the right
  boundary shifts outward by the smaller half. The function then validates the overlap
  (not spanning the central pad, ≥100 bp long) and returns the adjusted
  `(start_pos, end_pos, real_padding_start, real_padding_end)`, or `(-1,-1,-1,-1)`
  on rejection.

- `find_overlap_positions(ref_seq, start_query, end_query, additional_padding)` —
  locates the fragment inside a candidate's 3020 bp reference sequence by searching
  the first/last 50 bp (with reverse-complement fallback), then delegates boundary
  adjustment and validation to `adjust_positions`. Returns a 5-tuple
  `(start, end, real_padding_start, real_padding_end, reverse)`. A sentinel of
  `(-1,-1,-1,-1,False)` means the fragment could not be placed.

- `map_starrseq_to_deepcre(starrseq_data, gene_data, site_to_gene_ids)` — for each
  STARR-seq fragment it looks up the candidate gene(s) for that site (the
  `site_to_gene_ids` dict is built per-TF by the caller), reads
  `additional_padding` from each `gene_entry` (defaulting to 0 if absent), and
  calls `find_overlap_positions`. This alignment — not the mapping CSV's stored
  coordinates — is the source of truth for *where* the overlap is. Successful rows
  carry `overlap_start/end`, `real_padding_start`, `real_padding_end`, and `reverse`
  so that downstream steps can use the correct per-gene boundaries.

**Building & scoring the mutated sequences.**

- `get_starrseq_fragment(starr_seq, overlap_start, overlap_end, real_padding_start,
  real_padding_end)` — extracts the right slice of the STARR-seq sequence to splice
  in, using the per-gene `real_padding_start/end` (not the ideal globals) to decide
  whether the overlap abuts the left edge, the right edge, or sits fully inside one
  half.
- `compare_sequences(seq_1, seq_2)` — counts differing bases between two equal-length
  sequences. Returns `-1` if lengths differ (length mismatch is treated as an
  invalid splice, not as a count).
- `build_sequences(mapping_results, max_differences)` — splices the STARR-seq
  fragment into the reference window to produce the "mutated" sequence, counts how
  many bases differ from the reference via `compare_sequences`, and **drops** any
  sequence where `differences < 0` (length mismatch) or `differences > max_differences`
  (15 for WRKY, 16 for bHLH). Survivors are one-hot-encoded.
- `make_deepcre_predictions(seqs, meta)` — loads the deepCRE `.h5` model and runs a
  single batched `model.predict`, attaching `prediction_mutated` to each row.

**Joining in the experimental readout.**

- `merge_with_starrseq_results(...)` — left-joins the STARR-seq `enrichment` (and
  `condition`) onto the predictions by sequence id.
- `calculate_deltas(...)` — computes `delta_prediction` and `delta_enrichment`
  (mutated minus the matching reference variant) so changes, not just absolutes,
  can be correlated.
- `add_length_corrected_overlap_buckets(...)` — corrects short edge-spanning
  overlaps to 170 bp and assigns each row to its 200 bp bucket.

**Analysis & plotting.**

- `_fit_linear_model` (np.polyfit + `scipy.stats.spearmanr`) underlies every fit.
- `_plot_bucketed_correlation` / `_plot_individual_bucket_correlation` /
  the three `plot_*_correlation` wrappers draw the scatter plots
  (deepCRE vs STARR-seq; mutation-count vs each) coloured by bucket, with per-bucket
  and overall fit lines, and return per-bucket statistics rows.
- `plot_correlation_over_positions_fixed_window` /
  `…_fixed_number_elements` draw correlation/slope/p-value as a function of
  position along the gene. For fixed window, the buckets each must have at least
  50 points to be plotted.
- `save_bucket_statistics` writes those rows to `overlap_bucket_fit_parameters.csv`.

**The orchestrator.**

- `run_correlation_analysis(enrichment_df)` is the single entry the scripts call.
  It splits the dataframe into subsets — overall, `reference`, `synthetic`,
  `binding`, `non_binding`, and (when both conditions are present) `light`/`dark` —
  and for each subset emits the three scatter plots, the individual-bucket views,
  and the bucket-statistics CSV, plus the two position-series plot sets. The
  per-subset PNG/CSV naming is what produces the `deepcre_starrseq*/`,
  `mutation_deepcre*/`, `mutation_starrseq*/`, and `correlation_by_position_*/`
  subdirectories you see under each output root.

### 3.2 `starrseq_deepcre_correlation_WRKY.py` — WRKY entry point

TF-specific shell around `_common`. It assembles WRKY's analysis-ready dataframe
(`prepare_wrky_enrichment_df`) and chooses the output location (`main`).

**Inputs, and the columns taken from each:**

- `data/dCIS_WRKY_in_silico_mutated_GS2025d.fasta` → `load_starrseq_data`. Each
  header `WRKY_<chrom>:<start>-<end>_<status>_<id>` becomes a record carrying
  `full_name` (the whole header), `tf`, `chrom`/`start`/`end`, `binding_status`,
  the `reference` flag, and the raw `sequence`.
- `DEEP_CRE_RUN_FOLDER` → `load_relevant_deepcre_window_candidates` (WRKY-only,
  defined here): per gene folder it reads the 3020 bp `ref_seq` from
  `reference_sequence.fa` and `ref_fitness` = the fitness of the last entry in
  `saved_populations/pareto_front.json` — i.e. the deepCRE score of the *unmutated*
  reference window. Produces `gene`, `ref_seq`, `ref_fitness`, `start`, `end`.
- `data/WRKY_dCIS_dCRE_overlaps.csv` (Simon's file) — **only two columns are read**:
  `deepCIS_segment` and `Gene_ID`. `groupby("deepCIS_segment")["Gene_ID"].apply(list)`
  builds `site_to_gene_ids`, a per-site list of candidate genes. `deepCRE_region`,
  `Gene_Genomic_Location` and `Internal_deepCRE_Coord` are deliberately **ignored** —
  their coordinates are untrustworthy, so positions are re-derived by alignment.
- `data/plantstarr-seq_main_light_and_dark_simon_gernot.csv` (or the dark-only file)
  — columns `id`, `enrichment`, and `condition` are consumed downstream.

**How the pieces are joined (the shared `_common` pipeline):**

1. **Candidate lookup + alignment** (`map_starrseq_to_deepcre`). For each STARR-seq
   record it rebuilds the site key `<tf>_<chrom>:<start>-<end>` (which matches
   `deepCIS_segment`), looks up the candidate genes in `site_to_gene_ids`, and aligns
   the fragment inside each candidate's `ref_seq`. The mapping CSV thus acts purely
   as a *candidate-gene filter*; the alignment supplies the real `overlap_start/end`.
   Output rows carry `starr_full_name`, `gene`, `overlap_start/end`,
   `starr_binding_status`, `starr_reference`, `deepcre_ref_fitness`, and the two
   sequences.
2. **Build + score** (`build_sequences` → `make_deepcre_predictions`) adds
   `differences` (mutation count vs reference) and `prediction_mutated` (the model
   score of the mutated window) to each row.
3. **Enrichment merge** (`merge_with_starrseq_results`) — left-join on
   `predictions.starr_full_name == starrseq.id`, pulling in `enrichment` and
   `condition`. This is the join that connects prediction to experiment: the
   in-silico library was assayed and re-exported under those same header ids. It is
   **one-to-many** — each site was measured under both `Light` and `Dark`, so one
   prediction row fans out into a light row and a dark row (this duplication is
   exactly what later lets `run_correlation_analysis` stratify by `condition`).
   Rows with no enrichment match are dropped (`dropna(subset=["enrichment"])`).
4. **Deltas** (`calculate_deltas`) re-express both axes as *change from the site's
   unmutated baseline*:
   - `delta_prediction = prediction_mutated − deepcre_ref_fitness` — how far the
     model score of the mutated window moved from its reference window.
   - For the experimental side it groups by `starr_seq_base` (the
     `<tf>_<chrom>:<start>-<end>` prefix of `starr_full_name`, i.e. the site), takes
     the `enrichment` of that site's **reference** variant (`starr_reference == True`)
     as `reference_enrichment`, merges it back onto every variant of the same site,
     and computes `delta_enrichment = enrichment − reference_enrichment`.
   - Purpose: the correlation then asks "do the mutations the model thinks matter
     also move the *measured* enrichment?" instead of comparing absolute levels
     across unrelated sites (which differ for reasons unrelated to the TF motif).
5. **Buckets** (`add_length_corrected_overlap_buckets`) assign each row to its 200 bp
   position bucket in the 3020 bp frame.

`USE_NEW_DATA` toggles the STARR-seq CSV — light+dark → `correlation/new/`,
dark-only → `correlation/old/` — which is the source of the `old/`/`new/` split.
`main()` sets the output root and calls `run_correlation_analysis`.

### 3.3 `bHLH_overlaps/generate_bhlh_overlaps.py` — the bHLH mapping generator

This is the script that **produces `data/bHLH_dCIS_dCRE_overlaps.csv`** — the bHLH
counterpart of Simon's WRKY overlap file, written from scratch because the WRKY
file's coordinates are untrustworthy and its generator was lost.

Data flow:

1. `parse_fasta_sites(fasta)` — reads `dCIS_bHLH_in_silico_mutated_GS2025d.fasta`
   and deduplicates the 2,988 variant headers down to the **995 unique genomic
   binding sites** (`bHLH_<chrom>:<start>-<end>`).
2. `find_genes(gtf)` (from `evolution.extract_sequences`) — loads all genes from
   the TAIR10 GTF.
3. For each site, `overlaps_for_site` intersects it with **every gene on the same
   chromosome**, computing each gene's promoter and terminator windows via
   `find_start_end` and mapping overlap positions into the 3020 bp extracted frame
   via `genomic_to_relative_position` (strand-aware). One row per non-empty
   (site, gene-window) overlap.
4. `build_mapping` assembles the rows; `main()` (argparse CLI) writes the 7-column
   CSV (`site_id, gene_id, strand, region, start, end, additional_padding`) and
   prints a summary.

It reuses `evolution` functions directly rather than reimplementing the windowing
math. `bHLH_plan.md` is the full design spec (coordinate conventions, worked
example, edge cases) for this script.

### 3.4 `starrseq_deepcre_correlation_bHLH.py` — bHLH entry point

The bHLH sibling to `starrseq_deepcre_correlation_WRKY.py`. From step 3 onward
(`build_sequences` → predict → enrichment merge → deltas → buckets) it is **identical**
to WRKY — same columns, same two merges, same delta formulas described in §3.2. Only
the way the two *upstream* inputs are obtained differs, because no precomputed deepCRE
run folder exists for bHLH:

1. `ensure_refs_fasta(...)` — on first run only, it writes
   `data/bHLH_genes_of_interest.json` (the unique `gene_id`s from the bHLH overlap
   CSV) and invokes `evolution.extract_sequences` as a subprocess to produce
   `data/bHLH_reference_sequences.fa`. Subsequent runs skip this if the FASTA exists.
2. The mapping CSV `data/bHLH_dCIS_dCRE_overlaps.csv` is read once and used for two
   purposes: (a) the `site_id`→`gene_id` candidate-gene filter — **only `site_id` and
   `gene_id` are used** for `site_to_gene_ids`
   (`groupby("site_id")["gene_id"].apply(lambda s: list(s.unique()))`), exactly as
   in WRKY; and (b) the `additional_padding` column is extracted into
   `additional_padding_map` (a DataFrame indexed by
   `gene_id`, deduped) before `load_bhlh_window_candidates` is called.
   `.unique()` in (a) collapses the duplicate `(site_id, gene_id)` rows that short
   genes produce so each (site, gene) pair is aligned once. The remaining columns
   (`strand`, `region`, `start`, `end`) are still ignored — positions come from
   alignment.
3. `load_bhlh_window_candidates(refs_fasta_path, model_path, additional_padding_map)`
   — replaces WRKY's folder-walk. It parses each reference header
   `<chrom>_<gene_id>_gene:<start>-<end>` into `gene`, `ref_seq`, `start`, `end`,
   looks up `additional_padding` for that gene from `additional_padding_map` (a
   DataFrame indexed by `gene_id`), and attaches it to the gene entry so that
   `map_starrseq_to_deepcre` can pass it through to `adjust_positions`. `ref_fitness`
   is filled with a single batched `model.predict` over all reference windows. This
   is the bHLH equivalent of WRKY reading `pareto_front.json[-1][1]` — and it is
   what `delta_prediction = prediction_mutated − deepcre_ref_fitness` later subtracts
   against, so the bHLH delta is on the same "change vs reference window" basis as WRKY's.
4. `build_sequences(..., max_differences=16)` tolerates one more mutation than WRKY's
   15 (6 nt bHLH core motif vs WRKY's 5 nt).

Only the light+dark STARR-seq CSV is consumed — no `USE_NEW_DATA`, no
`correlation_bHLH/old/`. Outputs go to `correlation_bHLH/new/`.

### 3.5 `starrseq_deepcre_correlation_combined.py` — pooled entry point

This answers the validation question for WRKY and bHLH **together**. It re-implements
no pipeline: it imports `prepare_wrky_enrichment_df` and `prepare_bhlh_enrichment_df`,
calls each to get the two fully-prepared dataframes (predictions merged with
enrichment, deltas computed, buckets assigned), then `pd.concat([wrky_df, bhlh_df],
ignore_index=True)` and runs the same `run_correlation_analysis` over the union into
`correlation_combined/new/`.

Why a plain row concatenation is the right operation here (rather than a merge):

- **Identical columns.** Both dataframes exit the same `_common` steps with the same
  column set (`prediction_mutated`, `enrichment`, `condition`, `differences`,
  `delta_prediction`, `delta_enrichment`, the `overlap_bucket_*` columns, …), so
  stacking rows needs no alignment or renaming.
- **Deltas are computed per-TF *before* pooling.** `delta_prediction` and
  `delta_enrichment` are each derived inside the respective prepare-function, relative
  to that TF's own reference windows and reference variants. Concatenating afterwards
  therefore never mixes one TF's baseline into the other's. Because the
  `pareto_front` reference fitness equals a raw `model.predict` on the reference
  (§3.4 step 2), WRKY's and bHLH's `delta_prediction` share one scale and can sit on a
  single axis.
- **No double-counting.** Each TF is built from its own mapping CSV and reference set;
  a gene present in both TF libraries contributes once per TF (with that TF's own
  fragments), so the two halves are disjoint observations, not duplicates to merge.
- **Shared bucket frame.** Both assign buckets in the same 3020 bp coordinate frame,
  so e.g. the `800-999` bucket means the same gene position for both TFs and the
  pooled per-bucket fits stay coherent.

The pooled outputs use the identical subset/bucket layout as the per-TF roots (§5);
the only difference is that each plotted point may originate from either TF.

### 3.6 Design documents

- `bHLH_plan.md` — design spec for `generate_bhlh_overlaps.py` (the mapping file).
- `bHLH_analysis_plan.md` — design spec for the `_common.py` refactor and the bHLH
  correlation script (how the WRKY script was split into `_common` + per-TF shells).

---

## 4. Data files in detail

### External inputs (sources of truth)

- **`dCIS_WRKY_in_silico_mutated_GS2025d.fasta`** (2,379 records) and
  **`dCIS_bHLH_in_silico_mutated_GS2025d.fasta`** (2,988 records) — **my in-silico
  mutagenesis library**: synthetic variants of deepCIS-predicted TF binding sites
  that were ordered and assayed in STARR-seq. Header
  `<tf>_<chrom>:<start>-<end>_<status>_<id>` encodes the site and the variant type
  (`binding` / `non_binding` / `reference`). These are the sequences whose
  predictions are correlated against experiment.

- **`plantstarr-seq_main_light_and_dark_simon_gernot.csv`** (11,902 rows) and
  **`plantstarr-seq_main_dark_simon_gernot(1).csv`** (5,950 rows) — **experimental
  STARR-seq results**. Columns: `condition, id, GC, length, [sequence,]
  n_experiments, min_bc, min_ci, min_co, enrichment`. `enrichment` is the measured
  readout joined onto the predictions. The light+dark file adds a light condition
  and the raw `sequence` column; the `dark(1)` file is the earlier dark-only export
  (used when `USE_NEW_DATA=False`).

- **`WRKY_dCIS_dCRE_overlaps.csv`** (434 rows) — **Simon's** file of overlaps
  between the WRKY in-silico library and gene extraction windows. Columns:
  `deepCIS_segment, Gene_ID, deepCRE_region, Gene_Genomic_Location,
  Internal_deepCRE_Coord`. **Its coordinates are not trustworthy** — the WRKY
  script uses only the `deepCIS_segment`→`Gene_ID` association as a candidate-gene
  filter and re-derives positions by sequence alignment. `generate_bhlh_overlaps.py`
  is the trustworthy re-implementation of whatever produced this file.

- **`wrky_overlaps_reference_mapping.csv`** (389 rows; `gene_name, target_start,
  target_end`) and **`short_genes.json`** — **legacy / unused**. Neither is read by
  any code in this folder; they are leftovers from an earlier iteration and are not
  part of the current data flow.

### Generated data files

- **`bHLH_dCIS_dCRE_overlaps.csv`** (939 rows) — output of
  `generate_bhlh_overlaps.py` (§3.5). Input to the bHLH correlation script.
- **`bHLH_genes_of_interest.json`** — written by `ensure_refs_fasta`: the unique
  `gene_id`s from `bHLH_dCIS_dCRE_overlaps.csv`, used as the gene list for
  reference extraction.
- **`bHLH_reference_sequences.fa`** (863 records) + **`.fai`** — written by
  `evolution.extract_sequences` (invoked by `ensure_refs_fasta`): the per-gene
  3020 bp reference windows, headers `<chrom>_<gene_id>_gene:<start>-<end>`. This
  is the bHLH replacement for WRKY's `DEEP_CRE_RUN_FOLDER` reference sequences.
- **`*.fasta.fai`** — `pyfaidx` index files, created automatically the first time
  each FASTA is opened.

---

## 5. Output files (`correlation/`, `correlation_bHLH/`, `correlation_combined/`)

All three output roots have the identical layout, produced by
`run_correlation_analysis`. `correlation/` additionally has `old/` and `new/`
(dark-only vs light+dark STARR-seq); the others have only `new/`.

| Subdirectory pattern | Content |
| --- | --- |
| `deepcre_starrseq[_<subset>]/deepcre_starrseq_correlation*.png` | deepCRE prediction vs STARR-seq enrichment, bucketed scatter + fits |
| `mutation_deepcre[_<subset>]/…png` | mutation count vs deepCRE prediction |
| `mutation_starrseq[_<subset>]/…png` | mutation count vs STARR-seq enrichment |
| `*_bucket_800_999*.png` | the single bucket called out in `INDIVIDUAL_BUCKET_LABELS_TO_PLOT`, plotted on its own |
| `*/overlap_bucket_fit_parameters.csv` | per-bucket slope, intercept, Spearman r/p |
| `correlation_by_position_fixed_window/` | Spearman r, slope, p-value vs position (fixed-width windows) |
| `correlation_by_position_fixed_number_elements/` | same, fixed-count rolling windows |

`<subset>` ∈ { (none = overall), `reference`, `synthetic`, `binding`,
`non_binding`, `light`, `dark` }. Figures are 600 dpi (poster quality).

---

## 6. End-to-end data flow

```text
EXTERNAL INPUTS                          CODE                              OUTPUTS
─────────────────────────────────────────────────────────────────────────────────

dCIS_WRKY_…fasta  ─┐
WRKY_dCIS_dCRE_…   ─┤   (candidate-gene filter; coords ignored)
   overlaps.csv     │
DEEP_CRE_RUN_FOLDER ┼─► starrseq_deepcre_correlation_WRKY.py
   (ref seqs +      │      load → map(align) → build → predict →
    pareto fitness) │      merge → deltas → buckets ──► run_correlation_analysis
plantstarr-seq_…csv─┘                                         │
deepCRE .h5 model ──(used by every predict step)              └─► correlation/{old,new}/

dCIS_bHLH_…fasta ──► generate_bhlh_overlaps.py ──► data/bHLH_dCIS_dCRE_overlaps.csv
  +TAIR10 GTF                                              │
                                                           ▼
bHLH_dCIS_dCRE_overlaps.csv ─┐
genome FASTA + GTF ──────────┤ ensure_refs_fasta → evolution.extract_sequences
                             │      → bHLH_reference_sequences.fa
dCIS_bHLH_…fasta ───────────┼─► starrseq_deepcre_correlation_bHLH.py
plantstarr-seq_…csv ─────────┘      (same _common pipeline) ──► correlation_bHLH/new/

(WRKY df) + (bHLH df) ──► starrseq_deepcre_correlation_combined.py ─► correlation_combined/new/

All TF-agnostic steps (parse, align, build, predict, merge, deltas, bucket, plot)
live in _common.py and are shared by all three entry points.

Legacy / unused: wrky_overlaps_reference_mapping.csv, short_genes.json
```
