# Plan: Generate bHLH dCIS→deepCRE overlap mapping file

## Context

The STARR-seq × deepCRE correlation in `src/workflows/overlap_analysis/` exists only
for WRKY. To repeat it for bHLH, one input is missing: a file mapping each tested bHLH
binding site to the gene(s) whose deepCRE extraction window it overlaps, plus the
position of that overlap inside the window. The WRKY equivalent
(`data/WRKY_dCIS_dCRE_overlaps.csv`) was a one-off artifact with no surviving generator
and some coordinate kinks we explicitly want to avoid here.

We generate the bHLH file from:

- `src/workflows/overlap_analysis/data/dCIS_bHLH_in_silico_mutated_GS2025d.fasta`
  — headers `bHLH_<chrom>:<start>-<end>_<variant>_<id>`; 2988 records →
  **995 unique genomic sites** (all chrom `1`).
- `/home/gernot/ARCitect/ARCs/genRE/assays/Gene_Data/dataset/annotations/Arabidopsis_thaliana.TAIR10.52.gtf`
  (TAIR10, 1-based, seqnames `1..5,Mt,Pt`, 32 833 `gene` features).

deepCRE extracts, per gene, a promoter/TSS window and a terminator/TTS window
(`intragenic=500` inside + `extragenic=1000` outside the gene), concatenated as
`promoter(1500) + central N-padding(20) + terminator(1500)` = **3020 bp total**
(short genes get extra central padding so the total stays 3020).

## Downstream usage

The bHLH correlation script (to be built later, analogous to
`starrseq_deepcre_correlation_WRKY.py` but bHLH-specific) consumes this mapping as a
**candidate-gene filter**: per site, look up the list of `gene_id`s, then resolve precise
overlap positions by aligning the STARR-seq sequence's first/last 50bp against each
candidate's 3020bp extracted window via `str.find()` ± reverse complement (see
`find_overlap_positions` in the WRKY script). So `(site_id, gene_id)` is the strictly
required output; `strand, region, start, end, additional_padding` are useful provenance
for sanity-checking and other applications but are not consulted by the alignment step.
A 1bp offset in `start`/`end` driven by the FASTA header's coordinate convention is
therefore tolerable — the alignment is the source of truth.

## Decisions (confirmed)

- **Output: 7 columns** — `site_id, gene_id, strand, region, start, end, additional_padding`.
  - `site_id` = deduplicated site id `bHLH_<chrom>:<start>-<end>` (one per unique site;
    parallels WRKY's `deepCIS_segment` column and matches the search key built by
    `get_gene_candidates_for_starrseq_entry` in the WRKY script).
  - `gene_id` = TAIR gene id, with the `_gene` suffix stripped from `find_genes` output.
  - `strand` = `+` or `-`, the gene's strand from `find_genes`. Identical for all rows
    sharing a `gene_id`; included as gene-level provenance so a reader of any single
    row can tell which orientation the extracted sequence was built in.
  - `region` = `promoter` or `terminator` (the strings returned by
    `genomic_to_relative_position`); disambiguates short-gene rows where the same site
    can hit both windows.
  - `start`, `end` = 0-based half-open positions in the 3020bp extracted window.
  - `additional_padding` = the gene's extra central padding in bp (0 for normal genes,
    >0 for short genes); replaces the previously-planned short-gene warning.
- **Coordinates**: 0-based, half-open `[start, end)`, **padding included**, full range **0–3020**.
- **Reuse, do not reimplement**: import the windowing/mapping math directly from
  `evolution.extract_sequences`.
- **Scope**: all 995 unique sites; emit a row for any (site × gene-window) overlap. A
  site may map to several genes → several rows.
- **Build**: bHLH-specific script in `src/workflows/overlap_analysis/bHLH_overlaps/`.

## Reused evolution functions (imported directly)

`from evolution.extract_sequences import find_genes, find_start_end, genomic_to_relative_position`

- `find_genes(annotation_path, gene_name_attribute="gene_id", feature_type_filter=["gene"], genes_of_interest=[])`
  → DataFrame `chromosome, start, end, strand, gene_id`. Coords are **0-based** (BCBio);
  `gene_id` has a `_gene` suffix to strip. Empty `genes_of_interest` returns all genes.
- `find_start_end(start, end, 500, 1000, strand)` → `(prom_start, prom_end, term_start, term_end, additional_padding)`
  (genomic, 0-based; handles short-gene split internally).
- `genomic_to_relative_position(genomic_pos, prom_start, prom_end, term_start, term_end, strand, additional_padding)`
  → `(rel_pos, region)` or `None`. `rel_pos` is the 0-based offset in the 3020bp extracted
  sequence **including** `CENTRAL_PADDING=20` and `additional_padding`: promoter `[0,1500)`,
  terminator `[1520, 3020)` (normal genes). Handles minus-strand reversal.

## Overlap logic

Per unique site, looped against every gene on the same chromosome.

### Variables used below

All "genomic" positions are 0-based positions on the chromosome. All "extracted-sequence"
positions are 0-based positions inside the 3020bp window deepCRE extracts per gene.

| Name | Meaning |
|---|---|
| `site_genomic_start`, `site_genomic_end` | The genomic interval from the FASTA header. Treated **inclusive on both ends** — `site_genomic_end` is the last *included* base. |
| `gene_genomic_start`, `gene_genomic_end` | The gene body, from the GTF (via `find_genes`). |
| `strand` | `"+"` or `"-"`, the gene's strand. |
| `promoter_window_genomic_start`, `promoter_window_genomic_end` | Genomic interval of the gene's promoter window (1500bp). **Half-open** — `..._end` is *one past* the last included base. |
| `terminator_window_genomic_start`, `terminator_window_genomic_end` | Same for the terminator window. Half-open. |
| `additional_padding` | Extra N-padding inserted between the two windows for *short* genes; 0 for normal genes. |
| `window_genomic_start`, `window_genomic_end` | Stand-in for either the promoter or the terminator window inside step 2's per-window loop. |
| `overlap_first_base_genomic`, `overlap_last_base_genomic` | The leftmost and rightmost genomic bases of the site×window overlap. Inclusive on both ends. |
| `extracted_pos_for_overlap_first_base`, `extracted_pos_for_overlap_last_base` | The same two boundary bases after `genomic_to_relative_position` has mapped them into the 3020bp extracted sequence. |
| `extracted_seq_start`, `extracted_seq_end` | What we emit. Half-open positions inside the 3020bp extracted sequence; these become the CSV's `start` and `end` columns. |

### Step 1 — get the gene's two windows in genomic coords

```
(promoter_window_genomic_start,
 promoter_window_genomic_end,
 terminator_window_genomic_start,
 terminator_window_genomic_end,
 additional_padding) = find_start_end(
    gene_genomic_start, gene_genomic_end, 500, 1000, strand,
)
```

Both window intervals come back half-open in the genome. `find_start_end` already
handles:

- **+ strand**: the promoter window sits upstream of the gene body (lower genomic
  coords); the terminator window sits downstream (higher genomic coords).
- **- strand**: the two windows are swapped — the promoter window is at higher genomic
  coords and the terminator window is at lower genomic coords.
- **Short gene** (gene body shorter than `2 * intragenic = 1000`bp): the inside-the-gene
  part of each window is carved asymmetrically and `additional_padding > 0` comes back.
  The two windows can sit very close together, so a long-enough site can hit *both*.

### Step 2 — intersect the site with each window, separately

We intersect with each window separately because **the two windows are not contiguous
in the extracted sequence** — the central N-padding sits between them. A site that
spans the gene body would therefore produce *two disjoint chunks* in
extracted-sequence coordinates, and we want one row per chunk.

Letting `window_genomic_start, window_genomic_end` stand in for either the promoter
window or the terminator window:

```
overlap_first_base_genomic = max(site_genomic_start, window_genomic_start)
overlap_last_base_genomic  = min(site_genomic_end,   window_genomic_end - 1)
if overlap_first_base_genomic > overlap_last_base_genomic:
    no overlap with this window — skip
```

The bracket conventions are the bit worth pausing on:

- The **site** is inclusive–inclusive: `site_genomic_end` is the rightmost *included*
  base.
- The **window** is half-open: `window_genomic_end` is *one past* the rightmost
  included base.

For `min`/`max` to behave correctly, both ends need to be in the same convention. The
formula above puts everything in inclusive–inclusive: `site_genomic_end` is already
inclusive, and `window_genomic_end - 1` is the window's inclusive last base.

After this, `[overlap_first_base_genomic, overlap_last_base_genomic]` is the overlap
as inclusive–inclusive 0-based genomic coordinates. Its length in bases is
`overlap_last_base_genomic - overlap_first_base_genomic + 1`.

### Step 3 — map the two boundary bases into the extracted sequence

```
extracted_pos_for_overlap_first_base, region = genomic_to_relative_position(
    overlap_first_base_genomic,
    promoter_window_genomic_start, promoter_window_genomic_end,
    terminator_window_genomic_start, terminator_window_genomic_end,
    strand, additional_padding,
)
extracted_pos_for_overlap_last_base, _ = genomic_to_relative_position(
    overlap_last_base_genomic, ...same other args...
)
```

`genomic_to_relative_position` knows the layout of the 3020bp extracted sequence:

```
   0                       1500       1520 + additional_padding              3020
   |--- promoter (1500) ----|---- N ----|------ terminator (1500) -----------|
```

- For **+ strand**, larger genomic position → larger extracted-sequence position
  (monotonic increasing).
- For **- strand**, the extracted sequence is reverse-complemented, so larger genomic
  position → *smaller* extracted-sequence position (monotonic decreasing). The
  function handles this internally.

Because both boundary bases came from inside the same window in step 2, they map to
the same `region` — both `"promoter"` or both `"terminator"`. That string is the
row's `region`.

Then:

```
extracted_seq_start = min(extracted_pos_for_overlap_first_base,
                          extracted_pos_for_overlap_last_base)
extracted_seq_end   = max(extracted_pos_for_overlap_first_base,
                          extracted_pos_for_overlap_last_base) + 1
```

Two things going on:

- **`min`/`max` makes this strand-agnostic.** On + strand the lower genomic base maps
  to the lower extracted position; on - strand it maps to the higher one. Taking
  `min` picks the smaller of the two either way, so the half-open interval is
  correctly ordered regardless of strand.
- **`+ 1` converts inclusive back to half-open.**
  `extracted_seq_end - extracted_seq_start` equals the overlap length in bases.

### Step 4 — emit

```
(site_id, gene_id, strand, region,
 extracted_seq_start, extracted_seq_end, additional_padding)
```

`extracted_seq_start` and `extracted_seq_end` end up in the CSV under the column names
`start` and `end`.

### Edge cases handled by the same code, without special-casing

- **Normal gene**: the two windows are separated by the gene body, so a typical 250bp
  site hits at most one window → one row per `(site, gene)`.
- **Short gene**: a site can hit both windows → two rows; they share `gene_id`,
  `strand`, and `additional_padding` but differ in `region`, `start`, `end`.
- **No overlap with either window**: no row at all.
- **Clipped edge** (site pokes past the boundary of a window): step 2's
  `min(..., window_genomic_end - 1)` clamps `overlap_last_base_genomic` to the window's
  last base, so the row's length is just the in-window portion.
- **Minus-strand site**: the reverse mapping inside `genomic_to_relative_position` plus
  the `min`/`max` in step 3 jointly produce a correctly-ordered half-open
  extracted-sequence interval — no special handling in our code.

### Worked example

WRKY site `1:8049090-8049339`, gene `AT1G22740` on `+` strand, 0-based gene start
`8049089`:

- **Step 1**: `promoter_window_genomic_start = 8049089 - 1000 = 8048089`;
  `promoter_window_genomic_end = 8049089 + 500 = 8049589`. Promoter window is
  `[8048089, 8049589)`. (Terminator window is irrelevant for this example.)
- **Step 2** against the promoter window:
  - `overlap_first_base_genomic = max(8049090, 8048089) = 8049090`
  - `overlap_last_base_genomic  = min(8049339, 8049588) = 8049339`
  - Both site ends fall inside the window. Inclusive length:
    `8049339 - 8049090 + 1 = 250` bases.
- **Step 3**:
  - `extracted_pos_for_overlap_first_base = 8049090 - 8048089 = 1001`,
    `region = "promoter"`.
  - `extracted_pos_for_overlap_last_base  = 8049339 - 8048089 = 1250`.
  - `extracted_seq_start = min(1001, 1250) = 1001`,
    `extracted_seq_end   = max(1001, 1250) + 1 = 1251`.
  - Length `1251 - 1001 = 250` ✓.
- **Step 4**: emit
  `("WRKY_1:8049090-8049339", "AT1G22740", "+", "promoter", 1001, 1251, 0)`.

## Files

**New**: `src/workflows/overlap_analysis/bHLH_overlaps/generate_bhlh_overlaps.py`

- Module-level reuse of the three evolution functions above (no GTF/window math reimplemented).
- `parse_fasta_sites(fasta_path) -> list[Site]` — split headers, dedup to unique
  `tf_chrom:start-end` (reuses the `name.split('_')`, `parts[0]+'_'+parts[1]` approach from
  `starrseq_deepcre_correlation_WRKY.py:load_starrseq_data`).
- `overlaps_for_site(site, genes_df) -> list[row]` — implements the logic above, calling
  `find_start_end` + `genomic_to_relative_position`.
- `build_mapping(fasta_path, gtf_path, intragenic=500, extragenic=1000) -> DataFrame`.
- `main()` with argparse: `--fasta`, `--gtf`, `--output`, `--intragenic`, `--extragenic`;
  prints a summary (unique sites, sites with ≥1 overlap, rows, short-gene rows). Default
  `--gtf` = the TAIR10 path above; default `--output` =
  `src/workflows/overlap_analysis/data/bHLH_dCIS_dCRE_overlaps.csv`.
- Filter `genes_df` by chromosome before looping (optional `bisect` on sorted window starts;
  brute-force per-chrom is fine: 995 sites × ~7k chr1 genes).

**Output**: `src/workflows/overlap_analysis/data/bHLH_dCIS_dCRE_overlaps.csv`
(7 columns above; name parallels `WRKY_dCIS_dCRE_overlaps.csv`, which is the WRKY file
the correlation script actually consumes).

**New tests**: `test/workflows/overlap_analysis/bHLH_overlaps/test_generate_bhlh_overlaps.py`
(unittest, Arrange-Act-Assert; mock filesystem). Cover:

- `parse_fasta_sites` dedup (mock_open on a few headers incl. duplicate positions/variants).
- `overlaps_for_site` against real `find_start_end`/`genomic_to_relative_position` with
  hand-computed expectations: + strand site fully in promoter; minus-strand site; clipped
  window-edge (partial, shorter length); short gene (assert `additional_padding>0` and that
  start/end land past the padded terminator offset).
- one site → two genes (two rows).
- `build_mapping` end-to-end with `find_genes` **patched** to return a small DataFrame
  (avoids parsing the real GTF); assert the 7-column CSV via a `tempfile` dir, including
  that `region` matches the window that was hit and that `strand` matches the gene's
  strand from the patched DataFrame.

## Verification

1. Activate conda env `deepCREshap`.
2. `python -m pytest test/workflows/overlap_analysis/bHLH_overlaps/test_generate_bhlh_overlaps.py` — all pass.
3. Run the script → writes `data/bHLH_dCIS_dCRE_overlaps.csv`; review the printed
   summary (rows, how many sites overlapped a gene, how many short-gene rows).
4. Spot-check a couple rows against the GTF (`awk '$3=="gene"'`) to confirm start/end land in
   the expected window and that `end ≤ 3020`.
5. Hand off to user for the end-to-end smoke check (they run those themselves).
