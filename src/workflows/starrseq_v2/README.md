# STARR-seq v2 library build

Implements `Evolution/docs/starrseq_v2_design.md`.

**The round itself is defined by the factor grid in
`evolution.write_run_script`** — `FACTORS`, `BASELINE`, `arm_settings()`,
`grid_settings()`. Read those to see what was run. This folder supplies the
sequences and the gene metadata that grid reads.

```bash
# 1. gene selection, frame extraction, construct splicing, metadata
conda run -n deepCREshap python -m src.workflows.starrseq_v2.build_starrseq_v2_library

# 2. configs (in Evolution/)
conda run -n deepCREshap python -c "
import pandas as pd
from evolution.write_run_script import (
    GENE_METADATA_PATH, baseline_settings, arm_settings, grid_settings,
    write_starrseq_v2_configs,
)
metadata = pd.read_csv(GENE_METADATA_PATH)
write_starrseq_v2_configs('baseline', baseline_settings(), metadata)
write_starrseq_v2_configs('arms', arm_settings(), metadata, 'arm_subset')
write_starrseq_v2_configs('grid', grid_settings(), metadata, 'full_grid_subset')
"

# 3. run
conda run -n deepCREshap python -m evolution.genetic_algorithm \
    -i configs/starrseq_v2/<group>/<name>.json -c <cores> -r <run_id>
```

| file | what |
| --- | --- |
| `build_starrseq_v2_library.py` | the whole build: gene selection, extraction, subsets, directions, windows, construct splicing, metadata |
| `flowering_data/` | FLOR-ID download and the N. tabacum ortholog search that produce the two flowering gene lists |

Paths and parameters are hardcoded at the top of the module. Nothing takes
command line arguments.

## Geometry

Every frame is 3020 bp (`-i 500 -e 1000`), TSS at 1000, 20 bp of `N` padding at
1500–1520. The construct extracted through `construct_annotation.gtf` puts the
170 bp insert placeholder at **[780, 950)** and the barcode at [1076, 1094).

Four reference frames are possible, all mutating [780, 950) except the natural
off-site case:

| background | location | frame |
| --- | --- | --- |
| construct | correct | construct with the gene's [780, 950) window spliced in |
| construct | offsite | construct with another window of the same gene spliced in |
| natural | correct | the extracted gene frame, unchanged |
| natural | offsite | the extracted gene frame, mutated at the alternative window |

The construct cases need their own FASTA because the off-site window is spliced
into the fixed insert site. The natural cases share one file per species and
differ only in `mutation_start`/`mutation_end`, which `resolve()` takes from the
metadata. **The assay always measures a 170mer at the construct's fixed insert
site** — background and location say where the *optimization* happened, not
where the ordered sequence is measured.

## What the build produces

`build_starrseq_v2_library.main()` writes, into
`Evolution/data/starrseq_v2/`:

| output | content |
| --- | --- |
| `candidate_gene_metadata.csv` | one row per (gene, alternative window); the table the config grid reads |
| `starrseq_v1_mapping.csv` | the WRKY + bHLH fragment-to-gene alignments the v1 windows come from |
| `{ara,ntab}_flowering_GOF*_extracted_genes.fa` | 3020 bp gene frames |
| `starrseq_v1_extracted_genes.fa` | 3020 bp frames of the v1 genes |
| `{ara,ntab}_flowering_GOF_constructs.fa` | correct-window constructs |
| `{ara,ntab}_flowering_GOF_off_target_constructs.fa` | off-site constructs |
| `starrseq_v1_off_target_constructs.fa` | v1 fragment windows in the construct |

### Metadata columns

| column | meaning |
| --- | --- |
| `gene_id`, `species` | `arabidopsis` / `ntab` |
| `reason` | `flowering`, `GOF`, `LOF`, `starrseq_v1`, `random` |
| `arm_subset` | in the one-factor-at-a-time arms |
| `full_grid_subset` | in the full-grid probe (a subset of `arm_subset`) |
| `alternative_start`, `alternative_end` | the off-site window, `Int64`, missing outside the arm subset |
| `full_length_window` | the window spans exactly 170 bp |
| `direction` | `maximize` / `minimize` |

### Gene selection

Per species, up to `TARGET_GENE_COUNT_* = 500`:

* **arabidopsis** — STARR-seq v1 genes (from the WRKY and bHLH mapping modules),
  FLOR-ID flowering genes, the GOF and LOF panels read out of their extracted
  FASTAs, then random genes from the annotation to fill up. Extracted with
  `ARA_VCF`, so a natural-mutation arm is possible.
* **ntab** — the FLOR-ID orthologs from `flowering_data/`, then random genes.
  No VCF exists, so no natural-mutation arm.

Genes chosen for more than one reason keep one row per reason but count once
towards the target (`dict.fromkeys` dedup). The random draw excludes everything
already in the table.

### Subsets

`arm_subset` is 20 % of each species' genes (at least 50), drawn over unique
genes with the v1 genes excluded. `full_grid_subset` is 20 % of *that* (at
least 20). Every row of a drawn gene is flagged, so a gene never lands half in
and half out.

### Directions

GOF → `maximize`, LOF → `minimize`. Everything else is drawn balanced within
each (subset stratum × species) cell, the strata being full-grid, arm-but-not-
grid, and neither. A gene keeps one direction across all its rows.

### Off-site windows

A v1 gene takes its window from the measured fragment alignment. Every other
arm-subset gene draws one random 170 bp window, uniformly over
`RANDOM_WINDOW_STARTS` = `[0, 1330) ∪ [1520, 2850)`, which keeps the window
clear of the central `N` padding. Rows outside the arm subset keep no window.

## Deviations from the design document

### Agreed 2026-09-17, still standing

* **Arms enumerate 6, not 7** — the design table's count was the typo.
* **The grid is 16 cells, not 32** — the mutation-constraint dimension needs a
  variant panel and none exists for N. tabacum, so it is out of `GRID_FACTORS`.
* **3 variants per arm per unit, not 4** — matches the budget arithmetic.

### From the rewrite

* **N. tabacum genes come from the ARC genome**
  (`genomes/nicotiana_tabacum.fa` + the agat GTF), not the pseudo-chromosome
  build. Coordinates from the older pseudo build do not transfer.
* **ntab genes are not expression-stratified.** The design selected from
  `ntab_targets.csv` with low TPM → maximize, high → minimize. The build draws
  randomly and balances directions instead. `ntab_targets.csv` is unused.
* **Old fragments are reduced to the STARR-seq v1 genes** and enter only as
  `("construct", "v1_genes", "offsite")` — their measured fragment window in
  the construct. There is no native-context arm for them, and they are excluded
  from the arm-subset draw.
* **Blocks and arms are gone.** A run is one point in the `FACTORS` grid, and
  the config file name carries the factor levels plus the direction. Nothing is
  read back out of a results folder name.

## Open items

* **The output checks are not written.** `main()` ends on
  `# TODO: final length check, N counts, duplicates, nuclease sites, etc`, and
  `write_run_script.py` carries the same TODO. The design document asks for
  length 3020, residual `N` == 20, and spliced bytes identical to source.
* **No `N` check on the insert.** Random windows avoid the central padding by
  construction, but a v1 window comes from the alignment and is not constrained.
  An all-`N` mutation window hangs
  `evolution.mutation.check_indices_to_mutate` (`mutation.py:220-225`). The
  deleted `old_fragments.native_windows` had this guard.
* **Short v1 windows are dropped, not corrected.** A fragment overlap that is
  not exactly 170 bp fails `full_length_window` and is skipped. The deleted code
  length-corrected it inside the frame instead.
* **`max_number_mutations` is 20**, carried over from the single-model round.
  `starrseq_v2_probe()` writes a 10-gene-per-direction pilot to settle whether
  ensemble fronts saturate below the cap.
* **Harvest and post-hoc scoring do not exist.** Sections 5 and 6 of the design
  document — order table, restriction screening, scoring matrix — were removed
  along with the old architecture and are to be rebuilt. The cloning prefix,
  suffix, restriction site list and controls were never filled in.

## Notes

* Natural-context frames keep whatever `N` the genome has — some Arabidopsis
  frames carry hundreds. Only the mutation window matters for
  `check_indices_to_mutate`.
* One config file per (setting, direction). The split by direction is forced:
  `write_run_script` special-cases `weights`, and a setting's two directions
  need different ones.
* `models` *can* be fanned out per config — `write_run_script` broadcasts a flat
  list to every config and takes a list-of-lists per config. `models_for()`
  uses that for the single-model arm, cycling through the ensemble in sequence
  order, so no gene is pinned to one fixed model.
* Natural-mutation runs get 300 generations, everything else 1000; the search
  space is far smaller when mutations are VCF-restricted.

## Tests

```bash
conda run -n deepCREshap python -m pytest test/workflows/starrseq_v2/
cd ../../../../Evolution && conda run -n deepCREshap pytest test/test_write_run_script.py
```
