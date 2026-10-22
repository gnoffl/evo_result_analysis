# Region mutation breakdown — design

## Goal

Extend `evo_alg_pooled_plots` to break down the **possible mutation solution
space by genomic region** (promoter, 5'-UTR, 3'-UTR, terminator). For each gene
we count how many mutable positions / mutations fall into each region, then
compare the constrained (natural VCF) and unconstrained (any non-N position)
conditions, separately for the GOF and LOF gene sets.

## Region definition (1-based coordinates)

Sequences are 3020 bp. VCF positions are **1-based**, so all region borders are
stated 1-based. The reference-sequence iteration is 0-based and is converted to
1-based (`index + 1`) before assignment, so both sources share one coordinate
frame and one assignment function.

| Region     | Start | End  | Length |
|------------|-------|------|--------|
| promoter   | 1     | 1000 | 1000   |
| 5'-UTR     | 1001  | 1500 | 500    |
| (gap)      | 1501  | 1520 | 20     |
| 3'-UTR     | 1521  | 2020 | 500    |
| terminator | 2021  | 3020 | 1000   |

Positions 1501-1520 are always `N` in the reference and never carry a VCF
record, so they self-exclude. `assign_region` returns `None` for them; such
positions are dropped and never counted.

Borders are hardcoded as a module constant. The `evolution` package has no
canonical region-boundary definition to import (only run defaults), confirmed
before hardcoding.

## Metric

Raw per-gene **possible-mutation** counts per region (no length normalization).
Both conditions count the same thing — the number of mutations available to the
optimizer in that region. Region lengths differ 2x (promoter/terminator 1000 bp
vs UTRs 500 bp); this size confound is accepted and noted in the summary.

- **Constrained (VCF):** count VCF records whose position maps to each region.
  Each record (ALT allele) is one possible natural mutation. Matches
  `vcf_mutation_stats.count_mutations_in_vcf`, but per-region.
- **Unconstrained (reference):** count non-N positions of
  `reference_sequence_full` mapping to each region, **multiplied by 3** (each
  position admits three alternative SNPs). Extends
  `count_unconstrained_positions`; the x3 makes it comparable to the VCF record
  count.

## Components (`region_mutation_breakdown/region_mutation_breakdown.py`)

- `REGIONS`: list of `(name, start, end)` tuples, 1-based inclusive.
- `assign_region(position: int) -> str | None` — return region name or `None`.
- `collect_vcf_region_counts(vcf_dir: Path) -> pd.DataFrame` — per-gene,
  per-region VCF record counts. Columns: `gene, region, count`.
- `collect_unconstrained_region_counts(run_dir: Path) -> pd.DataFrame` —
  per-gene, per-region non-N position counts. Columns: `gene, region, count`.
- `build_region_dataframe(...)` — tidy long DataFrame:
  `gene, group, condition, region, count`.
- `plot_region_breakdown(data, group, output_dir, fmt)` — grouped bar plot for
  one gene group. x=region (fixed order), hue=condition, mean±sd error bars,
  reusing the visual style of
  `natural_unconstrained_mutation_vis._barplot_figure`. One figure per group.

## Data sources (reuse existing module constants)

Same paths as `natural_unconstrained_mutation_vis.py`:
- `GOF_RUN_DIR`, `LOF_RUN_DIR` — unconstrained reference sequences.
- `GOF_VCF_DIR`, `LOF_VCF_DIR` — constrained natural VCFs.

## Outputs (`region_mutation_breakdown/` subfolder, gitignored)

- `region_breakdown_GOF.png` — grouped bar plot, GOF genes.
- `region_breakdown_LOF.png` — grouped bar plot, LOF genes.
- `region_breakdown_per_gene.csv` — tidy per-gene counts.
- `region_breakdown_summary.csv` — mean/std count per group×condition×region.

## Testing (`test_region_mutation_breakdown.py`, mandatory, same session)

`unittest`, Arrange-Act-Assert. Cover:
- `assign_region` boundaries: 1, 1000, 1001, 1500 (5'-UTR edge), 1501-1520→None,
  1521, 2020, 2021, 3020; out-of-range (0, 3021) → None.
- `collect_vcf_region_counts` on a tiny mock VCF (temp file) with known
  positions across regions and in the gap.
- `collect_unconstrained_region_counts` on a mock fasta with known N/non-N
  layout (mock `pyfaidx.Fasta`).

## Non-goals (YAGNI)

- No length normalization / density.
- No significance testing between regions (can be added later if needed).
- No CLI / config; run as a script with the module-constant paths, matching
  sibling scripts.
