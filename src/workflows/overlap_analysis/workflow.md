# Overlap Analysis Workflow

This folder contains two related analysis workflows that validate evolutionary algorithm outputs by
comparing deepCRE binding predictions with experimental STAR-seq enrichment measurements.

---

## Directory Structure

```
overlap_analysis/
├── starrseq_deepcre_correlation_WRKY.py   # Main WRKY correlation pipeline
├── bHLH_overlaps/
│   └── generate_bhlh_overlaps.py          # bHLH site-to-gene mapping generator
├── bHLH_plan.md                            # Design spec for generate_bhlh_overlaps.py
├── data/                                   # Input data files
└── correlation/                            # Output plots and statistics
    ├── old/                               # Legacy pipeline outputs
    └── new/                               # Updated pipeline outputs
```

---

## Workflow 1: WRKY STAR-seq × deepCRE Correlation

**Script**: `starrseq_deepcre_correlation_WRKY.py`

**Goal**: Validate that deepCRE binding predictions for WRKY TF binding sites correlate with
experimentally measured STAR-seq enrichment scores.

### Inputs

| File | Description |
|------|-------------|
| `data/dCIS_WRKY_in_silico_mutated_GS2025d.fasta` | STAR-seq sequences with TF identity and position metadata in headers |
| `data/plantstarr-seq_main_light_and_dark_simon_gernot.csv` | STAR-seq enrichment scores per sequence under light and dark conditions |
| `data/WRKY_dCIS_dCRE_overlaps.csv` | Pre-computed mapping: WRKY binding sites → deepCRE gene extraction windows |
| `data/wrky_overlaps_reference_mapping.csv` | Reference coordinates for WRKY binding sites |
| Pareto front JSON files | Evolutionary algorithm outputs with mutated sequences and fitness scores |
| DeepCRE model (`.h5`) | Pre-trained TF-binding prediction neural network |

Toggle `USE_NEW_DATA` to switch between light+dark vs. dark-only STAR-seq conditions.

### Processing Steps

1. **Data loading**: Parse STAR-seq FASTA headers (TF identity, genomic coordinates, binding
   status); load deepCRE reference sequences and Pareto fronts from gene folders.

2. **Sequence mapping**: Match STAR-seq fragments to deepCRE reference sequences using 50 bp
   flanking queries; validate against genomic overlap window (INTRAGENIC=500 bp,
   EXTRAGENIC=1000 bp).

3. **Sequence building**: Insert STAR-seq fragments into deepCRE reference sequences to produce
   mutated variants; sequences with >15 mutations relative to reference are skipped.

4. **Prediction**: One-hot encode sequences; run through deepCRE model to obtain binding
   predictions.

5. **Merging**: Join deepCRE predictions with STAR-seq enrichment scores by sequence ID.

6. **Delta calculation**: Compute relative changes (mutated vs. reference) for both deepCRE
   prediction and STAR-seq enrichment.

7. **Bucketing**: Group overlaps by genomic start position into 200 bp buckets; correct short
   edge-spanning overlaps to 170 bp (`FULL_OVERLAP_LENGTH`).

8. **Visualization**: Generate scatter plots with Spearman correlations, per-bucket linear fits,
   and position-series rolling averages; stratify by condition (light/dark) and sequence type
   (reference/synthetic/binding/non-binding).

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BUCKET_SIZE` | 200 | Position window width for bucketing |
| `FULL_OVERLAP_LENGTH` | 170 | Corrected span for short edge overlaps |
| `INTRAGENIC` | 500 | Intragenic padding for extraction windows |
| `EXTRAGENIC` | 1000 | Extragenic padding for extraction windows |
| `INDIVIDUAL_BUCKET_LABELS_TO_PLOT` | `["800-999"]` | Buckets rendered as individual plots |

### Outputs (`correlation/new/` and `correlation/old/`)

| Subdirectory | Content |
|---|---|
| `deepcre_starrseq*/` | Scatter plots: deepCRE prediction vs. STAR-seq enrichment |
| `mutation_deepcre*/` | Scatter plots: mutation count vs. deepCRE prediction delta |
| `mutation_starrseq*/` | Scatter plots: mutation count vs. STAR-seq enrichment delta |
| `correlation_by_position_fixed_window/` | Spearman r, slope, p-value as function of position (fixed window) |
| `correlation_by_position_fixed_number_elements/` | Same, but with fixed number of elements per window |
| `overlap_bucket_fit_parameters.csv` | Per-bucket slope, intercept, Spearman r/p statistics |

Each analysis type is further split by sequence category: `binding`, `non_binding`, `reference`,
`synthetic`, `dark`, `light`.

Figures are rendered at 600 dpi for poster-quality output.

---

## Workflow 2: bHLH Site-to-Gene Overlap Mapping

**Script**: `bHLH_overlaps/generate_bhlh_overlaps.py`

**Goal**: Produce `bHLH_dCIS_dCRE_overlaps.csv` — a pre-computed mapping of bHLH TF binding
sites to the deepCRE extraction windows of overlapping genes. This is the bHLH equivalent of
`WRKY_dCIS_dCRE_overlaps.csv` and serves as input for a future bHLH correlation analysis.

### Inputs

| File | Description |
|------|-------------|
| `data/dCIS_bHLH_in_silico_mutated_GS2025d.fasta` | 2,988 sequences representing 995 unique bHLH binding sites |
| TAIR10 GTF annotation | 32,833 gene features used to define extraction windows |

### Processing Steps

1. Parse and deduplicate FASTA headers to extract unique genomic binding sites
   (`bHLH_<chrom>:<start>-<end>_<variant>_<id>`).

2. Load all genes via `evolution.find_genes()`.

3. For each site, find all genes on the same chromosome that overlap, then compute their
   promoter and terminator extraction windows (1500 bp each, with 500 bp intragenic +
   1000 bp extragenic padding) via `find_start_end()`.

4. Map overlapping genomic positions into the 3020 bp extracted-sequence coordinate frame
   (1500 bp promoter + 20 bp central + 1500 bp terminator) via
   `genomic_to_relative_position()` with strand-aware handling.

5. Emit one row per (binding site, gene-window) overlap.

### Outputs

| File | Columns | Description |
|------|---------|-------------|
| `data/bHLH_dCIS_dCRE_overlaps.csv` | `site_id`, `gene_id`, `strand`, `region`, `start`, `end`, `additional_padding` | Overlap mapping for downstream correlation analysis |

Summary statistics (unique sites, sites with ≥1 overlap, total rows) are printed to stdout.

---

## Data Files Overview

| File | Purpose |
|------|---------|
| `data/WRKY_dCIS_dCRE_overlaps.csv` | Pre-computed WRKY site → deepCRE window mapping (input to Workflow 1) |
| `data/bHLH_dCIS_dCRE_overlaps.csv` | Pre-computed bHLH site → deepCRE window mapping (output of Workflow 2) |
| `data/wrky_overlaps_reference_mapping.csv` | Reference coordinates for WRKY binding locations |
| `data/dCIS_WRKY_in_silico_mutated_GS2025d.fasta` | In-silico mutated WRKY sequences (~494 KB) |
| `data/dCIS_bHLH_in_silico_mutated_GS2025d.fasta` | In-silico mutated bHLH sequences (~623 KB) |
| `data/plantstarr-seq_main_light_and_dark_simon_gernot.csv` | STAR-seq enrichment scores under light/dark conditions |
| `data/plantstarr-seq_main_dark_simon_gernot(1).csv` | Dark-condition-only STAR-seq enrichment scores |
| `data/short_genes.json` | List of genes included in the analysis scope |

---

## External Dependencies

- `evolution` package: `find_genes`, `find_start_end`, `genomic_to_relative_position`,
  `sequences.one_hot_encode`
- `deepCRE` package: model loading and prediction
- `pyfaidx`: FASTA parsing
- `tensorflow.keras`: neural network inference
- `scipy.stats.spearmanr`: Spearman correlation
- `pandas`, `numpy`, `matplotlib`, `seaborn`: data manipulation and visualization

---

## Relationship Between Workflows

```
Workflow 2 (generate_bhlh_overlaps.py)
  └─→ data/bHLH_dCIS_dCRE_overlaps.csv
        └─→ (future bHLH correlation script, analogous to Workflow 1)

Workflow 1 (starrseq_deepcre_correlation_WRKY.py)
  uses: data/WRKY_dCIS_dCRE_overlaps.csv   ← same structure, produced by analogous WRKY mapping
  uses: data/dCIS_WRKY_in_silico_mutated_GS2025d.fasta
  uses: data/plantstarr-seq_*.csv
  outputs: correlation/new/ and correlation/old/
```

The old/new split in `correlation/` reflects two versions of the correlation pipeline run on
different subsets or preprocessing versions of the STAR-seq data.
