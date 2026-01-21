# Hoffie Collaboration Workflow

This workflow analyzes evolved sequences to identify candidates for wet lab validation, with a focus on understanding the genomic context of mutations (exons, introns, UTRs).

## Overview

The workflow consists of three main modules:

1. **`analysis/genomic_annotation.py`** - Annotates mutations with genomic features using GFF/GTF files
2. **`analysis/sequence_processing.py`** - Processes sequences (removes exons, extracts introns/UTRs)
3. **`workflows/hoffie/hoffie.py`** - Complete workflow integrating both modules

## Installation

Make sure you have the package installed in editable mode:

```bash
cd /path/to/evo_result_analysis
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

You need:
- **Mutation data**: JSON file from `summarize_mutations.py`
- **Annotation file**: GFF3 or GTF file with gene structure annotations
- **Reference sequences**: FASTA file (if doing sequence processing)

### 2. Run the Complete Workflow

```bash
python -m workflows.hoffie.hoffie \
    --mutations_json data/all_mutated_sequences_summary.json \
    --gff_file data/arabidopsis.gff3 \
    --output_dir results/hoffie_analysis
```

This will:
1. Load mutation data from your evolutionary algorithm results
2. Annotate each mutation with its genomic context (exon/intron/UTR/intergenic)
3. Generate a summary of mutation distribution across feature types
4. Select top candidates for wet lab validation
5. Export results to CSV and JSON formats

### 3. Filter for Specific Features

To only consider mutations in UTRs:

```bash
python -m workflows.hoffie.hoffie \
    --mutations_json data/mutations.json \
    --gff_file data/arabidopsis.gff3 \
    --output_dir results/utr_candidates \
    --feature_filter 5UTR 3UTR
```

### 4. Apply Fitness and Mutation Count Filters

To select candidates with high fitness and few mutations:

```bash
python -m workflows.hoffie.hoffie \
    --mutations_json data/mutations.json \
    --gff_file data/arabidopsis.gff3 \
    --output_dir results/high_fitness \
    --min_fitness 0.8 \
    --max_mutations 3
```

## Module Usage

### Genomic Annotation (analysis/genomic_annotation.py)

Annotate mutations with genomic features:

```bash
python -m analysis.genomic_annotation \
    --mutations_json data/mutations.json \
    --gff_file data/arabidopsis.gff3 \
    --output annotated_mutations.json
```

**Python API:**

```python
from analysis.summarize_mutations import MutationsGene
from analysis.genomic_annotation import GFFParser, annotate_mutations

# Load mutation data
with open('mutations.json') as f:
    data = json.load(f)
mutations_gene = MutationsGene.from_dict(data['AT1G09440'])

# Parse GFF
gff_parser = GFFParser('arabidopsis.gff3')

# Annotate mutations
annotated = annotate_mutations(mutations_gene, 'AT1G09440', gff_parser)

# Filter for UTR mutations
for gen, sequences in annotated.items():
    for seq in sequences:
        utr_mutations = [ann for ann in seq.annotations 
                        if ann.feature_type in ['5UTR', '3UTR']]
        if utr_mutations:
            print(f"Found {len(utr_mutations)} UTR mutations")
```

### Sequence Processing (analysis/sequence_processing.py)

Remove exons from sequences:

```bash
python -m analysis.sequence_processing \
    --fasta data/reference_sequences.fasta \
    --gff data/arabidopsis.gff3 \
    --output data/sequences_no_exons.fasta \
    --operation remove_exons
```

Extract only introns:

```bash
python -m analysis.sequence_processing \
    --fasta data/reference_sequences.fasta \
    --gff data/arabidopsis.gff3 \
    --output data/introns_only.fasta \
    --operation introns_only
```

Extract only UTRs:

```bash
python -m analysis.sequence_processing \
    --fasta data/reference_sequences.fasta \
    --gff data/arabidopsis.gff3 \
    --output data/utrs_only.fasta \
    --operation utrs_only
```

**Python API:**

```python
from analysis.sequence_processing import cut_exons_from_gene, GFFParser

# Parse GFF
gff_parser = GFFParser('arabidopsis.gff3')

# Remove exons from a sequence
original_sequence = "ATCGATCG..."
modified_sequence, removed_regions = cut_exons_from_gene(
    original_sequence,
    'AT1G09440',
    gff_parser
)

print(f"Original: {len(original_sequence)} bp")
print(f"After removing {len(removed_regions)} exons: {len(modified_sequence)} bp")
```

## Output Files

The workflow generates several output files:

### `annotated_mutations.json`
Complete annotation data with genomic coordinates and feature types for all mutations.

### `mutation_distribution.json`
Summary of how many mutations fall into each feature type:
```json
{
  "intron": 1234,
  "5UTR": 456,
  "3UTR": 234,
  "exon": 567,
  "CDS": 890
}
```

### `top_candidates.csv`
Top candidates for wet lab validation in CSV format, with columns:
- gene_id, chromosome, strand
- fitness, num_mutations
- For each mutation: genomic_position, feature_type, base_change

### `top_candidates.json`
Detailed JSON with all information about top candidates.

## Understanding Feature Types

- **`exon`**: Exonic regions (may include both coding and UTR portions)
- **`CDS`**: Coding sequence (protein-coding regions)
- **`5UTR`**: 5' untranslated region
- **`3UTR`**: 3' untranslated region
- **`intron`**: Intronic regions between exons
- **`intergenic`**: Outside of annotated gene boundaries

## Tips for Analyzing Results

1. **Check mutation distribution first**: Look at `mutation_distribution.json` to understand where mutations are occurring
2. **Filter by feature type**: Use `--feature_filter` to focus on specific regions of interest
3. **Balance fitness and mutations**: Higher fitness often requires more mutations; use `--min_fitness` and `--max_mutations` to find the right balance
4. **Examine candidates in CSV**: Import `top_candidates.csv` into Excel or similar for easy review

## Common Workflows

### Find UTR-only candidates
```bash
python -m workflows.hoffie.hoffie \
    -m mutations.json -g annotation.gff3 -o results/utr_only/ \
    --feature_filter 5UTR 3UTR --max_mutations 2
```

### Find high-fitness candidates with minimal mutations
```bash
python -m workflows.hoffie.hoffie \
    -m mutations.json -g annotation.gff3 -o results/minimal_muts/ \
    --min_fitness 0.9 --max_mutations 2
```

### Analyze specific generation
```bash
python -m workflows.hoffie.hoffie \
    -m mutations.json -g annotation.gff3 -o results/gen1000/ \
    --generation 1000
```

## Troubleshooting

**Gene not found in annotation**: Make sure gene IDs in your mutation data match the GFF file. The script tries to extract gene IDs from names like "1_AT1G09440_gene:..." automatically.

**Import errors**: Make sure you've run `pip install -e .` from the project root after updating the package structure.

**Empty results**: Check that your filters aren't too restrictive. Try running without filters first to see the full mutation distribution.

## Contact

For questions specific to this collaboration, contact the Hoffie lab team.
