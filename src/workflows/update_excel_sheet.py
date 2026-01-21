#!/usr/bin/env python3
"""
Script to update CSV files with GOF and LOF presence/absence columns.

This script reads gene IDs from list_genes.json and adds relevant columns to each CSV file:
- GOF CSV file: GOF_all_SNPs and GOF_natural_SNPs
- LOF CSV file: LOF_all_SNPs and LOF_natural_SNPs
"""

import json
import os
import pandas as pd
from typing import Any, Dict, List, Optional


def load_gene_data_from_json(json_path: str) -> Dict[str, List[str]]:
    """Load gene data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def categorize_experiment(experiment: str) -> List[str]:
    """Categorize an experiment path into its types."""
    categories = []
    if 'GOF/GOF_multi_mutation' in experiment:
        categories.append('GOF_all_SNPs')
    if 'GOF/GOF_multi_natural' in experiment:
        categories.append('GOF_natural_SNPs')
    if 'LOF/LOF_multi_mutation' in experiment:
        categories.append('LOF_all_SNPs')
    if 'LOF/LOF_multi_natural' in experiment:
        categories.append('LOF_natural_SNPs')
    return categories


def create_gene_mapping(gene_data: Dict[str, List[str]]) -> Dict[str, Dict[str, bool]]:
    """Create a mapping of gene IDs to their presence in experiments."""
    gene_mapping = {}
    
    for gene_id, experiments in gene_data.items():
        gene_mapping[gene_id] = {
            'GOF_all_SNPs': False,
            'GOF_natural_SNPs': False,
            'LOF_all_SNPs': False,
            'LOF_natural_SNPs': False
        }
        
        for experiment in experiments:
            categories = categorize_experiment(experiment)
            for category in categories:
                gene_mapping[gene_id][category] = True
    
    return gene_mapping


def find_gene_id_column(df: pd.DataFrame) -> Optional[str]:
    """Find the column name containing gene IDs."""
    # Prioritize gene_id_Atha columns first (which contain AT gene IDs)
    gene_id_priority = ['gene_id_atha', 'gene_id_Atha', 'gene_id']
    locus_variants = ['locus', 'gene']
    
    # First, try to find gene_id_Atha columns
    for col_name in df.columns:
        if col_name.lower().strip() in [v.lower() for v in gene_id_priority]:
            return col_name
    
    # Fall back to locus/gene columns
    for col_name in df.columns:
        if col_name.lower().strip() in [v.lower() for v in locus_variants]:
            return col_name
    
    return None


def count_genes_by_category(gene_mapping: Dict[str, Dict[str, bool]]) -> Dict[str, int]:
    """Count genes in each category."""
    return {
        'GOF_all_SNPs': sum(1 for g in gene_mapping.values() if g['GOF_all_SNPs']),
        'GOF_natural_SNPs': sum(1 for g in gene_mapping.values() if g['GOF_natural_SNPs']),
        'LOF_all_SNPs': sum(1 for g in gene_mapping.values() if g['LOF_all_SNPs']),
        'LOF_natural_SNPs': sum(1 for g in gene_mapping.values() if g['LOF_natural_SNPs'])
    }


def print_gene_statistics(gene_mapping: Dict[str, Dict[str, bool]]) -> None:
    """Print statistics about gene categories."""
    counts = count_genes_by_category(gene_mapping)
    print(f"Loaded {len(gene_mapping)} genes")
    for category, count in counts.items():
        print(f"  {category}: {count} genes")
    print()


def update_csv_with_columns(csv_path: str, output_path: str, 
                           gene_mapping: Dict[str, Dict[str, bool]], 
                           columns_to_add: List[str]) -> None:
    """Update CSV file by adding new columns with gene presence data."""
    # Read CSV with pandas, skipping the first row (title row) and using row 2 as header
    df = pd.read_csv(csv_path, header=1)
    
    if df.empty:
        print(f"Warning: {csv_path} is empty")
        return
    
    # Find gene ID column
    gene_id_col = find_gene_id_column(df)
    
    if gene_id_col is None:
        print(f"Warning: Could not find gene_id column")
        print(f"Available columns: {list(df.columns)}")
        # Use the 5th column (index 4) as fallback, or first column if there aren't enough
        gene_id_col = df.columns[4] if len(df.columns) > 4 else df.columns[0]
        print(f"Using column: '{gene_id_col}'")
    else:
        print(f"  Using gene ID column: '{gene_id_col}'")
    
    # Add new columns to dataframe based on gene mapping
    for col in columns_to_add:
        df[col] = df[gene_id_col].apply(
            lambda gene_id: 'present' if pd.notna(gene_id) and str(gene_id).strip() in gene_mapping and gene_mapping[str(gene_id).strip()][col] else 'absent'
        )
    
    # Write the updated dataframe to CSV with proper quoting for LibreOffice compatibility
    # QUOTE_ALL ensures all fields are quoted, preventing comma-related parsing issues
    df.to_csv(output_path, index=False, quoting=1)  # quoting=1 is csv.QUOTE_ALL
    
    print(f"  Created {os.path.basename(output_path)}")


def get_output_filename(csv_path: str) -> str:
    """Generate output filename with '_updated' suffix."""
    base, ext = os.path.splitext(csv_path)
    return f"{base}_updated{ext}"


def get_data_directory() -> str:
    """Get the data directory path relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    return os.path.join(project_root, 'data', 'update_excel_sheet')


def get_csv_configurations() -> List[Dict[str, Any]]:
    """Get configuration for CSV files to process."""
    return [
        {
            'file': 'GERNOT_target_list_prototype_SZ20251013_GOF.csv',
            'columns': ['GOF_all_SNPs', 'GOF_natural_SNPs']
        },
        {
            'file': 'GERNOT_target_list_prototype_SZ20251013_LOF.csv',
            'columns': ['LOF_all_SNPs', 'LOF_natural_SNPs']
        }
    ]


def process_csv_file(csv_path: str, gene_mapping: Dict[str, Dict[str, bool]], 
                    columns: List[str]) -> None:
    """Process a single CSV file."""
    if not os.path.exists(csv_path):
        print(f"Warning: {os.path.basename(csv_path)} not found, skipping...")
        return
    
    output_path = get_output_filename(csv_path)
    
    print(f"Processing {os.path.basename(csv_path)}...")
    print(f"  Adding columns: {', '.join(columns)}")
    
    update_csv_with_columns(csv_path, output_path, gene_mapping, columns)
    print()


def main():
    """Main function to orchestrate CSV file updates."""
    data_dir = get_data_directory()
    json_path = os.path.join(data_dir, 'list_genes.json')
    
    print("Loading gene mapping from list_genes.json...")
    gene_data = load_gene_data_from_json(json_path)
    gene_mapping = create_gene_mapping(gene_data)
    print_gene_statistics(gene_mapping)
    
    csv_configs = get_csv_configurations()
    
    for config in csv_configs:
        csv_path = os.path.join(data_dir, config['file'])
        process_csv_file(csv_path, gene_mapping, config['columns'])
    
    print("Done! All CSV files have been updated.")
    print("Note: Original files are unchanged. New files with '_updated' suffix have been created.")


if __name__ == '__main__':
    main()
