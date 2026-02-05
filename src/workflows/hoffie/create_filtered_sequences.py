#!/usr/bin/env python3
"""
Create filtered FASTA files containing exactly 3 versions for each gene:
1. Reference (0 mutations)
2. Maximum mutations variant
3. Chimeric variant (first half max mutations, second half reference)
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from pyfaidx import Fasta

from analysis.candidate_selection import get_sequences_without_restriction_sites


def extract_gene_and_mutation_count(header):
    """Extract gene name and mutation count from header."""
    # Example: >GOF_greedy_natural_260119_163805_381818__5_AT5G27150_gene:9553299-9557594_260119_164035_004014_mutations_090
    gene_match = re.search(r'__\d_([^_]+)_gene', header)
    mutation_match = re.search(r'mutations_(\d+)', header)
    
    if gene_match and mutation_match:
        gene = gene_match.group(1)
        mutations = int(mutation_match.group(1))
        return gene, mutations
    return None, None


def group_by_gene(fasta):
    """Group sequences by gene name."""
    genes = defaultdict(list)
    
    for record in fasta:
        header = record.name
        seq = str(record)
        gene, mutations = extract_gene_and_mutation_count(header)
        if gene is not None and mutations is not None:
            # Store as (mutations, header, seq) and use gene+mutations as key to handle true duplicates
            key = (gene, mutations)
            # Only keep the first occurrence of each mutation count per gene
            if not any(v[0] == mutations for v in genes[gene]):
                genes[gene].append((mutations, header, seq))
    
    # Sort each gene's sequences by mutation count
    for gene in genes:
        genes[gene].sort(key=lambda x: x[0])
    
    return genes


def create_chimeric_sequence(ref_seq, max_seq):
    """Create chimeric sequence: first half from max_seq, second half from ref_seq."""
    midpoint = len(ref_seq) // 2
    chimeric = max_seq[:midpoint] + ref_seq[midpoint:]
    return chimeric


def create_filtered_file(input_file, output_file):
    """Process a FASTA file and create filtered version."""
    print(f"Processing {input_file.name}...")
    
    fasta = Fasta(str(input_file))
    genes = group_by_gene(fasta)
    
    print(f"  Found {len(genes)} unique genes")
    
    with open(output_file, 'w') as out:
        for gene in sorted(genes.keys()):
            variants = genes[gene]
            
            # Find reference (0 mutations)
            ref_variant = next((v for v in variants if v[0] == 0), None)
            
            # Find max mutations variant
            max_variant = variants[-1]  # Last one after sorting
            
            if ref_variant is None:
                print(f"  Warning: No reference (0 mutations) found for gene {gene}, skipping")
                continue
            
            ref_mutations, ref_header, ref_seq = ref_variant
            max_mutations, max_header, max_seq = max_variant
            
            # Write reference sequence
            out.write(f">{ref_header}\n")
            # Write sequence in 80-character lines (standard FASTA format)
            for i in range(0, len(ref_seq), 80):
                out.write(f"{ref_seq[i:i+80]}\n")
            
            # Write max mutations sequence
            out.write(f">{max_header}\n")
            for i in range(0, len(max_seq), 80):
                out.write(f"{max_seq[i:i+80]}\n")
            
            # Create and write chimeric sequence
            chimeric_seq = create_chimeric_sequence(ref_seq, max_seq)
            chimeric_header = max_header.replace(
                f"mutations_{max_mutations:03d}",
                f"mutations_chimeric_{max_mutations:03d}"
            )
            out.write(f">{chimeric_header}\n")
            for i in range(0, len(chimeric_seq), 80):
                out.write(f"{chimeric_seq[i:i+80]}\n")
    
    print(f"  Created {output_file.name} with {len(genes) * 3} sequences")


def filter_restriction_sites(sequence_path: str, restriction_site_path: str) -> None:
    if not os.path.exists(sequence_path):
        print(f"Error: Path {sequence_path} does not exist")
        return
    if os.path.isdir(sequence_path):
        sequence_paths = [os.path.join(sequence_path, f) for f in os.listdir(sequence_path) if f.endswith('.fa') or f.endswith('.fasta')]
    elif os.path.isfile(sequence_path):
        sequence_paths = [sequence_path]
    else:
        print(f"Error: Path {sequence_path} is neither a file nor a directory")
        return
    for seq_path in sequence_paths:
        output_path = seq_path.replace('.fa', '_no_restriction_sites.fa').replace('.fasta', '_no_restriction_sites.fasta')
        get_sequences_without_restriction_sites(seq_path, restriction_site_path, output_path)
        print(f"Filtered sequences written to {output_path}")



def parse_args():
    parser = argparse.ArgumentParser(
        description="Create filtered FASTA files with reference, max mutations, and chimeric variants"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory containing input FASTA files (default: ./data)"
    )
    parser.add_argument("--remove_restriction_sites", "-r", action="store_true")
    parser.add_argument("--create_chimeras", "-c", action="store_true")

    parser.add_argument("--sequence_path", "-s", type=str, help="Path to sequence file or directory for restriction site filtering")
    parser.add_argument("--restriction_site_path", "-rp", type=str, help="Path to restriction site file")
    
    args = parser.parse_args()
    return args

def create_chimeras(data_dir: Path):
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    # Find all FASTA files
    fasta_files = list(data_dir.glob("*.fa"))
    
    if not fasta_files:
        print(f"No .fa files found in {data_dir}")
        return
    
    print(f"Found {len(fasta_files)} FASTA files to process in {data_dir}\n")
    
    for fasta_file in fasta_files:
        # Create output filename with _filtered suffix
        output_file = data_dir / f"{fasta_file.stem}_filtered.fa"
        create_filtered_file(fasta_file, output_file)
        print()
    
    print("All files processed successfully!")

def main():
    """Process all FASTA files in the specified directory."""
    args =  parse_args()

    if args.create_chimeras:
        create_chimeras(args.data_dir)
    
    if args.remove_restriction_sites:
        if args.sequence_path is None or args.restriction_site_path is None:
            print("Error: --sequence_path and --restriction_site_path must be provided for restriction site filtering")
            return
        filter_restriction_sites(args.sequence_path, args.restriction_site_path)


if __name__ == "__main__":
    main()
