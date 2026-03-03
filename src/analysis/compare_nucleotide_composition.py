#!/usr/bin/env python3
"""
Compare nucleotide composition between original extracted genes and mutated genes from evolutionary algorithm.
"""

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from pyfaidx import Fasta
from tqdm import tqdm


def count_nucleotides(sequence: str) -> Dict[str, int]:
    """Count occurrences of each nucleotide in a sequence."""
    counts = Counter(sequence.upper())
    return {
        'A': counts.get('A', 0),
        'C': counts.get('C', 0),
        'G': counts.get('G', 0),
        'T': counts.get('T', 0),
        'N': counts.get('N', 0),
        'Total': len(sequence)
    }


def get_mutated_sequence_from_pareto(pareto_file: str) -> str:
    """Extract the best mutated sequence from pareto front JSON."""
    with open(pareto_file, 'r') as f:
        pareto_data = json.load(f)
    
    # Get the first (best) sequence
    if isinstance(pareto_data, list) and len(pareto_data) > 0:
        best_sequence = pareto_data[0][0]  # [sequence, fitness, mutation_count]
        return best_sequence
    return None


def process_gene_folder(gene_folder: str) -> Tuple[Dict[str, int], Dict[str, int], str]:
    """Process a single gene folder to extract original and mutated nucleotide counts."""
    gene_name = os.path.basename(gene_folder)
    
    # Read reference sequence
    ref_seq_path = os.path.join(gene_folder, "reference_sequence.fa")
    if not os.path.exists(ref_seq_path):
        raise FileNotFoundError(f"Reference sequence not found: {ref_seq_path}")
    
    ref_fasta = Fasta(ref_seq_path)
    # Get the mutation window sequence (not the full sequence)
    ref_sequence = str(ref_fasta['reference_sequence_mutation_window_0_3020'])
    
    # Get mutated sequence from pareto front
    pareto_file = os.path.join(gene_folder, "saved_populations", "pareto_front.json")
    if not os.path.exists(pareto_file):
        raise FileNotFoundError(f"Pareto front not found: {pareto_file}")
    
    mutated_sequence = get_mutated_sequence_from_pareto(pareto_file)
    if not mutated_sequence:
        raise ValueError(f"Could not extract mutated sequence from {pareto_file}")
    
    # Count nucleotides
    original_counts = count_nucleotides(ref_sequence)
    mutated_counts = count_nucleotides(mutated_sequence)
    
    return original_counts, mutated_counts, gene_name


def aggregate_results(results_folder: str) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], List[str]]:
    """Process all gene folders and aggregate nucleotide counts."""
    gene_folders = [
        os.path.join(results_folder, folder) 
        for folder in os.listdir(results_folder) 
        if os.path.isdir(os.path.join(results_folder, folder))
    ]
    
    original_all = {'A': [], 'C': [], 'G': [], 'T': [], 'N': []}
    mutated_all = {'A': [], 'C': [], 'G': [], 'T': [], 'N': []}
    gene_names = []
    
    for gene_folder in tqdm(gene_folders, desc="Processing genes"):
        try:
            original_counts, mutated_counts, gene_name = process_gene_folder(gene_folder)
            gene_names.append(gene_name)
            
            for nucleotide in ['A', 'C', 'G', 'T', 'N']:
                original_all[nucleotide].append(original_counts[nucleotide])
                mutated_all[nucleotide].append(mutated_counts[nucleotide])
        
        except Exception as e:
            print(f"Error processing {os.path.basename(gene_folder)}: {e}")
    
    return original_all, mutated_all, gene_names


def plot_nucleotide_comparison(original_all: Dict[str, List[int]], 
                               mutated_all: Dict[str, List[int]], 
                               gene_names: List[str],
                               output_path: str):
    """Create line plots comparing nucleotide composition."""
    
    # Calculate totals for percentage
    original_totals = [sum(original_all[n][i] for n in ['A', 'C', 'G', 'T']) 
                      for i in range(len(gene_names))]
    mutated_totals = [sum(mutated_all[n][i] for n in ['A', 'C', 'G', 'T']) 
                     for i in range(len(gene_names))]
    
    # Calculate percentages
    original_pct = {
        n: [original_all[n][i] / original_totals[i] * 100 if original_totals[i] > 0 else 0 
            for i in range(len(gene_names))]
        for n in ['A', 'C', 'G', 'T']
    }
    mutated_pct = {
        n: [mutated_all[n][i] / mutated_totals[i] * 100 if mutated_totals[i] > 0 else 0 
            for i in range(len(gene_names))]
        for n in ['A', 'C', 'G', 'T']
    }
    
    # Calculate average percentages
    avg_original = {n: np.mean(original_pct[n]) for n in ['A', 'C', 'G', 'T']}
    avg_mutated = {n: np.mean(mutated_pct[n]) for n in ['A', 'C', 'G', 'T']}
    
    colors = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Average nucleotide composition
    nucleotides = ['A', 'C', 'G', 'T']
    x = np.arange(len(nucleotides))
    width = 0.35
    
    original_vals = [avg_original[n] for n in nucleotides]
    mutated_vals = [avg_mutated[n] for n in nucleotides]
    
    ax1.bar(x - width/2, original_vals, width, label='Original', alpha=0.8, color='lightgray', edgecolor='black')
    ax1.bar(x + width/2, mutated_vals, width, label='Mutated', alpha=0.8, 
            color=[colors[n] for n in nucleotides], edgecolor='black')
    
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_xlabel('Nucleotide', fontsize=12)
    ax1.set_title('Average Nucleotide Composition', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(nucleotides)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Line plot showing change per gene
    x_genes = np.arange(len(gene_names))
    
    for nucleotide in ['A', 'C', 'G', 'T']:
        changes = [mutated_pct[nucleotide][i] - original_pct[nucleotide][i] 
                  for i in range(len(gene_names))]
        ax2.plot(x_genes, changes, marker='o', label=nucleotide, 
                color=colors[nucleotide], linewidth=2, markersize=6)
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Gene Index', fontsize=12)
    ax2.set_ylabel('Change in Percentage (%)', fontsize=12)
    ax2.set_title('Nucleotide Composition Change\n(Mutated - Original)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("NUCLEOTIDE COMPOSITION COMPARISON")
    print("="*60)
    print(f"\nNumber of genes analyzed: {len(gene_names)}")
    print(f"\nAverage Original Composition:")
    for n in ['A', 'C', 'G', 'T']:
        print(f"  {n}: {avg_original[n]:.2f}%")
    print(f"\nAverage Mutated Composition:")
    for n in ['A', 'C', 'G', 'T']:
        print(f"  {n}: {avg_mutated[n]:.2f}%")
    print(f"\nAverage Change (Mutated - Original):")
    for n in ['A', 'C', 'G', 'T']:
        change = avg_mutated[n] - avg_original[n]
        print(f"  {n}: {change:+.2f}%")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Compare nucleotide composition between original and mutated genes'
    )
    parser.add_argument('--results_folder', '-r', required=True,
                       help='Path to the folder containing gene evolution results')
    parser.add_argument('--output', '-o', default='nucleotide_comparison.png',
                       help='Output path for the plot')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_folder):
        raise ValueError(f"Results folder does not exist: {args.results_folder}")
    
    print("Analyzing nucleotide composition...")
    original_all, mutated_all, gene_names = aggregate_results(args.results_folder)
    
    if not gene_names:
        print("No genes were successfully processed!")
        return
    
    plot_nucleotide_comparison(original_all, mutated_all, gene_names, args.output)


if __name__ == "__main__":
    main()
