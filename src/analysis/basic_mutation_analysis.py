#!/usr/bin/env python3
"""
Basic mutation analysis for deepCRE outputs.

Produces:
 - per-file per-gene rows in `summary.csv`
 - per-file plots: `position_counts_{file_basename}.{fmt}` and `rolling_mean_{file_basename}.{fmt}`

Requires project module: analysis.summarize_mutations (MutationsGene)
"""
import os
import json
import argparse
import csv
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

from analysis.summarize_mutations import MutationsGene

def analyze_gene(mg: MutationsGene, gene_name: str, file_label: str):
    """Analyze a single MutationsGene object and return a summary row and per-position counts."""
    gens = sorted(int(g) for g in mg.generation_dict.keys())
    final_gen = gens[-1]

    # gather per-individual mutation counts and counts per (pos,mut)
    individual_mut_counts = []
    pos_base_counts = Counter()
    pos_counts = Counter()

    for seq in mg.generation_dict[final_gen]:
        # seq is a MutatedSequence; if not, attempt to use what we have
        try:
            muts = seq.mutations
        except Exception:
            muts = []
        individual_mut_counts.append(len(muts))
        for pos, ref, mut in muts:
            pos_base_counts[(pos, mut)] += 1
            pos_counts[pos] += 1

    if len(individual_mut_counts) == 0:
        return None, Counter()

    n_individuals = len(individual_mut_counts)
    avg_mut = float(np.mean(individual_mut_counts))
    median_mut = float(np.median(individual_mut_counts))
    unique_positions = len(pos_counts)
    max_count = max(pos_base_counts.values()) if pos_base_counts else 0
    # top 3 hotspots
    top_positions = pos_counts.most_common(3)
    top_pos_summary = ';'.join([f"{p}:{c}" for p, c in top_positions])

    summary_row = {
        'file': os.path.basename(file_label),
        'gene': gene_name,
        'n_individuals': n_individuals,
        'avg_mut_per_ind': avg_mut,
        'median_mut_per_ind': median_mut,
        'unique_positions': unique_positions,
        'max_count_pos_base': max_count,
        'top_positions': top_pos_summary
    }

    return summary_row, pos_counts


def plot_per_gene_overlay(gene_data_list, window, outdir, label, fmt='png'):
    """Plot per-gene rolling means as overlay lines.
    
    Args:
        gene_data_list: List of tuples (gene_name, pos_counts_dict)
        window: Window size for rolling mean
        outdir: Output directory
        label: Label for filename
        fmt: Plot format
    """
    if not gene_data_list:
        return
    
    plt.clf()
    plt.figure(figsize=(12, 6))
    w = max(3, window if window % 2 == 1 else window+1)
    kernel = np.ones(w)/w
    
    # Generate distinct colors
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, len(gene_data_list)))
    
    for idx, (gene_name, pos_counts) in enumerate(gene_data_list):
        if not pos_counts:
            continue
        max_pos = max(pos_counts.keys())
        positions = list(range(max_pos+1))
        counts = np.array([pos_counts.get(i, 0) for i in positions])
        rolling = np.convolve(counts, kernel, mode='same')
        plt.plot(positions, rolling, alpha=0.6, linewidth=0.8, color=colors[idx], label=gene_name if len(gene_data_list) <= 20 else None)
    
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Rolling mean count', fontsize=12)
    plt.title(f'Per-gene rolling means (w={w}) - {label}', fontsize=13)
    if len(gene_data_list) <= 20:
        plt.legend(fontsize=8, loc='upper right', ncol=2)
    plt.grid(alpha=0.3)
    outfile = os.path.join(outdir, f"per_gene_overlay_{label}.{fmt}")
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved per-gene overlay plot: {outfile}")


def plot_heatmap(gene_data_list, window, outdir, label, fmt='png'):
    """Plot heatmap of per-gene rolling means.
    
    Args:
        gene_data_list: List of tuples (gene_name, pos_counts_dict)
        window: Window size for rolling mean
        outdir: Output directory
        label: Label for filename
        fmt: Plot format
    """
    if not gene_data_list:
        return
    
    w = max(3, window if window % 2 == 1 else window+1)
    kernel = np.ones(w)/w
    
    # Find max position across all genes
    max_pos_global = max(max(pc.keys()) if pc else 0 for _, pc in gene_data_list)
    
    # Build matrix: rows = genes, cols = positions
    heatmap_data = []
    gene_labels = []
    
    for gene_name, pos_counts in gene_data_list:
        if not pos_counts:
            continue
        positions = list(range(max_pos_global+1))
        counts = np.array([pos_counts.get(i, 0) for i in positions])
        rolling = np.convolve(counts, kernel, mode='same')
        heatmap_data.append(rolling)
        gene_labels.append(gene_name[:40])  # Truncate long gene names
    
    if not heatmap_data:
        return
    
    heatmap_data = np.array(heatmap_data)
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(14, max(8, len(gene_labels) * 0.3)))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Gene', fontsize=12)
    ax.set_title(f'Heatmap of rolling mean mutations (w={w}) - {label}', fontsize=13)
    
    # Set y-axis labels
    ax.set_yticks(range(len(gene_labels)))
    ax.set_yticklabels(gene_labels, fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rolling mean count', fontsize=11)
    
    outfile = os.path.join(outdir, f"heatmap_{label}.{fmt}")
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap: {outfile}")


def process_file(path, window=31, outdir='out', fmt='png'):
    """Process a JSON file containing multiple genes in MutationsGene.to_dict() format."""
    with open(path, 'r') as f:
        data = json.load(f)
    summary_rows = []
    # aggregate across genes for plotting
    position_totals = Counter()
    gene_data_list = []

    for gene, gene_dict in data.items():
        try:
            mg = MutationsGene.from_dict(gene_dict)
        except Exception as e:
            # skip non-conforming entries
            print(f"Skipping {gene} in {path}: {e}")
            continue

        summary_row, pos_counts = analyze_gene(mg, gene, path)
        if summary_row is None:
            continue
        summary_rows.append(summary_row)
        gene_data_list.append((gene, pos_counts))
        for p, c in pos_counts.items():
            position_totals[p] += c

    # Write plots for file-level aggregate
    label = os.path.basename(path)
    
    # Generate per-gene overlay and heatmap
    if gene_data_list:
        plot_per_gene_overlay(gene_data_list, window, outdir, label, fmt)
        plot_heatmap(gene_data_list, window, outdir, label, fmt)
    
    if position_totals:
        max_pos = max(position_totals)
        positions = list(range(max_pos+1))
        counts = np.array([position_totals.get(i, 0) for i in positions])
        # bar plot
        plt.clf()
        plt.figure(figsize=(10,4))
        plt.bar(positions, counts, width=1.0)
        plt.xlabel('Position')
        plt.ylabel('Counts')
        plt.title(f'Position counts - {os.path.basename(path)}')
        os.makedirs(outdir, exist_ok=True)
        barfile = os.path.join(outdir, f"position_counts_{os.path.basename(path)}.{fmt}")
        plt.savefig(barfile, dpi=150, bbox_inches='tight')

        # rolling mean
        w = max(3, window if window % 2 == 1 else window+1)
        kernel = np.ones(w)/w
        rolling = np.convolve(counts, kernel, mode='same')
        plt.clf()
        plt.figure(figsize=(10,4))
        plt.plot(positions, rolling)
        plt.xlabel('Position')
        plt.ylabel('Rolling mean count')
        plt.title(f'Rolling mean (w={w}) - {os.path.basename(path)}')
        rollfile = os.path.join(outdir, f"rolling_mean_{os.path.basename(path)}.{fmt}")
        plt.savefig(rollfile, dpi=150, bbox_inches='tight')

    return summary_rows


def process_results_folder(results_dir, window=31, outdir='out', fmt='png'):
    """Process an Evolution results directory where each subfolder is a gene folder containing 'reference_sequence.fa' and 'saved_populations'."""
    summary_rows = []
    position_totals = Counter()
    gene_data_list = []

    gene_folders = [os.path.join(results_dir, folder) for folder in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, folder))]
    for gene_folder in gene_folders:
        gene_name = os.path.basename(gene_folder)
        try:
            mg = MutationsGene(gene_folder)
        except Exception as e:
            print(f"Skipping {gene_name}: could not load MutationsGene from folder {gene_folder} ({e})")
            continue

        summary_row, pos_counts = analyze_gene(mg, gene_name, gene_folder)
        if summary_row is None:
            continue
        summary_rows.append(summary_row)
        gene_data_list.append((gene_name, pos_counts))
        for p, c in pos_counts.items():
            position_totals[p] += c

    # Write plots for folder-level aggregate
    label = os.path.basename(results_dir.rstrip(os.sep))
    
    # Generate per-gene overlay and heatmap
    if gene_data_list:
        plot_per_gene_overlay(gene_data_list, window, outdir, label, fmt)
        plot_heatmap(gene_data_list, window, outdir, label, fmt)
    
    if position_totals:
        max_pos = max(position_totals)
        positions = list(range(max_pos+1))
        counts = np.array([position_totals.get(i, 0) for i in positions])
        # bar plot
        plt.clf()
        plt.figure(figsize=(10,4))
        plt.bar(positions, counts, width=1.0)
        plt.xlabel('Position')
        plt.ylabel('Counts')
        plt.title(f'Position counts - {label}')
        os.makedirs(outdir, exist_ok=True)
        barfile = os.path.join(outdir, f"position_counts_{label}.{fmt}")
        plt.savefig(barfile, dpi=150, bbox_inches='tight')

        # rolling mean
        w = max(3, window if window % 2 == 1 else window+1)
        kernel = np.ones(w)/w
        rolling = np.convolve(counts, kernel, mode='same')
        plt.clf()
        plt.figure(figsize=(10,4))
        plt.plot(positions, rolling)
        plt.xlabel('Position')
        plt.ylabel('Rolling mean count')
        plt.title(f'Rolling mean (w={w}) - {label}')
        rollfile = os.path.join(outdir, f"rolling_mean_{label}.{fmt}")
        plt.savefig(rollfile, dpi=150, bbox_inches='tight')

    return summary_rows

def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', help='Single JSON file')
    group.add_argument('--input_dir', help='Directory with JSON files')
    group.add_argument('--results_dir', help='Directory containing Evolution results (each gene in a subfolder)')
    p.add_argument('--output', '-o', default='basic_analysis_out', help='Output folder')
    p.add_argument('--window', '-w', type=int, default=31, help='Rolling window size')
    p.add_argument('--format', '-f', default='png', choices=['png','pdf','svg','jpg'], help='Plot format')
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    all_rows = []

    if args.input:
        print(f"Processing file {args.input} ...")
        rows = process_file(args.input, window=args.window, outdir=args.output, fmt=args.format)
        all_rows.extend(rows)
    elif args.input_dir:
        files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.json')]
        for fp in files:
            print(f"Processing {fp} ...")
            rows = process_file(fp, window=args.window, outdir=args.output, fmt=args.format)
            all_rows.extend(rows)
    elif args.results_dir:
        print(f"Processing results folder {args.results_dir} ...")
        rows = process_results_folder(args.results_dir, window=args.window, outdir=args.output, fmt=args.format)
        all_rows.extend(rows)

    # write summary CSV
    csv_path = os.path.join(args.output, 'summary.csv')
    if all_rows:
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = list(all_rows[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)
        print(f"Wrote summary to {csv_path}")
    else:
        print("No valid gene data found; summary not written.")

if __name__ == '__main__':
    main()