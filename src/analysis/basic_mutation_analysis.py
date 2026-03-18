#!/usr/bin/env python3
"""
Basic mutation analysis for deepCRE evolutionary algorithm outputs.

This script analyzes mutation patterns from evolutionary optimization runs,
focusing on:
  1. WHERE mutations occur (positional hotspots)
  2. WHAT mutations occur (nucleotide transitions: A→C, A→T, etc.)
  3. PATTERNS across genes (overlays, heatmaps)

Produces:
 - per-file per-gene rows in `summary.csv`
 - per-file plots: position counts, rolling means, overlays, heatmaps
 - base change heatmaps: showing specific nucleotide transitions

Requires project module: analysis.summarize_mutations (MutationsGene)
"""
import os
import json
import argparse
import csv
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.summarize_mutations import MutationsGene


def create_run_output_dir(base_output: str, input_path: str) -> str:
    """
    Create a unique per-run output directory based on the input name.

    The resulting directory is:
        <base_output>/<input_name>_<timestamp>
    """
    input_name = os.path.basename(os.path.normpath(input_path))
    if os.path.isfile(input_path):
        input_name = os.path.splitext(input_name)[0]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_output, f"{input_name}_{timestamp}")

    # Avoid rare collisions (e.g., repeated runs within the same second).
    suffix = 1
    unique_dir = run_dir
    while os.path.exists(unique_dir):
        unique_dir = f"{run_dir}_{suffix}"
        suffix += 1

    os.makedirs(unique_dir, exist_ok=False)
    return unique_dir

def analyze_gene(mg: MutationsGene, gene_name: str, file_label: str):
    """
    Analyze a single MutationsGene object at the final generation.
    
    Args:
        mg: MutationsGene object containing evolution data
        gene_name: Name/identifier of the gene
        file_label: Label for the source file/folder
    
    Returns:
        Tuple of (summary_dict, pos_counts, base_change_counts)
        - summary_dict: Statistics about mutations
        - pos_counts: Counter of mutations per position
        - base_change_counts: Counter of (position, ref_base, mut_base) tuples
    """
    # Get the final generation from available generations
    gens = sorted(int(g) for g in mg.generation_dict.keys())
    final_gen = gens[-1]

    # Initialize tracking structures
    individual_mut_counts = []  # Number of mutations per individual
    pos_base_counts = Counter()  # Count of each (position, mutated_base) pair
    pos_counts = Counter()  # Total count per position
    base_change_counts = Counter()  # Count of each (pos, ref→mut) transition

    # Iterate through all sequences in the final generation
    for seq in mg.generation_dict[final_gen]:
        # Extract mutations from each sequence (MutatedSequence object)
        try:
            muts = seq.mutations  # List of (position, reference_base, mutated_base)
        except Exception:
            muts = []
        
        individual_mut_counts.append(len(muts))
        
        # Track each mutation by position and base change
        for pos, ref, mut in muts:
            pos_base_counts[(pos, mut)] += 1
            pos_counts[pos] += 1
            base_change_counts[(pos, ref, mut)] += 1  # Track ref→mut transition

    # Return empty if no individuals found
    if len(individual_mut_counts) == 0:
        return None, Counter(), Counter()

    # Calculate summary statistics
    n_individuals = len(individual_mut_counts)
    avg_mut = float(np.mean(individual_mut_counts))
    median_mut = float(np.median(individual_mut_counts))
    unique_positions = len(pos_counts)
    max_count = max(pos_base_counts.values()) if pos_base_counts else 0
    
    # Identify top 3 mutation hotspots
    top_positions = pos_counts.most_common(3)
    top_pos_summary = ';'.join([f"{p}:{c}" for p, c in top_positions])

    # Compile summary row for CSV output
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

    return summary_row, pos_counts, base_change_counts


def plot_per_gene_overlay(gene_data_list, window, outdir, label, fmt='png'):
    """
    Plot per-gene rolling means as overlay lines for comparison.
    
    Allows visual comparison of mutation patterns across multiple genes,
    showing whether certain regions are consistently mutated.
    
    Args:
        gene_data_list: List of tuples (gene_name, pos_counts_dict)
        window: Window size for rolling mean smoothing
        outdir: Output directory for saved plot
        label: Label for filename
        fmt: Plot format (png, pdf, svg, jpg)
    """
    if not gene_data_list:
        return
    
    plt.clf()
    plt.figure(figsize=(12, 6))
    
    # Ensure odd window size for symmetric smoothing
    w = max(3, window if window % 2 == 1 else window+1)
    kernel = np.ones(w)/w  # Uniform kernel for moving average
    
    # Generate distinct colors for each gene
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
    """
    Plot heatmap of per-gene rolling means.
    
    Creates a 2D heatmap where:
    - Rows represent different genes
    - Columns represent genomic positions
    - Color intensity shows mutation frequency (after smoothing)
    
    Args:
        gene_data_list: List of tuples (gene_name, pos_counts_dict)
        window: Window size for rolling mean smoothing
        outdir: Output directory for saved plot
        label: Label for filename
        fmt: Plot format (png, pdf, svg, jpg)
    """
    if not gene_data_list:
        return
    
    # Ensure odd window size for symmetric smoothing
    w = max(3, window if window % 2 == 1 else window+1)
    kernel = np.ones(w)/w  # Uniform kernel for moving average
    
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
    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlGnBu', interpolation='nearest')
    
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


def plot_base_change_heatmap(gene_data_list, window, outdir, label, fmt='png'):
    """
    Plot base change patterns across positions with three visualizations:
    1. Heatmap: Base changes (rows) vs positions (cols) with rolling mean
    2. Stacked area chart showing composition of all base changes at each position
    3. Line plot showing the dominant (most frequent) transition at each position
    
    Args:
        gene_data_list: List of tuples (gene_name, base_change_counts)
                       where base_change_counts is Counter of (pos, ref, mut)
        window: Window size for rolling mean smoothing
        outdir: Output directory for saved plot
        label: Label for filename
        fmt: Plot format (png, pdf, svg, jpg)
    """
    if not gene_data_list:
        return
    
    # Define all possible base changes (transitions and transversions)
    bases = ['A', 'C', 'G', 'T']
    base_changes = [f"{ref}→{mut}" for ref in bases for mut in bases if ref != mut]
    
    # Define colors for each base change type
    colors_dict = {
        'A→C': '#e6194b', 'A→G': '#f58231', 'A→T': '#ffe119',
        'C→A': '#bfef45', 'C→G': '#3cb44b', 'C→T': '#42d4f4',
        'G→A': '#4363d8', 'G→C': '#911eb4', 'G→T': '#f032e6',
        'T→A': '#fabebe', 'T→C': '#ffd8b1', 'T→G': '#aaffc3'
    }
    
    # Find max position across all genes
    max_pos = 0
    for gene_name, base_change_counts in gene_data_list:
        if base_change_counts:
            for pos, ref, mut in base_change_counts.keys():
                max_pos = max(max_pos, pos)
    
    if max_pos == 0:
        print(f"No base change data to plot for {label}")
        return
    
    # Build position-change matrix: aggregate counts across all genes
    position_change_matrix = defaultdict(lambda: defaultdict(int))
    
    for gene_name, base_change_counts in gene_data_list:
        for (pos, ref, mut), count in base_change_counts.items():
            change_key = f"{ref}→{mut}"
            if change_key in base_changes:
                position_change_matrix[change_key][pos] += count
    
    if not position_change_matrix:
        print(f"No base change data to plot for {label}")
        return
    
    # Prepare data with rolling mean smoothing
    positions = np.array(range(max_pos + 1))
    w = max(3, window if window % 2 == 1 else window + 1)
    kernel = np.ones(w) / w
    
    smoothed_data = {}
    for change in base_changes:
        counts = np.array([position_change_matrix[change].get(pos, 0) for pos in positions])
        rolling = np.convolve(counts, kernel, mode='same')
        smoothed_data[change] = rolling
    
    # === PLOT 0: Heatmap (Base changes vs Positions) ===
    plt.clf()
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create matrix for heatmap
    matrix_data = np.array([smoothed_data[change] for change in base_changes])
    
    # Show position labels every 500 bp
    tick_positions = list(range(0, len(positions), 500))
    tick_labels = [str(positions[i]) for i in tick_positions if i < len(positions)]
    tick_positions = [i for i in tick_positions if i < len(positions)]
    
    # Use seaborn for clean heatmap
    sns.heatmap(matrix_data, 
                xticklabels=False,  # We'll set custom ticks below
                yticklabels=base_changes,
                cmap='YlGnBu', 
                annot=False,
                cbar_kws={'label': 'Rolling mean count'},
                ax=ax)
    
    # Set custom x-axis ticks at every 500 bp
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=8)
    
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Base Change', fontsize=12)
    ax.set_title(f'Base Change Heatmap (window={w}, aggregated across genes) - {label}', fontsize=13)
    
    plt.tight_layout()
    outfile = os.path.join(outdir, f"base_change_heatmap_{label}.{fmt}")
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved base change heatmap: {outfile}")
    
    # === PLOT 0b: Per-Gene Base Change Heatmap (Genes vs Base Changes) ===
    plt.clf()
    
    # Build matrix: rows = genes, cols = base change types
    gene_change_matrix = defaultdict(lambda: defaultdict(int))
    gene_names = []
    
    for gene_name, base_change_counts in gene_data_list:
        gene_names.append(gene_name)
        for (pos, ref, mut), count in base_change_counts.items():
            change_key = f"{ref}→{mut}"
            if change_key in base_changes:
                gene_change_matrix[gene_name][change_key] += count
    
    if gene_change_matrix:
        # Create matrix for heatmap
        matrix_data = np.zeros((len(gene_names), len(base_changes)))
        for i, gene in enumerate(gene_names):
            for j, change in enumerate(base_changes):
                matrix_data[i, j] = gene_change_matrix[gene].get(change, 0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(8, len(gene_names) * 0.3)))
        
        sns.heatmap(matrix_data, 
                    xticklabels=base_changes,
                    yticklabels=[g[:50] for g in gene_names],  # Truncate long names
                    cmap='YlGnBu', 
                    annot=False,
                    cbar_kws={'label': 'Total count'},
                    ax=ax,
                    linewidths=0.5,
                    linecolor='lightgray')
        
        ax.set_xlabel('Base Change Type', fontsize=12)
        ax.set_ylabel('Gene', fontsize=12)
        ax.set_title(f'Base Change Profile per Gene - {label}', fontsize=13)
        
        plt.tight_layout()
        outfile = os.path.join(outdir, f"base_change_per_gene_{label}.{fmt}")
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"Saved per-gene base change heatmap: {outfile}")
    
    # === PLOT 1: Stacked Area Chart ===
    plt.clf()
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Stack the data
    stack_data = np.array([smoothed_data[change] for change in base_changes])
    colors = [colors_dict[change] for change in base_changes]
    
    ax.stackplot(positions, *stack_data, labels=base_changes, colors=colors, alpha=0.8)
    
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Rolling Mean Count', fontsize=12)
    ax.set_title(f'Base Change Composition (window={w}, stacked across genes) - {label}', fontsize=13)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, ncol=1)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    outfile = os.path.join(outdir, f"base_change_stacked_{label}.{fmt}")
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved stacked area chart: {outfile}")
    
    # === PLOT 2: Dominant Transition at Each Position ===
    plt.clf()
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Find dominant transition at each position
    dominant_changes = []
    dominant_counts = []
    
    for pos in positions:
        max_count = 0
        max_change = None
        for change in base_changes:
            count = smoothed_data[change][pos]
            if count > max_count:
                max_count = count
                max_change = change
        dominant_changes.append(max_change if max_change else base_changes[0])
        dominant_counts.append(max_count)
    
    # Create colored line segments
    for i in range(len(positions) - 1):
        change = dominant_changes[i]
        color = colors_dict.get(change, 'gray')
        ax.plot(positions[i:i+2], dominant_counts[i:i+2], 
               color=color, linewidth=2, alpha=0.8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_dict[change], label=change, alpha=0.8) 
                      for change in base_changes]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
             fontsize=9, ncol=1, title='Dominant Change')
    
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Count of Dominant Transition', fontsize=12)
    ax.set_title(f'Most Frequent Base Change per Position (window={w}) - {label}', fontsize=13)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    outfile = os.path.join(outdir, f"base_change_dominant_{label}.{fmt}")
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved dominant transition plot: {outfile}")


def plot_pre_post_base_percentages(gene_mg_list, outdir, label, fmt='png', window=50):
    """
    Plot per-position base composition percentages before and after mutation,
    derived directly from the full reference sequences and the mutated sequences
    stored in the final generation of each MutationsGene object.

    Produces two line plots (4 lines each — A, C, G, T):
      1. ``base_percentage_before_<label>.<fmt>`` – percentages in the reference
         sequences aggregated across all genes.
      2. ``base_percentage_after_<label>.<fmt>`` – percentages across all
         final-generation mutated sequences aggregated across all genes.

    A rolling mean (uniform kernel) smooths each line so local sequence-composition
    trends are visible above per-position noise.  A window of 50 bp is the default
    — roughly 3–5 % of a typical 1–2 kbp regulatory sequence, the same scale used
    for standard GC-content sliding-window analysis.

    Args:
        gene_mg_list: List of tuples ``(gene_name, MutationsGene)``
        outdir: Output directory for saved plots
        label: Label for filename
        fmt: Plot format (png, pdf, svg, jpg)
        window: Rolling-mean window size in bp (default 50)
    """
    if not gene_mg_list:
        return

    bases = ['A', 'C', 'G', 'T']
    base_colors = {'A': '#e6194b', 'C': '#3cb44b', 'G': '#4363d8', 'T': '#f58231'}

    # Per-position base counts accumulated across all genes
    ref_counts: defaultdict = defaultdict(Counter)   # from reference sequences
    mut_counts: defaultdict = defaultdict(Counter)   # from final-generation mutated sequences

    for _, mg in gene_mg_list:
        # Reference: one sequence per gene
        for pos, base in enumerate(mg.reference_sequence):
            if base in bases:
                ref_counts[pos][base] += 1

        # Mutated: every sequence in the final generation
        final_gen = sorted(mg.generation_dict.keys())[-1]
        for seq in mg.generation_dict[final_gen]:
            for pos, base in enumerate(seq.mutated_sequence):
                if base in bases:
                    mut_counts[pos][base] += 1

    if not ref_counts and not mut_counts:
        print(f"No sequence data to plot for {label}")
        return

    max_pos = max(
        max(ref_counts.keys(), default=0),
        max(mut_counts.keys(), default=0)
    )
    positions = np.arange(max_pos + 1)

    def _percentages(position_counts):
        perc = {}
        for base in bases:
            arr = np.zeros(len(positions))
            for i, pos in enumerate(positions):
                total = sum(position_counts[pos].values())
                if total > 0:
                    arr[i] = 100.0 * position_counts[pos].get(base, 0) / total
            perc[base] = arr
        return perc

    ref_perc = _percentages(ref_counts)
    mut_perc = _percentages(mut_counts)

    # Rolling-mean kernel — ensure odd size for symmetric smoothing
    w = max(3, window if window % 2 == 1 else window + 1)
    kernel = np.ones(w) / w

    def _smooth(arr):
        return np.convolve(arr, kernel, mode='same')

    os.makedirs(outdir, exist_ok=True)

    for perc, title_suffix, fname_suffix in [
        (ref_perc, 'Reference Sequences (Before Mutation)', 'before'),
        (mut_perc, 'Mutated Sequences (After Mutation)',    'after'),
    ]:
        plt.clf()
        fig, ax = plt.subplots(figsize=(12, 5))
        for base in bases:
            # faint raw signal
            ax.plot(positions, perc[base], color=base_colors[base], linewidth=0.4, alpha=0.25)
            # bold smoothed signal
            ax.plot(positions, _smooth(perc[base]), label=base, color=base_colors[base], linewidth=1.8)
        ax.set_xlabel('Position', fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(f'Base Composition – {title_suffix} – {label}  (rolling window={w} bp)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(title='Base')
        plt.tight_layout()
        outfile = os.path.join(outdir, f"base_percentage_{fname_suffix}_{label}.{fmt}")
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"Saved base percentage plot ({fname_suffix}): {outfile}")


def process_file(path, window=31, outdir='out', fmt='png'):
    """
    Process a JSON file containing multiple genes in MutationsGene.to_dict() format.
    
    Args:
        path: Path to JSON file
        window: Rolling mean window size
        outdir: Output directory
        fmt: Plot format
    
    Returns:
        List of summary row dictionaries for CSV output
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    summary_rows = []
    position_totals = Counter()  # Aggregate position counts across genes
    gene_data_list = []  # For position-based plots
    gene_base_change_list = []  # For base-change plots
    gene_mg_list = []  # (gene_name, MutationsGene) for sequence-based plots

    # Process each gene in the JSON file
    for gene, gene_dict in data.items():
        try:
            mg = MutationsGene.from_dict(gene_dict)
        except Exception as e:
            # Skip entries that don't conform to expected format
            print(f"Skipping {gene} in {path}: {e}")
            continue

        # Analyze the gene and extract mutation statistics
        summary_row, pos_counts, base_change_counts = analyze_gene(mg, gene, path)
        if summary_row is None:
            continue
        
        summary_rows.append(summary_row)
        gene_data_list.append((gene, pos_counts))
        gene_base_change_list.append((gene, base_change_counts))
        gene_mg_list.append((gene, mg))
        
        # Aggregate position counts across all genes
        for p, c in pos_counts.items():
            position_totals[p] += c

    # Generate plots for file-level aggregate
    label = os.path.basename(path)
    
    # Generate per-gene overlay, positional heatmap, and base-change heatmap
    if gene_data_list:
        plot_per_gene_overlay(gene_data_list, window, outdir, label, fmt)
        plot_heatmap(gene_data_list, window, outdir, label, fmt)
        plot_base_change_heatmap(gene_base_change_list, window, outdir, label, fmt)
        plot_pre_post_base_percentages(gene_mg_list, outdir, label, fmt, window=window)
    
    # Generate aggregate position plots if we have data
    if position_totals:
        max_pos = max(position_totals)
        positions = list(range(max_pos+1))
        counts = np.array([position_totals.get(i, 0) for i in positions])
        
        os.makedirs(outdir, exist_ok=True)
        
        # Bar plot: Raw mutation counts at each position
        plt.clf()
        plt.figure(figsize=(10,4))
        plt.bar(positions, counts, width=1.0)
        plt.xlabel('Position', fontsize=11)
        plt.ylabel('Total Mutation Count', fontsize=11)
        plt.title(f'Aggregate Position Counts - {os.path.basename(path)}', fontsize=12)
        plt.grid(alpha=0.3, axis='y')
        barfile = os.path.join(outdir, f"position_counts_{os.path.basename(path)}.{fmt}")
        plt.savefig(barfile, dpi=150, bbox_inches='tight')
        print(f"Saved bar plot: {barfile}")

        # Rolling mean plot: Smoothed mutation frequency across positions
        w = max(3, window if window % 2 == 1 else window+1)
        kernel = np.ones(w)/w
        rolling = np.convolve(counts, kernel, mode='same')
        plt.clf()
        plt.figure(figsize=(10,4))
        plt.plot(positions, rolling, linewidth=1.5)
        plt.xlabel('Position', fontsize=11)
        plt.ylabel('Rolling Mean Count', fontsize=11)
        plt.title(f'Smoothed Mutation Frequency (window={w}) - {os.path.basename(path)}', fontsize=12)
        plt.grid(alpha=0.3)
        rollfile = os.path.join(outdir, f"rolling_mean_{os.path.basename(path)}.{fmt}")
        plt.savefig(rollfile, dpi=150, bbox_inches='tight')
        print(f"Saved rolling mean plot: {rollfile}")

    return summary_rows


def process_results_folder(results_dir, window=31, outdir='out', fmt='png'):
    """
    Process an Evolution results directory structure.
    
    Expected structure:
        results_dir/
            gene1/
                reference_sequence.fa
                saved_populations/
            gene2/
                reference_sequence.fa
                saved_populations/
            ...
    
    Args:
        results_dir: Path to results directory
        window: Rolling mean window size
        outdir: Output directory
        fmt: Plot format
    
    Returns:
        List of summary row dictionaries for CSV output
    """
    summary_rows = []
    position_totals = Counter()
    gene_data_list = []  # For position-based plots
    gene_base_change_list = []  # For base-change plots
    gene_mg_list = []  # (gene_name, MutationsGene) for sequence-based plots

    # Iterate through each gene folder
    gene_folders = [os.path.join(results_dir, folder) for folder in os.listdir(results_dir) 
                    if os.path.isdir(os.path.join(results_dir, folder))]
    
    for gene_folder in gene_folders:
        gene_name = os.path.basename(gene_folder)
        try:
            # Load MutationsGene from the folder structure
            mg = MutationsGene(gene_folder)
        except Exception as e:
            print(f"Skipping {gene_name}: could not load MutationsGene from folder {gene_folder} ({e})")
            continue

        # Analyze the gene and extract mutation statistics
        summary_row, pos_counts, base_change_counts = analyze_gene(mg, gene_name, gene_folder)
        if summary_row is None:
            continue
        
        summary_rows.append(summary_row)
        gene_data_list.append((gene_name, pos_counts))
        gene_base_change_list.append((gene_name, base_change_counts))
        gene_mg_list.append((gene_name, mg))
        
        # Aggregate position counts across all genes
        for p, c in pos_counts.items():
            position_totals[p] += c

    # Generate plots for folder-level aggregate
    label = os.path.basename(results_dir.rstrip(os.sep))
    
    # Generate per-gene overlay, positional heatmap, and base-change heatmap
    if gene_data_list:
        plot_per_gene_overlay(gene_data_list, window, outdir, label, fmt)
        plot_heatmap(gene_data_list, window, outdir, label, fmt)
        plot_base_change_heatmap(gene_base_change_list, window, outdir, label, fmt)
        plot_pre_post_base_percentages(gene_mg_list, outdir, label, fmt, window=window)
    
    # Generate aggregate position plots if we have data
    if position_totals:
        max_pos = max(position_totals)
        positions = list(range(max_pos+1))
        counts = np.array([position_totals.get(i, 0) for i in positions])
        
        os.makedirs(outdir, exist_ok=True)
        
        # Bar plot: Raw mutation counts at each position
        plt.clf()
        plt.figure(figsize=(10,4))
        plt.bar(positions, counts, width=1.0)
        plt.xlabel('Position', fontsize=11)
        plt.ylabel('Total Mutation Count', fontsize=11)
        plt.title(f'Aggregate Position Counts - {label}', fontsize=12)
        plt.grid(alpha=0.3, axis='y')
        barfile = os.path.join(outdir, f"position_counts_{label}.{fmt}")
        plt.savefig(barfile, dpi=150, bbox_inches='tight')
        print(f"Saved bar plot: {barfile}")

        # Rolling mean plot: Smoothed mutation frequency across positions
        w = max(3, window if window % 2 == 1 else window+1)
        kernel = np.ones(w)/w
        rolling = np.convolve(counts, kernel, mode='same')
        plt.clf()
        plt.figure(figsize=(10,4))
        plt.plot(positions, rolling, linewidth=1.5)
        plt.xlabel('Position', fontsize=11)
        plt.ylabel('Rolling Mean Count', fontsize=11)
        plt.title(f'Smoothed Mutation Frequency (window={w}) - {label}', fontsize=12)
        plt.grid(alpha=0.3)
        rollfile = os.path.join(outdir, f"rolling_mean_{label}.{fmt}")
        plt.savefig(rollfile, dpi=150, bbox_inches='tight')
        print(f"Saved rolling mean plot: {rollfile}")

    return summary_rows

def main():
    """
    Main entry point for command-line execution.
    
    Supports three input modes:
    1. Single JSON file (--input)
    2. Directory of JSON files (--input_dir)
    3. Evolution results directory structure (--results_dir)
    
    Outputs:
    - summary.csv: Per-gene statistics
    - Various plots: position counts, rolling means, overlays, heatmaps
    - Base change heatmap: nucleotide transition patterns
    """
    p = argparse.ArgumentParser(
        description='Analyze mutation patterns from evolutionary algorithm outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', help='Single JSON file containing MutationsGene data')
    group.add_argument('--input_dir', help='Directory containing multiple JSON files')
    group.add_argument('--results_dir', help='Evolution results directory (each gene in a subfolder)')
    
    p.add_argument('--output', '-o', default='basic_analysis_out', 
                   help='Base output directory for run folders (default: basic_analysis_out)')
    p.add_argument('--window', '-w', type=int, default=31, 
                   help='Rolling window size for smoothing (default: 31)')
    p.add_argument('--format', '-f', default='png', choices=['png','pdf','svg','jpg'], 
                   help='Output plot format (default: png)')
    
    args = p.parse_args()

    # Create a unique folder for this run using the selected input name.
    selected_input = args.input or args.input_dir or args.results_dir
    os.makedirs(args.output, exist_ok=True)
    run_output = create_run_output_dir(args.output, selected_input)
    print(f"Run output directory: {run_output}")

    all_rows = []

    # Process input based on mode
    if args.input:
        # Mode 1: Single JSON file
        print(f"Processing single file: {args.input}")
        rows = process_file(args.input, window=args.window, outdir=run_output, fmt=args.format)
        all_rows.extend(rows)
        
    elif args.input_dir:
        # Mode 2: Directory of JSON files
        files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
                 if f.endswith('.json')]
        print(f"Found {len(files)} JSON files in {args.input_dir}")
        
        for fp in files:
            print(f"Processing: {fp}")
            rows = process_file(fp, window=args.window, outdir=run_output, fmt=args.format)
            all_rows.extend(rows)
            
    elif args.results_dir:
        # Mode 3: Evolution results folder structure
        print(f"Processing Evolution results folder: {args.results_dir}")
        rows = process_results_folder(args.results_dir, window=args.window, 
                                     outdir=run_output, fmt=args.format)
        all_rows.extend(rows)

    # Write summary CSV with per-gene statistics
    csv_path = os.path.join(run_output, 'summary.csv')
    if all_rows:
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = list(all_rows[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)
        print(f"\n✓ Summary CSV written to: {csv_path}")
        print(f"✓ Total genes analyzed: {len(all_rows)}")
    else:
        print("\n✗ No valid gene data found; summary not written.")

if __name__ == '__main__':
    main()