"""
Hoffie collaboration workflow for analyzing evolved sequences.

This workflow:
1. Loads mutation data from evolutionary algorithm results
2. Annotates mutations with genomic features (exons, introns, UTRs)
3. Filters candidates based on mutation types
4. Processes sequences (e.g., removes exons for specific analyses)

Usage example:
    python -m workflows.hoffie.hoffie \\
        --mutations_json data/mutations_summary.json \\
        --gff_file data/arabidopsis.gff3 \\
        --output_dir results/hoffie_analysis
"""

import argparse
import json
import os
from typing import Dict, List, Optional
from collections import defaultdict

from analysis.summarize_mutations import MutationsGene
from analysis.genomic_annotation import (
    annotate_all_genes,
    AnnotatedMutatedSequence
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hoffie collaboration workflow for analyzing evolved sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pass


if __name__ == "__main__":
    main()
