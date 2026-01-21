"""
Sequence processing module for manipulating sequences based on genomic annotations.

This module provides functionality to:
- Remove exons from sequences (keeping introns and regulatory regions)
- Extract specific regions (introns only, UTRs only, etc.)
- Generate modified FASTA files with processed sequences
"""

import argparse
import os
from typing import List, Optional, Tuple, Dict
from pyfaidx import Fasta
from dataclasses import dataclass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process sequences by removing or extracting genomic features."
    )

    return parser.parse_args()


def main():
    pass


if __name__ == "__main__":
    main()
