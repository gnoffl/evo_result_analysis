"""
Genomic annotation module for mapping mutations to genomic features.

This module provides functionality to:
- Parse GFF3/GTF annotation files
- Map sequence-relative mutation positions to genomic coordinates
- Classify mutations by feature type (exon, intron, UTR, intergenic)
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from analysis.summarize_mutations import MutatedSequence, MutationsGene


@dataclass
class GenomicFeature:
    """Represents a genomic feature from GFF/GTF annotation."""
    chromosome: str
    source: str
    feature_type: str  # e.g., 'gene', 'mRNA', 'exon', 'CDS', 'five_prime_UTR', 'three_prime_UTR'
    start: int  # 1-based, inclusive
    end: int    # 1-based, inclusive
    score: str
    strand: str  # '+' or '-'
    phase: str
    attributes: Dict[str, str]

    def overlaps(self, position: int) -> bool:
        """Check if a genomic position overlaps with this feature."""
        return self.start <= position <= self.end


@dataclass
class MutationAnnotation:
    """Annotation for a single mutation."""
    seq_position: int  # 0-based position in sequence
    genomic_position: int  # 1-based genomic coordinate
    ref_base: str
    mut_base: str
    feature_type: str  # 'exon', 'intron', '5UTR', '3UTR', 'intergenic', 'CDS'
    feature_id: Optional[str] = None  # ID of the overlapping feature
    gene_id: Optional[str] = None


@dataclass
class AnnotatedMutatedSequence:
    """Wraps a MutatedSequence with genomic annotations."""
    mutated_sequence: MutatedSequence
    gene_id: str
    chromosome: str
    gene_start: int  # 1-based genomic coordinate
    gene_end: int
    strand: str
    annotations: List[MutationAnnotation]
    generation: Optional[int] = None  # Generation number if applicable

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'mutations': repr(self.mutated_sequence),
            'fitness': self.mutated_sequence.fitness,
            'gene_id': self.gene_id,
            'chromosome': self.chromosome,
            'gene_start': self.gene_start,
            'gene_end': self.gene_end,
            'strand': self.strand,
            'generation': self.generation,
            'annotations': [asdict(ann) for ann in self.annotations]
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = []
        lines.append(f"Gene: {self.gene_id}")
        lines.append(f"Location: {self.chromosome} [{self.gene_start}-{self.gene_end}] ({self.strand} strand)")
        if self.generation is not None:
            lines.append(f"Generation: {self.generation}")
        lines.append(f"Fitness: {self.mutated_sequence.fitness:.6f}")
        lines.append(f"Number of mutations: {len(self.annotations)}")
        lines.append("\nMutations:")
        for ann in self.annotations:
            lines.append(f"  Seq pos {ann.seq_position:4d} -> Genomic pos {ann.genomic_position:10d}: {ann.ref_base}>{ann.mut_base}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Unambiguous string representation."""
        return f"AnnotatedMutatedSequence(gene_id={self.gene_id}, mutations={len(self.annotations)}, fitness={self.mutated_sequence.fitness:.6f})"
    
    @staticmethod
    def parse_sequence_name(seq_name: str) -> Tuple[str, str, int, int]:
        """Parse sequence name to extract genomic coordinates.
        
        Expected format: {index}_{gene_id}_gene:{start}-{end}_{timestamps...}
        Example: 3_Zm00001eb130940_gene:59589070-59596715_260205_125432_265348
        
        Args:
            seq_name: Sequence name from summarize_mutations output
            
        Returns:
            Tuple of (gene_id, chromosome, start, end)
        """
        # Split by underscores and find the gene:start-end pattern
        parts = seq_name.split('_')
        
        # Find the gene_id and coordinates part
        gene_id = None
        genomic_coords = None
        
        for i, part in enumerate(parts):
            if 'gene:' in part:
                gene_id = "_".join(parts[:i])  # Gene ID is before gene:coordinates
                genomic_coords = part.split('gene:')[1]
                break
        
        if not gene_id or not genomic_coords:
            raise ValueError(f"Could not parse genomic coordinates from sequence name: {seq_name}")
        
        # Parse start-end coordinates
        start_str, end_str = genomic_coords.split('-')
        start = int(start_str)
        end = int(end_str)
        
        return gene_id, gene_id, start, end
    
    @classmethod
    def from_mutated_sequence(
        cls,
        seq_name: str,
        mutated_sequence: MutatedSequence,
        intragenic: int = 500,
        extragenic: int = 1000,
        central_padding: int = 20,
        generation: Optional[int] = None
    ) -> 'AnnotatedMutatedSequence':
        """Create an AnnotatedMutatedSequence from a sequence name and MutatedSequence.
        
        The extracted sequences have two regions (promoter + terminator) separated by central padding:
        - Promoter: extragenic bp upstream + intragenic bp downstream of TSS
        - Central padding: N nucleotides (default 20)
        - Terminator: intragenic bp upstream + extragenic bp downstream of TTS
        
        Args:
            seq_name: Sequence name with embedded genomic coordinates
            mutated_sequence: MutatedSequence object with mutation data
            intragenic: Number of bases extracted inward from TSS/TTS (default: 500)
            extragenic: Number of bases extracted outward from TSS/TTS (default: 1000)
            central_padding: Number of N's between promoter and terminator (default: 20)
            generation: Generation number for this sequence (optional)
            
        Returns:
            AnnotatedMutatedSequence with genomic coordinates filled in
        """
        gene_id, chromosome, gene_start, gene_end = cls.parse_sequence_name(seq_name)
        
        # Determine strand based on coordinate order
        strand = '+' if gene_start < gene_end else '-'
        
        # Calculate the boundary between promoter and terminator regions
        promoter_length = intragenic + extragenic
        terminator_start_pos = promoter_length + central_padding
        
        # Create MutationAnnotation objects for each mutation
        annotations = []
        for seq_pos, ref_base, mut_base in mutated_sequence.mutations:
            # seq_pos is 0-based position in the extracted sequence
            
            if strand == '+':
                # Forward strand:
                # Promoter region [0, promoter_length): gene_start - extragenic to gene_start + intragenic
                # Terminator region [terminator_start_pos, ...): gene_end - intragenic to gene_end + extragenic
                
                if seq_pos < promoter_length:
                    # In promoter region
                    # Position 0 maps to (gene_start - extragenic + 1) in 1-based coords
                    # Position seq_pos maps to (gene_start - extragenic + seq_pos + 1)
                    genomic_pos = gene_start - extragenic + seq_pos + 1
                else:
                    # In terminator region (after central padding)
                    # terminator_start_pos maps to (gene_end - intragenic + 1) in 1-based coords
                    # Position seq_pos maps to (gene_end - intragenic + (seq_pos - terminator_start_pos) + 1)
                    genomic_pos = gene_end - intragenic + (seq_pos - terminator_start_pos) + 1
            else:
                # Reverse strand: sequence is reverse complemented
                # In the extracted sequence, position 0 corresponds to the promoter
                # For reverse: gene_start=TSS (higher coord), gene_end=TTS (lower coord)
                # Promoter region [0, promoter_length): gene_start - intragenic to gene_start + extragenic
                # Terminator region [terminator_start_pos, ...): gene_end - extragenic to gene_end + intragenic
                
                if seq_pos < promoter_length:
                    # In promoter region (upstream of TSS, higher coords for reverse)
                    # Position 0 maps to (gene_start + extragenic) in 1-based coords
                    # Because sequence is reverse complemented, position 0 is the rightmost
                    # The half-open end already equals the 1-based position of the last base
                    genomic_pos = gene_start + extragenic - seq_pos
                else:
                    # In terminator region (downstream of TTS, lower coords for reverse)
                    # terminator_start_pos maps to (gene_end + intragenic) in 1-based coords
                    # The half-open end already equals the 1-based position of the last base
                    genomic_pos = gene_end + intragenic - (seq_pos - terminator_start_pos)
            
            annotation = MutationAnnotation(
                seq_position=seq_pos,
                genomic_position=genomic_pos,
                ref_base=ref_base,
                mut_base=mut_base,
                feature_type='unknown',
                gene_id=gene_id
            )
            annotations.append(annotation)
        
        return cls(
            mutated_sequence=mutated_sequence,
            gene_id=gene_id,
            chromosome=chromosome,
            gene_start=gene_start,
            gene_end=gene_end,
            strand=strand,
            annotations=annotations,
            generation=generation
        )
    
    @classmethod
    def from_mutations_gene(
        cls,
        seq_name: str,
        mutations_gene: MutationsGene,
        generation: Optional[int] = None,
        intragenic: int = 500,
        extragenic: int = 1000,
        central_padding: int = 20
    ) -> Dict[int, List['AnnotatedMutatedSequence']]:
        """Create multiple AnnotatedMutatedSequence objects from a MutationsGene.
        
        Args:
            seq_name: Sequence name with embedded genomic coordinates
            mutations_gene: MutationsGene object with mutation data
            generation: Specific generation to process (None = all generations)
            intragenic: Number of bases extracted inward from TSS/TTS (default: 500)
            extragenic: Number of bases extracted outward from TSS/TTS (default: 1000)
            central_padding: Number of N's between promoter and terminator (default: 20)
            
        Returns:
            Dict[int, List[AnnotatedMutatedSequence]] mapping generation number to list of annotated sequences
        """
        results = {}
        
        # Determine which generations to process
        generations = [generation] if generation is not None else sorted(mutations_gene.generation_dict.keys())
        
        for gen in generations:
            if gen not in mutations_gene.generation_dict:
                raise ValueError(f"Generation {gen} not found in MutationsGene data")
                
            for mutated_seq in mutations_gene.generation_dict[gen]:
                annotated = cls.from_mutated_sequence(
                    seq_name, 
                    mutated_seq,
                    intragenic=intragenic,
                    extragenic=extragenic,
                    central_padding=central_padding,
                    generation=gen
                )
                if gen not in results:
                    results[gen] = []
                results[gen].append(annotated)
        
        return results

def annotate_mutations(
    mutations_gene: MutationsGene,
    gene_id: str,
    generation: Optional[int] = None
) -> Dict[int, List[AnnotatedMutatedSequence]]:
    return {}


def annotate_all_genes(
    mutations_data: Dict[str, MutationsGene],
    gff_file: str,
    generation: Optional[int] = None,
    output_file: Optional[str] = None
) -> Dict[str, Dict[int, List[AnnotatedMutatedSequence]]]:
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Annotate mutations with genomic features from GFF/GTF files.")
    return parser.parse_args()


def main():
    pass


if __name__ == "__main__":
    main()
