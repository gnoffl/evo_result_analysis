import unittest
import json
import tempfile
import os
from analysis.genomic_annotation import (
    AnnotatedMutatedSequence,
    MutationAnnotation,
    GenomicFeature
)
from analysis.summarize_mutations import MutatedSequence, MutationsGene


class TestParseSequenceName(unittest.TestCase):
    """Test cases for parsing sequence names."""

    def test_parse_forward_strand(self):
        """Test parsing a forward strand sequence name."""
        seq_name = "3_Zm00001eb130940_gene:100-500_260205_125432_265348"
        gene_id, chromosome, start, end = AnnotatedMutatedSequence.parse_sequence_name(seq_name)
        
        self.assertEqual(gene_id, "3_Zm00001eb130940")
        self.assertEqual(chromosome, "3_Zm00001eb130940")
        self.assertEqual(start, 100)
        self.assertEqual(end, 500)

    def test_parse_reverse_strand(self):
        """Test parsing a reverse strand sequence name."""
        seq_name = "1_Zm00001eb002590_gene:500-10000_260205_120109_617394"
        gene_id, chromosome, start, end = AnnotatedMutatedSequence.parse_sequence_name(seq_name)
        
        self.assertEqual(gene_id, "1_Zm00001eb002590")
        self.assertEqual(chromosome, "1_Zm00001eb002590")
        self.assertEqual(start, 500)
        self.assertEqual(end, 10000)

    def test_parse_invalid_format(self):
        """Test parsing an invalid sequence name."""
        seq_name = "invalid_name_without_gene_coords"
        with self.assertRaises(ValueError) as context:
            AnnotatedMutatedSequence.parse_sequence_name(seq_name)
        self.assertIn("Could not parse genomic coordinates", str(context.exception))


class TestGenomicCoordinateConversion(unittest.TestCase):
    """Test cases for converting sequence positions to genomic coordinates."""

    def test_forward_strand_promoter_region(self):
        """Test conversion in promoter region of forward strand gene."""
        seq_name = "1_TESTGENE_gene:10000-12000_timestamp"
        ref_seq = "A" * 3020  # 1000 + 500 + 20 + 500 + 1000
        
        # Create a mutation at position 0 (should be at gene_start - 1000 = 9000)
        mutated_seq_str = "0AC|0.95"
        mutated_seq = MutatedSequence.from_string(ref_seq, mutated_seq_str)
        
        annotated = AnnotatedMutatedSequence.from_mutated_sequence(
            seq_name, mutated_seq, intragenic=500, extragenic=1000, central_padding=20
        )
        
        self.assertEqual(annotated.strand, '+')
        self.assertEqual(len(annotated.annotations), 1)
        # Position 0 should map to 10000 - 1000 + 1 = 9001 (1-based)
        self.assertEqual(annotated.annotations[0].genomic_position, 9001)

    def test_forward_strand_promoter_boundary(self):
        """Test conversion at promoter/terminator boundary of forward strand gene."""
        seq_name = "1_TESTGENE_gene:10000-12000_timestamp"
        ref_seq = "A" * 3020
        
        # Position 1499 (last position of promoter) should be at gene_start - 1000 + 1499 + 1 = 10500 (1-based)
        mutated_seq_str = "1499AC|0.95"
        mutated_seq = MutatedSequence.from_string(ref_seq, mutated_seq_str)
        
        annotated = AnnotatedMutatedSequence.from_mutated_sequence(
            seq_name, mutated_seq, intragenic=500, extragenic=1000, central_padding=20
        )
        
        self.assertEqual(annotated.annotations[0].genomic_position, 10500)

    def test_forward_strand_terminator_region(self):
        """Test conversion in terminator region of forward strand gene."""
        seq_name = "1_TESTGENE_gene:10000-12000_timestamp"
        ref_seq = "A" * 3020
        
        # Position 1520 (first position after padding) should be at gene_end - 500 + 1 = 11501 (1-based)
        mutated_seq_str = "1520AC|0.95"
        mutated_seq = MutatedSequence.from_string(ref_seq, mutated_seq_str)
        
        annotated = AnnotatedMutatedSequence.from_mutated_sequence(
            seq_name, mutated_seq, intragenic=500, extragenic=1000, central_padding=20
        )
        
        self.assertEqual(annotated.annotations[0].genomic_position, 11501)

    def test_forward_strand_terminator_end(self):
        """Test conversion at end of terminator region of forward strand gene."""
        seq_name = "1_TESTGENE_gene:10000-12000_timestamp"
        ref_seq = "A" * 3020
        
        # Position 3019 (last position) should be at gene_end - 500 + 1499 + 1 = 13000 (1-based)
        mutated_seq_str = "3019AC|0.95"
        mutated_seq = MutatedSequence.from_string(ref_seq, mutated_seq_str)
        
        annotated = AnnotatedMutatedSequence.from_mutated_sequence(
            seq_name, mutated_seq, intragenic=500, extragenic=1000, central_padding=20
        )
        
        self.assertEqual(annotated.annotations[0].genomic_position, 13000)

    def test_reverse_strand_promoter_region(self):
        """Test conversion in promoter region of reverse strand gene."""
        seq_name = "1_TESTGENE_gene:12000-10000_timestamp"
        ref_seq = "A" * 3020
        
        # For reverse strand, position 0 should be at gene_start + 1000 = 13000 (1-based)
        mutated_seq_str = "0AC|0.95"
        mutated_seq = MutatedSequence.from_string(ref_seq, mutated_seq_str)
        
        annotated = AnnotatedMutatedSequence.from_mutated_sequence(
            seq_name, mutated_seq, intragenic=500, extragenic=1000, central_padding=20
        )
        
        self.assertEqual(annotated.strand, '-')
        self.assertEqual(annotated.annotations[0].genomic_position, 13000)

    def test_reverse_strand_promoter_end(self):
        """Test conversion at end of promoter region of reverse strand gene."""
        seq_name = "1_TESTGENE_gene:12000-10000_timestamp"
        ref_seq = "A" * 3020
        
        # Position 1499 should be at gene_start + 1000 - 1499 = 11501 (1-based)
        mutated_seq_str = "1499AC|0.95"
        mutated_seq = MutatedSequence.from_string(ref_seq, mutated_seq_str)
        
        annotated = AnnotatedMutatedSequence.from_mutated_sequence(
            seq_name, mutated_seq, intragenic=500, extragenic=1000, central_padding=20
        )
        
        self.assertEqual(annotated.annotations[0].genomic_position, 11501)

    def test_reverse_strand_terminator_region(self):
        """Test conversion in terminator region of reverse strand gene."""
        seq_name = "1_TESTGENE_gene:12000-10000_timestamp"
        ref_seq = "A" * 3020
        
        # Position 1520 (first after padding) should be at gene_end + 500 = 10500 (1-based)
        mutated_seq_str = "1520AC|0.95"
        mutated_seq = MutatedSequence.from_string(ref_seq, mutated_seq_str)
        
        annotated = AnnotatedMutatedSequence.from_mutated_sequence(
            seq_name, mutated_seq, intragenic=500, extragenic=1000, central_padding=20
        )
        
        self.assertEqual(annotated.annotations[0].genomic_position, 10500)

    def test_reverse_strand_terminator_end(self):
        """Test conversion in terminator region of reverse strand gene."""
        seq_name = "1_TESTGENE_gene:12000-10000_timestamp"
        ref_seq = "A" * 3020
        
        # Position 1520 (first after padding) should be at gene_end + 500 = 10500 (1-based)
        mutated_seq_str = "3019AC|0.95"
        mutated_seq = MutatedSequence.from_string(ref_seq, mutated_seq_str)
        
        annotated = AnnotatedMutatedSequence.from_mutated_sequence(
            seq_name, mutated_seq, intragenic=500, extragenic=1000, central_padding=20
        )
        
        self.assertEqual(annotated.annotations[0].genomic_position, 9001)

    def test_multiple_mutations(self):
        """Test conversion with multiple mutations."""
        seq_name = "1_TESTGENE_gene:10000-12000_timestamp"
        ref_seq = "A" * 3020
        
        # Multiple mutations in both regions
        mutated_seq_str = "0AC500AT1499AG1520AG2020AG|0.85"
        mutated_seq = MutatedSequence.from_string(ref_seq, mutated_seq_str)
        
        annotated = AnnotatedMutatedSequence.from_mutated_sequence(
            seq_name, mutated_seq, intragenic=500, extragenic=1000, central_padding=20
        )
        
        self.assertEqual(len(annotated.annotations), 5)
        self.assertEqual(annotated.annotations[0].genomic_position, 9001)   # 10000 - 1000 + 1 (1-based)
        self.assertEqual(annotated.annotations[1].genomic_position, 9501)   # 10000 - 1000 + 500 + 1
        self.assertEqual(annotated.annotations[2].genomic_position, 10500)  # 10000 - 1000 + 1499 + 1
        self.assertEqual(annotated.annotations[3].genomic_position, 11501)  # 12000 - 500 + 1
        self.assertEqual(annotated.annotations[4].genomic_position, 12001)  # 12000 - 500 + 480 + 1


class TestFromMutationsGene(unittest.TestCase):
    """Test cases for creating AnnotatedMutatedSequence from MutationsGene."""

    def setUp(self):
        """Set up test data."""
        self.seq_name = "1_TESTGENE_gene:10000-12000_timestamp"
        self.ref_seq = "A" * 3020

    def test_single_generation(self):
        """Test processing a single generation."""
        # Create test data
        seq1 = MutatedSequence.from_string(self.ref_seq, "0AC100AT|0.95")
        seq2 = MutatedSequence.from_string(self.ref_seq, "200AG|0.90")
        seq3 = MutatedSequence.from_string(self.ref_seq, "|0.89")
        
        mutations_gene = MutationsGene.__new__(MutationsGene)
        mutations_gene.reference_sequence = self.ref_seq
        mutations_gene.generation_dict = {100: [seq1, seq2, seq3]}
        
        annotated_list = AnnotatedMutatedSequence.from_mutations_gene(
            self.seq_name, mutations_gene, generation=100
        )
        
        self.assertEqual(len(annotated_list), 1)
        self.assertEqual(len(annotated_list[100]), 3)
        self.assertEqual(len(annotated_list[100][0].annotations), 2)
        self.assertEqual(len(annotated_list[100][1].annotations), 1)
        self.assertEqual(len(annotated_list[100][2].annotations), 0)

        #check position
        self.assertEqual(annotated_list[100][1].annotations[0].genomic_position, 9201)  # 10000 - 1000 + 200 + 1

    def test_all_generations(self):
        """Test processing all generations."""
        seq1 = MutatedSequence.from_string(self.ref_seq, "0AC|0.95")
        seq2 = MutatedSequence.from_string(self.ref_seq, "100AT|0.90")
        seq3 = MutatedSequence.from_string(self.ref_seq, "200AG|0.85")
        
        mutations_gene = MutationsGene.__new__(MutationsGene)
        mutations_gene.reference_sequence = self.ref_seq
        mutations_gene.generation_dict = {100: [seq1], 200: [seq2, seq3]}
        
        annotated_list = AnnotatedMutatedSequence.from_mutations_gene(
            self.seq_name, mutations_gene
        )
        
        self.assertEqual(len(annotated_list), 2)
        self.assertEqual(len(annotated_list[100]), 1)
        self.assertEqual(len(annotated_list[200]), 2)
        self.assertEqual(annotated_list[100][0].annotations[0].genomic_position, 9001)  # 10000 - 1000 + 1


class TestAnnotatedMutatedSequenceStringRepresentation(unittest.TestCase):
    """Test cases for string representations."""

    def test_str_method(self):
        """Test __str__ method produces readable output."""
        seq_name = "1_TESTGENE_gene:10000-12000_timestamp"
        ref_seq = "A" * 3020
        mutated_seq_str = "0AC500AT|0.95"
        mutated_seq = MutatedSequence.from_string(ref_seq, mutated_seq_str)
        
        annotated = AnnotatedMutatedSequence.from_mutated_sequence(seq_name, mutated_seq)
        
        str_repr = str(annotated)
        self.assertIn("TESTGENE", str_repr)
        self.assertIn("10000", str_repr)
        self.assertIn("12000", str_repr)
        self.assertIn("Fitness: 0.950000", str_repr)
        self.assertIn("Number of mutations: 2", str_repr)

    def test_repr_method(self):
        """Test __repr__ method produces concise output."""
        seq_name = "1_TESTGENE_gene:10000-12000_timestamp"
        ref_seq = "A" * 3020
        mutated_seq_str = "0AC500AT|0.95"
        mutated_seq = MutatedSequence.from_string(ref_seq, mutated_seq_str)
        
        annotated = AnnotatedMutatedSequence.from_mutated_sequence(seq_name, mutated_seq)
        
        repr_str = repr(annotated)
        self.assertIn("AnnotatedMutatedSequence", repr_str)
        self.assertIn("TESTGENE", repr_str)
        self.assertIn("mutations=2", repr_str)
        self.assertIn("fitness=0.950000", repr_str)



if __name__ == '__main__':
    unittest.main(defaultTest="TestAnnotatedMutatedSequenceStringRepresentation")
