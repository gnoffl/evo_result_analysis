"""List natural GOF/LOF genes by fitness delta at a fixed mutation budget.

For every gene of the constrained (natural) GOF and LOF runs
(``*_single_natural_*``), the pareto front entry with exactly ``k`` mutations is
compared against the 0-mutation reference entry, and the mutations it introduced
are reported in genome coordinates. One CSV per budget (k = 5 and k = 7), one
row per gene, GOF and LOF pooled and sorted by ``delta_mut`` descending: the
strongest gains of function end up at the top, the strongest losses at the
bottom.

Coordinate frames
-----------------
The 3020 bp optimized sequence is not contiguous genome (promoter 1-1000,
5'-UTR 1001-1500, an N gap 1501-1520, 3'-UTR 1521-2020, terminator 2021-3020),
so sequence positions cannot be mapped to the genome by an offset. Instead the
mapping is read from the ``GENOMIC_POS`` INFO field of the reextracted VCFs
written by ``evolution/extract_sequences.py``. Those runs are constrained to VCF
positions, so every introduced mutation has a VCF record by construction.

``extract_sequences.py`` complements the bases of minus-strand genes when it
builds the sequence, so its VCF ``REF``/``ALT`` are in the gene-sense frame. The
mutation strings written here are always in the **plus-strand** frame: for
minus-strand genes both bases are complemented back. Every emitted reference
base is checked against the genome FASTA at its genomic position, so a broken
mapping fails loudly instead of producing plausible coordinates.
"""

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd
from pyfaidx import Fasta

GOF_RUN_DIR = Path(
    "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset"
    "/GOF_LOF/GOF/GOF_single_natural_260311_122228_299093"
)
LOF_RUN_DIR = Path(
    "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset"
    "/GOF_LOF/LOF/LOF_single_natural_260311_122613_033117"
)
GOF_VCF_DIR = Path(
    "/home/gernot/Code/PhD_Code/Evolution/data/Arabidopsis_GOF_reextracted_vcfs"
)
LOF_VCF_DIR = Path(
    "/home/gernot/Code/PhD_Code/Evolution/data/Arabidopsis_LOF_reextracted_vcfs"
)
GENOME_PATH = Path(
    "/home/gernot/ARCitect/ARCs/genRE/assays/Gene_Data/dataset/genomes"
    "/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"
)

MUTATION_BUDGETS: Tuple[int, ...] = (5, 7)
OUTPUT_DIR = Path(__file__).parent / "natural_gof_lof_top_deltas"

COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}

# A pyfaidx Fasta, or any chromosome-name to sequence-string mapping (tests).
Genome = Union[Fasta, Mapping[str, str]]


def genome_base(genome: Genome, chromosome: str, genomic_position: int) -> str:
    """Read the plus-strand genome base at a genomic position.

    Args:
        genome: Indexable genome, keyed by chromosome name.
        chromosome: Chromosome name.
        genomic_position: 1-based position on the chromosome.

    Returns:
        The uppercase base at that position.
    """
    return str(genome[chromosome][genomic_position - 1]).upper()


def parse_gene_dir_name(dir_name: str) -> Tuple[str, str, int, int, str]:
    """Split a gene run directory name into its gene identity and location.

    Args:
        dir_name: Gene run directory name, e.g.
            ``1_AT1G01720_gene:267992-269819_260311_122242_380559``.

    Returns:
        Tuple of ``(gene_id, chromosome, gene_start, gene_end, strand)``.
        ``gene_start``/``gene_end`` are the coordinates as annotated (start >
        end for minus-strand genes); ``strand`` is ``"+"`` or ``"-"``.
    """
    chromosome, gene_id, coordinates = dir_name.split("_")[:3]
    start, end = (int(value) for value in coordinates.split(":")[1].split("-"))
    return gene_id, chromosome, start, end, "+" if start < end else "-"


def load_genomic_positions(vcf_path: Path) -> Dict[int, int]:
    """Map 0-based sequence positions to 1-based genomic positions.

    Args:
        vcf_path: Reextracted VCF of one gene, whose ``POS`` column holds
            1-based sequence positions and whose INFO column carries
            ``GENOMIC_POS``.

    Returns:
        Mapping from 0-based sequence position to 1-based genomic position.
    """
    positions = {}
    for line in vcf_path.read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        info = dict(entry.split("=", 1) for entry in fields[7].split(";"))
        positions[int(fields[1]) - 1] = int(info["GENOMIC_POS"])
    return positions


def load_reference_sequence(gene_dir: Path) -> str:
    """Read the full 3020 bp reference sequence of a gene run.

    Args:
        gene_dir: Gene run directory containing ``reference_sequence.fa``.

    Returns:
        The ``reference_sequence_full`` record as an uppercase string.
    """
    fasta = Fasta(str(gene_dir / "reference_sequence.fa"))
    return str(fasta["reference_sequence_full"]).upper()


def find_front_entry(pareto_front: List[list], mutation_count: int) -> Tuple[str, float]:
    """Pick the pareto front entry with an exact mutation count.

    Args:
        pareto_front: Parsed ``pareto_front.json``, a list of
            ``[sequence, fitness, mutation_count]`` entries.
        mutation_count: Number of mutations the wanted entry carries.

    Returns:
        Tuple of ``(sequence, fitness)``.

    Raises:
        ValueError: If no entry, or more than one entry, has that count.
    """
    matches = [entry for entry in pareto_front if entry[2] == mutation_count]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one entry with {mutation_count} mutations, "
            f"found {len(matches)}"
        )
    return matches[0][0], matches[0][1]


def format_mutations(
    reference_sequence: str,
    mutated_sequence: str,
    chromosome: str,
    strand: str,
    genomic_positions: Dict[int, int],
    genome: Genome,
) -> List[str]:
    """Describe the mutations of one sequence in plus-strand genome coordinates.

    Args:
        reference_sequence: The gene's 3020 bp reference sequence.
        mutated_sequence: An equally long optimized sequence.
        chromosome: Chromosome name of the gene.
        strand: ``"+"`` or ``"-"``; bases of minus-strand genes are
            complemented to the plus strand.
        genomic_positions: 0-based sequence position to 1-based genomic
            position, from :func:`load_genomic_positions`.
        genome: Indexable genome, used to verify every reference base.

    Returns:
        One ``"chromosome:genomic_position ref->alt"`` string per mutation, in
        ascending sequence-position order.

    Raises:
        ValueError: If the sequences differ in length, if a mutated position
            has no VCF record, or if the plus-strand reference base disagrees
            with the genome.
    """
    if len(reference_sequence) != len(mutated_sequence):
        raise ValueError("reference and mutated sequence lengths differ")

    mutations = []
    for position, (reference_letter, mutated_letter) in enumerate(
        zip(reference_sequence, mutated_sequence)
    ):
        if reference_letter == mutated_letter:
            continue
        if position not in genomic_positions:
            raise ValueError(
                f"mutated position {position} has no VCF record, so its genomic "
                "position is unknown"
            )
        genomic_position = genomic_positions[position]
        if strand == "-":
            reference_letter = COMPLEMENT[reference_letter]
            mutated_letter = COMPLEMENT[mutated_letter]
        expected = genome_base(genome, chromosome, genomic_position)
        if expected != reference_letter:
            raise ValueError(
                f"reference base mismatch at {chromosome}:{genomic_position}: "
                f"genome has {expected}, sequence implies {reference_letter}"
            )
        mutations.append(
            f"{chromosome}:{genomic_position} {reference_letter}->{mutated_letter}"
        )
    return mutations


def build_gene_row(
    gene_dir: Path,
    group: str,
    vcf_dir: Path,
    mutation_count: int,
    genome: Genome,
) -> Dict[str, object]:
    """Assemble the output row of one gene at one mutation budget.

    Args:
        gene_dir: Gene run directory.
        group: ``"GOF"`` or ``"LOF"``.
        vcf_dir: Directory holding the reextracted VCFs of this run's gene set.
        mutation_count: Mutation budget to report.
        genome: Indexable genome, used to verify every reference base.

    Returns:
        Row dictionary with the gene columns, the fitness columns, and one
        ``mutation<n>`` column per introduced mutation.
    """
    gene_id, chromosome, gene_start, gene_end, strand = parse_gene_dir_name(gene_dir.name)
    reference_sequence = load_reference_sequence(gene_dir)
    with open(gene_dir / "saved_populations" / "pareto_front.json") as handle:
        pareto_front = json.load(handle)

    fitness_reference = find_front_entry(pareto_front, 0)[1]
    mutated_sequence, fitness_mut = find_front_entry(pareto_front, mutation_count)
    vcf_name = "_".join(gene_dir.name.split("_")[:3])
    mutations = format_mutations(
        reference_sequence,
        mutated_sequence,
        chromosome,
        strand,
        load_genomic_positions(vcf_dir / f"{vcf_name}.vcf"),
        genome,
    )
    if len(mutations) != mutation_count:
        raise ValueError(
            f"{gene_dir.name}: pareto entry claims {mutation_count} mutations "
            f"but the sequence differs at {len(mutations)} positions"
        )

    row: Dict[str, object] = {
        "gene_id": gene_id,
        "group": group,
        "chromosome": chromosome,
        "gene_start": gene_start,
        "gene_end": gene_end,
        "strand": strand,
        "fitness_reference": fitness_reference,
        "fitness_mut": fitness_mut,
        "delta_mut": fitness_mut - fitness_reference,
    }
    for index, mutation in enumerate(mutations, start=1):
        row[f"mutation{index}"] = mutation
    return row


def build_table(
    runs: List[Tuple[Path, Path, str]],
    mutation_count: int,
    genome: Genome,
) -> pd.DataFrame:
    """Build the sorted table of all genes at one mutation budget.

    Args:
        runs: ``(run_dir, vcf_dir, group)`` triples to pool.
        mutation_count: Mutation budget to report.
        genome: Indexable genome, used to verify every reference base.

    Returns:
        DataFrame with one row per gene, sorted by ``delta_mut`` descending.
    """
    rows = []
    for run_dir, vcf_dir, group in runs:
        for gene_dir in sorted(run_dir.iterdir()):
            if not gene_dir.is_dir() or not gene_dir.name[0].isdigit():
                continue
            rows.append(build_gene_row(gene_dir, group, vcf_dir, mutation_count, genome))
    return pd.DataFrame(rows).sort_values("delta_mut", ascending=False, ignore_index=True)


def main(output_dir: Optional[Path] = None) -> None:
    """Write one CSV per mutation budget and print the extremes of each.

    Args:
        output_dir: Folder for the CSVs; defaults to :data:`OUTPUT_DIR`.
    """
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    genome = Fasta(str(GENOME_PATH))
    runs = [
        (GOF_RUN_DIR, GOF_VCF_DIR, "GOF"),
        (LOF_RUN_DIR, LOF_VCF_DIR, "LOF"),
    ]
    for mutation_count in MUTATION_BUDGETS:
        table = build_table(runs, mutation_count, genome)
        output_path = output_dir / f"natural_gof_lof_mut{mutation_count}.csv"
        table.to_csv(output_path, index=False)
        print(f"Saved {output_path} ({len(table)} genes)")
        print(table[["gene_id", "group", "delta_mut"]].head(5).to_string(index=False))
        print(table[["gene_id", "group", "delta_mut"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
