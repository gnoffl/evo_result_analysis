"""Find the N. tabacum ortholog of each Arabidopsis FLOR-ID flowering-time gene.

Translates both annotations from scratch into this folder, builds a fresh DIAMOND
database of the N. tabacum proteome, and keeps the single best blastp hit per
flowering gene. Nothing from an earlier build is reused.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

ARC = Path("/home/gernot/ARCitect/ARCs/genRE/assays/Gene_Data/dataset")
ARA_GENOME = ARC / "genomes/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"
ARA_ANNOTATION = ARC / "annotations/Arabidopsis_thaliana.TAIR10.52.gtf"
NTAB_GENOME = ARC / "genomes/nicotiana_tabacum.fa"
NTAB_ANNOTATION = ARC / "annotations/hlx-Nicotiana_tabacum-GCF_000715135.1-4097.gff"

HERE = Path(__file__).parent
ARA_GENES = HERE / "ara_flowering_genes_florid.json"
ARA_PROT_ALL = HERE / "ara_prot_all.faa"
ARA_PROT_FLOWERING = HERE / "ara_flowering_prot.faa"
NTAB_PROT_ALL = HERE / "ntab_prot_all.faa"
NTAB_PROT_PER_GENE = HERE / "ntab_prot_per_gene.faa"
NTAB_DB = HERE / "ntab_prot_per_gene"
HITS_TSV = HERE / "ara_flowering_vs_ntab.tsv"
MAPPING_CSV = HERE / "ara_ntab_flowering_orthologs.csv"

OUTPUT_JSON = HERE / "ntab_flowering_genes.json"

MAX_EVALUE = 1e-10
MIN_IDENTITY = 30.0
THREADS = 12


def conda_run(*command: str) -> None:
    """Run a command in the deepcre_tools environment, which holds gffread and diamond."""
    subprocess.run(["conda", "run", "-n", "deepcre_tools", *command], check=True)


def read_fasta(path: Path) -> Dict[str, str]:
    """Sequences of a FASTA keyed by the first word of the header."""
    sequences: Dict[str, str] = {}
    name = ""
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            sequences[name] = ""
        else:
            sequences[name] += line.strip()
    return sequences


def longest_per_gene(proteins: Dict[str, str], genes: Optional[List[str]] = None) -> Dict[str, str]:
    """Longest protein per gene, keyed by gene id (transcript suffix stripped).

    Args:
        genes: If given, only these gene ids are kept.
    """
    wanted = None if genes is None else set(genes)
    best: Dict[str, str] = {}
    for transcript, sequence in proteins.items():
        gene = transcript.rsplit(".", 1)[0]
        if wanted is not None and gene not in wanted:
            continue
        if len(sequence) > len(best.get(gene, "")):
            best[gene] = sequence
    return best


def write_fasta(sequences: Dict[str, str], path: Path) -> None:
    with open(path, "w") as handle:
        for name, sequence in sequences.items():
            handle.write(f">{name}\n{sequence}\n")


def best_hits(tsv: str) -> Dict[str, str]:
    """Best-scoring N. tabacum gene per Arabidopsis gene, hits below the cutoffs dropped."""
    hits: Dict[str, str] = {}
    for line in tsv.strip().splitlines():
        query, subject, identity, evalue = line.split("\t")[:4]
        if query not in hits and float(identity) >= MIN_IDENTITY and float(evalue) <= MAX_EVALUE:
            hits[query] = subject
    return hits


def build_query_fasta() -> None:
    """Translate TAIR10 and write the flowering-gene proteins DIAMOND queries with."""
    conda_run("gffread", "-y", str(ARA_PROT_ALL), "-g", str(ARA_GENOME), str(ARA_ANNOTATION))
    genes = json.loads(ARA_GENES.read_text())
    proteins = longest_per_gene(read_fasta(ARA_PROT_ALL), genes)
    write_fasta(proteins, ARA_PROT_FLOWERING)
    print(f"{len(proteins)} of {len(genes)} flowering genes have a protein")


def build_database() -> None:
    """Translate the N. tabacum annotation and make a DIAMOND database of it."""
    conda_run("gffread", "-y", str(NTAB_PROT_ALL), "-g", str(NTAB_GENOME), str(NTAB_ANNOTATION))
    proteins = longest_per_gene(read_fasta(NTAB_PROT_ALL))
    write_fasta(proteins, NTAB_PROT_PER_GENE)
    print(f"{len(proteins)} N. tabacum proteins in the database")
    conda_run("diamond", "makedb", "--in", str(NTAB_PROT_PER_GENE), "-d", str(NTAB_DB), "--quiet")


def run_diamond() -> str:
    conda_run(
        "diamond", "blastp", "-q", str(ARA_PROT_FLOWERING), "-d", str(NTAB_DB),
        "-o", str(HITS_TSV), "-f", "6", "qseqid", "sseqid", "pident", "evalue", "bitscore",
        "-e", str(MAX_EVALUE), "--max-target-seqs", "1", "-p", str(THREADS), "--quiet",
    )
    return HITS_TSV.read_text()


def main() -> None:
    build_query_fasta()
    build_database()
    hits = best_hits(run_diamond())

    with open(MAPPING_CSV, "w") as handle:
        handle.write("ara_gene_id,ntab_gene_id\n")
        for ara_gene, ntab_gene in sorted(hits.items()):
            handle.write(f"{ara_gene},{ntab_gene}\n")

    ntab_genes = sorted(set(hits.values()))
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(ntab_genes, indent=2))
    print(f"{len(hits)} Arabidopsis genes -> {len(ntab_genes)} N. tabacum genes")
    print(f"wrote {OUTPUT_JSON} and {MAPPING_CSV}")


if __name__ == "__main__":
    main()
