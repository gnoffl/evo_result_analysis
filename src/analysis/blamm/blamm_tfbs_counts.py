"""Count TFBS motif occurrences in reference vs. optimized pareto-front sequences.

For every gene of an evolutionary run this script pairs the unmodified
reference sequence with one optimized sequence taken from the pareto front,
scans both with ``blamm`` against a JASPAR motif database, and reports how
often each motif occurs in each sequence.

The optimized sequence is selected by mutation count. With ``MAX`` the entry
with the highest mutation count is used; with an integer the entry with exactly
that mutation count is used and genes lacking it are skipped with a warning.
A pareto front may contain several entries sharing a mutation count; because
all of them are non-dominated they share the same fitness, so the first
occurrence is taken.

Reference and optimized sequences of all genes are scanned in a single blamm
invocation using one background model, so both sides of the comparison are
subject to identical PWM score thresholds.

Example:
    python -m analysis.blamm.blamm_tfbs_counts \\
        --run-folder /path/to/GOF_single_mutation_251009_121226_109368 \\
        --motifs motifs.jaspar \\
        --motif-metadata motif_metadata.csv \\
        --output /path/to/output \\
        --mutation-count MAX
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.blamm.blamm_significance import (
    SIGNIFICANCE_FILE,
    motif_significance,
    summarise_significance,
)
from analysis.motives.deepcis_scanner import (
    GeneRunData,
    ParetoEntry,
    find_gene_folders,
    get_full_sequence,
)
from analysis.utils.io import print_section_header, print_status, print_subsection

#: Sentinel selecting the pareto entry with the highest mutation count.
MAX_MUTATIONS: str = "MAX"

#: Default p-value threshold handed to ``blamm scan -pt``.
DEFAULT_P_VALUE: float = 1e-4

#: Group identifier in the blamm manifest. A single group means a single
#: background model shared by reference and optimized sequences.
BLAMM_GROUP_ID: str = "evo"

#: Separator between gene name and variant in FASTA sequence identifiers.
SEQUENCE_ID_SEPARATOR: str = "__"

VARIANT_REFERENCE: str = "reference"
VARIANT_MUTATED: str = "mutated"

#: Pseudocount added to both sides of the log2 fold change, needed because
#: motifs regularly appear or disappear entirely between the two sequences.
LOG2_PSEUDOCOUNT: int = 1

#: Either the sentinel :data:`MAX_MUTATIONS` or an exact mutation count.
MutationCount = Union[int, str]

#: Columns of ``selected_entries.csv``, documenting the chosen pareto entry.
SELECTION_COLUMNS: List[str] = [
    "gene",
    "requested_mutation_count",
    "selected_mutation_count",
    "fitness",
    "pareto_index",
    "sequence_length",
]

#: Columns of ``skipped_genes.csv``. Declared explicitly so the file carries a
#: header even when no gene was skipped.
SKIPPED_COLUMNS: List[str] = ["gene", "reason"]

#: Column names of the GTF-like ``occurrences.txt`` written by ``blamm scan``.
OCCURRENCE_COLUMNS: List[str] = [
    "sequence_id",
    "source",
    "motif_id",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attribute",
]


def select_pareto_entry(
    pareto_front: Sequence[ParetoEntry], mutation_count: MutationCount
) -> Tuple[int, ParetoEntry]:
    """Select one pareto front entry by mutation count.

    Args:
        pareto_front: Entries as ``(mutable_sequence, fitness, mutation_count)``.
        mutation_count: Either :data:`MAX_MUTATIONS` for the entry with the most
            mutations, or an integer requesting that exact mutation count.

    Returns:
        Tuple of ``(index, entry)`` for the first entry matching the request.

    Raises:
        ValueError: If *pareto_front* is empty, or no entry has the requested
            mutation count.
    """
    if not pareto_front:
        raise ValueError("Pareto front is empty")

    counts = [int(round(entry[2])) for entry in pareto_front]
    if isinstance(mutation_count, str):
        if mutation_count.upper() != MAX_MUTATIONS:
            raise ValueError(
                f"mutation_count must be an integer or {MAX_MUTATIONS!r}, "
                f"got {mutation_count!r}"
            )
        target = max(counts)
    else:
        target = int(mutation_count)
    for index, count in enumerate(counts):
        if count == target:
            return index, pareto_front[index]
    raise ValueError(f"No pareto entry with mutation count {target}")


def build_sequence_records(
    gene_folders: Sequence[str], mutation_count: MutationCount
) -> Tuple[List[Tuple[str, str]], pd.DataFrame, pd.DataFrame]:
    """Load every gene and build the reference/optimized sequence pairs.

    Args:
        gene_folders: Absolute paths to gene folders of one run.
        mutation_count: :data:`MAX_MUTATIONS` or an exact mutation count.

    Returns:
        Tuple of ``(records, selections, skipped)`` where ``records`` is a list
        of ``(sequence_id, sequence)`` pairs ready to be written to FASTA,
        ``selections`` documents the pareto entry chosen per gene, and
        ``skipped`` lists genes that were left out together with the reason.
    """
    records: List[Tuple[str, str]] = []
    selections: List[Dict[str, object]] = []
    skipped: List[Dict[str, str]] = []

    for gene_folder in tqdm(gene_folders, desc="Loading genes"):
        gene_name = os.path.basename(gene_folder.rstrip(os.sep))
        try:
            gene_data = GeneRunData.load_gene_run_data(gene_folder)
            index, entry = select_pareto_entry(
                gene_data.pareto_front, mutation_count
            )
        except (ValueError, KeyError, OSError) as error:
            print_status(f"Skipping {gene_name}: {error}", "WARNING")
            skipped.append({"gene": gene_name, "reason": str(error)})
            continue

        mutable_sequence, fitness, selected_count = entry
        mutated_full = get_full_sequence(
            mutable_sequence,
            gene_data.reference_sequence_full,
            gene_data.mutation_start,
            gene_data.mutation_end,
        )
        records.append(
            (
                f"{gene_name}{SEQUENCE_ID_SEPARATOR}{VARIANT_REFERENCE}",
                gene_data.reference_sequence_full,
            )
        )
        records.append(
            (
                f"{gene_name}{SEQUENCE_ID_SEPARATOR}{VARIANT_MUTATED}",
                mutated_full,
            )
        )
        selections.append(
            {
                "gene": gene_name,
                "requested_mutation_count": str(mutation_count),
                "selected_mutation_count": int(round(selected_count)),
                "fitness": float(fitness),
                "pareto_index": index,
                "sequence_length": len(mutated_full),
            }
        )

    selections_table = pd.DataFrame(selections, columns=pd.Index(SELECTION_COLUMNS))
    skipped_table = pd.DataFrame(skipped, columns=pd.Index(SKIPPED_COLUMNS))
    return records, selections_table, skipped_table


def write_fasta(records: Sequence[Tuple[str, str]], fasta_path: str) -> None:
    """Write sequence records to a FASTA file.

    Args:
        records: ``(sequence_id, sequence)`` pairs. Identifiers must be unique
            and free of whitespace, since blamm truncates descriptors at the
            first whitespace character.
        fasta_path: Destination FASTA path.

    Raises:
        ValueError: If identifiers are not unique or contain whitespace.
    """
    identifiers = [record[0] for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("FASTA sequence identifiers must be unique")
    for identifier in identifiers:
        if any(character.isspace() for character in identifier):
            raise ValueError(
                f"FASTA identifier contains whitespace: {identifier!r}"
            )

    os.makedirs(os.path.dirname(os.path.abspath(fasta_path)), exist_ok=True)
    with open(fasta_path, "w") as handle:
        for identifier, sequence in records:
            handle.write(f">{identifier}\n{sequence}\n")


def write_manifest(fasta_path: str, manifest_path: str) -> None:
    """Write the blamm sequence manifest pointing at a single FASTA file.

    Args:
        fasta_path: Path to the FASTA file to register.
        manifest_path: Destination manifest (``.mf``) path.
    """
    with open(manifest_path, "w") as handle:
        handle.write(f"{BLAMM_GROUP_ID}\t{os.path.abspath(fasta_path)}\n")


def run_blamm(
    blamm_executable: str,
    motifs_path: str,
    manifest_path: str,
    work_dir: str,
    p_value: float,
    empirical_histograms: bool,
) -> str:
    """Run the blamm ``dict``, ``hist`` and ``scan`` steps.

    Args:
        blamm_executable: Path to the blamm binary.
        motifs_path: Path to the JASPAR motif file.
        manifest_path: Path to the sequence manifest.
        work_dir: Directory used as working directory; blamm writes its
            dictionary, histograms and thresholds there.
        p_value: Threshold passed to ``blamm scan -pt``.
        empirical_histograms: If True, derive score histograms from the input
            sequences (``hist -e``) instead of the theoretical background model.

    Returns:
        Path to the written ``occurrences.txt``.

    Raises:
        subprocess.CalledProcessError: If any blamm step fails.
    """
    motifs_path = os.path.abspath(motifs_path)
    manifest_path = os.path.abspath(manifest_path)
    occurrences_path = os.path.join(work_dir, "occurrences.txt")

    commands = [
        [blamm_executable, "dict", manifest_path],
        [blamm_executable, "hist"]
        + (["-e"] if empirical_histograms else [])
        + [motifs_path, manifest_path],
        [
            blamm_executable,
            "scan",
            "-rc",
            "-pt",
            repr(p_value),
            "-o",
            occurrences_path,
            motifs_path,
            manifest_path,
        ],
    ]
    for command in commands:
        print_status(f"Running: {' '.join(command)}")
        subprocess.run(command, cwd=work_dir, check=True)
    return occurrences_path


def parse_occurrences(occurrences_path: str) -> pd.DataFrame:
    """Read the GTF-like occurrence table written by ``blamm scan``.

    Args:
        occurrences_path: Path to ``occurrences.txt``.

    Returns:
        DataFrame with columns ``gene``, ``variant``, ``motif_id``, ``start``,
        ``end``, ``score`` and ``strand``. Empty input yields an empty frame
        with the same columns.

    Raises:
        ValueError: If a sequence identifier does not carry a known variant
            suffix.
    """
    columns = ["gene", "variant", "motif_id", "start", "end", "score", "strand"]
    if os.path.getsize(occurrences_path) == 0:
        return pd.DataFrame(columns=pd.Index(columns))

    occurrences = pd.read_csv(
        occurrences_path, sep="\t", header=None, names=OCCURRENCE_COLUMNS
    )
    split = occurrences["sequence_id"].str.rsplit(
        SEQUENCE_ID_SEPARATOR, n=1, expand=True
    )
    occurrences["gene"] = split[0]
    occurrences["variant"] = split[1]
    unknown = set(occurrences["variant"].unique()) - {
        VARIANT_REFERENCE,
        VARIANT_MUTATED,
    }
    if unknown:
        raise ValueError(f"Unexpected variant suffixes in occurrences: {unknown}")
    return occurrences.reindex(columns=columns)


def count_motif_hits(
    occurrences: pd.DataFrame, motif_metadata: pd.DataFrame
) -> pd.DataFrame:
    """Count occurrences per gene and motif for both sequence variants.

    Rows are emitted only for ``(gene, motif)`` pairs with at least one
    occurrence in either variant; all other pairs have counts of zero in both
    variants and carry no information.

    Args:
        occurrences: Parsed occurrence table from :func:`parse_occurrences`.
        motif_metadata: Table with ``motif_id``, ``motif_name`` and
            ``source_db`` columns.

    Returns:
        DataFrame with one row per ``(gene, motif_id)`` and the columns
        ``gene``, ``motif_id``, ``motif_name``, ``source_db``,
        ``count_reference``, ``count_mutated``, ``diff`` and ``log2_fold``.
    """
    counts = (
        occurrences.groupby(["gene", "motif_id", "variant"])
        .size()
        .unstack("variant", fill_value=0)
        .reset_index()
    )
    for variant in (VARIANT_REFERENCE, VARIANT_MUTATED):
        if variant not in counts.columns:
            counts[variant] = 0
    counts = counts.rename(
        columns={
            VARIANT_REFERENCE: "count_reference",
            VARIANT_MUTATED: "count_mutated",
        }
    )
    counts = counts.merge(
        motif_metadata[["motif_id", "motif_name", "source_db"]],
        on="motif_id",
        how="left",
    )
    counts["diff"] = counts["count_mutated"] - counts["count_reference"]
    counts["log2_fold"] = np.log2(
        (counts["count_mutated"] + LOG2_PSEUDOCOUNT)
        / (counts["count_reference"] + LOG2_PSEUDOCOUNT)
    )
    ordered_columns = [
        "gene",
        "motif_id",
        "motif_name",
        "source_db",
        "count_reference",
        "count_mutated",
        "diff",
        "log2_fold",
    ]
    return (
        counts.reindex(columns=ordered_columns)
        .sort_values(by=["gene", "motif_id"])
        .reset_index(drop=True)
    )


def aggregate_over_genes(
    per_gene_counts: pd.DataFrame, n_genes_analysed: int
) -> pd.DataFrame:
    """Aggregate per-gene motif counts over all genes of a run.

    Args:
        per_gene_counts: Output of :func:`count_motif_hits`.
        n_genes_analysed: Number of genes that entered the scan, used as the
            denominator of ``diff_per_gene``. Genes without any occurrence of a
            given motif count towards this denominator.

    Returns:
        DataFrame with one row per motif and the columns ``motif_id``,
        ``motif_name``, ``source_db``, ``total_reference``, ``total_mutated``,
        ``total_diff``, ``n_genes_analysed``, ``n_genes_with_hit``,
        ``diff_per_gene`` and ``log2_fold``, sorted by ``total_diff``.

    Raises:
        ValueError: If *n_genes_analysed* is not positive.
    """
    if n_genes_analysed <= 0:
        raise ValueError("n_genes_analysed must be positive")

    aggregated = (
        per_gene_counts.groupby(
            ["motif_id", "motif_name", "source_db"], dropna=False
        )
        .agg(
            total_reference=("count_reference", "sum"),
            total_mutated=("count_mutated", "sum"),
            n_genes_with_hit=("gene", "nunique"),
        )
        .reset_index()
    )
    aggregated["total_diff"] = (
        aggregated["total_mutated"] - aggregated["total_reference"]
    )
    aggregated["n_genes_analysed"] = n_genes_analysed
    aggregated["diff_per_gene"] = aggregated["total_diff"] / n_genes_analysed
    aggregated["log2_fold"] = np.log2(
        (aggregated["total_mutated"] + LOG2_PSEUDOCOUNT)
        / (aggregated["total_reference"] + LOG2_PSEUDOCOUNT)
    )
    ordered_columns = [
        "motif_id",
        "motif_name",
        "source_db",
        "total_reference",
        "total_mutated",
        "total_diff",
        "n_genes_analysed",
        "n_genes_with_hit",
        "diff_per_gene",
        "log2_fold",
    ]
    return (
        aggregated.reindex(columns=ordered_columns)
        .sort_values(by="total_diff", ascending=False)
        .reset_index(drop=True)
    )


def _file_checksum(path: str) -> str:
    """Return the SHA256 checksum of a file.

    Args:
        path: File to hash.

    Returns:
        Hexadecimal SHA256 digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: Sequence[str]) -> Optional[str]:
    """Run a command and return its stripped output, or None on failure.

    Args:
        command: Command and arguments to execute.

    Returns:
        Combined first line of stdout, or None if the command cannot be run.
    """
    try:
        completed = subprocess.run(
            list(command), capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output.splitlines()[0] if output else None


def collect_provenance(
    run_folder: str,
    motifs_path: str,
    motif_metadata_path: str,
    mutation_count: MutationCount,
    p_value: float,
    empirical_histograms: bool,
    blamm_executable: str,
    n_genes_found: int,
    n_genes_analysed: int,
    n_genes_skipped: int,
) -> Dict[str, object]:
    """Assemble a record of every input that determined the result.

    Args:
        run_folder: Evolutionary run that was analysed.
        motifs_path: JASPAR motif file used for scanning.
        motif_metadata_path: CSV describing the motifs.
        mutation_count: Requested mutation count or :data:`MAX_MUTATIONS`.
        p_value: Threshold passed to ``blamm scan``.
        empirical_histograms: Whether ``hist -e`` was used.
        blamm_executable: Path to the blamm binary.
        n_genes_found: Number of gene folders discovered in the run.
        n_genes_analysed: Number of genes that entered the scan.
        n_genes_skipped: Number of genes left out.

    Returns:
        JSON-serialisable dictionary describing the analysis.
    """
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_folder": os.path.abspath(run_folder),
        "mutation_count": str(mutation_count),
        "p_value": p_value,
        "histograms": "empirical" if empirical_histograms else "theoretical",
        "reverse_complement": True,
        "blamm_group_id": BLAMM_GROUP_ID,
        "motifs_path": os.path.abspath(motifs_path),
        "motifs_sha256": _file_checksum(motifs_path),
        "motif_metadata_path": os.path.abspath(motif_metadata_path),
        "motif_metadata_sha256": _file_checksum(motif_metadata_path),
        "blamm_executable": os.path.abspath(blamm_executable),
        "blamm_version": _command_output([blamm_executable, "-v"]),
        "analysis_git_commit": _command_output(
            [
                "git",
                "-C",
                os.path.dirname(os.path.abspath(__file__)),
                "rev-parse",
                "HEAD",
            ]
        ),
        "python_version": sys.version.split()[0],
        "n_genes_found": n_genes_found,
        "n_genes_analysed": n_genes_analysed,
        "n_genes_skipped": n_genes_skipped,
    }


def analyse_run(
    run_folder: str,
    motifs_path: str,
    motif_metadata_path: str,
    output_folder: str,
    mutation_count: MutationCount = MAX_MUTATIONS,
    p_value: float = DEFAULT_P_VALUE,
    empirical_histograms: bool = False,
    blamm_executable: str = "blamm",
) -> pd.DataFrame:
    """Run the full motif-count analysis for one evolutionary run.

    Args:
        run_folder: Root directory of the evolutionary run.
        motifs_path: JASPAR motif file for blamm.
        motif_metadata_path: CSV with ``motif_id``, ``motif_name``, ``source_db``.
        output_folder: Directory receiving all outputs.
        mutation_count: :data:`MAX_MUTATIONS` or an exact mutation count.
        p_value: Threshold passed to ``blamm scan -pt``.
        empirical_histograms: Whether to use ``hist -e``.
        blamm_executable: Path to the blamm binary.

    Returns:
        The aggregated per-motif table.

    Raises:
        ValueError: If no gene of the run could be analysed.
    """
    print_section_header("BLAMM TFBS COUNT ANALYSIS")
    work_dir = os.path.join(output_folder, "blamm_work")
    os.makedirs(work_dir, exist_ok=True)

    print_subsection("Selecting sequences")
    gene_folders = find_gene_folders(run_folder)
    records, selections, skipped = build_sequence_records(
        gene_folders, mutation_count
    )
    if selections.empty:
        raise ValueError(
            f"No gene in {run_folder} has mutation count {mutation_count}"
        )

    fasta_path = os.path.join(work_dir, "sequences.fa")
    manifest_path = os.path.join(work_dir, "sequences.mf")
    write_fasta(records, fasta_path)
    write_manifest(fasta_path, manifest_path)
    print_status(
        f"Wrote {len(records)} sequences for {len(selections)} genes "
        f"({len(skipped)} skipped)"
    )

    print_subsection("Scanning with blamm")
    occurrences_path = run_blamm(
        blamm_executable,
        motifs_path,
        manifest_path,
        work_dir,
        p_value,
        empirical_histograms,
    )

    print_subsection("Counting occurrences")
    motif_metadata = pd.read_csv(motif_metadata_path)
    occurrences = parse_occurrences(occurrences_path)
    per_gene_counts = count_motif_hits(occurrences, motif_metadata)
    aggregated = aggregate_over_genes(per_gene_counts, len(selections))

    print_subsection("Testing significance")
    significance = motif_significance(per_gene_counts, selections["gene"].tolist())
    significance.to_csv(
        os.path.join(output_folder, SIGNIFICANCE_FILE), index=False
    )
    print_status(summarise_significance(significance))

    per_gene_counts.to_csv(
        os.path.join(output_folder, "motif_counts_per_gene.csv"), index=False
    )
    aggregated.to_csv(
        os.path.join(output_folder, "motif_counts_aggregate.csv"), index=False
    )
    selections.to_csv(
        os.path.join(output_folder, "selected_entries.csv"), index=False
    )
    skipped.to_csv(
        os.path.join(output_folder, "skipped_genes.csv"), index=False
    )
    provenance = collect_provenance(
        run_folder=run_folder,
        motifs_path=motifs_path,
        motif_metadata_path=motif_metadata_path,
        mutation_count=mutation_count,
        p_value=p_value,
        empirical_histograms=empirical_histograms,
        blamm_executable=blamm_executable,
        n_genes_found=len(gene_folders),
        n_genes_analysed=len(selections),
        n_genes_skipped=len(skipped),
    )
    with open(os.path.join(output_folder, "run_parameters.json"), "w") as handle:
        json.dump(provenance, handle, indent=2)

    print_subsection("Summary")
    print_status(f"Genes found:    {len(gene_folders)}")
    print_status(f"Genes analysed: {len(selections)}", "SUCCESS")
    if not skipped.empty:
        print_status(
            f"Genes skipped:  {len(skipped)} "
            f"(see {os.path.join(output_folder, 'skipped_genes.csv')})",
            "WARNING",
        )
        for gene in skipped["gene"]:
            print_status(f"  skipped: {gene}", "WARNING")
    print_status(f"Motifs with occurrences: {len(aggregated)}", "SUCCESS")
    print_status(f"Results written to {output_folder}", "SUCCESS")
    return aggregated


def _parse_mutation_count(value: str) -> MutationCount:
    """Interpret the ``--mutation-count`` argument.

    Args:
        value: Raw command line value.

    Returns:
        :data:`MAX_MUTATIONS` or a non-negative integer.

    Raises:
        argparse.ArgumentTypeError: If the value is neither.
    """
    if value.upper() == MAX_MUTATIONS:
        return MAX_MUTATIONS
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--mutation-count must be an integer or '{MAX_MUTATIONS}', got {value!r}"
        )
    if count < 0:
        raise argparse.ArgumentTypeError("--mutation-count must not be negative")
    return count


def parse_arguments(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse and validate command line arguments.

    Args:
        args: Argument list to parse; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Count TFBS motif occurrences in reference and optimized "
            "pareto-front sequences using blamm."
        )
    )
    parser.add_argument(
        "--run-folder", "-r", required=True, help="Root folder of one evolutionary run."
    )
    parser.add_argument(
        "--motifs", "-m", required=True, help="JASPAR motif file for blamm."
    )
    parser.add_argument(
        "--motif-metadata",
        "-t",
        required=True,
        help="CSV mapping motif ids to names and source database.",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Directory receiving all outputs."
    )
    parser.add_argument(
        "--mutation-count",
        "-mc",
        type=_parse_mutation_count,
        default=MAX_MUTATIONS,
        help=(
            f"Mutation count of the optimized sequence, or '{MAX_MUTATIONS}' "
            "for the most heavily mutated entry (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--p-value",
        "-p",
        type=float,
        default=DEFAULT_P_VALUE,
        help="p-value threshold for blamm scan (default: %(default)s).",
    )
    parser.add_argument(
        "--empirical-histograms",
        action="store_true",
        help=(
            "Derive PWM score histograms from the input sequences instead of "
            "the theoretical background model."
        ),
    )
    parser.add_argument(
        "--blamm",
        default=shutil.which("blamm") or "blamm",
        help="Path to the blamm executable (default: %(default)s).",
    )
    parsed = parser.parse_args(args)

    if not os.path.isdir(parsed.run_folder):
        parser.error(f"Run folder does not exist: {parsed.run_folder}")
    if not os.path.isfile(parsed.motifs):
        parser.error(f"Motif file does not exist: {parsed.motifs}")
    if not parsed.motifs.endswith(".jaspar"):
        parser.error("--motifs must end in '.jaspar' (required by blamm)")
    if not os.path.isfile(parsed.motif_metadata):
        parser.error(f"Motif metadata does not exist: {parsed.motif_metadata}")
    if not 0.0 < parsed.p_value < 1.0:
        parser.error("--p-value must lie strictly between 0 and 1")
    if shutil.which(parsed.blamm) is None and not os.path.isfile(parsed.blamm):
        parser.error(f"blamm executable not found: {parsed.blamm}")
    return parsed


def main(args: Optional[Sequence[str]] = None) -> int:
    """Entry point for command line execution.

    Args:
        args: Argument list to parse; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    parsed = parse_arguments(args)
    os.makedirs(parsed.output, exist_ok=True)
    try:
        analyse_run(
            run_folder=parsed.run_folder,
            motifs_path=parsed.motifs,
            motif_metadata_path=parsed.motif_metadata,
            output_folder=parsed.output,
            mutation_count=parsed.mutation_count,
            p_value=parsed.p_value,
            empirical_histograms=parsed.empirical_histograms,
            blamm_executable=parsed.blamm,
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print_status(f"Analysis failed: {error}", "ERROR")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
