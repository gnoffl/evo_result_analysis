"""Convert MEME-format motif databases into a single JASPAR file for blamm.

``blamm`` expects Position Frequency Matrices (PFMs, integer counts) in JASPAR
format, while the MEME databases shipped with the MEME suite store
Position Probability Matrices (PPMs, per-position probabilities) together with
the number of observed sites (``nsites``). This module converts the latter into
the former via ``count = round(probability * nsites)`` and concatenates several
databases into one ``.jaspar`` file, alongside a metadata CSV that records
which database each motif came from.

Example:
    python -m analysis.blamm.meme_to_jaspar \\
        --meme-files JASPAR2024_CORE_plants_non-redundant_v2.meme \\
                     ArabidopsisDAPv1.meme \\
        --output-jaspar motifs.jaspar \\
        --output-metadata motif_metadata.csv
"""

import argparse
import os
import re
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence

import pandas as pd

from analysis.utils.io import print_status

#: Nucleotide order used in both MEME probability matrices and JASPAR output.
NUCLEOTIDES: str = "ACGT"

#: Number of sites assumed when a MEME motif does not declare ``nsites=``.
DEFAULT_NSITES: int = 100

_MOTIF_LINE = re.compile(r"^MOTIF\s+(\S+)(?:\s+(\S.*?))?\s*$")
_MATRIX_LINE = re.compile(
    r"^letter-probability matrix:\s*"
    r"alength=\s*(\d+)\s+"
    r"w=\s*(\d+)"
    r"(?:\s+nsites=\s*([0-9.eE+-]+))?"
)


class Motif(NamedTuple):
    """A single motif converted to integer counts.

    Attributes:
        motif_id: Identifier as given on the MEME ``MOTIF`` line.
        motif_name: Human-readable name (falls back to ``motif_id``).
        source_db: Basename of the MEME file the motif was read from.
        nsites: Number of sites the probability matrix was derived from.
        counts: Per-nucleotide count vectors, keyed by ``A``/``C``/``G``/``T``;
            every vector has length equal to the motif width.
    """

    motif_id: str
    motif_name: str
    source_db: str
    nsites: int
    counts: Dict[str, List[int]]

    @property
    def width(self) -> int:
        """Number of positions in the motif."""
        return len(self.counts["A"])


def _parse_probability_rows(
    lines: Sequence[str], start_index: int, width: int
) -> tuple:
    """Read ``width`` rows of four probabilities, skipping blank lines.

    Args:
        lines: All lines of the MEME file.
        start_index: Index of the first line after the matrix header.
        width: Number of matrix rows to read.

    Returns:
        Tuple of ``(rows, next_index)`` where ``rows`` is a list of
        ``width`` lists of four floats and ``next_index`` is the index of the
        first unconsumed line.

    Raises:
        ValueError: If the file ends early or a row does not hold four numbers.
    """
    rows: List[List[float]] = []
    index = start_index
    while len(rows) < width:
        if index >= len(lines):
            raise ValueError(
                f"MEME file ended after {len(rows)} of {width} matrix rows"
            )
        stripped = lines[index].strip()
        index += 1
        if not stripped:
            continue
        values = [float(field) for field in stripped.split()]
        if len(values) != len(NUCLEOTIDES):
            raise ValueError(
                f"Expected {len(NUCLEOTIDES)} probabilities per row, "
                f"got {len(values)}: {stripped!r}"
            )
        rows.append(values)
    return rows, index


def parse_meme_file(meme_path: str) -> List[Motif]:
    """Parse a MEME motif database into integer-count motifs.

    Only ``letter-probability matrix`` blocks are read; other matrix types
    (for example ``log-odds matrix``) that some databases additionally provide
    are ignored.

    Args:
        meme_path: Path to a MEME-format motif file.

    Returns:
        List of :class:`Motif` instances in file order.

    Raises:
        FileNotFoundError: If *meme_path* does not exist.
        ValueError: If a matrix block is malformed or uses an alphabet other
            than the four DNA nucleotides.
    """
    with open(meme_path) as handle:
        lines = handle.read().splitlines()

    source_db = os.path.basename(meme_path)
    motifs: List[Motif] = []
    current_id: Optional[str] = None
    current_name: Optional[str] = None

    index = 0
    while index < len(lines):
        line = lines[index]
        motif_match = _MOTIF_LINE.match(line)
        if motif_match:
            current_id = motif_match.group(1)
            current_name = motif_match.group(2) or current_id
            index += 1
            continue

        matrix_match = _MATRIX_LINE.match(line.strip())
        if not matrix_match:
            index += 1
            continue

        if current_id is None:
            raise ValueError(
                f"{source_db}: probability matrix at line {index + 1} "
                "is not preceded by a MOTIF line"
            )
        alphabet_length = int(matrix_match.group(1))
        if alphabet_length != len(NUCLEOTIDES):
            raise ValueError(
                f"{source_db}: motif {current_id} has alphabet length "
                f"{alphabet_length}, expected {len(NUCLEOTIDES)}"
            )
        width = int(matrix_match.group(2))
        raw_nsites = matrix_match.group(3)
        if raw_nsites is None:
            nsites = DEFAULT_NSITES
            print_status(
                f"{source_db}: motif {current_id} has no nsites, "
                f"assuming {DEFAULT_NSITES}",
                "WARNING",
            )
        else:
            nsites = max(1, int(round(float(raw_nsites))))

        rows, index = _parse_probability_rows(lines, index + 1, width)
        counts = {
            nucleotide: [
                int(round(row[nucleotide_index] * nsites)) for row in rows
            ]
            for nucleotide_index, nucleotide in enumerate(NUCLEOTIDES)
        }
        motifs.append(
            Motif(
                motif_id=current_id,
                motif_name=current_name or current_id,
                source_db=source_db,
                nsites=nsites,
                counts=counts,
            )
        )
        current_id = None
        current_name = None

    return motifs


def collect_motifs(meme_paths: Sequence[str]) -> List[Motif]:
    """Parse several MEME files and verify that motif identifiers are unique.

    Args:
        meme_paths: Paths to MEME motif databases, in the order they should be
            written to the combined JASPAR file.

    Returns:
        Concatenated list of motifs across all databases.

    Raises:
        ValueError: If the same motif identifier occurs in more than one
            database, since blamm reports only the identifier and the source
            would become ambiguous.
    """
    motifs: List[Motif] = []
    seen: Dict[str, str] = {}
    for meme_path in meme_paths:
        parsed = parse_meme_file(meme_path)
        print_status(f"Parsed {len(parsed)} motifs from {meme_path}")
        for motif in parsed:
            if motif.motif_id in seen:
                raise ValueError(
                    f"Duplicate motif id {motif.motif_id!r} in "
                    f"{motif.source_db} and {seen[motif.motif_id]}"
                )
            seen[motif.motif_id] = motif.source_db
        motifs.extend(parsed)
    return motifs


def write_jaspar(motifs: Sequence[Motif], output_path: str) -> None:
    """Write motifs to a single JASPAR-format file readable by blamm.

    Args:
        motifs: Motifs to write.
        output_path: Destination ``.jaspar`` file. Parent directories are
            created if missing.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as handle:
        for motif in motifs:
            handle.write(f">{motif.motif_id}\t{motif.motif_name}\n")
            for nucleotide in NUCLEOTIDES:
                values = " ".join(
                    f"{count:6d}" for count in motif.counts[nucleotide]
                )
                handle.write(f"{nucleotide}  [ {values} ]\n")


def write_metadata(motifs: Sequence[Motif], output_path: str) -> pd.DataFrame:
    """Write a CSV describing every motif in the combined JASPAR file.

    Args:
        motifs: Motifs to describe.
        output_path: Destination CSV path.

    Returns:
        The metadata table that was written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    metadata = pd.DataFrame(
        [
            {
                "motif_id": motif.motif_id,
                "motif_name": motif.motif_name,
                "source_db": motif.source_db,
                "width": motif.width,
                "nsites": motif.nsites,
            }
            for motif in motifs
        ]
    )
    metadata.to_csv(output_path, index=False)
    return metadata


def convert_meme_databases(
    meme_paths: Sequence[str], output_jaspar: str, output_metadata: str
) -> pd.DataFrame:
    """Convert MEME databases into one JASPAR file plus a metadata table.

    Args:
        meme_paths: Paths to MEME motif databases.
        output_jaspar: Destination ``.jaspar`` file for blamm.
        output_metadata: Destination CSV describing each motif.

    Returns:
        The metadata table that was written.
    """
    motifs = collect_motifs(meme_paths)
    write_jaspar(motifs, output_jaspar)
    metadata = write_metadata(motifs, output_metadata)
    print_status(
        f"Wrote {len(motifs)} motifs to {output_jaspar}", "SUCCESS"
    )
    return metadata


def parse_arguments(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse and validate command line arguments.

    Args:
        args: Argument list to parse; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert MEME motif databases into a single JASPAR file for blamm."
        )
    )
    parser.add_argument(
        "--meme-files",
        "-m",
        nargs="+",
        required=True,
        help="MEME-format motif databases to convert and concatenate.",
    )
    parser.add_argument(
        "--output-jaspar",
        "-o",
        required=True,
        help="Destination .jaspar file consumed by blamm.",
    )
    parser.add_argument(
        "--output-metadata",
        "-t",
        required=True,
        help="Destination CSV mapping motif ids to names and source database.",
    )
    parsed = parser.parse_args(args)

    for meme_path in parsed.meme_files:
        if not os.path.isfile(meme_path):
            parser.error(f"MEME file does not exist: {meme_path}")
    if not parsed.output_jaspar.endswith(".jaspar"):
        parser.error("--output-jaspar must end in '.jaspar' (required by blamm)")
    return parsed


def main(args: Optional[Sequence[str]] = None) -> int:
    """Entry point for command line execution.

    Args:
        args: Argument list to parse; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    parsed = parse_arguments(args)
    try:
        convert_meme_databases(
            parsed.meme_files, parsed.output_jaspar, parsed.output_metadata
        )
    except (ValueError, OSError) as error:
        print_status(f"Conversion failed: {error}", "ERROR")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
