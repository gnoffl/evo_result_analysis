"""deepCIS sliding-window predictions."""

from dataclasses import dataclass
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.utils.io import print_status
from evolution.sequences import one_hot_encode

# Number of TF families in the deepCIS output layer
N_TF_FAMILIES: int = 46

DEFAULT_WINDOW_SIZE: int = 250
DEFAULT_STEP_SIZE: int = 50
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_DELTA_THRESHOLD: float = 0.1

DEFAULT_EXTRAGENIC: int = 1000
DEFAULT_INTRAGENIC: int = 500
DEFAULT_CENTRAL_PADDING: int = 20

# Placeholder for real TF family names
TF_FAMILY_NAMES: List[str] = [f"TF_{i}" for i in range(N_TF_FAMILIES)]

if len(TF_FAMILY_NAMES) != N_TF_FAMILIES:
    raise ValueError(
        f"TF_FAMILY_NAMES must contain exactly {N_TF_FAMILIES} names, "
        f"got {len(TF_FAMILY_NAMES)}"
    )

ParetoEntry = Tuple[str, float, int]

@dataclass
class GeneRunData:
    """All data for one gene folder produced by the evolutionary algorithm.

    Attributes:
        gene_name: Basename of the gene folder (used as identifier throughout).
        gene_folder: Absolute path to the gene folder.
        pareto_front: List of ``(mutable_sequence, fitness, mutation_count)``
            tuples from ``saved_populations/pareto_front.json``.
        reference_sequence_full: Full reference DNA sequence read from
            ``reference_sequence.fa`` under the key ``reference_sequence_full``.
        mutation_start: 0-based start index of the mutable region in the full
            sequence (read from ``parameters.json``).
        mutation_end: 0-based exclusive end index of the mutable region.
    """

    gene_name: str
    gene_folder: str
    pareto_front: List[ParetoEntry]
    reference_sequence_full: str
    mutation_start: int
    mutation_end: int

    @classmethod
    def load_gene_run_data(cls, gene_folder: str) -> "GeneRunData":
        """Convenience loader that assembles a :class:`GeneRunData` from a folder.

        Args:
            gene_folder: Path to one gene folder produced by the evolutionary run.

        Returns:
            Populated :class:`GeneRunData` instance.
        """
        gene_name = os.path.basename(gene_folder.rstrip(os.sep))
        pareto_front = cls.load_pareto_front(gene_folder)
        params = cls.load_gene_params(gene_folder)
        mutation_start = int(params["mutation_start"])
        mutation_end = int(params["mutation_end"])
        reference_sequence_full = cls.load_reference_sequence_full(gene_folder)
        return cls(
            gene_name=gene_name,
            gene_folder=gene_folder,
            pareto_front=pareto_front,
            reference_sequence_full=reference_sequence_full,
            mutation_start=mutation_start,
            mutation_end=mutation_end,
        )

    @staticmethod
    def load_pareto_front(gene_folder: str) -> List[ParetoEntry]:
        """Load ``saved_populations/pareto_front.json`` for a gene folder.

        Returns:
            List of ``(mutable_sequence, fitness, mutation_count)`` tuples.

        Raises:
            FileNotFoundError: If the pareto front file does not exist.
        """
        path = os.path.join(gene_folder, "saved_populations", "pareto_front.json")
        with open(path) as fh:
            raw = json.load(fh)
        return [(str(entry[0]), float(entry[1]), int(entry[2])) for entry in raw]

    @staticmethod
    def load_gene_params(gene_folder: str) -> dict:
        """Load ``parameters.json`` from a gene folder.

        Returns:
            Parsed dictionary from the JSON file.

        Raises:
            FileNotFoundError: If ``parameters.json`` does not exist.
        """
        path = os.path.join(gene_folder, "parameters.json")
        with open(path) as fh:
            return json.load(fh)

    @staticmethod
    def load_reference_sequence_full(gene_folder: str) -> str:
        """Read the full reference sequence from ``reference_sequence.fa``.

        Expects a FASTA record named ``reference_sequence_full`` in the file.

        Args:
            gene_folder: Path to the gene folder.

        Returns:
            DNA string for the full reference sequence.

        Raises:
            ImportError: If ``pyfaidx`` is not installed.
            KeyError: If the expected FASTA record is absent.
        """
        try:
            from pyfaidx import Fasta  # type: ignore
        except ImportError as exc:
            raise ImportError("pyfaidx is required to read reference sequences.") from exc

        fa_path = os.path.join(gene_folder, "reference_sequence.fa")
        fasta = Fasta(fa_path)
        if "reference_sequence_full" not in fasta:
            key = list(fasta.keys())[0]
            return str(fasta[key])
        return str(fasta["reference_sequence_full"])

def find_gene_folders(run_folder: str) -> List[str]:
    """Discover all valid gene folders inside a run folder.

    A subfolder is considered valid when it contains *both*:
    - ``parameters.json``
    - ``saved_populations/pareto_front.json``

    Args:
        run_folder: Root directory of an evolutionary algorithm run.

    Returns:
        Sorted list of absolute paths to valid gene folders.
    """
    valid: List[str] = []
    for entry in sorted(os.listdir(run_folder)):
        folder = os.path.join(run_folder, entry)
        if not os.path.isdir(folder):
            continue
        has_params = os.path.isfile(os.path.join(folder, "parameters.json"))
        has_pareto = os.path.isfile(
            os.path.join(folder, "saved_populations", "pareto_front.json")
        )
        if has_params and has_pareto:
            valid.append(folder)
    print_status(f"Found {len(valid)} gene folders in {run_folder}")
    return valid

def get_full_sequence(
    mutable_sequence: str,
    reference_sequence_full: str,
    mutation_start: int,
    mutation_end: int,
) -> str:
    """Insert a (possibly mutated) mutable region into the full reference.

    Args:
        mutable_sequence: The mutable-region sequence from the pareto front
            (length must equal ``mutation_end - mutation_start``).
        reference_sequence_full: Full reference DNA string.
        mutation_start: 0-based start index of the mutable region.
        mutation_end: 0-based exclusive end index of the mutable region.

    Returns:
        Full-length DNA string with the mutable region replaced.
    """
    return (
        reference_sequence_full[:mutation_start]
        + mutable_sequence
        + reference_sequence_full[mutation_end:]
    )

def _get_zero_mutation_entry(pareto_front: List[ParetoEntry]) -> ParetoEntry:
    zero = [e for e in pareto_front if e[2] == 0]
    if zero:
        return zero[0]
    raise ValueError("No zero-mutation entry found in pareto front.")

def _get_max_mutation_entry(pareto_front: List[ParetoEntry]) -> ParetoEntry:
    """Return the pareto entry with the most mutations.

    Args:
        pareto_front: Loaded pareto front as a list of
            ``(mutable_sequence, fitness, mutation_count)`` tuples.

    Returns:
        The entry with the highest mutation count.
    """
    return max(pareto_front, key=lambda e: e[2])

def slide_windows(
    sequence: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step: int = DEFAULT_STEP_SIZE,
    padding_start: Optional[int] = None,
    padding_end: Optional[int] = None,
    extragenic: int = DEFAULT_EXTRAGENIC,
    intragenic: int = DEFAULT_INTRAGENIC,
    central_padding: int = DEFAULT_CENTRAL_PADDING,
) -> List[Tuple[int, int, str, bool]]:
    """Generate overlapping windows across a DNA sequence.

    Args:
        sequence: DNA string to slide over.
        window_size: Length of each window in bp (default 250).
        step: Step size between consecutive window starts (default 50).
        padding_start: 0-based start index of the central N-padding region.
            If None, inferred from ``extragenic + intragenic``.
        padding_end: Exclusive end index of the central N-padding region.
            If None, inferred from ``padding_start + central_padding``.
        extragenic: Extragenic bp used to extract the sequence (for inferring
            padding_start when it is not supplied explicitly).
        intragenic: Intragenic bp used to extract the sequence.
        central_padding: Length of the central N-padding.

    Returns:
        List of ``(start, end, subsequence, contains_padding)`` tuples where:
        * ``start`` is 0-based inclusive,
        * ``end``   is exclusive so ``end - start == window_size``,
        * ``contains_padding`` is True when the window overlaps the N-pad region.
    """
    if padding_start is None:
        padding_start = extragenic + intragenic
    if padding_end is None:
        padding_end = padding_start + central_padding

    windows: List[Tuple[int, int, str, bool]] = []
    for start in range(0, len(sequence) - window_size + 1, step):
        end = start + window_size
        subseq = sequence[start:end]
        contains_padding = not (end <= padding_start or start >= padding_end)
        windows.append((start, end, subseq, contains_padding))
    return windows

def load_deepcis_model(model_path: str):
    """Load a deepCIS TensorFlow/Keras model.

    Args:
        model_path: Path to a TF SavedModel directory or a ``.h5`` file.

    Returns:
        Loaded ``tf.keras.Model`` ready for inference.

    Raises:
        ImportError: If TensorFlow is not installed.
        ValueError: If the model cannot be loaded from ``model_path``.
    """
    import tensorflow as tf  # type: ignore
    print_status(f"Loading deepCIS model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

def predict_windows(
    model,
    windows_onehot: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Run deepCIS inference on a stack of one-hot encoded windows.

    The function automatically adapts the array layout to match the model's
    expected input shape (channels-first vs channels-last).

    Args:
        model: Loaded ``tf.keras.Model``.
        windows_onehot: Array of shape ``(n_windows, window_size, 4)``
            produced by stacking ``one_hot_encode`` outputs.
        batch_size: Number of windows per model call.

    Returns:
        ``np.ndarray`` of shape ``(n_windows, 46)`` with TF binding likelihoods.
    """
    inp = windows_onehot
    predictions = model.predict(inp, batch_size=batch_size, verbose=0)
    return np.asarray(predictions, dtype=np.float32)

def scan_single_gene_folder(
    model,
    gene_data: GeneRunData,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step: int = DEFAULT_STEP_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    extragenic: int = DEFAULT_EXTRAGENIC,
    intragenic: int = DEFAULT_INTRAGENIC,
    central_padding: int = DEFAULT_CENTRAL_PADDING,
) -> pd.DataFrame:
    """Scan the 0-mutation and max-mutation sequences of one gene with deepCIS.

    Reads the pareto front and reference sequence directly from *gene_data*
    (pre-loaded from the gene folder).  For each of the two sequences
    (``'reference'`` and ``'max_mutated'``), generates overlapping 250 bp
    windows, encodes them as one-hot arrays, and runs them through deepCIS
    in batches.

    Args:
        model: Loaded deepCIS ``tf.keras.Model``.
        gene_data: :class:`GeneRunData` loaded from a gene folder.
        window_size: Sliding-window size in bp (default 250).
        step: Step size between window starts in bp (default 50).
        batch_size: Number of windows per model call.
        extragenic: Extragenic bp used during sequence extraction.
        intragenic: Intragenic bp used during sequence extraction.
        central_padding: Length of the central N-pad (default 20).

    Returns:
        ``pd.DataFrame`` with columns::

            gene, sequence_type, window_start, window_end,
            contains_padding, tf_0, tf_1, …, tf_45
    """
    padding_start = extragenic + intragenic
    padding_end = padding_start + central_padding

    ref_entry = _get_zero_mutation_entry(gene_data.pareto_front)
    max_entry = _get_max_mutation_entry(gene_data.pareto_front)

    ref_full = get_full_sequence(
        ref_entry[0],
        gene_data.reference_sequence_full,
        gene_data.mutation_start,
        gene_data.mutation_end,
    )
    max_full = get_full_sequence(
        max_entry[0],
        gene_data.reference_sequence_full,
        gene_data.mutation_start,
        gene_data.mutation_end,
    )

    rows: List[dict] = []
    for seq_str, seq_type in [(ref_full, "reference"), (max_full, "max_mutated")]:
        windows = slide_windows(
            seq_str,
            window_size=window_size,
            step=step,
            padding_start=padding_start,
            padding_end=padding_end,
        )
        if not windows:
            continue

        _, _, subseqs, _ = zip(*windows)
        encoded = np.stack([one_hot_encode(s) for s in subseqs], axis=0)
        preds = predict_windows(model, encoded, batch_size=batch_size)

        for i, (start, end, _, contains_pad) in enumerate(windows):
            row: dict = {
                "gene": gene_data.gene_name,
                "sequence_type": seq_type,
                "window_start": start,
                "window_end": end,
                "contains_padding": contains_pad,
            }
            for tf_idx, tf_name in enumerate(TF_FAMILY_NAMES):
                row[tf_name] = float(preds[i, tf_idx])
            rows.append(row)

    return pd.DataFrame(rows)

def scan_all_genes(
    model_path: str,
    run_folder: str,
    output_path: str,
    name: Optional[str] = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    step: int = DEFAULT_STEP_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    extragenic: int = DEFAULT_EXTRAGENIC,
    intragenic: int = DEFAULT_INTRAGENIC,
    central_padding: int = DEFAULT_CENTRAL_PADDING,
    overwrite: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, GeneRunData]]:
    """Run deepCIS sliding-window scan over all gene folders in a run folder.

    Discovers gene folders automatically (any subfolder containing both
    ``parameters.json`` and ``saved_populations/pareto_front.json``), loads
    the model once, and calls :func:`scan_gene_folder` for each gene.

    Output filename pattern:
        ``deepcis_window_scan_{name}.csv``
    where *name* defaults to the basename of *run_folder*.

    Args:
        model_path: Path to deepCIS TF SavedModel directory or ``.h5`` file.
        run_folder: Root directory of the evolutionary algorithm run.
        output_path: Directory to write output CSV / parquet files.
        name: Optional label used in the output filename.  Defaults to the
            basename of *run_folder*.
        window_size: Sliding-window size in bp (default 250).
        step: Step size between window starts (default 50).
        batch_size: Inference batch size (default 64).
        extragenic: Extragenic bp used during sequence extraction (default 1000).
        intragenic: Intragenic bp used during sequence extraction (default 500).
        central_padding: Central N-padding length (default 20).
        overwrite: When False, load and return an existing output file without
            re-running inference (genes_data dict will be empty in that case).

    Returns:
        Tuple of:
        * Full results ``pd.DataFrame``.
        * ``Dict[str, GeneRunData]`` mapping gene name → loaded gene data
          (useful for downstream :func:`annotate_mutations_in_windows`).
    """
    run_name = name or os.path.basename(run_folder.rstrip(os.sep))
    csv_path = os.path.join(output_path, f"deepcis_window_scan_{run_name}.csv")

    if os.path.exists(csv_path) and not overwrite:
        print_status(f"Output already exists: {csv_path}", "WARNING")
        print_status("Loading existing file.  Pass --overwrite to re-run.", "INFO")
        return pd.read_csv(csv_path), {}

    gene_folders = find_gene_folders(run_folder)
    print_status(f"Found {len(gene_folders)} gene folders in {run_folder}")
    if not gene_folders:
        raise FileNotFoundError(
            f"No valid gene folders found in {run_folder}."
        )

    os.makedirs(output_path, exist_ok=True)
    model = load_deepcis_model(model_path)

    all_frames: List[pd.DataFrame] = []
    genes_data: Dict[str, GeneRunData] = {}

    for gene_folder in tqdm(gene_folders, desc="Scanning genes"):
        gene_name = os.path.basename(gene_folder)
        try:
            gene_data = GeneRunData.load_gene_run_data(gene_folder)
            genes_data[gene_name] = gene_data
            df = scan_single_gene_folder(
                model=model,
                gene_data=gene_data,
                window_size=window_size,
                step=step,
                batch_size=batch_size,
                extragenic=extragenic,
                intragenic=intragenic,
                central_padding=central_padding,
            )
            all_frames.append(df)
        except Exception as exc:
            print_status(f"Skipping gene {gene_name}: {exc}", "WARNING")

    result_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    result_df.to_csv(csv_path, index=False)
    print_status(f"Saved {len(result_df)} rows to {csv_path}", "SUCCESS")

    try:
        parquet_path = csv_path.replace(".csv", ".parquet")
        result_df.to_parquet(parquet_path, index=False)
        print_status(f"Also saved parquet: {parquet_path}", "SUCCESS")
    except Exception:
        pass

    return result_df, genes_data
