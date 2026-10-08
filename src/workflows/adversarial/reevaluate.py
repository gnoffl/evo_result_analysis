"""Re-evaluate evolved Pareto front sequences with sibling MSR models.

The evolutionary algorithm optimized each sequence against a single MSR model. This
script scores the same sequences with every model in a given folder, so the
predictions of the optimization model can be compared against those of its siblings
from the same training run.

See PLAN.md in this folder for the rationale behind the decisions taken here.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.overview.simple_result_stats import deduplicate_pareto_front


class PredictingModel(Protocol):
    """A model that predicts a batch of one hot encoded sequences at once."""

    def predict(self, sequences: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Predict a batch of sequences.

        Args:
            sequences: One hot encoded sequences of shape ``(n, length, 4)``.
            verbose: Verbosity of the underlying framework.

        Returns:
            One value per sequence, possibly with trailing singleton dimensions.
        """
        ...

DEFAULT_MODELS_FOLDER = "/home/gernot/Code/PhD_Code/Evolution/models/MSR_models_M0X0.75"
DEFAULT_RUN_FOLDERS = [
    "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/GOF_LOF/GOF/"
    "GOF_single_mutation_251009_121226_109368",
    "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/GOF_LOF/LOF/"
    "LOF_single_mutation_251020_180028_564570",
]
DEFAULT_OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

CSV_FILE_NAME = "adversarial_predictions.csv"
INPUT_FILE_NAME = "input.csv"
REFERENCE_RECORD_NAME = "reference_sequence_full"
PREDICTION_COLUMN_PREFIX = "prediction_"
CHECKSUM_KEY_PREFIX = "model_sha256:"
CHECKSUM_BLOCK_SIZE = 1024 * 1024

GENE_ID_PATTERN = re.compile(r"^[^_]+_([^_]+)_gene:")

METADATA_COLUMNS = [
    "gene_id",
    "sequence",
    "mutation_count",
    "max_number_mutations",
    "optimization_model",
    "original_fitness",
]


def build_output_folder(output_root: str, run_folder: str, models_folder: str) -> str:
    """Build the output folder path for one run and one set of models.

    The models folder is part of the name so that re-evaluating the same run with a
    different set of models cannot overwrite an existing result.

    Args:
        output_root: Root folder for all outputs.
        run_folder: Evolution run folder that was re-evaluated.
        models_folder: Folder holding the models used for the re-evaluation.

    Returns:
        Path of the form ``<output_root>/<run_name>__<models_folder_name>``.
    """
    run_name = os.path.basename(os.path.normpath(run_folder))
    models_name = os.path.basename(os.path.normpath(models_folder))
    return os.path.join(output_root, f"{run_name}__{models_name}")


def compute_file_checksum(file_path: str) -> str:
    """Compute the SHA-256 checksum of a file.

    Args:
        file_path: Path of the file to read.

    Returns:
        The checksum as a hexadecimal string.
    """
    checksum = hashlib.sha256()
    with open(file_path, "rb") as binary_file:
        for block in iter(lambda: binary_file.read(CHECKSUM_BLOCK_SIZE), b""):
            checksum.update(block)
    return checksum.hexdigest()


def get_analysis_commit() -> str:
    """Read the current commit of the repository this module lives in.

    Returns:
        The commit hash, or ``"unknown"`` if it cannot be determined, for example
        because git is unavailable or the code is not in a repository.
    """
    repository = os.path.dirname(os.path.abspath(__file__))
    try:
        completed = subprocess.run(
            ["git", "-C", repository, "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.decode().strip() or "unknown"


def write_input_record(
    output_folder: str,
    run_folder: str,
    models_folder: str,
    model_paths: Dict[str, str],
    predictions: pd.DataFrame,
) -> str:
    """Record what produced the outputs in this folder.

    Model files can be replaced in place, so the record holds a checksum per model
    rather than only its name, which is what makes a result retraceable.

    Args:
        output_folder: Folder the outputs were written to.
        run_folder: Evolution run folder that was re-evaluated.
        models_folder: Folder holding the models used for the re-evaluation.
        model_paths: Mapping of model name to model file path.
        predictions: Table as written to the output CSV.

    Returns:
        Path to the written record.
    """
    record = [
        ("timestamp", datetime.now().isoformat(timespec="seconds")),
        ("run_folder", os.path.abspath(run_folder)),
        ("models_folder", os.path.abspath(models_folder)),
        ("number_of_models", str(len(model_paths))),
        ("number_of_genes", str(predictions["gene_id"].nunique())),
        ("number_of_sequences", str(len(predictions))),
        ("optimization_models", ";".join(sorted(predictions["optimization_model"].unique()))),
        ("analysis_commit", get_analysis_commit()),
    ]
    for model_name, model_path in model_paths.items():
        record.append((f"{CHECKSUM_KEY_PREFIX}{model_name}", compute_file_checksum(model_path)))

    output_path = os.path.join(output_folder, INPUT_FILE_NAME)
    pd.DataFrame(record, columns=["key", "value"]).to_csv(output_path, index=False)
    return output_path


def clean_gene_id(sequence_name: str) -> str:
    """Reduce a run's sequence name to the bare gene identifier.

    The evolution runs name a sequence by chromosome, gene and coordinates, for example
    ``1_AT1G01720_gene:267992-269819``, of which only ``AT1G01720`` identifies the gene.

    Args:
        sequence_name: Value of ``sequence_name`` in a gene's ``parameters.json``.

    Returns:
        The gene identifier, or the input unchanged if it does not follow that naming
        scheme. Applying the function to an already cleaned identifier returns it
        unchanged, so it is safe to apply to existing tables.
    """
    match = GENE_ID_PATTERN.match(sequence_name)
    return match.group(1) if match else sequence_name


def discover_models(models_folder: str) -> Dict[str, str]:
    """Find all TensorFlow model files in a folder.

    Args:
        models_folder: Folder containing ``.h5`` model files.

    Returns:
        Mapping of model name (file name without the ``.h5`` extension) to full path,
        ordered by model name.

    Raises:
        FileNotFoundError: If the folder does not exist or contains no ``.h5`` files.
    """
    if not os.path.isdir(models_folder):
        raise FileNotFoundError(f"Models folder does not exist: {models_folder}")
    model_paths = {}
    for file_name in sorted(os.listdir(models_folder)):
        if file_name.endswith(".h5"):
            model_paths[file_name[: -len(".h5")]] = os.path.join(models_folder, file_name)
    if not model_paths:
        raise FileNotFoundError(f"No .h5 model files found in: {models_folder}")
    return model_paths


def load_models(model_paths: Dict[str, str]) -> Dict[str, PredictingModel]:
    """Load all TensorFlow models.

    Args:
        model_paths: Mapping of model name to model file path.

    Returns:
        Mapping of model name to the loaded model object.
    """
    from evolution.load_models import get_model_loader

    loader = get_model_loader("tensorflow")
    return {name: loader(path) for name, path in model_paths.items()}


def read_fasta_record(fasta_path: str, record_name: str) -> str:
    """Read a single record from a FASTA file.

    Args:
        fasta_path: Path to the FASTA file.
        record_name: Identifier of the record, without the leading ``>``.

    Returns:
        The sequence of the requested record, upper case and without line breaks.

    Raises:
        KeyError: If the record is not present in the file.
    """
    sequence_lines: List[str] = []
    inside_record = False
    with open(fasta_path, "r") as fasta_file:
        for line in fasta_file:
            line = line.strip()
            if line.startswith(">"):
                if inside_record:
                    break
                inside_record = line[1:].split()[0] == record_name
            elif inside_record:
                sequence_lines.append(line)
    if not sequence_lines:
        raise KeyError(f"Record '{record_name}' not found in {fasta_path}")
    return "".join(sequence_lines).upper()


def reconstruct_full_sequence(
    sequence: str, reference_sequence: str, mutation_start: int, mutation_end: int
) -> str:
    """Restore a full length sequence from a stored Pareto front sequence.

    Depending on the evolution run, the stored sequence is either already the full
    length sequence, or only the mutation window that has to be spliced back into the
    reference sequence.

    Args:
        sequence: Sequence as stored in the Pareto front.
        reference_sequence: Full length reference sequence of the gene.
        mutation_start: First position of the mutation window, inclusive.
        mutation_end: Last position of the mutation window, exclusive.

    Returns:
        The full length sequence.

    Raises:
        ValueError: If the stored sequence matches neither the length of the reference
            sequence nor the length of the mutation window.
    """
    if len(sequence) == len(reference_sequence):
        return sequence
    window_length = mutation_end - mutation_start
    if len(sequence) != window_length:
        raise ValueError(
            f"Stored sequence has length {len(sequence)}, which matches neither the "
            f"reference sequence ({len(reference_sequence)}) nor the mutation window "
            f"({window_length})."
        )
    return reference_sequence[:mutation_start] + sequence + reference_sequence[mutation_end:]


def select_front_representatives(
    pareto_front: List[Sequence], gene_id: str
) -> List[Tuple[str, float, int]]:
    """Reduce a Pareto front to one representative sequence per mutation count.

    Saturated fronts contain many distinct sequences that share the exact same fitness
    and mutation count. Those are indistinguishable to the optimization model, so a
    single representative per mutation count is kept, using the same rule as the
    pooled Pareto front plots.

    A warning is emitted if the fitness values within one mutation count are not all
    identical, because the representative would then be an arbitrary choice among
    genuinely different individuals.

    Args:
        pareto_front: Entries of ``[sequence, fitness, mutation_count]``.
        gene_id: Identifier of the gene, used in the warning message.

    Returns:
        One ``(sequence, fitness, mutation_count)`` tuple per mutation count, ordered
        by ascending mutation count.
    """
    # The front is read from JSON, so the entries are lists; deduplicate_pareto_front
    # expects tuples.
    entries: List[Tuple[str, float, int]] = [
        (str(entry[0]), float(entry[1]), int(entry[2])) for entry in pareto_front
    ]
    fitnesses_per_mutation_count: Dict[int, set] = {}
    for _, fitness, mutation_count in entries:
        fitnesses_per_mutation_count.setdefault(mutation_count, set()).add(fitness)
    ambiguous_counts = sorted(
        count for count, fitnesses in fitnesses_per_mutation_count.items() if len(fitnesses) > 1
    )
    if ambiguous_counts:
        warnings.warn(
            f"Gene {gene_id}: mutation counts {ambiguous_counts} contain differing "
            "fitness values; the selected representative is an arbitrary choice."
        )
    return deduplicate_pareto_front(entries)


def predict_sequences(
    sequences: List[str], models: Dict[str, PredictingModel]
) -> Dict[str, np.ndarray]:
    """Predict a batch of sequences with every model.

    Args:
        sequences: Full length nucleotide sequences.
        models: Mapping of model name to a model exposing ``predict``.

    Returns:
        Mapping of model name to a one dimensional array of predictions, one value per
        sequence.

    Raises:
        ValueError: If a model does not return exactly one value per sequence.
    """
    from evolution.sequences import one_hot_encode

    encoded = np.stack([one_hot_encode(sequence) for sequence in sequences])
    predictions = {}
    for model_name, model in models.items():
        prediction = np.asarray(model.predict(encoded, verbose=0))
        if prediction.size != len(sequences):
            raise ValueError(
                f"Model {model_name} returned {prediction.size} values for "
                f"{len(sequences)} sequences; exactly one value per sequence is required."
            )
        predictions[model_name] = prediction.reshape(len(sequences))
    return predictions


def find_gene_folders(run_folder: str) -> List[str]:
    """Find all gene folders of an evolution run.

    Args:
        run_folder: Folder holding one subfolder per optimized gene.

    Returns:
        Sorted list of gene folder paths that contain a ``parameters.json``.

    Raises:
        FileNotFoundError: If the run folder does not exist or contains no gene folders.
    """
    if not os.path.isdir(run_folder):
        raise FileNotFoundError(f"Run folder does not exist: {run_folder}")
    gene_folders = []
    for entry in sorted(os.listdir(run_folder)):
        gene_folder = os.path.join(run_folder, entry)
        if os.path.isfile(os.path.join(gene_folder, "parameters.json")):
            gene_folders.append(gene_folder)
    if not gene_folders:
        raise FileNotFoundError(f"No gene folders found in: {run_folder}")
    return gene_folders


def collect_gene_rows(
    gene_folder: str, models: Dict[str, PredictingModel]
) -> List[Dict[str, object]]:
    """Re-evaluate the Pareto front of a single gene with all models.

    Args:
        gene_folder: Folder of one optimized gene.
        models: Mapping of model name to a model exposing ``predict``.

    Returns:
        One row per mutation count present in the Pareto front. Returns an empty list
        if the gene has no saved Pareto front.
    """
    with open(os.path.join(gene_folder, "parameters.json"), "r") as parameters_file:
        parameters = json.load(parameters_file)
    pareto_front_path = os.path.join(gene_folder, "saved_populations", "pareto_front.json")
    if not os.path.isfile(pareto_front_path):
        warnings.warn(f"No Pareto front found for {gene_folder}, skipping.")
        return []
    with open(pareto_front_path, "r") as pareto_front_file:
        pareto_front = json.load(pareto_front_file)

    gene_id = clean_gene_id(parameters["sequence_name"])
    optimization_models = [
        os.path.basename(model["path"])[: -len(".h5")] for model in parameters["models"]
    ]
    reference_sequence = read_fasta_record(
        os.path.join(gene_folder, "reference_sequence.fa"), REFERENCE_RECORD_NAME
    )
    representatives = select_front_representatives(pareto_front, gene_id)
    sequences = [
        reconstruct_full_sequence(
            sequence,
            reference_sequence,
            parameters["mutation_start"],
            parameters["mutation_end"],
        )
        for sequence, _, _ in representatives
    ]
    predictions = predict_sequences(sequences, models)

    rows = []
    for index, (_, fitness, mutation_count) in enumerate(representatives):
        row: Dict[str, object] = {
            "gene_id": gene_id,
            "sequence": sequences[index],
            "mutation_count": mutation_count,
            "max_number_mutations": parameters["max_number_mutations"],
            "optimization_model": ";".join(optimization_models),
            "original_fitness": fitness,
        }
        for model_name, model_predictions in predictions.items():
            row[f"{PREDICTION_COLUMN_PREFIX}{model_name}"] = float(model_predictions[index])
        rows.append(row)
    return rows


def report_optimization_model_agreement(predictions: pd.DataFrame) -> Optional[float]:
    """Compare the re-computed optimization model prediction to the stored fitness.

    The optimization model is re-evaluated together with its siblings, so its
    prediction must reproduce the fitness stored in the Pareto front. This validates
    sequence reconstruction and one hot encoding end to end.

    Args:
        predictions: Table as written to the output CSV.

    Returns:
        The maximum absolute deviation, or None if the optimization model of a gene is
        not part of the evaluated model set.
    """
    deviations = []
    for optimization_model, group in predictions.groupby("optimization_model"):
        column = f"{PREDICTION_COLUMN_PREFIX}{optimization_model}"
        if column not in group.columns:
            return None
        deviations.append((group[column] - group["original_fitness"]).abs().max())
    return float(max(deviations))


def order_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Order the output columns as metadata first, then predictions by model name.

    Args:
        predictions: Table with metadata and prediction columns.

    Returns:
        The table with reordered columns.
    """
    prediction_columns = sorted(
        column for column in predictions.columns if column.startswith(PREDICTION_COLUMN_PREFIX)
    )
    return predictions[METADATA_COLUMNS + prediction_columns]


def reevaluate_run(
    run_folder: str,
    models: Dict[str, PredictingModel],
    model_paths: Dict[str, str],
    models_folder: str,
    output_root: str,
) -> str:
    """Re-evaluate every gene of one evolution run and write the result to a CSV.

    Args:
        run_folder: Folder holding one subfolder per optimized gene.
        models: Mapping of model name to a model exposing ``predict``.
        model_paths: Mapping of model name to model file path, for the input record.
        models_folder: Folder the models were loaded from.
        output_root: Root folder for outputs; a subfolder named after the run folder and
            the models folder is created inside it, see ``build_output_folder``.

    Returns:
        Path to the written CSV file.
    """
    gene_folders = find_gene_folders(run_folder)
    run_name = os.path.basename(os.path.normpath(run_folder))
    rows: List[Dict[str, object]] = []
    for gene_folder in tqdm(gene_folders, desc=run_name):
        rows.extend(collect_gene_rows(gene_folder, models))

    predictions = order_columns(pd.DataFrame(rows))
    output_folder = build_output_folder(output_root, run_folder, models_folder)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, CSV_FILE_NAME)
    predictions.to_csv(output_path, index=False)
    input_path = write_input_record(
        output_folder, run_folder, models_folder, model_paths, predictions
    )

    max_deviation = report_optimization_model_agreement(predictions)
    print(
        f"{run_name}: {predictions['gene_id'].nunique()} genes, {len(predictions)} sequences, "
        f"{len(models)} models -> {output_path}"
    )
    print(f"  inputs recorded in {input_path}")
    if max_deviation is None:
        print(
            "  optimization model not present in the model folder, "
            "cannot validate against the stored fitness"
        )
    else:
        print(f"  max |recomputed - stored| fitness of the optimization model: {max_deviation:.3g}")
    return output_path


def parse_arguments() -> argparse.Namespace:
    """Parse the command line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-folder",
        default=DEFAULT_MODELS_FOLDER,
        help="Folder containing the .h5 models to evaluate with.",
    )
    parser.add_argument(
        "--run-folders",
        nargs="+",
        default=DEFAULT_RUN_FOLDERS,
        help="Evolution run folders to re-evaluate, one CSV is written per run folder.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder for the output CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    """Re-evaluate all requested runs with all models in the models folder."""
    arguments = parse_arguments()
    model_paths = discover_models(arguments.models_folder)
    print(f"Loading {len(model_paths)} models from {arguments.models_folder}")
    models = load_models(model_paths)
    for run_folder in arguments.run_folders:
        reevaluate_run(
            run_folder,
            models,
            model_paths,
            arguments.models_folder,
            arguments.output_root,
        )


if __name__ == "__main__":
    main()
