"""Tests for workflows.adversarial.reevaluate."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# The evolution package is only needed inside the functions that predict, so a stub is
# installed for the tests that never touch a real model.
sys.modules.setdefault("evolution", MagicMock())
sys.modules.setdefault("evolution.load_models", MagicMock())
sys.modules.setdefault("evolution.sequences", MagicMock())

from workflows.adversarial import reevaluate  # noqa: E402


class TestDiscoverModels(unittest.TestCase):
    """Tests for model discovery in a folder."""

    def test_only_h5_files_are_returned_ordered_by_name(self):
        with tempfile.TemporaryDirectory() as folder:
            for file_name in ["model_b.h5", "model_a.h5", "notes.txt"]:
                open(os.path.join(folder, file_name), "w").close()

            models = reevaluate.discover_models(folder)

            self.assertEqual(["model_a", "model_b"], list(models))
            self.assertEqual(os.path.join(folder, "model_a.h5"), models["model_a"])

    def test_missing_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            reevaluate.discover_models("/nonexistent/models/folder")

    def test_folder_without_models_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                reevaluate.discover_models(folder)


class TestBuildOutputFolder(unittest.TestCase):
    """Tests for the output folder naming convention."""

    def test_run_and_models_folder_are_both_in_the_name(self):
        folder = reevaluate.build_output_folder(
            "/outputs", "/data/GOF_single_mutation_251009", "/models/MSR_models_M0X0.75"
        )

        self.assertEqual(
            "/outputs/GOF_single_mutation_251009__MSR_models_M0X0.75", folder
        )

    def test_different_models_folders_give_different_output_folders(self):
        first = reevaluate.build_output_folder("/outputs", "/data/run", "/models/set_a")
        second = reevaluate.build_output_folder("/outputs", "/data/run", "/models/set_b")

        self.assertNotEqual(first, second)

    def test_trailing_separators_are_ignored(self):
        with_separators = reevaluate.build_output_folder(
            "/outputs", "/data/run/", "/models/set_a/"
        )
        without_separators = reevaluate.build_output_folder(
            "/outputs", "/data/run", "/models/set_a"
        )

        self.assertEqual(without_separators, with_separators)


class TestComputeFileChecksum(unittest.TestCase):
    """Tests for the model file checksum."""

    def test_checksum_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as folder:
            file_path = os.path.join(folder, "model.h5")
            with open(file_path, "wb") as binary_file:
                binary_file.write(b"model weights")

            checksum = reevaluate.compute_file_checksum(file_path)

            self.assertEqual(hashlib.sha256(b"model weights").hexdigest(), checksum)

    def test_different_content_gives_a_different_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            checksums = []
            for content in [b"weights_a", b"weights_b"]:
                file_path = os.path.join(folder, "model.h5")
                with open(file_path, "wb") as binary_file:
                    binary_file.write(content)
                checksums.append(reevaluate.compute_file_checksum(file_path))

            self.assertNotEqual(checksums[0], checksums[1])


class TestWriteInputRecord(unittest.TestCase):
    """Tests for the provenance record."""

    def setUp(self):
        self.temporary_folder = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temporary_folder.name, "model_opt.h5")
        with open(self.model_path, "wb") as binary_file:
            binary_file.write(b"model weights")
        self.predictions = pd.DataFrame(
            {
                "gene_id": ["AT1G00001", "AT1G00001", "AT1G00002"],
                "optimization_model": ["model_opt"] * 3,
                "original_fitness": [0.1, 0.9, 0.5],
            }
        )

    def tearDown(self):
        self.temporary_folder.cleanup()

    def write(self):
        """Write the record and return it as a key to value mapping."""
        path = reevaluate.write_input_record(
            self.temporary_folder.name,
            "/data/GOF_run",
            "/models/MSR_models_M0X0.75",
            {"model_opt": self.model_path},
            self.predictions,
        )
        record = pd.read_csv(path)
        return dict(zip(record["key"], record["value"]))

    def test_inputs_and_counts_are_recorded(self):
        record = self.write()

        self.assertEqual("/data/GOF_run", record["run_folder"])
        self.assertEqual("/models/MSR_models_M0X0.75", record["models_folder"])
        self.assertEqual("1", str(record["number_of_models"]))
        self.assertEqual("2", str(record["number_of_genes"]))
        self.assertEqual("3", str(record["number_of_sequences"]))
        self.assertEqual("model_opt", record["optimization_models"])

    def test_a_checksum_is_recorded_per_model(self):
        record = self.write()

        key = f"{reevaluate.CHECKSUM_KEY_PREFIX}model_opt"
        self.assertEqual(hashlib.sha256(b"model weights").hexdigest(), record[key])

    def test_timestamp_and_commit_are_present(self):
        record = self.write()

        self.assertIn("timestamp", record)
        self.assertIn("analysis_commit", record)


class TestCleanGeneId(unittest.TestCase):
    """Tests for reducing a sequence name to the gene identifier."""

    def test_chromosome_and_coordinates_are_stripped(self):
        self.assertEqual(
            "AT1G01720", reevaluate.clean_gene_id("1_AT1G01720_gene:267992-269819")
        )

    def test_already_clean_identifier_is_unchanged(self):
        self.assertEqual("AT1G01720", reevaluate.clean_gene_id("AT1G01720"))

    def test_unknown_naming_scheme_is_unchanged(self):
        self.assertEqual("some_other_name", reevaluate.clean_gene_id("some_other_name"))


class TestReadFastaRecord(unittest.TestCase):
    """Tests for reading a single record from a FASTA file."""

    def setUp(self):
        self.temporary_folder = tempfile.TemporaryDirectory()
        self.fasta_path = os.path.join(self.temporary_folder.name, "reference_sequence.fa")
        with open(self.fasta_path, "w") as fasta_file:
            fasta_file.write(">reference_sequence_full\nacgt\nACGT\n")
            fasta_file.write(">reference_sequence_mutation_window_0_4\nTTTT\n")

    def tearDown(self):
        self.temporary_folder.cleanup()

    def test_first_record_is_read_and_upper_cased(self):
        sequence = reevaluate.read_fasta_record(self.fasta_path, "reference_sequence_full")

        self.assertEqual("ACGTACGT", sequence)

    def test_later_record_is_read(self):
        sequence = reevaluate.read_fasta_record(
            self.fasta_path, "reference_sequence_mutation_window_0_4"
        )

        self.assertEqual("TTTT", sequence)

    def test_missing_record_raises(self):
        with self.assertRaises(KeyError):
            reevaluate.read_fasta_record(self.fasta_path, "does_not_exist")


class TestReconstructFullSequence(unittest.TestCase):
    """Tests for restoring full length sequences."""

    def test_full_length_sequence_is_returned_unchanged(self):
        reconstructed = reevaluate.reconstruct_full_sequence("AAAA", "CCCC", 1, 3)

        self.assertEqual("AAAA", reconstructed)

    def test_window_is_spliced_into_the_reference(self):
        reconstructed = reevaluate.reconstruct_full_sequence("GG", "CCCC", 1, 3)

        self.assertEqual("CGGC", reconstructed)

    def test_unexpected_length_raises(self):
        with self.assertRaises(ValueError):
            reevaluate.reconstruct_full_sequence("GGG", "CCCC", 1, 3)


class TestSelectFrontRepresentatives(unittest.TestCase):
    """Tests for reducing a Pareto front to one entry per mutation count."""

    def test_one_representative_per_mutation_count_ordered_ascending(self):
        pareto_front = [
            ["SEQ_2_A", 0.9, 2],
            ["SEQ_2_B", 0.9, 2],
            ["SEQ_0", 0.1, 0],
            ["SEQ_5", 0.95, 5],
        ]

        representatives = reevaluate.select_front_representatives(pareto_front, "gene")

        self.assertEqual([0, 2, 5], [entry[2] for entry in representatives])
        self.assertEqual([0.1, 0.9, 0.95], [entry[1] for entry in representatives])

    def test_differing_fitness_within_a_mutation_count_warns(self):
        pareto_front = [["SEQ_A", 0.9, 1], ["SEQ_B", 0.8, 1], ["SEQ_0", 0.1, 0]]

        with self.assertWarns(UserWarning):
            reevaluate.select_front_representatives(pareto_front, "gene")


class TestPredictSequences(unittest.TestCase):
    """Tests for batched prediction with several models."""

    def setUp(self):
        self.one_hot_encode = lambda sequence: np.zeros((len(sequence), 4), dtype=np.float32)

    def test_predictions_are_flattened_to_one_value_per_sequence(self):
        model = MagicMock()
        model.predict.return_value = np.array([[0.25], [0.75]])
        with patch.dict(
            "sys.modules",
            {"evolution.sequences": MagicMock(one_hot_encode=self.one_hot_encode)},
        ):
            predictions = reevaluate.predict_sequences(["ACGT", "TGCA"], {"model_a": model})

        np.testing.assert_allclose([0.25, 0.75], predictions["model_a"])

    def test_wrong_number_of_values_raises(self):
        model = MagicMock()
        model.predict.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        with patch.dict(
            "sys.modules",
            {"evolution.sequences": MagicMock(one_hot_encode=self.one_hot_encode)},
        ):
            with self.assertRaises(ValueError):
                reevaluate.predict_sequences(["ACGT", "TGCA"], {"model_a": model})


class TestFindGeneFolders(unittest.TestCase):
    """Tests for locating the gene folders of a run."""

    def test_only_folders_with_parameters_are_returned(self):
        with tempfile.TemporaryDirectory() as run_folder:
            for gene in ["gene_b", "gene_a"]:
                os.makedirs(os.path.join(run_folder, gene))
                open(os.path.join(run_folder, gene, "parameters.json"), "w").close()
            os.makedirs(os.path.join(run_folder, "without_parameters"))
            open(os.path.join(run_folder, "summary.csv"), "w").close()

            gene_folders = reevaluate.find_gene_folders(run_folder)

            self.assertEqual(
                [os.path.join(run_folder, "gene_a"), os.path.join(run_folder, "gene_b")],
                gene_folders,
            )

    def test_run_folder_without_genes_raises(self):
        with tempfile.TemporaryDirectory() as run_folder:
            with self.assertRaises(FileNotFoundError):
                reevaluate.find_gene_folders(run_folder)


def write_gene_folder(
    run_folder, sequence_name, reference_sequence, pareto_front, optimization_model
):
    """Create a minimal gene folder on disk.

    Args:
        run_folder: Folder to create the gene folder in.
        sequence_name: Value written to ``sequence_name``.
        reference_sequence: Full length reference sequence.
        pareto_front: Entries of ``[sequence, fitness, mutation_count]``.
        optimization_model: Model name, without the ``.h5`` extension.

    Returns:
        Path to the created gene folder.
    """
    gene_folder = os.path.join(run_folder, f"{sequence_name}_timestamp")
    os.makedirs(os.path.join(gene_folder, "saved_populations"))
    parameters = {
        "sequence_name": sequence_name,
        "models": [{"path": f"/models/{optimization_model}.h5", "model_type": "tensorflow"}],
        "mutation_start": 0,
        "mutation_end": len(reference_sequence),
        "max_number_mutations": 3,
    }
    with open(os.path.join(gene_folder, "parameters.json"), "w") as parameters_file:
        json.dump(parameters, parameters_file)
    with open(os.path.join(gene_folder, "reference_sequence.fa"), "w") as fasta_file:
        fasta_file.write(f">{reevaluate.REFERENCE_RECORD_NAME}\n{reference_sequence}\n")
    front_path = os.path.join(gene_folder, "saved_populations", "pareto_front.json")
    with open(front_path, "w") as front_file:
        json.dump(pareto_front, front_file)
    return gene_folder


class TestCollectGeneRows(unittest.TestCase):
    """Tests for re-evaluating a single gene."""

    def setUp(self):
        self.temporary_folder = tempfile.TemporaryDirectory()
        self.gene_folder = write_gene_folder(
            self.temporary_folder.name,
            "1_AT1G00001_gene:267992-269819",
            "ACGT",
            [["TTTT", 0.9, 2], ["TTTA", 0.9, 2], ["ACGT", 0.1, 0]],
            "model_opt",
        )
        self.one_hot_encode = lambda sequence: np.zeros((len(sequence), 4), dtype=np.float32)

    def tearDown(self):
        self.temporary_folder.cleanup()

    def make_models(self):
        """Build two fake models returning fixed predictions."""
        optimization_model = MagicMock()
        optimization_model.predict.return_value = np.array([[0.1], [0.9]])
        other_model = MagicMock()
        other_model.predict.return_value = np.array([[0.2], [0.4]])
        return {"model_opt": optimization_model, "model_other": other_model}

    def collect(self):
        """Run ``collect_gene_rows`` with the encoding stubbed out."""
        with patch.dict(
            "sys.modules",
            {"evolution.sequences": MagicMock(one_hot_encode=self.one_hot_encode)},
        ):
            return reevaluate.collect_gene_rows(self.gene_folder, self.make_models())

    def test_one_row_per_mutation_count_with_all_predictions(self):
        rows = self.collect()

        self.assertEqual(2, len(rows))
        self.assertEqual([0, 2], [row["mutation_count"] for row in rows])
        # The long sequence name is reduced to the bare gene identifier.
        self.assertEqual(["AT1G00001", "AT1G00001"], [row["gene_id"] for row in rows])
        self.assertEqual("model_opt", rows[0]["optimization_model"])
        self.assertAlmostEqual(0.1, rows[0]["original_fitness"])
        self.assertAlmostEqual(0.1, rows[0]["prediction_model_opt"])
        self.assertAlmostEqual(0.2, rows[0]["prediction_model_other"])
        self.assertAlmostEqual(0.4, rows[1]["prediction_model_other"])

    def test_missing_pareto_front_is_skipped_with_a_warning(self):
        os.remove(os.path.join(self.gene_folder, "saved_populations", "pareto_front.json"))

        with self.assertWarns(UserWarning):
            rows = self.collect()

        self.assertEqual([], rows)


class TestReportOptimizationModelAgreement(unittest.TestCase):
    """Tests for the validation against the stored fitness."""

    def test_deviation_is_reported(self):
        predictions = pd.DataFrame(
            {
                "optimization_model": ["model_opt", "model_opt"],
                "original_fitness": [0.1, 0.9],
                "prediction_model_opt": [0.1, 0.8],
            }
        )

        deviation = reevaluate.report_optimization_model_agreement(predictions)

        self.assertAlmostEqual(0.1, deviation)

    def test_missing_optimization_model_returns_none(self):
        predictions = pd.DataFrame(
            {
                "optimization_model": ["model_absent"],
                "original_fitness": [0.1],
                "prediction_model_other": [0.2],
            }
        )

        self.assertIsNone(reevaluate.report_optimization_model_agreement(predictions))


class TestOrderColumns(unittest.TestCase):
    """Tests for the output column order."""

    def test_metadata_first_then_predictions_sorted(self):
        predictions = pd.DataFrame(
            {
                "prediction_b": [0.0],
                "original_fitness": [0.0],
                "prediction_a": [0.0],
                "gene_id": ["g"],
                "sequence": ["A"],
                "mutation_count": [0],
                "max_number_mutations": [3],
                "optimization_model": ["m"],
            }
        )

        ordered = reevaluate.order_columns(predictions)

        self.assertEqual(
            reevaluate.METADATA_COLUMNS + ["prediction_a", "prediction_b"], list(ordered.columns)
        )


if __name__ == "__main__":
    unittest.main()
