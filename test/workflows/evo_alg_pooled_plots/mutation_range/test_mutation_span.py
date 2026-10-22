"""Unit tests for the 5-mutation span workflow."""

import json
import os
import tempfile
import unittest

import pandas as pd

from analysis.mutations.summarize_mutations import MutatedSequence
from workflows.evo_alg_pooled_plots.mutation_range.mutation_span import (
    collect_all_runs,
    collect_run_spans,
    describe_all_runs,
    describe_spans,
    select_sequences_with_mutation_count,
)

REFERENCE_SEQUENCE = "ACGT" * 100


def _mutation_string(mutations, fitness):
    """Build a summarize_mutations-style mutation string.

    Args:
        mutations: Iterable of ``(position, source_base, new_base)`` tuples.
        fitness: Fitness value appended after the ``|`` separator.

    Returns:
        The encoded mutation string.
    """
    encoded = "".join(f"{pos}{ref}{mut}" for pos, ref, mut in mutations)
    return f"{encoded}|{fitness}"


def _mutations_at(positions):
    """Build valid mutations at the given positions of the test reference.

    Args:
        positions: Positions to mutate.

    Returns:
        List of ``(position, source_base, new_base)`` tuples.
    """
    return [
        (position, REFERENCE_SEQUENCE[position],
         "A" if REFERENCE_SEQUENCE[position] != "A" else "C")
        for position in positions
    ]


def _write_summary_json(path, gene_entries, generation=1999):
    """Write a minimal ``all_mutated_sequences_*.json`` file.

    Args:
        path: Destination file path.
        gene_entries: Mapping gene id -> list of ``(positions, fitness)``.
        generation: Generation key stored in the JSON.
    """
    data = {}
    for gene_id, entries in gene_entries.items():
        data[gene_id] = {
            "reference_sequence": REFERENCE_SEQUENCE,
            str(generation): [
                _mutation_string(_mutations_at(positions), fitness)
                for positions, fitness in entries
            ],
        }
    with open(path, "w") as handle:
        json.dump(data, handle)


class TestSelectSequencesWithMutationCount(unittest.TestCase):
    def test_selects_only_matching_count(self):
        # Arrange
        three = MutatedSequence.from_string(
            REFERENCE_SEQUENCE, _mutation_string(_mutations_at([1, 2, 3]), 0.5)
        )
        five = MutatedSequence.from_string(
            REFERENCE_SEQUENCE, _mutation_string(_mutations_at([1, 2, 3, 4, 5]), 0.9)
        )

        # Act
        selected = select_sequences_with_mutation_count([three, five], 5)

        # Assert
        self.assertEqual(selected, [five])

    def test_returns_empty_when_no_match(self):
        # Arrange
        three = MutatedSequence.from_string(
            REFERENCE_SEQUENCE, _mutation_string(_mutations_at([1, 2, 3]), 0.5)
        )

        # Act
        selected = select_sequences_with_mutation_count([three], 5)

        # Assert
        self.assertEqual(selected, [])


class TestCollectRunSpans(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.json_path = os.path.join(self.temp_dir.name, "run.json")

    def test_extracts_border_positions_and_span(self):
        # Arrange
        _write_summary_json(
            self.json_path,
            {"geneA": [([10, 20, 30, 40, 100], 0.8), ([5, 6], 0.2)]},
        )

        # Act
        table = collect_run_spans(self.json_path, n_mutations=5)

        # Assert
        self.assertEqual(len(table), 1)
        row = table.iloc[0]
        self.assertEqual(row["gene_id"], "geneA")
        self.assertEqual(row["first_position"], 10)
        self.assertEqual(row["last_position"], 100)
        self.assertEqual(row["span"], 90)
        self.assertEqual(row["generation"], 1999)
        self.assertEqual(row["reference_length"], len(REFERENCE_SEQUENCE))

    def test_skips_genes_without_matching_sequence(self):
        # Arrange
        _write_summary_json(
            self.json_path,
            {
                "geneA": [([10, 20, 30, 40, 100], 0.8)],
                "geneB": [([1, 2, 3], 0.4)],
            },
        )

        # Act
        table = collect_run_spans(self.json_path, n_mutations=5)

        # Assert
        self.assertEqual(list(table["gene_id"]), ["geneA"])

    def test_picks_highest_fitness_on_ties(self):
        # Arrange
        _write_summary_json(
            self.json_path,
            {
                "geneA": [
                    ([1, 2, 3, 4, 5], 0.1),
                    ([10, 11, 12, 13, 200], 0.9),
                ]
            },
        )

        # Act
        table = collect_run_spans(self.json_path, n_mutations=5)

        # Assert
        self.assertEqual(table.iloc[0]["span"], 190)

    def test_missing_file_raises(self):
        # Arrange
        missing = os.path.join(self.temp_dir.name, "nope.json")

        # Act / Assert
        with self.assertRaises(FileNotFoundError):
            collect_run_spans(missing)


class TestSummaries(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.run_one = os.path.join(self.temp_dir.name, "run_one.json")
        self.run_two = os.path.join(self.temp_dir.name, "run_two.json")
        _write_summary_json(
            self.run_one,
            {
                "geneA": [([10, 20, 30, 40, 50], 0.8)],
                "geneB": [([0, 1, 2, 3, 150], 0.7)],
            },
        )
        _write_summary_json(
            self.run_two, {"geneC": [([100, 101, 102, 103, 300], 0.6)]}
        )

    def test_describe_spans_columns(self):
        # Arrange
        table = collect_run_spans(self.run_one)

        # Act
        summary = describe_spans(table)

        # Assert
        self.assertEqual(
            list(summary.columns), ["first_position", "last_position", "span"]
        )
        self.assertEqual(summary.loc["count", "span"], 2)
        self.assertEqual(summary.loc["mean", "span"], (40 + 150) / 2)
        self.assertEqual(summary.loc["max", "first_position"], 10)
        self.assertEqual(summary.loc["min", "last_position"], 50)

    def test_collect_all_runs_adds_run_column(self):
        # Arrange
        run_files = {"one": self.run_one, "two": self.run_two}

        # Act
        table = collect_all_runs(run_files)

        # Assert
        self.assertEqual(table.columns[0], "run")
        self.assertEqual(sorted(table["run"].unique()), ["one", "two"])
        self.assertEqual(len(table), 3)

    def test_describe_all_runs_is_indexed_by_run(self):
        # Arrange
        table = collect_all_runs({"one": self.run_one, "two": self.run_two})

        # Act
        summary = describe_all_runs(table)

        # Assert
        self.assertIsInstance(summary.index, pd.MultiIndex)
        self.assertEqual(list(summary.index.names), ["run", "statistic"])
        self.assertEqual(summary.loc[("two", "mean"), "span"], 200)


if __name__ == "__main__":
    unittest.main()
