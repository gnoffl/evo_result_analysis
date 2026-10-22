"""Unit tests for the MutationPool class (Step 1 of mutation_distribution_analysis)."""

from __future__ import annotations

import tempfile
import unittest
from typing import Dict, List, Tuple
from unittest import mock
import pandas as pd

import pandas.testing as pdt

from analysis.mutations.summarize_mutations import MutatedSequence, MutationsGene
from workflows.mutation_distribution_analysis import mutation_pool as mp
from workflows.mutation_distribution_analysis.mutation_pool import MutationPool


def _make_sequence(
    reference: str, mutations: List[Tuple[int, str, str]], fitness: float
) -> MutatedSequence:
    """Construct a MutatedSequence with an explicit mutation list.

    Bypasses the regex-restricted ``from_string`` so tests can inject
    non-ACGT mutations (used by the filter test).
    """
    seq = MutatedSequence.__new__(MutatedSequence)
    seq.reference_sequence = reference
    seq.mutations = list(mutations)
    seq.mutated_sequence = reference
    seq.fitness = fitness
    return seq


def _make_gene(
    reference: str,
    sequences_by_gen: Dict[int, List[MutatedSequence]],
) -> MutationsGene:
    """Construct a MutationsGene with arbitrary per-generation sequence lists."""
    gene = MutationsGene.__new__(MutationsGene)
    gene.reference_sequence = reference
    gene.generation_dict = sequences_by_gen
    return gene


# ---------------------------------------------------------------------------
# from_summarized_json
# ---------------------------------------------------------------------------


class TestFromSummarizedJson(unittest.TestCase):
    """Tests for MutationPool.from_summarized_json."""

    def test_selects_highest_fitness(self) -> None:
        """The highest-fitness pareto-front sequence is picked for each gene."""
        reference = "ACGT" * 10
        low = _make_sequence(reference, [(0, "A", "T")], fitness=0.10)
        mid = _make_sequence(reference, [(0, "A", "T"), (5, "C", "G")], fitness=0.50)
        high = _make_sequence(reference, [(0, "A", "T"), (5, "C", "G"), (9, "T", "A")], fitness=0.99,)
        gene = _make_gene(reference, {1999: [low, mid, high]})

        with mock.patch.object(
            mp, "load_mutations_from_json", return_value={"gene_a": gene}
        ):
            pool = MutationPool.from_summarized_json("ignored.json", generation=1999)

        self.assertEqual(len(pool.mutations), 3)
        self.assertEqual(set(pool.mutations["position"]), {0, 5, 9})
        self.assertEqual(pool.gene_stats.loc[0, "n_mutations"], 3)
        self.assertAlmostEqual(pool.gene_stats.loc[0, "initial_fitness"], 0.10)
        self.assertAlmostEqual(pool.gene_stats.loc[0, "final_fitness"], 0.99)

    def test_filters_non_acgt_mutations(self) -> None:
        """Mutations with non-ACGT source or new bases are dropped."""
        reference = "ACGT" * 10
        seq = _make_sequence(
            reference,
            [
                (1, "N", "A"),  # invalid source -> drop
                (2, "A", "N"),  # invalid new -> drop
                (3, "A", "C"),  # valid -> keep
                (4, "C", "T"),  # valid -> keep
            ],
            fitness=0.8,
        )
        gene = _make_gene(reference, {1999: [seq]})

        with mock.patch.object(
            mp, "load_mutations_from_json", return_value={"gene_a": gene}
        ):
            pool = MutationPool.from_summarized_json("ignored.json", generation=1999)

        self.assertEqual(len(pool.mutations), 2)
        kept_positions = sorted(pool.mutations["position"].tolist())
        self.assertEqual(kept_positions, [3, 4])
        self.assertTrue(set(pool.mutations["source_base"]).issubset({"A", "C", "G", "T"}))
        self.assertTrue(set(pool.mutations["new_base"]).issubset({"A", "C", "G", "T"}))
        self.assertEqual(pool.gene_stats.loc[0, "n_mutations"], 2)

    def test_gene_stats_derived_correctly(self) -> None:
        """gene_stats fields match the underlying mutation list and reference."""
        reference_a = "ACGT" * 25  # length 100
        reference_b = "TGCA" * 50  # length 200

        gene_a = _make_gene(
            reference_a,
            {
                1999: [
                    _make_sequence(reference_a, [], fitness=0.2),
                    _make_sequence(reference_a, [(0, "A", "T")], fitness=0.3),
                    _make_sequence(reference_a, [(0, "A", "T"), (4, "A", "G"), (2, "G", "C")], fitness=0.7,),
                ]
            },
        )
        gene_b = _make_gene(
            reference_b,
            {
                1999: [
                    _make_sequence(reference_b, [], fitness=0.2),
                    _make_sequence(reference_b, [(1, "G", "A")], fitness=0.4)
                ]
            },
        )

        with mock.patch.object(
            mp,
            "load_mutations_from_json",
            return_value={"gene_a": gene_a, "gene_b": gene_b},
        ):
            pool = MutationPool.from_summarized_json("ignored.json", generation=1999)

        stats_by_gene = pool.gene_stats.set_index("gene_id")

        self.assertEqual(stats_by_gene.loc["gene_a", "n_mutations"], 3)
        self.assertAlmostEqual(stats_by_gene.loc["gene_a", "initial_fitness"], 0.2)
        self.assertAlmostEqual(stats_by_gene.loc["gene_a", "final_fitness"], 0.7)

        self.assertEqual(stats_by_gene.loc["gene_b", "n_mutations"], 1)
        self.assertAlmostEqual(stats_by_gene.loc["gene_b", "initial_fitness"], 0.2)
        self.assertAlmostEqual(stats_by_gene.loc["gene_b", "final_fitness"], 0.4)

    def test_mutations_dataframe_shape_and_columns(self) -> None:
        """mutations DataFrame has expected columns and total row count."""
        reference = "ACGT" * 10
        gene_a = _make_gene(reference, {1999: [_make_sequence(reference, [(0, "A", "T"), (4, "A", "G"), (8, "A", "C")], fitness=0.9,) ]},)
        gene_b = _make_gene(reference, {1999: [_make_sequence(reference, [(1, "C", "G"), (5, "C", "A")], fitness=0.8,)]},)

        with mock.patch.object(
            mp,
            "load_mutations_from_json",
            return_value={"gene_a": gene_a, "gene_b": gene_b},
        ):
            pool = MutationPool.from_summarized_json("ignored.json", generation=1999)

        self.assertEqual(
            list(pool.mutations.columns),
            ["gene_id", "position", "source_base", "new_base"],
        )
        self.assertEqual(len(pool.mutations), 5)  # 3 + 2
        per_gene = pool.mutations.groupby("gene_id").size().to_dict()
        self.assertEqual(per_gene, {"gene_a": 3, "gene_b": 2})
    
    def test_check_results(self) -> None:
        """Check that the results are correct for a simple case."""
        reference = "ACGT" * 10
        gene_a = _make_gene(reference, {1999: [
            _make_sequence(reference, [], fitness=0.5,),
            _make_sequence(reference, [(0, "A", "T"), (4, "A", "G")], fitness=0.9,),
        ]})
        gene_b = _make_gene(reference, {1999: [
            _make_sequence(reference, [], fitness=0.3,),
            _make_sequence(reference, [(0, "A", "T"), (4, "A", "C"), (8, "A", "G"), (12, "A", "T"), (17, "C", "G")], fitness=0.9,),
        ]})

        with mock.patch.object(
            mp, "load_mutations_from_json", return_value={"gene_a": gene_a, "gene_b": gene_b}
        ):
            pool = MutationPool.from_summarized_json("ignored.json", generation=1999)

        expected_mutations = pd.DataFrame(
            {
                "gene_id": ["gene_a", "gene_a", "gene_b", "gene_b", "gene_b", "gene_b", "gene_b"],
                "position": [0, 4, 0, 4, 8, 12, 17],
                "source_base": ["A", "A", "A", "A", "A", "A", "C"],
                "new_base": ["T", "G", "T", "C", "G", "T", "G"],
            }
        )
        expected_gene_stats = pd.DataFrame(
            {
                "gene_id": ["gene_a", "gene_b"],
                "n_mutations": [2, 5],
                "initial_fitness": [0.5, 0.3],
                "final_fitness": [0.9, 0.9],
            }
        )
        expected_references = {"gene_a": reference, "gene_b": reference}
        pdt.assert_frame_equal(pool.mutations.reset_index(drop=True), expected_mutations)
        pdt.assert_frame_equal(pool.gene_stats.reset_index(drop=True), expected_gene_stats)
        self.assertEqual(pool.references, expected_references)

    def test_empty_generation(self) -> None:
        """If the specified generation is empty for a gene, that gene is skipped."""
        reference = "ACGT" * 10
        gene_a = _make_gene(reference, {1999: []})

        with mock.patch.object(
            mp,
            "load_mutations_from_json",
            return_value={"gene_a": gene_a},
        ):
            with self.assertRaises(ValueError) as context:
                pool = MutationPool.from_summarized_json("ignored.json", generation=1999)

# ---------------------------------------------------------------------------
# _select_best_sequence
# ---------------------------------------------------------------------------


class TestSelectBestSequence(unittest.TestCase):
    """Tests for mp._select_best_sequence."""

    def test_picks_max_fitness(self) -> None:
        """The sequence with the highest fitness is returned."""
        reference = "ACGT" * 10
        low = _make_sequence(reference, [(0, "A", "T")], fitness=0.10)
        mid = _make_sequence(reference, [(1, "C", "G")], fitness=0.50)
        high = _make_sequence(reference, [(2, "G", "A")], fitness=0.99)
        gene = _make_gene(reference, {1999: [low, mid, high]})

        best = mp._select_best_sequence("gene_a", gene, 1999)

        self.assertIs(best, high)

    def test_raises_when_generation_missing(self) -> None:
        """Missing generation key raises ValueError mentioning gene and generation."""
        gene = _make_gene("ACGT" * 10, {1999: []})

        with self.assertRaisesRegex(ValueError, "gene_a.*42"):
            mp._select_best_sequence("gene_a", gene, 42)

    def test_raises_when_generation_empty(self) -> None:
        """Empty sequence list for the requested generation raises ValueError."""
        gene = _make_gene("ACGT" * 10, {1999: []})

        with self.assertRaisesRegex(ValueError, "gene_a.*1999"):
            mp._select_best_sequence("gene_a", gene, 1999)


# ---------------------------------------------------------------------------
# _filter_valid_mutations
# ---------------------------------------------------------------------------


class TestFilterValidMutations(unittest.TestCase):
    """Tests for mp._filter_valid_mutations."""

    def test_keeps_all_acgt(self) -> None:
        """All mutations with ACGT source and new bases are kept."""
        mutations = [(0, "A", "C"), (1, "G", "T"), (2, "C", "A")]

        kept, dropped = mp._filter_valid_mutations(mutations)

        self.assertEqual(kept, mutations)
        self.assertEqual(dropped, 0)

    def test_partitions_mixed_input(self) -> None:
        """Non-ACGT source or new base entries are dropped; rest is kept in order."""
        mutations = [
            (0, "N", "A"),  # invalid source
            (1, "A", "N"),  # invalid new
            (2, "A", "C"),  # valid
            (3, "g", "T"),  # invalid (lowercase)
            (4, "C", "T"),  # valid
        ]

        kept, dropped = mp._filter_valid_mutations(mutations)

        self.assertEqual(kept, [(2, "A", "C"), (4, "C", "T")])
        self.assertEqual(dropped, 3)

    def test_empty_input(self) -> None:
        """An empty input yields an empty kept list and zero drops."""
        kept, dropped = mp._filter_valid_mutations([])

        self.assertEqual(kept, [])
        self.assertEqual(dropped, 0)

    def test_all_invalid(self) -> None:
        """If every mutation is invalid, kept is empty and dropped counts all."""
        mutations = [(0, "N", "A"), (1, "A", "X"), (2, "Z", "Q")]

        kept, dropped = mp._filter_valid_mutations(mutations)

        self.assertEqual(kept, [])
        self.assertEqual(dropped, 3)


# ---------------------------------------------------------------------------
# _build_mutation_rows
# ---------------------------------------------------------------------------


class TestBuildMutationRows(unittest.TestCase):
    """Tests for mp._build_mutation_rows."""

    def test_returns_long_format_dicts(self) -> None:
        """Each tuple becomes a dict with the expected keys and the given gene_id."""
        mutations = [(0, "A", "C"), (5, "G", "T")]

        rows = mp._build_mutation_rows("gene_a", mutations)

        self.assertEqual(rows, [
            {"gene_id": "gene_a", "position": 0, "source_base": "A", "new_base": "C"},
            {"gene_id": "gene_a", "position": 5, "source_base": "G", "new_base": "T"},
        ])

    def test_empty_input_returns_empty_list(self) -> None:
        """An empty mutations list yields an empty row list."""
        rows = mp._build_mutation_rows("gene_a", [])

        self.assertEqual(rows, [])


# ---------------------------------------------------------------------------
# _build_gene_stats_row
# ---------------------------------------------------------------------------


class TestBuildGeneStatsRow(unittest.TestCase):
    """Tests for mp._build_gene_stats_row."""

    def test_populates_all_columns(self) -> None:
        """The returned dict matches GENE_STATS_COLUMNS with derived values."""
        reference = "ACGT" * 25  # length 100
        gene = _make_gene(reference,{1999: [
            _make_sequence(reference, [], fitness=0.2),
            _make_sequence(reference, [(0, "A", "T"), (4, "A", "G")], fitness=0.7),
        ]})

        row = mp._build_gene_stats_row("gene_a", gene, n_mutations=2, generation=1999)

        self.assertEqual(row["gene_id"], "gene_a")
        self.assertEqual(row["n_mutations"], 2)
        self.assertAlmostEqual(row["initial_fitness"], 0.2)         #type: ignore
        self.assertAlmostEqual(row["final_fitness"], 0.7)           #type: ignore

    def test_uses_provided_n_mutations(self) -> None:
        """``n_mutations`` is taken from the argument, not recomputed from the gene."""
        reference = "ACGT" * 10
        gene = _make_gene(reference,{1999: [_make_sequence(reference, [(0, "A", "T"), (1, "C", "G")], fitness=0.5)]})
        row = mp._build_gene_stats_row("gene_a", gene, n_mutations=7, generation=1999)
        self.assertEqual(row["n_mutations"], 7)


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip(unittest.TestCase):
    """Tests for MutationPool.save and MutationPool.load."""

    def test_roundtrip(self) -> None:
        """Saving a pool and reloading reproduces the original tables and refs."""
        reference = "ACGT" * 10
        gene = _make_gene(reference, {1999: [_make_sequence(reference, [(0, "A", "T"), (4, "A", "G")], fitness=0.9,)]})

        with mock.patch.object(
            mp, "load_mutations_from_json", return_value={"gene_a": gene}
        ):
            pool = MutationPool.from_summarized_json("ignored.json", generation=1999)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/pool.json"
            pool.save(output_path)
            reloaded = MutationPool.load(output_path)

        pdt.assert_frame_equal(
            reloaded.mutations.reset_index(drop=True),
            pool.mutations.reset_index(drop=True),
        )
        pdt.assert_frame_equal(
            reloaded.gene_stats.reset_index(drop=True),
            pool.gene_stats.reset_index(drop=True),
        )
        self.assertEqual(reloaded.references, pool.references)


if __name__ == "__main__":
    unittest.main()
