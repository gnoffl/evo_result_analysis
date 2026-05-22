"""Unit tests for the MutationPool class (Step 1 of mutation_distribution_analysis)."""

from __future__ import annotations

from typing import Dict, List, Tuple
from unittest import mock

import pandas as pd
import pandas.testing as pdt
import pytest

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


def test_from_summarized_json_selects_highest_fitness():
    """The highest-fitness pareto-front sequence is picked for each gene."""
    # Arrange
    reference = "ACGT" * 10
    low = _make_sequence(reference, [(0, "A", "T")], fitness=0.10)
    mid = _make_sequence(reference, [(0, "A", "T"), (5, "C", "G")], fitness=0.50)
    high = _make_sequence(
        reference,
        [(0, "A", "T"), (5, "C", "G"), (9, "T", "A")],
        fitness=0.99,
    )
    gene = _make_gene(reference, {1999: [low, mid, high]})

    with mock.patch.object(
        mp, "load_mutations_from_json", return_value={"gene_a": gene}
    ):
        # Act
        pool = MutationPool.from_summarized_json("ignored.json", generation=1999)

    # Assert
    assert len(pool.mutations) == 3
    assert set(pool.mutations["position"]) == {0, 5, 9}
    assert pool.gene_stats.loc[0, "n_mutations"] == 3
    assert pool.gene_stats.loc[0, "initial_fitness"] == pytest.approx(0.10)
    assert pool.gene_stats.loc[0, "final_fitness"] == pytest.approx(0.99)


def test_filters_non_acgt_mutations():
    """Mutations with non-ACGT source or new bases are dropped."""
    # Arrange
    reference = "ACGT" * 10
    seq = _make_sequence(
        reference,
        [
            (1, "N", "A"),   # invalid source -> drop
            (2, "A", "N"),   # invalid new -> drop
            (3, "A", "C"),   # valid -> keep
            (4, "C", "T"),   # valid -> keep
        ],
        fitness=0.8,
    )
    gene = _make_gene(reference, {1999: [seq]})

    with mock.patch.object(
        mp, "load_mutations_from_json", return_value={"gene_a": gene}
    ):
        # Act
        pool = MutationPool.from_summarized_json("ignored.json", generation=1999)

    # Assert
    assert len(pool.mutations) == 2
    kept_positions = sorted(pool.mutations["position"].tolist())
    assert kept_positions == [3, 4]
    assert set(pool.mutations["source_base"]).issubset({"A", "C", "G", "T"})
    assert set(pool.mutations["new_base"]).issubset({"A", "C", "G", "T"})
    # gene_stats should reflect kept mutations only
    assert pool.gene_stats.loc[0, "n_mutations"] == 2


def test_gene_stats_derived_correctly():
    """gene_stats fields match the underlying mutation list and reference."""
    # Arrange
    reference_a = "ACGT" * 25  # length 100
    reference_b = "TGCA" * 50  # length 200

    gene_a = _make_gene(
        reference_a,
        {
            1999: [
                _make_sequence(reference_a, [(0, "A", "T")], fitness=0.2),
                _make_sequence(
                    reference_a,
                    [(0, "A", "T"), (4, "A", "G")],
                    fitness=0.7,
                ),
            ]
        },
    )
    gene_b = _make_gene(
        reference_b,
        {1999: [_make_sequence(reference_b, [(1, "G", "A")], fitness=0.4)]},
    )

    with mock.patch.object(
        mp,
        "load_mutations_from_json",
        return_value={"gene_a": gene_a, "gene_b": gene_b},
    ):
        # Act
        pool = MutationPool.from_summarized_json("ignored.json", generation=1999)

    # Assert
    stats_by_gene = pool.gene_stats.set_index("gene_id")

    assert stats_by_gene.loc["gene_a", "n_mutations"] == 2
    assert stats_by_gene.loc["gene_a", "initial_fitness"] == pytest.approx(0.2)
    assert stats_by_gene.loc["gene_a", "final_fitness"] == pytest.approx(0.7)

    assert stats_by_gene.loc["gene_b", "n_mutations"] == 1
    assert stats_by_gene.loc["gene_b", "initial_fitness"] == pytest.approx(0.4)
    assert stats_by_gene.loc["gene_b", "final_fitness"] == pytest.approx(0.4)


def test_mutations_dataframe_shape_and_columns():
    """mutations DataFrame has expected columns and total row count."""
    # Arrange
    reference = "ACGT" * 10
    gene_a = _make_gene(
        reference,
        {
            1999: [
                _make_sequence(
                    reference,
                    [(0, "A", "T"), (4, "A", "G"), (8, "A", "C")],
                    fitness=0.9,
                )
            ]
        },
    )
    gene_b = _make_gene(
        reference,
        {
            1999: [
                _make_sequence(
                    reference,
                    [(1, "C", "G"), (5, "C", "A")],
                    fitness=0.8,
                )
            ]
        },
    )

    with mock.patch.object(
        mp,
        "load_mutations_from_json",
        return_value={"gene_a": gene_a, "gene_b": gene_b},
    ):
        # Act
        pool = MutationPool.from_summarized_json("ignored.json", generation=1999)

    # Assert
    assert list(pool.mutations.columns) == [
        "gene_id",
        "position",
        "source_base",
        "new_base",
    ]
    assert len(pool.mutations) == 5  # 3 + 2
    per_gene = pool.mutations.groupby("gene_id").size().to_dict()
    assert per_gene == {"gene_a": 3, "gene_b": 2}


# ---------------------------------------------------------------------------
# _select_best_sequence
# ---------------------------------------------------------------------------


def test_select_best_sequence_picks_max_fitness():
    """The sequence with the highest fitness is returned."""
    # Arrange
    reference = "ACGT" * 10
    low = _make_sequence(reference, [(0, "A", "T")], fitness=0.10)
    mid = _make_sequence(reference, [(1, "C", "G")], fitness=0.50)
    high = _make_sequence(reference, [(2, "G", "A")], fitness=0.99)
    gene = _make_gene(reference, {1999: [low, mid, high]})

    # Act
    best = mp._select_best_sequence("gene_a", gene, 1999)

    # Assert
    assert best is high


def test_select_best_sequence_raises_when_generation_missing():
    """Missing generation key raises ValueError mentioning gene and generation."""
    # Arrange
    gene = _make_gene("ACGT" * 10, {1999: []})

    # Act / Assert
    with pytest.raises(ValueError, match="gene_a.*42"):
        mp._select_best_sequence("gene_a", gene, 42)


def test_select_best_sequence_raises_when_generation_empty():
    """Empty sequence list for the requested generation raises ValueError."""
    # Arrange
    gene = _make_gene("ACGT" * 10, {1999: []})

    # Act / Assert
    with pytest.raises(ValueError, match="gene_a.*1999"):
        mp._select_best_sequence("gene_a", gene, 1999)


# ---------------------------------------------------------------------------
# _filter_valid_mutations
# ---------------------------------------------------------------------------


def test_filter_valid_mutations_keeps_all_acgt():
    """All mutations with ACGT source and new bases are kept."""
    # Arrange
    mutations = [(0, "A", "C"), (1, "G", "T"), (2, "C", "A")]

    # Act
    kept, dropped = mp._filter_valid_mutations(mutations)

    # Assert
    assert kept == mutations
    assert dropped == 0


def test_filter_valid_mutations_partitions_mixed_input():
    """Non-ACGT source or new base entries are dropped; rest is kept in order."""
    # Arrange
    mutations = [
        (0, "N", "A"),   # invalid source
        (1, "A", "N"),   # invalid new
        (2, "A", "C"),   # valid
        (3, "g", "T"),   # invalid (lowercase)
        (4, "C", "T"),   # valid
    ]

    # Act
    kept, dropped = mp._filter_valid_mutations(mutations)

    # Assert
    assert kept == [(2, "A", "C"), (4, "C", "T")]
    assert dropped == 3


def test_filter_valid_mutations_empty_input():
    """An empty input yields an empty kept list and zero drops."""
    # Act
    kept, dropped = mp._filter_valid_mutations([])

    # Assert
    assert kept == []
    assert dropped == 0


def test_filter_valid_mutations_all_invalid():
    """If every mutation is invalid, kept is empty and dropped counts all."""
    # Arrange
    mutations = [(0, "N", "A"), (1, "A", "X"), (2, "Z", "Q")]

    # Act
    kept, dropped = mp._filter_valid_mutations(mutations)

    # Assert
    assert kept == []
    assert dropped == 3


# ---------------------------------------------------------------------------
# _build_mutation_rows
# ---------------------------------------------------------------------------


def test_build_mutation_rows_returns_long_format_dicts():
    """Each tuple becomes a dict with the expected keys and the given gene_id."""
    # Arrange
    kept = [(0, "A", "C"), (5, "G", "T")]

    # Act
    rows = mp._build_mutation_rows("gene_a", kept)

    # Assert
    assert rows == [
        {"gene_id": "gene_a", "position": 0, "source_base": "A", "new_base": "C"},
        {"gene_id": "gene_a", "position": 5, "source_base": "G", "new_base": "T"},
    ]


def test_build_mutation_rows_empty_input_returns_empty_list():
    """An empty kept list yields an empty row list."""
    # Act
    rows = mp._build_mutation_rows("gene_a", [])

    # Assert
    assert rows == []


# ---------------------------------------------------------------------------
# _build_gene_stats_row
# ---------------------------------------------------------------------------


def test_build_gene_stats_row_populates_all_columns():
    """The returned dict matches GENE_STATS_COLUMNS with derived values."""
    # Arrange
    reference = "ACGT" * 25  # length 100
    gene = _make_gene(
        reference,
        {
            1999: [
                _make_sequence(reference, [(0, "A", "T")], fitness=0.2),
                _make_sequence(
                    reference, [(0, "A", "T"), (4, "A", "G")], fitness=0.7
                ),
            ]
        },
    )

    # Act
    row = mp._build_gene_stats_row("gene_a", gene, n_mutations=2, generation=1999)

    # Assert
    assert row == {
        "gene_id": "gene_a",
        "n_mutations": 2,
        "initial_fitness": pytest.approx(0.2),
        "final_fitness": pytest.approx(0.7),
    }


def test_build_gene_stats_row_uses_provided_n_mutations():
    """``n_mutations`` is taken from the argument, not recomputed from the gene."""
    # Arrange
    reference = "ACGT" * 10
    gene = _make_gene(
        reference,
        {
            1999: [
                _make_sequence(
                    reference, [(0, "A", "T"), (1, "C", "G")], fitness=0.5
                )
            ]
        },
    )

    # Act
    row = mp._build_gene_stats_row("gene_a", gene, n_mutations=7, generation=1999)

    # Assert
    assert row["n_mutations"] == 7


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    """Saving a pool and reloading reproduces the original tables and refs."""
    # Arrange
    reference = "ACGT" * 10
    gene = _make_gene(
        reference,
        {
            1999: [
                _make_sequence(
                    reference,
                    [(0, "A", "T"), (4, "A", "G")],
                    fitness=0.9,
                )
            ]
        },
    )

    with mock.patch.object(
        mp, "load_mutations_from_json", return_value={"gene_a": gene}
    ):
        pool = MutationPool.from_summarized_json("ignored.json", generation=1999)

    output_path = tmp_path / "pool.json"

    # Act
    pool.save(str(output_path))
    reloaded = MutationPool.load(str(output_path))

    # Assert
    pdt.assert_frame_equal(
        reloaded.mutations.reset_index(drop=True),
        pool.mutations.reset_index(drop=True),
    )
    pdt.assert_frame_equal(
        reloaded.gene_stats.reset_index(drop=True),
        pool.gene_stats.reset_index(drop=True),
    )
    assert reloaded.references == pool.references
