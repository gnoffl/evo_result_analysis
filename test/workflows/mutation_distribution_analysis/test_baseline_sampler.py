"""Unit tests for the baseline_sampler module (Steps 2 & 3)."""

from __future__ import annotations

import os
import random
import sys
import tempfile
import unittest
from typing import Dict, List, Tuple
from unittest import mock

import numpy as np
import pandas as pd

from workflows.mutation_distribution_analysis import baseline_sampler as bs
from workflows.mutation_distribution_analysis.baseline_sampler import (
    _apply_mutations,
    _build_gene_sampler,
    _filter_applicable,
    generate_baselines_fasta,
    sample_baseline_sequence,
)
from workflows.mutation_distribution_analysis.mutation_pool import (
    GENE_STATS_COLUMNS,
    MUTATION_COLUMNS,
    MutationPool,
)


def _make_pool(
    mutations_records: List[Dict[str, object]],
    refs: Dict[str, str],
    gene_stats_records: List[Dict[str, object]],
) -> MutationPool:
    """Construct a MutationPool from raw row dicts. Avoids re-running extraction."""
    mutations_df = pd.DataFrame(mutations_records, columns=MUTATION_COLUMNS)
    gene_stats_df = pd.DataFrame(gene_stats_records, columns=GENE_STATS_COLUMNS)
    return MutationPool(
        mutations=mutations_df,
        gene_stats=gene_stats_df,
        references=dict(refs),
    )


def _mut_row(
    gene_id: str, position: int, source_base: str, new_base: str
) -> Dict[str, object]:
    return {
        "gene_id": gene_id,
        "position": position,
        "source_base": source_base,
        "new_base": new_base,
    }


def _stats_row(
    gene_id: str,
    n_mutations: int,
    initial_fitness: float = 0.0,
    final_fitness: float = 1.0,
) -> Dict[str, object]:
    return {
        "gene_id": gene_id,
        "n_mutations": n_mutations,
        "initial_fitness": initial_fitness,
        "final_fitness": final_fitness,
    }


# ---------------------------------------------------------------------------
# _filter_applicable
# ---------------------------------------------------------------------------


class TestFilterApplicable(unittest.TestCase):
    """Tests for bs._filter_applicable."""

    def test_keeps_only_matching_source_base(self) -> None:
        """Rows whose source_base disagrees with the wildtype are dropped."""
        wildtype = "ACGT"
        mutations = pd.DataFrame(
            [
                _mut_row("g1", 0, "A", "T"),  # matches W[0]='A' -> keep
                _mut_row("g2", 1, "G", "T"),  # W[1]='C' -> drop
                _mut_row("g3", 2, "G", "A"),  # matches W[2]='G' -> keep
                _mut_row("g4", 3, "A", "C"),  # W[3]='T' -> drop
            ],
            columns=MUTATION_COLUMNS,
        )

        result = _filter_applicable(mutations, wildtype)

        expected_mutations = pd.DataFrame(
            [
                _mut_row("g1", 0, "A", "T"),  # matches W[0]='A' -> keep
                _mut_row("g3", 2, "G", "A"),  # matches W[2]='G' -> keep
            ],
            columns=MUTATION_COLUMNS,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(sorted(result["position"].tolist()), [0, 2])
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_mutations)

    def test_returns_empty_when_no_matches(self) -> None:
        """No matching rows yields an empty DataFrame with the correct columns."""
        wildtype = "AAAA"
        mutations = pd.DataFrame(
            [
                _mut_row("g1", 0, "C", "T"),
                _mut_row("g2", 1, "G", "T"),
            ],
            columns=MUTATION_COLUMNS,
        )

        result = _filter_applicable(mutations, wildtype)

        self.assertEqual(len(result), 0)
        self.assertEqual(list(result.columns), MUTATION_COLUMNS)

    def test_handles_duplicate_positions(self) -> None:
        """Multiplicity at a position is preserved when all entries match."""
        wildtype = "ACGT"
        mutations = pd.DataFrame(
            [
                _mut_row("g1", 1, "C", "A"),
                _mut_row("g2", 1, "C", "G"),
                _mut_row("g3", 1, "C", "T"),
            ],
            columns=MUTATION_COLUMNS,
        )

        result = _filter_applicable(mutations, wildtype)

        self.assertEqual(len(result), 3)
        self.assertEqual(set(result["new_base"]), {"A", "G", "T"})


# ---------------------------------------------------------------------------
# GeneMutationSampler / _build_gene_sampler
# ---------------------------------------------------------------------------


class TestGeneMutationSampler(unittest.TestCase):
    """Tests for bs._build_gene_sampler and the sampler it returns."""

    @staticmethod
    def _wide_applicable() -> pd.DataFrame:
        """20 rows spanning 20 distinct positions, all source='A', new='T'."""
        return pd.DataFrame(
            [_mut_row(f"g{p}", p, "A", "T") for p in range(20)],
            columns=MUTATION_COLUMNS,
        )

    def test_returns_k_mutations(self) -> None:
        """A seeded draw of k=3 yields exactly 3 mutation tuples."""
        sampler = _build_gene_sampler(self._wide_applicable(), 3)
        rng = np.random.default_rng(0)

        result = sampler.draw(rng)

        self.assertEqual(len(result), 3)

    def test_no_duplicate_positions(self) -> None:
        """Sampled positions are all distinct under position blocking."""
        applicable = pd.DataFrame([_mut_row(f"g{p}", (p // 5) + 1, "A", "T") for p in range(20)], columns=MUTATION_COLUMNS)
        sampler = _build_gene_sampler(applicable, 4)
        rng = np.random.default_rng(random.randint(0, 1000))

        result = sampler.draw(rng)
        positions = [pos for pos, _, _ in result]

        self.assertEqual(len(positions), len(set(positions)))

    def test_deterministic_under_seed(self) -> None:
        """Two calls with identical seeds produce identical sequences of draws."""
        sampler = _build_gene_sampler(self._wide_applicable(), 4)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)

        result_a = sampler.draw(rng_a)
        result_b = sampler.draw(rng_b)

        self.assertEqual(result_a, result_b)

    def test_raises_when_unique_positions_below_k(self) -> None:
        """k larger than the applicable position pool raises ValueError."""
        applicable = pd.DataFrame(
            [
                _mut_row("g1", 0, "A", "T"),
                _mut_row("g2", 1, "A", "G"),
            ],
            columns=MUTATION_COLUMNS,
        )

        with self.assertRaises(ValueError):
            _build_gene_sampler(applicable, 3)

    def test_zero_k_returns_empty_list(self) -> None:
        """k=0 returns [] and does not consume the RNG state."""
        sampler = _build_gene_sampler(self._wide_applicable(), 0)
        rng = np.random.default_rng(7)
        state_before = rng.bit_generator.state

        result = sampler.draw(rng)

        self.assertEqual(result, [])
        self.assertEqual(rng.bit_generator.state, state_before)

    def test_rejects_a_position_with_two_source_bases(self) -> None:
        """A position carrying two source bases signals an upstream bug."""
        applicable = pd.DataFrame(
            [
                _mut_row("g1", 0, "A", "T"),
                _mut_row("g2", 0, "C", "T"),
                _mut_row("g3", 1, "A", "G"),
            ],
            columns=MUTATION_COLUMNS,
        )

        with self.assertRaises(ValueError):
            _build_gene_sampler(applicable, 1)


class TestDrawFactorisation(unittest.TestCase):
    """The factorised draw reproduces the pool's conditional distributions.

    The position marginal is deliberately *not* the pool share — that is the
    point of the fix — but the substitution conditional
    ``P(new_base | position)`` must still be exactly the applicable-row
    frequency at that position, and ``source_base`` must be determined rather
    than drawn.
    """

    @staticmethod
    def _mixed_applicable() -> pd.DataFrame:
        """Two positions, with 3:1 and 1:1 splits between two substitutions."""
        rows = (
            [_mut_row("g1", 0, "A", "T")] * 3
            + [_mut_row("g1", 0, "A", "G")] * 1
            + [_mut_row("g1", 1, "C", "T")] * 2
            + [_mut_row("g1", 1, "C", "G")] * 2
        )
        return pd.DataFrame(rows, columns=MUTATION_COLUMNS)

    def test_source_base_is_uniquely_determined(self) -> None:
        """Each position has one source base, taken without consuming the RNG."""
        applicable = self._mixed_applicable()
        sampler = _build_gene_sampler(applicable, 2)

        self.assertEqual(list(sampler.source_bases), ["A", "C"])

    def test_new_base_frequencies_match_the_applicable_rows(self) -> None:
        """P(new_base | position) equals that substitution's row share."""
        applicable = self._mixed_applicable()
        sampler = _build_gene_sampler(applicable, 2)
        expected = {0: {"T": 0.75, "G": 0.25}, 1: {"T": 0.5, "G": 0.5}}
        n_replicates = 20000
        rng = np.random.default_rng(4242)
        hits: Dict[int, Dict[str, int]] = {
            0: {"T": 0, "G": 0}, 1: {"T": 0, "G": 0}
        }

        for _ in range(n_replicates):
            for position, source_base, new_base in sampler.draw(rng):
                self.assertEqual(source_base, "A" if position == 0 else "C")
                hits[position][new_base] += 1

        for position, expected_shares in expected.items():
            total = sum(hits[position].values())
            self.assertEqual(total, n_replicates)  # k = 2, both always drawn
            for new_base, expected_share in expected_shares.items():
                observed = hits[position][new_base] / total
                # 5 standard errors of a binomial proportion.
                tolerance = 5.0 * (
                    expected_share * (1 - expected_share) / total
                ) ** 0.5
                self.assertAlmostEqual(
                    observed, expected_share, delta=tolerance,
                    msg=f"position {position}, new_base {new_base}",
                )

    def test_position_inclusion_is_proportional_to_row_count(self) -> None:
        """A position appears in a draw with probability ``k * n_i / N``.

        This is the property the replaced successive sampler lacked.
        """
        rows = (
            [_mut_row("g1", 0, "A", "T")] * 40
            + [_mut_row("g1", 1, "A", "G")] * 30
            + [_mut_row("g1", 2, "A", "C")] * 20
            + [_mut_row("g1", 3, "A", "T")] * 6
            + [_mut_row("g1", 4, "A", "G")] * 4
        )
        applicable = pd.DataFrame(rows, columns=MUTATION_COLUMNS)
        sampler = _build_gene_sampler(applicable, 2)
        expected = 2.0 * np.array([40, 30, 20, 6, 4]) / 100.0
        n_replicates = 20000
        rng = np.random.default_rng(99)
        hits = np.zeros(5)

        for _ in range(n_replicates):
            for position, _source_base, _new_base in sampler.draw(rng):
                hits[position] += 1
        observed = hits / n_replicates

        for position, expected_probability in enumerate(expected):
            tolerance = 5.0 * (
                expected_probability
                * (1 - expected_probability)
                / n_replicates
            ) ** 0.5
            self.assertAlmostEqual(
                observed[position], expected_probability, delta=tolerance,
                msg=(
                    f"position {position}: observed {observed[position]:.4f} "
                    f"vs target {expected_probability:.4f}"
                ),
            )


# ---------------------------------------------------------------------------
# _apply_mutations
# ---------------------------------------------------------------------------


class TestApplyMutations(unittest.TestCase):
    """Tests for bs._apply_mutations."""

    def test_replaces_specified_positions_only(self) -> None:
        """Only the listed positions change; others are preserved."""
        wildtype = "AAAA"
        mutations = [(1, "A", "T"), (3, "A", "G")]

        result = _apply_mutations(wildtype, mutations)

        self.assertEqual(result, "ATAG")

    def test_returns_wildtype_when_no_mutations(self) -> None:
        """An empty mutation list yields the wildtype unchanged."""
        wildtype = "ACGTACGT"

        result = _apply_mutations(wildtype, [])

        self.assertEqual(result, wildtype)

    def test_asserts_source_base_matches_wildtype(self) -> None:
        """A mismatching source_base trips the defensive assertion."""
        wildtype = "AAAA"
        mutations = [(0, "C", "T")]  # W[0] is 'A', not 'C'

        with self.assertRaises(AssertionError):
            _apply_mutations(wildtype, mutations)


# ---------------------------------------------------------------------------
# sample_baseline_sequence
# ---------------------------------------------------------------------------


class TestSampleBaselineSequence(unittest.TestCase):
    """Tests for bs.sample_baseline_sequence."""

    def test_end_to_end_on_small_pool(self) -> None:
        """Returned sequence differs at exactly k positions and all changes are in-pool."""
        wildtype = "ACG"
        mutations_records = [
            _mut_row("g1", 0, "A", "T"),
            _mut_row("g1", 1, "C", "G"),
            _mut_row("g1", 2, "G", "A"),
        ]
        pool = _make_pool(
            mutations_records,
            refs={"g1": wildtype},
            gene_stats_records=[_stats_row("g1", n_mutations=2)],
        )
        rng = np.random.default_rng(0)

        result = sample_baseline_sequence(pool, "g1", n_mutations=2, rng=rng)

        diffs = [
            i for i, (w, r) in enumerate(zip(wildtype, result)) if w != r
        ]
        self.assertEqual(len(diffs), 2)
        for i in diffs:
            allowed = pool.mutations[
                (pool.mutations["position"] == i)
                & (pool.mutations["source_base"] == wildtype[i])
            ]["new_base"].tolist()
            self.assertIn(result[i], allowed)

    def test_raises_when_gene_id_missing(self) -> None:
        """Unknown gene_id raises KeyError naming the id."""
        pool = _make_pool(
            mutations_records=[],
            refs={"known_gene": "ACGT"},
            gene_stats_records=[_stats_row("known_gene", 0)],
        )
        rng = np.random.default_rng(0)

        with self.assertRaises(KeyError) as ctx:
            sample_baseline_sequence(pool, "missing_gene", 1, rng)
        self.assertIn("missing_gene", str(ctx.exception))

    def test_raises_when_pool_too_thin_for_k(self) -> None:
        """A pool with fewer unique applicable positions than k raises ValueError."""
        wildtype = "AAAA"
        mutations_records = [
            _mut_row("g1", 0, "A", "T"),
            _mut_row("g1", 1, "A", "G"),
        ]
        pool = _make_pool(
            mutations_records,
            refs={"g1": wildtype},
            gene_stats_records=[_stats_row("g1", n_mutations=4)],
        )
        rng = np.random.default_rng(0)

        with self.assertRaises(ValueError):
            sample_baseline_sequence(pool, "g1", n_mutations=3, rng=rng)


# ---------------------------------------------------------------------------
# generate_baselines_fasta
# ---------------------------------------------------------------------------


def _read_fasta(path: str) -> List[Tuple[str, str]]:
    """Parse a single-line-per-record FASTA into (header, sequence) tuples."""
    records: List[Tuple[str, str]] = []
    with open(path, "r") as f:
        lines = [line.rstrip("\n") for line in f]
    for i in range(0, len(lines), 2):
        header = lines[i]
        sequence = lines[i + 1]
        records.append((header, sequence))
    return records


def _two_gene_pool() -> MutationPool:
    """Two genes, plenty of applicable mutations per gene."""
    wildtype_a = "AAAAA"
    wildtype_b = "CCCCC"
    mutations_records: List[Dict[str, object]] = []
    for position in range(5):
        for new_base in ("C", "G", "T"):
            mutations_records.append(_mut_row("g_a", position, "A", new_base))
        for new_base in ("A", "G", "T"):
            mutations_records.append(_mut_row("g_b", position, "C", new_base))
    return _make_pool(
        mutations_records,
        refs={"g_a": wildtype_a, "g_b": wildtype_b},
        gene_stats_records=[
            _stats_row("g_a", n_mutations=2),
            _stats_row("g_b", n_mutations=3),
        ],
    )


class TestGenerateBaselinesFasta(unittest.TestCase):
    """Tests for bs.generate_baselines_fasta."""

    def test_writes_expected_number_of_records(self) -> None:
        """n_per_gene=3 across 2 genes yields 6 records."""
        pool = _two_gene_pool()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "baselines.fasta")

            generate_baselines_fasta(
                pool, output_path, n_per_gene=3, rng=np.random.default_rng(0)
            )

            records = _read_fasta(output_path)

        self.assertEqual(len(records), 6)

    def test_header_format_zero_padded(self) -> None:
        """Headers are exactly '>{gene_id}_baseline_{i:03d}'."""
        pool = _two_gene_pool()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "baselines.fasta")

            generate_baselines_fasta(
                pool, output_path, n_per_gene=2, rng=np.random.default_rng(0)
            )

            records = _read_fasta(output_path)

        headers = [r[0] for r in records]
        self.assertEqual(
            headers,
            [
                ">g_a_baseline_000",
                ">g_a_baseline_001",
                ">g_b_baseline_000",
                ">g_b_baseline_001",
            ],
        )

    def test_skips_gene_when_pool_too_thin(self) -> None:
        """A gene with too few applicable positions is skipped with a warning."""
        wildtype_a = "AAAA"
        wildtype_b = "GGGG"
        mutations_records = [
            _mut_row("g_a", 0, "A", "T"),
            _mut_row("g_a", 1, "A", "C"),
            _mut_row("g_a", 2, "A", "G"),
            _mut_row("g_a", 3, "A", "T"),
            _mut_row("g_b", 0, "G", "T"),
        ]
        pool = _make_pool(
            mutations_records,
            refs={"g_a": wildtype_a, "g_b": wildtype_b},
            gene_stats_records=[
                _stats_row("g_a", n_mutations=3),
                _stats_row("g_b", n_mutations=5),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "baselines.fasta")

            with mock.patch.object(bs, "print_status") as mock_warn:
                generate_baselines_fasta(
                    pool,
                    output_path,
                    n_per_gene=1,
                    rng=np.random.default_rng(0),
                )

            records = _read_fasta(output_path)

        self.assertEqual(len(records), 1)
        self.assertTrue(records[0][0].startswith(">g_a_baseline_"))
        self.assertTrue(mock_warn.called)
        warn_calls_text = " ".join(
            str(call_args) for call_args in mock_warn.call_args_list
        )
        self.assertIn("g_b", warn_calls_text)

    def test_deterministic_under_seed(self) -> None:
        """Two runs with the same seed produce byte-identical FASTA files."""
        pool = _two_gene_pool()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path_a = os.path.join(tmp_dir, "a.fasta")
            path_b = os.path.join(tmp_dir, "b.fasta")

            generate_baselines_fasta(
                pool, path_a, n_per_gene=2, rng=np.random.default_rng(123)
            )
            generate_baselines_fasta(
                pool, path_b, n_per_gene=2, rng=np.random.default_rng(123)
            )

            with open(path_a, "rb") as f_a, open(path_b, "rb") as f_b:
                self.assertEqual(f_a.read(), f_b.read())

    def test_each_baseline_differs_from_wildtype_at_k_positions(self) -> None:
        """Every emitted baseline differs from its wildtype at exactly k positions."""
        pool = _two_gene_pool()
        k_by_gene = pool.gene_stats.set_index("gene_id")["n_mutations"].to_dict()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "baselines.fasta")

            generate_baselines_fasta(
                pool, output_path, n_per_gene=3, rng=np.random.default_rng(0)
            )

            records = _read_fasta(output_path)

        for header, sequence in records:
            gene_id = header[1:].rsplit("_baseline_", 1)[0]
            wildtype = pool.references[gene_id]
            diffs = sum(1 for w, b in zip(wildtype, sequence) if w != b)
            self.assertEqual(diffs, int(k_by_gene[gene_id]))
            self.assertEqual(len(sequence), len(wildtype))


# ---------------------------------------------------------------------------
# main (CLI)
# ---------------------------------------------------------------------------


class TestCli(unittest.TestCase):
    """Tests for bs.main."""

    def test_cli_invokes_generate_baselines_fasta(self) -> None:
        """main() forwards parsed args + a seeded RNG to generate_baselines_fasta."""
        sentinel_pool = mock.MagicMock(spec=MutationPool)
        sentinel_rng = mock.MagicMock(spec=np.random.Generator)

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "pool.json")
            output_path = os.path.join(tmp_dir, "baselines.fasta")
            with open(input_path, "w") as f:
                f.write("{}")

            argv = [
                "baseline_sampler",
                "--input",
                input_path,
                "--output",
                output_path,
                "--n-per-gene",
                "5",
                "--seed",
                "1234",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                bs.MutationPool, "load", return_value=sentinel_pool
            ) as mock_load, mock.patch.object(
                bs.np.random, "default_rng", return_value=sentinel_rng
            ) as mock_rng, mock.patch.object(
                bs, "generate_baselines_fasta"
            ) as mock_generate:
                bs.main()

        mock_load.assert_called_once_with(input_path)
        mock_rng.assert_called_once_with(1234)
        mock_generate.assert_called_once_with(
            sentinel_pool,
            output_path,
            n_per_gene=5,
            rng=sentinel_rng,
        )


if __name__ == "__main__":
    unittest.main()
