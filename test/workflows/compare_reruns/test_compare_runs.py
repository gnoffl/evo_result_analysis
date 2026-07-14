"""Unit tests for the calculation functions of ``compare_runs``.

Plotting functions are intentionally not tested (per project convention: calculation
and rendering are kept separate, only calculation is unit-tested).
"""

import json
import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")  # headless backend for tests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workflows.compare_reruns.compare_runs import (
    plot_paired_scatter,
    plot_mutation_overlap,
    PER_GENE_COLUMNS,
    GeneComparisonSkipped,
    _paired_wilcoxon_pvalue,
    build_per_gene_table,
    compare_single_gene,
    compute_summary,
    discover_gene_fronts,
    extract_gene_key,
    format_summary,
    mutation_set,
    resolve_comparison_members,
)


def make_front(after_sequence, after_fitness, after_n_mutations,
               reference_sequence, before_fitness):
    """Build a minimal two-member Pareto front (after endpoint, reference)."""
    return [
        [after_sequence, after_fitness, after_n_mutations],
        [reference_sequence, before_fitness, 0.0],
    ]


class TestExtractGeneKey(unittest.TestCase):
    def test_extracts_chromosome_and_agi(self):
        # Arrange
        folder_name = "1_AT1G75760_gene:28448735-28446631_260224_163252_182620"
        # Act
        gene_key = extract_gene_key(folder_name)
        # Assert
        self.assertEqual(gene_key, "1_AT1G75760")

    def test_handles_organelle_prefix(self):
        self.assertEqual(
            extract_gene_key("Pt_ATCG00670_gene:71882-69909_260225_060145_367708"),
            "Pt_ATCG00670",
        )


class TestMutationSet(unittest.TestCase):
    def test_returns_position_and_base_of_substitutions(self):
        # Arrange
        reference_sequence = "AAAA"
        optimized_sequence = "AGTA"
        # Act
        mutations = mutation_set(reference_sequence, optimized_sequence)
        # Assert
        self.assertEqual(mutations, {(1, "G"), (2, "T")})

    def test_no_mutations_returns_empty_set(self):
        self.assertEqual(mutation_set("AAAA", "AAAA"), set())

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mutation_set("AAAA", "AAA")

    def test_position_only_returns_positions(self):
        # Act
        mutations = mutation_set("AAAA", "AGTA", position_only=True)
        # Assert
        self.assertEqual(mutations, {1, 2})

    def test_same_position_different_base_shared_only_in_position_only(self):
        # Arrange: both sequences mutate position 1, but to different bases.
        reference_sequence = "AAAA"
        optimized_a = "AGAA"
        optimized_b = "ACAA"
        # Act
        position_base_a = mutation_set(reference_sequence, optimized_a)
        position_base_b = mutation_set(reference_sequence, optimized_b)
        position_only_a = mutation_set(reference_sequence, optimized_a, position_only=True)
        position_only_b = mutation_set(reference_sequence, optimized_b, position_only=True)
        # Assert: position + base sees them as distinct, position-only as shared.
        self.assertEqual(position_base_a & position_base_b, set())
        self.assertEqual(position_only_a & position_only_b, {1})


class TestCompareSingleGene(unittest.TestCase):
    def setUp(self):
        self.reference_sequence = "AAAA"
        # Both endpoints have 2 mutations, so the common endpoint count is 2 and the
        # overlap members coincide with the endpoints.
        self.front_a = make_front("AGTA", 1.0, 2.0, self.reference_sequence, 0.9)
        self.front_b = make_front("AGCA", 0.95, 2.0, self.reference_sequence, 0.9)

    def test_happy_path_record(self):
        # Act
        record = compare_single_gene("g1", self.front_a, self.front_b)
        # Assert
        self.assertEqual(record["gene"], "g1")
        self.assertEqual(record["before"], 0.9)
        self.assertEqual(record["after_A"], 1.0)
        self.assertEqual(record["after_B"], 0.95)
        self.assertEqual(record["n_mutations_A"], 2.0)
        self.assertEqual(record["n_mutations_B"], 2.0)
        self.assertAlmostEqual(record["delta_after"], -0.05)
        self.assertEqual(record["delta_n_mutations"], 0.0)
        self.assertEqual(record["overlap_mutation_count"], 2)
        # shared exact = {(1,'G')}; a_only = {(2,'T')}; b_only = {(2,'C')}
        self.assertEqual(record["shared_mutations"], 1)
        self.assertEqual(record["a_only_mutations"], 1)
        self.assertEqual(record["b_only_mutations"], 1)

    def test_record_has_expected_columns(self):
        record = compare_single_gene("g1", self.front_a, self.front_b)
        self.assertEqual(list(record.keys()), PER_GENE_COLUMNS)

    def test_reference_sequence_mismatch_raises(self):
        # Arrange: run B has a different reference sequence
        front_b = make_front("AGCA", 0.95, 2.0, "AATA", 0.9)
        # Act / Assert
        with self.assertRaises(ValueError):
            compare_single_gene("g1", self.front_a, front_b)

    def test_before_fitness_mismatch_raises(self):
        # Arrange: same reference sequence, different before-fitness
        front_b = make_front("AGCA", 0.95, 2.0, self.reference_sequence, 0.8)
        with self.assertRaises(ValueError):
            compare_single_gene("g1", self.front_a, front_b)

    def test_before_fitness_within_tolerance_ok(self):
        front_b = make_front("AGCA", 0.95, 2.0, self.reference_sequence, 0.9 + 1e-9)
        record = compare_single_gene("g1", self.front_a, front_b)
        self.assertEqual(record["gene"], "g1")

    def test_endpoint_overlap_at_common_count_with_different_max_counts(self):
        # Arrange: run A reaches 4 mutations, run B only 2. The reported after/n
        # should come from each run's true endpoint (front[0]), while the mutation
        # overlap is computed at the common count min(4, 2) = 2.
        front_a = [
            ["GGGG", 1.0, 4.0],
            ["GGGA", 0.9, 3.0],
            ["GGAA", 0.8, 2.0],  # overlap member A at count 2
            ["GAAA", 0.7, 1.0],
            ["AAAA", 0.5, 0.0],
        ]
        front_b = [
            ["GCAA", 0.95, 2.0],  # endpoint B and overlap member B at count 2
            ["GAAA", 0.8, 1.0],
            ["AAAA", 0.5, 0.0],
        ]
        # Act
        record = compare_single_gene("g1", front_a, front_b, comparison_point="endpoint")
        # Assert: after/n_mutations reflect the true endpoints ...
        self.assertEqual(record["after_A"], 1.0)
        self.assertEqual(record["after_B"], 0.95)
        self.assertEqual(record["n_mutations_A"], 4.0)
        self.assertEqual(record["n_mutations_B"], 2.0)
        self.assertEqual(record["delta_n_mutations"], -2.0)
        # ... while the overlap is taken at the common count 2:
        self.assertEqual(record["overlap_mutation_count"], 2)
        # A@2 = "GGAA" -> {(0,'G'),(1,'G')}; B@2 = "GCAA" -> {(0,'G'),(1,'C')}
        self.assertEqual(record["shared_mutations"], 1)
        self.assertEqual(record["a_only_mutations"], 1)
        self.assertEqual(record["b_only_mutations"], 1)

    def test_half_max_reports_at_common_count_with_zero_delta(self):
        # Arrange: fronts whose half-max mutation count is 2 for both runs.
        front_a = [
            ["GGGG", 1.0, 4.0],
            ["GGGA", 0.8, 3.0],
            ["GGAA", 0.6, 2.0],  # half-max (fitness 0.6) and overlap member A
            ["GAAA", 0.4, 1.0],
            ["AAAA", 0.2, 0.0],
        ]
        front_b = [
            ["CCCA", 1.0, 3.0],
            ["GCAA", 0.7, 2.0],  # half-max reached at count 2 and overlap member B
            ["GAAA", 0.5, 1.0],
            ["AAAA", 0.2, 0.0],
        ]
        # Act
        record = compare_single_gene("g1", front_a, front_b, comparison_point="half_max")
        # Assert: after/n_mutations come from the common count 2 ...
        self.assertEqual(record["overlap_mutation_count"], 2)
        self.assertEqual(record["after_A"], 0.6)
        self.assertEqual(record["after_B"], 0.7)
        self.assertEqual(record["n_mutations_A"], 2.0)
        self.assertEqual(record["n_mutations_B"], 2.0)
        self.assertEqual(record["delta_n_mutations"], 0.0)
        # A@2 = "GGAA" -> {(0,'G'),(1,'G')}; B@2 = "GCAA" -> {(0,'G'),(1,'C')}
        self.assertEqual(record["shared_mutations"], 1)
        self.assertEqual(record["a_only_mutations"], 1)
        self.assertEqual(record["b_only_mutations"], 1)

    def test_position_only_matches_same_site_different_base(self):
        # front_a mutates pos1 -> G, front_b mutates pos1 -> C; same site.
        front_a = make_front("AGAA", 1.0, 1.0, self.reference_sequence, 0.9)
        front_b = make_front("ACAA", 0.95, 1.0, self.reference_sequence, 0.9)
        # position + base: no shared mutation
        record_exact = compare_single_gene("g1", front_a, front_b, position_only=False)
        self.assertEqual(record_exact["shared_mutations"], 0)
        # position only: the shared site counts as shared
        record_position = compare_single_gene("g1", front_a, front_b, position_only=True)
        self.assertEqual(record_position["shared_mutations"], 1)
        self.assertEqual(record_position["a_only_mutations"], 0)
        self.assertEqual(record_position["b_only_mutations"], 0)

    def test_endpoint_uses_largest_shared_count_across_gaps(self):
        # Arrange: front A has a gap (counts 5, 3, 0 -> no 4), front B is denser
        # (4, 3, 2, 1, 0). The endpoints are 5 and 4, which are not shared; the
        # largest count present on BOTH fronts is 3. The gene must NOT be skipped.
        front_a = [
            ["GGGGG", 1.0, 5.0],
            ["GGGAA", 0.8, 3.0],  # overlap member A at count 3
            ["AAAAA", 0.3, 0.0],
        ]
        front_b = [
            ["GGGGA", 0.95, 4.0],
            ["GGGAA", 0.9, 3.0],  # overlap member B at count 3
            ["GGAAA", 0.8, 2.0],
            ["GAAAA", 0.6, 1.0],
            ["AAAAA", 0.3, 0.0],
        ]
        # Act
        record = compare_single_gene("g1", front_a, front_b, comparison_point="endpoint")
        # Assert: reported after/n from the true endpoints ...
        self.assertEqual(record["after_A"], 1.0)
        self.assertEqual(record["after_B"], 0.95)
        self.assertEqual(record["n_mutations_A"], 5.0)
        self.assertEqual(record["n_mutations_B"], 4.0)
        # ... overlap taken at the largest shared count 3 (both "GGGAA")
        self.assertEqual(record["overlap_mutation_count"], 3)
        self.assertEqual(record["shared_mutations"], 3)
        self.assertEqual(record["a_only_mutations"], 0)
        self.assertEqual(record["b_only_mutations"], 0)

    def test_half_max_missing_common_count_raises_skipped(self):
        # Arrange: both half-max counts are computable (A -> 2, B -> 3), so the
        # common count is min(2, 3) = 2, but front B skips count 2 (gap 1 -> 3),
        # so the gene must be skipped in half_max mode.
        front_a = [
            ["GGGGAA", 1.0, 4.0],
            ["GGGAAA", 0.8, 3.0],
            ["GGAAAA", 0.6, 2.0],  # half-max (fitness 0.6) -> count 2
            ["GAAAAA", 0.4, 1.0],
            ["AAAAAA", 0.2, 0.0],
        ]
        front_b = [
            ["GGGGGG", 1.0, 6.0],
            ["GGGGGA", 0.9, 5.0],
            ["GGGGAA", 0.8, 4.0],
            ["GGGAAA", 0.7, 3.0],  # half-max (fitness 0.6) -> count 3
            ["GAAAAA", 0.3, 1.0],  # gap: no count 2
            ["AAAAAA", 0.2, 0.0],
        ]
        # Act / Assert
        with self.assertRaises(GeneComparisonSkipped):
            compare_single_gene("g1", front_a, front_b, comparison_point="half_max")


class TestResolveComparisonMembers(unittest.TestCase):
    def test_rejects_unknown_comparison_point(self):
        front = make_front("AGTA", 1.0, 2.0, "AAAA", 0.9)
        with self.assertRaises(ValueError):
            resolve_comparison_members(front, front, "not_a_point")


class TestBuildPerGeneTable(unittest.TestCase):
    def setUp(self):
        reference_sequence = "AAAA"
        self.fronts_a = {
            "g1": make_front("AGTA", 1.0, 2.0, reference_sequence, 0.9),
            "g2": make_front("AGAA", 1.0, 1.0, reference_sequence, 0.9),
        }
        self.fronts_b = {
            "g1": make_front("AGCA", 0.95, 2.0, reference_sequence, 0.9),
            "g3": make_front("AGAA", 1.0, 1.0, reference_sequence, 0.9),
        }

    def test_overlap_counts_and_columns(self):
        # Act
        per_gene, overlap_counts, excluded_genes = build_per_gene_table(
            self.fronts_a, self.fronts_b
        )
        # Assert
        self.assertEqual(overlap_counts, {"both": 1, "a_only": 1, "b_only": 1})
        self.assertEqual(list(per_gene.columns), PER_GENE_COLUMNS)
        self.assertIn("overlap_mutation_count", per_gene.columns)
        self.assertEqual(len(per_gene), 1)
        self.assertEqual(per_gene.iloc[0]["gene"], "g1")
        self.assertEqual(excluded_genes, [])

    def test_no_overlap_raises(self):
        with self.assertRaises(ValueError):
            build_per_gene_table({"g1": self.fronts_a["g1"]}, {"g9": self.fronts_b["g3"]})

    def test_skipped_gene_recorded_and_absent_from_table(self):
        # Arrange (half_max mode): g1 is comparable; g2's common half-max count is
        # absent on front B (gap), so it must be skipped and recorded.
        fronts_a = {
            "g1": make_front("AGTAA", 1.0, 2.0, "AAAAA", 0.9),
            "g2": [
                ["GGGGAA", 1.0, 4.0],
                ["GGGAAA", 0.8, 3.0],
                ["GGAAAA", 0.6, 2.0],  # half-max -> count 2
                ["GAAAAA", 0.4, 1.0],
                ["AAAAAA", 0.2, 0.0],
            ],
        }
        fronts_b = {
            "g1": make_front("AGCAA", 0.95, 2.0, "AAAAA", 0.9),
            "g2": [
                ["GGGGGG", 1.0, 6.0],
                ["GGGGGA", 0.9, 5.0],
                ["GGGGAA", 0.8, 4.0],
                ["GGGAAA", 0.7, 3.0],  # half-max -> count 3
                ["GAAAAA", 0.3, 1.0],  # gap: no count 2
                ["AAAAAA", 0.2, 0.0],
            ],
        }
        # Act
        per_gene, _, excluded_genes = build_per_gene_table(
            fronts_a, fronts_b, comparison_point="half_max"
        )
        # Assert
        self.assertEqual(list(per_gene["gene"]), ["g1"])
        self.assertEqual(len(excluded_genes), 1)
        skipped_gene, reason = excluded_genes[0]
        self.assertEqual(skipped_gene, "g2")
        self.assertIn("common count", reason)


class TestPairedWilcoxonPvalue(unittest.TestCase):
    def test_returns_pvalue_for_differing_series(self):
        # Arrange
        values_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        values_b = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        # Act
        p_value = _paired_wilcoxon_pvalue(values_a, values_b)
        # Assert
        self.assertIsNotNone(p_value)
        self.assertTrue(0.0 <= p_value <= 1.0)

    def test_all_equal_returns_none(self):
        values = np.array([1.0, 2.0, 3.0])
        self.assertIsNone(_paired_wilcoxon_pvalue(values, values.copy()))


class TestComputeSummary(unittest.TestCase):
    def setUp(self):
        self.per_gene = pd.DataFrame(
            [
                {
                    "gene": "g1", "before": 0.9, "after_A": 1.0, "after_B": 0.95,
                    "n_mutations_A": 2.0, "n_mutations_B": 3.0,
                    "delta_after": -0.05, "delta_n_mutations": 1.0,
                    "overlap_mutation_count": 2,
                    "shared_mutations": 1, "a_only_mutations": 1, "b_only_mutations": 2,
                },
                {
                    "gene": "g2", "before": 0.8, "after_A": 0.9, "after_B": 0.9,
                    "n_mutations_A": 4.0, "n_mutations_B": 2.0,
                    "delta_after": 0.0, "delta_n_mutations": -2.0,
                    "overlap_mutation_count": 4,
                    "shared_mutations": 2, "a_only_mutations": 0, "b_only_mutations": 0,
                },
            ],
            columns=PER_GENE_COLUMNS,
        )
        self.overlap_counts = {"both": 2, "a_only": 0, "b_only": 0}

    def test_aggregates(self):
        # Act
        summary = compute_summary(self.per_gene, self.overlap_counts, "run_a", "run_b")
        # Assert
        self.assertEqual(summary["n_genes_compared"], 2)
        self.assertAlmostEqual(summary["mean_after_a"], 0.95)
        self.assertAlmostEqual(summary["mean_after_b"], 0.925)
        self.assertAlmostEqual(summary["mean_delta_after"], -0.025)
        self.assertAlmostEqual(summary["mean_abs_delta_after"], 0.025)
        self.assertAlmostEqual(summary["mean_delta_n_mutations"], -0.5)
        self.assertAlmostEqual(summary["mean_abs_delta_n_mutations"], 1.5)
        self.assertEqual(summary["pooled_shared_mutations"], 3)
        self.assertEqual(summary["pooled_a_only_mutations"], 1)
        self.assertEqual(summary["pooled_b_only_mutations"], 2)

    def test_mean_shared_fraction(self):
        # g1: 1/(1+1+2)=0.25 ; g2: 2/(2+0+0)=1.0 ; mean = 0.625
        summary = compute_summary(self.per_gene, self.overlap_counts, "a", "b")
        self.assertAlmostEqual(summary["mean_shared_fraction"], 0.625)

    def test_threads_comparison_point_match_mode_and_skips(self):
        # Act
        summary = compute_summary(
            self.per_gene,
            self.overlap_counts,
            "a",
            "b",
            comparison_point="half_max",
            position_only=True,
            n_excluded=3,
        )
        # Assert
        self.assertEqual(summary["comparison_point"], "half_max")
        self.assertTrue(summary["position_only"])
        self.assertEqual(summary["n_excluded"], 3)
        # mean overlap count over {2, 4}
        self.assertAlmostEqual(summary["mean_overlap_mutation_count"], 3.0)

    def test_shared_fraction_ignores_genes_without_mutations(self):
        # Arrange: a gene with zero mutations should be excluded from the fraction mean
        per_gene = pd.DataFrame(
            [
                {
                    "gene": "g1", "before": 0.9, "after_A": 1.0, "after_B": 1.0,
                    "n_mutations_A": 0.0, "n_mutations_B": 0.0,
                    "delta_after": 0.0, "delta_n_mutations": 0.0,
                    "overlap_mutation_count": 0,
                    "shared_mutations": 0, "a_only_mutations": 0, "b_only_mutations": 0,
                },
                {
                    "gene": "g2", "before": 0.8, "after_A": 0.9, "after_B": 0.9,
                    "n_mutations_A": 2.0, "n_mutations_B": 2.0,
                    "delta_after": 0.0, "delta_n_mutations": 0.0,
                    "overlap_mutation_count": 2,
                    "shared_mutations": 2, "a_only_mutations": 1, "b_only_mutations": 1,
                },
            ],
            columns=PER_GENE_COLUMNS,
        )
        summary = compute_summary(per_gene, {"both": 2, "a_only": 0, "b_only": 0}, "a", "b")
        # Only g2 contributes: fraction 1.0
        self.assertAlmostEqual(summary["mean_shared_fraction"], 0.5)


class TestFormatSummary(unittest.TestCase):
    def _base_summary(self):
        return {
            "label_a": "run_a", "label_b": "run_b",
            "comparison_point": "endpoint", "position_only": False,
            "n_excluded": 0, "mean_overlap_mutation_count": 2.0,
            "overlap_counts": {"both": 1, "a_only": 0, "b_only": 0},
            "n_genes_compared": 1,
            "mean_before": 0.9, "mean_after_a": 1.0, "mean_after_b": 0.95,
            "median_after_a": 1.0, "median_after_b": 0.95,
            "mean_n_mutations_a": 2.0, "mean_n_mutations_b": 2.0,
            "mean_delta_after": -0.05, "mean_abs_delta_after": 0.05,
            "mean_delta_n_mutations": 0.0, "mean_abs_delta_n_mutations": 0.0,
            "wilcoxon_p_after": None, "wilcoxon_p_n_mutations": 0.5,
            "pooled_shared_mutations": 1, "pooled_a_only_mutations": 1,
            "pooled_b_only_mutations": 1, "mean_shared_fraction": 0.33,
        }

    def test_contains_labels_and_handles_undefined_pvalue(self):
        # Arrange
        summary = self._base_summary()
        # Act
        text = format_summary(summary)
        # Assert
        self.assertIn("run_a", text)
        self.assertIn("run_b", text)
        self.assertIn("undefined", text)

    def test_reports_comparison_point_match_mode_and_skips(self):
        # Arrange
        summary = self._base_summary()
        summary["comparison_point"] = "half_max"
        summary["position_only"] = True
        summary["n_excluded"] = 4
        # Act
        text = format_summary(summary)
        # Assert
        self.assertIn("half_max", text)
        self.assertIn("half-max point", text)
        self.assertIn("position only", text)
        self.assertIn("4", text)


class TestDiscoverGeneFronts(unittest.TestCase):
    def _write_front(self, run_dir, folder_name):
        gene_dir = os.path.join(run_dir, folder_name, "saved_populations")
        os.makedirs(gene_dir)
        with open(os.path.join(gene_dir, "pareto_front.json"), "w") as front_file:
            json.dump([["AGTA", 1.0, 2.0], ["AAAA", 0.9, 0.0]], front_file)

    def test_discovers_gene_folders(self):
        with tempfile.TemporaryDirectory() as run_dir:
            # Arrange
            self._write_front(run_dir, "1_AT1G75760_gene:1-2_t1")
            self._write_front(run_dir, "2_AT2G30933_gene:3-4_t2")
            # a stray file and a folder without a pareto front should be ignored
            with open(os.path.join(run_dir, "notes.txt"), "w") as stray:
                stray.write("ignore me")
            os.makedirs(os.path.join(run_dir, "3_AT3G00000_gene:5-6_t3"))
            # Act
            gene_fronts = discover_gene_fronts(run_dir)
            # Assert
            self.assertEqual(set(gene_fronts), {"1_AT1G75760", "2_AT2G30933"})

    def test_missing_run_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            discover_gene_fronts("/nonexistent/run/folder")

    def test_duplicate_gene_key_raises(self):
        with tempfile.TemporaryDirectory() as run_dir:
            self._write_front(run_dir, "1_AT1G75760_gene:1-2_t1")
            self._write_front(run_dir, "1_AT1G75760_gene:1-2_t2")
            with self.assertRaises(ValueError):
                discover_gene_fronts(run_dir)


class TestPlotAxInjectionContract(unittest.TestCase):
    """Ax-injection return contract of the Figure-returning plot functions.

    Rendering itself is not asserted (per this module's convention); these check
    the logic contract that, when an ``ax`` is passed, the function draws onto it
    and returns ``ax.get_figure()`` rather than creating a new standalone figure.
    """

    def test_plot_paired_scatter_returns_injected_axes_figure(self):
        per_gene = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.15, 0.25, 0.35]})
        fig, ax = plt.subplots()
        try:
            returned = plot_paired_scatter(per_gene, "a", "b", "x", "y", "title", ax=ax)
            self.assertIs(returned, ax.get_figure())
            self.assertGreater(len(ax.collections), 0)  # scatter drawn onto ax
        finally:
            plt.close(fig)

    def test_plot_mutation_overlap_returns_injected_axes_figure(self):
        per_gene = pd.DataFrame(
            {
                "shared_mutations": [2, 3, 1],
                "a_only_mutations": [1, 0, 2],
                "b_only_mutations": [0, 1, 1],
            }
        )
        fig, ax = plt.subplots()
        try:
            returned = plot_mutation_overlap(per_gene, ax=ax)
            self.assertIs(returned, ax.get_figure())
            self.assertGreater(len(ax.patches), 0)  # histogram drawn onto ax
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
