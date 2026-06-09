import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from workflows.overlap_analysis import _common as overlap_analysis


class TestOverlapAnalysisBuckets(unittest.TestCase):
    def test_add_bucketed_overlap_positions_derives_surrogates_and_buckets(self):
        df = pd.DataFrame(
            [
                {"overlap_start": 1500, "overlap_end": 1600},
                {"overlap_start": 2900, "overlap_end": 3020},
                {"overlap_start": 100, "overlap_end": 220},
                {"overlap_start": 0, "overlap_end": 100},
            ]
        )

        result = overlap_analysis.add_length_corrected_overlap_buckets(df)

        self.assertEqual(result.loc[0, "group_overlap_start"], 1430)
        self.assertEqual(result.loc[0, "group_overlap_end"], 1600)
        self.assertEqual(result.loc[0, "group_overlap_length"], 170)
        self.assertEqual(result.loc[0, "overlap_bucket_start"], 1400)
        self.assertEqual(result.loc[0, "overlap_bucket_label"], "1400-1599")

        self.assertEqual(result.loc[1, "group_overlap_start"], 2900)
        self.assertEqual(result.loc[1, "group_overlap_end"], 3070)
        self.assertEqual(result.loc[1, "group_overlap_length"], 170)
        self.assertEqual(result.loc[1, "overlap_bucket_start"], 2800)
        self.assertEqual(result.loc[1, "overlap_bucket_label"], "2800-2999")

        self.assertEqual(result.loc[2, "group_overlap_start"], 100)
        self.assertEqual(result.loc[2, "group_overlap_end"], 220)
        self.assertEqual(result.loc[2, "group_overlap_length"], 120)
        self.assertEqual(result.loc[2, "overlap_bucket_start"], 0)
        self.assertEqual(result.loc[2, "overlap_bucket_label"], "0-199")

        self.assertEqual(result.loc[3, "group_overlap_start"], -70)
        self.assertEqual(result.loc[3, "group_overlap_end"], 100)
        self.assertEqual(result.loc[3, "group_overlap_length"], 170)
        self.assertEqual(result.loc[3, "overlap_bucket_start"], -200)
        self.assertEqual(result.loc[3, "overlap_bucket_label"], "-200--1")


    def test_save_bucket_statistics_writes_expected_columns(self):
        sample_rows = [
            {
                "analysis": "deepcre_starrseq",
                "bucket_label": "0-199",
                "bucket_start": 0,
                "bucket_end": 199,
                "bucket_size": 200,
                "count_points": 5,
                "count_unique_points": 4,
                "slope": 0.1,
                "intercept": 0.2,
                "spearman_r": 0.3,
                "spearman_p": 0.04,
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = os.path.join(temp_dir, "correlation")
            output_path = os.path.join(output_root, "deepcre_starrseq", overlap_analysis.BUCKET_SUMMARY_FILE_NAME)
            with patch.object(overlap_analysis, "CORRELATION_OUTPUT_ROOT", output_root):
                overlap_analysis.save_bucket_statistics(sample_rows)

            written = pd.read_csv(output_path)

        expected_columns = [
            "analysis",
            "bucket_label",
            "bucket_start",
            "bucket_end",
            "bucket_size",
            "count_points",
            "count_unique_points",
            "slope",
            "intercept",
            "spearman_r",
            "spearman_p",
        ]
        self.assertEqual(list(written.columns), expected_columns)
        self.assertEqual(len(written), 1)
        self.assertEqual(written.loc[0, "count_points"], 5)
        self.assertEqual(written.loc[0, "count_unique_points"], 4)


class TestMapStarrseqToDeepcre(unittest.TestCase):
    def test_uses_site_to_gene_ids_dict_to_select_candidates(self):
        # Arrange: a STARR-seq fragment embedded in GENE_A's reference at a
        # known offset, plus an unrelated GENE_B that must not be selected.
        starr_seq = "A" * 50 + "C" * 70 + "G" * 50
        gene_a_ref = "T" * 200 + starr_seq + "T" * 1130
        gene_data = [
            {"gene": "GENE_A", "ref_seq": gene_a_ref, "ref_fitness": 0.5, "start": 1000, "end": 2000},
            {"gene": "GENE_B", "ref_seq": "N" * 1500, "ref_fitness": 0.7, "start": 5000, "end": 6000},
        ]
        starr_entry = {
            "full_name": "bHLH_1:1000-1249_binding_0",
            "tf": "bHLH",
            "chrom": "1",
            "start": 1000,
            "end": 1249,
            "binding_status": "binding",
            "reference": False,
            "sequence": starr_seq,
        }
        site_to_gene_ids = {"bHLH_1:1000-1249": ["GENE_A"]}

        # Act
        mapping_results = overlap_analysis.map_starrseq_to_deepcre(
            [starr_entry], gene_data, site_to_gene_ids
        )

        # Assert: only the dict-selected GENE_A maps, with the aligned offsets.
        self.assertEqual(len(mapping_results), 1)
        result = mapping_results[0]
        self.assertEqual(result["gene"], "GENE_A")
        self.assertEqual(result["overlap_start"], 200)
        self.assertEqual(result["overlap_end"], 370)
        self.assertEqual(result["deepcre_ref_fitness"], 0.5)

    def test_site_absent_from_dict_yields_no_mapping(self):
        # Arrange
        starr_entry = {
            "full_name": "bHLH_1:1000-1249_binding_0",
            "tf": "bHLH",
            "chrom": "1",
            "start": 1000,
            "end": 1249,
            "binding_status": "binding",
            "reference": False,
            "sequence": "A" * 50 + "C" * 70 + "G" * 50,
        }
        gene_data = [{"gene": "GENE_A", "ref_seq": "A" * 200, "ref_fitness": 0.1, "start": 1000, "end": 2000}]

        # Act: the site key is not present in the mapping dict.
        mapping_results = overlap_analysis.map_starrseq_to_deepcre(
            [starr_entry], gene_data, {}
        )

        # Assert
        self.assertEqual(mapping_results, [])


class TestBuildSequencesMaxDifferences(unittest.TestCase):
    @staticmethod
    def _make_mapping(num_differences: int) -> dict:
        """Build a mapping whose ref/mutated sequences differ in exactly N bases."""
        ref_region_len = 50
        starr_seq = "A" * (ref_region_len - num_differences) + "C" * num_differences
        ref_seq = "G" * 100 + "A" * ref_region_len + "G" * 100
        return {
            "ref_seq": ref_seq,
            "starr_sequence": starr_seq,
            "overlap_start": 100,
            "overlap_end": 100 + ref_region_len,
            "reverse": False,
            "starr_full_name": f"bHLH_test_{num_differences}",
            "gene": "GENE_X",
        }

    def test_keeps_at_threshold_and_drops_above_for_n_15(self):
        # Arrange
        mappings = [self._make_mapping(15), self._make_mapping(16)]

        # Act
        with patch.object(overlap_analysis, "one_hot_encode", lambda seq: np.zeros(4)):
            _, meta_data = overlap_analysis.build_sequences(mappings, max_differences=15)

        # Assert
        self.assertEqual(len(meta_data), 1)
        self.assertEqual(meta_data[0]["differences"], 15)

    def test_keeps_at_threshold_and_drops_above_for_n_16(self):
        # Arrange
        mappings = [self._make_mapping(16), self._make_mapping(17)]

        # Act
        with patch.object(overlap_analysis, "one_hot_encode", lambda seq: np.zeros(4)):
            _, meta_data = overlap_analysis.build_sequences(mappings, max_differences=16)

        # Assert
        self.assertEqual(len(meta_data), 1)
        self.assertEqual(meta_data[0]["differences"], 16)


class TestReverseComplement(unittest.TestCase):
    def test_complements_and_reverses_uppercase(self):
        self.assertEqual(overlap_analysis.reverse_complement("ACGT"), "ACGT")
        self.assertEqual(overlap_analysis.reverse_complement("AACC"), "GGTT")

    def test_preserves_case(self):
        self.assertEqual(overlap_analysis.reverse_complement("acgt"), "acgt")

    def test_leaves_unmapped_characters_in_place(self):
        # 'N' has no complement mapping; it is reversed but not translated.
        self.assertEqual(overlap_analysis.reverse_complement("ANG"), "CNT")


class TestCompareSequences(unittest.TestCase):
    def test_counts_mismatches_for_equal_length(self):
        self.assertEqual(overlap_analysis.compare_sequences("AAAA", "AAAA"), 0)
        self.assertEqual(overlap_analysis.compare_sequences("AAAA", "AATA"), 1)
        self.assertEqual(overlap_analysis.compare_sequences("AAAA", "TTTT"), 4)

    def test_returns_longer_length_when_sizes_differ(self):
        self.assertEqual(overlap_analysis.compare_sequences("AAA", "AAAAA"), -1)


class TestGetStarrseqFragment(unittest.TestCase):
    def test_takes_suffix_when_overlap_starts_at_left_edge(self):
        # overlap_start == 0 -> fragment is the trailing seq_length bases.
        self.assertEqual(
            overlap_analysis.get_starrseq_fragment("ABCDEFGHIJ", 0, 4, overlap_analysis.IDEAL_PADDING_START, overlap_analysis.IDEAL_PADDING_END), "GHIJ"
        )

    def test_takes_suffix_when_overlap_starts_at_padding_end(self):
        fragment = overlap_analysis.get_starrseq_fragment(
            "ABCDEFGHIJ",
            overlap_analysis.IDEAL_PADDING_END,
            overlap_analysis.IDEAL_PADDING_END + 4,
            overlap_analysis.IDEAL_PADDING_START, overlap_analysis.IDEAL_PADDING_END
        )
        self.assertEqual(fragment, "GHIJ")

    def test_takes_prefix_when_overlap_ends_at_padding_start(self):
        fragment = overlap_analysis.get_starrseq_fragment(
            "ABCDEFGHIJ",
            overlap_analysis.IDEAL_PADDING_START - 4,
            overlap_analysis.IDEAL_PADDING_START,
            overlap_analysis.IDEAL_PADDING_START, overlap_analysis.IDEAL_PADDING_END
        )
        self.assertEqual(fragment, "ABCD")

    def test_returns_full_sequence_for_internal_overlap(self):
        fragment = overlap_analysis.get_starrseq_fragment("ABCDEFGHIJ", 100, 150, overlap_analysis.IDEAL_PADDING_START, overlap_analysis.IDEAL_PADDING_END)
        self.assertEqual(fragment, "ABCDEFGHIJ")


class TestFindOverlapPositions(unittest.TestCase):
    def test_forward_match_extends_end_by_fifty(self):
        start_query = "A" * 10
        end_query = "C" * 10
        ref_seq = "T" * 200 + start_query + "G" * 90 + end_query + "T" * 50

        start_pos, end_pos, _, _, reverse = overlap_analysis.find_overlap_positions(
            ref_seq, start_query, end_query, additional_padding=0
        )

        self.assertEqual(start_pos, 200)
        self.assertEqual(end_pos, 350)
        self.assertFalse(reverse)

    def test_reverse_complement_match_sets_reverse_flag(self):
        start_query = "A" * 10
        end_query = "C" * 10
        # Reference holds only the reverse complements: rc(C*10)=G*10 first,
        # rc(A*10)=T*10 later, so the forward search fails and the function
        # falls back to the reverse-complement branch.
        ref_seq = "G" * 10 + "AC" * 70 + "T" * 10 + "AC" * 5

        start_pos, end_pos, _, _, reverse = overlap_analysis.find_overlap_positions(
            ref_seq, start_query, end_query, additional_padding=0
        )

        self.assertTrue(reverse)
        self.assertEqual(start_pos, 0)
        self.assertEqual(end_pos, 200)

    def test_missing_start_query_anchors_to_left_edge(self):
        end_query = "C" * 10
        ref_seq = "G" * 300 + end_query + "G" * 100

        start_pos, end_pos, _, _, reverse = overlap_analysis.find_overlap_positions(
            ref_seq, "A" * 10, end_query, additional_padding=0
        )

        self.assertEqual(start_pos, 0)
        self.assertEqual(end_pos, 350)
        self.assertFalse(reverse)

    def test_missing_end_query_anchors_to_padding_start(self):
        start_query = "A" * 10
        ref_seq = "T" * 200 + start_query + "G" * 100

        start_pos, end_pos, _, _, reverse = overlap_analysis.find_overlap_positions(
            ref_seq, start_query, "C" * 10, additional_padding=0
        )

        self.assertEqual(start_pos, 200)
        self.assertEqual(end_pos, overlap_analysis.IDEAL_PADDING_START)
        self.assertFalse(reverse)

    def test_short_overlap_is_rejected(self):
        start_query = "A" * 10
        end_query = "C" * 10
        ref_seq = "T" * 200 + start_query + "G" * 30 + end_query + "T" * 50

        result = overlap_analysis.find_overlap_positions(
            ref_seq, start_query, end_query, additional_padding=0
        )

        self.assertEqual(result, (-1, -1, -1, -1, False))

    def test_no_match_returns_sentinel(self):
        ref_seq = "AC" * 100

        result = overlap_analysis.find_overlap_positions(
            ref_seq, "T" * 10, "G" * 10, additional_padding=0
        )

        self.assertEqual(result, (-1, -1, -1, -1, False))


class TestCalculateDeltas(unittest.TestCase):
    def test_computes_prediction_and_enrichment_deltas_against_reference(self):
        # Arrange: a reference entry and a mutated entry sharing a base id.
        df = pd.DataFrame(
            [
                {
                    "starr_full_name": "bHLH_1:100-200_reference_0",
                    "starr_reference": True,
                    "enrichment": 2.0,
                    "prediction_mutated": 0.8,
                    "deepcre_ref_fitness": 0.5,
                },
                {
                    "starr_full_name": "bHLH_1:100-200_binding_0",
                    "starr_reference": False,
                    "enrichment": 5.0,
                    "prediction_mutated": 0.9,
                    "deepcre_ref_fitness": 0.5,
                },
            ]
        )

        # Act
        result = overlap_analysis.calculate_deltas(df)

        # Assert: mutated deltas measured against its reference window.
        mutated = result[result["starr_reference"] == False].iloc[0]
        self.assertAlmostEqual(mutated["delta_prediction"], 0.4)
        self.assertAlmostEqual(mutated["delta_enrichment"], 3.0)
        self.assertEqual(mutated["starr_seq_base"], "bHLH_1:100-200")
        reference = result[result["starr_reference"] == True].iloc[0]
        self.assertAlmostEqual(reference["delta_enrichment"], 0.0)


class TestMergeWithStarrseqResults(unittest.TestCase):
    def test_maps_enrichment_and_condition_and_drops_id(self):
        # Arrange
        predictions_df = pd.DataFrame(
            [
                {"starr_full_name": "X1", "gene": "G1"},
                {"starr_full_name": "X2", "gene": "G2"},
            ]
        )
        starr_seq_results = pd.DataFrame(
            [
                {"id": "X1", "enrichment": 1.0, "condition": "light", "extra": "x"},
                {"id": "X2", "enrichment": 2.0, "condition": "dark", "extra": "y"},
            ]
        )

        # Act
        merged = overlap_analysis.merge_with_starrseq_results(
            predictions_df, starr_seq_results
        )

        # Assert: enrichment + condition carried over, join key and unrelated
        # columns dropped.
        self.assertNotIn("id", merged.columns)
        self.assertNotIn("extra", merged.columns)
        self.assertIn("condition", merged.columns)
        enrichment_x1 = merged.loc[
            merged["starr_full_name"] == "X1", "enrichment"
        ].iloc[0]
        self.assertEqual(enrichment_x1, 1.0)

    def test_omits_condition_when_absent(self):
        # Arrange
        predictions_df = pd.DataFrame([{"starr_full_name": "X1"}])
        starr_seq_results = pd.DataFrame([{"id": "X1", "enrichment": 4.2}])

        # Act
        merged = overlap_analysis.merge_with_starrseq_results(
            predictions_df, starr_seq_results
        )

        # Assert
        self.assertNotIn("condition", merged.columns)
        self.assertEqual(merged.loc[0, "enrichment"], 4.2)


class _FakeRecord:
    """Minimal stand-in for a ``pyfaidx`` record (``.name`` + ``str()``)."""

    def __init__(self, name: str, sequence: str) -> None:
        self.name = name
        self._sequence = sequence

    def __str__(self) -> str:
        return self._sequence


class _FakeFasta:
    """Iterable/indexable stand-in for a ``pyfaidx.Fasta`` object."""

    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def __getitem__(self, name):
        for record in self._records:
            if record.name == name:
                return record
        raise KeyError(name)


class TestLoadStarrseqData(unittest.TestCase):
    def test_parses_binding_reference_and_non_binding_headers(self):
        # Arrange: a 4-part binding header, a 3-part reference header, and a
        # 3-part non-binding header.
        records = [
            _FakeRecord("bHLH_1:100-200_binding_7", "ACGT"),
            _FakeRecord("WRKY_2:300-400_reference", "TTTT"),
            _FakeRecord("WRKY_2:300-400_nonbind", "GGGG"),
        ]

        # Act
        with patch.object(
            overlap_analysis, "Fasta", return_value=_FakeFasta(records)
        ):
            data = overlap_analysis.load_starrseq_data("unused.fasta")

        # Assert
        self.assertEqual(len(data), 3)

        binding = data[0]
        self.assertEqual(binding["tf"], "bHLH")
        self.assertEqual(binding["tf_id"], "7")
        self.assertEqual(binding["chrom"], "1")
        self.assertEqual(binding["start"], 100)
        self.assertEqual(binding["end"], 200)
        self.assertEqual(binding["binding_status"], "binding")
        self.assertFalse(binding["reference"])
        self.assertEqual(binding["sequence"], "ACGT")

        reference = data[1]
        self.assertTrue(reference["reference"])
        self.assertEqual(reference["binding_status"], "binding")

        non_binding = data[2]
        self.assertFalse(non_binding["reference"])
        self.assertEqual(non_binding["binding_status"], "non_binding")


class TestMakeDeepcrePredictions(unittest.TestCase):
    def test_attaches_flattened_predictions_to_metadata(self):
        # Arrange
        seqs = np.zeros((2, 3020, 4))
        meta_data = [
            {"gene": "G1", "starr_full_name": "X1"},
            {"gene": "G2", "starr_full_name": "X2"},
        ]
        fake_model = Mock()
        fake_model.predict.return_value = np.array([[0.1], [0.2]])

        # Act
        with patch.object(overlap_analysis, "load_model", return_value=fake_model):
            result = overlap_analysis.make_deepcre_predictions(seqs, meta_data)

        # Assert
        self.assertAlmostEqual(result["prediction_mutated"].iloc[0], 0.1)
        self.assertAlmostEqual(result["prediction_mutated"].iloc[1], 0.2)
        self.assertEqual(list(result["gene"]), ["G1", "G2"])
        fake_model.predict.assert_called_once()


class TestModuleConfiguration(unittest.TestCase):
    def test_configure_matplotlib_sets_poster_font_sizes(self):
        overlap_analysis.configure_matplotlib()
        self.assertEqual(overlap_analysis.mpl.rcParams["axes.titlesize"], 20)
        self.assertEqual(
            overlap_analysis.mpl.rcParams["savefig.dpi"],
            overlap_analysis.OUTPUT_DPI,
        )

    def test_set_output_root_updates_module_state(self):
        original = overlap_analysis.CORRELATION_OUTPUT_ROOT
        try:
            overlap_analysis.set_output_root("/tmp/example-root")
            self.assertEqual(
                overlap_analysis.CORRELATION_OUTPUT_ROOT, "/tmp/example-root"
            )
        finally:
            overlap_analysis.set_output_root(original)


class TestPlottingSmoke(unittest.TestCase):
    """Exercise the plotting routines end-to-end against a temp output root.

    These verify the routines run without error, return their statistics rows,
    and write the expected files; the rendered figures themselves are not
    inspected.
    """

    def setUp(self):
        overlap_analysis.plt.switch_backend("Agg")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._root = os.path.join(self._tmp.name, "correlation")
        patcher = patch.object(
            overlap_analysis, "CORRELATION_OUTPUT_ROOT", self._root
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _bucketed_df(self) -> pd.DataFrame:
        base = pd.DataFrame(
            [
                {"overlap_start": 800, "overlap_end": 970,
                 "prediction_mutated": 0.1, "enrichment": 1.0,
                 "differences": 3, "starr_reference": False, "gene": "G1"},
                {"overlap_start": 800, "overlap_end": 970,
                 "prediction_mutated": 0.2, "enrichment": 2.0,
                 "differences": 4, "starr_reference": False, "gene": "G1"},
                {"overlap_start": 800, "overlap_end": 970,
                 "prediction_mutated": 0.3, "enrichment": 3.0,
                 "differences": 5, "starr_reference": False, "gene": "G2"},
            ]
        )
        return overlap_analysis.add_length_corrected_overlap_buckets(base)

    def test_deepcre_starrseq_correlation_returns_stats_and_writes_png(self):
        df = self._bucketed_df()

        rows = overlap_analysis.plot_deepcre_starrseq_correlation(df)

        labels = {row["bucket_label"] for row in rows}
        self.assertIn("all", labels)
        self.assertIn("800-999", labels)
        overall = next(row for row in rows if row["bucket_label"] == "all")
        self.assertEqual(overall["count_points"], 3)
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    self._root,
                    "deepcre_starrseq",
                    "deepcre_starrseq_correlation.png",
                )
            )
        )

    def test_mutation_correlation_plots_return_rows(self):
        df = self._bucketed_df()
        self.assertTrue(overlap_analysis.plot_mutation_starrseq_correlation(df))
        self.assertTrue(overlap_analysis.plot_mutation_deepcre_correlation(df))

    def test_individual_bucket_views_writes_files(self):
        df = self._bucketed_df()

        overlap_analysis.plot_individual_bucket_views(df, "800-999")

        self.assertTrue(
            os.path.exists(
                os.path.join(
                    self._root,
                    "deepcre_starrseq",
                    "deepcre_starrseq_correlation_bucket_800_999.png",
                )
            )
        )

    def test_simply_plot_multi_writes_png(self):
        series = [([1, 2, 3], [0.1, 0.2, 0.3], "all", "#555555")]

        overlap_analysis.simply_plot_multi(
            series, "x", "y", "title", "corr_by_position", "subfolder"
        )

        self.assertTrue(
            os.path.exists(
                os.path.join(self._root, "subfolder", "corr_by_position.png")
            )
        )

    def test_correlation_over_positions_fixed_window_writes_png(self):
        df = pd.DataFrame(
            {
                "group_overlap_start": [800] * 15,
                "prediction_mutated": np.linspace(0.0, 1.0, 15),
                "enrichment": np.linspace(1.0, 5.0, 15),
            }
        )

        overlap_analysis.plot_correlation_over_positions_fixed_window(
            [(df, "all", "#555555")]
        )

        self.assertTrue(
            os.path.exists(
                os.path.join(
                    self._root,
                    "correlation_by_position_fixed_window",
                    "correlation_by_position.png",
                )
            )
        )

    def test_correlation_over_positions_fixed_number_elements_writes_png(self):
        size = 201
        df = pd.DataFrame(
            {
                "group_overlap_start": np.arange(size),
                "prediction_mutated": np.linspace(0.0, 1.0, size),
                "enrichment": np.linspace(1.0, 5.0, size),
            }
        )

        overlap_analysis.plot_correlation_over_positions_fixed_number_elements(
            [(df, "all", "#555555")]
        )

        self.assertTrue(
            os.path.exists(
                os.path.join(
                    self._root,
                    "correlation_by_position_fixed_number_elements",
                    "correlation_by_position.png",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
