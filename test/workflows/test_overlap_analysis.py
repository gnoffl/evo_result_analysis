import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from workflows.overlap_analysis import starrseq_deepcre_correlation_WRKY as overlap_analysis


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

        result = overlap_analysis.add_bucketed_overlap_positions(df)

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


if __name__ == "__main__":
    unittest.main()
