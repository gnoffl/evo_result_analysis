"""Unit tests for ``starrseq_deepcre_correlation_combined``.

The two per-TF preparation steps and the shared analysis stage are mocked so
that the pooling glue (concatenating the WRKY and bHLH dataframes and handing
the union to ``_common.run_correlation_analysis``) can be exercised without any
model inference, file I/O, or plotting.
"""
import unittest
from unittest.mock import patch

import pandas as pd

from workflows.overlap_analysis import starrseq_deepcre_correlation_combined as combined


class TestCombinedMain(unittest.TestCase):
    def test_pools_both_tf_dataframes_into_one_analysis(self):
        # Arrange: WRKY contributes 2 rows, bHLH contributes 3 rows.
        wrky_df = pd.DataFrame({"gene": ["W1", "W2"], "enrichment": [1.0, 2.0]})
        bhlh_df = pd.DataFrame({"gene": ["B1", "B2", "B3"], "enrichment": [3.0, 4.0, 5.0]})

        # Act
        with patch.object(combined._common, "configure_matplotlib"), \
                patch.object(combined._common, "set_output_root") as mock_set_root, \
                patch.object(combined, "prepare_wrky_enrichment_df", return_value=wrky_df), \
                patch.object(combined, "prepare_bhlh_enrichment_df", return_value=bhlh_df), \
                patch.object(combined._common, "run_correlation_analysis") as mock_run:
            combined.main()

        # Assert: the output root is the combined directory, and the analysis
        # receives a single dataframe with the pooled WRKY + bHLH rows.
        mock_set_root.assert_called_once_with(combined.CORRELATION_OUTPUT_ROOT)
        mock_run.assert_called_once()
        pooled_df = mock_run.call_args[0][0]
        self.assertEqual(len(pooled_df), len(wrky_df) + len(bhlh_df))
        self.assertEqual(list(pooled_df["gene"]), ["W1", "W2", "B1", "B2", "B3"])
        # Concatenation must reset the index so pooled rows are uniquely indexed.
        self.assertEqual(list(pooled_df.index), [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
