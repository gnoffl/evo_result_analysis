"""Pooled WRKY + bHLH STARR-seq x deepCRE correlation.

This is the pooled sibling of the per-TF scripts. It reuses each TF's
data-preparation step (:func:`prepare_wrky_enrichment_df` and
:func:`prepare_bhlh_enrichment_df`), concatenates the two analysis-ready
dataframes, and runs the shared, TF-agnostic analysis from :mod:`_common` over
the union.

Pooling is a simple row concatenation: each TF dataframe is built independently
from its own reference windows and mapping (so genes shared between the two TF
sets are never double-counted), both come out of the identical downstream
pipeline with the same columns, and both read the same light+dark STARR-seq
enrichment file. Per-TF mutation thresholds (WRKY <=15, bHLH <=16) are applied
inside each preparation step before concatenation.
"""
import os

import pandas as pd

from workflows.overlap_analysis import _common
from workflows.overlap_analysis.starrseq_deepcre_correlation_WRKY import prepare_wrky_enrichment_df
from workflows.overlap_analysis.starrseq_deepcre_correlation_bHLH import prepare_bhlh_enrichment_df

BASE_DIR = os.path.dirname(__file__)
CORRELATION_OUTPUT_ROOT = os.path.join(BASE_DIR, "correlation_combined", "new")


def main():
    _common.configure_matplotlib()
    _common.set_output_root(CORRELATION_OUTPUT_ROOT)
    wrky_df = prepare_wrky_enrichment_df()
    bhlh_df = prepare_bhlh_enrichment_df()
    combined_df = pd.concat([wrky_df, bhlh_df], ignore_index=True)
    _common.run_correlation_analysis(combined_df)


if __name__ == "__main__":
    main()
