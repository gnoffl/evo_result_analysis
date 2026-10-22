"""Significance testing helpers shared by the analysis and workflow layers.

The paper's per-feature significance statements (per TF family in the deepCIS
peak analysis, per motif in the blamm analysis) all follow the same recipe: a
per-gene vector of changes ``optimized - reference`` is tested against zero with
a two-sided Wilcoxon signed-rank test, the p-values are Benjamini-Hochberg
corrected across the features tested in that analysis, and the resulting
q-values are rendered as significance stars. Keeping that recipe in one module
guarantees the star thresholds are identical wherever they appear.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

#: q-value thresholds (most stringent first) mapped to their star annotation.
SIGNIFICANCE_STAR_THRESHOLDS: List[Tuple[float, str]] = [
    (0.001, "***"),
    (0.01, "**"),
    (0.05, "*"),
]


def wilcoxon_pvalue_vs_zero(values: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value of ``values`` against zero.

    Args:
        values: One observation per replicate (e.g. per gene).

    Returns:
        The p-value, or NaN if the test is undefined — which happens when every
        observation is zero, since the default ``zero_method="wilcox"`` discards
        zero differences and leaves nothing to rank.
    """
    try:
        return float(wilcoxon(values)[1])
    except ValueError:
        return float("nan")


def benjamini_hochberg_qvalues(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg corrected q-values, leaving NaN p-values as NaN.

    NaN entries are excluded from the correction entirely, so they neither
    inflate the number of tests nor receive a q-value.

    Args:
        pvalues: p-values indexed by feature.

    Returns:
        Series with the same index holding the q-values.
    """
    qvalues = pd.Series(np.nan, index=pvalues.index)
    finite = pvalues.notna()
    if finite.any():
        qvalues[finite] = multipletests(pvalues[finite].to_numpy(), method="fdr_bh")[1]
    return qvalues


def qvalue_to_stars(qvalue: float) -> str:
    """Return the significance star string for a q-value.

    Args:
        qvalue: Corrected p-value; NaN counts as not significant.

    Returns:
        The star string from :data:`SIGNIFICANCE_STAR_THRESHOLDS`, or the empty
        string if the q-value clears no threshold.
    """
    if pd.isna(qvalue):
        return ""
    for threshold, stars in SIGNIFICANCE_STAR_THRESHOLDS:
        if qvalue < threshold:
            return stars
    return ""
