"""Unit tests for analysis.utils.statistics."""

import math
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from analysis.utils.statistics import (
    SIGNIFICANCE_STAR_THRESHOLDS,
    benjamini_hochberg_qvalues,
    qvalue_to_stars,
    wilcoxon_pvalue_vs_zero,
)


class TestWilcoxonPvalueVsZero(unittest.TestCase):
    """Tests for the signed-rank p-value helper."""

    def test_consistent_shift_is_significant(self):
        # Arrange
        values = np.ones(10)

        # Act
        pvalue = wilcoxon_pvalue_vs_zero(values)

        # Assert
        self.assertLess(pvalue, 0.05)

    def test_all_zero_input_yields_nan(self):
        # Arrange
        values = np.zeros(10)

        # Act
        pvalue = wilcoxon_pvalue_vs_zero(values)

        # Assert
        self.assertTrue(math.isnan(pvalue))

    def test_symmetric_input_is_not_significant(self):
        # Arrange
        values = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])

        # Act
        pvalue = wilcoxon_pvalue_vs_zero(values)

        # Assert
        self.assertGreater(pvalue, 0.5)

    def test_sign_does_not_change_two_sided_pvalue(self):
        # Arrange
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # Act
        positive = wilcoxon_pvalue_vs_zero(values)
        negative = wilcoxon_pvalue_vs_zero(-values)

        # Assert
        self.assertAlmostEqual(positive, negative)


class TestBenjaminiHochbergQvalues(unittest.TestCase):
    """Tests for the multiple-testing correction helper."""

    def test_qvalues_are_at_least_the_pvalues(self):
        # Arrange
        pvalues = pd.Series([0.001, 0.01, 0.04, 0.5], index=list("abcd"))

        # Act
        qvalues = benjamini_hochberg_qvalues(pvalues)

        # Assert
        self.assertTrue((qvalues >= pvalues - 1e-12).all())
        self.assertAlmostEqual(qvalues["d"], 0.5)

    def test_nan_pvalues_stay_nan_and_do_not_count_as_tests(self):
        # Arrange
        with_nan = pd.Series([0.01, float("nan"), 0.02], index=list("abc"))
        without_nan = pd.Series([0.01, 0.02], index=list("ac"))

        # Act
        qvalues = benjamini_hochberg_qvalues(with_nan)
        reference = benjamini_hochberg_qvalues(without_nan)

        # Assert
        self.assertTrue(math.isnan(qvalues["b"]))
        self.assertAlmostEqual(qvalues["a"], reference["a"])
        self.assertAlmostEqual(qvalues["c"], reference["c"])

    def test_all_nan_input_returns_all_nan(self):
        # Arrange
        pvalues = pd.Series([float("nan"), float("nan")], index=list("ab"))

        # Act
        qvalues = benjamini_hochberg_qvalues(pvalues)

        # Assert
        self.assertTrue(qvalues.isna().all())


class TestQvalueToStars(unittest.TestCase):
    """Tests for the star rendering helper."""

    def test_star_tiers(self):
        # Arrange / Act / Assert
        self.assertEqual(qvalue_to_stars(0.0005), "***")
        self.assertEqual(qvalue_to_stars(0.005), "**")
        self.assertEqual(qvalue_to_stars(0.02), "*")
        self.assertEqual(qvalue_to_stars(0.2), "")

    def test_thresholds_are_exclusive(self):
        # Arrange / Act / Assert
        self.assertEqual(qvalue_to_stars(0.05), "")
        self.assertEqual(qvalue_to_stars(0.01), "*")
        self.assertEqual(qvalue_to_stars(0.001), "**")

    def test_nan_is_not_significant(self):
        # Arrange / Act / Assert
        self.assertEqual(qvalue_to_stars(float("nan")), "")

    def test_thresholds_are_ordered_most_stringent_first(self):
        # Arrange
        thresholds = [threshold for threshold, _ in SIGNIFICANCE_STAR_THRESHOLDS]

        # Act / Assert
        self.assertEqual(thresholds, sorted(thresholds))


if __name__ == "__main__":
    unittest.main()
