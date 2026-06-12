"""Tests for the one-off mutation-assessment script.

Only the sequence-manipulation logic is checked (no model is loaded).
"""

import unittest

from workflows.hoffie.mutation_assessment.assess_mutations import insert_poly_c


class TestInsertPolyC(unittest.TestCase):
    def test_grows_run_by_one_and_keeps_1500_bp(self):
        # Arrange: 1500 bp promoter with a CGTCCT-flanked 14x C run.
        promoter = ("A" * 100 + "CGTCCT" + "C" * 14 + "T" * 1380)[:1500]
        self.assertEqual(len(promoter), 1500)

        # Act
        result = insert_poly_c(promoter, "CGTCCT")

        # Assert
        self.assertEqual(len(result), 1500)  # stays 1500 bp
        self.assertIn("CGTCCT" + "C" * 15, result)  # run grew 14 -> 15
        self.assertEqual(result[-1], promoter[-2])  # last promoter base dropped


if __name__ == "__main__":
    unittest.main()
