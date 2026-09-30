import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from workflows.starrseq_v2.probe_analysis import best_fitness, collect_run, normalize_per_gene

FRONT = [["AAA", 0.9, 60.0], ["AAC", 0.7, 20.0], ["AAG", 0.6, 15.0], ["AAT", 0.2, 0.0]]


class TestBestFitness(unittest.TestCase):
    def test_maximize_respects_mutation_limit(self):
        self.assertEqual(best_fitness(FRONT, 20, maximize=True), 0.7)
        self.assertEqual(best_fitness(FRONT, 60, maximize=True), 0.9)

    def test_minimize_picks_lowest(self):
        self.assertEqual(best_fitness(FRONT, 20, maximize=False), 0.2)


class TestCollectRun(unittest.TestCase):
    def test_missing_checkpoint_gives_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "parameters.json").write_text(json.dumps({"sequence_name": "gene_a"}))
            (run_dir / "saved_populations").mkdir()
            (run_dir / "saved_populations" / "pareto_front.json").write_text(json.dumps(FRONT))

            rows = collect_run(run_dir, maximize=True)

        by_key = {(row["generation"], row["mutation_limit"]): row["fitness"] for row in rows}
        self.assertTrue(math.isnan(by_key[(1000, 20)]))
        self.assertEqual(by_key[(2000, 20)], 0.7)
        self.assertEqual(by_key[(2000, 60)], 0.9)


class TestNormalizePerGene(unittest.TestCase):
    def test_rescales_between_unmutated_and_best(self):
        results = pd.DataFrame({
            "gene": ["a", "a", "a", "b", "b", "b"],
            "generation": [2000, 2000, 1000, 2000, 2000, 1000],
            "mutation_limit": [0, 60, 60, 0, 60, 20],
            "fitness": [0.2, 0.8, 0.5, 0.5, 0.1, 0.3],
        })

        normalized = normalize_per_gene(results)

        self.assertEqual(normalized["mutation_limit"].tolist(), [60, 60, 60, 20])
        for actual, expected in zip(normalized["fitness"], [1.0, 0.5, 1.0, 0.5]):
            self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
