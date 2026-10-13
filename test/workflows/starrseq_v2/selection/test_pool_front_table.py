import json
import os
import tempfile
import unittest
from typing import Any, Dict, List

import pandas as pd

from workflows.starrseq_v2.selection import pool_front_table as pft

BASELINE_NAME = "ensemble_ntab_model_construct_correct_unconstrained_ntab_genes_maximize"
OFFSITE_NAME = "single_ara_model_construct_offsite_unconstrained_ara_genes_minimize"
NATURAL_OFFSITE_NAME = "ensemble_ara_model_natural_offsite_unconstrained_ara_genes_maximize"
TIMESTAMP = "_261008_233230_989743"
# the N. tabacum scaffold recurs inside the gene id, which is what defeats naive splitting
NTAB_GENE = "Nicotiana_tabacum_NW_015787655.1_000001"
NTAB_RECORD = f"NW_015787655.1_{NTAB_GENE}_gene:68121-69160"
ARA_GENE = "AT1G22810"
ARA_RECORD = f"1_{ARA_GENE}_gene:8075037-8074230"


def factors_of(name: str) -> Dict[str, str]:
    return pft.config_factors()[name]


def metadata_rows() -> pd.DataFrame:
    """The ara gene under three reasons with both directions and two windows."""
    return pd.DataFrame({
        "gene_id": [ARA_GENE, ARA_GENE, ARA_GENE, NTAB_GENE],
        "species": ["arabidopsis", "arabidopsis", "arabidopsis", "ntab"],
        "reason": ["starrseq_v1", "GOF", "LOF", "flowering"],
        "alternative_start": [2734, 186, 186, 500],
        "alternative_end": [2904, 356, 356, 670],
        "direction": ["minimize", "maximize", "minimize", "maximize"],
        "full_length_window": [True, True, True, True],
    })


def write_json(path: str, content: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(content, f)


def write_run(runs_root: str, config_dir: str, record_name: str, front: List[list], **parameters: Any) -> str:
    """Write one run folder and return its name."""
    run_name = record_name + "_261008_233255_874430"
    run_path = os.path.join(runs_root, config_dir, run_name)
    write_json(os.path.join(run_path, "parameters.json"), {
        "sequence_name": record_name,
        "models": [{"path": "/models/ntab/b.h5"}, {"path": "/models/ntab/a.h5"}],
        "number_of_generations": 1000,
        "vcf_path": None,
        "mutation_start": 780,
        "mutation_end": 950,
        **parameters,
    })
    write_json(os.path.join(run_path, "saved_populations", "pareto_front.json"), front)
    return run_name


def stats_entry(start_fitness: float, max_mutations: float, half_max: int) -> Dict[str, Any]:
    return {"start_fitness": start_fitness, "final_fitness": 0.9, "max_mutations": max_mutations,
            "num_mutations_half_max_effect": half_max}


class TestConfigFactors(unittest.TestCase):
    def test_parses_levels_with_underscores(self):
        self.assertEqual(factors_of(OFFSITE_NAME), {
            "models": "single", "model_sp": "ara_model", "background": "construct",
            "location": "offsite", "mutations": "unconstrained", "gene_sp": "ara_genes",
            "direction": "minimize",
        })

    def test_v1_genes_are_a_gene_species(self):
        factors = factors_of("ensemble_ntab_model_construct_offsite_unconstrained_v1_genes_maximize")
        self.assertEqual(factors["gene_sp"], "v1_genes")

    def test_names_are_unique(self):
        # 2 levels for five factors, 3 gene species, 2 directions
        self.assertEqual(len(pft.config_factors()), 2 ** 5 * 3 * 2)


class TestGroupOf(unittest.TestCase):
    def test_all_four_groups(self):
        self.assertEqual(pft.group_of(factors_of(BASELINE_NAME)), "baseline")
        self.assertEqual(
            pft.group_of(factors_of(BASELINE_NAME.replace("ntab_genes", "ara_genes"))), "baseline"
        )
        self.assertEqual(
            pft.group_of(factors_of(BASELINE_NAME.replace("ensemble", "single"))), "arms"
        )
        self.assertEqual(pft.group_of(factors_of(OFFSITE_NAME)), "grid")
        self.assertEqual(
            pft.group_of(factors_of("ensemble_ntab_model_construct_offsite_unconstrained_v1_genes_maximize")),
            "starrseq_v1",
        )


class TestSourceWindow(unittest.TestCase):
    def test_correct_location_is_the_insert_site(self):
        parameters = {"sequence_name": ARA_RECORD, "mutation_start": 780, "mutation_end": 950}
        natural_correct = factors_of(NATURAL_OFFSITE_NAME.replace("offsite", "correct"))
        self.assertEqual(pft.source_window(factors_of(BASELINE_NAME), parameters), (780, 950))
        self.assertEqual(pft.source_window(natural_correct, parameters), (780, 950))

    def test_construct_offsite_reads_the_record_name_not_the_mutation_window(self):
        parameters = {"sequence_name": ARA_RECORD + "_off_2734", "mutation_start": 780, "mutation_end": 950}
        self.assertEqual(pft.source_window(factors_of(OFFSITE_NAME), parameters), (2734, 2904))

    def test_natural_offsite_reads_the_mutation_window(self):
        parameters = {"sequence_name": ARA_RECORD, "mutation_start": 186, "mutation_end": 356}
        self.assertEqual(pft.source_window(factors_of(NATURAL_OFFSITE_NAME), parameters), (186, 356))


class TestFrontEntries(unittest.TestCase):
    def read(self, front: List[list]) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pareto_front.json")
            write_json(path, front)
            return pft.front_entries(path)

    def test_gaps_and_ties_keep_file_order(self):
        entries = self.read([["A" * 170, 0.9, 5.0], ["C" * 170, 0.8, 2.0], ["G" * 170, 0.8, 2.0],
                             ["T" * 170, 0.5, 0.0]])
        self.assertEqual(entries["front_index"].tolist(), [0, 1, 2, 3])
        self.assertEqual(entries["n_mutations"].tolist(), [5, 2, 2, 0])
        self.assertEqual(entries["tied_at_count"].tolist(), [False, True, True, False])

    def test_front_of_length_one(self):
        entries = self.read([["A" * 170, 0.5, 0.0]])
        self.assertEqual(len(entries), 1)
        self.assertFalse(entries["tied_at_count"].iloc[0])

    def test_counts_n(self):
        entries = self.read([["NNA" + "C" * 167, 0.5, 0.0]])
        self.assertEqual(entries["n_count"].iloc[0], 2)


class TestCollapseMetadata(unittest.TestCase):
    def test_one_row_per_gene_with_reasons_joined(self):
        collapsed = pft.collapse_metadata(metadata_rows()).set_index("gene_id")
        self.assertEqual(len(collapsed), 2)
        self.assertEqual(collapsed.loc[ARA_GENE, "reason"], "GOF|LOF|starrseq_v1")
        self.assertEqual(collapsed.loc[ARA_GENE, "species"], "arabidopsis")


class TestAddEntryFlags(unittest.TestCase):
    def test_flags_and_signed_delta(self):
        table = pd.DataFrame({
            "direction": ["maximize", "maximize", "minimize", "minimize"],
            "n_mutations": [0, 3, 0, 3],
            "fitness": [0.5, 0.75, 0.5, 0.25],
            "start_fitness": [0.5, 0.5, 0.5, 0.5],
            "num_mutations_half_max_effect": [3, 3, 1, 1],
            "max_mutations_reached": [3, 3, 3, 3],
        })
        flagged = pft.add_entry_flags(table)
        self.assertEqual(flagged["fitness_delta_vs_reference"].tolist(), [0.0, 0.25, 0.0, 0.25])
        self.assertEqual(flagged["is_reference"].tolist(), [True, False, True, False])
        self.assertEqual(flagged["is_half_max"].tolist(), [False, True, False, False])
        self.assertEqual(flagged["is_max_mutations"].tolist(), [False, True, False, True])


class TestPoolFronts(unittest.TestCase):
    """A tree with one baseline and one construct+offsite config."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = os.path.join(self.tmp.name, "runs")
        self.analysis = os.path.join(self.tmp.name, "analysis")
        baseline_dir, offsite_dir = BASELINE_NAME + TIMESTAMP, OFFSITE_NAME + TIMESTAMP
        baseline_run = write_run(
            self.runs, baseline_dir, NTAB_RECORD, [["A" * 170, 0.9, 4.0], ["N" * 170, 0.5, 0.0]]
        )
        # two windows of the same gene, the case a metadata lookup cannot resolve
        first_window = write_run(
            self.runs, offsite_dir, ARA_RECORD + "_off_186", [["A" * 170, 0.5, 0.0]],
            models=[{"path": "/models/Ara_ssr/c.h5"}],
        )
        second_window = write_run(
            self.runs, offsite_dir, ARA_RECORD + "_off_2734", [["A" * 170, 0.2, 1.0], ["C" * 170, 0.4, 0.0]],
            models=[{"path": "/models/Ara_ssr/c.h5"}],
        )
        write_json(os.path.join(self.runs, baseline_dir, "problem_configs.json"), [])
        write_json(
            os.path.join(self.analysis, BASELINE_NAME, f"stats_{BASELINE_NAME}.json"),
            {baseline_run: stats_entry(0.5, 4.0, 4)},
        )
        write_json(os.path.join(self.analysis, OFFSITE_NAME, f"stats_{OFFSITE_NAME}.json"), {
            first_window: stats_entry(0.5, 0.0, 0),
            second_window: stats_entry(0.4, 1.0, 1),
        })
        self.run_overview, self.front_table = pft.pool_fronts(self.runs, self.analysis, metadata_rows())

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_row_per_run_and_per_entry(self):
        # the ara gene has three metadata rows; nothing may fan out
        self.assertEqual(len(self.run_overview), 3)
        self.assertEqual(len(self.front_table), 5)
        self.assertFalse(self.front_table.duplicated(["run_dir", "front_index"]).any())

    def test_run_columns(self):
        runs = self.run_overview.set_index("record_name")
        baseline = runs.loc[NTAB_RECORD]
        self.assertEqual(baseline["gene_id"], NTAB_GENE)
        self.assertEqual(baseline["group"], "baseline")
        self.assertEqual(baseline["config_name"], BASELINE_NAME)
        self.assertEqual(baseline["run_dir"], f"{BASELINE_NAME}{TIMESTAMP}/{NTAB_RECORD}_261008_233255_874430")
        self.assertEqual(baseline["model_paths"], "a.h5|b.h5")
        self.assertEqual(baseline["n_models"], 2)
        self.assertEqual(runs.loc[ARA_RECORD + "_off_186", ["source_start", "source_end"]].tolist(), [186, 356])
        self.assertEqual(runs.loc[ARA_RECORD + "_off_2734", ["source_start", "source_end"]].tolist(), [2734, 2904])
        self.assertEqual(runs.loc[ARA_RECORD + "_off_2734", "reason"], "GOF|LOF|starrseq_v1")

    def test_entry_columns(self):
        is_second_window = self.front_table["record_name"] == ARA_RECORD + "_off_2734"
        deltas = self.front_table.loc[is_second_window, "fitness_delta_vs_reference"].tolist()
        # minimize: the fitness dropped from 0.4 to 0.2, which is a gain
        self.assertAlmostEqual(deltas[0], 0.2)
        baseline = self.front_table[self.front_table["gene_id"] == NTAB_GENE]
        self.assertEqual(baseline["n_count"].tolist(), [0, 170])


if __name__ == "__main__":
    unittest.main()
