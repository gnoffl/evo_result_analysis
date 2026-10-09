"""Unit tests for the runtime-benchmark loading and statistics module."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark_calc import (
    ScalingFit,
    check_expected_cells,
    find_non_monotonic_curves,
    fit_log_log_scaling,
    fit_sweep_scaling,
    flag_poor_power_law_fits,
    load_benchmark_runtimes,
    load_timings_file,
    parse_run_directory_name,
    regime_runtime_ratio,
    split_label,
    summarize_runtimes,
)


def make_genetic_record(
    sequence_name: str,
    population_size: int,
    number_of_generations: int,
    algorithm_seconds: float,
    max_number_mutations: int = 20,
    n_evaluations: Optional[int] = None,
) -> Dict[str, object]:
    """Build one genetic-algorithm run record as the algorithm writes it.

    Args:
        sequence_name: Name of the timed sequence.
        population_size: Population size of the run.
        number_of_generations: Number of generations of the run.
        algorithm_seconds: Wall time of the optimization.
        max_number_mutations: Mutation cap of the run.
        n_evaluations: Evaluation count; when None the consistent value
            ``population_size * (number_of_generations + 1)`` is used.

    Returns:
        The record as a dictionary.
    """
    if n_evaluations is None:
        n_evaluations = population_size * (number_of_generations + 1)
    return {
        "sequence_name": sequence_name,
        "population_size": population_size,
        "number_of_generations": number_of_generations,
        "max_number_mutations": max_number_mutations,
        "n_evaluations": n_evaluations,
        "realized_mutations": max_number_mutations,
        "model_load_seconds": 1.5,
        "algorithm_seconds": algorithm_seconds,
    }


def make_greedy_record(
    sequence_name: str, max_number_mutations: int, algorithm_seconds: float
) -> Dict[str, object]:
    """Build one greedy run record as the algorithm writes it (no evaluation count).

    Args:
        sequence_name: Name of the timed sequence.
        max_number_mutations: Mutation cap of the run.
        algorithm_seconds: Wall time of the optimization.

    Returns:
        The record as a dictionary.
    """
    return {
        "sequence_name": sequence_name,
        "max_number_mutations": max_number_mutations,
        "realized_mutations": max_number_mutations,
        "model_load_seconds": 2.0,
        "algorithm_seconds": algorithm_seconds,
    }


def write_run_folder(
    device_folder: Path, directory_name: str, records: List[Dict[str, object]]
) -> Path:
    """Write one run folder holding a ``timings.json`` with the given records.

    Args:
        device_folder: Folder the run folder is created in.
        directory_name: Basename of the run folder.
        records: Run records to put into the ``runs`` list.

    Returns:
        Path of the created run folder.
    """
    run_folder = device_folder / directory_name
    run_folder.mkdir(parents=True)
    total = sum(float(record["algorithm_seconds"]) for record in records)
    with open(run_folder / "timings.json", "w", encoding="utf-8") as handle:
        json.dump({"total_seconds": total, "runs": records}, handle)
    return run_folder


class ParseRunDirectoryNameTest(unittest.TestCase):
    """Tests for splitting a run folder name into its parts."""

    def test_splits_algorithm_label_and_regime(self) -> None:
        # Arrange / Act
        parsed = parse_run_directory_name("ga_pop50_natural_260821_213811_209722")
        # Assert
        self.assertEqual(parsed, ("ga", "pop50", "natural"))

    def test_splits_greedy_unconstrained(self) -> None:
        # Arrange / Act
        parsed = parse_run_directory_name(
            "greedy_maxmut100_unconstrained_260821_214210_874872"
        )
        # Assert
        self.assertEqual(parsed, ("greedy", "maxmut100", "unconstrained"))

    def test_rejects_unknown_regime(self) -> None:
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            parse_run_directory_name("ga_pop50_synthetic_260821_213811_209722")

    def test_rejects_unknown_algorithm(self) -> None:
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            parse_run_directory_name("annealing_pop50_natural_260821_213811_209722")

    def test_rejects_short_name(self) -> None:
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            parse_run_directory_name("ga_pop50_natural")


class SplitLabelTest(unittest.TestCase):
    """Tests for mapping a sweep label onto a config field and value."""

    def test_known_prefixes(self) -> None:
        # Arrange / Act / Assert
        self.assertEqual(split_label("pop25"), ("population_size", 25))
        self.assertEqual(split_label("gen2000"), ("number_of_generations", 2000))
        self.assertEqual(split_label("maxmut2999"), ("max_number_mutations", 2999))

    def test_rejects_unknown_prefix(self) -> None:
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            split_label("mutrate5")

    def test_rejects_non_integer_value(self) -> None:
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            split_label("popmany")


class LoadTimingsFileTest(unittest.TestCase):
    """Tests for loading a single timings.json, including its consistency checks."""

    def test_builds_one_row_per_sequence(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as temporary_dir:
            device_folder = Path(temporary_dir)
            run_folder = write_run_folder(
                device_folder,
                "ga_pop50_natural_260821_213811_209722",
                [
                    make_genetic_record("benchmark_00", 50, 1000, 500.0),
                    make_genetic_record("benchmark_01", 50, 1000, 520.0),
                ],
            )
            # Act
            frame = load_timings_file(
                run_folder / "timings.json", "ga", "CPU", "pop50", "natural"
            )
        # Assert
        self.assertEqual(len(frame), 2)
        self.assertEqual(list(frame["device"].unique()), ["CPU"])
        self.assertEqual(list(frame["regime"].unique()), ["natural"])
        self.assertEqual(list(frame["swept_parameter"].unique()), ["population_size"])
        self.assertEqual(list(frame["swept_value"].unique()), [50])
        self.assertEqual(frame["n_evaluations"].tolist(), [50050, 50050])

    def test_greedy_records_have_no_evaluation_count(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as temporary_dir:
            device_folder = Path(temporary_dir)
            run_folder = write_run_folder(
                device_folder,
                "greedy_maxmut20_natural_260821_214211_107741",
                [make_greedy_record("benchmark_00", 20, 15.0)],
            )
            # Act
            frame = load_timings_file(
                run_folder / "timings.json", "greedy", "CPU", "maxmut20", "natural"
            )
        # Assert
        self.assertTrue(frame["n_evaluations"].isna().all())
        self.assertTrue(frame["population_size"].isna().all())
        self.assertEqual(frame["swept_value"].tolist(), [20])

    def test_rejects_record_contradicting_the_label(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as temporary_dir:
            device_folder = Path(temporary_dir)
            run_folder = write_run_folder(
                device_folder,
                "ga_pop50_natural_260821_213811_209722",
                [make_genetic_record("benchmark_00", 200, 1000, 500.0)],
            )
            # Act / Assert
            with self.assertRaises(ValueError) as caught:
                load_timings_file(
                    run_folder / "timings.json", "ga", "CPU", "pop50", "natural"
                )
        self.assertIn("population_size=200", str(caught.exception))

    def test_rejects_inconsistent_evaluation_count(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as temporary_dir:
            device_folder = Path(temporary_dir)
            run_folder = write_run_folder(
                device_folder,
                "ga_pop50_natural_260821_213811_209722",
                [
                    make_genetic_record(
                        "benchmark_00", 50, 1000, 500.0, n_evaluations=12345
                    )
                ],
            )
            # Act / Assert
            with self.assertRaises(ValueError) as caught:
                load_timings_file(
                    run_folder / "timings.json", "ga", "CPU", "pop50", "natural"
                )
        self.assertIn("12345", str(caught.exception))

    def test_rejects_empty_run_list(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as temporary_dir:
            device_folder = Path(temporary_dir)
            run_folder = write_run_folder(
                device_folder, "ga_pop50_natural_260821_213811_209722", []
            )
            # Act / Assert
            with self.assertRaises(ValueError):
                load_timings_file(
                    run_folder / "timings.json", "ga", "CPU", "pop50", "natural"
                )


class LoadBenchmarkRuntimesTest(unittest.TestCase):
    """Tests for walking the benchmark tree."""

    DEVICE_FOLDERS = {"GA": "CPU", "greedy": "CPU"}

    def _build_tree(self, root: Path) -> None:
        """Write a two-device benchmark tree with two cells each."""
        genetic_folder = root / "GA"
        write_run_folder(
            genetic_folder,
            "ga_pop50_unconstrained_260821_213811_209722",
            [make_genetic_record(f"benchmark_{index:02d}", 50, 1000, 100.0 + index)
             for index in range(3)],
        )
        write_run_folder(
            genetic_folder,
            "ga_pop100_unconstrained_260821_213811_193061",
            [make_genetic_record(f"benchmark_{index:02d}", 100, 1000, 200.0 + index)
             for index in range(3)],
        )
        greedy_folder = root / "greedy"
        write_run_folder(
            greedy_folder,
            "greedy_maxmut20_natural_260821_214211_107741",
            [make_greedy_record(f"benchmark_{index:02d}", 20, 15.0 + index)
             for index in range(3)],
        )

    def test_loads_every_cell_and_derives_seconds_per_evaluation(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._build_tree(root)
            # Act
            runtimes = load_benchmark_runtimes(root, self.DEVICE_FOLDERS)
        # Assert
        self.assertEqual(len(runtimes), 9)
        self.assertEqual(set(runtimes["algorithm"]), {"ga", "greedy"})
        genetic_rows = runtimes[runtimes["algorithm"] == "ga"]
        expected = genetic_rows["algorithm_seconds"] / genetic_rows["n_evaluations"]
        pd.testing.assert_series_equal(
            genetic_rows["seconds_per_evaluation"], expected, check_names=False
        )
        greedy_rows = runtimes[runtimes["algorithm"] == "greedy"]
        self.assertTrue(greedy_rows["seconds_per_evaluation"].isna().all())

    def test_missing_timings_file_raises(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._build_tree(root)
            (root / "GA" / "ga_pop50_unconstrained_260821_213811_209722" / "timings.json").unlink()
            # Act / Assert
            with self.assertRaises(FileNotFoundError):
                load_benchmark_runtimes(root, self.DEVICE_FOLDERS)

    def test_missing_device_folder_raises(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "GA").mkdir()
            write_run_folder(
                root / "GA",
                "ga_pop50_unconstrained_260821_213811_209722",
                [make_genetic_record("benchmark_00", 50, 1000, 100.0)],
            )
            # Act / Assert
            with self.assertRaises(FileNotFoundError):
                load_benchmark_runtimes(root, self.DEVICE_FOLDERS)

    def test_one_folder_may_hold_both_algorithms(self) -> None:
        """The same-hardware tree keeps ga and greedy runs in one device folder."""
        # Arrange
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            write_run_folder(
                root / "gpu",
                "ga_pop50_unconstrained_260821_213811_209722",
                [make_genetic_record("benchmark_00", 50, 1000, 100.0)],
            )
            write_run_folder(
                root / "gpu",
                "greedy_maxmut20_natural_260821_214211_107742",
                [make_greedy_record("benchmark_00", 20, 15.0)],
            )
            # Act
            runtimes = load_benchmark_runtimes(root, {"gpu": "GPU"})
        # Assert
        self.assertEqual(set(runtimes["algorithm"]), {"ga", "greedy"})
        self.assertEqual(set(runtimes["device"]), {"GPU"})


class CheckExpectedCellsTest(unittest.TestCase):
    """Tests for the completeness report."""

    def test_reports_shortfall(self) -> None:
        # Arrange
        runtimes = pd.DataFrame(
            {
                "algorithm": ["ga"] * 4,
                "device": ["CPU"] * 4,
                "run_directory": ["cell_a", "cell_a", "cell_b", "cell_b"],
            }
        )
        # Act
        report = check_expected_cells(runtimes, {"ga": 15})
        # Assert
        self.assertEqual(len(report), 1)
        self.assertEqual(report.loc[0, "n_cells"], 2)
        self.assertEqual(report.loc[0, "n_expected_cells"], 15)
        self.assertFalse(bool(report.loc[0, "complete"]))
        self.assertEqual(report.loc[0, "n_sequences_min"], 2)


class SummarizeRuntimesTest(unittest.TestCase):
    """Tests for the per-cell median and interquartile summary."""

    def test_median_and_iqr(self) -> None:
        # Arrange
        runtimes = pd.DataFrame(
            {
                "algorithm": ["ga"] * 5,
                "device": ["CPU"] * 5,
                "regime": ["unconstrained"] * 5,
                "swept_parameter": ["population_size"] * 5,
                "swept_value": [50] * 5,
                "algorithm_seconds": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        # Act
        summary = summarize_runtimes(runtimes)
        # Assert
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.loc[0, "n"], 5)
        self.assertEqual(summary.loc[0, "median"], 30.0)
        self.assertEqual(summary.loc[0, "q1"], 20.0)
        self.assertEqual(summary.loc[0, "q3"], 40.0)
        self.assertEqual(summary.loc[0, "iqr"], 20.0)
        self.assertEqual(summary.loc[0, "minimum"], 10.0)
        self.assertEqual(summary.loc[0, "maximum"], 50.0)


class FitLogLogScalingTest(unittest.TestCase):
    """Tests for the power-law fit."""

    def test_recovers_exact_power_law(self) -> None:
        # Arrange: y = 3 * x ** 2 exactly
        x_values = [1.0, 2.0, 4.0, 8.0, 16.0]
        y_values = [3.0 * value**2 for value in x_values]
        # Act
        fit = fit_log_log_scaling(x_values, y_values)
        # Assert
        self.assertIsInstance(fit, ScalingFit)
        self.assertAlmostEqual(fit.exponent, 2.0, places=10)
        self.assertAlmostEqual(fit.coefficient, 3.0, places=8)
        self.assertAlmostEqual(fit.r_squared, 1.0, places=10)
        self.assertEqual(fit.n_points, 5)
        self.assertAlmostEqual(fit.exponent_ci_low, 2.0, places=6)
        self.assertAlmostEqual(fit.exponent_ci_high, 2.0, places=6)

    def test_confidence_interval_brackets_the_exponent(self) -> None:
        # Arrange: linear scaling with a little noise
        x_values = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        y_values = np.array([1.05, 1.9, 4.2, 7.8, 16.4, 31.0])
        # Act
        fit = fit_log_log_scaling(x_values, y_values)
        # Assert
        self.assertLess(fit.exponent_ci_low, fit.exponent)
        self.assertLess(fit.exponent, fit.exponent_ci_high)
        self.assertLess(fit.exponent_ci_low, 1.0)
        self.assertGreater(fit.exponent_ci_high, 1.0)

    def test_rejects_too_few_points(self) -> None:
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            fit_log_log_scaling([1.0, 2.0], [1.0, 2.0])

    def test_rejects_non_positive_values(self) -> None:
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            fit_log_log_scaling([1.0, 2.0, 0.0], [1.0, 2.0, 3.0])

    def test_rejects_constant_x(self) -> None:
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            fit_log_log_scaling([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])

    def test_rejects_mismatched_lengths(self) -> None:
        # Arrange / Act / Assert
        with self.assertRaises(ValueError):
            fit_log_log_scaling([1.0, 2.0, 4.0], [1.0, 2.0])


class FitSweepScalingTest(unittest.TestCase):
    """Tests for fitting one power law per sweep curve."""

    def _frame(self) -> pd.DataFrame:
        rows = []
        for regime, coefficient in (("unconstrained", 1.0), ("natural", 4.0)):
            for swept_value in (25, 50, 100, 200):
                for index in range(3):
                    rows.append(
                        {
                            "algorithm": "ga",
                            "device": "CPU",
                            "regime": regime,
                            "swept_parameter": "population_size",
                            "swept_value": swept_value,
                            "algorithm_seconds": coefficient * swept_value,
                        }
                    )
        return pd.DataFrame(rows)

    def test_one_row_per_curve_with_linear_exponent(self) -> None:
        # Arrange
        frame = self._frame()
        # Act
        fits = fit_sweep_scaling(frame)
        # Assert
        self.assertEqual(len(fits), 2)
        for _, row in fits.iterrows():
            self.assertAlmostEqual(row["exponent"], 1.0, places=10)
            self.assertAlmostEqual(row["r_squared"], 1.0, places=10)
            self.assertEqual(row["n_levels"], 4)
            self.assertEqual(row["n_points"], 12)
        coefficients = dict(zip(fits["regime"], fits["coefficient"]))
        self.assertAlmostEqual(coefficients["unconstrained"], 1.0, places=8)
        self.assertAlmostEqual(coefficients["natural"], 4.0, places=8)

    def test_skips_single_level_curves(self) -> None:
        # Arrange
        frame = self._frame()
        single_level = frame[frame["swept_value"] == 50]
        # Act
        fits = fit_sweep_scaling(single_level)
        # Assert
        self.assertTrue(fits.empty)


class FlagPoorPowerLawFitsTest(unittest.TestCase):
    """Tests for flagging curves a power law does not describe."""

    def test_returns_only_rows_below_the_floor_sorted_ascending(self) -> None:
        # Arrange
        fits = pd.DataFrame(
            {
                "swept_parameter": ["a", "b", "c"],
                "r_squared": [0.99, 0.30, 0.66],
            }
        )
        # Act
        poor = flag_poor_power_law_fits(fits, r_squared_floor=0.95)
        # Assert
        self.assertEqual(poor["swept_parameter"].tolist(), ["b", "c"])

    def test_empty_when_every_curve_fits(self) -> None:
        # Arrange
        fits = pd.DataFrame({"swept_parameter": ["a"], "r_squared": [0.999]})
        # Act
        poor = flag_poor_power_law_fits(fits, r_squared_floor=0.95)
        # Assert
        self.assertTrue(poor.empty)

    def test_empty_input_passes_through(self) -> None:
        # Arrange / Act / Assert
        self.assertTrue(flag_poor_power_law_fits(pd.DataFrame()).empty)


class FindNonMonotonicCurvesTest(unittest.TestCase):
    """Tests for detecting sweep steps where the runtime falls."""

    def _frame(self, medians: Dict[int, float]) -> pd.DataFrame:
        rows = []
        for swept_value, runtime in medians.items():
            for index in range(2):
                rows.append(
                    {
                        "algorithm": "ga",
                        "device": "CPU",
                        "regime": "unconstrained",
                        "swept_parameter": "population_size",
                        "swept_value": swept_value,
                        "algorithm_seconds": runtime,
                    }
                )
        return pd.DataFrame(rows)

    def test_reports_the_offending_step(self) -> None:
        # Arrange: 25 -> 50 falls, 50 -> 100 rises
        frame = self._frame({50: 98.0, 25: 138.0, 100: 198.0})
        # Act
        offenders = find_non_monotonic_curves(frame)
        # Assert
        self.assertEqual(len(offenders), 1)
        self.assertEqual(offenders.loc[0, "swept_value_from"], 25)
        self.assertEqual(offenders.loc[0, "swept_value_to"], 50)
        self.assertEqual(offenders.loc[0, "median_from"], 138.0)
        self.assertEqual(offenders.loc[0, "median_to"], 98.0)

    def test_monotonic_curve_yields_nothing(self) -> None:
        # Arrange
        frame = self._frame({25: 50.0, 50: 100.0, 100: 200.0})
        # Act
        offenders = find_non_monotonic_curves(frame)
        # Assert
        self.assertTrue(offenders.empty)

    def test_equal_medians_do_not_count_as_a_fall(self) -> None:
        # Arrange
        frame = self._frame({25: 100.0, 50: 100.0})
        # Act
        offenders = find_non_monotonic_curves(frame)
        # Assert
        self.assertTrue(offenders.empty)


class RegimeRuntimeRatioTest(unittest.TestCase):
    """Tests for the natural-over-unconstrained runtime ratio."""

    def test_ratio_per_cell(self) -> None:
        # Arrange
        runtimes = pd.DataFrame(
            {
                "algorithm": ["ga"] * 4,
                "device": ["CPU"] * 4,
                "label": ["pop50", "pop50", "pop100", "pop100"],
                "regime": ["unconstrained", "natural", "unconstrained", "natural"],
                "algorithm_seconds": [100.0, 400.0, 200.0, 600.0],
            }
        )
        # Act
        ratios = regime_runtime_ratio(runtimes)
        # Assert
        by_label = ratios.set_index("label")["ratio_natural_over_unconstrained"]
        self.assertAlmostEqual(by_label["pop50"], 4.0)
        self.assertAlmostEqual(by_label["pop100"], 3.0)

    def test_cells_with_one_regime_are_dropped(self) -> None:
        # Arrange
        runtimes = pd.DataFrame(
            {
                "algorithm": ["ga"] * 3,
                "device": ["CPU"] * 3,
                "label": ["pop50", "pop50", "maxmut2999"],
                "regime": ["unconstrained", "natural", "unconstrained"],
                "algorithm_seconds": [100.0, 400.0, 170.0],
            }
        )
        # Act
        ratios = regime_runtime_ratio(runtimes)
        # Assert
        self.assertEqual(ratios["label"].tolist(), ["pop50"])


if __name__ == "__main__":
    unittest.main()
