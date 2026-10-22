"""Unit tests for the runtime-benchmark orchestration script."""

import tempfile
import unittest
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

import pandas as pd

from workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark import (
    build_mutation_cap_frame,
    cost_sweep_rows,
    device_title_suffix,
    plot_device_figures,
    plot_sweep,
    write_tables,
)


def make_genetic_frame() -> pd.DataFrame:
    """Build a tidy frame holding the pivot cell, a non-pivot cell and the cap cell.

    Returns:
        Frame with three cells of two sequences each, all for the genetic algorithm
        on CPU.
    """
    cells = (
        # label, regime, population_size, generations, cap, swept parameter/value
        ("pop100", "unconstrained", 100, 1000, 20, "population_size", 100),
        ("pop100", "natural", 100, 1000, 20, "population_size", 100),
        ("pop200", "unconstrained", 200, 1000, 20, "population_size", 200),
        ("gen200", "unconstrained", 100, 200, 20, "number_of_generations", 200),
        ("maxmut2999", "unconstrained", 100, 1000, 2999, "max_number_mutations", 2999),
    )
    rows = []
    for label, regime, population, generations, cap, parameter, value in cells:
        for index in range(2):
            rows.append(
                {
                    "algorithm": "ga",
                    "device": "CPU",
                    "regime": regime,
                    "label": label,
                    "swept_parameter": parameter,
                    "swept_value": value,
                    "population_size": population,
                    "number_of_generations": generations,
                    "max_number_mutations": cap,
                    "n_evaluations": population * (generations + 1),
                    "realized_mutations": min(cap, 20),
                    "sequence_name": f"benchmark_{index:02d}",
                    "algorithm_seconds": 100.0 + index,
                    "model_load_seconds": 1.0,
                    "seconds_per_evaluation": 0.001,
                    "run_directory": f"ga_{label}_{regime}_260821_213811_00000{index}",
                }
            )
    return pd.DataFrame(rows)


class BuildMutationCapFrameTest(unittest.TestCase):
    """Tests for selecting the cells that differ only in the mutation cap."""

    def test_selects_pivot_and_lifted_cap_cells_only(self) -> None:
        # Arrange
        frame = make_genetic_frame()
        # Act
        cap_frame = build_mutation_cap_frame(frame, "CPU")
        # Assert
        self.assertEqual(sorted(cap_frame["label"].unique()), ["maxmut2999", "pop100"])
        self.assertEqual(list(cap_frame["swept_parameter"].unique()),
                         ["max_number_mutations"])
        self.assertEqual(sorted(cap_frame["swept_value"].unique()), [20, 2999])
        self.assertEqual(list(cap_frame["regime"].unique()), ["unconstrained"])

    def test_does_not_mutate_the_input_frame(self) -> None:
        # Arrange
        frame = make_genetic_frame()
        original = frame.copy()
        # Act
        build_mutation_cap_frame(frame, "CPU")
        # Assert
        pd.testing.assert_frame_equal(frame, original)

    def test_other_device_yields_empty_frame(self) -> None:
        # Arrange
        frame = make_genetic_frame()
        # Act
        cap_frame = build_mutation_cap_frame(frame, "GPU")
        # Assert
        self.assertTrue(cap_frame.empty)


class CostSweepRowsTest(unittest.TestCase):
    """Tests that the cap is a control for the genetic algorithm but a sweep for greedy."""

    def _frame(self) -> pd.DataFrame:
        genetic = make_genetic_frame()
        greedy = genetic[genetic["label"] == "maxmut2999"].copy()
        greedy["algorithm"] = "greedy"
        greedy["label"] = "maxmut100"
        greedy["swept_value"] = 100
        return pd.concat([genetic, greedy], ignore_index=True)

    def test_drops_the_genetic_cap_control(self) -> None:
        # Arrange
        frame = self._frame()
        # Act
        kept = cost_sweep_rows(frame)
        # Assert
        genetic_kept = kept[kept["algorithm"] == "ga"]
        self.assertNotIn("max_number_mutations", set(genetic_kept["swept_parameter"]))

    def test_keeps_the_greedy_cap_sweep(self) -> None:
        # Arrange
        frame = self._frame()
        # Act
        kept = cost_sweep_rows(frame)
        # Assert
        greedy_kept = kept[kept["algorithm"] == "greedy"]
        self.assertFalse(greedy_kept.empty)
        self.assertEqual(
            set(greedy_kept["swept_parameter"]), {"max_number_mutations"}
        )

    def test_does_not_mutate_the_input_frame(self) -> None:
        # Arrange
        frame = self._frame()
        original = frame.copy()
        # Act
        cost_sweep_rows(frame)
        # Assert
        pd.testing.assert_frame_equal(frame, original)


class DeviceTitleSuffixTest(unittest.TestCase):
    """Tests for the device suffix that carries the GPU caveat."""

    def test_suffix_names_the_device(self) -> None:
        # Arrange / Act / Assert
        self.assertEqual(device_title_suffix("CPU"), ", CPU")
        self.assertIn("GPU", device_title_suffix("GPU"))

    def test_gpu_titles_carry_the_caveat_only_for_the_contended_data(self) -> None:
        # Arrange
        module = "workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark"
        # Act / Assert
        with patch(f"{module}.GPU_TITLE_SUFFIX", " (contended)"):
            self.assertIn("contended", device_title_suffix("GPU"))
            self.assertNotIn("contended", device_title_suffix("CPU"))
        with patch(f"{module}.GPU_TITLE_SUFFIX", ""):
            self.assertEqual(device_title_suffix("GPU"), ", GPU")


class PlotSweepTest(unittest.TestCase):
    """Tests that the plot type follows the number of levels on the x axis."""

    def _sweep_frame(self, levels: Sequence[int]) -> pd.DataFrame:
        rows = []
        for level_index, regime in enumerate(("unconstrained", "natural")):
            for swept_value in levels:
                for sequence_index in range(4):
                    rows.append(
                        {
                            "algorithm": "ga",
                            "device": "CPU",
                            "regime": regime,
                            "swept_parameter": "population_size",
                            "swept_value": swept_value,
                            "algorithm_seconds": (level_index + 1)
                            * swept_value
                            * (1.0 + 0.02 * sequence_index),
                        }
                    )
        return pd.DataFrame(rows)

    def test_three_levels_draw_a_scaling_plot(self) -> None:
        # Arrange
        frame = self._sweep_frame([25, 50, 100])
        with patch(
            "workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark.plot_scaling_fit"
        ) as scaling, patch(
            "workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark.plot_runtime_sweep"
        ) as boxes:
            # Act
            plot_sweep(frame, "population_size", Path("/tmp"), "stem", "title")
        # Assert
        scaling.assert_called_once()
        boxes.assert_not_called()
        self.assertEqual(
            set(scaling.call_args.kwargs["fits"]), {"unconstrained", "natural"}
        )
        self.assertEqual(scaling.call_args.kwargs["x_column"], "swept_value")
        self.assertEqual(scaling.call_args.kwargs["xlabel"], "Population size")

    def test_two_levels_draw_boxplots(self) -> None:
        # Arrange
        frame = self._sweep_frame([25, 50])
        with patch(
            "workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark.plot_scaling_fit"
        ) as scaling, patch(
            "workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark.plot_runtime_sweep"
        ) as boxes:
            # Act
            plot_sweep(frame, "population_size", Path("/tmp"), "stem", "title")
        # Assert
        boxes.assert_called_once()
        scaling.assert_not_called()

    def test_empty_frame_draws_nothing(self) -> None:
        # Arrange
        frame = self._sweep_frame([25, 50, 100]).iloc[0:0]
        with patch(
            "workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark.plot_scaling_fit"
        ) as scaling, patch(
            "workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark.plot_runtime_sweep"
        ) as boxes:
            # Act
            plot_sweep(frame, "population_size", Path("/tmp"), "stem", "title")
        # Assert
        scaling.assert_not_called()
        boxes.assert_not_called()

    def test_scaling_plot_recovers_the_exponent_per_regime(self) -> None:
        # Arrange: both regimes are exactly linear in the swept value
        frame = self._sweep_frame([25, 50, 100, 200])
        with patch(
            "workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark.plot_scaling_fit"
        ) as scaling:
            # Act
            plot_sweep(frame, "population_size", Path("/tmp"), "stem", "title")
        # Assert
        for fit in scaling.call_args.kwargs["fits"].values():
            self.assertAlmostEqual(fit.exponent, 1.0, places=6)


class PlotDeviceFiguresTest(unittest.TestCase):
    """Tests that each device gets its own figures and never shares one."""

    def _frame_with_both_devices(self) -> pd.DataFrame:
        cpu = make_genetic_frame()
        gpu = cpu.copy()
        gpu["device"] = "GPU"
        greedy = cpu[cpu["label"] == "pop100"].copy()
        greedy["algorithm"] = "greedy"
        greedy["swept_parameter"] = "max_number_mutations"
        greedy["swept_value"] = 20
        return pd.concat([cpu, gpu, greedy], ignore_index=True)

    def test_file_names_are_device_specific(self) -> None:
        # Arrange
        frame = self._frame_with_both_devices()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            # Act
            plot_device_figures(frame, output_dir, "CPU")
            cpu_files = sorted(path.name for path in output_dir.iterdir())
            plot_device_figures(frame, output_dir, "GPU")
            all_files = sorted(path.name for path in output_dir.iterdir())
        # Assert
        gpu_files = [name for name in all_files if name not in cpu_files]
        self.assertTrue(cpu_files)
        self.assertTrue(gpu_files)
        self.assertTrue(all("_cpu_" in name for name in cpu_files))
        self.assertTrue(all("_gpu_" in name for name in gpu_files))

    def test_greedy_and_genetic_get_separate_files(self) -> None:
        # Arrange
        frame = self._frame_with_both_devices()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            # Act
            plot_device_figures(frame, output_dir, "CPU")
            written = sorted(path.name for path in output_dir.iterdir())
        # Assert
        self.assertTrue(any(name.startswith("greedy_cpu_") for name in written))
        self.assertTrue(any(name.startswith("ga_cpu_") for name in written))

    def test_device_absent_from_the_frame_writes_nothing(self) -> None:
        # Arrange
        frame = make_genetic_frame()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            # Act
            plot_device_figures(frame, output_dir, "GPU")
            # Assert
            self.assertEqual(list(output_dir.iterdir()), [])


class WriteTablesTest(unittest.TestCase):
    """Tests for the CSV outputs."""

    def test_writes_every_expected_csv(self) -> None:
        # Arrange
        frame = make_genetic_frame()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            # Act
            write_tables(frame, output_dir)
            # Assert
            written = sorted(path.name for path in output_dir.iterdir())
        self.assertEqual(
            written,
            [
                "cell_completeness.csv",
                "non_monotonic_steps.csv",
                "regime_runtime_ratios.csv",
                "runtime_benchmark_tidy.csv",
                "runtime_summary.csv",
                "scaling_fits.csv",
                "unfittable_curves.csv",
            ],
        )

    def test_scaling_fits_exclude_the_mutation_cap_control(self) -> None:
        # Arrange
        frame = make_genetic_frame()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            # Act
            write_tables(frame, output_dir)
            fits = pd.read_csv(output_dir / "scaling_fits.csv")
        # Assert
        if not fits.empty:
            self.assertNotIn("max_number_mutations", set(fits["swept_parameter"]))


if __name__ == "__main__":
    unittest.main()
