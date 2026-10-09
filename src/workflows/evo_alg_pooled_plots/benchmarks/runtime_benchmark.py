"""Produce the runtime-benchmark tables and figures of the two optimizers.

Thin orchestration script: it loads the benchmark timings, writes the tidy frame
and the summary tables, and draws the figures. All loading and statistics live in
``runtime_benchmark_calc.py``, all drawing in ``runtime_benchmark_plot.py``.

Run with the project's conda environment:

    conda run -n deepCREshap python -m workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark

Two benchmark trees exist, and ``USE_MIXED_HARDWARE_DATA`` picks between them:

- ``False`` (default): the same-hardware runs, one job running every config in
  sequence on one node per device, the GPU one uncontended. Outputs land in
  ``benchmarks_same_hardware`` next to this file (gitignored).
- ``True``: the older HPC-queue runs, whose cells may sit on compute nodes of
  differing hardware and whose GPU cells shared devices.
  Outputs land in ``benchmarks_mixed_hardware``.

Only ``algorithm_seconds`` — the wall time of the optimization itself — is plotted.
``model_load_seconds`` stays in the tidy CSV but is cold-start noise, and the greedy
algorithm reports no evaluation count, so it appears only in the wall-time figures
and never in the cost-per-evaluation ones.

Every figure covers exactly one device: CPU and GPU never share axes. In the
mixed-hardware tree the GPU array jobs let 15 tasks share the GPUs, so those wall
times are contended upper bounds (see ``GPU_CONTENTION_CAVEAT``) and putting them
next to a clean CPU box would invite reading a speedup off a contended number. Each
device therefore gets the same full set of figures, and there the GPU titles say
that they are contended.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark_calc import (
    GPU_CONTENTION_CAVEAT,
    MIXED_HARDWARE_DEVICE_FOLDERS,
    MIXED_HARDWARE_ROOT,
    SAME_HARDWARE_DEVICE_FOLDERS,
    SAME_HARDWARE_ROOT,
    RUNTIME_COLUMN,
    check_expected_cells,
    find_non_monotonic_curves,
    fit_log_log_scaling,
    fit_sweep_scaling,
    flag_poor_power_law_fits,
    load_benchmark_runtimes,
    regime_runtime_ratio,
    summarize_runtimes,
)
from workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark_plot import (
    SWEPT_PARAMETER_LABELS,
    plot_runtime_sweep,
    plot_scaling_fit,
)

# Which benchmark tree to analyze. The same-hardware runs are the default: one job
# ran every config in sequence on one node with one GPU, so the hardware never
# changes between the timed runs and the GPU is uncontended. Set this to True to
# re-analyze the older HPC-queue runs, which also carry the CPU cells.
USE_MIXED_HARDWARE_DATA = False

if USE_MIXED_HARDWARE_DATA:
    BENCHMARK_ROOT = MIXED_HARDWARE_ROOT
    DEVICE_FOLDERS = MIXED_HARDWARE_DEVICE_FOLDERS
    OUTPUT_DIR = Path(__file__).parent / "benchmarks_mixed_hardware"
else:
    BENCHMARK_ROOT = SAME_HARDWARE_ROOT
    DEVICE_FOLDERS = SAME_HARDWARE_DEVICE_FOLDERS
    OUTPUT_DIR = Path(__file__).parent / "benchmarks_same_hardware"

FMT = "png"

# Pivot of the genetic-algorithm sweep: the cell all other cells vary away from.
PIVOT_POPULATION_SIZE = 100
PIVOT_GENERATIONS = 1000

# Only the mixed-hardware GPU cells shared their devices; the same-hardware ones ran
# alone, so their titles must not claim contention.
GPU_TITLE_SUFFIX = " (contended)" if USE_MIXED_HARDWARE_DATA else ""

# Every figure covers one device only, so a contended GPU measurement can never be
# put side by side with a clean CPU one in the same panel.
DEVICES = tuple(sorted(set(DEVICE_FOLDERS.values())))

# Above this many levels on a quantitative x axis, a sweep is drawn as a log-log
# scaling plot rather than as boxplots: with three or more levels the question is
# how the runtime scales, and a fitted exponent answers it where a row of boxes does
# not. At two levels there is no scaling to fit and the replicate spread is the point.
MAX_LEVELS_FOR_BOXPLOT = 2


def cost_sweep_rows(runtimes: pd.DataFrame) -> pd.DataFrame:
    """Drop the genetic algorithm's mutation-cap control, keep every cost sweep.

    The mutation cap means different things to the two algorithms. For the genetic
    algorithm it is a control: the ``maxmut2999`` cell exists to show that lifting
    the cap does *not* change the runtime, so fitting a scaling law across it or
    demanding that it be monotonic would be meaningless. For the greedy algorithm
    the cap is the number of optimization steps, hence its only cost driver, and it
    must be fitted and checked like any other sweep.

    Args:
        runtimes: Tidy frame from ``load_benchmark_runtimes``.

    Returns:
        The frame without the genetic algorithm's mutation-cap rows.
    """
    is_genetic_cap_control = (runtimes["algorithm"] == "ga") & (
        runtimes["swept_parameter"] == "max_number_mutations"
    )
    return pd.DataFrame(runtimes[~is_genetic_cap_control])



def build_mutation_cap_frame(runtimes: pd.DataFrame, device: str) -> pd.DataFrame:
    """Select the genetic-algorithm cells that differ only in the mutation cap.

    The lifted-cap cell (``maxmut2999``) and the pivot of the population sweep
    share the population size and generation count and are both unconstrained, so
    they differ only in ``max_number_mutations``. The two carry different
    ``swept_parameter`` labels, so this relabels both as a mutation-cap sweep to
    put them on one axis.

    Args:
        runtimes: Tidy frame from ``load_benchmark_runtimes``.
        device: Device to select, ``"CPU"`` or ``"GPU"``.

    Returns:
        Frame of the two cells with ``swept_parameter`` set to
        ``"max_number_mutations"`` and ``swept_value`` to the cap.
    """
    selection = runtimes[
        (runtimes["algorithm"] == "ga")
        & (runtimes["device"] == device)
        & (runtimes["regime"] == "unconstrained")
        & (runtimes["population_size"] == PIVOT_POPULATION_SIZE)
        & (runtimes["number_of_generations"] == PIVOT_GENERATIONS)
    ]
    relabelled = selection.copy()
    relabelled["swept_parameter"] = "max_number_mutations"
    relabelled["swept_value"] = relabelled["max_number_mutations"].astype(int)
    return relabelled


def write_tables(runtimes: pd.DataFrame, output_dir: Path) -> None:
    """Write the tidy frame and the summary tables as CSVs.

    Args:
        runtimes: Tidy frame from ``load_benchmark_runtimes``.
        output_dir: Folder to write into.
    """
    runtimes.to_csv(output_dir / "runtime_benchmark_tidy.csv", index=False)
    check_expected_cells(runtimes).to_csv(
        output_dir / "cell_completeness.csv", index=False
    )
    summarize_runtimes(runtimes).to_csv(
        output_dir / "runtime_summary.csv", index=False
    )
    cost_sweeps = cost_sweep_rows(runtimes)
    fits = fit_sweep_scaling(cost_sweeps)
    fits.to_csv(output_dir / "scaling_fits.csv", index=False)
    flag_poor_power_law_fits(fits).to_csv(
        output_dir / "unfittable_curves.csv", index=False
    )
    find_non_monotonic_curves(cost_sweeps).to_csv(
        output_dir / "non_monotonic_steps.csv", index=False
    )
    regime_runtime_ratio(runtimes).to_csv(
        output_dir / "regime_runtime_ratios.csv", index=False
    )
    print(f"Wrote tables into {output_dir}")


def print_fit_diagnostics(runtimes: pd.DataFrame) -> None:
    """Print the sweep curves whose exponent must not be quoted as a scaling law.

    A curve that is not monotonic in the swept parameter, or that a power law fits
    poorly, reports something other than the parameter it sweeps. Printing these
    keeps a bad exponent from being read off ``scaling_fits.csv`` unnoticed.

    Args:
        runtimes: Tidy frame from ``load_benchmark_runtimes``.
    """
    cost_sweeps = cost_sweep_rows(runtimes)
    non_monotonic = find_non_monotonic_curves(cost_sweeps)
    if not non_monotonic.empty:
        print("\nRuntime falls as the swept parameter grows in these steps:")
        print(non_monotonic.to_string(index=False))
    poor_fits = flag_poor_power_law_fits(fit_sweep_scaling(cost_sweeps))
    if not poor_fits.empty:
        print("\nPower law does not describe these curves; do not quote the exponent:")
        print(
            poor_fits[
                [
                    "algorithm",
                    "device",
                    "regime",
                    "swept_parameter",
                    "exponent",
                    "r_squared",
                ]
            ].to_string(index=False)
        )


def device_title_suffix(device: str) -> str:
    """Return the title suffix that names the device and, for GPU, its caveat.

    Args:
        device: ``"CPU"`` or ``"GPU"``.

    Returns:
        The suffix to append to a figure title.
    """
    return f", {device}{GPU_TITLE_SUFFIX if device == 'GPU' else ''}"


def plot_sweep(
    runtimes: pd.DataFrame,
    swept_parameter: str,
    output_dir: Path,
    file_stem: str,
    title: str,
) -> None:
    """Draw one sweep, choosing the plot type from how many levels it covers.

    Every parameter swept here is a quantity, not a category, so a sweep of three or
    more levels is drawn as a log-log scatter of the individual sequences with a
    fitted power law per regime: that shows the *scaling*, which is the question a
    runtime benchmark asks, whereas boxplots side by side leave the reader to
    eyeball the slope. Boxplots are kept only where there is no scaling to show —
    two levels, which is the mutation-cap control, where the point is whether the
    two cells differ at all and the replicate spread is what decides it.

    The exponent's confidence interval and $R^2$ go into the legend, so a curve the
    power law does not fit is visible in the figure itself and not only in
    ``unfittable_curves.csv``.

    Args:
        runtimes: Tidy frame already filtered to one algorithm, device and sweep.
            An empty frame draws nothing.
        swept_parameter: Config field on the x axis, used for its axis label.
        output_dir: Folder to write the figure into.
        file_stem: Basename of the file, without extension.
        title: Figure title.
    """
    if runtimes.empty:
        return
    n_levels = runtimes["swept_value"].nunique()
    if n_levels <= MAX_LEVELS_FOR_BOXPLOT:
        plot_runtime_sweep(
            runtimes,
            swept_parameter=swept_parameter,
            hue_column="regime",
            output_dir=output_dir,
            file_stem=file_stem,
            fmt=FMT,
            title=title,
        )
        return
    fits = {
        regime: fit_log_log_scaling(group["swept_value"], group[RUNTIME_COLUMN])
        for regime, group in runtimes.groupby("regime", observed=True)
    }
    plot_scaling_fit(
        runtimes,
        fits=fits,
        x_column="swept_value",
        hue_column="regime",
        output_dir=output_dir,
        file_stem=file_stem,
        fmt=FMT,
        title=title,
        xlabel=SWEPT_PARAMETER_LABELS.get(swept_parameter, swept_parameter),
    )


def plot_device_figures(runtimes: pd.DataFrame, output_dir: Path, device: str) -> None:
    """Draw the full set of sweep figures for one device.

    CPU and GPU are never drawn on the same axes. The GPU array jobs let 15 tasks
    share the GPUs, so a CPU box and a GPU box side by side would invite reading a
    speedup off a contended measurement; keeping the two apart means each figure
    shows one measurement regime only, and the GPU titles carry the caveat.

    Within a device, the two mutation regimes are the hue, and the greedy algorithm
    gets its own figure rather than sharing one with the genetic algorithm.

    Args:
        runtimes: Tidy frame from ``load_benchmark_runtimes``.
        output_dir: Folder to write the figures into.
        device: Device to draw, ``"CPU"`` or ``"GPU"``.
    """
    suffix = device_title_suffix(device)
    stem_device = device.lower()

    genetic = runtimes[
        (runtimes["algorithm"] == "ga") & (runtimes["device"] == device)
    ]
    for swept_parameter in ("population_size", "number_of_generations"):
        plot_sweep(
            genetic[genetic["swept_parameter"] == swept_parameter],
            swept_parameter=swept_parameter,
            output_dir=output_dir,
            file_stem=f"ga_{stem_device}_{swept_parameter}_sweep",
            title=(
                f"Genetic algorithm{suffix}: runtime vs "
                f"{swept_parameter.replace('_', ' ')}"
            ),
        )

    plot_sweep(
        build_mutation_cap_frame(runtimes, device),
        swept_parameter="max_number_mutations",
        output_dir=output_dir,
        file_stem=f"ga_{stem_device}_mutation_cap_control",
        title=(
            f"Genetic algorithm{suffix}: lifting the mutation cap at the sweep pivot"
        ),
    )

    greedy = runtimes[
        (runtimes["algorithm"] == "greedy") & (runtimes["device"] == device)
    ]
    plot_sweep(
        greedy,
        swept_parameter="max_number_mutations",
        output_dir=output_dir,
        file_stem=f"greedy_{stem_device}_mutation_cap_sweep",
        title=f"Greedy algorithm{suffix}: runtime vs mutation cap",
    )


def plot_evaluation_scaling(
    runtimes: pd.DataFrame, output_dir: Path, device: str
) -> None:
    """Draw runtime against the genetic algorithm's evaluation count for one device.

    Both regimes run the same number of model evaluations at a given population
    size and generation count, so the vertical offset between the two series is the
    cost of the constraint bookkeeping and the slope is the cost per evaluation.
    The mutation-cap control cells are excluded, since they are not part of the
    cost sweep.

    Args:
        runtimes: Tidy frame from ``load_benchmark_runtimes``.
        output_dir: Folder to write the figure into.
        device: Device to draw, ``"CPU"`` or ``"GPU"``.
    """
    cost_sweeps = cost_sweep_rows(runtimes)
    selection = cost_sweeps[
        (cost_sweeps["algorithm"] == "ga") & (cost_sweeps["device"] == device)
    ]
    if selection.empty:
        return
    fits = {
        regime: fit_log_log_scaling(group["n_evaluations"], group[RUNTIME_COLUMN])
        for regime, group in selection.groupby("regime", observed=True)
    }
    plot_scaling_fit(
        selection,
        fits=fits,
        x_column="n_evaluations",
        hue_column="regime",
        output_dir=output_dir,
        file_stem=f"ga_{device.lower()}_evaluation_scaling",
        fmt=FMT,
        title=(
            f"Genetic algorithm{device_title_suffix(device)}: runtime vs model "
            "evaluations"
        ),
    )


def main(
    benchmark_root: Path = BENCHMARK_ROOT,
    device_folders: Dict[str, str] = DEVICE_FOLDERS,
    output_dir: Path = OUTPUT_DIR,
    devices: Tuple[str, ...] = DEVICES,
) -> None:
    """Load the benchmark, write the tables and draw every figure.

    Args:
        benchmark_root: Folder holding the benchmark run folders.
        device_folders: Mapping of subfolder of ``benchmark_root`` to device.
        output_dir: Folder to write tables and figures into; created if missing.
        devices: Devices to draw figures for.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    runtimes = load_benchmark_runtimes(benchmark_root, device_folders)
    print(f"Loaded {len(runtimes)} timed runs from {benchmark_root}")
    print(check_expected_cells(runtimes).to_string(index=False))
    if USE_MIXED_HARDWARE_DATA:
        print(f"Caveat: {GPU_CONTENTION_CAVEAT}")
    write_tables(runtimes, output_dir)
    print_fit_diagnostics(runtimes)
    for device in devices:
        plot_device_figures(runtimes, output_dir, device)
        plot_evaluation_scaling(runtimes, output_dir, device)


if __name__ == "__main__":
    main()
