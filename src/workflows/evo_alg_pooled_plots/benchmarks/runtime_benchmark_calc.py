"""Loading and summary statistics for the runtime benchmark of the two optimizers.

This module is the calculation half of the runtime-benchmark toolbox: it reads the
``timings.json`` files written by the benchmark runs, assembles them into one tidy
frame, summarizes the per-cell runtime distributions, and fits power-law scaling
exponents. It contains **no plotting and no orchestration** — figures live in
``runtime_benchmark_plot.py`` and the concrete analysis in ``runtime_benchmark.py``.

Benchmark design (``write_run_script.benchmark_runtime_script`` in the *Evolution*
package): 30 synthetic sequences of 3020 bp are optimized by each of two
algorithms. The genetic algorithm (``ga``) is swept over population size (with the
number of generations fixed at 1000) and over the number of generations (with the
population fixed at 100), plus one extra cell whose mutation cap is lifted to 2999
to show that the cap does not drive its runtime. The greedy algorithm (``greedy``)
is swept over its mutation cap. Every cell is run twice: ``unconstrained`` (any
base of the sequence may be mutated) and ``natural`` (only the 300 SNPs listed in
the sequence's VCF). The 30 sequences are the replicates of a cell.

Timing fields come from ``run_genetic_algorithm`` / ``run_greedy_algorithm``:
``algorithm_seconds`` is the wall time of the optimization itself and is the only
runtime measure used here; ``model_load_seconds`` is kept in the frame but is
cold-start noise (0.25 s to 19 s depending on whether the process was first in its
folder) and is not part of any runtime comparison.

``n_evaluations`` (population size x (generations + 1)) is written by the genetic
algorithm only. The greedy algorithm's ``timings.json`` has no such field, so its
evaluation count is not available and greedy is compared on wall time alone.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# The older benchmark tree: the cells were submitted as HPC array jobs, so they may
# have landed on compute nodes of differing hardware and the GPU cells shared their
# devices (see GPU_CONTENTION_CAVEAT). It holds both devices, one folder per
# (algorithm, device).
MIXED_HARDWARE_ROOT = Path(
    "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/benchmark"
)
MIXED_HARDWARE_DEVICE_FOLDERS: Dict[str, str] = {
    "GA": "CPU",
    "GA_GPU": "GPU",
    "greedy": "CPU",
    "greedy_GPU": "GPU",
}

# The newer benchmark tree: one single job per device ran every config in sequence on
# one node (GPU protocol ``260825_180334_bench_runtime_gpu_same_hardware``), so the
# hardware never changes between the timed runs and the GPU is uncontended. Both
# algorithms sit in the same ``cpu`` / ``gpu`` folder.
SAME_HARDWARE_ROOT = MIXED_HARDWARE_ROOT / "same_hardware"
SAME_HARDWARE_DEVICE_FOLDERS: Dict[str, str] = {"cpu": "CPU", "gpu": "GPU"}

# The GPU array jobs of the mixed-hardware tree requested one GPU per task and
# allowed 15 tasks to run at
# once (``-l gpu=1``, ``-tc 15``), so the GPU cells shared devices: their run
# folders start staggered over ~25 minutes while all CPU cells start in the same
# second, and GPU pop25/unconstrained came out slower than GPU pop50/unconstrained,
# which cannot happen without contention. GPU wall times are therefore upper
# bounds of unknown tightness, not clean single-device measurements.
GPU_CONTENTION_CAVEAT = (
    "GPU cells shared devices (15 concurrent tasks, one GPU each), so GPU wall "
    "times are contended upper bounds, not clean measurements."
)

# Label prefix of a run folder -> the config field that the sweep varies.
SWEPT_PARAMETERS: Dict[str, str] = {
    "pop": "population_size",
    "gen": "number_of_generations",
    "maxmut": "max_number_mutations",
}

ALGORITHMS: Tuple[str, ...] = ("ga", "greedy")
REGIMES: Tuple[str, ...] = ("unconstrained", "natural")

# Below this coefficient of determination, a fitted power law does not describe the
# measured curve and its exponent and confidence interval are not interpretable as
# a scaling law. The CPU/unconstrained population sweep of this benchmark is such a
# case: its runtime is not monotonic in the population size.
POWER_LAW_R_SQUARED_FLOOR = 0.95

RUNTIME_COLUMN = "algorithm_seconds"
MODEL_LOAD_COLUMN = "model_load_seconds"
EVALUATIONS_COLUMN = "n_evaluations"
SECONDS_PER_EVALUATION_COLUMN = "seconds_per_evaluation"

# Fields the run records may carry. The genetic algorithm writes all of them; the
# greedy algorithm writes only the mutation cap, so the rest become NaN.
_CONFIG_FIELDS: Tuple[str, ...] = (
    "population_size",
    "number_of_generations",
    "max_number_mutations",
    EVALUATIONS_COLUMN,
)

# Number of run folders in the benchmark tree, from the sweep definition: 15 GA
# cells (4 population sizes + 3 generation counts, each unconstrained and natural,
# plus the lifted-cap cell) and 10 greedy cells (5 mutation caps, both regimes).
EXPECTED_CELLS_PER_DEVICE: Dict[str, int] = {"ga": 15, "greedy": 10}


def parse_run_directory_name(directory_name: str) -> Tuple[str, str, str]:
    """Split a benchmark run folder name into algorithm, sweep label and regime.

    Folder names are written by ``get_run_folder`` as
    ``<algorithm>_<label>_<regime>_<DDMMYY>_<HHMMSS>_<microseconds>``, e.g.
    ``ga_pop50_natural_260821_213811_209722``.

    Args:
        directory_name: Basename of the run folder.

    Returns:
        Tuple of ``(algorithm, label, regime)``.

    Raises:
        ValueError: If the name has too few parts, or the algorithm or regime
            field is not one of the known values.
    """
    parts = directory_name.split("_")
    if len(parts) != 6:
        raise ValueError(
            f"Run folder {directory_name!r} does not look like "
            "<algorithm>_<label>_<regime>_<date>_<time>_<microseconds>"
        )
    algorithm, regime = parts[0], parts[-4]
    label = "_".join(parts[1:-4])
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Run folder {directory_name!r} has unknown algorithm {algorithm!r}, "
            f"expected one of {ALGORITHMS}"
        )
    if regime not in REGIMES:
        raise ValueError(
            f"Run folder {directory_name!r} has unknown regime {regime!r}, "
            f"expected one of {REGIMES}"
        )
    if not label:
        raise ValueError(f"Run folder {directory_name!r} carries no sweep label")
    return algorithm, label, regime


def split_label(label: str) -> Tuple[str, int]:
    """Split a sweep label into the swept config field and its value.

    Args:
        label: Sweep label of a run folder, e.g. ``"pop50"`` or ``"maxmut2999"``.

    Returns:
        Tuple of ``(swept_parameter, swept_value)``, where ``swept_parameter`` is
        the name of the config field the label refers to.

    Raises:
        ValueError: If the label starts with no known prefix, or its remainder is
            not an integer.
    """
    for prefix, parameter in SWEPT_PARAMETERS.items():
        if label.startswith(prefix):
            remainder = label[len(prefix) :]
            if not remainder.isdigit():
                raise ValueError(
                    f"Sweep label {label!r} has non-integer value {remainder!r}"
                )
            return parameter, int(remainder)
    raise ValueError(
        f"Sweep label {label!r} starts with none of the known prefixes "
        f"{sorted(SWEPT_PARAMETERS)}"
    )


def _validate_run_record(
    record: Dict[str, object],
    directory_name: str,
    algorithm: str,
    swept_parameter: str,
    swept_value: int,
) -> None:
    """Check that a run record agrees with the folder name it was found under.

    The folder name is the only place the sweep cell is labelled, so a mislabelled
    or misplaced folder would otherwise silently pool runs of different settings
    into one cell. Two checks catch that: the swept config field must equal the
    value in the label, and — for the genetic algorithm — ``n_evaluations`` must
    equal ``population_size * (number_of_generations + 1)``, the identity
    ``run_genetic_algorithm`` uses to write it.

    Args:
        record: One entry of the ``runs`` list of a ``timings.json``.
        directory_name: Basename of the run folder, for the error message.
        algorithm: ``"ga"`` or ``"greedy"``.
        swept_parameter: Config field the folder's label refers to.
        swept_value: Value the folder's label claims for that field.

    Raises:
        ValueError: If the record contradicts the folder name, or a field the
            algorithm is expected to write is missing.
    """
    if swept_parameter not in record:
        raise ValueError(
            f"Run {record.get('sequence_name')!r} in {directory_name} has no "
            f"{swept_parameter!r} field, which its label claims to sweep"
        )
    found = record[swept_parameter]
    if found != swept_value:
        raise ValueError(
            f"Run {record.get('sequence_name')!r} in {directory_name} has "
            f"{swept_parameter}={found}, but the folder label says {swept_value}"
        )
    if algorithm != "ga":
        return
    missing = [field for field in _CONFIG_FIELDS if field not in record]
    if missing:
        raise ValueError(
            f"Genetic-algorithm run {record.get('sequence_name')!r} in "
            f"{directory_name} is missing the fields {missing}"
        )
    expected_evaluations = int(record["population_size"]) * (  # type: ignore[arg-type]
        int(record["number_of_generations"]) + 1  # type: ignore[arg-type]
    )
    if int(record[EVALUATIONS_COLUMN]) != expected_evaluations:  # type: ignore[arg-type]
        raise ValueError(
            f"Run {record.get('sequence_name')!r} in {directory_name} reports "
            f"{record[EVALUATIONS_COLUMN]} evaluations, but population_size * "
            f"(number_of_generations + 1) is {expected_evaluations}"
        )


def load_timings_file(
    timings_path: Path, algorithm: str, device: str, label: str, regime: str
) -> pd.DataFrame:
    """Load one ``timings.json`` into a tidy frame of one row per sequence.

    Args:
        timings_path: Path of the ``timings.json`` written by ``summarize_timings``.
        algorithm: ``"ga"`` or ``"greedy"``.
        device: ``"CPU"`` or ``"GPU"``.
        label: Sweep label of the run folder, e.g. ``"pop50"``.
        regime: ``"unconstrained"`` or ``"natural"``.

    Returns:
        Frame with one row per timed sequence, carrying the cell identifiers, the
        config fields, ``realized_mutations`` and the two timing columns.

    Raises:
        ValueError: If the file holds no runs, or a run contradicts the folder name.
    """
    with open(timings_path, "r", encoding="utf-8") as handle:
        content = json.load(handle)
    records = content["runs"]
    if not records:
        raise ValueError(f"{timings_path} holds no timed runs")
    swept_parameter, swept_value = split_label(label)
    directory_name = timings_path.parent.name
    rows: List[Dict[str, object]] = []
    for record in records:
        _validate_run_record(
            record, directory_name, algorithm, swept_parameter, swept_value
        )
        row: Dict[str, object] = {
            "device": device,
            "algorithm": algorithm,
            "regime": regime,
            "label": label,
            "swept_parameter": swept_parameter,
            "swept_value": swept_value,
            "sequence_name": record["sequence_name"],
            "realized_mutations": record["realized_mutations"],
            RUNTIME_COLUMN: record[RUNTIME_COLUMN],
            MODEL_LOAD_COLUMN: record[MODEL_LOAD_COLUMN],
            "run_directory": directory_name,
        }
        for field in _CONFIG_FIELDS:
            row[field] = record.get(field, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def load_benchmark_runtimes(
    benchmark_root: Path,
    device_folders: Dict[str, str],
) -> pd.DataFrame:
    """Load every ``timings.json`` of the benchmark tree into one tidy frame.

    Args:
        benchmark_root: Folder holding one subfolder per device (mixed-hardware
            tree: per algorithm and device), each holding one run folder per sweep
            cell. Pair it with the matching ``device_folders``, e.g.
            :data:`SAME_HARDWARE_ROOT` with
            :data:`SAME_HARDWARE_DEVICE_FOLDERS`.
        device_folders: Mapping of subfolder name to the device its runs were timed
            on. The algorithm is read from each run folder's own name, so a folder
            may hold runs of both algorithms.

    Returns:
        Tidy frame with one row per (device, algorithm, regime, sweep cell,
        sequence), sorted by cell. ``seconds_per_evaluation`` is added for the
        genetic algorithm and is NaN for greedy, which reports no evaluation count.

    Raises:
        FileNotFoundError: If ``benchmark_root`` or one of the expected device
            subfolders does not exist, or a run folder holds no ``timings.json``.
    """
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"Benchmark root {benchmark_root} does not exist")
    frames: List[pd.DataFrame] = []
    for folder_name, device in sorted(device_folders.items()):
        device_folder = benchmark_root / folder_name
        if not device_folder.is_dir():
            raise FileNotFoundError(
                f"Expected benchmark subfolder {device_folder} does not exist"
            )
        for run_folder in sorted(path for path in device_folder.iterdir() if path.is_dir()):
            algorithm, label, regime = parse_run_directory_name(run_folder.name)
            timings_path = run_folder / "timings.json"
            if not timings_path.is_file():
                raise FileNotFoundError(f"No timings.json in {run_folder}")
            frames.append(
                load_timings_file(timings_path, algorithm, device, label, regime)
            )
    if not frames:
        raise FileNotFoundError(f"No benchmark run folders found under {benchmark_root}")
    runtimes = pd.concat(frames, ignore_index=True)
    runtimes[SECONDS_PER_EVALUATION_COLUMN] = (
        runtimes[RUNTIME_COLUMN] / runtimes[EVALUATIONS_COLUMN]
    )
    sort_columns = ["algorithm", "device", "regime", "swept_parameter", "swept_value"]
    return runtimes.sort_values(sort_columns + ["sequence_name"]).reset_index(drop=True)


def check_expected_cells(
    runtimes: pd.DataFrame,
    expected_cells: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Report the number of loaded sweep cells and sequences per algorithm/device.

    The benchmark is only interpretable if every cell of the sweep is present, so
    this returns the counts for inspection rather than asserting them: a cell that
    failed on the cluster shows up as a shortfall the caller can print.

    Args:
        runtimes: Tidy frame from :func:`load_benchmark_runtimes`.
        expected_cells: Expected number of run folders per algorithm. Defaults to
            :data:`EXPECTED_CELLS_PER_DEVICE`.

    Returns:
        Frame with one row per (algorithm, device), columns ``n_cells``,
        ``n_expected_cells``, ``n_sequences_min``, ``n_sequences_max`` and
        ``complete``.
    """
    if expected_cells is None:
        expected_cells = EXPECTED_CELLS_PER_DEVICE
    sequences_per_cell = runtimes.groupby(
        ["algorithm", "device", "run_directory"], observed=True
    ).size()
    rows: List[Dict[str, object]] = []
    for (algorithm, device), counts in sequences_per_cell.groupby(level=[0, 1]):
        expected = expected_cells.get(str(algorithm), np.nan)
        rows.append(
            {
                "algorithm": algorithm,
                "device": device,
                "n_cells": len(counts),
                "n_expected_cells": expected,
                "n_sequences_min": int(counts.min()),
                "n_sequences_max": int(counts.max()),
                "complete": len(counts) == expected,
            }
        )
    return pd.DataFrame(rows)


def summarize_runtimes(
    runtimes: pd.DataFrame,
    group_columns: Sequence[str] = (
        "algorithm",
        "device",
        "regime",
        "swept_parameter",
        "swept_value",
    ),
    value_column: str = RUNTIME_COLUMN,
) -> pd.DataFrame:
    """Summarize a runtime distribution per sweep cell by median and quartiles.

    The 30 sequences of a cell are the replicates. Their runtimes are right-skewed
    (one sequence occasionally saturates the mutation cap late), so the median and
    the interquartile range describe a cell rather than mean and standard deviation.

    Args:
        runtimes: Tidy frame from :func:`load_benchmark_runtimes`.
        group_columns: Columns identifying a cell.
        value_column: Column to summarize, defaults to ``"algorithm_seconds"``.

    Returns:
        Frame with one row per cell and columns ``n``, ``median``, ``q1``, ``q3``,
        ``iqr``, ``minimum`` and ``maximum``.
    """
    grouped = runtimes.groupby(list(group_columns), observed=True)[value_column]
    summary = grouped.agg(
        n="size",
        median="median",
        q1=lambda values: values.quantile(0.25),
        q3=lambda values: values.quantile(0.75),
        minimum="min",
        maximum="max",
    )
    summary["iqr"] = summary["q3"] - summary["q1"]
    return summary.reset_index()


@dataclass
class ScalingFit:
    """Power-law fit ``y = coefficient * x ** exponent`` from an OLS fit in log space.

    Attributes:
        exponent: Slope of the ordinary least-squares fit of log(y) on log(x).
        exponent_ci_low: Lower end of the exponent's confidence interval.
        exponent_ci_high: Upper end of the exponent's confidence interval.
        coefficient: ``exp(intercept)``, the value of y at ``x == 1``.
        r_squared: Coefficient of determination of the fit in log space.
        n_points: Number of (x, y) pairs the fit used.
        confidence: Confidence level of the interval, e.g. 0.95.
    """

    exponent: float
    exponent_ci_low: float
    exponent_ci_high: float
    coefficient: float
    r_squared: float
    n_points: int
    confidence: float


def fit_log_log_scaling(
    x_values: Iterable[float], y_values: Iterable[float], confidence: float = 0.95
) -> ScalingFit:
    """Fit a power law to (x, y) by ordinary least squares on the logarithms.

    A runtime that is linear in the swept parameter has exponent 1; a quadratic one
    has exponent 2. Reporting the exponent with a confidence interval says how
    tightly the data pins the scaling down, which a bare slope does not.

    The interval is the usual OLS interval on the slope: ``exponent +/- t * s``,
    where ``s`` is the standard error of the slope and ``t`` the two-sided critical
    value of Student's t distribution at the given confidence level with ``n - 2``
    degrees of freedom, ``n`` being the number of points.

    That interval assumes the residuals are independent noise around a power law.
    When they are not — when the measured curve is systematically not a power law,
    which shows up as a low ``r_squared`` — the interval is far too narrow and the
    exponent should not be quoted at all. Check ``r_squared`` against
    :data:`POWER_LAW_R_SQUARED_FLOOR` before reading an exponent as a scaling law.

    Args:
        x_values: Positive x values, e.g. population sizes.
        y_values: Positive y values, e.g. runtimes in seconds.
        confidence: Confidence level of the exponent interval, defaults to 0.95.

    Returns:
        The :class:`ScalingFit`.

    Raises:
        ValueError: If fewer than three points are given, the two sequences differ
            in length, any value is not strictly positive, or all x values are equal.
    """
    x_array = np.asarray(list(x_values), dtype=float)
    y_array = np.asarray(list(y_values), dtype=float)
    if x_array.size != y_array.size:
        raise ValueError(
            f"Got {x_array.size} x values and {y_array.size} y values"
        )
    if x_array.size < 3:
        raise ValueError(
            f"A power-law fit with a confidence interval needs at least three "
            f"points, got {x_array.size}"
        )
    if np.any(x_array <= 0) or np.any(y_array <= 0):
        raise ValueError("A log-log fit needs strictly positive x and y values")
    if np.all(x_array == x_array[0]):
        raise ValueError("All x values are equal, the exponent is undefined")
    regression = stats.linregress(np.log(x_array), np.log(y_array))
    critical_value = float(
        stats.t.ppf(0.5 + confidence / 2.0, df=x_array.size - 2)
    )
    half_width = critical_value * float(regression.stderr)  # type: ignore[attr-defined]
    slope = float(regression.slope)  # type: ignore[attr-defined]
    return ScalingFit(
        exponent=slope,
        exponent_ci_low=slope - half_width,
        exponent_ci_high=slope + half_width,
        coefficient=float(np.exp(regression.intercept)),  # type: ignore[attr-defined]
        r_squared=float(regression.rvalue) ** 2,  # type: ignore[attr-defined]
        n_points=int(x_array.size),
        confidence=confidence,
    )


def fit_sweep_scaling(
    runtimes: pd.DataFrame,
    x_column: str = "swept_value",
    group_columns: Sequence[str] = (
        "algorithm",
        "device",
        "regime",
        "swept_parameter",
    ),
    value_column: str = RUNTIME_COLUMN,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Fit one power law per sweep curve, using the individual sequences as points.

    Every sequence of every cell of a curve enters the fit, so the confidence
    interval reflects both the spread within a cell and the fit across cells.
    Curves with fewer than two distinct x values, or fewer than three points, are
    skipped — a single cell pins down no exponent.

    Args:
        runtimes: Tidy frame from :func:`load_benchmark_runtimes`.
        x_column: Column holding the swept value, defaults to ``"swept_value"``.
        group_columns: Columns identifying one sweep curve.
        value_column: Column holding the runtime, defaults to ``"algorithm_seconds"``.
        confidence: Confidence level of the exponent intervals, defaults to 0.95.

    Returns:
        Frame with one row per fitted curve, carrying the group columns, the
        :class:`ScalingFit` fields, and ``n_levels``, the number of distinct x
        values the curve covers.
    """
    rows: List[Dict[str, object]] = []
    for group_values, group in runtimes.groupby(list(group_columns), observed=True):
        usable = group.dropna(subset=[x_column, value_column])
        n_levels = usable[x_column].nunique()
        if n_levels < 2 or len(usable) < 3:
            continue
        fit = fit_log_log_scaling(
            usable[x_column], usable[value_column], confidence=confidence
        )
        row: Dict[str, object] = dict(zip(group_columns, group_values))
        row.update(
            {
                "n_levels": int(n_levels),
                "n_points": fit.n_points,
                "exponent": fit.exponent,
                "exponent_ci_low": fit.exponent_ci_low,
                "exponent_ci_high": fit.exponent_ci_high,
                "coefficient": fit.coefficient,
                "r_squared": fit.r_squared,
                "confidence": fit.confidence,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def flag_poor_power_law_fits(
    fits: pd.DataFrame, r_squared_floor: float = POWER_LAW_R_SQUARED_FLOOR
) -> pd.DataFrame:
    """Return the fitted curves whose power law does not describe the measurement.

    A low coefficient of determination means the residuals are systematic rather
    than noise, so both the exponent and its confidence interval are artefacts of
    forcing a straight line through a curve that is not one. Such rows must not be
    quoted as scaling laws, and this function is how a caller finds them.

    Args:
        fits: Frame from :func:`fit_sweep_scaling`.
        r_squared_floor: Minimum acceptable coefficient of determination, defaults
            to :data:`POWER_LAW_R_SQUARED_FLOOR`.

    Returns:
        The subset of rows below the floor, sorted by ``r_squared`` ascending. An
        empty frame means every fitted curve is well described by a power law.
    """
    if fits.empty:
        return fits
    poor = fits[fits["r_squared"] < r_squared_floor]
    return poor.sort_values("r_squared").reset_index(drop=True)


def find_non_monotonic_curves(
    runtimes: pd.DataFrame,
    group_columns: Sequence[str] = ("algorithm", "device", "regime", "swept_parameter"),
    value_column: str = RUNTIME_COLUMN,
) -> pd.DataFrame:
    """Return the sweep curves whose median runtime does not rise with the parameter.

    Every parameter this benchmark sweeps adds work, so a curve whose median falls
    from one level to the next reports something other than the swept parameter —
    either a real threshold effect (for instance a batch size below which the model
    call is dominated by its own overhead) or interference between concurrent runs.
    Either way it invalidates a power-law reading of that curve, so it is reported
    rather than fitted over.

    Args:
        runtimes: Tidy frame from :func:`load_benchmark_runtimes`.
        group_columns: Columns identifying one sweep curve.
        value_column: Column holding the runtime, defaults to ``"algorithm_seconds"``.

    Returns:
        Frame with one row per offending step, carrying the group columns plus
        ``swept_value_from``, ``swept_value_to``, ``median_from`` and ``median_to``.
    """
    medians = (
        runtimes.groupby(list(group_columns) + ["swept_value"], observed=True)[value_column]
        .median()
        .reset_index()
    )
    rows: List[Dict[str, object]] = []
    for group_values, group in medians.groupby(list(group_columns), observed=True):
        ordered = group.sort_values("swept_value")
        values = ordered[value_column].to_numpy()
        levels = ordered["swept_value"].to_numpy()
        for index in range(len(values) - 1):
            if values[index + 1] >= values[index]:
                continue
            row: Dict[str, object] = dict(zip(group_columns, group_values))
            row.update(
                {
                    "swept_value_from": levels[index],
                    "swept_value_to": levels[index + 1],
                    "median_from": values[index],
                    "median_to": values[index + 1],
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def regime_runtime_ratio(
    runtimes: pd.DataFrame,
    group_columns: Sequence[str] = ("algorithm", "device", "label"),
    value_column: str = RUNTIME_COLUMN,
) -> pd.DataFrame:
    """Compute the natural-over-unconstrained median runtime ratio per sweep cell.

    The two regimes run the same number of model evaluations for the genetic
    algorithm, so a ratio different from 1 isolates the cost of the constraint
    bookkeeping rather than of the scoring. For the greedy algorithm the regimes
    differ in the number of candidate mutations per step as well, so its ratio
    mixes both effects.

    Args:
        runtimes: Tidy frame from :func:`load_benchmark_runtimes`.
        group_columns: Columns identifying a cell across the two regimes.
        value_column: Column holding the runtime, defaults to ``"algorithm_seconds"``.

    Returns:
        Frame with one row per cell that has both regimes, columns
        ``median_unconstrained``, ``median_natural`` and ``ratio_natural_over_unconstrained``.
    """
    medians = (
        runtimes.groupby(list(group_columns) + ["regime"], observed=True)[value_column]
        .median()
        .unstack("regime")
    )
    both_regimes = medians.dropna(subset=list(REGIMES))
    ratios = pd.DataFrame(
        {
            "median_unconstrained": both_regimes["unconstrained"],
            "median_natural": both_regimes["natural"],
            "ratio_natural_over_unconstrained": both_regimes["natural"]
            / both_regimes["unconstrained"],
        }
    )
    return ratios.reset_index()
