"""Render-time matplotlib compositions of supplementary publication figures.

Same construction principle as ``compose_plots_mpl.py``: one function per figure,
hardcoded input paths, a single matplotlib figure built inside the shared
:func:`publication_style` context with the existing plot functions handed an
``ax``. Supplementary figures are typically single-panel, but they use the same
stylesheet so typography matches the main figures exactly.

Run directly::

    conda run -n deepCREshap python src/workflows/paper_plots/supplementary_plots_mpl.py
"""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

from workflows.adversarial.plot import (
    draw_example_panel,
    draw_pooled_panel,
    prepare_predictions,
    select_example_genes,
)
from workflows.adversarial.reevaluate import (
    CSV_FILE_NAME as ADVERSARIAL_CSV_FILE_NAME,
    clean_gene_id,
)
from workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark_calc import (
    MIXED_HARDWARE_DEVICE_FOLDERS,
    MIXED_HARDWARE_ROOT,
    RUNTIME_COLUMN,
    ScalingFit,
    fit_log_log_scaling,
    load_benchmark_runtimes,
)
from workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark_plot import (
    REGIME_ORDER,
    SWEPT_PARAMETER_LABELS,
    plot_scaling_fit,
)
from workflows.evo_alg_pooled_plots.tf_comparison.mut5_comparison import (
    LEFT_COLUMNS as MUT5_LEFT_COLUMNS,
    MUTATED_COLUMN as MUT5_MUTATED_COLUMN,
    PAIRED_RUN_A as MUT5_PAIRED_RUN_A,
    PAIRED_RUN_B as MUT5_PAIRED_RUN_B,
    RIGHT_COLUMNS as MUT5_RIGHT_COLUMNS,
    RUNS as MUT5_RUNS,
    SINGLE_RUNS as MUT5_SINGLE_RUNS,
)
from workflows.evo_alg_pooled_plots.tf_comparison.tf_comparison_calc import (
    TF_COLUMN,
    build_matrix,
    order_tfs_by_group_contrast,
    paired_tf_significance,
    single_run_tf_significance,
)
from workflows.evo_alg_pooled_plots.tf_comparison.tf_comparison_plot import (
    plot_heatmap,
    q_to_stars,
)
from workflows.paper_plots.style import (
    DOUBLE_COLUMN_MM,
    figure_size_inches,
    panel_label,
    publication_style,
    save_publication_figure,
)

OUTPUT_DIR = "src/workflows/paper_plots/figures/supplements"

# Significance cutoff for keeping a TF family in the figure: a TF is shown when
# its paired ara max-vs-min contrast reaches at least the * tier. Figure 3G uses
# the stricter *** tier; the supplement shows the full significant set.
_SIGNIFICANT_ALPHA = 0.05

# Subtle grey for the group dividers (matches figure 3G).
_SEPARATOR_COLOR = "0.45"
_SEPARATOR_WIDTH = 1.0

# Asterisk glyphs sit high in their text box, so seaborn's va="center" leaves
# them above the cell centre; this nudge (in cell units) re-centres them.
_STAR_VERTICAL_NUDGE = 0.12

# The deepCIS scan suffix carried by every TF-family name from the tnt motif set;
# dropping it keeps the tick labels short without merging distinct families.
_TNT_SUFFIX = "_tnt"


def _short_tf_name(tf_name: str) -> str:
    """Strip the ``_tnt`` motif-set suffix from a TF family name.

    Names from other motif sets (e.g. ``Homeobox_ecoli``) are left untouched, so
    families that differ only by motif set stay distinguishable.

    Args:
        tf_name: TF family name as it appears in the comparison matrix.

    Returns:
        The name without a trailing ``_tnt``, otherwise unchanged.
    """
    if tf_name.endswith(_TNT_SUFFIX):
        return tf_name[: -len(_TNT_SUFFIX)]
    return tf_name


def _mut5_cell_stars(significance: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Build the intra-run significance stars for the four 5-mutation runs.

    Mirrors ``mut5_comparison.main``: the ara pair reuses the ``q_a`` / ``q_b``
    columns of the paired analysis, while GOF and LOF are each tested
    independently via :func:`single_run_tf_significance`.

    Args:
        significance: The paired ara max-vs-min significance frame, carrying the
            per-run ``q_a`` / ``q_b`` columns.

    Returns:
        Mapping of run label to ``{tf: stars}`` for the cell annotations.
    """
    _, paired_label_a = MUT5_PAIRED_RUN_A
    _, paired_label_b = MUT5_PAIRED_RUN_B
    cell_stars: Dict[str, Dict[str, str]] = {
        paired_label_a: dict(
            zip(significance[TF_COLUMN], significance["q_a"].map(q_to_stars))
        ),
        paired_label_b: dict(
            zip(significance[TF_COLUMN], significance["q_b"].map(q_to_stars))
        ),
    }
    for run_dir, label in MUT5_SINGLE_RUNS:
        intra = single_run_tf_significance(run_dir, mutated_column=MUT5_MUTATED_COLUMN)
        cell_stars[label] = dict(
            zip(intra[TF_COLUMN], intra["q_intra"].map(q_to_stars))
        )
    return cell_stars


def _significant_mut5_matrix(
    significance: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Build the per-gene diff matrix restricted to the significant TF families.

    The four-run matrix is ordered by group contrast exactly as in
    ``mut5_comparison.main`` (max-favoured families first), then filtered to the
    TFs whose paired ara max-vs-min contrast reaches :data:`_SIGNIFICANT_ALPHA`.

    Args:
        significance: The paired ara max-vs-min significance frame.

    Returns:
        ``(matrix, row_stars)``: the filtered TF x run matrix in display order,
        and the inter-run contrast stars keyed by TF name.
    """
    significant_tfs = set(
        significance.loc[significance["q_contrast"] < _SIGNIFICANT_ALPHA, TF_COLUMN]
    )
    matrix = order_tfs_by_group_contrast(
        build_matrix(
            MUT5_RUNS,
            normalization="per_gene",
            mutated_column=MUT5_MUTATED_COLUMN,
        ),
        MUT5_LEFT_COLUMNS,
        MUT5_RIGHT_COLUMNS,
    )
    matrix = matrix.loc[[tf for tf in matrix.index if tf in significant_tfs]]
    row_stars = dict(
        zip(significance[TF_COLUMN], significance["q_contrast"].map(q_to_stars))
    )
    return matrix, row_stars


def _draw_mut5_tf_heatmap(ax: plt.Axes, colorbar_ax: plt.Axes) -> None:
    """Draw the 5-mutation TF heatmap: significant families across four runs.

    Wide (transposed) layout — TF families along the x axis, the four runs as
    rows — annotated with intra-run significance stars only (no numbers). The
    x tick labels carry the inter-run contrast stars. A subtle grey divider
    separates the maximization rows (ara max, GOF) from the minimization rows
    (ara min, LOF), and a second one splits the max-favoured from the
    min-favoured TF block where the ordering score changes sign.

    Args:
        ax: Axes to draw the heatmap onto.
        colorbar_ax: Thin axes to hold the colour bar.
    """
    paired_dir_a, _ = MUT5_PAIRED_RUN_A
    paired_dir_b, _ = MUT5_PAIRED_RUN_B
    significance = paired_tf_significance(
        paired_dir_a, paired_dir_b, mutated_column=MUT5_MUTATED_COLUMN
    )
    matrix, row_stars = _significant_mut5_matrix(significance)

    plot_heatmap(
        matrix,
        annotate=True,
        cell_stars=_mut5_cell_stars(significance),
        row_stars=row_stars,
        separator_after_column=None,
        annotate_values=False,
        add_colorbar=False,
        transpose=True,
        ax=ax,
    )
    ax.figure.colorbar(
        ax.collections[0], cax=colorbar_ax, label="Δ binding peaks / gene"
    )

    n_tfs = matrix.shape[0]
    # Shorten the TF names but keep the contrast stars produced by plot_heatmap.
    ax.set_xticks(np.arange(n_tfs) + 0.5)
    ax.set_xticklabels(
        [
            f"{_short_tf_name(tf)} {row_stars[tf]}".strip()
            if row_stars.get(tf)
            else _short_tf_name(tf)
            for tf in matrix.index
        ],
        rotation=90,
        ha="center",
    )

    for annotation in ax.texts:
        text_x, text_y = annotation.get_position()
        annotation.set_position((text_x, text_y + _STAR_VERTICAL_NUDGE))

    # Horizontal divider between the maximization and minimization run groups.
    ax.axhline(
        len(MUT5_LEFT_COLUMNS), color=_SEPARATOR_COLOR, linewidth=_SEPARATOR_WIDTH
    )
    # Vertical divider where the ordering score (max-group minus min-group mean)
    # changes sign: max-favoured TFs left, min-favoured right.
    left_mean = matrix[MUT5_LEFT_COLUMNS].mean(axis=1).fillna(0.0)
    right_mean = matrix[MUT5_RIGHT_COLUMNS].mean(axis=1).fillna(0.0)
    n_more_in_max = int(((left_mean - right_mean) > 0).sum())
    if 0 < n_more_in_max < n_tfs:
        ax.axvline(
            n_more_in_max, color=_SEPARATOR_COLOR, linewidth=_SEPARATOR_WIDTH
        )


def _populate_fig_s1(fig: Figure) -> None:
    """Draw supplementary figure S1 (single panel) onto ``fig``.

    Must be called inside a :func:`publication_style` context.

    Args:
        fig: An (empty) figure to populate.
    """
    grid = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 0.02], wspace=0.04)
    heatmap_ax = fig.add_subplot(grid[0, 0])
    colorbar_ax = fig.add_subplot(grid[0, 1])
    _draw_mut5_tf_heatmap(heatmap_ax, colorbar_ax)


def fig_s1() -> None:
    """Compose supplementary figure S1: TF binding change at a 5-mutation budget.

    The four-run per-gene TF diff heatmap from the dedicated 5-mutation deepCIS
    rerun, restricted to the TF families whose paired ara max-vs-min contrast is
    significant (q < 0.05). Cells are annotated with intra-run significance stars
    only; the colour encodes the change in binding peaks per gene.
    """
    with publication_style():
        fig = plt.figure(
            figsize=figure_size_inches(DOUBLE_COLUMN_MM, 75.0), layout="constrained"
        )
        _populate_fig_s1(fig)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_publication_figure(
            fig, os.path.join(OUTPUT_DIR, "figS1_tf_comparison_mut5.svg")
        )
        plt.close(fig)


# Supplementary figure S2: runtime scaling of the two optimizers on CPU. GPU cells
# of the benchmark shared devices, so their wall times are contended upper bounds
# and are left out of the figure entirely.
_BENCHMARK_DEVICE = "CPU"

# Panels of figure S2: (algorithm, swept config field, panel title). The genetic
# algorithm's two cost drivers are the population size and the generation count;
# the greedy algorithm's only one is the mutation cap, which is its step count.
_RUNTIME_PANELS: List[Tuple[str, str, str]] = [
    ("ga", "population_size", "Evolutionary algorithm"),
    ("ga", "number_of_generations", "Evolutionary algorithm"),
    ("greedy", "max_number_mutations", "Greedy algorithm"),
]

# Regime colours taken from the magma colour map, at a dark and a light sample far
# enough apart to stay distinguishable in greyscale as well as in colour.
_REGIME_COLORS: Dict[str, str] = {
    "unconstrained": "#641a80",
    "natural": "#f9795d",
}

# Placement of the R² label at the right end of a fit line: an offset in points, and
# the factor the x axis is widened by to make room for the labels.
_FIT_LABEL_OFFSET_POINTS = (4.0, 2.0)
_FIT_LABEL_FONTSIZE = 6.0
_X_AXIS_HEADROOM = 1.9

# Panel labels sit far enough left to clear the y axis of a log-log panel.
_PANEL_LABEL_X = -0.22


def _cpu_benchmark_runtimes() -> pd.DataFrame:
    """Load the benchmark timings and keep the CPU rows.

    Read from the mixed-hardware tree: it is the only one with CPU cells.

    Returns:
        Tidy runtime frame restricted to ``device == "CPU"``.
    """
    runtimes = load_benchmark_runtimes(
        MIXED_HARDWARE_ROOT, MIXED_HARDWARE_DEVICE_FOLDERS
    )
    return pd.DataFrame(runtimes[runtimes["device"] == _BENCHMARK_DEVICE])


def _regime_scaling_fits(panel_runtimes: pd.DataFrame) -> Dict[str, ScalingFit]:
    """Fit a power law to the runtime curve of each mutation regime.

    The fit is of the form ``t = c * x ** k``, with ``t`` the optimization wall time
    in seconds, ``x`` the swept parameter, ``k`` the exponent and ``c`` the
    coefficient; ``k = 1`` is linear scaling. Every regime gets its line, and the R²
    written next to that line says how well the power law describes the curve: below
    the benchmark's R² floor of 0.95 it does not describe it at all, and the exponent
    of such a fit must not be quoted as a scaling law.

    Args:
        panel_runtimes: Rows of one panel — one algorithm, one device, one swept
            parameter, both regimes.

    Returns:
        The fits, keyed by regime.
    """
    fits: Dict[str, ScalingFit] = {}
    present = set(panel_runtimes["regime"].unique())
    for regime in [name for name in REGIME_ORDER if name in present]:
        group = panel_runtimes[panel_runtimes["regime"] == regime]
        fits[regime] = fit_log_log_scaling(
            group["swept_value"], group[RUNTIME_COLUMN]
        )
    return fits


def _label_fit_lines(
    ax: plt.Axes, panel_runtimes: pd.DataFrame, fits: Dict[str, ScalingFit]
) -> None:
    """Write each fit's R² at the right end of its line, in the line's colour.

    Putting the goodness of fit on the line itself rather than into a legend keeps
    it next to the curve it belongs to and leaves the upper part of the panel, where
    the steepest curves run, free.

    Args:
        ax: Axes the fits were drawn onto. Modified in place.
        panel_runtimes: Rows of the panel, used for each regime's x range.
        fits: Fits per regime, as returned by :func:`_regime_scaling_fits`.
    """
    for regime, fit in fits.items():
        group = panel_runtimes[panel_runtimes["regime"] == regime]
        x_end = float(group["swept_value"].max())
        ax.annotate(
            f"R² {fit.r_squared:.2f}",
            xy=(x_end, fit.coefficient * x_end**fit.exponent),
            xytext=_FIT_LABEL_OFFSET_POINTS,
            textcoords="offset points",
            color=_REGIME_COLORS[regime],
            fontsize=_FIT_LABEL_FONTSIZE,
            va="center",
            ha="left",
        )
    # The labels sit outside the data range, so the axis is widened to hold them.
    left, right = ax.get_xlim()
    ax.set_xlim(left, right * _X_AXIS_HEADROOM)


def _set_sweep_ticks(ax: plt.Axes, swept_values: np.ndarray) -> None:
    """Put one tick per swept level on the logarithmic x axis, labelled as integers.

    Matplotlib's log locator places decade and minor ticks, which on the narrow
    ranges these sweeps cover overlaps its own labels. The swept levels are the
    only x values measured, so they are the only ones worth a tick.

    Args:
        ax: Axes with a logarithmic x axis. Modified in place.
        swept_values: The distinct swept values of the panel.
    """
    levels = np.sort(np.unique(swept_values.astype(float)))
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{int(level)}" for level in levels])
    ax.xaxis.set_minor_locator(NullLocator())


def _draw_runtime_scaling_panel(
    ax: plt.Axes,
    runtimes: pd.DataFrame,
    algorithm: str,
    swept_parameter: str,
    title: str,
) -> None:
    """Draw one CPU runtime-scaling panel of figure S2.

    Each of the 30 benchmark sequences of every sweep level is a point, coloured by
    mutation regime, on log-log axes; the fitted power law of a regime is drawn as
    a line whose goodness of fit is written next to its right end
    panel. The regime legend is shared by the whole figure, so no panel draws one.

    Args:
        ax: Axes to draw onto.
        runtimes: CPU runtime frame from :func:`_cpu_benchmark_runtimes`.
        algorithm: Algorithm to select, ``"ga"`` or ``"greedy"``.
        swept_parameter: Config field on the x axis.
        title: Panel title.

    Raises:
        ValueError: If the selection is empty, so a silently missing sweep cannot
            be mistaken for an empty panel.
    """
    panel_runtimes = pd.DataFrame(
        runtimes[
            (runtimes["algorithm"] == algorithm)
            & (runtimes["swept_parameter"] == swept_parameter)
        ]
    )
    if panel_runtimes.empty:
        raise ValueError(
            f"No CPU benchmark rows for algorithm {algorithm!r} and swept "
            f"parameter {swept_parameter!r}"
        )
    fits = _regime_scaling_fits(panel_runtimes)
    plot_scaling_fit(
        panel_runtimes,
        fits=fits,
        x_column="swept_value",
        hue_column="regime",
        palette=_REGIME_COLORS,
        show_legend=False,
        show_title=True,
        title=title,
        xlabel=SWEPT_PARAMETER_LABELS.get(swept_parameter, swept_parameter),
        ax=ax,
    )
    _set_sweep_ticks(ax, panel_runtimes["swept_value"].to_numpy())
    _label_fit_lines(ax, panel_runtimes, fits)


def _populate_fig_s2(fig: Figure) -> None:
    """Draw supplementary figure S2 (three panels in one row) onto ``fig``.

    Must be called inside a :func:`publication_style` context.

    Args:
        fig: An (empty) figure to populate.
    """
    runtimes = _cpu_benchmark_runtimes()
    grid = fig.add_gridspec(nrows=1, ncols=len(_RUNTIME_PANELS))
    for index, (algorithm, swept_parameter, title) in enumerate(_RUNTIME_PANELS):
        ax = fig.add_subplot(grid[0, index])
        _draw_runtime_scaling_panel(ax, runtimes, algorithm, swept_parameter, title)
        if index > 0:
            # The three panels share one quantity on the y axis, so labelling the
            # leftmost one is enough and keeps the panels wide.
            ax.set_ylabel("")
        panel_label(ax, "ABC"[index], x=_PANEL_LABEL_X)
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="-",
                color=_REGIME_COLORS[regime],
                label=regime,
                markersize=3,
            )
            for regime in REGIME_ORDER
        ],
        title="Mutation regime",
        loc="outside lower center",
        ncol=len(REGIME_ORDER),
        frameon=False,
    )


def fig_s2() -> None:
    """Compose supplementary figure S2: CPU runtime scaling of both optimizers.

    Panel A and B give the two cost drivers of the genetic algorithm, the
    population size and the number of generations; panel C gives the greedy
    algorithm's mutation cap, which is its number of optimization steps. Only CPU
    measurements are shown: the GPU cells of the benchmark shared devices between
    15 concurrent tasks, so their wall times are contended upper bounds.
    """
    with publication_style():
        fig = plt.figure(
            figsize=figure_size_inches(DOUBLE_COLUMN_MM, 65.0), layout="constrained"
        )
        _populate_fig_s2(fig)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_publication_figure(
            fig, os.path.join(OUTPUT_DIR, "figS2_runtime_scaling.svg")
        )
        plt.close(fig)


# Supplementary figure S3: the adversarial re-evaluation of both GOF/LOF runs.
# The re-evaluation CSVs written by ``workflows.adversarial.reevaluate``; each
# folder name encodes the evolution run and the model set it was re-scored with.
_ADVERSARIAL_OUTPUT_ROOT = "src/workflows/adversarial/outputs"
_ADVERSARIAL_RUNS: List[Tuple[str, str]] = [
    (
        "GOF_single_mutation_251009_121226_109368__MSR_models_M0X0.75",
        "Gain of function",
    ),
    (
        "LOF_single_mutation_251020_180028_564570__MSR_models_M0X0.75",
        "Loss of function",
    ),
]

# The example genes are picked at evenly spaced ranks of the mean normalized gap,
# so with three of them they are the closest, the median and the farthest agreeing
# gene; the ranking itself is described in the caption rather than in the titles.
_NUMBER_OF_EXAMPLE_GENES = 3

# Column titles, written on the top row only: both rows hold the same four panel
# kinds, so the pooled panel and the three gap ranks are named once for both. The
# gene shown in an example panel differs per row and is given in the caption.
_ADVERSARIAL_COLUMN_TITLES: List[str] = [
    "pooled",
    "minimum difference",
    "median difference",
    "maximum difference",
]

# Colours of the two figure elements, taken from the magma samples the other
# supplementary figures use, so the whole supplement reads as one set: the
# optimization model's front in the dark purple, the sibling models in the light
# salmon.
_ADVERSARIAL_REFERENCE_COLOR = "#641a80"
_ADVERSARIAL_OTHERS_COLOR = "#f9795d"

# Panel labels sit left of the y axis; the row identifier sits further out still.
_ADVERSARIAL_PANEL_LABEL_X = -0.30
_ROW_LABEL_X = -0.46


def _load_adversarial_run(run_folder: str) -> pd.DataFrame:
    """Load and prepare one adversarial re-evaluation table.

    Args:
        run_folder: Folder name under :data:`_ADVERSARIAL_OUTPUT_ROOT`.

    Returns:
        The expanded and per gene normalized table, as returned by
        :func:`prepare_predictions`.
    """
    predictions = pd.read_csv(
        os.path.join(_ADVERSARIAL_OUTPUT_ROOT, run_folder, ADVERSARIAL_CSV_FILE_NAME)
    )
    predictions["gene_id"] = predictions["gene_id"].map(clean_gene_id)
    return prepare_predictions(predictions)


def _draw_adversarial_row(
    axes_row: List[plt.Axes],
    prepared: pd.DataFrame,
    is_top_row: bool,
    is_bottom_row: bool,
) -> List[Artist]:
    """Draw one run's four panels: the pooled comparison and three example genes.

    The leftmost panel pools all genes after the per gene normalization; the three
    others show the raw, unnormalized predictions of the closest, the median and
    the farthest agreeing gene, so the pooled band can be read against the
    individual curves it summarizes.

    Args:
        axes_row: The four axes of the row, left to right.
        prepared: Table as returned by :func:`prepare_predictions`.
        is_top_row: Whether this row carries the shared column titles. Both rows
            show the same four panel kinds, so naming them once is enough.
        is_bottom_row: Whether this row carries the shared x axis label.

    Returns:
        The legend handles of the panels, shared by the whole figure.
    """
    legend_handles = draw_pooled_panel(
        axes_row[0],
        prepared,
        show_min_max=False,
        reference_color=_ADVERSARIAL_REFERENCE_COLOR,
        others_color=_ADVERSARIAL_OTHERS_COLOR,
    )
    example_genes = select_example_genes(prepared, _NUMBER_OF_EXAMPLE_GENES)
    for ax, gene_id in zip(axes_row[1:], example_genes):
        # The single-gene panels draw every element, including the min-max band the
        # pooled panel leaves out, so their handles carry the figure's legend.
        legend_handles = draw_example_panel(
            ax,
            prepared,
            gene_id,
            reference_color=_ADVERSARIAL_REFERENCE_COLOR,
            others_color=_ADVERSARIAL_OTHERS_COLOR,
        )
    axes_row[1].set_ylabel("prediction")

    for ax, column_title in zip(axes_row, _ADVERSARIAL_COLUMN_TITLES):
        if is_top_row:
            ax.set_title(column_title)
        if not is_bottom_row:
            # One x axis label per column is enough and keeps the rows compact.
            ax.set_xlabel("")
    return legend_handles


def _populate_fig_s3(fig: Figure) -> None:
    """Draw supplementary figure S3 (two runs by four panels) onto ``fig``.

    Must be called inside a :func:`publication_style` context.

    Args:
        fig: An (empty) figure to populate.
    """
    grid = fig.add_gridspec(nrows=len(_ADVERSARIAL_RUNS), ncols=4)
    legend_handles: List[Artist] = []
    for row, (run_folder, row_label) in enumerate(_ADVERSARIAL_RUNS):
        axes_row = [fig.add_subplot(grid[row, column]) for column in range(4)]
        legend_handles = _draw_adversarial_row(
            axes_row,
            _load_adversarial_run(run_folder),
            is_top_row=row == 0,
            is_bottom_row=row == len(_ADVERSARIAL_RUNS) - 1,
        )
        for column, ax in enumerate(axes_row):
            panel_label(
                ax,
                "ABCDEFGH"[row * 4 + column],
                x=_ADVERSARIAL_PANEL_LABEL_X,
            )
        axes_row[0].text(
            _ROW_LABEL_X,
            0.5,
            row_label,
            transform=axes_row[0].transAxes,
            rotation=90,
            fontweight="bold",
            va="center",
            ha="center",
        )
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=len(legend_handles),
        frameon=False,
    )


def fig_s3() -> None:
    """Compose supplementary figure S3: sibling MSR models re-score the fronts.

    Every Pareto-front sequence of the gain-of-function and the loss-of-function
    run is re-scored with the 11 MSR models of the same training run that were not
    used for the optimization. Panels A and E pool all genes after normalizing each
    gene to the range its optimization model achieved; the remaining panels show
    the raw predictions of the closest, the median and the farthest agreeing gene,
    ranked by the mean normalized gap between the other models' median curve and
    the optimization model's front.
    """
    with publication_style():
        fig = plt.figure(
            figsize=figure_size_inches(DOUBLE_COLUMN_MM, 105.0), layout="constrained"
        )
        _populate_fig_s3(fig)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_publication_figure(
            fig, os.path.join(OUTPUT_DIR, "figS3_adversarial_reevaluation.svg")
        )
        plt.close(fig)


SUPPLEMENTARY_FIGURES: List = [fig_s1, fig_s2, fig_s3]


def main() -> None:
    """Render every supplementary figure into :data:`OUTPUT_DIR`."""
    for figure_function in SUPPLEMENTARY_FIGURES:
        figure_function()
        print(f"Rendered {figure_function.__name__}")


if __name__ == "__main__":
    main()
