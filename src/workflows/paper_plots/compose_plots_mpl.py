"""Render-time matplotlib compositions of publication figures.

Each figure is one function with hardcoded input paths, so the exact data
backing every published panel is recorded in version control and never lost.
Unlike ``compose_plots_svg.py`` (which scales pre-rendered SVGs to fit a page),
these functions build a single matplotlib figure under one shared stylesheet and
call the existing ``src/analysis`` / ``src/workflows`` plot functions with an
``ax`` handed to them, so typography and geometry are uniform by construction.

Run directly::

    conda run -n deepCREshap python src/workflows/paper_plots/compose_plots_mpl.py
"""

import json
import os
from typing import Any, Dict

import matplotlib.pyplot as plt

from analysis.overview.simple_result_stats import (
    draw_visualize_start_vs_max_fitness_by_mutations,
    hist_half_max_mutations,
    show_average_pareto_front,
)
from workflows.paper_plots.style import (
    DOUBLE_COLUMN_MM,
    figure_size_inches,
    panel_label,
    publication_style,
    save_publication_figure,
    sync_axis_limits,
)

OUTPUT_DIR = "src/workflows/paper_plots/figures/mpl_compositions"

# Unused-but-required positional args for the plot functions: when an ``ax`` is
# provided, nothing is saved, so ``name``/``output_format`` are never consumed.
_UNUSED_NAME = "composed"
_UNUSED_FORMAT = "svg"

# Neutral grey for the Pareto markers and histogram bars (more subdued than the
# default matplotlib blue for print).
_NEUTRAL_COLOR = "0.35"

# Scatter marker area (points^2); small enough to keep the dense minimization
# panel readable without losing the sparse maximization panel.
_SCATTER_MARKER_SIZE = 14.0

# The average Pareto fronts saturate well before the 90-mutation expansion limit,
# so the display is cropped to the informative range (saturation stays visible)
# instead of wasting most of the panel on a flat tail.
_PARETO_X_MAX = 45.0


def _load_stats(stats_json_path: str) -> Dict[str, Dict[str, Any]]:
    """Load a precomputed per-gene stats JSON produced by ``get_stats_per_gene``.

    Args:
        stats_json_path: Path to a ``stats_<name>.json`` file.

    Returns:
        The parsed stats dictionary keyed by gene.
    """
    with open(stats_json_path, "r") as stats_file:
        return json.load(stats_file)


def _populate_fig2(fig: plt.Figure) -> None:
    """Draw all six panels of figure 2 onto ``fig``.

    Split out from :func:`fig2` so the figure can be built once and then either
    saved as the publication SVG or previewed as a raster during development.
    Must be called inside a :func:`publication_style` context so the panels
    inherit the shared typography.

    Args:
        fig: An (empty) figure to populate with a 2x3 grid of panels.
    """
    runs = [
        {
            "row_label": "maximization",
            "results_folder": (
                "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/"
                "paper_runs/single_mutation/"
                "arabidopsis_toolkit_msr_max_single_260224_155640_550482"
            ),
            "stats_json": (
                "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/"
                "paper_runs/single_mutation/ara_msr_max_single/"
                "stats_ara_msr_max_single.json"
            ),
        },
        {
            "row_label": "minimization",
            "results_folder": (
                "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/"
                "paper_runs/single_mutation/"
                "arabidopsis_toolkit_msr_min_single_260224_115131_937839"
            ),
            "stats_json": (
                "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/"
                "paper_runs/single_mutation/ara_msr_min_single/"
                "stats_ara_msr_min_single.json"
            ),
        },
    ]

    panel_letters = [["A", "B", "C"], ["D", "E", "F"]]

    # Pre-load stats so the scatter panels can share one colour normalisation.
    stats_per_run = [_load_stats(run["stats_json"]) for run in runs]
    global_max_mutations = max(
        stat["num_mutations_half_max_effect"]
        for stats in stats_per_run
        for stat in stats.values()
        if "num_mutations_half_max_effect" in stat
    )

    # Columns: scatter | shared colorbar | pareto | histogram. The thin colorbar
    # column keeps the three data columns equal width (a per-panel colorbar would
    # otherwise shrink only the scatter cell). Spacing is managed by the figure's
    # constrained layout.
    grid = fig.add_gridspec(nrows=2, ncols=4, width_ratios=[1.0, 0.05, 1.0, 1.0])
    colorbar_ax = fig.add_subplot(grid[:, 1])

    pareto_axes = []
    hist_axes = []
    scatter_mappable = None

    for row_index, (run, stats) in enumerate(zip(runs, stats_per_run)):
        scatter_ax = fig.add_subplot(grid[row_index, 0])
        pareto_ax = fig.add_subplot(grid[row_index, 2])
        hist_ax = fig.add_subplot(grid[row_index, 3])

        scatter_mappable = draw_visualize_start_vs_max_fitness_by_mutations(
            stats,
            _UNUSED_NAME,
            _UNUSED_FORMAT,
            relative=False,
            titles=False,
            ax=scatter_ax,
            add_colorbar=False,
            vmin=0.0,
            vmax=global_max_mutations,
        )
        # Small, edgeless markers keep the dense minimisation panel legible.
        scatter_mappable.set_sizes([_SCATTER_MARKER_SIZE])

        show_average_pareto_front(
            run["results_folder"],
            _UNUSED_FORMAT,
            titles=False,
            ax=pareto_ax,
            color=_NEUTRAL_COLOR,
            error_style="band",
        )
        hist_half_max_mutations(
            stats,
            _UNUSED_NAME,
            _UNUSED_FORMAT,
            titles=False,
            ax=hist_ax,
            color=_NEUTRAL_COLOR,
        )
        # Thin white separators between histogram bars for print crispness.
        for bar in hist_ax.patches:
            bar.set_edgecolor("white")
            bar.set_linewidth(0.4)

        for column_index, ax in enumerate((scatter_ax, pareto_ax, hist_ax)):
            panel_label(ax, panel_letters[row_index][column_index])

        # Row identifier in the left margin (which optimisation objective).
        scatter_ax.text(
            -0.42,
            0.5,
            run["row_label"],
            transform=scatter_ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontweight="bold",
        )

        # Keep the tick numbers on every panel, but show the x-axis label only on
        # the bottom row (the top row's x-axis is identical to the one below it).
        if row_index == 0:
            for ax in (scatter_ax, pareto_ax, hist_ax):
                ax.set_xlabel("")

        pareto_axes.append(pareto_ax)
        hist_axes.append(hist_ax)

    # One shared colorbar for both scatter panels (they use a common norm).
    fig.colorbar(
        scatter_mappable, cax=colorbar_ax, label="Mutations at Half Max Effect"
    )

    # Make the two runs comparable within the Pareto and histogram columns.
    sync_axis_limits(pareto_axes)
    sync_axis_limits(hist_axes)

    # Crop the flat, uninformative tail of the Pareto fronts (fills the panels).
    for ax in pareto_axes:
        ax.set_xlim(right=_PARETO_X_MAX)


def fig2() -> None:
    """Compose figure 2: two single-mutation runs (max top, min bottom).

    Data columns are scatter (start vs. final fitness coloured by mutations at
    half max, magma), average Pareto front, and histogram of mutations at half
    max, separated by a thin column holding one colorbar shared by both scatter
    panels. The Pareto and histogram columns each share x/y limits across the two
    runs so the runs are directly comparable; scatter y is left independent
    (maximisation saturates near 1, minimisation near 0 by objective).
    """
    with publication_style():
        fig = plt.figure(
            figsize=figure_size_inches(DOUBLE_COLUMN_MM, 100.0), layout="constrained"
        )
        _populate_fig2(fig)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_publication_figure(fig, os.path.join(OUTPUT_DIR, "fig2_composed.svg"))
        plt.close(fig)


if __name__ == "__main__":
    fig2()
