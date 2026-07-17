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
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from analysis.mutations.analyze_mutations import (
    COLORS,
    calculate_net_nucleotide_change,
    calculate_positional_nucleotide_change,
    make_line_plot_rolling_window,
    plot_net_nucleotide_change,
)
from analysis.overview.simple_result_stats import (
    draw_visualize_start_vs_max_fitness_by_mutations,
    hist_half_max_mutations,
    show_average_pareto_front,
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
from workflows.mutation_distance_analysis.mutation_distance_analysis import (
    _to_proportions,
    compute_random_distances,
    compute_real_distances,
    plot_difference,
)
from workflows.mutation_distribution_analysis.mutation_pool import MutationPool
from workflows.overlap_analysis._common import (
    BINDING_STATUS_COLORS,
    HIGHLIGHT_WINDOW_CENTER,
    POSITION_SERIES_COLORS,
    compute_correlation_by_position,
    plot_overlay_highlight_correlation,
    simply_plot_multi,
)
from workflows.overlap_analysis.starrseq_deepcre_correlation_bHLH import (
    prepare_bhlh_enrichment_df,
)
from workflows.overlap_analysis.starrseq_deepcre_correlation_WRKY import (
    prepare_wrky_enrichment_df,
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

# Bar transparency. Matches ``hist_half_max_mutations`` (alpha=0.7 in
# simple_result_stats.py) so fig3's bars blend to the same lighter grey as fig2's
# histograms (0.35 over white at alpha 0.7 ≈ 0.55).
_BAR_ALPHA = 0.7

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


# --- Figure 3: mutation signatures (net change, distances) + TF contrast -----

# The two maximization runs compared in the left 2x2 grid (net nucleotide change
# and mutation-distance difference), top row then bottom row.
_FIG3_RUNS = [
    {
        "species_label": "A. thaliana",
        "mutated_sequences_json": (
            "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/"
            "paper_runs/single_mutation/ara_msr_max_single/"
            "all_mutated_sequences_ara_msr_max_single_gen1999.json"
        ),
    },
    {
        "species_label": "Z. mays",
        "mutated_sequences_json": (
            "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/"
            "paper_runs/single_mutation/zea_msr_max_single/"
            "all_mutated_sequences_zea_msr_max_single_gen1999.json"
        ),
    },
]

# Panel E — the four-run TF-comparison directories (display order: the two
# maximization runs then the two minimization runs). These mirror
# ``minmax_comparison.py`` so panel E matches that standalone figure's data,
# ordering and cell-significance stars.
_ARA_MAX_DIR = (
    "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/"
    "paper_runs/single_mutation/ara_msr_max_single"
)
_GOF_DIR = (
    "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/"
    "GOF_LOF/GOF/GOF_single"
)
_ARA_MIN_DIR = (
    "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/"
    "paper_runs/single_mutation/ara_msr_min_single"
)
_LOF_DIR = (
    "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/"
    "GOF_LOF/LOF/LOF_single"
)
_FIG3_TF_RUNS: List[Tuple[str, str]] = [
    (_ARA_MAX_DIR, "ara max"),
    (_GOF_DIR, "GOF"),
    (_ARA_MIN_DIR, "ara min"),
    (_LOF_DIR, "LOF"),
]
# Left group = maximization runs, right group = minimization runs. Ordering
# (order_tfs_by_group_contrast) and the vertical divider both use this split.
_FIG3_TF_LEFT_COLUMNS = ["ara max", "GOF"]
_FIG3_TF_RIGHT_COLUMNS = ["ara min", "LOF"]
# The single runs (no paired partner) that get an independent intra-run test for
# their cell stars; the ara pair reuses the q_a / q_b from the paired analysis.
_FIG3_TF_SINGLE_RUNS: List[Tuple[str, str]] = [(_GOF_DIR, "GOF"), (_LOF_DIR, "LOF")]

# Crop the mutation-distance difference panels to short distances: the signal is
# concentrated near zero and the long tail is flat and uninformative.
_DISTANCE_MAX = 30
# Random-baseline replicates per gene and RNG seed for the distance null.
_DISTANCE_REPLICATES_PER_GENE = 10
_DISTANCE_SEED = 42
# Keep only TFs whose paired ara max-vs-min contrast reaches the *** tier; the
# cutoff matches ``q_to_stars`` (q < 0.001). Every selected TF then carries three
# stars, so the redundant row-label stars are dropped.
_THREE_STAR_ALPHA = 0.001
# Panel E group/run dividers: soft grey, thin (the default black/2.0 reads harsh).
_SEPARATOR_COLOR = "0.45"
_SEPARATOR_WIDTH = 1.0
# Downward shift (in cell-height units) to centre the high-sitting asterisks.
_STAR_VERTICAL_NUDGE = 0.12


def _distance_difference_proportions(
    mutated_sequences_json: str,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Compute real and random inter-mutation distance proportions for one run.

    Builds a :class:`MutationPool` in memory from the run's summarized-mutations
    JSON (no cached pool file is needed), then returns the normalized real and
    random-baseline distance distributions ready for :func:`plot_difference`.

    Args:
        mutated_sequences_json: Path to an ``all_mutated_sequences_*.json`` file.

    Returns:
        ``(real_proportions, random_proportions)`` as ``{distance: proportion}``
        mappings.
    """
    pool = MutationPool.from_summarized_json(mutated_sequences_json)
    real_distances = compute_real_distances(pool)
    rng = np.random.default_rng(_DISTANCE_SEED)
    random_distances = compute_random_distances(
        pool, _DISTANCE_REPLICATES_PER_GENE, rng
    )
    return _to_proportions(real_distances), _to_proportions(random_distances)


def _significant_tf_cell_stars(
    significance: "Any",
) -> Dict[str, Dict[str, str]]:
    """Build the intra-run cell-significance stars for panel E's four runs.

    Mirrors ``minmax_comparison.py``: the ara pair reuses the ``q_a`` / ``q_b``
    columns already produced by the paired analysis, while the single runs
    (GOF, LOF) are each tested independently via
    :func:`single_run_tf_significance`.

    Args:
        significance: The paired ara max-vs-min significance frame (with the
            default ``q_a`` / ``q_b`` per-run columns).

    Returns:
        Mapping of run label to ``{tf: stars}`` for the numeric cell annotations.
    """
    cell_stars: Dict[str, Dict[str, str]] = {
        "ara max": dict(zip(significance[TF_COLUMN], significance["q_a"].map(q_to_stars))),
        "ara min": dict(zip(significance[TF_COLUMN], significance["q_b"].map(q_to_stars))),
    }
    for run_dir, label in _FIG3_TF_SINGLE_RUNS:
        intra = single_run_tf_significance(run_dir)
        cell_stars[label] = dict(zip(intra[TF_COLUMN], intra["q_intra"].map(q_to_stars)))
    return cell_stars


def _draw_significant_tf_heatmap(ax: plt.Axes, colorbar_ax: plt.Axes) -> None:
    """Draw panel E: the three-star ara max-vs-min TFs across all four runs.

    Runs the paired ara max/min contrast and keeps only the TFs reaching the
    *** significance tier (:data:`_THREE_STAR_ALPHA`). The four-run per-gene diff
    matrix is ordered exactly as in ``minmax_comparison.py``
    (:func:`order_tfs_by_group_contrast`), which places the max-favoured TFs
    above the min-favoured ones. Subtle grey dividers separate the maximization
    from the minimization runs (vertical) and the two TF blocks (horizontal,
    where the ordering score changes sign). Intra-run significance stars annotate
    the cells; row labels carry no stars — every selected TF is *** by
    construction. The heatmap's colour bar is drawn into ``colorbar_ax`` so it
    can sit in a dedicated thin column next to the (narrow) heatmap.

    Args:
        ax: Axes to draw the heatmap onto.
        colorbar_ax: Thin axes to hold the shared colour bar.
    """
    significance = paired_tf_significance(_ARA_MAX_DIR, _ARA_MIN_DIR)
    three_star_tfs = set(
        significance.loc[
            significance["q_contrast"] < _THREE_STAR_ALPHA, TF_COLUMN
        ]
    )

    matrix = order_tfs_by_group_contrast(
        build_matrix(_FIG3_TF_RUNS, normalization="per_gene"),
        _FIG3_TF_LEFT_COLUMNS,
        _FIG3_TF_RIGHT_COLUMNS,
    )
    matrix = matrix.loc[[tf for tf in matrix.index if tf in three_star_tfs]]

    plot_heatmap(
        matrix,
        annotate=True,
        cell_stars=_significant_tf_cell_stars(significance),
        separator_after_column=None,
        annotate_values=False,
        add_colorbar=False,
        ax=ax,
    )
    ax.figure.colorbar(ax.collections[0], cax=colorbar_ax, label="diff_calc / gene")

    # Force every TF label to show: in this shorter axes seaborn's default "auto"
    # y-ticks would otherwise thin the labels to every other TF.
    n_rows, n_columns = matrix.shape
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(matrix.index, rotation=0)

    # Asterisk glyphs sit high in their text box, so seaborn's va="center" leaves
    # them above the cell centre; nudge each annotation down to centre it.
    for annotation in ax.texts:
        text_x, text_y = annotation.get_position()
        annotation.set_position((text_x, text_y + _STAR_VERTICAL_NUDGE))

    # Subtle grey dividers: vertical between the max and min run groups,
    # horizontal where the ordering score (max-group minus min-group mean)
    # changes sign (max-favoured TFs above, min-favoured below).
    ax.axvline(
        len(_FIG3_TF_LEFT_COLUMNS),
        color=_SEPARATOR_COLOR,
        linewidth=_SEPARATOR_WIDTH,
    )
    left_mean = matrix[_FIG3_TF_LEFT_COLUMNS].mean(axis=1).fillna(0.0)
    right_mean = matrix[_FIG3_TF_RIGHT_COLUMNS].mean(axis=1).fillna(0.0)
    n_more_in_max = int(((left_mean - right_mean) > 0).sum())
    if 0 < n_more_in_max < n_rows:
        ax.axhline(
            n_more_in_max,
            color=_SEPARATOR_COLOR,
            linewidth=_SEPARATOR_WIDTH,
        )


def _color_bars_neutral(ax: plt.Axes) -> None:
    """Paint all of an axes' bars a single neutral grey.

    The bar panels carry no colour meaning (sign is already read from the zero
    line), so a uniform grey keeps colour reserved for the heatmap (panel E).

    Args:
        ax: Axes whose bar patches should be recoloured.
    """
    for bar in ax.patches:
        bar.set_color(_NEUTRAL_COLOR)
        bar.set_alpha(_BAR_ALPHA)


def _populate_fig3(fig: plt.Figure) -> None:
    """Draw all seven panels of figure 3 onto ``fig``.

    Layout is one flat 4-column x 5-row grid (no nested outer halves). Row 0 holds
    the two rolling-mean line panels (A: ara, B: zea), each spanning two columns;
    row 1 is a short strip holding the shared A/C/G/T legend; rows 2-4 have equal
    height. The two net-nucleotide-change bar plots (C: ara, D: zea) sit
    one-per-column in the left two columns of row 2; the two mutation-distance
    difference panels (E: ara, F: zea) each span the left two columns on rows 3
    and 4; the significant-TF heatmap (G) spans the right two columns across
    rows 2-4, with a dedicated thin colour-bar column split off inside it. The
    line and bar plots share one A/C/G/T nucleotide colour scheme, explained by
    the single legend in row 1. Must be called inside a :func:`publication_style`
    context.

    Args:
        fig: An (empty) figure to populate.
    """
    # Flat grid: row 0 the line panels, row 1 the (short) shared legend spanning
    # all columns, rows 2-4 equal height. Columns are equal width; A/B and E/F
    # span two columns each.
    grid = fig.add_gridspec(
        nrows=5, ncols=4, height_ratios=[1.0, 0.2, 1.0, 1.0, 1.0]
    )

    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")

    # Heatmap spans the right two columns of rows 2-4; split off a thin colour-bar
    # column beside its (label-narrowed) body.
    heatmap_cells = grid[2:5, 2:4].subgridspec(
        nrows=1, ncols=2, width_ratios=[1.0, 0.045], wspace=0.05
    )

    # --- A, B: net-change rolling-mean line plots (row 0, two columns each) ----
    line_axes = []
    line_cells = [grid[0, 0:2], grid[0, 2:4]]
    for column_index, (run, letter) in enumerate(zip(_FIG3_RUNS, ["A", "B"])):
        line_ax = fig.add_subplot(line_cells[column_index])
        _, _, net_change_by_position = calculate_positional_nucleotide_change(
            run["mutated_sequences_json"]
        )
        make_line_plot_rolling_window(
            net_change_by_position,
            f"{_UNUSED_NAME}_diff",
            _UNUSED_FORMAT,
            titles=False,
            ax=line_ax,
            plot_sum=False,
        )
        # The shared figure legend replaces the per-panel one.
        panel_legend = line_ax.get_legend()
        if panel_legend is not None:
            panel_legend.remove()
        line_ax.set_title(run["species_label"], fontstyle="italic")
        # A and B share the y-axis label: keep it on A only.
        if column_index == 1:
            line_ax.set_ylabel("")
        panel_label(line_ax, letter)
        line_axes.append(line_ax)

    # --- C, D: net-nucleotide-change bar plots (row 2, one per left column) ---
    bar_axes = []
    bar_cells = [grid[2, 0], grid[2, 1]]
    for column_index, (run, letter) in enumerate(zip(_FIG3_RUNS, ["C", "D"])):
        bar_ax = fig.add_subplot(bar_cells[column_index])
        net_change = calculate_net_nucleotide_change(run["mutated_sequences_json"])
        # plot_net_nucleotide_change already colours the bars by A/C/G/T, matching
        # the line plots, so no neutral recolouring here.
        plot_net_nucleotide_change(
            net_change,
            _UNUSED_NAME,
            _UNUSED_FORMAT,
            titles=False,
            ax=bar_ax,
        )
        bar_ax.axhline(0, color="black", linewidth=0.8)
        bar_ax.set_title(run["species_label"], fontstyle="italic")
        # C and D share the y-axis label: keep it on C only.
        if column_index == 1:
            bar_ax.set_ylabel("")
        panel_label(bar_ax, letter)
        bar_axes.append(bar_ax)

    # --- E, F: mutation-distance difference panels (rows 3-4, left columns) ---
    distance_axes = []
    distance_cells = [grid[3, 0:2], grid[4, 0:2]]
    for row_index, (run, letter) in enumerate(zip(_FIG3_RUNS, ["E", "F"])):
        distance_ax = fig.add_subplot(distance_cells[row_index])
        real_proportions, random_proportions = _distance_difference_proportions(
            run["mutated_sequences_json"]
        )
        plot_difference(
            real_proportions,
            random_proportions,
            _UNUSED_NAME,
            OUTPUT_DIR,
            max_distance=_DISTANCE_MAX,
            ax=distance_ax,
        )
        distance_ax.set_title(run["species_label"], fontstyle="italic")
        _color_bars_neutral(distance_ax)
        # No colour meaning left to explain, so drop the legend entirely.
        existing_legend = distance_ax.get_legend()
        if existing_legend is not None:
            existing_legend.remove()
        # E and F share the x-axis label: keep it on F (the bottom panel) only.
        if row_index == 0:
            distance_ax.set_xlabel("")
        panel_label(distance_ax, letter)
        distance_axes.append(distance_ax)

    # --- G: significant-TF heatmap (right two columns, spans rows 2-4) --------
    heatmap_ax = fig.add_subplot(heatmap_cells[0, 0])
    colorbar_ax = fig.add_subplot(heatmap_cells[0, 1])
    _draw_significant_tf_heatmap(heatmap_ax, colorbar_ax)
    heatmap_label = panel_label(heatmap_ax, "G")

    # Make the two runs directly comparable within each panel type.
    sync_axis_limits(line_axes, sync_x=False, sync_y=True)
    sync_axis_limits(bar_axes, sync_x=False, sync_y=True)
    sync_axis_limits(distance_axes)

    # One shared legend for the A/C/G/T nucleotide colours (line and bar plots),
    # drawn in the reserved top row so no panel carries its own.
    nucleotide_handles = [
        Line2D([0], [0], color=COLORS[nucleotide], label=nucleotide)
        for nucleotide in ["A", "C", "G", "T"]
    ]
    legend_ax.legend(
        handles=nucleotide_handles,
        loc="center",
        ncol=4,
        frameon=False,
    )

    # G's long TF row labels shrink the heatmap axes far to the right, so its
    # panel letter (anchored in axes fraction) lands right of B's. Re-anchor it in
    # figure coordinates to B's letter x, just above G's own top, so the two
    # right-column letters line up. Requires a draw so the constrained layout has
    # resolved the final axes positions.
    fig.canvas.draw()
    b_axes_position = line_axes[1].get_position()
    # -0.08 and +0.02 mirror panel_label's default x and (y - 1) offsets.
    b_label_x = b_axes_position.x0 - 0.08 * b_axes_position.width
    heatmap_position = heatmap_ax.get_position()
    heatmap_label.set_transform(fig.transFigure)
    heatmap_label.set_position(
        (b_label_x, heatmap_position.y1 + 0.02 * heatmap_position.height)
    )


def fig3() -> None:
    """Compose figure 3: mutation signatures plus the significant-TF contrast.

    Panels A/B are the rolling-mean net nucleotide change along the sequence for
    the ara and zea maximization runs; panels C/D are the total net nucleotide
    change (A, C, G, T bar plots) for the same runs; panels E/F are the
    per-distance difference between the real and random-baseline inter-mutation
    distance distributions (shared limits). A/B and C/D share one A/C/G/T
    nucleotide colour scheme with a single figure legend. Panel G is the four-run
    per-gene TF diff heatmap restricted to the TFs whose ara max-vs-min paired
    contrast reaches the *** tier, ordered by group contrast so the max-favoured
    TFs sit above the min-favoured ones.
    """
    with publication_style():
        fig = plt.figure(
            figsize=figure_size_inches(DOUBLE_COLUMN_MM, 250.0), layout="constrained"
        )
        _populate_fig3(fig)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_publication_figure(fig, os.path.join(OUTPUT_DIR, "fig3_composed.svg"))
        plt.close(fig)


# --- Figure 6: STARR-seq x deepCRE positional correlation --------------------

# Publication marker/line weights for the position-series panels (A, B). The
# standalone plots use s=30 / linewidth=3, far too heavy for a compact panel.
_POSITION_SCATTER_SIZE = 4.0
_POSITION_SCATTER_ALPHA = 0.12
_POSITION_LINE_WIDTH = 1.4
# Overlay-scatter marker areas for panel C (standalone uses 35 / 55).
_OVERLAY_ALL_SIZE = 6.0
_OVERLAY_HIGHLIGHT_SIZE = 12.0
# Highlight-window connector drawn on A and B: a thin dashed vertical line at the
# fixed-window center that panel C zooms into (the peak-correlation window). Kept
# black (not the panel-C highlight red) so the figure is not overloaded with
# colour.
_HIGHLIGHT_MARKER_COLOR = "black"
_HIGHLIGHT_MARKER_WIDTH = 1.0
# Panel D: display order and pretty labels for the binding-status boxes. The
# order mirrors panel C's colour legend (binding first, then non-binding).
_BINDING_STATUS_ORDER = ["binding", "non_binding"]
_BINDING_STATUS_LABELS = {"binding": "Binding", "non_binding": "Non-binding"}
# fig6 grid geometry: two equal-width plot columns and four rows -- the A/B
# panels, a thin full-width strip for their legend, the C/D panels, and a thin
# full-width strip for their legend.
_FIG6_HEIGHT_RATIOS = [1.0, 0.15, 1.4, 0.22]
# Legend labels for panel C's two fit lines (drawn unlabelled by the overlay
# plot): the fit through all points and the fit through the highlight window.
_FIT_ALL_LABEL = "fit all data"
_FIT_HIGHLIGHT_LABEL = "fit best fit data"
# Panel C's highlight-window fit line is recoloured to black in the composition
# (the standalone overlay plot draws it in a dark red) and pushed behind every
# scatter point (all grey and highlight points sit at zorder >= 1) so it never
# occludes the data.
_HIGHLIGHT_FIT_COLOR = "black"
_HIGHLIGHT_FIT_ZORDER = 0.5


def _build_fig6_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the WRKY + bHLH deepCRE prediction pipelines once and pool the results.

    Mirrors ``starrseq_deepcre_correlation_combined.py``: each TF's analysis-ready
    dataframe is built independently (own reference windows and mapping, so genes
    shared between the two TF sets are never double-counted). Panels A/B use the
    row-concatenated pool of both TFs; panels C/D use the WRKY-only dataframe.
    Both TF dataframes come out of the identical downstream pipeline with the same
    columns.

    This runs the full deepCRE prediction pipeline (loads the TF model and scans
    the reference windows) for both TFs, so it is the slow part of :func:`fig6`;
    there is no cached point-level dataframe to load instead. The WRKY pipeline is
    run only once and reused for both the pool and the WRKY-only panels.

    Returns:
        ``(combined_df, wrky_df)`` where ``combined_df`` is the pooled
        WRKY + bHLH dataframe for panels A/B and ``wrky_df`` is the WRKY-only
        dataframe for panels C/D.
    """
    wrky_df = prepare_wrky_enrichment_df()
    bhlh_df = prepare_bhlh_enrichment_df()
    combined_df = pd.concat([wrky_df, bhlh_df], ignore_index=True)
    return combined_df, wrky_df


def _fig6_position_series(
    combined_df: pd.DataFrame,
) -> List[Tuple[pd.DataFrame, str, str]]:
    """Split the pooled dataframe into the all/light/dark position-series inputs.

    Matches ``run_correlation_analysis``: the pooled dataframe plus its light and
    dark ``condition`` subsets, each with its :data:`POSITION_SERIES_COLORS`
    colour, ready for :func:`compute_correlation_by_position`.

    Args:
        combined_df: The pooled analysis-ready dataframe.

    Returns:
        ``[(all, "all", grey), (light, "light", gold), (dark, "dark", blue)]``.
    """
    condition = combined_df["condition"].str.lower()
    return [
        (combined_df, "all", POSITION_SERIES_COLORS["all"]),
        (combined_df[condition == "light"].copy(), "light", POSITION_SERIES_COLORS["light"]),
        (combined_df[condition == "dark"].copy(), "dark", POSITION_SERIES_COLORS["dark"]),
    ]


def _draw_binding_status_boxplot(
    ax: plt.Axes, highlight_points: pd.DataFrame
) -> None:
    """Draw panel D: enrichment by binding status for the highlight-window points.

    Boxes the ``enrichment`` of the exact points panel C colours by binding status
    (the peak-correlation highlight window of the WRKY dataset), split into the
    binding and non-binding groups and coloured with the same
    :data:`BINDING_STATUS_COLORS` as panel C so the two panels read as one. A
    sample-size annotation sits above each box and a dashed zero line marks no
    enrichment.

    Args:
        ax: Axes to draw the boxplot onto.
        highlight_points: The deduplicated highlight-window points returned by
            :func:`plot_overlay_highlight_correlation` with
            ``color_by_binding=True``; must carry ``enrichment`` and
            ``starr_binding_status`` columns.
    """
    present_order = [
        status
        for status in _BINDING_STATUS_ORDER
        if (highlight_points["starr_binding_status"] == status).any()
    ]
    palette = {status: BINDING_STATUS_COLORS[status] for status in present_order}

    # hue == x with legend off is the non-deprecated way to colour by category;
    # saturation=1 keeps the box fills identical to panel C's point colours
    # (seaborn otherwise desaturates them to 0.75).
    sns.boxplot(
        data=highlight_points,
        x="starr_binding_status",
        y="enrichment",
        order=present_order,
        hue="starr_binding_status",
        hue_order=present_order,
        palette=palette,
        saturation=1.0,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("STARR-seq Enrichment")
    # Set ticks before labels so matplotlib does not warn about a FixedFormatter
    # without a matching FixedLocator.
    ax.set_xticks(range(len(present_order)))
    ax.set_xticklabels([_BINDING_STATUS_LABELS[status] for status in present_order])

    # Sample size above each box (just under the top of the axes).
    y_top = ax.get_ylim()[1]
    for position, status in enumerate(present_order):
        count = int((highlight_points["starr_binding_status"] == status).sum())
        ax.text(
            position,
            y_top,
            f"n={count}",
            ha="center",
            va="top",
        )


def _populate_fig6(
    fig: plt.Figure, combined_df: pd.DataFrame, wrky_df: pd.DataFrame
) -> None:
    """Draw all four panels of figure 6 onto ``fig``.

    Layout is a 4-column x 4-row grid. Row 0: panel A (Spearman correlation by
    overlap position) and panel B (p-value of that correlation, log y) over the
    pooled WRKY + bHLH dataset, two columns each. Row 1: a thin full-width strip
    holding their all/light/dark condition legend. Row 2: panel C (the WRKY
    deepCRE-vs-STARR-seq scatter with the peak-correlation window highlighted and
    coloured by binding status) spanning three columns and panel D (a boxplot of
    those highlight-window points' enrichment, binding vs non-binding) the last
    column. Row 3: a thin full-width strip holding the all/binding/non-binding +
    fit-line legend. A and B each mark the highlighted
    window's center with a thin dashed vertical line so the reader sees which
    window panels C/D zoom into. Panels C and D share the binding-status colours,
    so one legend keys both. Must be called inside a :func:`publication_style`
    context.

    Args:
        fig: An (empty) figure to populate.
        combined_df: The pooled WRKY + bHLH analysis-ready dataframe (panels A/B)
            from :func:`_build_fig6_dataframes`.
        wrky_df: The WRKY-only analysis-ready dataframe (panels C/D) from
            :func:`_build_fig6_dataframes`.
    """
    # Four equal-width columns: A and B take two columns each on the top row,
    # while C spans three columns and D one on the panel row below; each panel row
    # is followed by a thin full-width strip that holds that row's legend.
    grid = fig.add_gridspec(
        nrows=4, ncols=4, height_ratios=_FIG6_HEIGHT_RATIOS
    )
    correlation_ax = fig.add_subplot(grid[0, 0:2])
    pvalue_ax = fig.add_subplot(grid[0, 2:4])
    top_legend_ax = fig.add_subplot(grid[1, :])
    top_legend_ax.axis("off")
    overlay_ax = fig.add_subplot(grid[2, 0:3])
    boxplot_ax = fig.add_subplot(grid[2, 3])
    bottom_legend_ax = fig.add_subplot(grid[3, :])
    bottom_legend_ax.axis("off")

    named_dfs = _fig6_position_series(combined_df)
    corr_series, _, pval_series, _ = compute_correlation_by_position(named_dfs)

    simply_plot_multi(
        corr_series,
        "Overlap start position",
        "Spearman correlation",
        _UNUSED_NAME,
        _UNUSED_NAME,
        _UNUSED_NAME,
        ax=correlation_ax,
        scatter_size=_POSITION_SCATTER_SIZE,
        scatter_alpha=_POSITION_SCATTER_ALPHA,
        line_width=_POSITION_LINE_WIDTH,
    )
    simply_plot_multi(
        pval_series,
        "Overlap start position",
        "P-value (Spearman)",
        _UNUSED_NAME,
        _UNUSED_NAME,
        _UNUSED_NAME,
        log=True,
        ax=pvalue_ax,
        scatter_size=_POSITION_SCATTER_SIZE,
        scatter_alpha=_POSITION_SCATTER_ALPHA,
        line_width=_POSITION_LINE_WIDTH,
    )

    # Capture the shared all/light/dark handles from A and drop the per-panel
    # legends both A and B auto-created; the combined strip legend is built below.
    shared_handles, shared_labels = correlation_ax.get_legend_handles_labels()
    for ax in (correlation_ax, pvalue_ax):
        panel_legend = ax.get_legend()
        if panel_legend is not None:
            panel_legend.remove()

    # Panel C: WRKY-only overlay, highlight-window points coloured by binding
    # status. The returned highlight points feed panel D so both panels show the
    # exact same set of points.
    _, highlight_points, _, _ = plot_overlay_highlight_correlation(
        wrky_df, color_by_binding=True, ax=overlay_ax
    )
    overlay_scatters = overlay_ax.collections
    # Collection 0 is the grey full-dataset background; the remaining collections
    # are the binding/non-binding highlight groups, drawn larger.
    for scatter_index, scatter in enumerate(overlay_scatters):
        size = _OVERLAY_ALL_SIZE if scatter_index == 0 else _OVERLAY_HIGHLIGHT_SIZE
        scatter.set_sizes([size])
    # Recolour panel C's highlight-window fit line black (the overlay plot draws
    # it in a dark red) and push it behind every scatter point so it no longer
    # occludes the data; the all-data fit line stays grey. Done before reading the
    # line handles so the legend proxy picks up the black colour. Line order is
    # [all-data fit, highlight-window fit].
    overlay_fit_lines = list(overlay_ax.get_lines())
    if len(overlay_fit_lines) >= 2:
        overlay_fit_lines[1].set_color(_HIGHLIGHT_FIT_COLOR)
        overlay_fit_lines[1].set_zorder(_HIGHLIGHT_FIT_ZORDER)

    # Extract panel C's all/binding/non-binding scatter key, append its two fit
    # lines, and drop the in-panel legend the overlay plot auto-created.
    overlay_handles, overlay_labels = overlay_ax.get_legend_handles_labels()
    overlay_legend = overlay_ax.get_legend()
    if overlay_legend is not None:
        overlay_legend.remove()
    fit_labels = [_FIT_ALL_LABEL, _FIT_HIGHLIGHT_LABEL]
    overlay_handles = list(overlay_handles) + overlay_fit_lines[:2]
    overlay_labels = list(overlay_labels) + fit_labels[: len(overlay_fit_lines[:2])]

    # Panel D: enrichment boxplot of the same highlight-window points, split by
    # binding status (shares panel C's colours).
    _draw_binding_status_boxplot(boxplot_ax, highlight_points)

    # Each panel row's legend sits in the thin full-width strip below it, laid
    # out horizontally: the all/light/dark conditions under A/B, the
    # binding-status + fit-line key under C/D.
    top_legend_ax.legend(
        shared_handles,
        shared_labels,
        loc="center",
        ncol=len(shared_labels),
        frameon=False,
        title="STARR-seq condition",
    )
    bottom_legend_ax.legend(
        overlay_handles,
        overlay_labels,
        loc="center",
        ncol=len(overlay_labels),
        frameon=False,
        title="Binding status",
        markerscale=2.2,
    )

    # Strip the long standalone titles; the axis labels and panel letters carry
    # the meaning in a multi-panel figure.
    for ax in (correlation_ax, pvalue_ax, overlay_ax):
        ax.set_title("")

    # Connect the top panels to panel C: the peak-correlation window C highlights
    # is the fixed window plotted at x = HIGHLIGHT_WINDOW_CENTER on A and B.
    for ax in (correlation_ax, pvalue_ax):
        ax.axvline(
            HIGHLIGHT_WINDOW_CENTER,
            color=_HIGHLIGHT_MARKER_COLOR,
            linewidth=_HIGHLIGHT_MARKER_WIDTH,
            linestyle="--",
            zorder=0,
        )

    # Both top panels share the same overlap-position x-axis; harmonise it.
    sync_axis_limits([correlation_ax, pvalue_ax], sync_y=False)

    for ax, letter in (
        (correlation_ax, "A"),
        (pvalue_ax, "B"),
        (overlay_ax, "C"),
        (boxplot_ax, "D"),
    ):
        panel_label(ax, letter)


def fig6() -> None:
    """Compose figure 6: STARR-seq x deepCRE positional correlation.

    Panels A and B summarise the fixed-window positional analysis of the pooled
    WRKY + bHLH dataset: the Spearman correlation between deepCRE predictions and
    STARR-seq enrichment as a function of overlap start position (A) and the
    p-value of that correlation on a log axis (B), each overlaying the ``all``,
    ``light`` and ``dark`` STARR-seq conditions. Panel C is the WRKY-only
    point-level deepCRE-vs-STARR-seq scatter with the peak-correlation window
    (marked by the dashed line on A/B) highlighted on top of the full WRKY dataset
    and coloured by binding status. Panel D boxes the enrichment of those exact
    highlight-window points, split into binding vs non-binding.

    The dataframes are rebuilt from scratch on every call (both deepCRE
    prediction pipelines run); there is no cached point-level dataframe.
    """
    with publication_style():
        fig = plt.figure(
            figsize=figure_size_inches(DOUBLE_COLUMN_MM, 170.0), layout="constrained"
        )
        combined_df, wrky_df = _build_fig6_dataframes()
        _populate_fig6(fig, combined_df, wrky_df)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_publication_figure(fig, os.path.join(OUTPUT_DIR, "fig6_composed.svg"))
        plt.close(fig)


if __name__ == "__main__":
    # fig2()
    # fig3()
    fig6()
