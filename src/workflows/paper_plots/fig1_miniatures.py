"""Small standalone inset illustrations for figure 1.

Each function here renders one tiny, publication-styled panel meant to sit as
an illustrative inset inside the larger composed figure 1 (a graphical
abstract), reusing the exact data sources and plotting building blocks of the
figures they are miniaturized from:

- ``pareto_front_mini``: the example gene's final Pareto front, from
  :mod:`workflows.evo_alg_pooled_plots.example_gene_pareto.example_gene_pareto`
  (the same source as figure 1's earlier incremental series).
- ``deepcis_scan_mini``: the example gene's reference-vs-optimized deepCIS
  scan, from the same CSV and gene/TF pair as figure 4's scan track, with the
  introduced-mutation and TSS/TTS markers switched off; keeps a small
  "Reference"/"Optimized" legend since the two curves are otherwise
  indistinguishable.
- ``fig6c_mini``: the STARR-seq enrichment vs. deepCRE prediction points
  inside the best-fit overlap window from figure 6 panel C, without the
  binding-status split or the full-data background points, and without a
  legend.
- ``fig3g_mini``: the significant-TF heatmap from figure 3 panel G, restricted
  to the top/bottom 3 TFs by group contrast, without significance stars, with
  the ``_tnt`` suffix dropped from TF names, and the colour bar labelled
  "introduction frequency".

Each panel is saved as a standalone SVG under ``OUTPUT_DIR``.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps
from matplotlib.figure import Figure

from analysis.motives.deepcis_visualize import _plot_gene_tf
from workflows.evo_alg_pooled_plots.example_gene_pareto.example_gene_pareto import (
    load_final_front,
)
from workflows.overlap_analysis._common import (
    HIGHLIGHT_WINDOW_CENTER,
    compute_overlay_correlation_data,
)
from workflows.overlap_analysis.starrseq_deepcre_correlation_WRKY import (
    prepare_wrky_enrichment_df,
)
from workflows.evo_alg_pooled_plots.tf_comparison.tf_comparison_calc import (
    build_matrix,
    order_tfs_by_group_contrast,
    paired_tf_significance,
    TF_COLUMN,
)
from workflows.evo_alg_pooled_plots.tf_comparison.tf_comparison_plot import plot_heatmap
from workflows.paper_plots.compose_plots_mpl import (
    _ARA_MAX_DIR,
    _ARA_MIN_DIR,
    _FIG3_TF_LEFT_COLUMNS,
    _FIG3_TF_RIGHT_COLUMNS,
    _FIG3_TF_RUNS,
    _THREE_STAR_ALPHA,
)
from workflows.paper_plots.style import figure_size_inches, publication_style

OUTPUT_DIR = "src/workflows/paper_plots/figures/mpl_compositions/fig1_miniatures"

# Size of every miniature panel, in millimetres.
_MINI_SIZE_MM = (55.0, 36.0)
# The deepCIS scan miniature is wider than the others: its x-axis spans the
# full 3020 bp extraction window, so it needs more room than the other panels.
_DEEPCIS_MINI_SIZE_MM = (55.0, 36.0)
# Upper y-axis limit for the deepCIS scan miniature, giving the "Reference"/
# "Optimized" legend headroom in the top-left corner above the curves (which
# otherwise occupy the full 0-1 binding-score range).
_DEEPCIS_MINI_Y_MAX = 1.3

# Colormap and shade for the Pareto-front miniature (matches the darkest end of
# the full incremental series' purple/magenta gradient).
_PARETO_CMAP = "Purples"
_PARETO_COLOR_SHADE = 0.85

# Colour for the fig 6c miniature's scatter/fit-line, taken from the magma
# colormap instead of the full figure's binding-status highlight colour.
_FIG6C_MINI_CMAP = "magma"
_FIG6C_MINI_COLOR_SHADE = 0.2

# Same example gene, deepCIS scan CSV, and TF family as figure 4's scan track
# (compose_plots_mpl.py's ``_FIG4_DEEPCIS_*`` constants), so the inset matches
# the panel it is a miniature of.
_DEEPCIS_SCAN_CSV = (
    "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/GOF_LOF/GOF/"
    "deepdive_simons_gene/deepcis_scan/"
    "deepcis_window_scan_deepdive_simons_gene_mut5.csv"
)
_DEEPCIS_GENE = "3_AT3G60640_gene:22415750-22417548_260312_000750_357934"
_DEEPCIS_TF = "LOBAS2_tnt"

# The figure 3 panel G heatmap miniature keeps only the top/bottom 3 TFs by
# group contrast, so it needs more vertical room per row than the other
# minis but stays narrow (short TF labels, no colour-bar column reservation).
_TF_HEATMAP_MINI_SIZE_MM = (55.0, 40.0)
# Number of TFs kept at each end of the group-contrast ordering.
_TF_HEATMAP_MINI_N_EXTREME = 3
# Suffix stripped from TF names for display (matches the deepCIS model naming
# convention, not a meaningful part of the TF family name).
_TF_NAME_SUFFIX = "_tnt"


def _save_mini_figure(fig: Figure, filename: str) -> None:
    """Save a miniature panel figure to ``OUTPUT_DIR/filename`` as SVG only.

    Args:
        fig: The figure to save.
        filename: File name including the ``.svg`` extension.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight")
    plt.close(fig)


def pareto_front_mini() -> None:
    """Render and save the final-Pareto-front miniature.

    Loads the example gene's final front via
    :func:`example_gene_pareto.load_final_front` and scatters it in a single
    dark purple shade, with axis labels but no title or legend.
    """
    mutation_counts, predictions = load_final_front()
    color = colormaps[_PARETO_CMAP](_PARETO_COLOR_SHADE)
    with publication_style():
        fig, ax = plt.subplots(figsize=figure_size_inches(*_MINI_SIZE_MM))
        ax.scatter(mutation_counts, predictions, color=color, edgecolors="none")
        ax.set_xlabel("Mutation count")
        ax.set_ylabel("deepCRE prediction")
        fig.tight_layout(pad=0.1)
    _save_mini_figure(fig, "pareto_front_mini.svg")


def deepcis_scan_mini() -> None:
    """Render and save the deepCIS scan miniature.

    Draws the same example gene/TF reference-vs-optimized track as figure 4's
    scan panel, but without the introduced-mutation markers or the TSS/TTS
    markers. Unlike the other two miniatures, this one keeps a small
    "Reference"/"Optimized" legend in the top-left corner, since the two
    curves are otherwise indistinguishable; the y-axis is extended above 1.0
    to give that legend headroom above the curves.
    """
    scan_data = pd.read_csv(_DEEPCIS_SCAN_CSV)
    gene_data = scan_data[scan_data["gene"] == _DEEPCIS_GENE]
    if gene_data.empty:
        raise ValueError(f"Gene {_DEEPCIS_GENE!r} not found in {_DEEPCIS_SCAN_CSV}.")

    with publication_style():
        fig, ax = plt.subplots(figsize=figure_size_inches(*_DEEPCIS_MINI_SIZE_MM))
        _plot_gene_tf(
            gene_data,
            _DEEPCIS_GENE,
            _DEEPCIS_TF,
            ax,
            include_title=False,
            highlight_peaks=False,
            show_difference=False,
            show_tss_tts=False,
            mutation_positions=None,
        )
        ax.set_xlabel("Position in extraction window")
        ax.set_ylabel("TF binding score")
        ax.set_ylim(top=_DEEPCIS_MINI_Y_MAX)
        ax.grid(False)
        existing_legend = ax.get_legend()
        if existing_legend is not None:
            labels = [text.get_text() for text in existing_legend.get_texts()]
            handles = existing_legend.legend_handles
            existing_legend.remove()
            ax.legend(handles=handles, labels=labels, loc="upper left", frameon=False)
        fig.tight_layout(pad=0.1)
    _save_mini_figure(fig, "deepcis_scan_mini.svg")


def fig6c_mini() -> None:
    """Render and save the figure 6 panel C miniature.

    Scatters only the WRKY points inside the best-fit overlap window (the same
    window figure 6 panel C highlights, centered at
    :data:`HIGHLIGHT_WINDOW_CENTER`) with their linear fit, without the
    full-data background points, the binding-status colour split, or a legend.
    """
    wrky_df = prepare_wrky_enrichment_df()
    _, highlight_points, _, highlight_fit = compute_overlay_correlation_data(
        wrky_df,
        "prediction_mutated",
        "enrichment",
        HIGHLIGHT_WINDOW_CENTER,
    )
    slope, intercept, _, _ = highlight_fit
    color = colormaps[_FIG6C_MINI_CMAP](_FIG6C_MINI_COLOR_SHADE)

    with publication_style():
        fig, ax = plt.subplots(figsize=figure_size_inches(*_MINI_SIZE_MM))
        ax.scatter(
            highlight_points["prediction_mutated"],
            highlight_points["enrichment"],
            color=color,
            alpha=0.85,
            edgecolors="none",
        )
        if not highlight_points.empty and pd.notna(slope) and pd.notna(intercept):
            line_x = pd.Series(
                [
                    highlight_points["prediction_mutated"].min(),
                    highlight_points["prediction_mutated"].max(),
                ]
            )
            ax.plot(line_x, slope * line_x + intercept, color=color)
        ax.set_xlabel("deepCRE prediction")
        ax.set_ylabel("STARR-seq enrichment")
        fig.tight_layout(pad=0.1)
    _save_mini_figure(fig, "fig6c_mini.svg")


def fig3g_mini() -> None:
    """Render and save the figure 3 panel G heatmap miniature.

    Reuses the same three-star ara max-vs-min TF selection and group-contrast
    ordering as figure 3 panel G, then keeps only the top/bottom
    :data:`_TF_HEATMAP_MINI_N_EXTREME` TFs. Unlike panel G, only the two ara
    runs are shown (the GOF/LOF columns are dropped), relabelled "max"/"min".
    Significance stars are dropped (cells are unannotated), the ``_tnt``
    model-naming suffix is stripped from TF labels, the "run" x-axis label is
    dropped, and the colour bar is labelled "introduction frequency".
    """
    significance = paired_tf_significance(_ARA_MAX_DIR, _ARA_MIN_DIR)
    three_star_tfs = set(
        significance.loc[significance["q_contrast"] < _THREE_STAR_ALPHA, TF_COLUMN]
    )

    matrix = order_tfs_by_group_contrast(
        build_matrix(_FIG3_TF_RUNS, normalization="per_gene"),
        _FIG3_TF_LEFT_COLUMNS,
        _FIG3_TF_RIGHT_COLUMNS,
    )
    matrix = matrix.loc[[tf for tf in matrix.index if tf in three_star_tfs]]
    matrix = pd.concat(
        [matrix.head(_TF_HEATMAP_MINI_N_EXTREME), matrix.tail(_TF_HEATMAP_MINI_N_EXTREME)]
    )
    matrix.index = [
        name[: -len(_TF_NAME_SUFFIX)] if name.endswith(_TF_NAME_SUFFIX) else name
        for name in matrix.index
    ]
    matrix = matrix[["ara max", "ara min"]].rename(
        columns={"ara max": "max", "ara min": "min"}
    )

    with publication_style():
        fig, ax = plt.subplots(figsize=figure_size_inches(*_TF_HEATMAP_MINI_SIZE_MM))
        plot_heatmap(
            matrix,
            annotate=False,
            cbar_label="introduction frequency",
            separator_after_column=1,
            ax=ax,
        )
        ax.set_xlabel("")
        fig.tight_layout(pad=0.1)
    _save_mini_figure(fig, "fig3g_mini.svg")


def main() -> None:
    """Render and save all four figure 1 miniature insets."""
    pareto_front_mini()
    deepcis_scan_mini()
    fig6c_mini()
    fig3g_mini()


if __name__ == "__main__":
    main()
