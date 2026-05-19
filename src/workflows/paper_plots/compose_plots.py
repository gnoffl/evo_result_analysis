"""
Example composition script: combine a set of existing PDF plots into a single
publication-ready figure.

Edit the ``PANELS`` list and ``OUTPUT_PATH`` / ``FIGSIZE`` for your specific
figure. Run directly:

    python src/workflows/compose_example_figure.py

Layout reference (fractions of the output page, origin = top-left):

    (0.0, 0.0) ─────────────── (1.0, 0.0)
         │                           │
         │   x0,y0 ── x1,y0          │
         │    │          │           │
         │   x0,y1 ── x1,y1          │
         │                           │
    (0.0, 1.0) ─────────────── (1.0, 1.0)

Common layouts
--------------
Two panels side by side (equal halves):
    left:  (0.0, 0.0, 0.5, 1.0)
    right: (0.5, 0.0, 1.0, 1.0)

2 × 2 grid:
    top-left:     (0.0, 0.0, 0.5, 0.5)
    top-right:    (0.5, 0.0, 1.0, 0.5)
    bottom-left:  (0.0, 0.5, 0.5, 1.0)
    bottom-right: (0.5, 0.5, 1.0, 1.0)

One wide top + two narrow bottom:
    top:          (0.0, 0.0, 1.0, 0.5)
    bottom-left:  (0.0, 0.5, 0.5, 1.0)
    bottom-right: (0.5, 0.5, 1.0, 1.0)
"""

import os

from workflows.figure_composition import compose_figures

# ---------------------------------------------------------------------------
# Edit here: paths to your existing PDFs and their positions in the figure
# ---------------------------------------------------------------------------

def fig2():
    images = [
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/ara_msr_max_single/average_pareto_front_arabidopsis_toolkit_msr_max_single_260224_155640_550482.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/ara_msr_max_single/hist_half_max_mutations_ara_msr_max_single.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/ara_msr_min_single/average_pareto_front_arabidopsis_toolkit_msr_min_single_260224_115131_937839.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/ara_msr_min_single/hist_half_max_mutations_ara_msr_min_single.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/zea_msr_max_single/average_pareto_front_maize_toolkit_msr_max_single_260225_023558_509964.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/zea_msr_max_single/hist_half_max_mutations_zea_msr_max_single.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/zea_msr_min_single/average_pareto_front_maize_toolkit_msr_min_single_260224_233447_380062.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/zea_msr_min_single/hist_half_max_mutations_zea_msr_min_single.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/random_max_msr_single/average_pareto_front_random_sequences_max_msr_single_260224_122135_065103.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/random_max_msr_single/hist_half_max_mutations_random_max_msr_single.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/ara_msr_max_greedy/average_pareto_front_ara_greedy_msr_max_260304_165758_335723.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/ara_msr_max_greedy/hist_half_max_mutations_ara_msr_max_greedy.svg"
    ]
    PANELS = [
        {
            "path": images[0],   # <-- replace with actual path
            "rect": (0.0, 0.0, 0.25, 0.333),   # left half
            "page": 0,
        },
        {
            "path": images[1],   # <-- replace with actual path
            "rect": (0.25, 0.0, 0.5, 0.333),   # right half
            "page": 0,
        },
        {
            "path": images[2],   # <-- replace with actual path
            "rect": (0.5, 0.0, 0.75, 0.333),   # right half
            "page": 0,
        },
        {
            "path": images[3],   # <-- replace with actual path
            "rect": (0.75, 0.0, 1.0, 0.333),   # right half
            "page": 0,
        },
        {
            "path": images[4],   # <-- replace with actual path
            "rect": (0.0, 0.333, 0.25, 0.666),   # right half
            "page": 0,
        },
        {
            "path": images[5],   # <-- replace with actual path
            "rect": (0.25, 0.333, 0.5, 0.666),   # right half
            "page": 0,
        },
        {
            "path": images[6],   # <-- replace with actual path
            "rect": (0.5, 0.333, 0.75, 0.666),   # right half
            "page": 0,
        },
        {
            "path": images[7],   # <-- replace with actual path
            "rect": (0.75, 0.333, 1.0, 0.666),   # right half
            "page": 0,
        },
        {
            "path": images[8],   # <-- replace with actual path
            "rect": (0, 0.666, 0.25, 1.0),
        },
        {
            "path": images[9],   # <-- replace with actual path
            "rect": (0.25, 0.666, 0.5, 1.0),
        },
        {
            "path": images[10],   # <-- replace with actual path
            "rect": (0.5, 0.666, 0.75, 1.0),
        },
        {
            "path": images[11],   # <-- replace with actual path
            "rect": (0.75, 0.666, 1.0, 1.0),
        },
    ]
    # A4 landscape: (11.69, 8.27) | A4 portrait: (8.27, 11.69)
    # Two-column journal (140 mm wide): (5.51, 3.94)
    # Single-column journal (85 mm wide): (3.35, 3.35)
    FIGSIZE = (11.69, 6)  # A4 portrait, inches

    compose_figures(
        panels=PANELS,
        output_path="src/workflows/paper_plots/figures/fig2_composed.svg",
        figsize=FIGSIZE,
        labels=True,
        label_fontsize=14.0,
    )


def fig3():
    images = [
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/ara_msr_max_single/rolling_mean_mutations_ara_msr_max_single_diff_31.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/ara_msr_max_single/mutation_distances_ara_msr_max_single_smaller_distances.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/zea_msr_max_single/rolling_mean_mutations_zea_msr_max_single_diff_31.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/zea_msr_max_single/mutation_distances_zea_msr_max_single_smaller_distances.svg",
    ]
    panels = [
        {
            "path": images[0],
            "rect": (0, 0, 0.4, 0.25)
        },
        {
            "path": images[1],
            "rect": (0.4, 0, 1, 0.25)
        },
        {
            "path": images[2],
            "rect": (0, 0.25, 0.4, 0.5)
        },
        {
            "path": images[3],
            "rect": (0.4, 0.25, 1, 0.5)
        },
    ]
    FIGSIZE = (6, 8)
    compose_figures(
        panels=panels,
        output_path="src/workflows/paper_plots/figures/fig3_composed.svg",
        figsize=FIGSIZE,
        labels=True,
        label_fontsize=14.0,
    )

def fig4():
    images = [
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/GOF_LOF/GOF/GOF_single/average_pareto_front_GOF_single_mutation_251009_121226_109368.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/GOF_LOF/GOF/GOF_single/hist_half_max_mutations_GOF_single.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/GOF_LOF/LOF/LOF_single/average_pareto_front_LOF_single_mutation_251020_180028_564570.svg",
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/GOF_LOF/LOF/LOF_single/hist_half_max_mutations_LOF_single.svg",
        ""
    ]
    panels = [
        {
            "path": images[0],
            "rect": (0, 0, 0.4, 0.25)
        },
        {
            "path": images[1],
            "rect": (0.4, 0, 1, 0.25)
        },
        {
            "path": images[2],
            "rect": (0, 0.25, 0.4, 0.5)
        },
        {
            "path": images[3],
            "rect": (0.4, 0.25, 1, 0.5)
        },
    ]
    FIGSIZE = (6, 8)
    compose_figures(
        panels=panels,
        output_path="src/workflows/paper_plots/figures/fig4_composed.svg",
        figsize=FIGSIZE,
        labels=True,
        label_fontsize=14.0,
    )

def fig_5_starrseq():
    images = [
        "/home/gernot/ARCitect/ARCs/genRE/assays/Evolution/protocols/Tobias/motif_mutation/analysis/medium_window_WRKY/sequence_heat_map_positive_reverse_mutation_only_only_core.png",
        "/home/gernot/ARCitect/ARCs/genRE/assays/Evolution/protocols/Tobias/motif_mutation/analysis/medium_window_WRKY/sequence_heat_map_reverse_flank_impacted.png",
    ]
    panels = [
        {
            "path": images[0],
            "rect": (0, 0, 0.5, 1)
        },
        {
            "path": images[1],
            "rect": (0.5, 0, 1, 1)
        },
    ]
    FIGSIZE = (12, 6)
    compose_figures(
        panels=panels,
        output_path="src/workflows/paper_plots/figures/fig5_starrseq_composed.svg",
        figsize=FIGSIZE,
        labels=True,
        label_fontsize=14.0,
    )

if __name__ == "__main__":
    # fig2()
    # fig3()
    # fig4()
    fig_5_starrseq()
