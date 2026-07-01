"""Reproduce the minimization-vs-maximization TF comparison figure.

This is a thin orchestration script over the TF-comparison toolbox
(``tf_comparison_calc`` + ``tf_comparison_plot``). It compares four evolutionary
runs — two maximization runs (Arabidopsis "ara max", "GOF") on the left and two
minimization runs ("ara min", "LOF") on the right — and renders an annotated
diverging heatmap of per-TF binding change, ordered so that max and min read as
mirror images.

Two significance layers are overlaid (see the folder's ``summary.md`` for the
rationale):

1. **Inter-run paired contrast** for the two ara runs (they share genes): the
   ``q_contrast`` stars are appended to the TF row labels.
2. **Intra-run** tests for every run: ``q`` stars are appended to each cell's
   numeric annotation. The ara runs reuse the ``q_a`` / ``q_b`` columns already
   produced by the paired analysis; GOF/LOF are tested independently.

Run it directly to regenerate the CSVs and PNGs in ``minmax_comparison/``.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from workflows.evo_alg_pooled_plots.tf_comparison.tf_comparison_calc import (
    TF_COLUMN,
    build_matrix,
    order_tfs_by_group_contrast,
    paired_tf_significance,
    single_run_tf_significance,
    top_bottom_tfs,
)
from workflows.evo_alg_pooled_plots.tf_comparison.tf_comparison_plot import (
    plot_heatmap,
    q_to_stars,
)

_ARA_MAX_DIR = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/ara_msr_max_single"
_GOF_DIR = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/GOF_LOF/GOF/GOF_single"
_ARA_MIN_DIR = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/ara_msr_min_single"
_LOF_DIR = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/GOF_LOF/LOF/LOF_single"

# Runs in display order: maximization runs first, then minimization runs.
RUNS: List[Tuple[str, str]] = [
    (_ARA_MAX_DIR, "ara max"),
    (_GOF_DIR, "GOF"),
    (_ARA_MIN_DIR, "ara min"),
    (_LOF_DIR, "LOF"),
]
LEFT_COLUMNS: List[str] = ["ara max", "GOF"]
RIGHT_COLUMNS: List[str] = ["ara min", "LOF"]

# The two ara runs share genes and drive the paired inter-run contrast.
PAIRED_RUN_A: Tuple[str, str] = (_ARA_MAX_DIR, "ara max")
PAIRED_RUN_B: Tuple[str, str] = (_ARA_MIN_DIR, "ara min")
# Runs that only get an independent intra-run test (no paired partner).
SINGLE_RUNS: List[Tuple[str, str]] = [(_GOF_DIR, "GOF"), (_LOF_DIR, "LOF")]

OUTPUT_DIR = Path(__file__).parent / "minmax_comparison"
OUTPUT_BASENAME = "compare_TFs"
SIGNIFICANCE_BASENAME = "compare_TFs_significance"
FIGURE_FORMAT = "png"
TOP_BOTTOM_N_TFS: Optional[int] = None

# (normalization key, filename suffix, colour-bar label) per heatmap variant.
NORMALIZATIONS: List[Tuple[str, str, str]] = [
    ("per_gene", "_per_gene", "diff_calc / gene"),
    ("fold_change", "_log_fold_change", "log2 fold change"),
]


def main(
    runs: List[Tuple[str, str]] = RUNS,
    left_columns: List[str] = LEFT_COLUMNS,
    right_columns: List[str] = RIGHT_COLUMNS,
    paired_run_a: Tuple[str, str] = PAIRED_RUN_A,
    paired_run_b: Tuple[str, str] = PAIRED_RUN_B,
    single_runs: List[Tuple[str, str]] = SINGLE_RUNS,
    output_dir: os.PathLike = OUTPUT_DIR,
    top_bottom_n: Optional[int] = TOP_BOTTOM_N_TFS,
) -> None:
    """Compute both significance layers, write CSVs, and save the heatmaps.

    Args:
        runs: ``(run_directory, label)`` tuples in display order.
        left_columns: Labels of the left (e.g. maximization) group, for ordering
            and the heatmap separator.
        right_columns: Labels of the right (e.g. minimization) group.
        paired_run_a: ``(directory, label)`` of the A side of the paired contrast.
        paired_run_b: ``(directory, label)`` of the B side of the paired contrast.
        single_runs: ``(directory, label)`` tuples tested only intra-run.
        output_dir: Folder for the CSV and PNG outputs (created if absent).
        top_bottom_n: Keep only the top-N/bottom-N TFs per heatmap; None keeps all.
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    paired_dir_a, paired_label_a = paired_run_a
    paired_dir_b, paired_label_b = paired_run_b

    # Inter-run paired significance for the two runs that share genes.
    significance = paired_tf_significance(paired_dir_a, paired_dir_b)
    significance_path = output_dir / f"{SIGNIFICANCE_BASENAME}.csv"
    significance.to_csv(significance_path, index=False)
    print(f"Saved {significance_path}")
    row_stars = dict(zip(significance[TF_COLUMN], significance["q_contrast"].map(q_to_stars)))

    # Intra-run stars. The paired runs reuse q_a / q_b from the paired analysis;
    # the single runs are tested independently.
    cell_stars: Dict[str, Dict[str, str]] = {
        paired_label_a: dict(zip(significance[TF_COLUMN], significance["q_a"].map(q_to_stars))),
        paired_label_b: dict(zip(significance[TF_COLUMN], significance["q_b"].map(q_to_stars))),
    }
    for run_dir, label in single_runs:
        intra_sig = single_run_tf_significance(run_dir)
        safe_label = label.replace(" ", "_")
        intra_path = output_dir / f"{SIGNIFICANCE_BASENAME}_{safe_label}.csv"
        intra_sig.to_csv(intra_path, index=False)
        print(f"Saved {intra_path}")
        cell_stars[label] = dict(zip(intra_sig[TF_COLUMN], intra_sig["q_intra"].map(q_to_stars)))

    top_bottom_suffix = f"_top{top_bottom_n}" if top_bottom_n is not None else ""
    for normalization, suffix, cbar_label in NORMALIZATIONS:
        matrix = order_tfs_by_group_contrast(
            build_matrix(runs, normalization=normalization), left_columns, right_columns
        )
        if top_bottom_n is not None:
            matrix = top_bottom_tfs(matrix, top_bottom_n)
        fig = plot_heatmap(
            matrix,
            annotate=True,
            cbar_label=cbar_label,
            row_stars=row_stars,
            cell_stars=cell_stars,
            separator_after_column=len(left_columns),
        )
        output_path = output_dir / f"{OUTPUT_BASENAME}{suffix}{top_bottom_suffix}.{FIGURE_FORMAT}"
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
