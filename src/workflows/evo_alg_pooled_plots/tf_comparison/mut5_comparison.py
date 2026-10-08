"""Reproduce the minmax TF comparison figure at a fixed 5-mutation budget.

This is a variant of ``minmax_comparison.py`` (see that module's docstring for
the general design) that compares each run's **reference** sequence against its
**exactly-5-mutations** pareto entry, instead of its maximally-mutated entry.
Everything else — the four runs, the left/right grouping, the paired and
intra-run significance layers — mirrors ``minmax_comparison.py``.

The underlying data comes from a *separate* deepCIS rerun
(``run_deepcis_peak_pipeline.sh --mutation-count 5``) written to dedicated
``*_mut5`` output folders, so it does not touch the max-mutation outputs that
``minmax_comparison.py`` reads. That rerun used the current pipeline, which
labels the optimized sequence ``"optimized"`` rather than the older pipeline's
``"max_mutated"`` (see ``tf_comparison_calc.py``'s ``mutated_column``
parameter) — every call below passes ``mutated_column="optimized"``
accordingly.

Run it directly to regenerate the CSVs and PNGs in ``mut5_comparison/``.
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

# Optimized-sequence label used by the current pipeline (as opposed to the
# older "max_mutated" label that minmax_comparison.py's runs use).
MUTATED_COLUMN = "optimized"

_ARA_MAX_DIR = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/ara_msr_max_single_mut5"
_GOF_DIR = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/GOF_LOF/GOF/GOF_single_mut5"
_ARA_MIN_DIR = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/ara_msr_min_single_mut5"
_LOF_DIR = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/GOF_LOF/LOF/LOF_single_mut5"

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

OUTPUT_DIR = Path(__file__).parent / "mut5_comparison"
OUTPUT_BASENAME = "compare_TFs_mut5"
SIGNIFICANCE_BASENAME = "compare_TFs_mut5_significance"
FIGURE_FORMAT = "png"
# Draw the heatmaps rotated by 90° (TFs along the x axis) for a wide figure.
TRANSPOSE_HEATMAP = True
# The transposed cells are narrow, so the value+star annotations need a smaller font.
ANNOTATION_FONTSIZE = 7.0
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
    transpose: bool = TRANSPOSE_HEATMAP,
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
        transpose: Draw TFs along the x axis (True) or y axis (False, matching
            ``minmax_comparison.py``'s orientation).
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    paired_dir_a, paired_label_a = paired_run_a
    paired_dir_b, paired_label_b = paired_run_b

    # Inter-run paired significance for the two runs that share genes.
    significance = paired_tf_significance(
        paired_dir_a, paired_dir_b, mutated_column=MUTATED_COLUMN
    )
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
        intra_sig = single_run_tf_significance(run_dir, mutated_column=MUTATED_COLUMN)
        safe_label = label.replace(" ", "_")
        intra_path = output_dir / f"{SIGNIFICANCE_BASENAME}_{safe_label}.csv"
        intra_sig.to_csv(intra_path, index=False)
        print(f"Saved {intra_path}")
        cell_stars[label] = dict(zip(intra_sig[TF_COLUMN], intra_sig["q_intra"].map(q_to_stars)))

    top_bottom_suffix = f"_top{top_bottom_n}" if top_bottom_n is not None else ""
    for normalization, suffix, cbar_label in NORMALIZATIONS:
        matrix = order_tfs_by_group_contrast(
            build_matrix(runs, normalization=normalization, mutated_column=MUTATED_COLUMN),
            left_columns,
            right_columns,
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
            transpose=transpose,
            annotation_fontsize=ANNOTATION_FONTSIZE if transpose else None,
        )
        output_path = output_dir / f"{OUTPUT_BASENAME}{suffix}{top_bottom_suffix}.{FIGURE_FORMAT}"
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
    main(top_bottom_n=5, transpose=False)
