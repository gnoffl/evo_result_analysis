"""Single-run TF comparison for a random-start maximization run.

This is the single-run analogue of ``minmax_comparison.py``: a thin orchestration
script over the TF-comparison toolbox (``tf_comparison_calc`` +
``tf_comparison_plot``). It analyses one maximization run whose starting
sequences are *random* (not natural plant genes) and renders an annotated
diverging heatmap of per-TF binding change, ordered by overall mean.

One significance layer is overlaid (see the folder's ``summary.md`` for the
rationale): the **intra-run** test for the single run, whose ``q_intra`` stars
are appended to each cell's numeric annotation. No inter-run (row-label) stars
apply because there is only one run.

Question it answers: *starting from random sequence, which TF families does the
optimizer systematically introduce/remove to raise predicted expression?*

Run it directly to regenerate the CSV and PNGs in ``random_start_comparison/``.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from workflows.evo_alg_pooled_plots.tf_comparison.tf_comparison_calc import (
    TF_COLUMN,
    build_matrix,
    order_tfs_by_mean,
    per_gene_tf_binding_summary,
    single_run_tf_significance,
    top_bottom_tfs,
)
from workflows.evo_alg_pooled_plots.tf_comparison.tf_comparison_plot import (
    plot_heatmap,
    q_to_stars,
)

# Hardcoded run directory (as in minmax_comparison).
# NOTE: deepcis_scan must hold exactly one *_annotated_peaks_*.csv, or
# load_per_gene_diffs raises FileNotFoundError. Remove any stale timestamped copy.
_RANDOM_MAX_DIR = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/paper_runs/single_mutation/random_max_msr_single"

RUN_DIR: str = _RANDOM_MAX_DIR
LABEL: str = "random max"

# Random-start sequences are named ``random_sequence_<index>_<timestamp>``, whose
# first two fields are always ``random_sequence``; 3 fields (``random_sequence_000``)
# keep each sequence a distinct replicate. Natural-gene runs use the default 2.
CORE_ID_FIELDS: int = 3

OUTPUT_DIR = Path(__file__).parent / "random_start_comparison"
OUTPUT_BASENAME = "random_start_comparison"
SIGNIFICANCE_BASENAME = "random_start_comparison_significance"
BINDING_SUMMARY_BASENAME = "random_start_comparison_binding_summary"
FIGURE_FORMAT = "png"
TOP_BOTTOM_N_TFS: Optional[int] = None

# (normalization key, filename suffix, colour-bar label) per heatmap variant.
NORMALIZATIONS: List[Tuple[str, str, str]] = [
    ("per_gene", "_per_gene", "diff_calc / gene"),
    ("fold_change", "_log_fold_change", "log2 fold change"),
]


def main(
    run_dir: str = RUN_DIR,
    label: str = LABEL,
    output_dir: os.PathLike = OUTPUT_DIR,
    top_bottom_n: Optional[int] = TOP_BOTTOM_N_TFS,
    core_id_fields: int = CORE_ID_FIELDS,
) -> None:
    """Write the per-TF binding summary and significance CSVs, and save the heatmaps.

    Args:
        run_dir: Random-start maximization run directory.
        label: Column/label to give the single run in the matrix and figure.
        output_dir: Folder for the CSV and PNG outputs (created if absent).
        top_bottom_n: Keep only the top-N/bottom-N TFs per heatmap; None keeps all.
        core_id_fields: Number of leading underscore-separated fields identifying a
            replicate (3 for random-start sequences); see
            :func:`load_per_gene_diffs`.
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Descriptive per-TF binding: mean +/- std over genes for the reference,
    # optimized, and diff peak counts.
    binding_summary = per_gene_tf_binding_summary(run_dir, core_id_fields)
    binding_summary_path = output_dir / f"{BINDING_SUMMARY_BASENAME}.csv"
    binding_summary.to_csv(binding_summary_path, index=False)
    print(f"Saved {binding_summary_path}")

    # Intra-run significance for the single run.
    significance = single_run_tf_significance(run_dir, core_id_fields)
    significance_path = output_dir / f"{SIGNIFICANCE_BASENAME}.csv"
    significance.to_csv(significance_path, index=False)
    print(f"Saved {significance_path}")
    cell_stars: Dict[str, Dict[str, str]] = {
        label: dict(zip(significance[TF_COLUMN], significance["q_intra"].map(q_to_stars)))
    }

    top_bottom_suffix = f"_top{top_bottom_n}" if top_bottom_n is not None else ""
    for normalization, suffix, cbar_label in NORMALIZATIONS:
        matrix = order_tfs_by_mean(build_matrix([(run_dir, label)], normalization=normalization))
        if top_bottom_n is not None:
            matrix = top_bottom_tfs(matrix, top_bottom_n)
        fig = plot_heatmap(
            matrix,
            annotate=True,
            cbar_label=cbar_label,
            row_stars=None,
            cell_stars=cell_stars,
            separator_after_column=None,
        )
        output_path = output_dir / f"{OUTPUT_BASENAME}{suffix}{top_bottom_suffix}.{FIGURE_FORMAT}"
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
