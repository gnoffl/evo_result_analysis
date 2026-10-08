"""Compare the TF binding changes of differently *constrained* GOF / LOF runs.

Each direction (GOF, LOF) was optimized four times under a different constraint
on which mutations the evolutionary algorithm was allowed to make:

* ``unconstrained`` — any single-nucleotide substitution
* ``natural`` — only substitutions observed in natural accessions
* ``CRISPR`` / ``CRISPR 3PAM`` — only substitutions reachable by base editing
  (the two CRISPR runs differ in how permissive the PAM requirement is)

All four runs of a direction share the same genes (105 for GOF, 68 for LOF), so
their per-TF diff columns are directly comparable. This module only *arranges*
that data: values, ordering and statistics all come from ``tf_comparison_calc``
and the heatmap from ``tf_comparison_plot``, exactly as
``minmax_comparison.py`` / ``mut5_comparison.py`` do.

Output per direction, in ``constraint_comparison/``:

* one heatmap per normalization, TFs ordered by mean value, with **intra-run**
  significance stars in each cell (Wilcoxon of that run's per-gene diff vs 0)
* the intra-run significance CSV per run
* a paired significance CSV for every one of the six run pairs, so any
  constraint-vs-constraint contrast can be looked up even though the figures
  show none of them

All these runs were scanned with the older pipeline, which labels the optimized
sequence ``"max_mutated"`` — the ``tf_comparison_calc`` default, so no
``mutated_column`` argument is needed anywhere below.

Note: the ``per_gene`` normalization divides by the gene count read from a run's
``stats_*.json``. The two CRISPR run folders contain only ``deepcis_scan`` and
no such file, so that normalization raises until the stats JSONs are added.
"""

import itertools
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from workflows.evo_alg_pooled_plots.tf_comparison.tf_comparison_calc import (
    TF_COLUMN,
    build_matrix,
    order_tfs_by_mean,
    paired_tf_significance,
    single_run_tf_significance,
)
from workflows.evo_alg_pooled_plots.tf_comparison.tf_comparison_plot import (
    plot_heatmap,
    q_to_stars,
)

_DATASET = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset"

# (run directory, display label) per constraint, in display order.
GOF_RUNS: List[Tuple[str, str]] = [
    (f"{_DATASET}/GOF_LOF/GOF/GOF_single", "unconstrained"),
    (f"{_DATASET}/GOF_LOF/GOF/GOF_single_natural", "natural"),
    (f"{_DATASET}/GOF_LOF_Crispr/GOF_Master", "CRISPR"),
    (f"{_DATASET}/GOF_LOF_Crispr/GOF_3PAM_Master", "CRISPR 3PAM"),
]
LOF_RUNS: List[Tuple[str, str]] = [
    (f"{_DATASET}/GOF_LOF/LOF/LOF_single", "unconstrained"),
    (f"{_DATASET}/GOF_LOF/LOF/LOF_single_natural", "natural"),
    (f"{_DATASET}/GOF_LOF_Crispr/LOF_Master", "CRISPR"),
    (f"{_DATASET}/GOF_LOF_Crispr/LOF_3PAM_Master", "CRISPR 3PAM"),
]

OUTPUT_DIR = Path(__file__).parent / "constraint_comparison"
FIGURE_FORMAT = "png"
# Tall layout: TFs along the y axis, the four runs along the x axis. The cells are
# wide enough for the default annotation font, so no size override is needed.
TRANSPOSE_HEATMAP = False
ANNOTATION_FONTSIZE = 7.0

# (normalization key, filename suffix, colour-bar label) per heatmap variant.
NORMALIZATIONS: List[Tuple[str, str, str]] = [
    ("per_gene", "_per_gene", "diff_calc / gene"),
    ("fold_change", "_log_fold_change", "log2 fold change"),
]


def _safe(label: str) -> str:
    """Return a label usable in a filename or a pandas column suffix."""
    return label.replace(" ", "_")


def write_intra_run_significance(
    runs: List[Tuple[str, str]], direction: str, output_dir: Path
) -> Dict[str, Dict[str, str]]:
    """Write one intra-run significance CSV per run and return its stars.

    Args:
        runs: ``(run_directory, label)`` tuples.
        direction: ``"GOF"`` or ``"LOF"``, used in the output filenames.
        output_dir: Existing folder to write the CSVs into.

    Returns:
        ``{run label: {tf: stars}}`` from the ``q_intra`` column, ready to pass to
        :func:`plot_heatmap` as ``cell_stars``.
    """
    cell_stars: Dict[str, Dict[str, str]] = {}
    for run_dir, label in runs:
        significance = single_run_tf_significance(run_dir)
        path = output_dir / f"{direction}_intra_{_safe(label)}.csv"
        significance.to_csv(path, index=False)
        print(f"Saved {path}")
        cell_stars[label] = dict(
            zip(significance[TF_COLUMN], significance["q_intra"].map(q_to_stars))
        )
    return cell_stars


def write_pairwise_significance(
    runs: List[Tuple[str, str]], direction: str, output_dir: Path
) -> None:
    """Write a paired significance CSV for every pair of runs.

    The figures show no cross-run contrast, so these CSVs are the only place the
    constraint-vs-constraint tests live.

    Args:
        runs: ``(run_directory, label)`` tuples; all pairs are tested.
        direction: ``"GOF"`` or ``"LOF"``, used in the output filenames.
        output_dir: Existing folder to write the CSVs into.
    """
    for (dir_a, label_a), (dir_b, label_b) in itertools.combinations(runs, 2):
        name_a, name_b = _safe(label_a), _safe(label_b)
        significance = paired_tf_significance(dir_a, dir_b, name_a, name_b)
        path = output_dir / f"{direction}_paired_{name_a}_vs_{name_b}.csv"
        significance.to_csv(path, index=False)
        print(f"Saved {path}")


def compare_constraints(
    runs: List[Tuple[str, str]],
    direction: str,
    output_dir: os.PathLike = OUTPUT_DIR,
    transpose: bool = TRANSPOSE_HEATMAP,
) -> None:
    """Write the significance CSVs and heatmaps for one direction's four runs.

    Args:
        runs: ``(run_directory, label)`` tuples in display order.
        direction: ``"GOF"`` or ``"LOF"``; prefixes every output filename.
        output_dir: Folder for the CSV and figure outputs (created if absent).
        transpose: Draw TFs along the x axis (True) or the y axis (False).
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cell_stars = write_intra_run_significance(runs, direction, output_dir)
    write_pairwise_significance(runs, direction, output_dir)

    for normalization, suffix, cbar_label in NORMALIZATIONS:
        matrix = order_tfs_by_mean(build_matrix(runs, normalization=normalization))
        figure = plot_heatmap(
            matrix,
            annotate=True,
            cbar_label=cbar_label,
            cell_stars=cell_stars,
            separator_after_column=None,
            transpose=transpose,
            annotation_fontsize=ANNOTATION_FONTSIZE if transpose else None,
        )
        path = output_dir / f"{direction}_constraints{suffix}.{FIGURE_FORMAT}"
        figure.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(figure)
        print(f"Saved {path}")


def main() -> None:
    """Produce the GOF and the LOF constraint comparison, each in its own figure."""
    compare_constraints(GOF_RUNS, "GOF")
    compare_constraints(LOF_RUNS, "LOF")


if __name__ == "__main__":
    main()
