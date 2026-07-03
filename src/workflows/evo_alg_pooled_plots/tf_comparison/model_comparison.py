"""2x2 gene-source x fitness-model TF comparison, grouped by model.

This is a thin orchestration script over the TF-comparison toolbox
(``tf_comparison_calc`` + ``tf_comparison_plot``). It compares four maximization
runs forming a 2x2 design: natural genes from two species (Arabidopsis "ara",
*Zea mays* "zea") each optimized under two deepCRE fitness models (ara model,
zea model). All four runs maximize predicted expression.

The figure is grouped by model — left group = both ara-model runs, right group =
both zea-model runs, with a divider after column 2 — to answer: *does the
predictor model drive the TF strategy?* Rows are ordered by ``mean(ara-model) -
mean(zea-model)``.

Significance layers (see the folder's ``summary.md`` for the rationale):

- **Cell (intra-run) stars** for all four runs, appended to each cell's numeric
  annotation. They are reused from the two stratified paired tables' ``q_a`` /
  ``q_b`` columns (one paired table per gene source), so no separate single-run
  tests are needed.
- **Row-label (inter-run) stars are dropped from the figure** by choice; the
  inter-run significance lives in CSVs only.

Statistics written as CSV only (not shown on the figure):

- Two stratified paired model contrasts (one per gene source).
- **A** — pooled model main effect (``pooled_model_tf_significance``).
- **C** — interaction / species-specificity (``interaction_model_tf_significance``).

The pairing structure: the two ara-gene runs share the same ara genes; the two
zea-gene runs share the same zea genes; ara and zea gene sets are disjoint. So
the model swap is paired by gene within each gene source, with a consistent
A = ara model, B = zea model convention.

Run it directly to regenerate the CSVs and PNGs in ``model_comparison/``.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from workflows.evo_alg_pooled_plots.tf_comparison.tf_comparison_calc import (
    TF_COLUMN,
    build_matrix,
    interaction_model_tf_significance,
    order_tfs_by_group_contrast,
    paired_tf_significance,
    pooled_model_tf_significance,
    top_bottom_tfs,
)
from workflows.evo_alg_pooled_plots.tf_comparison.tf_comparison_plot import (
    plot_heatmap,
    q_to_stars,
)

# Hardcoded run directories (as in minmax_comparison). Each run holds one
# stats_*.json and exactly one *_annotated_peaks_*.csv in deepcis_scan.
_COMPARISON_ROOT = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/comparison_zea_ara/SSR_regular_extractionwindow/analysis_gernot"
_ARA_GENE_ARA_MODEL_DIR = f"{_COMPARISON_ROOT}/ara_genes_ara_model_ssr_regular"
_ZEA_GENE_ARA_MODEL_DIR = f"{_COMPARISON_ROOT}/zea_genes_ara_model_ssr_regular"
_ARA_GENE_ZEA_MODEL_DIR = f"{_COMPARISON_ROOT}/ara_genes_zea_model_ssr_regular"
_ZEA_GENE_ZEA_MODEL_DIR = f"{_COMPARISON_ROOT}/zea_genes_zea_model_ssr_regular"

# Labels (gene source / model). Tweakable at review.
LABEL_ARA_GENE_ARA_MODEL = "araG/araM"
LABEL_ZEA_GENE_ARA_MODEL = "zeaG/araM"
LABEL_ARA_GENE_ZEA_MODEL = "araG/zeaM"
LABEL_ZEA_GENE_ZEA_MODEL = "zeaG/zeaM"

# Runs in display order (grouped by model: ara model first, then zea model).
RUNS: List[Tuple[str, str]] = [
    (_ARA_GENE_ARA_MODEL_DIR, LABEL_ARA_GENE_ARA_MODEL),
    (_ZEA_GENE_ARA_MODEL_DIR, LABEL_ZEA_GENE_ARA_MODEL),
    (_ARA_GENE_ZEA_MODEL_DIR, LABEL_ARA_GENE_ZEA_MODEL),
    (_ZEA_GENE_ZEA_MODEL_DIR, LABEL_ZEA_GENE_ZEA_MODEL),
]
LEFT_COLUMNS: List[str] = [LABEL_ARA_GENE_ARA_MODEL, LABEL_ZEA_GENE_ARA_MODEL]
RIGHT_COLUMNS: List[str] = [LABEL_ARA_GENE_ZEA_MODEL, LABEL_ZEA_GENE_ZEA_MODEL]

# Paired-by-gene-source contrasts, A = ara model, B = zea model in both.
# Each pair is ((model_a_dir, model_a_label), (model_b_dir, model_b_label)).
ARA_GENE_PAIR: Tuple[Tuple[str, str], Tuple[str, str]] = (
    (_ARA_GENE_ARA_MODEL_DIR, LABEL_ARA_GENE_ARA_MODEL),
    (_ARA_GENE_ZEA_MODEL_DIR, LABEL_ARA_GENE_ZEA_MODEL),
)
ZEA_GENE_PAIR: Tuple[Tuple[str, str], Tuple[str, str]] = (
    (_ZEA_GENE_ARA_MODEL_DIR, LABEL_ZEA_GENE_ARA_MODEL),
    (_ZEA_GENE_ZEA_MODEL_DIR, LABEL_ZEA_GENE_ZEA_MODEL),
)

OUTPUT_DIR = Path(__file__).parent / "model_comparison"
OUTPUT_BASENAME = "model_comparison"
FIGURE_FORMAT = "png"
TOP_BOTTOM_N_TFS: Optional[int] = None

# (normalization key, filename suffix, colour-bar label) per heatmap variant.
NORMALIZATIONS: List[Tuple[str, str, str]] = [
    ("per_gene", "_per_gene", "diff_calc / gene"),
    ("fold_change", "_log_fold_change", "log2 fold change"),
]

# Column-name suffixes that make the CSV "_a"/"_b" sides self-describing. The two
# axes differ: the paired tables contrast the two MODELS within a gene source
# (A = ara model, B = zea model), while the interaction table contrasts the two
# GENE SOURCES (A = ara genes, B = zea genes).
MODEL_A_LABEL = "ara_model"
MODEL_B_LABEL = "zea_model"
GENE_GROUP_A_LABEL = "ara_genes"
GENE_GROUP_B_LABEL = "zea_genes"


def _paired_contrast_cell_stars(
    pair: Tuple[Tuple[str, str], Tuple[str, str]],
    output_dir: Path,
    csv_basename: str,
) -> Dict[str, Dict[str, str]]:
    """Run one gene source's paired contrast, save its CSV, return its cell stars.

    The paired table covers both of the pair's runs over that gene source's
    shared genes, so its ``q_<MODEL_A_LABEL>`` / ``q_<MODEL_B_LABEL>`` columns give
    the two runs' intra-run stars directly.

    Args:
        pair: ``((model_a_dir, model_a_label), (model_b_dir, model_b_label))``.
        output_dir: Folder to write the significance CSV into.
        csv_basename: File name (without extension) for the CSV.

    Returns:
        ``{model_a_run_label: {tf: stars}, model_b_run_label: {tf: stars}}`` from
        the ara-model and zea-model q-value columns respectively.
    """
    (dir_a, run_label_a), (dir_b, run_label_b) = pair
    significance = paired_tf_significance(
        dir_a, dir_b, label_a=MODEL_A_LABEL, label_b=MODEL_B_LABEL
    )
    significance_path = output_dir / f"{csv_basename}.csv"
    significance.to_csv(significance_path, index=False)
    print(f"Saved {significance_path}")
    return {
        run_label_a: dict(
            zip(significance[TF_COLUMN], significance[f"q_{MODEL_A_LABEL}"].map(q_to_stars))
        ),
        run_label_b: dict(
            zip(significance[TF_COLUMN], significance[f"q_{MODEL_B_LABEL}"].map(q_to_stars))
        ),
    }


def main(
    runs: List[Tuple[str, str]] = RUNS,
    left_columns: List[str] = LEFT_COLUMNS,
    right_columns: List[str] = RIGHT_COLUMNS,
    ara_gene_pair: Tuple[Tuple[str, str], Tuple[str, str]] = ARA_GENE_PAIR,
    zea_gene_pair: Tuple[Tuple[str, str], Tuple[str, str]] = ZEA_GENE_PAIR,
    output_dir: os.PathLike = OUTPUT_DIR,
    top_bottom_n: Optional[int] = TOP_BOTTOM_N_TFS,
) -> None:
    """Compute all significance tables, write CSVs, and save the heatmaps.

    Args:
        runs: ``(run_directory, label)`` tuples in display order (grouped by model).
        left_columns: Labels of the left (ara-model) group, for ordering and the
            heatmap separator.
        right_columns: Labels of the right (zea-model) group.
        ara_gene_pair: The two ara-gene runs as
            ``((ara_model_dir, label), (zea_model_dir, label))`` (A = ara model).
        zea_gene_pair: The two zea-gene runs, same A/B convention.
        output_dir: Folder for the CSV and PNG outputs (created if absent).
        top_bottom_n: Keep only the top-N/bottom-N TFs per heatmap; None keeps all.
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Two stratified paired contrasts (one per gene source). Their q_a / q_b give
    # the intra-run cell stars for all four runs.
    cell_stars: Dict[str, Dict[str, str]] = {}
    cell_stars.update(
        _paired_contrast_cell_stars(
            ara_gene_pair, output_dir, f"{OUTPUT_BASENAME}_significance_ara_genes"
        )
    )
    cell_stars.update(
        _paired_contrast_cell_stars(
            zea_gene_pair, output_dir, f"{OUTPUT_BASENAME}_significance_zea_genes"
        )
    )

    # Directory-only pairs (A = ara model, B = zea model) for the pooled / interaction tests.
    ara_dir_pair = (ara_gene_pair[0][0], ara_gene_pair[1][0])
    zea_dir_pair = (zea_gene_pair[0][0], zea_gene_pair[1][0])

    # A — pooled model main effect across both gene sources.
    pooled = pooled_model_tf_significance([ara_dir_pair, zea_dir_pair])
    pooled_path = output_dir / f"{OUTPUT_BASENAME}_significance_pooled_model.csv"
    pooled.to_csv(pooled_path, index=False)
    print(f"Saved {pooled_path}")

    # C — interaction / species-specificity of the model effect.
    interaction = interaction_model_tf_significance(
        ara_dir_pair, zea_dir_pair, label_a=GENE_GROUP_A_LABEL, label_b=GENE_GROUP_B_LABEL
    )
    interaction_path = output_dir / f"{OUTPUT_BASENAME}_significance_interaction.csv"
    interaction.to_csv(interaction_path, index=False)
    print(f"Saved {interaction_path}")

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
            row_stars=None,
            cell_stars=cell_stars,
            separator_after_column=len(left_columns),
        )
        output_path = output_dir / f"{OUTPUT_BASENAME}{suffix}{top_bottom_suffix}.{FIGURE_FORMAT}"
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
