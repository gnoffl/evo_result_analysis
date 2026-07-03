"""Integration smoke test for the model_comparison thin script."""

import json
import os
import tempfile
import unittest
from typing import List, Tuple

import pandas as pd

from workflows.evo_alg_pooled_plots.tf_comparison import model_comparison


def _write_full_run(
    run_dir: str,
    summary_df: pd.DataFrame,
    peaks: List[Tuple[str, str, str, int]],
) -> None:
    """Create a run dir with stats JSON, peak summary, and annotated peaks.

    The gene count for the stats JSON is inferred from the distinct core gene ids
    in ``peaks`` so per-gene normalization stays consistent.
    """
    scan_dir = os.path.join(run_dir, "deepcis_scan")
    os.makedirs(scan_dir, exist_ok=True)

    core_genes = {"_".join(gene.split("_")[:2]) for gene, _, _, _ in peaks}
    stats = {gene: {"final_fitness": 0.5} for gene in core_genes}
    with open(os.path.join(run_dir, "stats_run.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle)

    summary_df.to_csv(os.path.join(scan_dir, "run_peak_summary.csv"), index=False)

    records = [
        {"gene": gene, "tf": tf, "signal_type": signal, "peak_area": 1.0}
        for gene, tf, signal, count in peaks
        for _ in range(count)
    ]
    frame = pd.DataFrame(records, columns=["gene", "tf", "signal_type", "peak_area"])
    frame.to_csv(os.path.join(scan_dir, "run_annotated_peaks_x.csv"), index=False)


def _summary(tfs: List[str], diffs: List[int]) -> pd.DataFrame:
    """Minimal peak summary with the columns build_matrix needs."""
    return pd.DataFrame(
        {
            "tf": tfs,
            "diff_calc": diffs,
            "max_mutated": [abs(d) + 1 for d in diffs],
            "reference": [1 for _ in diffs],
        }
    )


def _pair_peaks(
    gene_indices: List[int], sig_mutated_count: int
) -> List[Tuple[str, str, str, int]]:
    """Annotated peaks over ``gene_indices`` for a SIG and a FLAT TF.

    ``sig_mutated_count`` controls SIG's optimized peak count (reference 1), so a
    larger value on one run of a pair makes the paired model contrast significant.
    """
    peaks: List[Tuple[str, str, str, int]] = []
    for gene_index in gene_indices:
        gene = f"{gene_index}_GENE{gene_index}_loc_{gene_index}"
        peaks += [(gene, "SIG", "reference", 1), (gene, "SIG", "max_mutated", sig_mutated_count)]
        peaks += [(gene, "FLAT", "reference", 2), (gene, "FLAT", "max_mutated", 2)]
    return peaks


class ModelComparisonIntegrationTest(unittest.TestCase):
    """End-to-end run of the 2x2 orchestration against synthetic runs in a temp dir."""

    def test_creates_expected_csv_and_png_outputs(self) -> None:
        # Arrange: four synthetic runs (ara-gene pair 0-7, zea-gene pair 8-15).
        with tempfile.TemporaryDirectory() as tmp:
            ara_gene_ara_model = os.path.join(tmp, "araG_araM")
            zea_gene_ara_model = os.path.join(tmp, "zeaG_araM")
            ara_gene_zea_model = os.path.join(tmp, "araG_zeaM")
            zea_gene_zea_model = os.path.join(tmp, "zeaG_zeaM")
            output_dir = os.path.join(tmp, "out")

            ara_genes = list(range(8))
            zea_genes = list(range(8, 16))
            # Within each gene source the ara model introduces SIG (mutated 3) and
            # the zea model leaves it flat (mutated 1) -> significant model contrast.
            _write_full_run(ara_gene_ara_model, _summary(["SIG", "FLAT"], [2, 0]), _pair_peaks(ara_genes, 3))
            _write_full_run(ara_gene_zea_model, _summary(["SIG", "FLAT"], [0, 0]), _pair_peaks(ara_genes, 1))
            _write_full_run(zea_gene_ara_model, _summary(["SIG", "FLAT"], [2, 0]), _pair_peaks(zea_genes, 3))
            _write_full_run(zea_gene_zea_model, _summary(["SIG", "FLAT"], [0, 0]), _pair_peaks(zea_genes, 1))

            runs = [
                (ara_gene_ara_model, "araG/araM"),
                (zea_gene_ara_model, "zeaG/araM"),
                (ara_gene_zea_model, "araG/zeaM"),
                (zea_gene_zea_model, "zeaG/zeaM"),
            ]
            ara_gene_pair = (
                (ara_gene_ara_model, "araG/araM"),
                (ara_gene_zea_model, "araG/zeaM"),
            )
            zea_gene_pair = (
                (zea_gene_ara_model, "zeaG/araM"),
                (zea_gene_zea_model, "zeaG/zeaM"),
            )

            # Act
            model_comparison.main(
                runs=runs,
                left_columns=["araG/araM", "zeaG/araM"],
                right_columns=["araG/zeaM", "zeaG/zeaM"],
                ara_gene_pair=ara_gene_pair,
                zea_gene_pair=zea_gene_pair,
                output_dir=output_dir,
            )

            # Assert: all four significance CSVs and both heatmap PNGs were written.
            expected = [
                "model_comparison_significance_ara_genes.csv",
                "model_comparison_significance_zea_genes.csv",
                "model_comparison_significance_pooled_model.csv",
                "model_comparison_significance_interaction.csv",
                "model_comparison_per_gene.png",
                "model_comparison_log_fold_change.png",
            ]
            for name in expected:
                path = os.path.join(output_dir, name)
                self.assertTrue(os.path.exists(path), f"Expected output {name} was not created")
                self.assertGreater(os.path.getsize(path), 0, f"Output {name} is empty")


if __name__ == "__main__":
    unittest.main()
