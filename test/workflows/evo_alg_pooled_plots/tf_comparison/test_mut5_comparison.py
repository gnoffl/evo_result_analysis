"""Integration smoke test for the mut5_comparison thin script."""

import json
import os
import tempfile
import unittest
from typing import List, Tuple

import pandas as pd

from workflows.evo_alg_pooled_plots.tf_comparison import mut5_comparison


def _write_full_run(
    run_dir: str,
    summary_df: pd.DataFrame,
    peaks: List[Tuple[str, str, str, int]],
) -> None:
    """Create a run dir with stats JSON, peak summary, and annotated peaks.

    Mirrors ``test_minmax_comparison._write_full_run`` but uses the
    ``"optimized"`` signal-type label (the current pipeline's label) instead of
    the older pipeline's ``"max_mutated"``, matching what mut5_comparison.py's
    ``MUTATED_COLUMN`` expects.
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
            "optimized": [abs(d) + 1 for d in diffs],
            "reference": [1 for _ in diffs],
        }
    )


def _shared_gene_peaks(prefix: str) -> List[Tuple[str, str, str, int]]:
    """Annotated peaks for the two ara runs over 8 shared genes.

    ``prefix`` distinguishes a maximization-like run (SIG introduced) from a
    minimization-like run (SIG removed) so the paired contrast is significant.
    """
    peaks: List[Tuple[str, str, str, int]] = []
    for i in range(8):
        gene = f"{i}_GENE{i}_loc_{i}"
        if prefix == "max":
            peaks += [(gene, "SIG", "reference", 1), (gene, "SIG", "optimized", 3)]
        else:
            peaks += [(gene, "SIG", "reference", 1)]
        peaks += [(gene, "FLAT", "reference", 2), (gene, "FLAT", "optimized", 2)]
    return peaks


class Mut5ComparisonIntegrationTest(unittest.TestCase):
    """End-to-end run of the orchestration against synthetic runs in a temp dir."""

    def test_creates_expected_csv_and_png_outputs(self) -> None:
        # Arrange: four synthetic runs plus an output dir, all under one temp root.
        with tempfile.TemporaryDirectory() as tmp:
            ara_max_dir = os.path.join(tmp, "ara_max")
            gof_dir = os.path.join(tmp, "gof")
            ara_min_dir = os.path.join(tmp, "ara_min")
            lof_dir = os.path.join(tmp, "lof")
            output_dir = os.path.join(tmp, "out")

            _write_full_run(ara_max_dir, _summary(["SIG", "FLAT"], [2, 0]), _shared_gene_peaks("max"))
            _write_full_run(ara_min_dir, _summary(["SIG", "FLAT"], [-1, 0]), _shared_gene_peaks("min"))
            _write_full_run(
                gof_dir,
                _summary(["SIG", "FLAT"], [2, 0]),
                [("1_G_a_0", "SIG", "reference", 1), ("1_G_a_0", "SIG", "optimized", 3)],
            )
            _write_full_run(
                lof_dir,
                _summary(["SIG", "FLAT"], [-1, 0]),
                [("1_G_a_0", "SIG", "reference", 3), ("1_G_a_0", "SIG", "optimized", 1)],
            )

            runs = [
                (ara_max_dir, "ara max"),
                (gof_dir, "GOF"),
                (ara_min_dir, "ara min"),
                (lof_dir, "LOF"),
            ]

            # Act
            mut5_comparison.main(
                runs=runs,
                left_columns=["ara max", "GOF"],
                right_columns=["ara min", "LOF"],
                paired_run_a=(ara_max_dir, "ara max"),
                paired_run_b=(ara_min_dir, "ara min"),
                single_runs=[(gof_dir, "GOF"), (lof_dir, "LOF")],
                output_dir=output_dir,
            )

            # Assert: both significance CSVs and both heatmap PNGs were written.
            expected = [
                "compare_TFs_mut5_significance.csv",
                "compare_TFs_mut5_significance_GOF.csv",
                "compare_TFs_mut5_significance_LOF.csv",
                "compare_TFs_mut5_per_gene.png",
                "compare_TFs_mut5_log_fold_change.png",
            ]
            for name in expected:
                self.assertTrue(
                    os.path.exists(os.path.join(output_dir, name)),
                    f"Expected output {name} was not created",
                )

    def test_top_bottom_n_adds_suffixed_heatmaps(self) -> None:
        # Arrange: same synthetic runs as the default-output test.
        with tempfile.TemporaryDirectory() as tmp:
            ara_max_dir = os.path.join(tmp, "ara_max")
            gof_dir = os.path.join(tmp, "gof")
            ara_min_dir = os.path.join(tmp, "ara_min")
            lof_dir = os.path.join(tmp, "lof")
            output_dir = os.path.join(tmp, "out")

            _write_full_run(ara_max_dir, _summary(["SIG", "FLAT"], [2, 0]), _shared_gene_peaks("max"))
            _write_full_run(ara_min_dir, _summary(["SIG", "FLAT"], [-1, 0]), _shared_gene_peaks("min"))
            _write_full_run(
                gof_dir,
                _summary(["SIG", "FLAT"], [2, 0]),
                [("1_G_a_0", "SIG", "reference", 1), ("1_G_a_0", "SIG", "optimized", 3)],
            )
            _write_full_run(
                lof_dir,
                _summary(["SIG", "FLAT"], [-1, 0]),
                [("1_G_a_0", "SIG", "reference", 3), ("1_G_a_0", "SIG", "optimized", 1)],
            )

            runs = [
                (ara_max_dir, "ara max"),
                (gof_dir, "GOF"),
                (ara_min_dir, "ara min"),
                (lof_dir, "LOF"),
            ]

            # Act
            mut5_comparison.main(
                runs=runs,
                left_columns=["ara max", "GOF"],
                right_columns=["ara min", "LOF"],
                paired_run_a=(ara_max_dir, "ara max"),
                paired_run_b=(ara_min_dir, "ara min"),
                single_runs=[(gof_dir, "GOF"), (lof_dir, "LOF")],
                output_dir=output_dir,
                top_bottom_n=5,
                transpose=False,
            )

            # Assert: the top/bottom-N heatmaps are written with the "_top5" suffix.
            expected = [
                "compare_TFs_mut5_per_gene_top5.png",
                "compare_TFs_mut5_log_fold_change_top5.png",
            ]
            for name in expected:
                self.assertTrue(
                    os.path.exists(os.path.join(output_dir, name)),
                    f"Expected output {name} was not created",
                )


if __name__ == "__main__":
    unittest.main()
