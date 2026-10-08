"""Integration smoke test for the constraint_comparison thin script."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from workflows.evo_alg_pooled_plots.tf_comparison import constraint_comparison


def _write_full_run(
    run_dir: str,
    summary_df: pd.DataFrame,
    peaks: List[Tuple[str, str, str, int]],
) -> None:
    """Create a run dir with stats JSON, peak summary, and annotated peaks.

    Uses the older pipeline's ``"max_mutated"`` signal-type label, which is what
    the GOF/LOF constraint runs carry.
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


def _peaks(mutated_sig_count: int) -> List[Tuple[str, str, str, int]]:
    """Annotated peaks over 8 genes shared by every run.

    ``mutated_sig_count`` sets how many SIG peaks the optimized sequence has
    (the reference always has 1), so different constraints get different effects.
    """
    peaks: List[Tuple[str, str, str, int]] = []
    for index in range(8):
        gene = f"{index}_GENE{index}_loc_{index}"
        peaks += [
            (gene, "SIG", "reference", 1),
            (gene, "SIG", "max_mutated", mutated_sig_count),
            (gene, "FLAT", "reference", 2),
            (gene, "FLAT", "max_mutated", 2),
        ]
    return peaks


class CompareConstraintsIntegrationTest(unittest.TestCase):
    """End-to-end run of the orchestration against synthetic runs in a temp dir."""

    def _build_runs(self, root: str) -> List[Tuple[str, str]]:
        """Write four synthetic constraint runs under ``root`` and return them."""
        runs = []
        for label, mutated_sig_count, diff in [
            ("unconstrained", 4, 3),
            ("natural", 3, 2),
            ("CRISPR", 2, 1),
            ("CRISPR 3PAM", 1, 0),
        ]:
            run_dir = os.path.join(root, label.replace(" ", "_"))
            _write_full_run(
                run_dir, _summary(["SIG", "FLAT"], [diff, 0]), _peaks(mutated_sig_count)
            )
            runs.append((run_dir, label))
        return runs

    def test_writes_heatmaps_intra_and_all_pairwise_csvs(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._build_runs(tmp)
            output_dir = os.path.join(tmp, "out")

            # Act
            constraint_comparison.compare_constraints(
                runs, "GOF", output_dir=output_dir
            )

            # Assert: both heatmaps, one intra CSV per run, and all 6 pair CSVs.
            expected = [
                "GOF_constraints_per_gene.png",
                "GOF_constraints_log_fold_change.png",
                "GOF_intra_unconstrained.csv",
                "GOF_intra_natural.csv",
                "GOF_intra_CRISPR.csv",
                "GOF_intra_CRISPR_3PAM.csv",
                "GOF_paired_unconstrained_vs_natural.csv",
                "GOF_paired_unconstrained_vs_CRISPR.csv",
                "GOF_paired_unconstrained_vs_CRISPR_3PAM.csv",
                "GOF_paired_natural_vs_CRISPR.csv",
                "GOF_paired_natural_vs_CRISPR_3PAM.csv",
                "GOF_paired_CRISPR_vs_CRISPR_3PAM.csv",
            ]
            for name in expected:
                self.assertTrue(
                    os.path.exists(os.path.join(output_dir, name)),
                    f"Expected output {name} was not created",
                )

    def test_intra_run_csv_reports_the_expected_median_diff(self) -> None:
        # Arrange: the "unconstrained" run has 4 optimized vs 1 reference SIG peak
        # per gene, so its per-gene SIG diff is +3 in all 8 genes.
        with tempfile.TemporaryDirectory() as tmp:
            runs = self._build_runs(tmp)
            output_dir = os.path.join(tmp, "out")
            os.makedirs(output_dir)

            # Act
            cell_stars = constraint_comparison.write_intra_run_significance(
                runs, "GOF", Path(output_dir)
            )
            intra = pd.read_csv(os.path.join(output_dir, "GOF_intra_unconstrained.csv"))

            # Assert
            sig_row = intra[intra["tf"] == "SIG"].iloc[0]
            self.assertEqual(sig_row["median_diff"], 3.0)
            self.assertEqual(sig_row["n_nonzero"], 8)
            self.assertEqual(
                set(cell_stars),
                {"unconstrained", "natural", "CRISPR", "CRISPR 3PAM"},
            )
            # A consistent +3 shift across 8 genes is significant; FLAT never moves.
            self.assertNotEqual(cell_stars["unconstrained"]["SIG"], "")
            self.assertEqual(cell_stars["unconstrained"]["FLAT"], "")


if __name__ == "__main__":
    unittest.main()
