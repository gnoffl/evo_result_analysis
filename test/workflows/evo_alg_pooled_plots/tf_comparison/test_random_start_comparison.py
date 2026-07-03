"""Integration smoke test for the random_start_comparison thin script."""

import json
import os
import tempfile
import unittest
from typing import List, Tuple

import pandas as pd

from workflows.evo_alg_pooled_plots.tf_comparison import random_start_comparison


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


class RandomStartComparisonIntegrationTest(unittest.TestCase):
    """End-to-end run of the orchestration against a synthetic run in a temp dir."""

    def test_creates_expected_csv_and_png_outputs(self) -> None:
        # Arrange: one synthetic run plus an output dir under one temp root.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "random_max")
            output_dir = os.path.join(tmp, "out")

            # Random-start sequence naming: first two fields are always
            # "random_sequence"; each sequence is a distinct replicate only when
            # split on 3 fields (which the script's CORE_ID_FIELDS default handles).
            peaks: List[Tuple[str, str, str, int]] = []
            for i in range(8):
                sequence = f"random_sequence_{i:03d}_260224_ts_{i}"
                peaks += [(sequence, "SIG", "reference", 1), (sequence, "SIG", "max_mutated", 3)]
                peaks += [(sequence, "FLAT", "reference", 2), (sequence, "FLAT", "max_mutated", 2)]
            _write_full_run(run_dir, _summary(["SIG", "FLAT"], [2, 0]), peaks)

            # Act
            random_start_comparison.main(
                run_dir=run_dir,
                label="random max",
                output_dir=output_dir,
            )

            # Assert: the significance CSV and both heatmap PNGs were written.
            expected = [
                "random_start_comparison_significance.csv",
                "random_start_comparison_per_gene.png",
                "random_start_comparison_log_fold_change.png",
            ]
            for name in expected:
                path = os.path.join(output_dir, name)
                self.assertTrue(os.path.exists(path), f"Expected output {name} was not created")
                self.assertGreater(os.path.getsize(path), 0, f"Output {name} is empty")

            # Regression guard: the 8 sequences must be kept as 8 replicates (not
            # collapsed to a single "random_sequence" gene, which would force p=1).
            significance = pd.read_csv(
                os.path.join(output_dir, "random_start_comparison_significance.csv")
            )
            self.assertTrue((significance["n_genes"] == 8).all())
            self.assertLess(significance.set_index("tf").loc["SIG", "p_intra"], 0.05)


if __name__ == "__main__":
    unittest.main()
