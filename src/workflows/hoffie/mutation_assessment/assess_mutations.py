"""One-off check: do the lab-found mutations change deepCRE predictions?
Includes a bar plot comparing original vs lab_variant deepCRE predictions.

For each design we rebuild the 3020 bp deepCRE input (first 1500 bp from the
optimized zma_* T03 sequence, the rest from the T03 reference background),
apply the mutation the collaborators found, and predict both the original and
the mutated variant with the Zmay SSR model.

Only the poly-C insertion (zma_min / zma_max_nat) falls inside the deepCRE
window. The pHP4/pHP5 GATGC->T change is outside the window -- its context
(GATGCGGG, GGGTTTTACTGATG, ACTGATGCATACATGATG, CATATGCAGCATCTATTCATATG,
GATGGCATATGCAGC, TACGAGTTTAAGATGGATGGAAATATCGATCTAGGATAGGTATACATGTTGATG) is
absent from the reference and every T03 record -- so it can't affect the
prediction and is skipped.
"""

import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyfaidx import Fasta
from tensorflow.keras.models import load_model  # type: ignore

from evolution.sequences import one_hot_encode

MODEL_PATH = (
    "/home/gernot/Code/PhD_Code/Evolution/models/"
    "Zmay_S0X0.75dP7K25g_NC_050105.1_ssr_train_models_250705_211854.h5"
)
SEQ_DIR = "src/workflows/hoffie/data/ubi/sequences"
BACKGROUND_PATH = f"{SEQ_DIR}/T03_full_ref.fasta"
OUT_CSV = "src/workflows/hoffie/mutation_assessment/prediction_results.csv"
OUTPUT_PATH = "src/workflows/hoffie/mutation_assessment/assess_mutations_sequences.fa"

# condition -> (pareto FASTA, 5' flank of the poly-C run, or None if the lab
# mutation lies outside the deepCRE window).
CONDITIONS = {
    "zma_min": (
        "zma_min_260122_155305_101018_pareto_fronts_no_restriction_sites_no_restriction_sites.fa",
        "TACGCCGCTCGTCCT",
    ),
    "zma_max_nat": (
        "zma_max_nat_260126_105908_714145_pareto_fronts_no_restriction_sites_no_restriction_sites.fa",
        "TACGCCGCTCGGCCT",
    ),
    "zma_min_nat": (
        "zma_min_nat_260122_155305_079827_pareto_fronts_no_restriction_sites_no_restriction_sites.fa",
        "TACGCCGCTCGACCT",
    ),
    "zma_max": (
        "zma_max_260126_105908_740831_pareto_fronts_no_restriction_sites_no_restriction_sites.fa",
        None,
    ),
}
PLOT_OUTPUT_DIR = "src/workflows/hoffie/mutation_assessment"


def get_t03_seq(fasta_file: str) -> str:
    """Return the T03 _mutations_045 sequence (the one sent to the lab)."""
    fasta = Fasta(f"{SEQ_DIR}/{fasta_file}")
    header = next(
        k for k in fasta.keys() if "_T03_" in k and k.endswith("mutations_045")
    )
    print(header)
    return str(fasta[header]).upper()


def insert_poly_c(promoter: str, flank: str) -> str:
    """Add one C to the poly-C run after `flank`, then trim back to 1500 bp.

    Trimming drops the last promoter base before the 20 bp N spacer so the
    reconstructed deepCRE input stays exactly 3020 bp long.
    """
    run_start = promoter.index(flank) + len(flank)
    return (promoter[:run_start] + "C" + promoter[run_start:])[:1500]


def save_generated_sequences(names: List[str], variants: List[str], seqs: List[str]) -> None:
    """Save the generated sequences to a FASTA file for record-keeping."""
    with open(OUTPUT_PATH, "w") as f:
        for name, variant, seq in zip(names, variants, seqs):
            header = f">{name}_{variant}"
            f.write(f"{header}\n{seq}\n")


def main() -> None:
    background = str(Fasta(BACKGROUND_PATH)["ref"]).upper()  # 3020 bp

    names, variants, seqs = [], [], []
    for name, (fasta_file, flank) in CONDITIONS.items():
        promoter = get_t03_seq(fasta_file)[:1500]
        names.append(name)
        variants.append("original")
        seqs.append(promoter + background[1500:])
        if flank is not None:
            names.append(name)
            variants.append("lab_variant")
            seqs.append(insert_poly_c(promoter, flank) + background[1500:])
    
    save_generated_sequences(names, variants, seqs)

    model = load_model(MODEL_PATH)
    encoded = np.array([one_hot_encode(seq) for seq in seqs])
    predictions = model.predict(encoded).flatten()

    df = pd.DataFrame(
        {
            "condition": names,
            "variant": variants,
            "length": [len(seq) for seq in seqs],
            "prediction": predictions,
        }
    )
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False))

    for name, (_, flank) in CONDITIONS.items():
        if flank is None:
            continue
        sub = df[df["condition"] == name].set_index("variant")["prediction"]
        print(f"{name}: lab_variant - original = {sub['lab_variant'] - sub['original']:+.4f}")




def load_prediction_results(csv_path: str) -> pd.DataFrame:
    """Load deepCRE prediction results from CSV.

    Args:
        csv_path: Path to the predictions CSV (condition, variant, length, prediction).

    Returns:
        DataFrame with columns: condition, variant, length, prediction.
    """
    return pd.read_csv(csv_path)


def save_figure(fig: plt.Figure, filename: str, output_dir: str, fmt: str = "png") -> None:
    """Save figure to output_dir/filename.fmt.

    Args:
        fig: Matplotlib figure to save.
        filename: Base filename without extension.
        output_dir: Directory to save the figure in.
        fmt: File format (default: png).
    """
    path = Path(output_dir) / f"{filename}.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=150)


def plot_prediction_comparison(
    df: pd.DataFrame,
    output_dir: str,
    fmt: str = "png",
) -> None:
    """Bar plot comparing original vs lab_variant deepCRE predictions per condition.

    Shows that the lab-introduced mutations do not meaningfully change deepCRE
    predictions relative to the originally sent sequences.

    Args:
        df: DataFrame with columns condition, variant, prediction.
        output_dir: Directory to write the figure.
        fmt: Output file format.
    """
    sns.set_theme(style="whitegrid")

    # Build a (condition, variant) -> prediction lookup to avoid chained indexing
    predictions: dict[tuple[str, str], float] = {
        (str(row["condition"]), str(row["variant"])): float(row["prediction"])
        for _, row in df.iterrows()
    }

    conditions = list(df["condition"].unique())
    x_positions = np.arange(len(conditions))
    bar_width = 0.35

    palette = {"original": "#4C72B0", "lab_variant": "#DD8452"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for offset, variant in zip([-bar_width / 2, bar_width / 2], ["original", "lab_variant"]):
        values = [predictions.get((condition, variant), float("nan")) for condition in conditions]
        bars = ax.bar(
            x_positions + offset,
            values,
            width=bar_width,
            label=variant.replace("_", " ").capitalize(),
            color=palette[variant],
            edgecolor="white",
        )
        # Annotate each bar with its value
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Annotate delta (lab_variant - original) above each pair
    for x_pos, condition in zip(x_positions, conditions):
        original_val = predictions.get((condition, "original"))
        lab_val = predictions.get((condition, "lab_variant"))
        if original_val is None or lab_val is None:
            continue
        delta = lab_val - original_val
        max_val = max(original_val, lab_val)
        ax.text(
            x_pos,
            max_val + 0.05,
            f"Δ={delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="dimgray",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_xlabel("Condition")
    ax.set_ylabel("deepCRE prediction score")
    ax.set_title(
        "Lab variants do not meaningfully alter deepCRE predictions",
        fontsize=11,
    )
    ax.set_ylim(0, 1.15)
    ax.legend(title="Variant")

    # Note for condition without lab variant (use the pre-built lookup)
    conditions_without_lab = [
        c for c in conditions if (c, "lab_variant") not in predictions
    ]
    if conditions_without_lab:
        note = f"* {', '.join(conditions_without_lab)}: lab mutation outside deepCRE window — no variant tested"
        fig.text(0.5, -0.04, note, ha="center", fontsize=8, color="dimgray", style="italic")

    sns.despine()
    save_figure(fig, "prediction_comparison", output_dir, fmt)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    if "--plot-only" in sys.argv:
        prediction_df = load_prediction_results(OUT_CSV)
        plot_prediction_comparison(prediction_df, PLOT_OUTPUT_DIR)
    else:
        main()
        prediction_df = load_prediction_results(OUT_CSV)
        plot_prediction_comparison(prediction_df, PLOT_OUTPUT_DIR)
