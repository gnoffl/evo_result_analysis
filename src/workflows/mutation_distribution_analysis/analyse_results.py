"""Compute average deepCIS scores after random mutations, and plot them against
before/after optimization, grouped by species."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ARA_RANDOM_MUTATIONS_DIR = "src/workflows/mutation_distribution_analysis/mutated_sequences/ara_random_mutated_predictions.csv"
ZEA_RANDOM_MUTATIONS_DIR = "src/workflows/mutation_distribution_analysis/mutated_sequences/zea_random_mutated_predictions.csv"
AVERAGE_PREDICTIONS_PATH = "src/workflows/mutation_distribution_analysis/visualizations/average_predictions_after_random_mutations.txt"


def average_predications_after_random_mutations():
    ara_random_mutations_df = pd.read_csv(ARA_RANDOM_MUTATIONS_DIR)
    zea_random_mutations_df = pd.read_csv(ZEA_RANDOM_MUTATIONS_DIR)

    print("Average predictions after random mutations:")
    avg_ara = ara_random_mutations_df['prediction'].mean()
    avg_zea = zea_random_mutations_df['prediction'].mean()
    print(f"Arabidopsis: {avg_ara}")
    print(f"Maize: {avg_zea}")
    os.makedirs(os.path.dirname(AVERAGE_PREDICTIONS_PATH), exist_ok=True)
    with open(AVERAGE_PREDICTIONS_PATH, "w") as f:
        f.write("Average predictions after random mutations:\n")
        f.write(f"Arabidopsis: {avg_ara}\n")
        f.write(f"Maize: {avg_zea}\n")


def load_random_mutation_averages(filepath: str) -> dict[str, float]:
    """Parse average deepCIS scores after random mutations from text file.

    Args:
        filepath: Path to the average_predictions_after_random_mutations.txt file.

    Returns:
        Dictionary mapping species name to average prediction score.
    """
    averages = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Arabidopsis:"):
                averages["Arabidopsis"] = float(line.split(":")[1].strip())
            elif line.startswith("Maize:"):
                averages["Maize"] = float(line.split(":")[1].strip())
    return averages


def load_evolution_summary(filepath: str, run_key: str) -> dict[str, float]:
    """Parse start and final fitness means from an evolution run summary file.

    Reads only the block matching run_key (the first occurrence).

    Args:
        filepath: Path to a summary_*.txt file.
        run_key: The run name to match (e.g. 'ara_msr_max_single').

    Returns:
        Dictionary with keys 'start_fitness_mean' and 'final_fitness_mean'.
    """
    result: dict[str, float] = {}
    inside_block = False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line == f"Summary for {run_key}:":
                inside_block = True
                continue
            if inside_block:
                if line.startswith("Summary for"):
                    break
                if line.startswith("start_fitness_mean:"):
                    result["start_fitness_mean"] = float(line.split(":")[1].strip())
                elif line.startswith("final_fitness_mean:"):
                    result["final_fitness_mean"] = float(line.split(":")[1].strip())
    return result


def build_plot_dataframe(
    ara_summary_path: str,
    zea_summary_path: str,
    random_mutations_path: str,
) -> pd.DataFrame:
    """Assemble a tidy DataFrame with one row per (species, condition) combination.

    Args:
        ara_summary_path: Path to Arabidopsis evolution summary file.
        zea_summary_path: Path to Maize evolution summary file.
        random_mutations_path: Path to random mutations averages file.

    Returns:
        DataFrame with columns: species, condition, score.
    """
    random_averages = load_random_mutation_averages(random_mutations_path)

    ara = load_evolution_summary(ara_summary_path, "ara_msr_max_single")
    zea = load_evolution_summary(zea_summary_path, "zea_msr_max_single")

    rows = [
        {"species": "Arabidopsis", "condition": "Before optimization", "score": ara["start_fitness_mean"]},
        {"species": "Arabidopsis", "condition": "After random mutations", "score": random_averages["Arabidopsis"]},
        {"species": "Arabidopsis", "condition": "After optimization", "score": ara["final_fitness_mean"]},
        {"species": "Maize", "condition": "Before optimization", "score": zea["start_fitness_mean"]},
        {"species": "Maize", "condition": "After random mutations", "score": random_averages["Maize"]},
        {"species": "Maize", "condition": "After optimization", "score": zea["final_fitness_mean"]},
    ]
    return pd.DataFrame(rows)


def plot_optimization_vs_random(
    data: pd.DataFrame,
    output_dir: str | None = None,
    fmt: str = "png",
    ax: plt.Axes | None = None,
) -> None:
    """Plot grouped bar chart of deepCIS scores across conditions and species.

    Args:
        data: Tidy DataFrame with columns: species, condition, score.
        output_dir: Directory to save the figure. Required when ``ax`` is None
            (standalone mode); ignored when ``ax`` is given.
        fmt: Output file format (e.g. 'png', 'svg', 'pdf').
        ax: Axes to draw onto. When None, a standalone figure is created and
            saved (unchanged behaviour); when given, the plot is drawn onto
            ``ax`` and nothing is saved.
    """
    condition_order = ["Before optimization", "After random mutations", "After optimization"]

    own_figure = ax is None
    if own_figure:
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(7, 5))

    sns.barplot(
        data=data,
        x="species",
        y="score",
        hue="condition",
        hue_order=condition_order,
        ax=ax,
    )

    ax.set_xlabel("Species")
    ax.set_ylabel("Average deepCIS prediction score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Condition", bbox_to_anchor=(1.01, 1), loc="upper left")

    if own_figure:
        fig.tight_layout()
        save_figure(fig, "optimization_vs_random_mutations", output_dir, fmt)
        plt.close(fig)


def save_figure(fig: plt.Figure, filename: str, output_dir: str, fmt: str = "png") -> None:
    """Save figure to output_dir/filename.fmt.

    Args:
        fig: Matplotlib figure to save.
        filename: Base filename without extension.
        output_dir: Directory to save the figure.
        fmt: File format extension.
    """
    path = Path(output_dir) / f"{filename}.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved figure to {path}")


def main():
    average_predications_after_random_mutations()

    base = Path(__file__).parent
    ara_summary = (
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/"
        "paper_runs/single_mutation/ara_msr_max_single/summary_ara_msr_max_single.txt"
    )
    zea_summary = (
        "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/"
        "paper_runs/single_mutation/zea_msr_max_single/summary_zea_msr_max_single.txt"
    )
    output_dir = str(base / "visualizations")
    Path(output_dir).mkdir(exist_ok=True)

    data = build_plot_dataframe(ara_summary, zea_summary, AVERAGE_PREDICTIONS_PATH)
    plot_optimization_vs_random(data, output_dir, fmt="png")


if __name__ == "__main__":
    main()