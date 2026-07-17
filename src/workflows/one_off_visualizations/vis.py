from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps
from matplotlib.lines import Line2D

from analysis.motives.deepcis_visualize import (
    _get_padding_edge_positions,
    extract_plot_data,
)

OUTPUT_DIR = Path(__file__).parent / "plots"

SCAN_CSV = Path(
    "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/"
    "GOF_LOF/GOF/deepdive_simons_gene/deepcis_scan/"
    "deepcis_window_scan_deepdive_simons_gene_mut5.csv"
)

def random_bar_plot():
    values = [
        {"label": "Start Value", "value": 0.74},
        {"label": "Final Value", "value": 0.00004},
    ]

    fig, ax = plt.subplots()
    labels = [value["label"] for value in values]
    heights = [value["value"] for value in values]
    #select color from viridis color map
    colormap = colormaps["viridis"]
    colors = [colormap(0.1), colormap(0.1)]
    #make bars horizontal (left to right)
    ax.barh(labels, heights, color=colors, alpha=0.6)
    #set x-axis to log scale
    ax.set_xscale("log")
    #invert y-axis so first label is on top
    ax.invert_yaxis()
    # make layout wide and short
    fig.set_size_inches(6, 3)
    ax.set_xlabel("Average deepCRE prediction")
    ax.set_title("Start vs Final deepCRE prediction")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"start_vs_final.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

def optimized_track(
    scan_csv: Path = SCAN_CSV,
    tf_col: str = "LOBAS2_tnt",
    tss_position: float = 1000.0,
    tts_position: float = 2020.0,
    fmt: str = "png",
) -> None:
    """Plot only the optimized binding track for a single TF family.

    This is a stripped-down version of ``deepcis_visualize``: it shows just the
    optimized (mutated) sequence's binding-prediction curve for the requested TF
    family, keeping the TSS/TTS and padding markers but dropping the reference
    line, the difference line, and any peak highlighting.

    Args:
        scan_csv: Path to the deepCIS window-scan CSV file.
        tf_col: TF-family column to plot. Default: "LOBAS2_tnt".
        tss_position: X-coordinate (bp) for the TSS marker.
        tts_position: X-coordinate (bp) for the TTS marker.
        fmt: Output file format. Default: "png".
    """
    tf_label = tf_col[: -len("_tnt")] if tf_col.endswith("_tnt") else tf_col
    scan_data = pd.read_csv(scan_csv)
    gene = scan_data["gene"].unique()[0]

    # Reuse the tested extraction helper; keep only the optimized track.
    _, _, optimized_x, optimized_y, _, _, x_min, x_max = extract_plot_data(
        scan_data, tf_col
    )
    padding_edges = _get_padding_edge_positions(scan_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        optimized_x,
        optimized_y,
        color="blue",
        linewidth=2,
        label="Optimized",
        zorder=2,
    )

    # TSS/TTS markers (kept out of the automatic legend; added via a proxy).
    for marker_position in (tss_position, tts_position):
        ax.axvline(
            marker_position,
            color="#2f4f4f",
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
            zorder=1,
        )

    # Padding markers: 0, 1, or 2 vertical dotted lines.
    for padding_position in padding_edges:
        ax.axvline(
            padding_position,
            color="#7a7a7a",
            linestyle=":",
            linewidth=1.8,
            alpha=0.9,
            zorder=1,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Genomic Position (bp)")
    ax.set_ylabel("Binding Prediction Score")
    ax.set_title(f"{gene} - {tf_label} (optimized)")
    ax.grid(True, alpha=0.3)

    legend_handles = [
        Line2D([0], [0], color="blue", linewidth=2, label="Optimized"),
        Line2D(
            [0], [0], color="#2f4f4f", linestyle="--", linewidth=1.4, label="TSS/TTS"
        ),
    ]
    if padding_edges:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#7a7a7a",
                linestyle=":",
                linewidth=1.8,
                label="Area containing Padding",
            )
        )
    ax.legend(handles=legend_handles, loc="best")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{tf_label}_optimized.{fmt}", bbox_inches="tight", dpi=150
    )
    plt.close(fig)


if __name__ == "__main__":
    random_bar_plot()
    optimized_track()