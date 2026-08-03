"""Incremental Pareto-front scatter for a single example gene.

Each Pareto-front JSON file is a list of ``[sequence, deepcre_prediction,
mutation_count]`` triples. One figure is produced: the starting point (the
unmutated natural sequence, 0 mutations) plus every saved Pareto front,
coloured by a single sequential gradient so later generations are darker. This
shows how the front advances over the optimization.

Front files (progression order); the file without a generation tag is final:

    pareto_front_gen_00005.json
    pareto_front_gen_00020.json
    pareto_front_gen_00100.json
    pareto_front_gen_01000.json
    pareto_front.json            (final)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle

from workflows.paper_plots.style import figure_size_inches, publication_style

# --- Hardcoded configuration (one-off script) --------------------------------

RUN_DIR = Path(
    "/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/GOF_LOF/GOF/"
    "GOF_single_natural_260311_122228_299093/"
    "3_AT3G60640_gene:22415750-22417548_260312_000750_357934/saved_populations"
)

OUTPUT_DIR = Path(__file__).parent / "plots"
OUTPUT_FORMAT = "svg"

# Sequential colormap for the generation gradient (light = early, dark = late).
GRADIENT_CMAP = "magma"
# Colour of the distinct starting-point marker.
START_COLOR = "black"

# Pareto-front file names: ``pareto_front_gen_<N>.json`` per generation, plus a
# final ``pareto_front.json`` with no generation tag.
FRONT_GLOB = "pareto_front*.json"
_GEN_PATTERN = re.compile(r"pareto_front_gen_(\d+)\.json$")

# Fractional padding added around the data when computing axis limits.
AXIS_MARGIN = 0.05

# Colormap and shade used for the small final-front illustration figure.
MINI_FRONT_CMAP = "Purples"
MINI_FRONT_COLOR_SHADE = 0.85
# Size of the small final-front illustration figure, in millimetres.
MINI_FRONT_SIZE_MM = (45.0, 36.0)


# --- Calculation -------------------------------------------------------------


def load_mutation_prediction(file_path: Path) -> tuple[list[float], list[float]]:
    """Load mutation counts and deepCRE predictions from a front JSON file.

    Args:
        file_path: Path to a JSON file holding a list of
            ``[sequence, deepcre_prediction, mutation_count]`` triples.

    Returns:
        A tuple ``(mutation_counts, predictions)`` of parallel lists.
    """
    with open(file_path) as handle:
        entries = json.load(handle)
    mutation_counts = [entry[2] for entry in entries]
    predictions = [entry[1] for entry in entries]
    return mutation_counts, predictions


def sample_gradient_colors(count: int) -> list[tuple[float, float, float, float]]:
    """Sample ``count`` evenly spaced RGBA colors from the generation gradient.

    The very brightest end of the colormap is skipped for on-white contrast.

    Args:
        count: Number of colors to sample (one per Pareto front).

    Returns:
        A list of RGBA tuples, earliest generation first.
    """
    colormap = colormaps[GRADIENT_CMAP]
    if count <= 0:
        return []
    if count == 1:
        return [colormap(0.1)]
    return [colormap(0.8 - 0.8 * index / (count - 1)) for index in range(count)]


def order_front_files(file_names: list[str]) -> list[tuple[str, str]]:
    """Order Pareto-front file names by generation and build legend labels.

    A ``pareto_front_gen_<N>.json`` file is placed by its generation ``N``; the
    untagged ``pareto_front.json`` is treated as the final front and sorted
    last.

    Args:
        file_names: Pareto-front file names (no directory), e.g. from a glob.

    Returns:
        A list of ``(file_name, label)`` tuples in progression order (earliest
        generation first, final front last).
    """
    final_sort_key = float("inf")
    tagged = []
    for name in file_names:
        match = _GEN_PATTERN.search(name)
        if match:
            generation = int(match.group(1))
            tagged.append((generation, name, f"Gen {generation}"))
        elif name == "pareto_front.json":
            tagged.append((final_sort_key, name, "Final"))
    tagged.sort(key=lambda item: item[0])
    return [(name, label) for _, name, label in tagged]


def compute_shared_limits(
    all_mutation_counts: list[float],
    all_predictions: list[float],
    margin: float = AXIS_MARGIN,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute x/y axis limits with a fractional margin.

    Args:
        all_mutation_counts: Pooled mutation counts across every front.
        all_predictions: Pooled deepCRE predictions across every front.
        margin: Fraction of the data range to pad on each side.

    Returns:
        A tuple ``(xlim, ylim)`` where each entry is a ``(low, high)`` pair.

    Raises:
        ValueError: If either input list is empty.
    """
    if not all_mutation_counts or not all_predictions:
        raise ValueError("Cannot compute limits from empty data.")

    def _padded(values: list[float]) -> tuple[float, float]:
        low, high = min(values), max(values)
        span = high - low
        pad = span * margin if span > 0 else abs(high) * margin or 1.0
        return low - pad, high + pad

    return _padded(all_mutation_counts), _padded(all_predictions)


def extract_zero_mutation_points(
    mutation_counts: list[float], predictions: list[float]
) -> tuple[list[float], list[float]]:
    """Select the points with a mutation count of zero (the natural sequence).

    Args:
        mutation_counts: Mutation count per sequence.
        predictions: deepCRE prediction per sequence (parallel to counts).

    Returns:
        A tuple ``(zero_counts, zero_predictions)`` keeping only entries whose
        mutation count is zero.
    """
    zero_counts = []
    zero_predictions = []
    for count, prediction in zip(mutation_counts, predictions):
        if count == 0:
            zero_counts.append(count)
            zero_predictions.append(prediction)
    return zero_counts, zero_predictions


# --- Plotting ----------------------------------------------------------------


def _build_fixed_legend_handles(
    all_front_styles: list[tuple[str, tuple]],
) -> list[Line2D]:
    """Build proxy legend handles for every front plus the start marker.

    Using proxy artists keeps the legend identical across all cumulative frames
    (every front listed from the first frame on), independent of which fronts a
    given frame actually draws.

    Args:
        all_front_styles: ``(label, color)`` for every front, in progression
            order.

    Returns:
        Proxy ``Line2D`` handles: one dot per front, then the start star.
    """
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=color, label=label)
        for label, color in all_front_styles
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            color=START_COLOR,
            markersize=11,
            label="Start (natural seq)",
        )
    )
    return handles


def plot_fronts_up_to(
    front_data: list[tuple[str, list[float], list[float], tuple]],
    start_counts: list[float],
    start_predictions: list[float],
    all_front_styles: list[tuple[str, tuple]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    dim_non_final: bool = False,
) -> Figure:
    """Scatter the start point plus the given (prefix of) Pareto fronts.

    Each front keeps its assigned gradient colour, so a front looks identical
    across the cumulative frames. The legend lists every front (from
    ``all_front_styles``) on every frame, so it is fully present from frame 0.
    Axis limits are passed in (shared across frames) so every frame is directly
    comparable.

    Args:
        front_data: List of ``(label, mutation_counts, predictions, color)``
            tuples to draw, in progression order. Pass a prefix for an early
            cumulative frame; pass an empty list for the start-only frame.
        start_counts: Mutation counts of the starting point(s) (0 mutations).
        start_predictions: deepCRE predictions of the starting point(s).
        all_front_styles: ``(label, color)`` for every front (all frames),
            driving the fixed legend.
        xlim: x-axis ``(low, high)`` limits.
        ylim: y-axis ``(low, high)`` limits.
        dim_non_final: If True, fade every front except the final one to a low
            opacity so the final front and start point stand out (used for the
            last frame).

    Returns:
        The created matplotlib figure.
    """
    final_label = all_front_styles[-1][0] if all_front_styles else None

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, mutation_counts, predictions, color in front_data:
        is_final = label == final_label
        alpha = 1 if (is_final or not dim_non_final) else 0.3
        ax.scatter(
            mutation_counts,
            predictions,
            color=color,
            alpha=alpha,
            edgecolors="none",
        )

    ax.scatter(
        start_counts,
        start_predictions,
        color=START_COLOR,
        marker=MarkerStyle("*"),
        s=120,
        edgecolors="none",
        zorder=5,
    )

    ax.set_xlabel("Mutation count")
    ax.set_ylabel("deepCRE prediction")
    ax.set_title("Pareto front over generations")
    ax.legend(
        handles=_build_fixed_legend_handles(all_front_styles),
        title="Generation",
        loc="lower right",
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.tight_layout()
    return fig


def plot_final_front_mini(
    mutation_counts: list[float], predictions: list[float]
) -> Figure:
    """Draw a minimal scatter of the final Pareto front for use as an inset.

    No legend or title is drawn, but axis labels/ticks are kept so the inset
    remains readable on its own. The figure is small by design so it can be
    embedded as an illustrative inset inside a larger composed figure. Uses the
    shared publication stylesheet for fonts and line weights.

    Args:
        mutation_counts: Mutation counts of the final Pareto front.
        predictions: deepCRE predictions of the final Pareto front (parallel
            to ``mutation_counts``).

    Returns:
        The created matplotlib figure.
    """
    color = colormaps[MINI_FRONT_CMAP](MINI_FRONT_COLOR_SHADE)
    with publication_style():
        fig, ax = plt.subplots(figsize=figure_size_inches(*MINI_FRONT_SIZE_MM))
        ax.scatter(mutation_counts, predictions, color=color, edgecolors="none")
        ax.set_xlabel("Mutation count")
        ax.set_ylabel("deepCRE prediction")
        fig.tight_layout(pad=0.1)
    return fig


def save_figure(fig: Figure, filename: str) -> None:
    """Save a figure to ``OUTPUT_DIR/filename.OUTPUT_FORMAT``.

    Args:
        fig: The matplotlib figure to save.
        filename: File name without extension.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{filename}.{OUTPUT_FORMAT}"
    fig.savefig(str(path), bbox_inches="tight", dpi=150)


def _frame_slug(index: int, label: str) -> str:
    """Build a zero-padded, filesystem-safe frame file name (no extension)."""
    clean_label = label.lower().replace(" ", "_")
    return f"step_{index:02d}_{clean_label}"


def main() -> None:
    """Build and save the cumulative series of Pareto-front figures.

    Frame 0 shows the start point only; each subsequent frame adds the next
    Pareto front, so the series builds up to the full picture. All frames share
    one axis range.
    """
    sns.set_theme(style="whitegrid")

    file_names = [path.name for path in RUN_DIR.glob(FRONT_GLOB)]
    ordered_fronts = order_front_files(file_names)
    colors = sample_gradient_colors(len(ordered_fronts))

    colored_fronts = []
    for (file_name, label), color in zip(ordered_fronts, colors):
        mutation_counts, predictions = load_mutation_prediction(RUN_DIR / file_name)
        colored_fronts.append((label, mutation_counts, predictions, color))

    all_counts = [count for _, counts, _, _ in colored_fronts for count in counts]
    all_predictions = [pred for _, _, preds, _ in colored_fronts for pred in preds]
    xlim, ylim = compute_shared_limits(all_counts, all_predictions)

    # Start point: the 0-mutation entry (identical across fronts); take it from
    # the earliest front.
    _, first_counts, first_predictions, _ = colored_fronts[0]
    start_counts, start_predictions = extract_zero_mutation_points(
        first_counts, first_predictions
    )

    all_front_styles = [(label, color) for label, _, _, color in colored_fronts]

    # Frame 0: start only. Frame k: start + first k fronts. On the final frame,
    # fade everything except the start point and the final Pareto front.
    last_frame_index = len(colored_fronts)
    for frame_index in range(last_frame_index + 1):
        fig = plot_fronts_up_to(
            colored_fronts[:frame_index],
            start_counts,
            start_predictions,
            all_front_styles,
            xlim,
            ylim,
            dim_non_final=(frame_index == last_frame_index),
        )
        if frame_index == 0:
            slug = _frame_slug(0, "start")
        else:
            frame_label = colored_fronts[frame_index - 1][0]
            slug = _frame_slug(frame_index, frame_label)
        save_figure(fig, slug)
        plt.close(fig)


def load_final_front() -> tuple[list[float], list[float]]:
    """Load the mutation counts and predictions of this gene's final Pareto front.

    Used by :mod:`workflows.paper_plots.fig1_miniatures` to build a small inset
    illustration from the same data source as the full incremental series above.

    Returns:
        A tuple ``(mutation_counts, predictions)`` for the final front.
    """
    file_names = [path.name for path in RUN_DIR.glob(FRONT_GLOB)]
    final_file_name, _ = order_front_files(file_names)[-1]
    return load_mutation_prediction(RUN_DIR / final_file_name)


if __name__ == "__main__":
    main()

