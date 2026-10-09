"""Plotting for the runtime benchmark of the two sequence-optimizing algorithms.

This module is the visualization half of the runtime-benchmark toolbox: it turns
the tidy runtime frame of ``runtime_benchmark_calc.py`` into sweep boxplots and
power-law scaling plots. It contains **no statistics and no IO decisions** — the
frame is filtered and the fits are computed by the caller, so the same two
functions serve a CPU-only figure, a CPU-versus-GPU figure and a panel of a
composed paper figure.

Both functions follow the ax-injection convention of this package: with ``ax=None``
they create, style, optionally save and close their own figure; with ``ax`` given
they draw onto it and touch neither the theme, the layout nor the disk.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

from workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark_calc import (  # noqa: E402
    RUNTIME_COLUMN,
    ScalingFit,
)

# Colours for the two mutation regimes and the two devices. Kept distinct from
# each other so a figure that shows both cannot be misread.
REGIME_PALETTE: Dict[str, str] = {
    "unconstrained": "#4878a8",
    "natural": "#d1834b",
}
REGIME_ORDER: List[str] = ["unconstrained", "natural"]

DEVICE_PALETTE: Dict[str, str] = {
    "CPU": "#5f5f5f",
    "GPU": "#68a357",
}
DEVICE_ORDER: List[str] = ["CPU", "GPU"]

# Axis labels for the swept config fields, so a caller only passes the column.
SWEPT_PARAMETER_LABELS: Dict[str, str] = {
    "population_size": "Population size",
    "number_of_generations": "Number of generations",
    "max_number_mutations": "Mutation cap",
}

RUNTIME_AXIS_LABEL = "Optimization wall time (s)"

_PALETTES: Dict[str, Dict[str, str]] = {
    "regime": REGIME_PALETTE,
    "device": DEVICE_PALETTE,
}
_ORDERS: Dict[str, List[str]] = {"regime": REGIME_ORDER, "device": DEVICE_ORDER}

# Geometry of the per-sequence dot overlay: the 30 sequences of a cell are drawn
# on top of the box, so the reader sees the replicate spread the box summarizes.
_POINT_SIZE = 2.0
_POINT_ALPHA = 0.5
_POINT_JITTER = 0.12
_POINT_COLOUR = "black"

# Outline styling. Most cells of this benchmark have an interquartile range of a few
# seconds against a median of hundreds, so on a logarithmic axis their box collapses
# to a hairline and its fill colour disappears. Painting the outline — box edge,
# median, whiskers and caps — in the box's own hue keeps the collapsed box readable,
# and the fill is lightened so the outline stands out where a box is not collapsed.
_BOX_FILL_ALPHA = 0.35
_BOX_LINE_WIDTH = 1.6


def _resolve_hue(
    hue_column: str, data: pd.DataFrame
) -> Tuple[Optional[Dict[str, str]], Optional[List[str]]]:
    """Return the palette and category order to use for a hue column.

    Args:
        hue_column: Column to colour by; ``"regime"`` and ``"device"`` have fixed
            palettes and orders.
        data: Frame being plotted, used to restrict the order to present values.

    Returns:
        Tuple of ``(palette, hue_order)``; ``(None, None)`` for an unknown column,
        which lets seaborn choose.
    """
    if hue_column not in _PALETTES:
        return None, None
    present = set(data[hue_column].unique())
    hue_order = [value for value in _ORDERS[hue_column] if value in present]
    return _PALETTES[hue_column], hue_order


def color_box_outlines(ax: plt.Axes) -> None:
    """Repaint every box's outline, median, whiskers and caps in its own hue.

    Seaborn fills each box with its hue colour but draws all line artists in one
    dark grey. A box whose interquartile range is tiny collapses on a logarithmic
    axis to a line of that grey, so its hue — and with it the regime or device it
    stands for — becomes invisible. This gives every line artist belonging to a box
    that box's colour and lightens the fill, so a collapsed box still reads as a
    coloured mark.

    Artists are matched to their box by horizontal centre rather than by seaborn's
    drawing order: a box's median spans its full width, its caps a fraction of it
    and its whisker is vertical at its centre, so all three share the box's centre.
    A line is assigned to the nearest box centre within half a box width, which
    leaves the jittered replicate dots and any other artist untouched.

    Args:
        ax: Axes holding a drawn seaborn boxplot. Modified in place.
    """
    box_centers: List[float] = []
    box_colors: List[tuple] = []
    box_half_widths: List[float] = []
    for patch in ax.patches:
        red, green, blue = patch.get_facecolor()[:3]
        patch.set_edgecolor((red, green, blue, 1.0))
        patch.set_facecolor((red, green, blue, _BOX_FILL_ALPHA))
        patch.set_linewidth(_BOX_LINE_WIDTH)
        if not isinstance(patch, PathPatch):
            # A legend swatch: restyled like a box, but it sits on no axis
            # position, so no line artist belongs to it.
            continue
        vertices = patch.get_path().vertices
        left, right = float(vertices[:, 0].min()), float(vertices[:, 0].max())
        box_centers.append((left + right) / 2.0)
        box_half_widths.append((right - left) / 2.0)
        box_colors.append((red, green, blue, 1.0))
    if not box_centers:
        return

    centers = np.asarray(box_centers)
    for line in ax.lines:
        x_data = np.asarray(line.get_xdata(), dtype=float)
        if x_data.size == 0:
            continue
        center = (float(x_data.min()) + float(x_data.max())) / 2.0
        nearest = int(np.argmin(np.abs(centers - center)))
        if abs(centers[nearest] - center) > box_half_widths[nearest]:
            continue
        line.set_color(box_colors[nearest])
        line.set_linewidth(_BOX_LINE_WIDTH)


def plot_runtime_sweep(
    runtimes: pd.DataFrame,
    swept_parameter: Optional[str] = None,
    hue_column: str = "regime",
    value_column: str = RUNTIME_COLUMN,
    log_y: bool = True,
    show_points: bool = True,
    color_outlines: bool = True,
    output_dir: Optional[Path] = None,
    file_stem: str = "runtime_sweep",
    fmt: str = "png",
    show_legend: bool = True,
    show_title: bool = True,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Draw boxplots of runtime against the swept parameter, split by a hue column.

    The swept values are placed on a categorical axis, since the sweeps are a
    handful of chosen levels rather than a continuum. Each box summarizes the
    sequences of one cell; the sequences are overlaid as jittered dots.

    Args:
        runtimes: Tidy frame from ``load_benchmark_runtimes``, already filtered to
            the cells that belong in this figure.
        swept_parameter: Config field on the x axis. When None it is taken from the
            frame's ``swept_parameter`` column, which must then hold one value.
        hue_column: Column to split the boxes by, ``"regime"`` or ``"device"``.
        value_column: Column on the y axis, defaults to ``"algorithm_seconds"``.
        log_y: Whether to use a logarithmic y axis. The sweeps span two orders of
            magnitude, so this is the default.
        show_points: Whether to overlay the individual sequences as dots.
        color_outlines: Whether to repaint each box's outline, median, whiskers and
            caps in its own hue via :func:`color_box_outlines`. On by default,
            because most cells of this benchmark collapse to a hairline on the
            logarithmic axis and would otherwise be indistinguishable.
        output_dir: Directory to save the figure into. Ignored when ``ax`` is given;
            when None in standalone mode the figure is not saved.
        file_stem: Basename of the saved file, without extension.
        fmt: File format, e.g. ``"png"``, ``"svg"``, ``"pdf"``.
        show_legend: Whether to draw the hue legend. Set False for a panel whose
            legend is provided elsewhere.
        show_title: Whether to draw the axis title. Set False for panel use.
        title: Title text. When None a title is derived from the swept parameter.
        ax: Axes to draw onto. When None a standalone figure is created.

    Returns:
        The Axes the boxes were drawn onto.

    Raises:
        ValueError: If the frame is empty, or ``swept_parameter`` is None and the
            frame covers more than one swept parameter.
    """
    if runtimes.empty:
        raise ValueError("Got an empty runtime frame to plot")
    if swept_parameter is None:
        found = sorted(runtimes["swept_parameter"].unique())
        if len(found) != 1:
            raise ValueError(
                f"The frame covers the swept parameters {found}; pass "
                "swept_parameter explicitly to pick one for the x axis"
            )
        swept_parameter = str(found[0])
    data = pd.DataFrame(runtimes[runtimes["swept_parameter"] == swept_parameter])
    if data.empty:
        raise ValueError(f"No rows with swept_parameter == {swept_parameter!r}")

    palette, hue_order = _resolve_hue(hue_column, data)
    order = sorted(data["swept_value"].unique())

    own_figure = ax is None
    if ax is None:
        sns.set_theme(style="whitegrid", font_scale=1.1)
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    sns.boxplot(
        data=data,
        x="swept_value",
        y=value_column,
        hue=hue_column,
        order=order,
        hue_order=hue_order,
        palette=palette,
        showfliers=not show_points,
        ax=ax,
    )
    if color_outlines:
        # before the dots are added, so only the boxes' own line artists are on
        # the axes to be matched
        color_box_outlines(ax)
    if show_points:
        # the dots only mark the replicates, so every hue level gets the same
        # colour; a dict rather than color= keeps seaborn from reading it as a
        # gradient palette while hue is set
        levels = hue_order if hue_order is not None else sorted(data[hue_column].unique())
        sns.stripplot(
            data=data,
            x="swept_value",
            y=value_column,
            hue=hue_column,
            order=order,
            hue_order=hue_order,
            dodge=True,
            jitter=_POINT_JITTER,
            size=_POINT_SIZE,
            alpha=_POINT_ALPHA,
            palette={level: _POINT_COLOUR for level in levels},
            legend=False,
            ax=ax,
        )
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(SWEPT_PARAMETER_LABELS.get(swept_parameter, swept_parameter))
    ax.set_ylabel(RUNTIME_AXIS_LABEL if value_column == RUNTIME_COLUMN else value_column)
    if show_title:
        ax.set_title(
            title
            if title is not None
            else f"Runtime vs {SWEPT_PARAMETER_LABELS.get(swept_parameter, swept_parameter).lower()}"
        )
    if show_legend:
        ax.legend(title=hue_column.capitalize(), fontsize=8, title_fontsize=8)
    elif ax.get_legend() is not None:
        ax.get_legend().remove()  # type: ignore[union-attr]

    if own_figure:
        fig.tight_layout()  # type: ignore[union-attr]
        if output_dir is not None:
            output_path = Path(output_dir) / f"{file_stem}.{fmt}"
            fig.savefig(output_path, bbox_inches="tight", dpi=150)  # type: ignore[union-attr]
            plt.close(fig)
            print(f"Saved: {output_path}")
    return ax


def plot_scaling_fit(
    runtimes: pd.DataFrame,
    fits: Dict[str, ScalingFit],
    x_column: str = "n_evaluations",
    hue_column: str = "regime",
    value_column: str = RUNTIME_COLUMN,
    palette: Optional[Dict[str, str]] = None,
    output_dir: Optional[Path] = None,
    file_stem: str = "runtime_scaling",
    fmt: str = "png",
    show_legend: bool = True,
    show_title: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Scatter runtime against a cost driver on log-log axes, with the fitted power laws.

    One power law is drawn per hue level, over the range of x values that level
    covers, and its exponent is written into the legend with its confidence
    interval. Linear scaling shows up as an exponent of 1; anything else is the
    finding.

    Args:
        runtimes: Tidy frame from ``load_benchmark_runtimes``, already filtered.
        fits: Mapping of hue level to the :class:`ScalingFit` for that level, from
            ``fit_log_log_scaling``. Levels missing from the mapping get points but
            no line.
        x_column: Column on the x axis, e.g. ``"n_evaluations"``.
        hue_column: Column to split the series by, ``"regime"`` or ``"device"``.
        value_column: Column on the y axis, defaults to ``"algorithm_seconds"``.
        palette: Colour per hue level, overriding the module palette of
            ``hue_column``. Use it when a figure needs its own colour scheme; None
            keeps the module palette.
        output_dir: Directory to save the figure into. Ignored when ``ax`` is given;
            when None in standalone mode the figure is not saved.
        file_stem: Basename of the saved file, without extension.
        fmt: File format, e.g. ``"png"``, ``"svg"``, ``"pdf"``.
        show_legend: Whether to draw the legend carrying the exponents.
        show_title: Whether to draw the axis title. Set False for panel use.
        title: Title text. When None a title is derived from the x column.
        xlabel: Label for the x axis. When None it is derived from ``x_column``,
            which reads badly for a generic column such as ``"swept_value"``.
        ax: Axes to draw onto. When None a standalone figure is created.

    Returns:
        The Axes the points were drawn onto.

    Raises:
        ValueError: If the frame is empty, or holds no positive x/y pairs to plot.
    """
    if runtimes.empty:
        raise ValueError("Got an empty runtime frame to plot")
    data = runtimes.dropna(subset=[x_column, value_column])
    data = pd.DataFrame(data[(data[x_column] > 0) & (data[value_column] > 0)])
    if data.empty:
        raise ValueError(
            f"No rows with positive {x_column} and {value_column} to plot on log axes"
        )
    module_palette, hue_order = _resolve_hue(hue_column, data)
    if palette is None:
        palette = module_palette
    if hue_order is None:
        hue_order = sorted(str(value) for value in data[hue_column].unique())

    own_figure = ax is None
    if ax is None:
        sns.set_theme(style="whitegrid", font_scale=1.1)
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    for level in hue_order:
        level_data = data[data[hue_column] == level]
        if level_data.empty:
            continue
        colour = palette[level] if palette is not None else None
        fit = fits.get(level)
        label = str(level)
        if fit is not None:
            label = (
                f"{level}: exponent {fit.exponent:.2f} "
                f"[{fit.exponent_ci_low:.2f}, {fit.exponent_ci_high:.2f}], "
                f"R² {fit.r_squared:.3f}"
            )
        ax.scatter(
            level_data[x_column],
            level_data[value_column],
            s=8,
            alpha=0.55,
            color=colour,
            label=label,
        )
        if fit is not None:
            x_line = np.geomspace(
                float(level_data[x_column].min()), float(level_data[x_column].max()), 50
            )
            ax.plot(
                x_line,
                fit.coefficient * x_line**fit.exponent,
                color=colour,
                linewidth=1.6,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        xlabel if xlabel is not None else x_column.replace("_", " ").capitalize()
    )
    ax.set_ylabel(RUNTIME_AXIS_LABEL if value_column == RUNTIME_COLUMN else value_column)
    if show_title:
        ax.set_title(
            title
            if title is not None
            else f"Runtime vs {x_column.replace('_', ' ')}"
        )
    if show_legend:
        ax.legend(fontsize=7, loc="upper left")
    elif ax.get_legend() is not None:
        ax.get_legend().remove()  # type: ignore[union-attr]

    if own_figure:
        fig.tight_layout()  # type: ignore[union-attr]
        if output_dir is not None:
            output_path = Path(output_dir) / f"{file_stem}.{fmt}"
            fig.savefig(output_path, bbox_inches="tight", dpi=150)  # type: ignore[union-attr]
            plt.close(fig)
            print(f"Saved: {output_path}")
    return ax
