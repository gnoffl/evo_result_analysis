"""Shared styling primitives for publication figures.

This module is the single source of truth for the typography and geometry of
publication figures. Individual analysis/plotting functions draw onto axes that
are created inside a :func:`publication_style` context, so the stylesheet here
governs every panel uniformly (nothing is rescaled after rendering).

The defaults target a generic vector figure (roughly a 180 mm double-column
width, ~7-8 pt sans-serif text, editable text in vector output). The exact
numbers are expected to be tuned once a target journal is fixed; override them
per call via ``extra_rc`` or edit :data:`PUBLICATION_RC`.

Example
-------
>>> from workflows.paper_plots.style import (
...     publication_style, figure_size_inches, panel_label,
...     save_publication_figure, DOUBLE_COLUMN_MM,
... )
>>> with publication_style():
...     fig = plt.figure(figsize=figure_size_inches(DOUBLE_COLUMN_MM, 120))
...     ax = fig.add_subplot()
...     some_plot_function(data, ax=ax)   # draws onto the shared-style axes
...     panel_label(ax, "A")
...     save_publication_figure(fig, "figures/fig1.pdf")
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text

# Common journal column widths in millimetres. Adjust to the target journal.
SINGLE_COLUMN_MM: float = 85.0
DOUBLE_COLUMN_MM: float = 180.0

_MM_PER_INCH: float = 25.4

# The publication stylesheet as an rcParams dict. Applied on top of matplotlib's
# default style (see :func:`publication_style`), so the result is deterministic
# regardless of any global seaborn/matplotlib state set elsewhere.
PUBLICATION_RC: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "legend.title_fontsize": 7.0,
    "figure.titlesize": 9.0,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Cleaner scatter marks: no dark rings from overlapping semi-transparent
    # points, and ticks that point outward (standard for print figures).
    "scatter.edgecolors": "none",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    # Keep text as editable text (not paths) in vector output.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def mm_to_inch(millimetres: float) -> float:
    """Convert a length in millimetres to inches.

    Args:
        millimetres: Length in millimetres.

    Returns:
        The equivalent length in inches.
    """
    return millimetres / _MM_PER_INCH


def figure_size_inches(width_mm: float, height_mm: float) -> tuple[float, float]:
    """Return a matplotlib ``figsize`` (in inches) from millimetre dimensions.

    Args:
        width_mm: Figure width in millimetres.
        height_mm: Figure height in millimetres.

    Returns:
        ``(width_inch, height_inch)`` suitable for ``plt.figure(figsize=...)``.
    """
    return mm_to_inch(width_mm), mm_to_inch(height_mm)


@contextlib.contextmanager
def publication_style(extra_rc: Optional[dict] = None) -> Iterator[None]:
    """Context manager that activates the publication stylesheet.

    The publication rcParams are applied on top of matplotlib's ``default``
    style, so styling is deterministic irrespective of any global state (e.g. a
    prior ``seaborn.set_context`` call). Original rcParams are restored on exit.

    Args:
        extra_rc: Optional rcParams overrides merged on top of
            :data:`PUBLICATION_RC` for this context only (e.g. a larger base
            font for a poster variant).

    Yields:
        None. Create figures/axes inside the ``with`` block so they inherit the
        publication style.
    """
    rc = dict(PUBLICATION_RC)
    if extra_rc:
        rc.update(extra_rc)
    with plt.style.context(["default", rc]):    #type: ignore
        yield


def panel_label(
    ax: Axes,
    label: str,
    *,
    x: float = -0.08,
    y: float = 1.02,
    fontsize: Optional[float] = None,
    fontweight: str = "bold",
) -> Text:
    """Stamp a bold panel label (e.g. ``"A"``) on an axes.

    The label is positioned in axes-fraction coordinates, so its placement is
    independent of the data and the panel's physical size.

    Args:
        ax: Axes to label.
        label: Label text, typically a single uppercase letter.
        x: Horizontal position in axes-fraction coordinates (0 = left edge).
            Defaults to slightly left of the axes to sit in the margin.
        y: Vertical position in axes-fraction coordinates (1 = top edge).
        fontsize: Label font size in points. ``None`` uses the active
            ``font.size`` rcParam.
        fontweight: Font weight for the label. Defaults to ``"bold"``.

    Returns:
        The created :class:`matplotlib.text.Text` artist.
    """
    return ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        va="bottom",
        ha="right",
    )


def save_publication_figure(
    fig: Figure,
    path: str,
    *,
    dpi: int = 600,
    transparent: bool = False,
) -> None:
    """Save a composed figure with consistent publication output settings.

    Args:
        fig: The figure to save.
        path: Output path. The extension determines the format; use a vector
            format (``.pdf`` or ``.svg``) for publication.
        dpi: Resolution for any raster elements; ignored for pure vector
            content. Defaults to 600.
        transparent: Whether the figure background should be transparent.
            Defaults to False.
    """
    if not path.lower().endswith(".svg"):
        raise ValueError(
            f"Unsupported file extension in {path}. Use .svg for publications."
        )
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)
    # replace .svg at the end with .png for simpler agent inspection and google docs integration
    png_path = path[:-4] + ".png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", transparent=transparent)


def sync_axis_limits(
    axes: Sequence[Axes],
    *,
    sync_x: bool = True,
    sync_y: bool = True,
) -> None:
    """Give a group of axes a common set of x/y limits.

    The shared limits are the union of the individual limits (minimum of the
    lower bounds, maximum of the upper bounds), so no data is clipped. Use this
    to make panels of the same plot type directly comparable across runs.

    Args:
        axes: The axes to harmonise. A group of fewer than two axes is a no-op.
        sync_x: Whether to synchronise the x-axis limits. Defaults to True.
        sync_y: Whether to synchronise the y-axis limits. Defaults to True.
    """
    axes = list(axes)
    if len(axes) < 2:
        return

    if sync_x:
        lower = min(ax.get_xlim()[0] for ax in axes)
        upper = max(ax.get_xlim()[1] for ax in axes)
        for ax in axes:
            ax.set_xlim(lower, upper)

    if sync_y:
        lower = min(ax.get_ylim()[0] for ax in axes)
        upper = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(lower, upper)
