"""Plotting for cross-run transcription-factor (TF) comparisons.

This module is the visualization half of the TF-comparison toolbox: it turns a
TF × run matrix (and optional significance stars) into an annotated diverging
heatmap. It contains **no statistics and no IO** — values and stars are computed
by ``tf_comparison_calc.py`` and passed in by a thin orchestration script.

``row_stars`` / ``cell_stars`` realise the "compute everything, display a subset"
mechanism: a caller computes whatever stats it wants, writes its CSVs, and passes
only the chosen q-value columns here as stars.
"""

from typing import Dict, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

COLORMAP = "RdBu_r"
# q-value thresholds (most stringent first) mapped to the star annotation.
STAR_THRESHOLDS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]


def q_to_stars(qvalue: float) -> str:
    """Return the significance star string for a q-value (empty if not significant)."""
    if pd.isna(qvalue):
        return ""
    for threshold, stars in STAR_THRESHOLDS:
        if qvalue < threshold:
            return stars
    return ""


def plot_heatmap(
    matrix: pd.DataFrame,
    annotate: bool,
    cbar_label: str = "diff_calc / gene",
    row_stars: Optional[Dict[str, str]] = None,
    cell_stars: Optional[Dict[str, Dict[str, str]]] = None,
    separator_after_column: Optional[int] = None,
) -> Figure:
    """Draw the cross-run TF heatmap on a symmetric diverging scale.

    When ``row_stars`` maps a TF to a non-empty star string, that string is
    appended to the TF's row label to mark inter-run significance.

    When ``cell_stars`` maps a run label to a ``{tf: stars}`` dict, the stars
    are appended to each cell's numeric annotation to mark intra-run significance
    (i.e. whether diff vs reference is significant within that run).

    Args:
        matrix: TF × run DataFrame of normalized diff values.
        annotate: Whether to annotate cells with numeric values (and stars).
        cbar_label: Colour bar axis label.
        row_stars: Optional inter-run significance stars keyed by TF name.
        cell_stars: Optional intra-run significance stars keyed by run label then
            TF name; only used when ``annotate`` is True.
        separator_after_column: Optional column index after which to draw a thick
            vertical divider (e.g. to split a left group from a right group). The
            divider is drawn only when ``0 < separator_after_column < n_runs``;
            ``None`` draws no separator (the flat / no-grouping case).

    Returns:
        Matplotlib Figure with a single heatmap axes.
    """
    limit = float(np.nanmax(np.abs(matrix.to_numpy()))) or 1.0
    n_tfs, n_runs = matrix.shape
    fig, ax = plt.subplots(figsize=(max(4.0, 0.9 * n_runs + 2.5), max(4.0, 0.3 * n_tfs + 1.0)))

    annot_arg: Union[bool, pd.DataFrame] = annotate
    fmt_arg = ".2f"
    if annotate and cell_stars:
        annot_data = pd.DataFrame("", index=matrix.index, columns=matrix.columns)
        for col in matrix.columns:
            col_stars = cell_stars.get(str(col), {})
            for tf in matrix.index:
                val = matrix.loc[tf, col]
                if pd.isna(val):
                    annot_data.loc[tf, col] = ""
                else:
                    stars = col_stars.get(str(tf), "")
                    annot_data.loc[tf, col] = f"{val:.2f}{stars}"
        annot_arg = annot_data
        fmt_arg = ""

    sns.heatmap(
        matrix,
        cmap=COLORMAP,
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=annot_arg,
        fmt=fmt_arg,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    if separator_after_column is not None and 0 < separator_after_column < n_runs:
        ax.axvline(separator_after_column, color="black", linewidth=2.0)
    ax.set_xlabel("run")
    ax.set_ylabel("TF family")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    if row_stars:
        labels = [
            f"{tf} {row_stars[tf]: <3}" if row_stars.get(tf) else str(tf) + "    " for tf in matrix.index
        ]
        ax.set_yticklabels(labels, rotation=0)
    fig.tight_layout()
    return fig
