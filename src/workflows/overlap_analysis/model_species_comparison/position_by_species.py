"""Position-resolved STARR-seq agreement, per model of both species.

Where :mod:`correlation_by_species` asks how well a model agrees with the
measurement overall, this asks *where along the sequence* it agrees, and whether
that location is the same for every model. The pooled analysis puts the
highest-correlation window at centre 928 (see
:data:`_common.HIGHLIGHT_WINDOW_CENTER`); the question here is whether each
cross-validation model of each training species reproduces that, and how strong
the correlation is inside that region.

Reuses :func:`_common.compute_correlation_by_position` unchanged, so the curves
are the same quantity the existing ``correlation_by_position_fixed_window``
plots show: for each window of :data:`_common.BUCKET_SIZE` consecutive
``group_overlap_start`` values, the Spearman correlation between
``prediction_mutated`` and ``enrichment``, keyed by the window centre
``group_overlap_start + BUCKET_SIZE // 2``.

Runs off the per-model caches written by :mod:`correlation_by_species`, so it
performs no model inference. Each model's curve is itself cached, so re-running
is cheap.

Two caveats on comparing peaks across models:

- The window grid is identical for every model (it is set by the fragment
  positions, which do not depend on the model), so the upward bias of taking a
  maximum over many overlapping windows is the same for all models and cancels
  in the comparison. Absolute peak heights are still optimistic.
- Peak *location* is read off the raw curve; the plotted curves are smoothed
  with a centred rolling mean of :data:`_common.ROLLING_WINDOW_SIZE` windows
  purely for legibility, and the smoothed peak is reported alongside.
"""
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.utils.io import print_section_header, print_status, print_subsection
from workflows.overlap_analysis import _common
from workflows.overlap_analysis.model_species_comparison import _summary
from workflows.overlap_analysis.model_species_comparison._models import (
    SpeciesModel,
    list_species_models,
)
from workflows.overlap_analysis.model_species_comparison.correlation_by_species import (
    cache_path_for as enrichment_cache_path_for,
)

BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, "cache")
CURVES_CSV_PATH = os.path.join(BASE_DIR, "position_curves_by_model.csv")
PEAK_SUMMARY_CSV_PATH = os.path.join(BASE_DIR, "position_peak_summary.csv")
CURVES_PLOT_PATH = os.path.join(BASE_DIR, "position_correlation_curves.png")
PEAK_PLOT_PATH = os.path.join(BASE_DIR, "position_peak_summary.png")

# Region of interest on the window-centre axis. A window centred at ``c`` covers
# ``group_overlap_start`` in ``[c - BUCKET_SIZE // 2, c + BUCKET_SIZE // 2)``, so
# this region covers fragments whose overlap starts between 700 and 1100.
REGION_OF_INTEREST = (800.0, 1000.0)

# Minimum number of fragments a window must contain to be eligible as the global
# correlation peak. ``_common.MIN_POINTS_PER_FIXED_WINDOW`` (50) is low enough to
# admit the sparse windows at either end of the 3020 bp frame: the last window
# holds ~52 fragments drawn from only ~8 genes, where the Spearman correlation is
# dominated by between-gene variation and reaches values far above anything in
# the well-populated interior (~700-900 fragments from ~100 genes). Those edge
# windows are not comparable to interior ones, so they are excluded from
# peak-finding. Region measurements are unaffected, being confined to
# :data:`REGION_OF_INTEREST` where support is uniformly high.
MIN_WINDOW_SUPPORT_FOR_PEAK = 200

SPECIES_COLORS = {"Ara": "#2166ac", "Slyc": "#d95f02"}


def curve_cache_path_for(model: SpeciesModel) -> str:
    """Return the cache path for one model's position curve.

    Args:
        model: The cross-validation model.

    Returns:
        Absolute path to the per-model curve cache CSV.
    """
    return os.path.join(
        CACHE_DIR, f"position_curve_{model.species}_{model.held_out_chromosome}.csv"
    )


def compute_window_support(enrichment_df: pd.DataFrame) -> pd.DataFrame:
    """Count the fragments and genes behind each position window.

    Mirrors the windowing of :func:`_common.compute_correlation_by_position`
    exactly - a window starts at each distinct ``group_overlap_start`` value,
    spans :data:`_common.BUCKET_SIZE`, and is keyed by its centre - so the counts
    join onto that function's output by ``position``.

    Args:
        enrichment_df: One model's enrichment dataframe, carrying
            ``group_overlap_start`` and ``gene``.

    Returns:
        Dataframe with columns ``position``, ``count_fragments`` and
        ``count_genes``, one row per window.
    """
    starts = enrichment_df["group_overlap_start"]
    support_rows = []
    for start in sorted(starts.unique()):
        window_df = enrichment_df[
            (starts >= start) & (starts < start + _common.BUCKET_SIZE)
        ]
        support_rows.append(
            {
                "position": float(start + _common.BUCKET_SIZE / 2),
                "count_fragments": len(window_df),
                "count_genes": window_df["gene"].nunique(),
            }
        )
    return pd.DataFrame(support_rows)


def position_curve_for_model(model: SpeciesModel) -> pd.DataFrame:
    """Load or compute the position-resolved correlation curve for one model.

    Args:
        model: The cross-validation model.

    Returns:
        Tidy curve with columns ``species``, ``held_out_chromosome``,
        ``position``, ``correlation``, ``slope``, ``p_value`` and the
        ``*_rolling`` smoothed variants.

    Raises:
        FileNotFoundError: If the model's enrichment cache is missing, i.e.
            ``correlation_by_species`` has not been run for it yet.
    """
    curve_path = curve_cache_path_for(model)
    if os.path.exists(curve_path):
        print_status(f"cache hit: {os.path.basename(curve_path)}", "INFO")
        return pd.read_csv(curve_path)

    enrichment_path = enrichment_cache_path_for(model)
    if not os.path.exists(enrichment_path):
        raise FileNotFoundError(
            f"Missing enrichment cache {enrichment_path}. Run "
            "correlation_by_species first so the per-model predictions exist."
        )
    enrichment_df = pd.read_csv(enrichment_path)
    label = f"{model.species}/{model.held_out_chromosome}"
    _, _, _, curve_df = _common.compute_correlation_by_position(
        [(enrichment_df, label, SPECIES_COLORS[model.species])]
    )
    curve_df = curve_df.drop(columns=["label"])
    curve_df = curve_df.merge(
        compute_window_support(enrichment_df), on="position", how="left"
    )
    curve_df.insert(0, "held_out_chromosome", model.held_out_chromosome)
    curve_df.insert(0, "species", model.species)
    os.makedirs(CACHE_DIR, exist_ok=True)
    curve_df.to_csv(curve_path, index=False)
    print_status(
        f"wrote cache: {os.path.basename(curve_path)} ({len(curve_df)} windows)",
        "SUCCESS",
    )
    return curve_df


def summarize_peak(
    curve_df: pd.DataFrame,
    region: Tuple[float, float] = REGION_OF_INTEREST,
    min_support: int = MIN_WINDOW_SUPPORT_FOR_PEAK,
) -> Dict[str, float]:
    """Locate one model's correlation peak and measure it inside the region.

    The global peak is taken only over windows holding at least ``min_support``
    fragments; see :data:`MIN_WINDOW_SUPPORT_FOR_PEAK` for why the sparse windows
    at the ends of the frame are excluded. The unrestricted peak is reported too,
    as ``peak_position_any_support``, so nothing is hidden.

    Args:
        curve_df: One model's tidy curve, as returned by
            :func:`position_curve_for_model`.
        region: Inclusive ``(low, high)`` bounds on the window-centre axis.
        min_support: Minimum fragments a window needs to be eligible as the peak.

    Returns:
        Mapping with the supported global peak (``peak_position``,
        ``peak_correlation``, ``count_fragments_at_peak``,
        ``count_genes_at_peak``), the smoothed peak (``peak_position_rolling``,
        ``peak_correlation_rolling``), the unrestricted peak
        (``peak_position_any_support``, ``peak_correlation_any_support``),
        whether the supported peak falls inside the region (``peak_in_region``),
        the in-region strength (``max_correlation_in_region``,
        ``position_of_region_max``, ``mean_correlation_in_region``), and window
        counts.

    Raises:
        ValueError: If the curve has no window with a defined correlation, no
            window meeting ``min_support``, or no window inside the region.
    """
    defined_df = curve_df.dropna(subset=["correlation"])
    if defined_df.empty:
        raise ValueError("Curve contains no window with a defined correlation")

    unrestricted_peak_row = defined_df.loc[defined_df["correlation"].idxmax()]
    supported_df = defined_df[defined_df["count_fragments"] >= min_support]
    if supported_df.empty:
        raise ValueError(f"No window holds at least {min_support} fragments")
    peak_row = supported_df.loc[supported_df["correlation"].idxmax()]

    rolling_df = curve_df.dropna(subset=["correlation_rolling"])
    rolling_df = rolling_df[rolling_df["count_fragments"] >= min_support]
    rolling_peak_row = rolling_df.loc[rolling_df["correlation_rolling"].idxmax()]

    low, high = region
    region_df = defined_df[
        (defined_df["position"] >= low) & (defined_df["position"] <= high)
    ]
    if region_df.empty:
        raise ValueError(f"No window falls inside the region {region}")
    region_peak_row = region_df.loc[region_df["correlation"].idxmax()]

    return {
        "peak_position": float(peak_row["position"]),
        "peak_correlation": float(peak_row["correlation"]),
        "count_fragments_at_peak": int(peak_row["count_fragments"]),
        "count_genes_at_peak": int(peak_row["count_genes"]),
        "peak_position_rolling": float(rolling_peak_row["position"]),
        "peak_correlation_rolling": float(rolling_peak_row["correlation_rolling"]),
        "peak_position_any_support": float(unrestricted_peak_row["position"]),
        "peak_correlation_any_support": float(unrestricted_peak_row["correlation"]),
        "peak_in_region": bool(low <= float(peak_row["position"]) <= high),
        "max_correlation_in_region": float(region_peak_row["correlation"]),
        "position_of_region_max": float(region_peak_row["position"]),
        "mean_correlation_in_region": float(region_df["correlation"].mean()),
        "count_windows": len(defined_df),
        "count_windows_supported": len(supported_df),
        "count_windows_in_region": len(region_df),
    }


def build_peak_summary(
    curves: List[pd.DataFrame], region: Tuple[float, float] = REGION_OF_INTEREST
) -> pd.DataFrame:
    """Summarise every model's peak into one table.

    Args:
        curves: One tidy curve per model.
        region: Inclusive ``(low, high)`` bounds on the window-centre axis.

    Returns:
        One row per model, sorted by species then held-out chromosome.
    """
    summary_rows = []
    for curve_df in curves:
        row = {
            "species": str(curve_df["species"].iloc[0]),
            "held_out_chromosome": str(curve_df["held_out_chromosome"].iloc[0]),
        }
        row.update(summarize_peak(curve_df, region))
        summary_rows.append(row)
    return (
        pd.DataFrame(summary_rows)
        .sort_values(["species", "held_out_chromosome"])
        .reset_index(drop=True)
    )


def plot_position_curves(
    all_curves_df: pd.DataFrame,
    output_path: str,
    region: Tuple[float, float] = REGION_OF_INTEREST,
    min_support: int = MIN_WINDOW_SUPPORT_FOR_PEAK,
) -> None:
    """Overlay every model's smoothed position curve, colored by species.

    One thin line per model plus a thick per-species median, with the region of
    interest shaded, so a shared peak location shows up as the curves rising
    together inside the shaded band.

    Windows holding fewer than ``min_support`` fragments are omitted, since their
    correlations are not comparable to the well-populated interior; the title
    records the omission.

    Args:
        all_curves_df: Concatenated tidy curves for all models.
        output_path: Image path to write; parent directories are created.
        region: Inclusive ``(low, high)`` bounds on the window-centre axis.
        min_support: Minimum fragments a window needs to be plotted.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_curves_df = all_curves_df[all_curves_df["count_fragments"] >= min_support]
    figure, axes = plt.subplots(figsize=(11, 6))
    axes.axvspan(region[0], region[1], color="#000000", alpha=0.07, zorder=0)
    axes.axhline(0.0, color="#999999", linestyle="--", linewidth=1, zorder=1)

    for species, species_df in all_curves_df.groupby("species"):
        color = SPECIES_COLORS[str(species)]
        for _, model_df in species_df.groupby("held_out_chromosome"):
            sorted_df = model_df.sort_values("position")
            axes.plot(
                sorted_df["position"], sorted_df["correlation_rolling"],
                color=color, linewidth=0.8, alpha=0.45, zorder=2,
            )
        median_curve = (
            species_df.groupby("position")["correlation_rolling"].median().sort_index()
        )
        model_count = species_df["held_out_chromosome"].nunique()
        axes.plot(
            median_curve.index, median_curve.values,
            color=color, linewidth=2.5, zorder=3,
            label=f"{species} median ({model_count} models)",
        )

    axes.set_xlabel("Overlap start position (window centre)")
    axes.set_ylabel(
        f"Spearman correlation\n(rolling mean of {_common.ROLLING_WINDOW_SIZE} windows)"
    )
    axes.set_title(
        "Position-resolved agreement with STARR-seq, per cross-validation model\n"
        f"(windows with fewer than {min_support} fragments omitted)"
    )
    axes.legend()
    figure.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(figure)


def plot_peak_summary(
    summary_df: pd.DataFrame,
    output_path: str,
    region: Tuple[float, float] = REGION_OF_INTEREST,
) -> None:
    """Plot peak location and in-region strength per species.

    Left panel: where each model's peak sits, with the region shaded, answering
    whether the peak location is shared. Right panel: the in-region maximum
    correlation, comparing strength where it matters.

    Args:
        summary_df: Output of :func:`build_peak_summary`.
        output_path: Image path to write; parent directories are created.
        region: Inclusive ``(low, high)`` bounds on the window-centre axis.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid")
    species_order = sorted(summary_df["species"].unique())
    palette = {species: SPECIES_COLORS[species] for species in species_order}
    figure, axes_array = plt.subplots(1, 2, figsize=(12, 5))

    panels = [
        ("peak_position", "Position of the global correlation peak", True),
        ("max_correlation_in_region", "Max Spearman r inside the region", False),
    ]
    for axes, (column, panel_title, shade_region) in zip(axes_array, panels):
        if shade_region:
            axes.axhspan(region[0], region[1], color="#000000", alpha=0.07, zorder=0)
        sns.boxplot(
            data=summary_df, x="species", y=column, order=species_order,
            hue="species", hue_order=species_order, palette=palette, legend=False,
            showfliers=False, width=0.5, ax=axes,
        )
        sns.stripplot(
            data=summary_df, x="species", y=column, order=species_order,
            color="#222222", size=7, jitter=0.15, ax=axes,
        )
        axes.set_xlabel("Training species")
        axes.set_title(panel_title)
    axes_array[0].set_ylabel("Overlap start position (window centre)")
    axes_array[1].set_ylabel("Spearman r")
    figure.suptitle(
        "Correlation peak location and strength, region "
        f"{int(region[0])}-{int(region[1])}"
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(figure)


def main() -> None:
    """Build every model's position curve and summarise the peaks by species."""
    _common.configure_matplotlib()
    models = list_species_models()
    print_section_header("POSITION-RESOLVED AGREEMENT BY MODEL TRAINING SPECIES")

    curves: List[pd.DataFrame] = []
    for _, model in enumerate(models, start=1):

        curves.append(position_curve_for_model(model))

    all_curves_df = pd.concat(curves, ignore_index=True)
    all_curves_df.to_csv(CURVES_CSV_PATH, index=False)

    print_subsection("Peak summary")
    summary_df = build_peak_summary(curves)
    summary_df.to_csv(PEAK_SUMMARY_CSV_PATH, index=False)
    plot_position_curves(all_curves_df, CURVES_PLOT_PATH)
    plot_peak_summary(summary_df, PEAK_PLOT_PATH)

    print(summary_df.to_string(index=False))
    peak_in_region_count = int(summary_df["peak_in_region"].sum())
    print_status(
        f"{peak_in_region_count}/{len(summary_df)} models peak inside "
        f"{REGION_OF_INTEREST[0]:.0f}-{REGION_OF_INTEREST[1]:.0f}",
        "INFO",
    )

    region_summary = _summary.compare_species(
        summary_df.rename(columns={"max_correlation_in_region": "spearman_r"})
    )
    print(region_summary.to_string(index=False))
    print_status("done", "SUCCESS")


if __name__ == "__main__":
    main()
