"""Visualize how sibling MSR models score sequences evolved against one MSR model.

Reads the CSV files written by ``reevaluate.py`` and produces, per evolution run:

* ``pooled.pdf`` -- all genes pooled after per gene normalization, showing the
  optimization model's Pareto front against the spread of the other models.
* ``examples.pdf`` -- three individual genes with raw, unnormalized predictions.

See PLAN.md in this folder for the rationale behind the decisions taken here.
"""

import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot)
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from workflows.adversarial.reevaluate import (  # noqa: E402
    CSV_FILE_NAME,
    DEFAULT_MODELS_FOLDER,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RUN_FOLDERS,
    PREDICTION_COLUMN_PREFIX,
    build_output_folder,
    clean_gene_id,
)

NORMALIZED_FITNESS_COLUMN = "normalized_original_fitness"
NORMALIZED_PREFIX = "normalized_"
EXPANDED_COLUMN = "is_expanded"

REFERENCE_COLOR = "#c0392b"
OTHERS_COLOR = "#2f4f4f"
POOLED_FILE_NAME = "pooled"
EXAMPLES_FILE_NAME = "examples"
GAP_COLUMN = "mean_normalized_gap"
GAP_FILE_NAME = f"{GAP_COLUMN}.csv"
GAP_LABEL = "mean normalized gap"
NUMBER_OF_EXAMPLE_GENES = 3


def default_csv_paths() -> List[str]:
    """Build the CSV paths that ``reevaluate.py`` writes with its default arguments.

    Returns:
        One CSV path per default run folder.
    """
    return [
        os.path.join(
            build_output_folder(DEFAULT_OUTPUT_ROOT, run_folder, DEFAULT_MODELS_FOLDER),
            CSV_FILE_NAME,
        )
        for run_folder in DEFAULT_RUN_FOLDERS
    ]


def get_prediction_columns(predictions: pd.DataFrame) -> List[str]:
    """List the prediction columns of a re-evaluation table.

    Args:
        predictions: Table as written by ``reevaluate.py``.

    Returns:
        Sorted list of prediction column names.
    """
    return sorted(
        column for column in predictions.columns if column.startswith(PREDICTION_COLUMN_PREFIX)
    )


def get_other_model_columns(predictions: pd.DataFrame) -> List[str]:
    """List the prediction columns of the models not used for the optimization.

    Args:
        predictions: Table as written by ``reevaluate.py``.

    Returns:
        Sorted list of prediction column names, excluding every optimization model.

    Raises:
        ValueError: If no models are left after excluding the optimization models.
    """
    optimization_models = set()
    for entry in predictions["optimization_model"].unique():
        optimization_models.update(str(entry).split(";"))
    excluded = {f"{PREDICTION_COLUMN_PREFIX}{name}" for name in optimization_models}
    other_columns = [column for column in get_prediction_columns(predictions) if column not in excluded]
    if not other_columns:
        raise ValueError(
            "No models left after excluding the optimization models "
            f"{sorted(optimization_models)}."
        )
    return other_columns


def expand_gene_front(gene_predictions: pd.DataFrame, max_number_mutations: int) -> pd.DataFrame:
    """Fill the gaps of one gene's Pareto front by carrying whole rows forward.

    A Pareto front is monotone in the mutation count, so a missing mutation count
    means the algorithm found nothing strictly better than at the previous count. This
    is the row level analogue of ``expand_pareto_front`` in
    ``analysis.overview.simple_result_stats``, which fills a gap with the previous
    fitness: here the sequence and all model predictions are carried forward as well,
    so the other models can be plotted on the same grid.

    Args:
        gene_predictions: Rows of a single gene, one per mutation count present.
        max_number_mutations: Highest mutation count of the grid, inclusive.

    Returns:
        One row per mutation count in ``0 ... max_number_mutations``, with an
        additional boolean column marking the rows that were filled in.

    Raises:
        ValueError: If the gene has no entry at mutation count 0, which would leave the
            start of the grid undefined.
    """
    indexed = gene_predictions.drop_duplicates("mutation_count").set_index("mutation_count")
    if 0 not in indexed.index:
        raise ValueError(
            f"Gene {gene_predictions['gene_id'].iloc[0]} has no entry at mutation count 0."
        )
    if indexed.index.max() > max_number_mutations:
        warnings.warn(
            f"Gene {gene_predictions['gene_id'].iloc[0]} has entries above mutation count "
            f"{max_number_mutations}, which are dropped by the expansion."
        )
    grid = range(max_number_mutations + 1)
    expanded = indexed.reindex(grid).ffill()
    expanded[EXPANDED_COLUMN] = [mutation_count not in indexed.index for mutation_count in grid]
    return expanded.reset_index()


def normalize_gene_front(gene_predictions: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalize one gene so the optimization model's fitness spans ``[0, 1]``.

    With ``f_ref(m)`` the optimization model's fitness at mutation count ``m``,
    ``a = min_m f_ref(m)`` and ``b = max_m f_ref(m)``, every value ``p`` of the gene,
    including the predictions of the other models, is mapped to ``(p - a) / (b - a)``.
    The other models are therefore shifted by the same baseline ``a`` and are not
    bounded to ``[0, 1]``, which keeps it visible when a model scores the unmutated
    reference sequence differently from the optimization model.

    Args:
        gene_predictions: Rows of a single gene.

    Returns:
        The input table with added ``normalized_*`` columns, or None if the
        optimization model's fitness is constant across the front, which makes the
        normalization undefined.
    """
    minimum_fitness = gene_predictions["original_fitness"].min()
    maximum_fitness = gene_predictions["original_fitness"].max()
    fitness_range = maximum_fitness - minimum_fitness
    if fitness_range == 0:
        warnings.warn(
            f"Gene {gene_predictions['gene_id'].iloc[0]} has a constant fitness across "
            "its Pareto front and cannot be normalized; it is dropped."
        )
        return None
    normalized = gene_predictions.copy()
    normalized[NORMALIZED_FITNESS_COLUMN] = (
        normalized["original_fitness"] - minimum_fitness
    ) / fitness_range
    for column in get_prediction_columns(gene_predictions):
        normalized[f"{NORMALIZED_PREFIX}{column}"] = (
            normalized[column] - minimum_fitness
        ) / fitness_range
    return normalized


def prepare_predictions(
    predictions: pd.DataFrame, minimum_fitness_range: float = 0.0
) -> pd.DataFrame:
    """Expand and normalize every gene of a re-evaluation table.

    Args:
        predictions: Table as written by ``reevaluate.py``.
        minimum_fitness_range: Genes whose optimization model fitness spans less than
            this range across the Pareto front are dropped. The normalization divides
            by that range, so a gene that the optimization model could barely improve
            turns small absolute disagreements of the other models into very large
            normalized values.

    Returns:
        Concatenation of the expanded and normalized per gene tables.

    Raises:
        ValueError: If no gene survives the normalization.
    """
    prepared = []
    dropped_genes = []
    for gene_id, gene_predictions in predictions.groupby("gene_id", sort=True):
        fitness_range = (
            gene_predictions["original_fitness"].max() - gene_predictions["original_fitness"].min()
        )
        if fitness_range < minimum_fitness_range:
            dropped_genes.append(gene_id)
            continue
        max_number_mutations = int(gene_predictions["max_number_mutations"].iloc[0])
        expanded = expand_gene_front(gene_predictions, max_number_mutations)
        expanded["gene_id"] = gene_id
        normalized = normalize_gene_front(expanded)
        if normalized is not None:
            prepared.append(normalized)
    if dropped_genes:
        print(
            f"  dropped {len(dropped_genes)} genes with a fitness range below "
            f"{minimum_fitness_range}"
        )
    if not prepared:
        raise ValueError("No gene could be normalized.")
    return pd.concat(prepared, ignore_index=True)


def summarize_spread(
    prepared: pd.DataFrame, value_columns: List[str]
) -> pd.DataFrame:
    """Summarize the spread of several models per mutation count.

    All values of all given columns at one mutation count are pooled, so for the
    pooled figure the spread runs over models and genes at once.

    Args:
        prepared: Table with one row per gene and mutation count.
        value_columns: Columns to pool.

    Returns:
        Table indexed by mutation count with the columns ``median``, ``lower_quartile``,
        ``upper_quartile``, ``minimum`` and ``maximum``.
    """
    pooled = prepared.melt(
        id_vars=["mutation_count"], value_vars=value_columns, value_name="value"
    )
    grouped = pooled.groupby("mutation_count")["value"]
    return pd.DataFrame(
        {
            "median": grouped.median(),
            "lower_quartile": grouped.quantile(0.25),
            "upper_quartile": grouped.quantile(0.75),
            "minimum": grouped.min(),
            "maximum": grouped.max(),
        }
    )


MINIMUM_MAXIMUM_ALPHA = 0.15
INTERQUARTILE_ALPHA = 0.35


def draw_front_with_spread(
    axes: plt.Axes,
    mutation_counts: np.ndarray,
    reference_values: np.ndarray,
    spread: pd.DataFrame,
    reference_label: str,
    others_label: str,
    show_min_max: bool = True,
    reference_color: str = REFERENCE_COLOR,
    others_color: str = OTHERS_COLOR,
) -> List[Artist]:
    """Draw one Pareto front together with the spread of the other models.

    Args:
        axes: Axes to draw onto.
        mutation_counts: x values.
        reference_values: Optimization model values, one per mutation count.
        spread: Spread summary as returned by ``summarize_spread``.
        reference_label: Legend label of the optimization model line.
        others_label: Legend label of the other models' median line.
        show_min_max: Draw the min-max band. On the pooled panels the extremes of
            the per gene normalization cover the whole panel, so the band carries
            no information there and can be left out.
        reference_color: Colour of the optimization model's line.
        others_color: Colour of the other models' line and bands.

    Returns:
        Legend handles in drawing order. The bands are labelled through explicit
        patches instead of ``fill_between(label=...)``, so callers pass these to
        ``legend(handles=...)``.
    """
    spread_mutation_counts = spread.index.to_numpy()
    if show_min_max:
        axes.fill_between(
            spread_mutation_counts,
            spread["minimum"].to_numpy(),
            spread["maximum"].to_numpy(),
            color=others_color,
            alpha=MINIMUM_MAXIMUM_ALPHA,
            linewidth=0,
        )
    axes.fill_between(
        spread_mutation_counts,
        spread["lower_quartile"].to_numpy(),
        spread["upper_quartile"].to_numpy(),
        color=others_color,
        alpha=INTERQUARTILE_ALPHA,
        linewidth=0,
    )
    others_line = axes.plot(
        spread_mutation_counts,
        spread["median"].to_numpy(),
        color=others_color,
        linewidth=1.6,
    )[0]
    reference_line = axes.plot(
        mutation_counts,
        reference_values,
        color=reference_color,
        linewidth=2.0,
    )[0]
    axes.set_xlabel("number of mutations")
    minimum_maximum_handles = (
        [
            Patch(
                facecolor=others_color,
                alpha=MINIMUM_MAXIMUM_ALPHA,
                label="other models, min-max",
            )
        ]
        if show_min_max
        else []
    )
    return minimum_maximum_handles + [
        Patch(
            facecolor=others_color,
            alpha=INTERQUARTILE_ALPHA,
            label="other models, interquartile range",
        ),
        Line2D(
            [], [], color=others_color, linewidth=others_line.get_linewidth(), label=others_label
        ),
        Line2D(
            [],
            [],
            color=reference_color,
            linewidth=reference_line.get_linewidth(),
            label=reference_label,
        ),
    ]


def set_robust_ylimits(
    axes: plt.Axes,
    spread: pd.DataFrame,
    reference_values: np.ndarray,
    short_note: bool = False,
    annotate_truncation: bool = True,
) -> None:
    """Limit the y axis to the interquartile band and annotate what is cut off.

    The normalization divides by the range the optimization model achieved, so genes it
    could barely improve produce extreme normalized values for the other models. Those
    would compress everything else into a few pixels, therefore the axis is limited to
    the interquartile band and the optimization model's line. Whenever the min-max band
    reaches beyond the visible area, this is stated in the plot, so a truncated band
    cannot be mistaken for a narrow one.

    Args:
        axes: Axes to limit.
        spread: Spread summary as returned by ``summarize_spread``.
        reference_values: Values of the optimization model's line.
        short_note: Write the truncation note in an abbreviated form. The long
            sentence does not fit into a narrow panel of a composed figure.
        annotate_truncation: Write the note at all. The note describes the min-max
            band, so it is pointless on a panel that does not draw that band.
    """
    lower = min(float(np.min(reference_values)), float(spread["lower_quartile"].min()))
    upper = max(float(np.max(reference_values)), float(spread["upper_quartile"].max()))
    padding = 0.1 * (upper - lower) if upper > lower else 0.1
    lower, upper = lower - padding, upper + padding
    axes.set_ylim(lower, upper)

    band_minimum = float(spread["minimum"].min())
    band_maximum = float(spread["maximum"].max())
    cut_off = []
    if band_minimum < lower:
        cut_off.append(f"down to {band_minimum:.3g}")
    if band_maximum > upper:
        cut_off.append(f"up to {band_maximum:.3g}")
    if cut_off and annotate_truncation:
        prefix = "min-max " if short_note else "min-max band extends beyond the axis, "
        axes.text(
            0.98,
            0.02,
            prefix + " and ".join(cut_off),
            transform=axes.transAxes,
            ha="right",
            va="bottom",
            fontsize="x-small",
            color=OTHERS_COLOR,
        )


def draw_pooled_panel(
    axes: plt.Axes,
    prepared: pd.DataFrame,
    short_note: bool = False,
    show_min_max: bool = True,
    reference_color: str = REFERENCE_COLOR,
    others_color: str = OTHERS_COLOR,
) -> List[Artist]:
    """Draw the pooled, per gene normalized comparison onto one axes.

    All genes are pooled after the per gene normalization, so the optimization
    model's median front is shown against the spread of the other models over both
    models and genes. This is the panel body of :func:`plot_pooled`, without the
    figure, the title and the legend, so a composed figure can reuse it.

    Args:
        axes: Axes to draw onto.
        prepared: Table as returned by ``prepare_predictions``.
        short_note: Abbreviate the truncation note, see :func:`set_robust_ylimits`.
        show_min_max: Draw the min-max band and, with it, the note about the part
            of it that the robust y limits cut off.
        reference_color: Colour of the optimization model's line.
        others_color: Colour of the other models' line and bands.

    Returns:
        Legend handles in drawing order.
    """
    other_columns = get_other_model_columns(prepared)
    normalized_other_columns = [f"{NORMALIZED_PREFIX}{column}" for column in other_columns]
    spread = summarize_spread(prepared, normalized_other_columns)
    reference = prepared.groupby("mutation_count")[NORMALIZED_FITNESS_COLUMN].median()
    legend_handles = draw_front_with_spread(
        axes,
        reference.index.to_numpy(),
        reference.to_numpy(),
        spread,
        reference_label="optimization model (median over genes)",
        others_label=f"other models, median ({len(other_columns)} models)",
        show_min_max=show_min_max,
        reference_color=reference_color,
        others_color=others_color,
    )
    set_robust_ylimits(
        axes,
        spread,
        reference.to_numpy(),
        short_note=short_note,
        annotate_truncation=show_min_max,
    )
    axes.set_ylabel("normalized prediction")
    return legend_handles


def draw_example_panel(
    axes: plt.Axes,
    prepared: pd.DataFrame,
    gene_id: str,
    reference_color: str = REFERENCE_COLOR,
    others_color: str = OTHERS_COLOR,
) -> List[Artist]:
    """Draw one gene's raw, unnormalized predictions onto one axes.

    This is the panel body of :func:`plot_examples`, without the figure, the titles
    and the legend, so a composed figure can reuse it.

    Args:
        axes: Axes to draw onto.
        prepared: Table as returned by ``prepare_predictions``.
        gene_id: Gene to draw.
        reference_color: Colour of the optimization model's line.
        others_color: Colour of the other models' line and bands.

    Returns:
        Legend handles in drawing order.
    """
    other_columns = get_other_model_columns(prepared)
    gene_predictions = prepared[prepared["gene_id"] == gene_id].sort_values("mutation_count")
    spread = summarize_spread(gene_predictions, other_columns)
    return draw_front_with_spread(
        axes,
        gene_predictions["mutation_count"].to_numpy(),
        gene_predictions["original_fitness"].to_numpy(),
        spread,
        reference_label="optimization model",
        others_label=f"other models, median ({len(other_columns)} models)",
        reference_color=reference_color,
        others_color=others_color,
    )


def plot_pooled(prepared: pd.DataFrame, run_name: str, output_folder: str, output_format: str) -> str:
    """Plot all genes pooled, using the per gene normalized values.

    Args:
        prepared: Table as returned by ``prepare_predictions``.
        run_name: Name of the evolution run, used as the figure title.
        output_folder: Folder to write the figure to.
        output_format: File extension of the figure.

    Returns:
        Path to the written figure.
    """
    figure, axes = plt.subplots(figsize=(7.0, 4.5))
    legend_handles = draw_pooled_panel(axes, prepared)
    number_of_genes = prepared["gene_id"].nunique()
    axes.set_title(
        f"{run_name}\n{number_of_genes} genes, per gene normalized to the "
        "optimization model's range"
    )
    axes.legend(handles=legend_handles, loc="best", fontsize="small", frameon=False)
    figure.tight_layout()
    output_path = os.path.join(output_folder, f"{POOLED_FILE_NAME}.{output_format}")
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def compute_mean_normalized_gap(prepared: pd.DataFrame) -> pd.Series:
    """Measure per gene how far the other models sit from the optimization model.

    The value is the mean absolute vertical gap between the other models' median
    normalized curve and the optimization model's normalized fitness, averaged over all
    mutation counts. It is expressed in units of the range the optimization model
    achieved for that gene, because that is what the normalization divides by: 0 means
    the median traces the front exactly, 1 means it sits a full achieved range away
    from it on average. Larger values mean the other models track the front less well.

    Averaging over the whole front rather than only its endpoint is deliberate: the
    optimization model saturates well before the highest mutation count and the other
    models usually catch up eventually, so an endpoint value is close to zero for
    almost every gene and does not distinguish a model that follows the front from one
    that lags far behind it. The absolute difference makes the measure direction
    agnostic, so it applies to gain of function as well as loss of function runs.

    Args:
        prepared: Table as returned by ``prepare_predictions``.

    Returns:
        Series of gaps indexed by gene id, sorted ascending.
    """
    normalized_other_columns = [
        f"{NORMALIZED_PREFIX}{column}" for column in get_other_model_columns(prepared)
    ]
    gaps = prepared[["gene_id"]].copy()
    gaps["gap"] = (
        prepared[normalized_other_columns].median(axis=1) - prepared[NORMALIZED_FITNESS_COLUMN]
    ).abs()
    return gaps.groupby("gene_id")["gap"].mean().sort_values()


def select_example_genes(prepared: pd.DataFrame, number_of_genes: int) -> List[str]:
    """Pick example genes spanning the range of the mean normalized gap.

    Genes are taken at evenly spaced ranks of the gap, so the closest and the farthest
    gene are always among them and, for three genes, the median gene sits between them.
    This shows the full range rather than a typical case.

    Args:
        prepared: Table as returned by ``prepare_predictions``.
        number_of_genes: How many genes to pick.

    Returns:
        Gene ids ordered from the smallest to the largest gap.
    """
    gaps = compute_mean_normalized_gap(prepared)
    if len(gaps) <= number_of_genes:
        return list(gaps.index)
    positions = np.linspace(0, len(gaps) - 1, number_of_genes).round().astype(int)
    return [gaps.index[position] for position in positions]


def plot_examples(
    prepared: pd.DataFrame,
    run_name: str,
    output_folder: str,
    output_format: str,
    example_genes: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """Plot individual genes with raw, unnormalized predictions.

    Args:
        prepared: Table as returned by ``prepare_predictions``.
        run_name: Name of the evolution run, used in the figure title.
        output_folder: Folder to write the figure to.
        output_format: File extension of the figure.
        example_genes: Gene ids to plot. When None, genes spanning the range of the
            mean normalized gap are selected.

    Returns:
        The path to the written figure and the plotted gene ids.

    Raises:
        ValueError: If a requested gene id is not present in the table.
    """
    if example_genes is None:
        example_genes = select_example_genes(prepared, NUMBER_OF_EXAMPLE_GENES)
    missing = [gene for gene in example_genes if gene not in set(prepared["gene_id"])]
    if missing:
        raise ValueError(f"Requested example genes not present in the data: {missing}")

    gaps = compute_mean_normalized_gap(prepared)
    figure, axes_list = plt.subplots(
        1, len(example_genes), figsize=(4.0 * len(example_genes), 4.0), squeeze=False
    )
    legend_handles: List[Artist] = []
    for axes, gene_id in zip(axes_list[0], example_genes):
        legend_handles = draw_example_panel(axes, prepared, gene_id)
        axes.set_title(f"{gene_id}\n{GAP_LABEL} {gaps[gene_id]:.2f}", fontsize="small")
    axes_list[0][0].set_ylabel("prediction")
    axes_list[0][0].legend(handles=legend_handles, loc="best", fontsize="x-small", frameon=False)
    figure.suptitle(f"{run_name}: individual genes, raw predictions", fontsize="medium")
    figure.tight_layout()
    output_path = os.path.join(output_folder, f"{EXAMPLES_FILE_NAME}.{output_format}")
    figure.savefig(output_path)
    plt.close(figure)
    return output_path, list(example_genes)


def plot_csv(
    csv_path: str,
    output_format: str,
    example_genes: Optional[List[str]] = None,
    minimum_fitness_range: float = 0.0,
) -> Dict[str, str]:
    """Produce both figures and the mean normalized gap table for one re-evaluation CSV.

    Args:
        csv_path: Path to a CSV written by ``reevaluate.py``.
        output_format: File extension of the figures.
        example_genes: Gene ids for the example figure, or None to select genes
            spanning the range of the mean normalized gap.
        minimum_fitness_range: Minimum span of the optimization model's fitness for a
            gene to be included, see ``prepare_predictions``.

    Returns:
        Mapping of output name to the written path.
    """
    predictions = pd.read_csv(csv_path)
    # Tables written before gene id cleaning was introduced still carry the full
    # sequence name; cleaning is idempotent, so this is safe for newer tables too.
    predictions["gene_id"] = predictions["gene_id"].map(clean_gene_id)
    output_folder = os.path.dirname(os.path.abspath(csv_path))
    run_name = os.path.basename(output_folder)
    print(f"{run_name}: {predictions['gene_id'].nunique()} genes in the CSV")
    prepared = prepare_predictions(predictions, minimum_fitness_range)

    gap_path = os.path.join(output_folder, GAP_FILE_NAME)
    gaps = compute_mean_normalized_gap(prepared)
    gaps.rename(GAP_COLUMN).to_csv(gap_path)
    pooled_path = plot_pooled(prepared, run_name, output_folder, output_format)
    examples_path, plotted_genes = plot_examples(
        prepared, run_name, output_folder, output_format, example_genes
    )
    print(f"  {prepared['gene_id'].nunique()} genes plotted")
    print(f"  {report_expansion(prepared)}")
    print(
        f"  {GAP_LABEL}: median {gaps.median():.3f}, "
        f"range {gaps.min():.3f} to {gaps.max():.3f}"
    )
    print(f"  {gap_path}")
    print(f"  {pooled_path}")
    print(f"  {examples_path} ({', '.join(plotted_genes)})")
    return {
        POOLED_FILE_NAME: pooled_path,
        EXAMPLES_FILE_NAME: examples_path,
        GAP_FILE_NAME: gap_path,
    }


def report_expansion(prepared: pd.DataFrame) -> str:
    """Describe how much of the plotted grid was filled in rather than evaluated.

    Rows added by the expansion carry the sequence and all predictions of a lower
    mutation count, so a region where most genes are expanded shows a band that is
    carried forward rather than measured. This is reported so such a region cannot be
    mistaken for consensus between the models.

    Args:
        prepared: Table as returned by ``prepare_predictions``.

    Returns:
        A one line summary.
    """
    expanded_fraction = prepared[EXPANDED_COLUMN].mean()
    per_mutation_count = prepared.groupby("mutation_count")[EXPANDED_COLUMN].mean()
    mostly_expanded = per_mutation_count.index[per_mutation_count > 0.5]
    summary = f"{expanded_fraction:.1%} of the plotted rows are carried forward by the expansion"
    if len(mostly_expanded):
        summary += (
            f"; carried forward for the majority of genes from mutation count "
            f"{int(mostly_expanded.min())} on"
        )
    return summary


def parse_arguments() -> argparse.Namespace:
    """Parse the command line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        nargs="+",
        default=default_csv_paths(),
        help="Re-evaluation CSV files to plot; figures are written next to each CSV.",
    )
    parser.add_argument(
        "--example-genes",
        nargs="+",
        default=None,
        help="Gene ids for the example figure; by default the best, median and worst "
        "agreeing gene are shown.",
    )
    parser.add_argument(
        "--output-format",
        default="pdf",
        help="File extension of the written figures.",
    )
    parser.add_argument(
        "--min-fitness-range",
        type=float,
        default=0.0,
        help="Drop genes whose optimization model fitness spans less than this range "
        "across the Pareto front; the normalization divides by that range, so genes "
        "the optimization model could barely improve produce extreme normalized values.",
    )
    return parser.parse_args()


def main() -> None:
    """Plot every requested re-evaluation CSV."""
    arguments = parse_arguments()
    for csv_path in arguments.csv:
        plot_csv(
            csv_path,
            arguments.output_format,
            arguments.example_genes,
            arguments.min_fitness_range,
        )


if __name__ == "__main__":
    main()
