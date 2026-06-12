import numpy as np
from pathlib import Path
from typing import List, Optional, cast
from matplotlib.figure import Figure
from tensorflow.keras.models import load_model  # type: ignore
from pyfaidx import Fasta
import matplotlib.axes
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import scipy.stats as stats
from evolution.sequences import one_hot_encode


DEEPCRE_PATH = "/home/gernot/Code/PhD_Code/Evolution/models/Atha_S0X0.75dP7K25g_NC_003075.7_ssr_train_models_250705_211854.h5"
CONSTRUCT_INSERT_PATH = "src/workflows/overlap_analysis/correct_construct/construct_inserts.fa"
PREDICTIONS_CACHE_PATH = "src/workflows/overlap_analysis/correct_construct/predictions_cache.csv"
STARRSEQ_PATH = "src/workflows/overlap_analysis/data/plantstarr-seq_main_light_and_dark_simon_gernot.csv"
OUTPUT_DIR = "src/workflows/overlap_analysis/correct_construct/plots"
FMT = "png"

_BINDING_PALETTE = {"binding": "#2196F3", "non_binding": "#FF5722"}
_TF_PALETTE = {"WRKY": "#2196F3", "bHLH": "#FF5722"}
_CONDITION_MARKERS = {"Light": "o", "Dark": "^"}

def predict_sequences(cache_path: str = PREDICTIONS_CACHE_PATH) -> pd.DataFrame:
    """Run deepCRE on all construct inserts and return per-sequence predictions.

    Results are written to cache_path on first run. Subsequent calls load from
    the cache instead of re-running the model.

    Args:
        cache_path: CSV path to read from / write to.

    Returns:
        DataFrame with columns: id, barcode, prediction.
    """
    if Path(cache_path).exists():
        return pd.read_csv(cache_path)

    inserts_fasta = Fasta(CONSTRUCT_INSERT_PATH)
    model = load_model(DEEPCRE_PATH)
    keys = []
    encoded = []
    for key in inserts_fasta.keys():
        seq = str(inserts_fasta[key])
        encoded_seq = one_hot_encode(seq)
        keys.append(key)
        encoded.append(encoded_seq)
    encoded_array = np.array(encoded)
    predictions = model.predict(encoded_array).flatten().tolist()
    results = pd.DataFrame({"keys": keys, "prediction": predictions})
    results["barcode"] = results["keys"].apply(lambda x: int(x.split("_")[1]))
    results["id"] = results["keys"].apply(lambda x: "_".join(x.split("_")[2:]))
    results = pd.DataFrame(results[["id", "barcode", "prediction"]])
    results.to_csv(cache_path, index=False)
    return results

def _parse_binding_category(id_str: str) -> str:
    """Parse binding category from a sequence ID.

    Args:
        id_str: Sequence identifier string containing 'non_binding' or 'binding'.

    Returns:
        'non_binding' if the sequence is non-binding, 'binding' otherwise
        (covers both 'binding' and 'reference_binding').
    """
    suffix = "_".join(id_str.split("_")[1:])
    return "non_binding" if "non_binding" in suffix else "binding"


def load_starrseq_data(path: str) -> pd.DataFrame:
    """Load and filter STAR-seq data for bHLH and WRKY sequences.

    Args:
        path: Path to the STAR-seq CSV file.

    Returns:
        DataFrame with columns: id, condition, enrichment, tf_family, binding_category.
    """
    df = pd.read_csv(path)
    df = df[df["id"].str.startswith("bHLH_") | df["id"].str.startswith("WRKY_")]
    df = df[["id", "condition", "enrichment"]].copy()
    df["tf_family"] = df["id"].str.split("_").str[0]
    df["binding_category"] = df["id"].apply(_parse_binding_category)
    return df


def average_barcode_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Average deepCRE predictions across both barcodes for each sequence.

    Args:
        predictions_df: DataFrame with columns: id, barcode, prediction.

    Returns:
        DataFrame with columns: id, prediction (mean across barcodes).
    """
    return predictions_df.groupby("id")[["prediction"]].mean().reset_index()


def prepare_barcode_comparison_data(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot predictions to compare barcode 1 vs barcode 2 for each sequence.

    Args:
        predictions_df: DataFrame with columns: id, barcode, prediction.

    Returns:
        DataFrame with columns: id, barcode_1, barcode_2, tf_family.
    """
    pivoted = predictions_df.pivot(
        index="id", columns="barcode", values="prediction"
    ).reset_index()
    pivoted.columns = ["id", "barcode_1", "barcode_2"]
    pivoted["tf_family"] = pivoted["id"].str.split("_").str[0]
    return pivoted


def prepare_correlation_data(
    predictions_df: pd.DataFrame, starrseq_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge averaged predictions with STAR-seq enrichment data.

    Args:
        predictions_df: DataFrame with columns: id, barcode, prediction.
        starrseq_df: DataFrame from load_starrseq_data.

    Returns:
        Merged DataFrame with columns: id, prediction, condition, enrichment,
        tf_family, binding_category.
    """
    avg_preds = average_barcode_predictions(predictions_df)
    return starrseq_df.merge(avg_preds, on="id", how="inner")


def save_figure(fig: Figure, filename: str, output_dir: str, fmt: str = "png") -> None:
    """Save a matplotlib figure and close it.

    Args:
        fig: Figure to save.
        filename: Output filename without extension.
        output_dir: Directory to write the file into (created if missing).
        fmt: File format, e.g. 'png', 'pdf', 'svg'.
    """
    path = Path(output_dir) / f"{filename}.{fmt}"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _format_p_value(p: float) -> str:
    """Format a p-value for display.

    Args:
        p: Raw p-value from a statistical test.

    Returns:
        'p < 0.001' when the value is very small, otherwise 'p = X.XXX'.
    """
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _draw_regression(
    ax: matplotlib.axes.Axes,
    x: pd.Series,
    y: pd.Series,
    color: str,
    label: Optional[str] = None,
) -> None:
    """Fit a linear regression to x/y and draw the line.

    When label is provided, r and p are appended and the line is included in
    the legend. Skips silently when fewer than two points are available.

    Args:
        ax: Axes to draw on.
        x: Predictor values (deepCRE prediction).
        y: Response values (STAR-seq enrichment).
        color: Line color.
        label: Base legend label; r and p are appended when given.
    """
    if len(x) < 2:
        return
    _lr = stats.linregress(x, y)
    slope, intercept, r, p = float(_lr[0]), float(_lr[1]), float(_lr[2]), float(_lr[3])  # type: ignore[arg-type]
    x_range = np.linspace(x.min(), x.max(), 100)
    line_label = f"{label} (r={r:.2f}, {_format_p_value(p)})" if label else None
    ax.plot(x_range, slope * x_range + intercept, color=color, linewidth=1.5, label=line_label)


def plot_barcode_comparison(
    data: pd.DataFrame, output_dir: str, fmt: str = "png"
) -> None:
    """Scatter plot of barcode 1 vs barcode 2 predictions to assess barcode influence.

    Args:
        data: DataFrame from prepare_barcode_comparison_data.
        output_dir: Directory to save the figure.
        fmt: Output format (e.g., 'png', 'pdf', 'svg').
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 6))

    for tf, group in data.groupby("tf_family"):
        ax.scatter(
            group["barcode_1"],
            group["barcode_2"],
            label=tf,
            alpha=0.4,
            s=15,
            color=_TF_PALETTE[tf],
        )

    lim_min = min(data["barcode_1"].min(), data["barcode_2"].min())
    lim_max = max(data["barcode_1"].max(), data["barcode_2"].max())
    ax.plot(
        [lim_min, lim_max], [lim_min, lim_max],
        "k--", linewidth=1, alpha=0.6, label="y = x",
    )

    ax.set_xlabel("deepCRE prediction (barcode 1)")
    ax.set_ylabel("deepCRE prediction (barcode 2)")
    ax.set_title("Barcode influence on deepCRE predictions")
    ax.legend(title="TF family")
    save_figure(fig, "barcode_comparison", output_dir, fmt)


def _resolve_condition_params(condition: Optional[str]) -> tuple:
    """Return (plot_label, filename_suffix) for a given condition filter.

    Args:
        condition: 'Light', 'Dark', or None for both conditions combined.

    Returns:
        Tuple of (condition_label, filename).
    """
    if condition is not None:
        return condition, f"prediction_vs_enrichment_{condition.lower()}"
    return "Light + Dark", "prediction_vs_enrichment_combined"


def _populate_single_condition_panel(
    ax: matplotlib.axes.Axes,
    subset: pd.DataFrame,
) -> None:
    """Draw scatter points and regression lines for a single-condition panel.

    Points are colored by binding_category; one regression line per category.

    Args:
        ax: Axes to draw on.
        subset: Rows for one TF family and one condition.
    """
    for cat, group in subset.groupby("binding_category"):
        ax.scatter(
            group["prediction"], group["enrichment"],
            color=_BINDING_PALETTE[cat], label=cat, alpha=0.4, s=15,
        )
        _draw_regression(
            ax, group["prediction"], group["enrichment"],
            _BINDING_PALETTE[cat], label=f"{cat} fit",
        )


def _populate_combined_condition_panel(
    ax: matplotlib.axes.Axes,
    subset: pd.DataFrame,
) -> None:
    """Draw scatter points and regression lines for a combined Light+Dark panel.

    Color encodes binding_category; marker shape encodes condition.
    Regression lines are drawn per binding_category pooled across conditions.

    Args:
        ax: Axes to draw on.
        subset: Rows for one TF family across all conditions.
    """
    for cond, cond_group in subset.groupby("condition"):
        for cat, group in cond_group.groupby("binding_category"):
            ax.scatter(
                group["prediction"], group["enrichment"],
                color=_BINDING_PALETTE[cat],
                marker=_CONDITION_MARKERS[cond],  # type: ignore[arg-type]
                alpha=0.4, s=15,
                label=f"{cat} ({cond})",
            )
    for cat, group in subset.groupby("binding_category"):
        _draw_regression(
            ax, group["prediction"], group["enrichment"],
            _BINDING_PALETTE[cat], label=f"{cat} fit",
        )


def plot_prediction_vs_enrichment(
    data: pd.DataFrame,
    condition: Optional[str],
    output_dir: str,
    fmt: str = "png",
) -> None:
    """Scatter plot of deepCRE prediction (x) vs STAR-seq enrichment (y).

    One panel per TF family. When condition is None both Light and Dark are
    shown together: color encodes binding_category and marker encodes condition.

    Args:
        data: DataFrame from prepare_correlation_data.
        condition: 'Light', 'Dark', or None for all conditions combined.
        output_dir: Directory to save the figure.
        fmt: Output format (e.g., 'png', 'pdf', 'svg').
    """
    sns.set_theme(style="whitegrid")
    condition_label, filename = _resolve_condition_params(condition)
    plot_data = data if condition is None else data[data["condition"] == condition]

    tf_families = sorted(plot_data["tf_family"].unique())
    fig, axes_array = plt.subplots(
        1, len(tf_families), figsize=(6 * len(tf_families), 5), sharey=True, squeeze=False
    )
    axes_list: List[matplotlib.axes.Axes] = axes_array[0].tolist()

    for ax, tf in zip(axes_list, tf_families):
        subset = cast(pd.DataFrame, plot_data[plot_data["tf_family"] == tf].copy())
        if condition is None:
            _populate_combined_condition_panel(ax, subset)
        else:
            _populate_single_condition_panel(ax, subset)
        ax.set_xlabel("deepCRE prediction")
        ax.set_title(tf)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(title="Binding status", fontsize=8, title_fontsize=8, loc="lower right")

    axes_list[0].set_ylabel("STAR-seq enrichment")  # type: ignore[union-attr]
    fig.suptitle(
        f"deepCRE prediction vs. STAR-seq enrichment ({condition_label})",
        y=1.02,
    )
    save_figure(fig, filename, output_dir, fmt)


def plot_overall_correlation(
    data: pd.DataFrame, output_dir: str, fmt: str = "png"
) -> None:
    """Scatter all deepCRE predictions against all STAR-seq enrichment values.

    No subsetting by TF family, binding category, or condition. A single
    regression line is drawn and annotated with slope, r, and p-value.

    Args:
        data: DataFrame from prepare_correlation_data.
        output_dir: Directory to save the figure.
        fmt: Output format (e.g., 'png', 'pdf', 'svg').
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(data["prediction"], data["enrichment"], alpha=0.3, s=10, color="#555555")

    _lr = stats.linregress(data["prediction"], data["enrichment"])
    slope, intercept, r, p = float(_lr[0]), float(_lr[1]), float(_lr[2]), float(_lr[3])  # type: ignore[arg-type]
    x_range = np.linspace(data["prediction"].min(), data["prediction"].max(), 100)
    ax.plot(
        x_range, slope * x_range + intercept,
        color="#E53935", linewidth=1.5,
        label=f"slope={slope:.3f}, r={r:.2f}, {_format_p_value(p)}",
    )

    ax.set_xlabel("deepCRE prediction")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("STAR-seq enrichment")
    ax.set_title("Overall deepCRE prediction vs. STAR-seq enrichment")
    ax.legend(fontsize=9)
    save_figure(fig, "prediction_vs_enrichment_overall", output_dir, fmt)


if __name__ == "__main__":
    predictions_df = predict_sequences()
    print(predictions_df.head())

    starrseq_df = load_starrseq_data(STARRSEQ_PATH)
    barcode_data = prepare_barcode_comparison_data(predictions_df)
    correlation_data = prepare_correlation_data(predictions_df, starrseq_df)

    plot_barcode_comparison(barcode_data, OUTPUT_DIR, FMT)
    plot_overall_correlation(correlation_data, OUTPUT_DIR, FMT)
    for cond in [None, "Light", "Dark"]:
        plot_prediction_vs_enrichment(correlation_data, cond, OUTPUT_DIR, FMT)