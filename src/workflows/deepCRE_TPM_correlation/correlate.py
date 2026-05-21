"""Analyze correlation between deepCRE model predictions and RNA-seq TPM measurements."""

import os
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load TPM measurements and model predictions.

    Returns:
        Tuple of (tpm_data, pred_data) DataFrames indexed by gene_id.
    """
    tpm_data = pd.read_csv("src/workflows/deepCRE_TPM_correlation/arabidopsis_leaf_counts.csv", index_col="gene_id")
    pred_data = pd.read_csv("src/workflows/deepCRE_TPM_correlation/Atha_S0X0.75dP7K25g_NC_003075.7_ssr_train_models_250705_211854_deepcre_crosspredict_arabidopsis_full_ara_prediction_260521_112603.csv")
    pred_data.rename(columns={"genes": "gene_id"}, inplace=True)
    pred_data.set_index("gene_id", inplace=True)
    return tpm_data, pred_data


def merge_data(tpm_data: pd.DataFrame, pred_data: pd.DataFrame) -> pd.DataFrame:
    """Merge TPM and prediction data on gene ID."""
    return tpm_data[["logMaxTPM"]].join(pred_data, how="inner")


def analyze_model(
    merged: pd.DataFrame, model_col: str
) -> dict:
    """Perform regression and correlation analysis for a single model.

    Args:
        merged: DataFrame with model predictions and logMaxTPM.
        model_col: Name of the model prediction column.

    Returns:
        Dictionary with regression/correlation results and data.
    """
    data = merged[[model_col, "logMaxTPM"]].dropna()
    x = data[model_col].values
    y = data["logMaxTPM"].values

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    corr_coef, corr_pval = pearsonr(x, y)

    return {
        "data": data,
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "corr_coef": corr_coef,
        "corr_pval": corr_pval,
    }

def plot_single_dataset(results: Dict, model_col: str, ax: plt.Axes) -> None:
    data = results[model_col]["data"]
    x = data[model_col].values
    y = data["logMaxTPM"].values

    ax.scatter(x, y, alpha=0.1, s=30, color="blue")

    x_line = np.array([x.min(), x.max()])
    y_line = (results[model_col]["slope"] * x_line + results[model_col]["intercept"])
    ax.plot(x_line, y_line, "r-", linewidth=2, label="Linear fit")

def create_plots(results: dict, model_cols: List[str]) -> None:
    """Create and save separate scatter plots with regression lines for each model."""
    #copy model_cols
    local_model_cols = model_cols.copy()
    while True:
        if not local_model_cols:
            break
        model_col = local_model_cols.pop(0)
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 6))

        plot_single_dataset(results, model_col, ax)
        high_low = False
        if model_col.endswith("_low"):
            other_model_col = model_col.replace("_low", "_high")
            high_low = True
        elif model_col.endswith("_high"):
            other_model_col = model_col.replace("_high", "_low")
            high_low = True
        else:
            other_model_col = None
        
        if other_model_col is not None and other_model_col in results:
            local_model_cols.remove(other_model_col)
            plot_single_dataset(results=results, model_col=other_model_col, ax=ax)

        title = "msr" if "_M0X0." in model_col else "ssr"
        title = title + " (only high and low predictions)" if high_low else title + " (all predictions)"
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Model Prediction")
        ax.set_ylabel("logMaxTPM")
        plt.tight_layout()
        filename = f"deepcre_tpm_correlation_model_{model_col}.png"
        out_path = os.path.join("src/workflows/deepCRE_TPM_correlation", filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {filename}")
        plt.close(fig)


def analyze_subset(results: dict, model_col: str, subset_name: str, filter_low: bool = False) -> dict:
    """Analyze a subset of predictions."""
    if subset_name == "low":
        pred_min, pred_max = 0, 0.4
    elif subset_name == "high":
        pred_min, pred_max = 0.6, 1.0
    else:
        raise ValueError("subset_name must be 'low' or 'high'")

    data = results[model_col]["data"]
    subset = data[(data[model_col] >= pred_min) & (data[model_col] <= pred_max)]
    subset = subset[subset["logMaxTPM"] > 0.2]
    subset = subset.rename(columns={model_col: f"{model_col}_{subset_name}"})

    if len(subset) < 2:
        return {
            "model": model_col + f"_{subset_name}",
            "n_genes": len(subset),
            "slope": None,
            "intercept": None,
            "r_value": None,
            "p_value": None,
            "corr_coef": None,
            "corr_pval": None,
        }

    return analyze_model(subset, f"{model_col}_{subset_name}")


def save_summary(results: dict, model_cols: List[str]) -> None:
    """Save correlation analysis results to CSV."""
    summary_df = pd.DataFrame({
        "Model": model_cols,
        "N_genes": [len(results[m]["data"]) for m in model_cols],
        "Slope": [results[m]["slope"] for m in model_cols],
        "Intercept": [results[m]["intercept"] for m in model_cols],
        "R_value": [results[m]["r_value"] for m in model_cols],
        "P_value_regression": [results[m]["p_value"] for m in model_cols],
        "Correlation_coefficient": [results[m]["corr_coef"] for m in model_cols],
        "P_value_correlation": [results[m]["corr_pval"] for m in model_cols],
    })

    summary_df.to_csv("src/workflows/deepCRE_TPM_correlation/correlation_analysis_summary.csv", index=False)


def main() -> None:
    """Run correlation analysis for both models."""
    tpm_data, pred_data = load_data()
    merged = merge_data(tpm_data, pred_data)
    model_cols = pred_data.columns[:2].tolist()

    print(f"Merged data shape: {merged.shape}")
    print(f"Model columns: {model_cols}\n")

    results = {col: analyze_model(merged, col) for col in model_cols}


    for model_col in model_cols:
        low_pred = analyze_subset(results, model_col, "low")
        high_pred = analyze_subset(results, model_col, "high")
        results[f"{model_col}_low"] = low_pred
        results[f"{model_col}_high"] = high_pred

    model_cols = [[col, f"{col}_low", f"{col}_high"] for col in model_cols]
    model_cols = [item for sublist in model_cols for item in sublist]
    save_summary(results, model_cols)
    create_plots(results, model_cols)


if __name__ == "__main__":
    main()
