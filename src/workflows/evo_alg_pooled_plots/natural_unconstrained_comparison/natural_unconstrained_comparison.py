"""Compare final fitness of natural-only vs unconstrained evolutionary runs.

The same genes were optimized twice: once allowing all mutations
("unconstrained") and once allowing only naturally occurring mutations
("natural"). Restricting to natural mutations is expected to weaken the
optimization. This script tests that, per gene, the final fitness differs
between the two conditions.

Because the same gene appears in both conditions, the two final-fitness values
are paired. We therefore use a Wilcoxon signed-rank test on the per-gene paired
differences (natural - unconstrained), run separately for the maximization
(GOF) and minimization (LOF) gene sets.

The hypothesis that natural is weaker is directional and stated a priori, so the
test is one-sided, with the side chosen per gene set. "Weaker" means:
  * maximization (GOF): natural reaches a *lower* final fitness  -> diff < 0
    (one-sided alternative "less")
  * minimization (LOF): natural reaches a *higher* final fitness -> diff > 0
    (one-sided alternative "greater")

Each run directory holds one ``stats_*.json`` mapping gene id -> stats dict with
``start_fitness`` and ``final_fitness``. Gene ids carry timestamp suffixes that
differ between runs, so genes are matched by their core id (the first two
underscore-separated fields, e.g. ``5_AT5G13640``).

It is a one-off analysis script: run it directly.
"""

import glob
import json
import os
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy.stats import rankdata, wilcoxon  # noqa: E402

# (unconstrained run dir, natural-only run dir, direction, label)
GOF_UNCONSTRAINED = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/GOF_LOF/GOF/GOF_single"
GOF_NATURAL = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/GOF_LOF/GOF/GOF_single_natural"
LOF_UNCONSTRAINED = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/GOF_LOF/LOF/LOF_single"
LOF_NATURAL = "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset/GOF_LOF/LOF/LOF_single_natural"

# (name, one-sided alternative for "natural is weaker", log y-axis?, unconstrained dir, natural dir)
# "less": natural < unconstrained (maximization); "greater": natural > unconstrained (minimization).
# Log scale is used only for minimization, whose values span several orders of
# magnitude; maximization values cluster near 1.0 and read better on a linear axis.
COMPARISONS = [
    ("maximization", "less", False, GOF_UNCONSTRAINED, GOF_NATURAL),
    ("minimization", "greater", True, LOF_UNCONSTRAINED, LOF_NATURAL),
]

FINAL_FITNESS_KEY = "final_fitness"
START_FITNESS_KEY = "start_fitness"
OUTPUT_BASENAME = "natural_unconstrained_comparison"

FIGURE_FORMAT = "png"
CONDITION_ORDER = ["unconstrained", "natural"]
CONDITION_PALETTE = {"unconstrained": "#4878CF", "natural": "#D65F5F"}
# p-value thresholds (most stringent first) mapped to the star annotation.
STAR_THRESHOLDS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]


def core_gene_id(gene_id: str) -> str:
    """Return the core gene id (first two underscore-separated fields).

    Args:
        gene_id: Full gene id from a stats JSON key, e.g.
            ``5_AT5G13640_gene:4392936-4397541_251010_182339_304179``.

    Returns:
        The core id, e.g. ``5_AT5G13640``.
    """
    return "_".join(gene_id.split("_")[:2])


def load_final_fitness(run_dir: str) -> Dict[str, float]:
    """Load per-gene final fitness from a run's ``stats_*.json``.

    Args:
        run_dir: Run directory containing exactly one ``stats_*.json``.

    Returns:
        Mapping of core gene id to ``final_fitness``.

    Raises:
        FileNotFoundError: If there is not exactly one ``stats_*.json``.
    """
    matches = glob.glob(os.path.join(run_dir, "stats_*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one stats_*.json in {run_dir}, found {matches}"
        )
    with open(matches[0], "r", encoding="utf-8") as handle:
        stats = json.load(handle)
    return {core_gene_id(gene): entry[FINAL_FITNESS_KEY] for gene, entry in stats.items()}


def pair_final_fitness(unconstrained_dir: str, natural_dir: str) -> pd.DataFrame:
    """Pair per-gene final fitness across the two conditions by core gene id.

    Genes present in only one condition are dropped (inner join).

    Args:
        unconstrained_dir: Run directory allowing all mutations.
        natural_dir: Run directory allowing only natural mutations.

    Returns:
        DataFrame indexed by core gene id with ``unconstrained`` and
        ``natural`` final-fitness columns, sorted by gene id.
    """
    unconstrained = load_final_fitness(unconstrained_dir)
    natural = load_final_fitness(natural_dir)
    shared = sorted(set(unconstrained) & set(natural))
    return pd.DataFrame(
        {
            "unconstrained": [unconstrained[gene] for gene in shared],
            "natural": [natural[gene] for gene in shared],
        },
        index=pd.Index(shared, name="core_gene"),
    )


def rank_biserial_effect_size(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for a Wilcoxon signed-rank test.

    Zero differences are dropped (matching scipy's default ``zero_method``).
    The value is ``(T_plus - T_minus) / (T_plus + T_minus)`` where ``T_plus``
    and ``T_minus`` are the summed ranks of the positive and negative absolute
    differences. It ranges from -1 to 1; the sign follows the sign of the
    typical difference.

    Args:
        differences: Per-gene paired differences (natural - unconstrained).

    Returns:
        Rank-biserial correlation, or ``nan`` if all differences are zero.
    """
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return float("nan")
    ranks = rankdata(np.abs(nonzero))
    rank_sum_positive = ranks[nonzero > 0].sum()
    rank_sum_negative = ranks[nonzero < 0].sum()
    total = rank_sum_positive + rank_sum_negative
    return float((rank_sum_positive - rank_sum_negative) / total)


def wilcoxon_paired_test(paired: pd.DataFrame, alternative: str = "two-sided") -> Dict[str, float]:
    """Wilcoxon signed-rank test of natural vs unconstrained final fitness.

    Args:
        paired: DataFrame with ``unconstrained`` and ``natural`` columns, one
            row per paired gene (as returned by :func:`pair_final_fitness`).
        alternative: Sidedness passed to ``scipy.stats.wilcoxon``, applied to the
            paired differences (natural - unconstrained). Use ``"less"`` to test
            that natural is lower (maximization weaker), ``"greater"`` that
            natural is higher (minimization weaker), or ``"two-sided"``.

    Returns:
        Dict with the number of pairs, per-condition median final fitness,
        median paired difference (natural - unconstrained), the Wilcoxon
        statistic, p-value (for the requested sidedness), and rank-biserial
        effect size.
    """
    differences = (paired["natural"] - paired["unconstrained"]).to_numpy()
    statistic, p_value = wilcoxon(differences, alternative=alternative)
    return {
        "n_pairs": int(len(paired)),
        "median_unconstrained": float(paired["unconstrained"].median()),
        "median_natural": float(paired["natural"].median()),
        "median_difference": float(np.median(differences)),
        "wilcoxon_statistic": float(statistic),
        "p_value": float(p_value),
        "rank_biserial": rank_biserial_effect_size(differences),
    }


def p_value_to_stars(p_value: float) -> str:
    """Return the significance star string for a p-value (``"ns"`` if none)."""
    for threshold, stars in STAR_THRESHOLDS:
        if p_value < threshold:
            return stars
    return "ns"


def save_figure(fig: Figure, filename: str, output_dir: str, fmt: str = FIGURE_FORMAT) -> None:
    """Save ``fig`` to ``output_dir/filename.fmt`` at 150 dpi."""
    path = os.path.join(output_dir, f"{filename}.{fmt}")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved {path}")


def plot_paired_fitness(
    paired: pd.DataFrame,
    title: str,
    p_value: float,
    log_scale: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """Draw a paired box plot of final fitness with per-gene connecting lines.

    One box per condition (unconstrained, natural). Each gene's two values are
    joined by a faint grey line so the consistency of the shift is visible. A
    significance bar above the boxes is annotated with the one-sided p-value and
    its star level.

    Args:
        paired: DataFrame with ``unconstrained`` and ``natural`` columns, one row
            per gene (as returned by :func:`pair_final_fitness`).
        title: Axes title (e.g. the comparison name).
        p_value: One-sided Wilcoxon p-value to annotate.
        log_scale: If True, use a logarithmic y-axis. Requires all values > 0.
        ax: Axes to draw onto. When None (standalone), a figure is created and
            returned with the original explicit font sizes. When given, the plot
            is drawn onto ``ax`` and its parent figure is returned; inline font
            sizes are left to the active stylesheet.

    Returns:
        Matplotlib Figure with a single axes.
    """
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(4.5, 5.0))
    else:
        fig = ax.get_figure()
    x_positions = {"unconstrained": 0, "natural": 1}

    for _, row in paired.iterrows():
        ax.plot(
            [x_positions["unconstrained"], x_positions["natural"]],
            [row["unconstrained"], row["natural"]],
            color="grey",
            alpha=0.25,
            linewidth=0.6,
            zorder=1,
        )

    long_form = paired.melt(value_vars=CONDITION_ORDER, var_name="condition", value_name="final_fitness")
    sns.boxplot(
        data=long_form,
        x="condition",
        y="final_fitness",
        order=CONDITION_ORDER,
        hue="condition",
        palette=CONDITION_PALETTE,
        legend=False,
        showfliers=False,
        width=0.5,
        boxprops={"alpha": 0.7},
        ax=ax,
        zorder=2,
    )

    data_max = float(paired[CONDITION_ORDER].to_numpy().max())
    data_min = float(paired[CONDITION_ORDER].to_numpy().min())
    if log_scale:
        ax.set_yscale("log")
        bar_y, text_y, top = data_max * 1.6, data_max * 2.4, data_max * 7.0
    else:
        span = data_max - data_min
        bar_y, text_y, top = data_max + 0.04 * span, data_max + 0.07 * span, data_max + 0.22 * span

    # Significance bracket above the data, with a tick at each end.
    tick = (text_y - bar_y) * 0.4
    ax.plot([0, 0, 1, 1], [bar_y - tick, bar_y, bar_y, bar_y - tick], color="black", linewidth=1.0)
    significance_text = ax.text(
        0.5,
        text_y,
        f"{p_value_to_stars(p_value)}\np = {p_value:.2e}",
        ha="center",
        va="bottom",
    )
    ax.set_ylim(top=top)

    ax.set_xlabel("mutation set")
    ax.set_ylabel("final fitness")
    ax.set_title(title)
    if own_figure:
        significance_text.set_fontsize(9)
        fig.tight_layout()
    return fig


def main() -> None:
    """Run the paired test for both gene sets and save CSVs and paired plots."""
    output_folder = os.path.join(os.path.dirname(__file__), "results_comparison")
    os.makedirs(output_folder, exist_ok=True)
    summary_rows = []
    for name, alternative, log_scale, unconstrained_dir, natural_dir in COMPARISONS:
        paired = pair_final_fitness(unconstrained_dir, natural_dir)
        result = wilcoxon_paired_test(paired, alternative=alternative)
        summary_rows.append({"comparison": name, "alternative": alternative, **result})

        paired_path = os.path.join(output_folder, f"{OUTPUT_BASENAME}_{name}_paired.csv")
        paired.to_csv(paired_path)
        print(f"Saved {paired_path}")

        fig = plot_paired_fitness(paired, name, result["p_value"], log_scale=log_scale)
        save_figure(fig, f"{OUTPUT_BASENAME}_{name}", output_folder)
        plt.close(fig)

        print(
            f"{name} (alternative={alternative}): n={result['n_pairs']}, "
            f"median unconstrained={result['median_unconstrained']:.4f}, "
            f"median natural={result['median_natural']:.4f}, "
            f"p={result['p_value']:.3e}, r={result['rank_biserial']:.3f}"
        )

    summary = pd.DataFrame(summary_rows).set_index("comparison")
    summary_path = os.path.join(output_folder, f"{OUTPUT_BASENAME}_significance.csv")
    summary.to_csv(summary_path)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
