"""Compare probe-run fitness at generation 1000/2000 and 20/max mutations per gene."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROBE_DIR = Path("/home/gernot/ARCitect/ARCs/dream/assays/Evolution_runs/dataset/starrseq_v2/probe")
OUTPUT_DIR = Path(__file__).parent / "probe_analysis"
MAX_MUTATIONS = 60
# limit 0 is the unmutated reference sequence, used as the zero point of the normalization
MUTATION_LIMITS = [0, 20, MAX_MUTATIONS]
# pareto_front.json is the final front (gen 2000); identical to pareto_front_gen_01999.json where both exist
FRONT_FILES = {1000: "pareto_front_gen_01000.json", 2000: "pareto_front.json"}


def best_fitness(front: list, mutation_limit: int, maximize: bool) -> float:
    """Best fitness among front entries with at most `mutation_limit` mutations."""
    fitnesses = [fitness for _, fitness, mutations in front if mutations <= mutation_limit]
    return max(fitnesses) if maximize else min(fitnesses)


def collect_run(run_dir: Path, maximize: bool) -> list:
    """One row per (generation, mutation limit); fitness is NaN if the checkpoint is missing."""
    gene = json.loads((run_dir / "parameters.json").read_text())["sequence_name"]
    rows = []
    for generation, file_name in FRONT_FILES.items():
        front_path = run_dir / "saved_populations" / file_name
        front = json.loads(front_path.read_text()) if front_path.exists() else None
        for mutation_limit in MUTATION_LIMITS:
            fitness = best_fitness(front, mutation_limit, maximize) if front else float("nan")
            rows.append({"gene": gene, "generation": generation, "mutation_limit": mutation_limit, "fitness": fitness})
    return rows


def normalize_per_gene(results: pd.DataFrame) -> pd.DataFrame:
    """Rescale each gene's fitness gain: 0 = unmutated, 1 = gen 2000 with max mutations; drops limit-0 rows."""
    final_front = results[results["generation"] == 2000]
    unmutated = results["gene"].map(final_front[final_front["mutation_limit"] == 0].set_index("gene")["fitness"])
    best = results["gene"].map(final_front[final_front["mutation_limit"] == MAX_MUTATIONS].set_index("gene")["fitness"])
    normalized = results.assign(fitness=(results["fitness"] - unmutated) / (best - unmutated))
    return normalized[normalized["mutation_limit"] > 0]


def plot_direction(results: pd.DataFrame, direction: str, output_path: Path) -> None:
    """Barplot of mean fitness over genes (± SD), one bar per (generation, mutation limit)."""
    summary = results.groupby(["generation", "mutation_limit"])["fitness"].agg(["mean", "std", "count"])
    number_of_genes = results["gene"].nunique()
    labels = [
        f"gen {generation}\n≤{limit} mut\nn={count}" + (f"\n({number_of_genes - count} missing)" if count < number_of_genes else "")
        for (generation, limit), count in summary["count"].items()
    ]
    _, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, summary["mean"], yerr=summary["std"], capsize=5)
    ax.set_ylabel(f"relative fitness gain\n(0 = unmutated, 1 = gen 2000, ≤{MAX_MUTATIONS} mut)\nmean ± SD over genes")
    ax.set_title(f"probe runs: {direction}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for direction in ["maximize", "minimize"]:
        (direction_dir,) = PROBE_DIR.glob(f"{direction}_*")
        rows = []
        for run_dir in sorted(path for path in direction_dir.iterdir() if path.is_dir()):
            rows += collect_run(run_dir, maximize=direction == "maximize")
        results = pd.DataFrame(rows)
        results.to_csv(OUTPUT_DIR / f"probe_fitness_{direction}.csv", index=False)
        plot_direction(normalize_per_gene(results), direction, OUTPUT_DIR / f"probe_fitness_{direction}.png")


if __name__ == "__main__":
    main()
