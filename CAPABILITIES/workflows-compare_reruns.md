# src/workflows/compare_reruns/

Compares two evolutionary-algorithm run folders on the genes they share: before/after deepCRE fitness, mutation count reached, and overlap of introduced mutations (position+base or position only), at either the optimized endpoint or the half-max point. Inputs are run folders with `<gene>/saved_populations/pareto_front.json`; outputs go to `<output_dir>/<comparison_point>[_position_only]/` as `per_gene.csv`, `summary.txt`, optional `excluded_genes.txt`, and three figures. Planning notes in `plan_half_max_and_common_count_overlap.md`.

## compare_runs.py  [script]
Run-vs-run consistency analysis. Run: `python src/workflows/compare_reruns/compare_runs.py --run-a DIR --run-b DIR --output-dir DIR [--format png] [--comparison-point endpoint|half_max] [--position-only]`. Imports `calculate_half_max_mutations` from `analysis.overview.simple_result_stats` and `get_data_at_mutation_count` from `workflows.candidate_selection`. Relies on Pareto ordering front[0] = optimized endpoint, front[-1] = 0-mutation reference.

- `COMPARISON_POINTS = ("endpoint", "half_max")`.
- `PER_GENE_COLUMNS` — gene, before, after_A/B, n_mutations_A/B, delta_after, delta_n_mutations, overlap_mutation_count, shared/a_only/b_only_mutations.
- `GeneComparisonSkipped(Exception)` — per-gene skip (common count missing on a front); distinct from ValueError (runs incomparable → abort).
- `extract_gene_key(folder_name: str) -> str` — `"{chr}_{AGI}"` from a gene folder name (first two `_`-fields).
- `discover_gene_fronts(run_dir: str) -> dict[str, str]` — gene key → `saved_populations/pareto_front.json` path; raises on missing dir or duplicate keys.
- `load_pareto_front(pareto_path: str) -> list` — `json.load` of a pareto_front.json (list of `[sequence, fitness, mutation_count]`).
- `mutation_set(reference_sequence: str, optimized_sequence: str, position_only: bool = False) -> set[tuple[int, str]] | set[int]` — substitutions vs reference as {(0-based pos, new base)} or {pos}; requires equal lengths.
- `resolve_comparison_members(front_a, front_b, comparison_point: str) -> tuple[reported_a, reported_b, overlap_a, overlap_b, overlap_count]` — endpoint: report front[0], overlap at the largest mutation count present on both fronts; half_max: report and overlap at min of the two half-max counts.
- `compare_single_gene(gene_key: str, front_a, front_b, fitness_tolerance: float = 1e-6, comparison_point: str = "endpoint", position_only: bool = False) -> dict` — validates identical reference sequence and before-fitness (ValueError otherwise), returns one `PER_GENE_COLUMNS` record.
- `build_per_gene_table(fronts_a: dict, fronts_b: dict, fitness_tolerance=1e-6, comparison_point="endpoint", position_only=False) -> tuple[pd.DataFrame, dict[str, int], list[tuple[str, str]]]` — per-gene table over shared genes, gene-overlap counts (both/a_only/b_only), skipped genes with reasons.
- `compute_summary(per_gene, overlap_counts, label_a, label_b, comparison_point="endpoint", position_only=False, n_excluded=0) -> dict` — means/medians/deltas, paired Wilcoxon p (after fitness, mutation count; None if all differences zero), pooled shared/A-only/B-only counts, mean per-gene shared fraction.
- `format_summary(summary: dict) -> str` — text report.
- `save_figure(fig: Figure, filename: str, output_dir: str, fmt: str = "png") -> None` — save `output_dir/filename.fmt` (dpi 150, tight) and close.
- `plot_paired_scatter(per_gene, x_column, y_column, xlabel, ylabel, title, ax=None) -> Figure` — ax-injectable A-vs-B scatter with y = x line.
- `plot_mutation_overlap(per_gene, position_only=False, ax=None) -> Figure` — ax-injectable histogram of per-gene shared fraction shared/(shared+a_only+b_only).
- `render_all_figures(per_gene, label_a, label_b, output_dir, fmt="png", comparison_point="endpoint", position_only=False) -> None` — writes `scatter_after`, `scatter_n_mutations`, `mutation_overlap`.
- `run_comparison(run_a: str, run_b: str, output_dir: str, fmt="png", fitness_tolerance=1e-6, comparison_point="endpoint", position_only=False) -> None` — full pipeline: load, table, summary, CSV/txt/figures.
