# src/analysis/overview/

Run-level overview statistics of evolutionary runs, read from per-gene `saved_populations/pareto_front.json` (a list of [sequence, fitness, n_mutations], stored order front[0] = optimized endpoint, front[-1] = reference). It computes start/final fitness, max mutations, mutations needed for the half-max effect, average normalized pareto fronts and convergence loss over generations. It also compares methods on the same genes: front areas and ranks, diversity/conservation, and PCA of mutation vectors. Outputs are JSON/txt stats and plots; most plotting functions accept `ax=` and save only when `ax` is None.

## simple_result_stats.py  [library]
Per-run stats and plots; imported by paper_plots, candidate_selection, adversarial, compare_reruns and compare_methods. CLI: `python -m analysis.overview.simple_result_stats -r <run> -n <name> -o <out> [--all | --summary --fitness_plot --mutations_plot --plot_random_pareto --plot_average_pareto --plot_average_loss --plot_half_max_mutations --plot_half_max_mutations_hist] [-s stats.json] [-m 90] [-l 1999] [-f png] [--no_titles] [--overwrite]`.

- `get_pareto_front(gene_path: str) -> Optional[List[Tuple[str, float, int]]]` — loads `<gene>/saved_populations/pareto_front.json`; None if missing.
- `add_basic_stats(stats, fitnesses, num_mutations, gene) -> None` — in place: final_fitness = fitnesses[0], start_fitness = fitnesses[-1], max_mutations, origin_chromosome = gene.split("_")[0].
- `get_min_mutation_count_for_fitness(pareto_front, target_fitness: float) -> float` — fewest mutations whose fitness reaches the target; assumes the stored order (optimized → reference) and a gap-free monotone front; handles both optimization directions.
- `calculate_half_max_mutations(pareto_front) -> int` — mutations needed to reach the midpoint of front[0] and front[-1] fitness.
- `get_stats_per_gene(results_folder: str, name: str, output_folder: str = ".", overwrite: bool = False) -> Dict[str, Dict[str, Any]]` — the stats above plus `num_mutations_half_max_effect` for every gene dir; caches to `stats_<name>.json` and reuses that file unless overwrite.
- `summary_stat_calculation(stats) -> dict` — mean/std of final/start fitness, max_mutations and half-max mutations.
- `split_stats(stats) -> (main_chrom_stats, organelle_scaffold_stats)` — split by whether origin_chromosome is numeric.
- `summarize_stats(stats, name, output_folder, group_origin=True) -> str` — writes `summary_<name>.txt` (plus per-group summaries).
- `visualize_start_vs_max_fitness(stats, name, output_format, output_folder=".", titles=True, ax=None) -> None` — scatter of start vs final fitness, main chromosome (blue) vs organelle/scaffold (red).
- `visualize_start_vs_max_fitness_by_mutations(stats, name, output_format, output_folder=".", titles=True) -> None` — calls the draw_ function twice (absolute and relative).
- `fit_logit_linear(x_values: np.ndarray, y_values: np.ndarray, initial_guess_epsilon: float = 1e-6) -> Tuple[float, float, float]` — fits y = sigmoid(slope·x + intercept) by nonlinear least squares on the (0,1) scale (scipy curve_fit); the initial guess comes from a logit-space polyfit, which is also the fallback if the fit does not converge. Returns (slope, intercept, R² on the original scale).
- `draw_visualize_start_vs_max_fitness_by_mutations(stats, name, output_format, relative=False, output_folder=".", titles=True, ax=None, add_colorbar=True, vmin=None, vmax=None, mean_start_line=False, fit_line=False, mean_start_line_color="0.5", fit_line_color="black") -> PathCollection` — start vs final fitness scatter, coloured (magma) by half-max mutations (absolute, or divided by max_mutations); optional dashed mean-start line and logit-linear fit curve. Returns the mappable for shared colorbars.
- `plot_pareto_front(pareto_path: str, out_path: str, titles=True, ax=None) -> None` — fitness vs n_mutations scatter for one front JSON.
- `show_random_fronts(results_folder, output_format, num_samples=4, output_folder=".", titles=True)` — plot_pareto_front for randomly chosen genes (np.random, unseeded).
- `deduplicate_pareto_front(pareto_front) -> list` — sorts ASCENDING by mutation count (so the reference comes first, reversing the stored order) and keeps the first entry per count.
- `expand_pareto_front(pareto_front, max_number_mutation: int) -> list` — deduplicates, then fills to exactly max+1 entries (0..max mutations) by carrying the previous fitness forward with sequence ""; counts above max are dropped. Result is ascending in mutations.
- `normalize_front(pareto_front) -> list` — min-max scales fitness to [0,1]; ValueError if fitness is constant.
- `show_average_pareto_front(results_folder, output_format, output_folder=".", max_number_mutation=90, titles=True, ax=None, color='#1f77b4', error_style='bars') -> None` — expands and normalizes each gene's front, then plots mean ± std per mutation count (`'bars'` or `'band'`).
- `calculate_loss_pareto_front(current_front, target_front, max_number_mutation) -> float` — |Σ(target − current)| over the expanded fronts (the absolute value of the signed sum, not a sum of absolute differences).
- `calculate_loss_over_generations_single_gene(results_folder, gene, max_number_mutation, last_generation) -> Dict[int, float]` — loss of each `pareto_front_gen_*.json` against the final front; the final generation is set to 0.
- `calculate_loss_over_generations(results_folder, max_number_mutation, last_generation) -> Dict[str, Dict[int, float]]` — the above for all genes; a gene that errors is printed and skipped.
- `join_losses_for_visualization(loss_data) -> (generations, mean_losses, std_losses)` — pools per-generation losses across genes.
- `plot_loss_over_generations(results_folder, name, max_number_mutation, output_format, last_generation, output_folder=".", titles=True, ax=None) -> None` — mean ± std loss vs generation.
- `plot_half_max_mutations_vs_initial_fitness(stats, name, output_format, output_folder=".", titles=True, ax=None) -> None` — scatter.
- `distribution_half_max_mutations(stats_path: str) -> None` — prints the frequency and cumulative % of half-max mutation counts.
- `hist_half_max_mutations(stats, name, output_format, output_folder=".", titles=True, ax=None, color=None, max_mutations_shown=None) -> None` — histogram with integer bins; optional overflow bin labelled "≥N".

## compare_methods.py  [script]
Compares several methods (runs) on the same genes. Requires scikit-learn. `python -m analysis.overview.compare_methods -r <run1> <run2> -me <name1> <name2> -o <out> [-mu <mut_json1> <mut_json2>] [--final|--diversity|--mutation_pca|--progress|--all] [-mm 90] [-mb 100] [-mp 3000] [-fmt png]`.

- `check_genes_present(gene_paths, methods) -> None` — ValueError if any gene lacks a method.
- `get_gene_paths(gene_folder_paths: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]` — gene name (first two "_" tokens of the folder name) → method → pareto_front.json path; FileNotFoundError if a file is missing.
- `calculate_differences_between_fronts(fronts) -> Dict[str, Dict[str, float]]` — "area" = Σ fitness over the expanded front; reference = the best area (direction inferred from the first front). Returns summed_diff (|area difference|) and abs_diff (Σ|pointwise difference|) per method.
- `add_normalized_fronts(fronts, normalized_fronts_all) -> None` — in place: min-max normalizes all methods' fronts for one gene jointly and appends (fitness, mutations) lists.
- `get_plot_vals_normalized_fronts(normalized_fronts) -> (mean_mutations, mean_fitness, std_fitness)`.
- `plot_normalized_fronts(normalized_fronts, output_dir, output_format, ax=None) -> None` — averaged normalized fronts per method; in standalone mode it also plots the differences (that file name is hardcoded `.png`).
- `plot_interesting_pareto_fronts_values(fronts, gene_name, tag, output_dir, output_format, ax=None) -> None` — raw fronts per method; xlim fixed at (-1, 91).
- `get_differences_and_mutations(fronts) -> Dict[str, (diffs, mutations)]` — pointwise fitness minus the best-area reference front.
- `plot_differences_between_fronts(fronts, gene_name, tag, output_dir, output_format, output_name=None, ax=None) -> None`.
- `plot_interesting_pareto_fronts(fronts, gene_name, tag, output_dir, output_format) -> None` — calls both of the above.
- `update_ranks_and_areas(ranks_and_areas, differences) -> None` — accumulates ranks, areas and best counts (smallest diff = rank 1).
- `compare_methods_final(results_paths: Dict[str, str], output_dir, output_format, max_mutations=90) -> None` — normalized/"interesting" front plots plus `ranks_and_areas.json`. **[bug?]** the per-gene `differences = {m: sum([value]) for ... for value in values.values()}` keeps only the last method's value, so the min/max interesting-gene choice reflects one method only.
- `normalize_gene_name(gene_name: str) -> str` — strips a trailing `_dddddd_dddddd_dddddd` timestamp.
- `load_mutation_data(input_data: Dict[str, Tuple[results_path, mutation_json]]) -> (method→gene→MutationsGene, gene_names)` — **[broken]** on a gene-list mismatch it computes `set - dict` and raises TypeError instead of the intended ValueError. Not the same as analyze_mutations.load_mutation_data.
- `count_mutations_per_method(method_dict, gene) -> Dict[str, int]` — number of distinct final-generation mutations per method.
- `get_ranks_from_sorted(curr_dict: Dict[str, float]) -> List[float]` — tie-averaged ranks with values sorted descending.
- `update_comparison_dict(comparison_dict, curr_dict, max_best=True, summation_field="mutation_count")` — accumulates sums, ranks and fractional best counts. **[bug]** with max_best=False and ties, the ranks (computed in descending order) don't line up with the methods (sorted ascending).
- `rank_by_mutation_count(method_dict, gene_names) -> dict`.
- `get_sampled_pareto_fronts(pareto_front: List[MutatedSequence], max_bootstrap: int) -> List[List[MutatedSequence]]` — random fronts with one individual per mutation count (min(#combinations, max_bootstrap) draws, with replacement, unseeded).
- `calculate_conservation_statistic_pareto_front(pareto_front, max_bootstrap, mutable_positions) -> float` — mean conservation statistic over the sampled fronts.
- `calculate_conservation_per_method(method_dict, gene, max_bootstrap, mutable_positions) -> Dict[str, float]`.
- `rank_by_conservation(method_dict, gene_names, max_bootstrap, mutable_positions) -> dict`.
- `compare_diversity_methods(input_data, output_dir, max_bootstrap, mutable_positions) -> None` — writes `diversity_comparison.json`.
- `get_all_mutations(method_dict, gene) -> set` — union of final-generation mutations across methods.
- `get_method_vectors(method_dict, gene, all_mutations) -> Dict[str, List[List[int]]]` — binary mutation-presence vector per individual.
- `pca_transform_all_methods(method_vectors) -> (PCA, transformed)` — sklearn 2-component PCA fitted jointly on all methods.
- `plot_pca_all_methods(method_vectors, gene, output_dir, output_format, ax=None)`.
- `pca_transform_single_method(vectors) -> (PCA, transformed)`.
- `plot_pca_single_method(vectors, gene, method, output_dir, output_format, ax=None)`.
- `pca_visualization(input_data, output_dir) -> None` — PCA plots for 4 random genes.
- `get_best_area(method_paths, max_mutations) -> (best_area, maximization)`.
- `get_area(pareto_front_path, max_mutations) -> float` — Σ fitness of the expanded front.
- `visualize_progress(generation_losses, output_dir, output_format, logarithmic=True, ax=None) -> None` — mean loss to the best front vs generation, per method. **[bug]** `i % 1000 != -10` is always true, so every error bar is NaN and none are drawn.
- `compare_method_progress(results_paths, max_mutations, output_format, output_dir) -> None` — area loss of every `pareto_front_gen_*.json` against the best method's final area; log and linear plots.

## compare_analysis_results.py  [one-off]
Bar plots of "accuracy" (line 2 of `summary*.txt`, i.e. final_fitness_mean) across 8 hardcoded relative `analysis/...` folders from old run-configuration comparisons.
