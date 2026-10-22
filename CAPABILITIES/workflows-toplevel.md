# src/workflows/ (top-level modules)

Three standalone modules sit directly in `src/workflows/`. `candidate_selection.py` holds the GOF/LOF wet-lab candidate selection from Pareto fronts and also the generic helpers for front lookup and restriction-site screening. `figure_composition_svg.py` composes existing SVG panels into one labelled multi-panel SVG. `update_excel_sheet.py` is a one-shot script that adds GOF/LOF presence columns to collaborator CSV target lists.

## candidate_selection.py  [library | script]
Picks the best-per-gene and best-per-mutation-count GOF/LOF candidates by merging single- and multi-mutation runs. Also provides front lookup and restriction-site filtering helpers, used by `compare_reruns/compare_runs.py` and (via a broken import) `hoffie/create_filtered_sequences.py`. CLI: `-p` (preliminary: `-r results_folder -o out.csv [-max/-min start-fitness filter]`) or `-f` (final: `-s single.csv -m multi.csv -o out.csv -n n_candidates`). Imports `calculate_half_max_mutations` and `get_min_mutation_count_for_fitness` from `analysis.overview.simple_result_stats`.

- `reverse_complement(seq: str) -> str` — uppercase ACGTN only. Any other character (lowercase, IUPAC) raises KeyError.
- `get_restriction_regexes(restriction_site_path: str) -> Dict[str, re.Pattern]` — site name → regex for every record of a site FASTA, plus `<name>_reverse` for the reverse complement. Only `N` is treated as a wildcard.
- `get_sequences_without_restriction_sites(sequence_path: str, restriction_site_path: str, output_path: str, prefixes: Optional[Dict[str, str]] = None, postfixes: Optional[Dict[str, str]] = None) -> None` — writes the FASTA records that match no site. The per-record prefixes/postfixes are prepended/appended before screening **and are kept in the written sequence**. Hits are printed.
- `get_data_at_mutation_count(pareto_front: List[Tuple[str, float, float]], target_mutation_count: int) -> Tuple[str, float, float]` — binary search for the entry whose rounded mutation count equals the target exactly. Assumes `front[0]` has the most mutations (counts descending). Raises ValueError if the count is absent (fronts can have gaps).
- `get_pareto_front_paths(results_folder: str) -> List[str]` — the `<subdir>/saved_populations/pareto_front.json` paths that exist, one per subdirectory.
- `get_gene_name_from_path(pareto_path: str) -> str` — field 1 of the run-dir name split on `_`. This is wrong for chromosome names that contain `_`.
- `process_mutations(pareto_front: List[Tuple[str, float, float]]) -> Dict[str, Optional[float]]` — for counts 1,2,3,5,7,10,15,20, the |fitness − reference fitness| and the sequence at that exact count (NaN and "" when absent).
- `process_fitness_thresholds(pareto_front: List[Tuple[str, float, float]]) -> Dict[str, List[float]]` — for fitness 0.0..1.0 in steps of 0.1 (assumes fitness in [0,1]), the minimal mutation count to reach it and that entry's sequence. Values are NaN outside the reference–final range.
- `process_single_gene(pareto_path: str) -> Tuple[Dict, bool]` — the per-gene summary dict (reference sequence, start/final fitness, max mutation count, half-max mutations) merged with the two dicts above, and a flag saying whether the run maximised.
- `filter_and_sort_results(selected_df: pd.DataFrame, starting_fitness_max: Optional[float], starting_fitness_min: Optional[float], ascending: bool) -> pd.DataFrame` — filters on `start_fitness` and sorts by it.
- `preliminary_selection(results_folder: str, output_path: str, starting_fitness_max: Optional[float] = None, starting_fitness_min: Optional[float] = None) -> None` — one summary row per gene of a results folder, written to CSV.
- `load_selected_genes(csv_path: str) -> pd.DataFrame` — reads such a CSV with `gene` as the index.
- `draw_line_plot(vis_folder: str, gene: str, single_points: List[Tuple[float, float]], multi_points: List[Tuple[float, float]], ax: Optional[plt.Axes] = None) -> None` — fitness (x) vs. mutation count (y) for the single- and multi-mutation run of one gene. Axis limits are hardcoded to x ∈ [-0.03, 1.03], y ∈ [-3, 93]. With `ax=None` it makes and saves its own figure; otherwise it only draws.
- `get_line_plot_data(row) -> Tuple[list, list]` — the `(fitness, n_mutations)` point lists for both runs, from the `mutations_for_fitness_*_single/_multi` columns of a merged row plus the start and final points.
- `compare_trajectories_mutations(merged: pd.DataFrame)` — one trajectory plot per gene into the hardcoded `visualization/GOF_LOF/GOF/compare_trajectories`.
- `get_best_fitness_and_seq(row, mutations, single_col, multi_col) -> Tuple[float, str]` — the larger of the single and multi delta and its sequence, NaN-tolerant. Always takes the larger value, so for minimisation runs this works only because the stored deltas are absolute.
- `find_best_mutation_deltas(merged: pd.DataFrame)` — in place, adds `best_fitness_delta_for_<n>_mutations`, `per_mutation_delta_for_<n>_mutations` (delta / n) and `best_sequence_for_<n>_mutations` for n in 1,2,3,5,7,10,15,20.
- `get_sorted_values(row, per_mutation_columns: List[str], min_delta: float) -> List[Tuple[str, float]]` — the per-mutation efficiencies sorted descending, with anything below `min_delta` total gain pushed to −inf.
- `find_best_mutation_count_per_gene(row: pd.Series, rank: int = 1, min_delta: float = 0.3) -> Union[int, float]` — the mutation count with the rank-th best per-mutation efficiency, NaN when its total delta is below `min_delta`.
- `find_best_mutations_per_gene(merged: pd.DataFrame, n_candidates: int)` — in place, adds `best_mutation_count_rank_<i>` for i = 1..n.
- `selected_per_mutation(merged: pd.DataFrame, n_candidates: int) -> pd.DataFrame` — the top n genes per mutation count; `selection_reason = "top_per_mutation"`.
- `selected_per_gene(merged: pd.DataFrame, n_candidates: int) -> pd.DataFrame` — the top n mutation counts per gene; `selection_reason = "top_per_gene"`.
- `combine_selections(per_gene_bests: pd.DataFrame, per_mutation_bests: pd.DataFrame) -> pd.DataFrame` — the union, deduplicated on (gene, mutations), with overlaps marked `"both"`.
- `final_selection(single_path: str, multi_path: str, output_path: str, n_candidates: int, min_delta: float = 0.3) -> None` — merges the two summary CSVs on gene (suffixes `_single`/`_multi`) and writes the combined selection.
- `one_off_error_correction(path)` **[one-off]** — rewrites a CSV in place, replacing the gene index by its second `_` field.

## figure_composition_svg.py  [library]
Places existing SVG files as nested `<svg>` panels on one new canvas, at positions given as fractions of the canvas, with optional A/B/C labels. Used by `paper_plots/compose_plots_svg.py`. **[broken]** its own module docstring example imports `workflows.figure_composition`, which does not exist; the module is `figure_composition_svg`.

- `compose_figures(panels: list[dict], output_path: str, figsize: tuple[float, float] = (8.27, 11.69), labels: bool = False, label_fontsize: float = 14.0, label_color: tuple[float, float, float] = (0.0, 0.0, 0.0), margin: float = 0.01, panel_scale: float = 0.92) -> None` — each panel dict is `{"path": str, "rect": (x0, y0, x1, y1)}` with the rect in canvas fractions; `figsize` is in inches at 96 px/inch. Panels keep their own `viewBox` and are scaled to fit, anchored bottom-right when `panel_scale < 1` so the label corner stays free. Matplotlib's full-canvas background rect is stripped from each source and white backgrounds are drawn instead, so the output is never transparent.

## update_excel_sheet.py  [script]
Adds `GOF_all_SNPs` / `GOF_natural_SNPs` (and the LOF equivalents) presence columns to two hardcoded collaborator target-list CSVs in `data/update_excel_sheet/`, from the gene→experiment-paths map in `list_genes.json`. Run `python src/workflows/update_excel_sheet.py`; writes `<name>_updated.csv` next to the input and leaves the originals alone. No test file exists for this module.

- `load_gene_data_from_json(json_path: str) -> Dict[str, List[str]]` — reads the gene → experiment path list JSON.
- `categorize_experiment(experiment: str) -> List[str]` — the category names whose substring (`GOF/GOF_multi_mutation`, `GOF/GOF_multi_natural`, and the LOF pair) occurs in the path.
- `create_gene_mapping(gene_data: Dict[str, List[str]]) -> Dict[str, Dict[str, bool]]` — gene id → the four booleans.
- `find_gene_id_column(df: pd.DataFrame) -> Optional[str]` — the gene-id column, preferring `gene_id_atha`/`gene_id` over `locus`/`gene` (case-insensitive), else None.
- `count_genes_by_category(gene_mapping) -> Dict[str, int]`, `print_gene_statistics(gene_mapping) -> None` — the per-category gene counts, and printing them.
- `update_csv_with_columns(csv_path: str, output_path: str, gene_mapping, columns_to_add: List[str]) -> None` — reads the CSV with `header=1` (row 1 is a title row), writes `present`/`absent` columns, and saves with `QUOTE_ALL` for LibreOffice. Falls back to column 5 when no gene-id column is found.
- `get_output_filename(csv_path: str) -> str`, `get_data_directory() -> str`, `get_csv_configurations() -> List[Dict[str, Any]]`, `process_csv_file(csv_path, gene_mapping, columns) -> None` — the `_updated` suffix, the `data/update_excel_sheet` path, the two hardcoded file/column configs, and the per-file driver.