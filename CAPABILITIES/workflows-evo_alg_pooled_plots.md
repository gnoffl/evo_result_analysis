# src/workflows/evo_alg_pooled_plots/

Cross-run ("pooled") analyses of finished evolutionary runs, each reading run folders under the ARCitect dataset tree (hardcoded paths) and writing CSVs/PNGs next to the script. `tf_comparison/` is a calc/plot toolbox for per-TF deepCIS binding change across runs (TF × run heatmaps, Wilcoxon/Mann-Whitney + BH stars) plus thin scripts per comparison; `benchmarks/` loads optimizer `timings.json` files and fits log-log runtime scaling; `natural_unconstrained_comparison/` contrasts VCF-constrained ("natural") vs unconstrained runs (final fitness, solution-space size, per-region mutation counts, % of actual mutations at VCF positions); `example_gene_pareto/` and `mutation_range/` handle a single example gene's Pareto fronts and mutation positions/spans. Several modules are imported by `workflows/paper_plots/` (compose_plots_mpl, fig1_miniatures, supplementary_plots_mpl).

Shared conventions: run dir = `<run>/stats_*.json` + `<run>/deepcis_scan/*_peak_summary.csv` + `*_annotated_peaks_*.csv`; gene subdirs start with a digit and hold `reference_sequence.fa` + `saved_populations/pareto_front.json` (`[sequence, fitness, mutation_count]` entries, `front[0]` = most mutated); "core gene id" = first 2 `_`-fields (e.g. `1_AT1G01150`). Optimized-sequence label is `"max_mutated"` (old pipeline) or `"optimized"` (current pipeline).

## tf_comparison/tf_comparison_calc.py  [library]
Loading + statistics for cross-run per-TF binding diffs (no plotting). Uses `analysis.utils.statistics` (`wilcoxon_pvalue_vs_zero`, `benjamini_hochberg_qvalues`). Imported by paper_plots.

- `TF_COLUMN`, `DIFF_COLUMN`, `REF_COLUMN`, `MUTATED_COLUMN` (`"max_mutated"`), `ANNOTATED_PEAKS_GLOB`, `SIGNAL_TYPE_COLUMN`, `GENE_COLUMN` — column-name schema of the deepCIS peak CSVs.
- `count_genes(run_dir: str) -> int` — number of genes = length of the run's single `stats_*.json` (raises if not exactly one).
- `load_run_diffs(run_dir: str, label: str, normalization: str = "per_gene", mutated_column: str = MUTATED_COLUMN) -> pd.Series` — per-TF value from the run's `*_peak_summary.csv`: `"per_gene"` = `diff_calc / count_genes`; `"fold_change"` = log2(mutated/reference) aggregate peak counts (NaN where ratio ≤0/inf). Any other normalization string silently returns the un-normalized summary DataFrame.
- `build_matrix(runs: List[Tuple[str, str]], normalization: str = "per_gene", mutated_column: str = MUTATED_COLUMN) -> pd.DataFrame` — TF × run matrix (columns in given order, NaN for absent TFs) from `load_run_diffs`.
- `order_tfs_by_group_contrast(matrix: pd.DataFrame, left_columns: List[str], right_columns: List[str]) -> pd.DataFrame` — sort rows by `mean(left) - mean(right)` descending (NaN group mean → 0).
- `order_tfs_by_mean(matrix: pd.DataFrame) -> pd.DataFrame` — sort rows by row mean descending.
- `top_bottom_tfs(matrix: pd.DataFrame, n: int) -> pd.DataFrame` — keep first n + last n rows of an already-ordered matrix (deduplicated).
- `load_per_gene_tf_counts(run_dir: str, n_core_fields: int = 2, mutated_column: str = MUTATED_COLUMN) -> pd.DataFrame` — per (core_gene, tf) reference / optimized peak counts and `diff` from the newest (lexicographically max) `*_annotated_peaks_*.csv`; only combos with ≥1 peak present. Use `n_core_fields=3` for random-start `random_sequence_<i>_...` ids.
- `load_per_gene_diffs(run_dir: str, n_core_fields: int = 2, mutated_column: str = MUTATED_COLUMN) -> pd.DataFrame` — `diff` column only of the above.
- `paired_tf_significance(run_a_directory: str, run_b_directory: str, label_a: str = "a", label_b: str = "b", mutated_column: str = MUTATED_COLUMN) -> pd.DataFrame` — per TF over shared genes (absent = 0): two-sided Wilcoxon of `diff_a - diff_b`, of `diff_a` vs 0 and `diff_b` vs 0; each p column BH-corrected across TFs (`q_contrast`, `q_<label_a>`, `q_<label_b>`); sorted by `q_contrast`.
- `single_run_tf_significance(run_dir: str, n_core_fields: int = 2, mutated_column: str = MUTATED_COLUMN) -> pd.DataFrame` — per-TF Wilcoxon of per-gene diff vs 0 within one run (0-filled genes × TFs), BH → `q_intra`; columns tf, n_genes, median_diff, n_nonzero, p_intra, q_intra.
- `per_gene_tf_binding_summary(run_dir: str, n_core_fields: int = 2, mutated_column: str = MUTATED_COLUMN) -> pd.DataFrame` — per-TF mean and sample SD (ddof=1) over genes (0-filled) of reference, optimized and diff peak counts. Output column names are always `mean_max_mutated`/`std_max_mutated` even when `mutated_column="optimized"`.
- `pooled_model_tf_significance(model_pairs: List[Tuple[str, str]], mutated_column: str = MUTATED_COLUMN) -> pd.DataFrame` — model main effect: concatenates within-pair per-gene contrasts `D = diff_a - diff_b` across disjoint gene sources, Wilcoxon vs 0 per TF, BH → `q_model`.
- `interaction_model_tf_significance(group_a_pair: Tuple[str, str], group_b_pair: Tuple[str, str], label_a: str = "a", label_b: str = "b", mutated_column: str = MUTATED_COLUMN) -> pd.DataFrame` — per-TF two-sided Mann-Whitney U of D vectors between two gene groups (interaction), BH → `q_interaction`; NaN if MWU undefined.

## tf_comparison/tf_comparison_plot.py  [library]
Annotated diverging TF × run heatmap (no stats/IO). Imported by paper_plots. Forces `matplotlib.use("Agg")` at import.

- `COLORMAP` (`"RdBu_r"`), `STAR_THRESHOLDS` (= `analysis.utils.statistics.SIGNIFICANCE_STAR_THRESHOLDS`).
- `q_to_stars(qvalue: float) -> str` — thin alias of `analysis.utils.statistics.qvalue_to_stars` ("" if not significant / NaN).
- `plot_heatmap(matrix: pd.DataFrame, annotate: bool, cbar_label: str = "diff_calc / gene", row_stars: Optional[Dict[str, str]] = None, cell_stars: Optional[Dict[str, Dict[str, str]]] = None, separator_after_column: Optional[int] = None, annotate_values: bool = True, add_colorbar: bool = True, transpose: bool = False, annotation_fontsize: Optional[float] = None, ax: Optional[plt.Axes] = None) -> Figure` — seaborn heatmap, symmetric colour limits ±max|value|, centred at 0; `row_stars` appended to TF tick labels, `cell_stars` `{run: {tf: stars}}` appended to cell values; optional thick divider; `transpose=True` puts TFs on x. ax-injection: with `ax` given no sizing/tight_layout.

## tf_comparison/minmax_comparison.py  [script]
Maximization (ara max, GOF) vs minimization (ara min, LOF) TF heatmaps; run directly, writes to `minmax_comparison/`. Uses tf_comparison_calc + tf_comparison_plot.

- `RUNS`, `LEFT_COLUMNS`, `RIGHT_COLUMNS`, `PAIRED_RUN_A`, `PAIRED_RUN_B`, `SINGLE_RUNS` — `(run_dir, label)` config for the four runs (paired ara runs share genes).
- `NORMALIZATIONS` — `(key, file suffix, cbar label)` for per_gene and fold_change variants (same constant repeated in every tf_comparison script).
- `main(runs=RUNS, left_columns=..., right_columns=..., paired_run_a=..., paired_run_b=..., single_runs=..., output_dir: os.PathLike = OUTPUT_DIR, top_bottom_n: Optional[int] = None) -> None` — paired significance (row stars = q_contrast, cell stars = q_a/q_b), intra-run tests for GOF/LOF, heatmaps per normalization ordered by group contrast.

## tf_comparison/mut5_comparison.py  [script]
Same as minmax_comparison but at exactly-5-mutation pareto entries (`*_mut5` deepCIS reruns, `mutated_column="optimized"`); run directly → `mut5_comparison/`. Its run constants are imported by paper_plots/supplementary_plots_mpl.

- `MUTATED_COLUMN` (`"optimized"`), `RUNS`, `LEFT_COLUMNS`, `RIGHT_COLUMNS`, `PAIRED_RUN_A`, `PAIRED_RUN_B`, `SINGLE_RUNS` — `*_mut5` run dirs/labels.
- `main(..., top_bottom_n: Optional[int] = None, transpose: bool = True) -> None` — as minmax `main`, plus transposed layout option; `__main__` runs it twice (full transposed, then top-5 untransposed).

## tf_comparison/random_start_comparison.py  [script]
Single random-start maximization run: binding summary CSV, intra-run significance, heatmaps ordered by mean; run directly → `random_start_comparison/`. Uses `n_core_fields=3`. (Module comment claims deepcis_scan must hold exactly one annotated-peaks CSV — stale: the loader now picks the newest.)

- `RUN_DIR`, `LABEL`, `CORE_ID_FIELDS` (3).
- `main(run_dir: str = RUN_DIR, label: str = LABEL, output_dir=..., top_bottom_n: Optional[int] = None, core_id_fields: int = 3) -> None`.

## tf_comparison/model_comparison.py  [script]
2×2 gene source (ara/zea) × deepCRE model (ara/zea) TF comparison grouped by model; run directly → `model_comparison/`. Writes stratified paired CSVs, pooled main effect (`pooled_model_tf_significance`) and interaction (`interaction_model_tf_significance`) CSVs; heatmap shows only cell (intra-run) stars.

- `RUNS`, `LEFT_COLUMNS`, `RIGHT_COLUMNS`, `ARA_GENE_PAIR`, `ZEA_GENE_PAIR` — run dirs under `comparison_zea_ara/SSR_regular_extractionwindow/analysis_gernot`.
- `main(runs=..., left_columns=..., right_columns=..., ara_gene_pair=..., zea_gene_pair=..., output_dir=..., top_bottom_n: Optional[int] = None) -> None`.

## tf_comparison/constraint_comparison.py  [script]
GOF and LOF each under four mutation constraints (unconstrained, natural, CRISPR, CRISPR 3PAM): intra-run stars per cell, paired CSV for all 6 run pairs, heatmaps ordered by mean; run directly → `constraint_comparison/`. `per_gene` normalization raises for the CRISPR runs (no `stats_*.json`).

- `GOF_RUNS`, `LOF_RUNS` — `(run_dir, label)` for the four constraints.
- `write_intra_run_significance(runs: List[Tuple[str, str]], direction: str, output_dir: Path) -> Dict[str, Dict[str, str]]` — `single_run_tf_significance` per run → CSV; returns `{label: {tf: stars}}` cell stars.
- `write_pairwise_significance(runs: List[Tuple[str, str]], direction: str, output_dir: Path) -> None` — `paired_tf_significance` for every run pair → CSV.
- `compare_constraints(runs: List[Tuple[str, str]], direction: str, output_dir: os.PathLike = OUTPUT_DIR, transpose: bool = False) -> None` — both of the above plus one heatmap per normalization.

## benchmarks/runtime_benchmark_calc.py  [library]
Load optimizer runtime benchmark `timings.json` files into a tidy frame; summaries, power-law scaling fits, sanity checks (no plotting). Imported by paper_plots/supplementary_plots_mpl.

- `MIXED_HARDWARE_ROOT`, `MIXED_HARDWARE_DEVICE_FOLDERS`, `SAME_HARDWARE_ROOT`, `SAME_HARDWARE_DEVICE_FOLDERS` — benchmark trees and subfolder→device maps.
- `GPU_CONTENTION_CAVEAT`, `SWEPT_PARAMETERS` (label prefix → config field), `ALGORITHMS` (`ga`, `greedy`), `REGIMES` (`unconstrained`, `natural`), `POWER_LAW_R_SQUARED_FLOOR` (0.95), `RUNTIME_COLUMN` (`algorithm_seconds`), `MODEL_LOAD_COLUMN`, `EVALUATIONS_COLUMN`, `SECONDS_PER_EVALUATION_COLUMN`, `EXPECTED_CELLS_PER_DEVICE`.
- `parse_run_directory_name(directory_name: str) -> Tuple[str, str, str]` — `<algorithm>_<label>_<regime>_<date>_<time>_<us>` → (algorithm, label, regime); validates.
- `split_label(label: str) -> Tuple[str, int]` — e.g. `"pop50"` → (`"population_size"`, 50).
- `load_timings_file(timings_path: Path, algorithm: str, device: str, label: str, regime: str) -> pd.DataFrame` — one row per sequence; validates the swept field and (GA) `n_evaluations == pop*(gen+1)`.
- `load_benchmark_runtimes(benchmark_root: Path, device_folders: Dict[str, str]) -> pd.DataFrame` — all run folders → tidy frame, adds `seconds_per_evaluation` (NaN for greedy).
- `check_expected_cells(runtimes: pd.DataFrame, expected_cells: Optional[Dict[str, int]] = None) -> pd.DataFrame` — cells/sequences found per (algorithm, device) vs expected.
- `summarize_runtimes(runtimes: pd.DataFrame, group_columns: Sequence[str] = (...), value_column: str = RUNTIME_COLUMN) -> pd.DataFrame` — per-cell n, median, q1, q3, iqr, min, max.
- `ScalingFit` (dataclass: exponent, exponent_ci_low, exponent_ci_high, coefficient, r_squared, n_points, confidence) — power law `y = coefficient * x**exponent`.
- `fit_log_log_scaling(x_values: Iterable[float], y_values: Iterable[float], confidence: float = 0.95) -> ScalingFit` — OLS (`scipy.stats.linregress`) on log x/log y, slope CI = ±t(n−2)·SE; needs ≥3 positive points and ≥2 distinct x. Generic.
- `fit_sweep_scaling(runtimes: pd.DataFrame, x_column: str = "swept_value", group_columns: Sequence[str] = (...), value_column: str = RUNTIME_COLUMN, confidence: float = 0.95) -> pd.DataFrame` — one fit per sweep curve using every sequence as a point; skips curves with <2 levels or <3 points.
- `flag_poor_power_law_fits(fits: pd.DataFrame, r_squared_floor: float = 0.95) -> pd.DataFrame` — rows with r² below floor.
- `find_non_monotonic_curves(runtimes: pd.DataFrame, group_columns: Sequence[str] = (...), value_column: str = RUNTIME_COLUMN) -> pd.DataFrame` — steps where the per-level median runtime decreases.
- `regime_runtime_ratio(runtimes: pd.DataFrame, group_columns: Sequence[str] = ("algorithm", "device", "label"), value_column: str = RUNTIME_COLUMN) -> pd.DataFrame` — natural/unconstrained median ratio per cell.

## benchmarks/runtime_benchmark_plot.py  [library]
Runtime sweep boxplots and log-log scaling plots (ax-injection). Uses runtime_benchmark_calc. Imported by paper_plots/supplementary_plots_mpl. Forces `matplotlib.use("Agg")` at import.

- `REGIME_PALETTE`, `REGIME_ORDER`, `DEVICE_PALETTE`, `DEVICE_ORDER`, `SWEPT_PARAMETER_LABELS`, `RUNTIME_AXIS_LABEL`.
- `color_box_outlines(ax: plt.Axes) -> None` — repaint each seaborn box's outline/median/whiskers/caps in its fill hue (matched by x-centre) and lighten fill; generic for any seaborn boxplot on log axes.
- `plot_runtime_sweep(runtimes: pd.DataFrame, swept_parameter: Optional[str] = None, hue_column: str = "regime", value_column: str = RUNTIME_COLUMN, log_y: bool = True, show_points: bool = True, color_outlines: bool = True, output_dir: Optional[Path] = None, file_stem: str = "runtime_sweep", fmt: str = "png", show_legend: bool = True, show_title: bool = True, title: Optional[str] = None, ax: Optional[plt.Axes] = None) -> plt.Axes` — boxplots of runtime per swept value (categorical x) with jittered per-sequence dots.
- `plot_scaling_fit(runtimes: pd.DataFrame, fits: Dict[str, ScalingFit], x_column: str = "n_evaluations", hue_column: str = "regime", value_column: str = RUNTIME_COLUMN, palette: Optional[Dict[str, str]] = None, output_dir=None, file_stem: str = "runtime_scaling", fmt: str = "png", show_legend: bool = True, show_title: bool = True, title=None, xlabel=None, ax=None) -> plt.Axes` — log-log scatter per hue level plus fitted power-law line; exponent/CI/R² in legend.

## benchmarks/runtime_benchmark.py  [script]
Orchestrates the benchmark: `conda run -n deepCREshap python -m workflows.evo_alg_pooled_plots.benchmarks.runtime_benchmark`; module flag `USE_MIXED_HARDWARE_DATA` picks the tree; writes tables + one figure set per device into `benchmarks_same_hardware/` or `benchmarks_mixed_hardware/`.

- `cost_sweep_rows(runtimes: pd.DataFrame) -> pd.DataFrame` — drop GA mutation-cap control rows.
- `build_mutation_cap_frame(runtimes: pd.DataFrame, device: str) -> pd.DataFrame` — GA unconstrained cells at pivot (pop 100, gen 1000) relabelled as a mutation-cap sweep.
- `write_tables(runtimes: pd.DataFrame, output_dir: Path) -> None` — tidy, completeness, summary, fits, poor fits, non-monotonic steps, regime ratios CSVs.
- `print_fit_diagnostics(runtimes: pd.DataFrame) -> None` — prints non-monotonic steps and poor fits.
- `device_title_suffix(device: str) -> str`; `plot_sweep(runtimes, swept_parameter, output_dir, file_stem, title) -> None` — boxplot if ≤2 levels, else log-log fit per regime; `plot_device_figures(runtimes, output_dir, device) -> None`; `plot_evaluation_scaling(runtimes, output_dir, device) -> None` — GA runtime vs n_evaluations.
- `main(benchmark_root: Path = ..., device_folders=..., output_dir: Path = ..., devices: Tuple[str, ...] = ...) -> None`.

## example_gene_pareto/example_gene_pareto.py  [library]
Pareto-front-over-generations scatter series for one hardcoded example gene (`RUN_DIR` = its `saved_populations`); run directly → `plots/step_XX_*.svg`. `load_final_front_and_rejected` is imported by paper_plots/fig1_miniatures. Uses `workflows.paper_plots.style`.

- `RUN_DIR`, `FRONT_GLOB`, `POPULATION_GLOB`, `GRADIENT_CMAP`, `OUTPUT_DIR`.
- `load_mutation_prediction(file_path: Path) -> tuple[list[float], list[float]]` — (mutation_counts, predictions) from a `[seq, prediction, count]` JSON.
- `load_sequence_mutation_prediction(file_path: Path) -> tuple[list[str], list[float], list[float]]` — same plus sequences.
- `all_population_files(run_dir: Path | None = None) -> list[Path]` — `population_gen_<N>_before.json` files sorted by N (default `RUN_DIR`).
- `sample_gradient_colors(count: int) -> list[tuple]` — evenly spaced RGBA from magma (0.8→0.0).
- `order_front_files(file_names: list[str]) -> list[tuple[str, str]]` — sort `pareto_front_gen_<N>.json` by N, untagged `pareto_front.json` last as "Final"; returns (name, label).
- `compute_shared_limits(all_mutation_counts: list[float], all_predictions: list[float], margin: float = 0.05) -> tuple[tuple, tuple]` — padded x/y limits.
- `extract_zero_mutation_points(mutation_counts: list[float], predictions: list[float]) -> tuple[list, list]` — keep 0-mutation entries.
- `plot_fronts_up_to(front_data, start_counts, start_predictions, all_front_styles, xlim, ylim, dim_non_final: bool = False) -> Figure` — cumulative frame: start star + given fronts, fixed legend.
- `plot_final_front_mini(mutation_counts: list[float], predictions: list[float]) -> Figure` — 45×36 mm inset scatter under `publication_style()`.
- `save_figure(fig: Figure, filename: str) -> None` — saves to module `OUTPUT_DIR` / `OUTPUT_FORMAT`.
- `load_final_front() -> tuple[list[float], list[float]]` — final front of hardcoded `RUN_DIR` (no args).
- `load_final_front_and_rejected() -> tuple[tuple, tuple]` — final front + all population-snapshot candidates (dedup by sequence) not on the final front; hardcoded `RUN_DIR`.

## example_gene_pareto/mutation_locations.py  [library]
Prints 0-based positions (in the 3020 bp window) of selected front members' mutations for the example gene; run directly. Uses `analysis.mutations.summarize_mutations.MutationsGene`. `GENE_RUN_DIR`, `FINAL_GENERATION`, `find_front_member` imported by paper_plots/compose_plots_mpl.

- `GENE_RUN_DIR`, `FINAL_GENERATION` (20000), `REQUESTED_MUTATION_COUNTS`.
- `find_front_member(gene: MutationsGene, mutation_count: int, generation: int) -> MutatedSequence` — the unique front member with that mutation count (wraps `MutationsGene.search_mutation_count`); raises if 0 or >1.
- `format_member_mutations(member: MutatedSequence, mutation_count: int) -> str` — text block "pos ref -> alt" per mutation.

## mutation_range/mutation_span.py  [script]
First/last mutation position and span of the exactly-n-mutation (default 5) front entries per gene for six runs; run directly → `results/*.csv`. Uses `analysis.mutations.summarize_mutations.load_mutations_from_json`.

- `RUN_JSON_FILES` — run name → `all_mutated_sequences_*.json` path (ara/zea max/min, GOF, LOF).
- `select_sequences_with_mutation_count(sequences: List[MutatedSequence], n_mutations: int) -> List[MutatedSequence]` — filter by exact mutation count.
- `collect_run_spans(json_file: str, n_mutations: int = 5) -> pd.DataFrame` — one row per (gene, generation in the JSON's `generation_dict`) with first/last 0-based position, span, fitness; highest-fitness entry if several.
- `describe_spans(span_table: pd.DataFrame) -> pd.DataFrame` — `describe()` of position/span columns.
- `collect_all_runs(run_json_files: Dict[str, str] | None = None, n_mutations: int = 5) -> pd.DataFrame`; `describe_all_runs(all_runs_table: pd.DataFrame) -> pd.DataFrame` — per-run describe stacked.

## natural_gof_lof_top_deltas.py  [script]
For natural (VCF-constrained) GOF/LOF runs, per gene: fitness delta of the exactly-k-mutation front entry (k=5,7) vs the 0-mutation entry, mutations in plus-strand genome coordinates (verified against TAIR10 FASTA); run directly → `natural_gof_lof_top_deltas/natural_gof_lof_mut{k}.csv`.

- `GOF_RUN_DIR`, `LOF_RUN_DIR`, `GOF_VCF_DIR`, `LOF_VCF_DIR`, `GENOME_PATH` (TAIR10), `COMPLEMENT`, `Genome` (type alias: pyfaidx Fasta or str mapping).
- `genome_base(genome: Genome, chromosome: str, genomic_position: int) -> str` — uppercase base at a 1-based genomic position.
- `parse_gene_dir_name(dir_name: str) -> Tuple[str, str, int, int, str]` — `<chr>_<gene>_gene:<start>-<end>_...` → (gene_id, chromosome, start, end, strand; strand "-" if start > end).
- `load_genomic_positions(vcf_path: Path) -> Dict[int, int]` — reextracted VCF: 0-based sequence position → 1-based genomic position (from INFO `GENOMIC_POS`).
- `load_reference_sequence(gene_dir: Path) -> str` — `reference_sequence_full` record of `reference_sequence.fa`, uppercased.
- `find_front_entry(pareto_front: List[list], mutation_count: int) -> Tuple[str, float]` — (sequence, fitness) of the unique entry with that count; raises otherwise.
- `format_mutations(reference_sequence: str, mutated_sequence: str, chromosome: str, strand: str, genomic_positions: Dict[int, int], genome: Genome) -> List[str]` — `"chr:pos ref->alt"` per differing position, complemented to plus strand for "-" genes; raises if no VCF record or genome ref mismatch.
- `build_gene_row(gene_dir: Path, group: str, vcf_dir: Path, mutation_count: int, genome: Genome) -> Dict[str, object]`; `build_table(runs: List[Tuple[Path, Path, str]], mutation_count: int, genome: Genome) -> pd.DataFrame` — sorted by `delta_mut` desc.

## natural_unconstrained_comparison/natural_unconstrained_comparison.py  [library]
Paired one-sided Wilcoxon of per-gene final fitness, natural vs unconstrained runs (GOF "less", LOF "greater"), paired boxplots; run directly → `results_comparison/`. `core_gene_id` is imported by actual_mutation_region_breakdown.

- `COMPARISONS`, `CONDITION_ORDER`, `CONDITION_PALETTE`, `STAR_THRESHOLDS` (p-value thresholds, local copy).
- `core_gene_id(gene_id: str) -> str` — first two `_`-fields (e.g. `5_AT5G13640`).
- `load_final_fitness(run_dir: str) -> Dict[str, float]` — core gene id → `final_fitness` from the single `stats_*.json`.
- `pair_final_fitness(unconstrained_dir: str, natural_dir: str) -> pd.DataFrame` — inner join on core id; columns unconstrained, natural.
- `rank_biserial_effect_size(differences: np.ndarray) -> float` — matched-pairs rank-biserial (zeros dropped). Generic.
- `wilcoxon_paired_test(paired: pd.DataFrame, alternative: str = "two-sided") -> Dict[str, float]` — Wilcoxon on natural − unconstrained + medians + rank-biserial.
- `p_value_to_stars(p_value: float) -> str` — `***/**/*`, else `"ns"`.
- `save_figure(fig: Figure, filename: str, output_dir: str, fmt: str = "png") -> None` — savefig 150 dpi + print.
- `plot_paired_fitness(paired: pd.DataFrame, title: str, p_value: float, log_scale: bool = False, ax: Optional[plt.Axes] = None) -> Figure` — two-box plot with per-gene connecting lines and significance bracket (ax-injection).

## natural_unconstrained_comparison/natural_unconstrained_mutation_vis.py  [library]
Solution-space size: VCF positions/records (constrained) vs non-N positions ×3 (unconstrained) per gene, Wilcoxon/MWU tests, log bar plots; run directly → `natural_unconstrained_mutation_vis/`. Uses `vcf_mutation_stats.collect_stats`. Its run/VCF dir constants are imported by region_mutation_breakdown, actual_mutation_region_breakdown and paper_plots.

- `GOF_RUN_DIR`, `LOF_RUN_DIR` (natural runs), `GOF_VCF_DIR`, `LOF_VCF_DIR` (reextracted VCF dirs in the Evolution repo).
- `count_unconstrained_positions(fa_path: Path) -> int` — non-N bases in `reference_sequence_full`.
- `collect_unconstrained_stats(run_dir: Path) -> Tuple[List[int], List[int]]` — per gene (digit-prefixed subdirs): (non-N positions, positions × 3).
- `build_comparison_dataframe(...8 per-group lists...) -> pd.DataFrame` — tidy group/condition/positions/mutations.
- `run_statistical_tests(gof_constrained: List[int], lof_constrained: List[int], gof_unconstrained: List[int], lof_unconstrained: List[int], metric_name: str) -> pd.DataFrame` — paired Wilcoxon within group, two-sided MWU GOF vs LOF within condition (uncorrected).
- `plot_solution_space_comparison(data: pd.DataFrame, output_dir: str, fmt: str = "png", test_results: Optional[pd.DataFrame] = None) -> None` — two log-y mean±SD bar figures with brackets (bracket x positions assume seaborn patch order).

## natural_unconstrained_comparison/region_mutation_breakdown.py  [library]
Possible mutations per gene split by construct region (promoter 1–1000, 5'UTR 1001–1500, 3'UTR 1521–2020, terminator 2021–3020; 1-based), constrained (VCF record count) vs unconstrained (non-N × 3); run directly → `region_mutation_breakdown/`. Imports dir constants from natural_unconstrained_mutation_vis. `REGION_ORDER`/`assign_region` reused by actual_mutation_region_breakdown; several functions used by paper_plots/compose_plots_mpl.

- `REGIONS`, `REGION_ORDER`, `DEFAULT_CONDITION_PALETTE`.
- `assign_region(position: int) -> Optional[str]` — 1-based construct position → region name (None in the 1501–1520 N gap or out of range).
- `collect_vcf_region_counts(vcf_dir: Path) -> pd.DataFrame` — per VCF file × region record counts (gene = file stem).
- `collect_unconstrained_region_counts(run_dir: Path) -> pd.DataFrame` — per gene dir × region non-N positions × 3.
- `build_region_dataframe(gof_vcf, lof_vcf, gof_unconstrained, lof_unconstrained) -> pd.DataFrame` — tidy gene/group/condition/region/count.
- `plot_region_breakdown(data: pd.DataFrame, group: str, output_dir: Optional[Path] = None, fmt: str = "png", show_legend: bool = True, show_title: bool = True, palette: Optional[dict] = None, ax: Optional[plt.Axes] = None) -> plt.Axes` — log-y mean±SD bars by region × condition.
- `plot_region_breakdown_boxes(data, group, output_dir=None, fmt="png", show_legend=True, show_title=True, palette=None, ax=None, show_gene_points: bool = False) -> plt.Axes` — same as boxes on a symlog y axis (0 visible), hue-coloured outlines/medians, optional per-gene dots.

## natural_unconstrained_comparison/actual_mutation_region_breakdown.py  [library]
For UNCONSTRAINED GOF/LOF runs (`*_single_mutation_*`): % of mutations in the most-mutated front member (`front[0]`) that fall on natural VCF positions, per region; per-gene boxplots and pooled percent with gene-level cluster bootstrap CI (fig 4 panel J); run directly → `actual_mutation_region_breakdown/`. Imports `core_gene_id`, VCF dir constants, `REGION_ORDER`/`assign_region` from siblings and `MutatedSequence` from analysis.mutations.summarize_mutations. Used by paper_plots/compose_plots_mpl.

- `GOF_UNCONSTRAINED_RUN_DIR`, `LOF_UNCONSTRAINED_RUN_DIR`, `DEFAULT_GROUP_PALETTE`, `BOOTSTRAP_RESAMPLES` (10000), `BOOTSTRAP_SEED`, `BOOTSTRAP_CONFIDENCE` (95).
- `load_vcf_positions_by_gene(vcf_dir: Path) -> Dict[str, Set[int]]` — core gene id → set of 1-based VCF positions.
- `load_most_mutated_individual(gene_dir: Path) -> MutatedSequence` — `pareto_front.json[0]` diffed vs reference (reads FASTA record index 1, `reference_sequence_mutation_window_0_3020`; identical to `_full` in the checked gene).
- `collect_actual_mutation_region_counts(run_dir: Path, vcf_positions_by_gene: Dict[str, Set[int]]) -> pd.DataFrame` — per gene × region total and VCF-allowed mutation counts; prints skipped genes without VCF.
- `build_group_dataframe(gof_data: pd.DataFrame, lof_data: pd.DataFrame) -> pd.DataFrame` — adds group and `percent_allowed` (NaN if 0 mutations).
- `build_summary_dataframe(data: pd.DataFrame) -> pd.DataFrame` — per group/region n_genes and pooled percent (no CI).
- `pooled_percent_with_bootstrap_ci(allowed_per_gene: np.ndarray, total_per_gene: np.ndarray, n_resamples: int = 10000, seed: int = ..., confidence: float = 95.0) -> Tuple[float, float, float]` — ratio-of-sums % with percentile cluster bootstrap over genes (zero-denominator resamples discarded). Generic.
- `build_pooled_ci_dataframe(data: pd.DataFrame, n_resamples=..., seed=..., confidence=...) -> pd.DataFrame` — above per group × region (seed + cell index).
- `plot_actual_mutation_allowance(data, output_dir=None, fmt="png", show_legend=True, show_title=True, palette=None, ax=None) -> plt.Axes` — per-gene percent boxplots by region × group.
- `plot_pooled_allowance_bars(data, output_dir=None, fmt="png", show_legend=True, show_title=True, palette=None, ax=None, n_resamples=..., seed=...) -> plt.Axes` — pooled % bars with asymmetric bootstrap error bars.
- `plot_pooled_allowance_points(data, ..., show_gene_points: bool = True) -> plt.Axes` — pooled % as diamond + CI whisker over jittered per-gene dots (the panel used by compose_plots_mpl).

## natural_unconstrained_comparison/vcf_mutation_stats.py  [library]
Per-gene VCF record counts (unique positions and total records) for GOF/LOF reextracted VCFs; `__main__` writes `vcf_mutation_stats/vcf_mutation_stats.csv`. `collect_stats` imported by natural_unconstrained_mutation_vis.

- `count_mutations_in_vcf(vcf_path: Path) -> Tuple[int, int]` — (unique POS count, record count).
- `collect_stats(vcf_dir: Path) -> Tuple[List[int], List[int]]` — above over all `*.vcf`, sorted.
- `build_summary_row(label: str, unique_locations: List[int], total_mutations: List[int]) -> dict` — mean / SD (population SD, ddof=0) / min / max.
