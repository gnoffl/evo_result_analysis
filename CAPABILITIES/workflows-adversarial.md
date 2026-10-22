# src/workflows/adversarial/

Checks whether Pareto-front sequences evolved against one MSR deepCRE model are also scored well by the sibling models of the same training run. `reevaluate.py` reads evolution run folders (`<gene>/parameters.json`, `saved_populations/pareto_front.json`, `reference_sequence.fa`), keeps one representative per mutation count, rescores with every `.h5` model in a folder and writes `outputs/<run>__<models>/adversarial_predictions.csv` + `input.csv` (provenance incl. model SHA-256 checksums, git commit). `plot.py` expands/normalizes those tables per gene and writes `pooled.<fmt>`, `examples.<fmt>`, `mean_normalized_gap.csv`; its panel functions are reused by supplementary fig S3. Rationale in `PLAN.md`.

## reevaluate.py  [script]
Re-score Pareto fronts with all sibling models. Run: `python src/workflows/adversarial/reevaluate.py [--models-folder DIR] [--run-folders DIR ...] [--output-root DIR]` (defaults = MSR_models_M0X0.75 and the GOF/LOF single-mutation runs). Imports `deduplicate_pareto_front` from `analysis.overview.simple_result_stats`.

- `DEFAULT_MODELS_FOLDER` — `/home/gernot/Code/PhD_Code/Evolution/models/MSR_models_M0X0.75`.
- `DEFAULT_RUN_FOLDERS` — Evolution_runs GOF_single_mutation_251009_121226_109368 and LOF_single_mutation_251020_180028_564570 (unconstrained runs).
- `DEFAULT_OUTPUT_ROOT` — `src/workflows/adversarial/outputs` (absolute, module-relative).
- `CSV_FILE_NAME = "adversarial_predictions.csv"`, `INPUT_FILE_NAME = "input.csv"`, `PREDICTION_COLUMN_PREFIX = "prediction_"`, `METADATA_COLUMNS` (gene_id, sequence, mutation_count, max_number_mutations, optimization_model, original_fitness), `GENE_ID_PATTERN` (regex `^[^_]+_([^_]+)_gene:`).
- `PredictingModel` — Protocol: `.predict(sequences: np.ndarray (n, L, 4), verbose=0) -> np.ndarray`.
- `build_output_folder(output_root: str, run_folder: str, models_folder: str) -> str` — `<output_root>/<run_name>__<models_folder_name>`.
- `compute_file_checksum(file_path: str) -> str` — SHA-256 hex digest, read in 1 MiB blocks.
- `get_analysis_commit() -> str` — `git rev-parse HEAD` of this repo, or `"unknown"`.
- `write_input_record(output_folder: str, run_folder: str, models_folder: str, model_paths: Dict[str, str], predictions: pd.DataFrame) -> str` — writes key/value provenance CSV (timestamp, paths, counts, commit, per-model checksum).
- `clean_gene_id(sequence_name: str) -> str` — `1_AT1G01720_gene:267992-269819` → `AT1G01720`; idempotent (returns input if pattern does not match).
- `discover_models(models_folder: str) -> Dict[str, str]` — name→path for all `.h5` files, sorted; raises if none.
- `load_models(model_paths: Dict[str, str]) -> Dict[str, PredictingModel]` — wraps `evolution.load_models.get_model_loader("tensorflow")`.
- `read_fasta_record(fasta_path: str, record_name: str) -> str` — plain-Python reader for one FASTA record (upper-cased); KeyError if absent.
- `reconstruct_full_sequence(sequence: str, reference_sequence: str, mutation_start: int, mutation_end: int) -> str` — returns `sequence` if full length, else splices a mutation-window sequence into the reference ([start, end)); ValueError on other lengths.
- `select_front_representatives(pareto_front: List[Sequence], gene_id: str) -> List[Tuple[str, float, int]]` — one entry per mutation count via `deduplicate_pareto_front`, ascending count; warns if fitness differs within a count.
- `predict_sequences(sequences: List[str], models: Dict[str, PredictingModel]) -> Dict[str, np.ndarray]` — one-hot (wraps `evolution.sequences.one_hot_encode`), batch-predict with every model; 1-D array per model.
- `find_gene_folders(run_folder: str) -> List[str]` — sorted subfolders containing `parameters.json` (does not require pareto_front.json); raises if none.
- `collect_gene_rows(gene_folder: str, models: Dict[str, PredictingModel]) -> List[Dict[str, object]]` — full per-gene pipeline (params → reference → representatives → reconstruct → predict) → one row per mutation count; [] if no Pareto front.
- `report_optimization_model_agreement(predictions: pd.DataFrame) -> Optional[float]` — max |recomputed optimization-model prediction − stored fitness|; None if optimization model not in the evaluated set.
- `order_columns(predictions: pd.DataFrame) -> pd.DataFrame` — metadata columns then sorted `prediction_*`.
- `reevaluate_run(run_folder: str, models: Dict[str, PredictingModel], model_paths: Dict[str, str], models_folder: str, output_root: str) -> str` — whole run → CSV + input record, prints validation; returns CSV path.

## plot.py  [script]
Plot sibling-model agreement from `reevaluate.py` CSVs. Run: `python src/workflows/adversarial/plot.py [--csv PATH ...] [--example-genes ID ...] [--output-format pdf] [--min-fitness-range 0.0]`; figures written next to each CSV. Imports constants/helpers from `reevaluate.py`. Sets the `Agg` backend at import. Panel functions are imported by `paper_plots/supplementary_plots_mpl.py` (fig S3).

- `REFERENCE_COLOR = "#c0392b"`, `OTHERS_COLOR = "#2f4f4f"` — default line/band colours; `NORMALIZED_PREFIX = "normalized_"`, `NORMALIZED_FITNESS_COLUMN`, `EXPANDED_COLUMN = "is_expanded"`, `GAP_COLUMN = "mean_normalized_gap"`, `NUMBER_OF_EXAMPLE_GENES = 3`.
- `default_csv_paths() -> List[str]` — CSV paths `reevaluate.py` writes with its defaults.
- `get_prediction_columns(predictions: pd.DataFrame) -> List[str]` — sorted `prediction_*` columns.
- `get_other_model_columns(predictions: pd.DataFrame) -> List[str]` — prediction columns minus all optimization models (`;`-split); raises if none left.
- `expand_gene_front(gene_predictions: pd.DataFrame, max_number_mutations: int) -> pd.DataFrame` — reindex one gene to counts 0..max, forward-filling whole rows (sequence + all predictions), adds `is_expanded`; row-level analogue of `simple_result_stats.expand_pareto_front`. Requires count 0.
- `normalize_gene_front(gene_predictions: pd.DataFrame) -> Optional[pd.DataFrame]` — adds `normalized_*` columns: (p − a)/(b − a) with a, b = min/max of the optimization model's `original_fitness` for that gene (other models not clipped); None if range is 0.
- `prepare_predictions(predictions: pd.DataFrame, minimum_fitness_range: float = 0.0) -> pd.DataFrame` — expand + normalize every gene, dropping genes whose fitness range < threshold.
- `summarize_spread(prepared: pd.DataFrame, value_columns: List[str]) -> pd.DataFrame` — per mutation count, pooled over the given columns (and genes): median, lower/upper quartile, min, max.
- `draw_front_with_spread(axes, mutation_counts, reference_values, spread, reference_label, others_label, show_min_max=True, reference_color=..., others_color=...) -> List[Artist]` — ax-injectable: min–max band, IQR band, others' median line, reference line; returns legend proxy handles.
- `set_robust_ylimits(axes, spread, reference_values, short_note=False, annotate_truncation=True) -> None` — y-limits to IQR ∪ reference line (+10% pad) and text note of how far the min–max band is cut off.
- `draw_pooled_panel(axes, prepared, short_note=False, show_min_max=True, reference_color=..., others_color=...) -> List[Artist]` — ax-injectable pooled normalized panel (median over genes of optimization model vs other-model spread).
- `draw_example_panel(axes, prepared, gene_id: str, reference_color=..., others_color=...) -> List[Artist]` — ax-injectable single-gene raw-prediction panel.
- `plot_pooled(prepared, run_name: str, output_folder: str, output_format: str) -> str` — standalone pooled figure → `pooled.<fmt>`.
- `compute_mean_normalized_gap(prepared: pd.DataFrame) -> pd.Series` — per gene mean over mutation counts of |median normalized other-models − normalized optimization fitness|, sorted ascending.
- `select_example_genes(prepared: pd.DataFrame, number_of_genes: int) -> List[str]` — genes at evenly spaced ranks of the gap (min, …, max).
- `plot_examples(prepared, run_name, output_folder, output_format, example_genes=None) -> Tuple[str, List[str]]` — standalone example-gene figure → `examples.<fmt>`.
- `plot_csv(csv_path: str, output_format: str, example_genes=None, minimum_fitness_range: float = 0.0) -> Dict[str, str]` — full per-CSV pipeline (clean ids, prepare, gap CSV, both figures).
- `report_expansion(prepared: pd.DataFrame) -> str` — one-line summary of the fraction of forward-filled rows.
