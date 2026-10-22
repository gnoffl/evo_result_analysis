# src/analysis/motives/

TF-binding analysis of evolved promoter sequences. The main pipeline (driven by `analysis_scripts/run_deepcis_peak_pipeline.sh`) goes: `deepcis_scanner` runs 250 bp sliding windows of the reference and one optimized sequence per gene folder through deepCIS (46 TF-family outputs) and writes a long CSV; `peak_scanner`/`peak_annotation` call binding peaks per gene × TF on the reference, optimized or difference signal; `analyze_peaks` checks peak overlap with the mutable region and summarizes/plots peak counts per TF; `deepcis_visualize` plots per-gene/TF signal tracks. `analyze_mapping` is a separate, older script for MOCA-blue EPM motif-count CSVs.

Coordinate conventions: scan rows carry `window_start` (0-based, inclusive) and `window_end` (exclusive) on the full extracted sequence (TSS at 1000, TTS at 2020). Peak `peak_start`/`peak_end` are in **window-start space** (bp positions of window starts), not window centers. Downstream code converts them ad hoc: `analyze_peaks` adds a hardcoded `+250` to `peak_end` to get covered bp; `deepcis_visualize` adds `+125` to both ends to put them on window-center coordinates.

## gene_matching.py  [library]
Resolves bare gene IDs against full run-specific gene-folder names.

- `resolve_gene_ids(requested_ids: Iterable[str], available_names: Iterable[str]) -> Tuple[List[str], List[str]]` — substring match of each ID within the names; returns (sorted matched names, sorted unmatched IDs). One ID can match several names.

## deepcis_scanner.py  [library + script]
Loads gene-folder run data and runs deepCIS sliding-window prediction. CLI: `python -m analysis.motives.deepcis_scanner -r RUN_FOLDER -o OUTPUT [-m MODEL.h5] [-n NAME] [-w 250] [-s 10] [-b 64] [-mc N] [-g GENE ...] [--overwrite]`; writes `OUTPUT/deepcis_scan/deepcis_window_scan_{NAME}.csv`.

- `N_TF_FAMILIES` — 46 (deepCIS output width).
- `TF_FAMILY_NAMES` — ordered list of the 46 deepCIS output column names (e.g. `"WRKY_tnt"`, `"bHLH_tnt"`, `"ARF_ecoli"`); index = model output index. Only copy of this list in the repo.
- `DEFAULT_WINDOW_SIZE` (250), `DEFAULT_STEP_SIZE` (10), `DEFAULT_BATCH_SIZE` (64), `DEFAULT_DELTA_THRESHOLD` (0.1), `DEFAULT_EXTRAGENIC` (1000), `DEFAULT_INTRAGENIC` (500), `DEFAULT_CENTRAL_PADDING` (20).
- `ParetoEntry` — type alias `Tuple[str, float, int]` = (mutable_sequence, fitness, mutation_count).
- `GeneRunData(gene_name, gene_folder, pareto_front, reference_sequence_full, mutation_start, mutation_end)` — dataclass for one gene folder of an evolution run; mutation_start/end are 0-based, end exclusive.
  - `.load_gene_run_data(gene_folder: str) -> GeneRunData` — classmethod; assembles from `saved_populations/pareto_front.json`, `parameters.json`, `reference_sequence.fa`.
  - `.load_pareto_front(gene_folder: str) -> List[ParetoEntry]` — static; reads `saved_populations/pareto_front.json`, casts to (str, float, int). Keeps file order.
  - `.load_gene_params(gene_folder: str) -> dict` — static; reads `parameters.json`.
  - `.load_reference_sequence_full(gene_folder: str) -> str` — static; reads record `reference_sequence_full` from `reference_sequence.fa` via pyfaidx, falling back to the first record.
- `find_gene_folders(run_folder: str) -> List[str]` — sorted subfolders containing both `parameters.json` and `saved_populations/pareto_front.json`.
- `get_full_sequence(mutable_sequence: str, reference_sequence_full: str, mutation_start: int, mutation_end: int) -> str` — splices a mutable region into the full reference (no length check).
- `find_padding_region(sequence: str) -> Tuple[int, int]` — (start, end-exclusive) of the longest run of `N`; `(-1, -1)` if none.
- `slide_windows(sequence: str, window_size: int = 250, step: int = 10, padding_start: Optional[int] = None, padding_end: Optional[int] = None) -> List[Tuple[int, int, str, bool]]` — (start, end, subseq, contains_padding) windows; padding inferred via `find_padding_region` when not given; appends one extra end-aligned window if the last regular window doesn't reach the sequence end (breaks uniform spacing; does not happen for 3020 bp sequences with step 10). Docstring wrongly says default step 50 and lists nonexistent args.
- `load_deepcis_model(model_path: str)` — `tf.keras.models.load_model`.
- `predict_windows(model, windows_onehot: np.ndarray, batch_size: int = 64) -> np.ndarray` — `model.predict` on `(n_windows, window_size, 4)` one-hot; returns float32 `(n_windows, 46)`. (Docstring claims layout adaptation; none is done.)
- `scan_single_gene_folder(model, gene_data: GeneRunData, window_size: int = 250, step: int = 10, batch_size: int = 64, mutation_count: Optional[int] = None) -> pd.DataFrame` — scans reference + one optimized sequence (max-mutation pareto entry, or exact `mutation_count`, else ValueError); columns `gene, sequence_type ('reference'|'optimized'), window_start, window_end, contains_padding, <46 TF names>`. One-hot via `evolution.sequences.one_hot_encode`.
- `scan_all_genes(model_path: str, run_folder: str, output_path: str, name: Optional[str] = None, window_size: int = 250, step: int = 10, batch_size: int = 64, overwrite: bool = False, mutation_count: Optional[int] = None, genes: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, GeneRunData]]` — scans every gene folder (optionally substring-filtered by `genes`), skips failing genes with a warning, writes CSV. If CSV exists and not `overwrite`, loads it and returns `{}` for genes_data; raises if the cached CSV uses the stale `max_mutated` label.
- `parse_arguments(args=None)` / `run_deepcis_scan(args) -> int` — CLI parsing (with path/positivity validation) and runner.

## peak_annotation.py  [library]
Peak calling on one 1-D sliding-window signal (one gene × TF).

- `PeakAnnotator(df: pd.DataFrame, threshold_peak: float, step_size: Optional[int] = None, sigma: float = 10., window_size: int = 250, lambda_weight: float = 1.0)` — `df` is required and must have `window_start` with uniform spacing (raises otherwise); step_size inferred from the first two `window_start`s when None. sigma is in bp (converted to elements via /step). Docstring defaults (sigma 20, threshold 0) are wrong.
  - `.detect_peaks(signal_column: str = "signal") -> pd.DataFrame` — runs on `self.df` (not an argument; docstring examples are outdated). A: moving average over `window_size//step` elements, regions where |avg| > threshold_peak. B: zero-pad, Gaussian smooth, forward diff. C: per region, pick edges (l, r) maximizing normalized flank-derivative term + lambda × normalized (|sum signal[l:r]| − (r−l)·threshold); split [l, r) into ceil(width/W) evenly spaced peaks of W elements. Returns columns `region_idx, peak_rank, peak_start, peak_end, peak_middle_start, peak_middle_end, edge_peak, peak_area` (bp in window-start space; `peak_end = window_start[r-1] + step - 1`; `peak_area` = signed signal sum).
  - `.validate_signal_column(signal_column: str) -> None` — checks column exists and is numeric.
  - `.distribute_peaks_over_area(start: int, finish: int, num_peaks: int) -> List[Tuple[int, int]]` — evenly spaced fixed-width (W elements) sub-intervals clipped to [start, finish).
  - `.calculate_mass_contributions(signal_cum_sum: np.ndarray, scan_range: Tuple[int,int,int,int]) -> np.ndarray` — matrix of threshold-reduced |mass| for all (l, r) candidates; used for the 90th-percentile normalizer.
  - `.calculate_scan_range(region_start: int, region_end: int, deriv: np.ndarray) -> Tuple[int,int,int,int]` — (region_start, region_end, left_end, right_end) candidate-edge search bounds, each extended by W elements.
  - `.get_middle_window(start: int, end: int) -> Tuple[int, int]` — step-sized window centered on `(start+end+1)/2`.
  - `.empty_result_df() -> pd.DataFrame` — static; empty frame with the output columns.

## peak_scanner.py  [library + script]
Runs `PeakAnnotator` over all gene × TF combinations of a deepcis_scanner CSV. CLI: `python -m analysis.motives.peak_scanner SCAN_CSV [-g GENE ...] [-t TF ...] [-s reference|optimized|difference|all] [-o OUT.csv] [--annotator-window-size 250] [--annotator-step-size N] [--annotator-threshold-peak 0.2] [--annotator-sigma 50] [--annotator-lambda-weight 2.0]`. Note CLI/`run` defaults (sigma 50, lambda 2) differ from the class defaults (sigma 10, lambda 1).

- `DeepCISPeakScanner(genes: Optional[List[str]] = None, tfs: Optional[List[str]] = None, signal_type: str = "difference", output_path=None, annotator_window_size: int = 250, annotator_step_size: Optional[int] = None, annotator_threshold_peak: float = 0.2, annotator_sigma: float = 10.0, annotator_lambda_weight: float = 1.0)` — signal_type ∈ {reference, optimized, difference}; difference = optimized − reference on exactly matching windows.
  - `.scan(scanner_data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame` — loads/validates (rejects stale `max_mutated` label), runs peak detection per gene × TF (failures logged and skipped), returns concatenated peaks with `gene, tf, signal_type` prepended. Does not save; side effect: creates `output_dir` (default `data/peak_annotations`).
  - `.validate_genes(df: pd.DataFrame) -> List[str]` — substring-resolves `self.genes`; raises on any unmatched ID.
  - `.validate_tfs(df, available_tfs: List[str]) -> List[str]` — checks requested TF columns exist.
- `run(scanner_data, output_path, genes=None, tfs=None, signal_types: List[str] = ["difference"], window_size: int = 250, step_size: Optional[int] = None, threshold_peak: float = 0.2, sigma: float = 50.0, lambda_weight: float = 2.0) -> pd.DataFrame` — one scanner per signal type, concatenates, saves CSV to `output_path`. genes/tfs also accept comma-separated strings.

## analyze_peaks.py  [script]
Mutable-region vs. peak overlap and per-TF peak-count summary/plots from peak_scanner output. CLI: `python -m analysis.motives.analyze_peaks PEAKS_CSV [--run-directory RUN] [--excluded-genes-file JSON] [--peak_type reference] [--output-folder DIR] [-n BASE] [--expected_overlap] [--summarize-peaks] [--visualize] [--only_overlapping] [--expected-peak-locations CSV] [--figure-format png] [--value-limit X] [--top-bottom-n 3] [--all]`.

- `PEAK_GENE_COLUMN`, `PEAK_TF_COLUMN`, `PEAK_START_COLUMN`, `PEAK_END_COLUMN`, `PEAK_SCORE_COLUMN` (`"peak_area"`), `DIFF_CALC_COLUMN`, `DIVERGING_COLORMAP` (`"RdBu_r"`) — column-name / style constants.
- `analyze_wrky_peak_overlaps(peaks: pd.DataFrame, run_directory, output_folder: Path, tf_substring: str = "WRKY", peak_type: str = "reference", base_name: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]` — for each gene folder's mutable region (from `parameters.json`, end made inclusive) finds overlapping peaks of TFs containing `tf_substring` and picks the best (max overlap, then closest center); peak extent taken as `[peak_start, peak_end+250]`. Writes `{base}_{peak_type}.csv` and `..._expected_peak_locations.csv`; returns (per-gene table, `derive_peak_locations` output). Despite the name, works for any TF substring.
- `derive_peak_locations(full_results_df: pd.DataFrame) -> pd.DataFrame` — `gene_name` = 2nd `_`-token of the gene folder name, plus `target_start/target_end`.
- `filter_overlapping_peaks(peaks_df: pd.DataFrame, expected_peak_locations: pd.DataFrame) -> pd.DataFrame` — keeps peaks (gene matched by substring) whose midpoint `(start+end+250)/2` lies in the target region.
- `summarize_peaks(peaks_df: pd.DataFrame, output_folder: Path, expected_peak_locations: pd.DataFrame, base_name: str = "") -> pd.DataFrame` — wide per-TF peak counts: columns per signal type, difference peaks split into `diff_added`/`diff_removed` by sign of `peak_area`, and `diff_calc = optimized − reference` (docstring wrongly says added − removed); filters via `filter_overlapping_peaks` if locations non-empty; writes `{base}_peak_summary.csv`.
- `add_zeros(summary: pd.DataFrame) -> pd.DataFrame` — reindexes to all (tf, signal_type) combinations with 0 fill. Unused in module.
- `save_figure(fig: Figure, filename: str, output_dir, fmt: str = "png") -> Path` — `savefig` to `output_dir/filename.fmt`, dpi 150, tight bbox.
- `prepare_diff_calc_data(summary_df: pd.DataFrame) -> pd.DataFrame` — (`tf`, `diff_calc`) sorted descending.
- `select_top_bottom(plot_df: pd.DataFrame, n: int) -> pd.DataFrame` — head(n)+tail(n), or the whole frame if ≤ 2n rows.
- `compute_symmetric_limit(diff_calc_values) -> float` — max |value| (nan-safe), 1.0 if empty/zero.
- `plot_diff_calc_barplot(plot_df: pd.DataFrame, value_limit: float, title: str = "", ax: Optional[plt.Axes] = None) -> Figure` — diverging horizontal bars colored on symmetric RdBu_r scale + colorbar; ax-injectable.
- `plot_diff_calc_heatmap(plot_df: pd.DataFrame, value_limit: float, title: str = "", ax: Optional[plt.Axes] = None) -> Figure` — single-column annotated seaborn heatmap, same scale; ax-injectable.
- `visualize_peak_summary(output_folder, summary_df: pd.DataFrame, base_name: str = "", fmt: str = "png", value_limit: Optional[float] = None, top_bottom_n: int = 3) -> List[Path]` — writes bar + heatmap figures; limit computed from full data before top/bottom selection. (Docstring claims it auto-locates the CSV; it takes `summary_df`.)
- `build_parser() -> ArgumentParser` — CLI parser. Note: `--only_overlapping` only triggers a validation error; filtering actually happens whenever expected locations are available.

## deepcis_visualize.py  [library + script]
Per-gene × TF line plots of deepCIS window scores (reference, optimized, difference), with optional padding / TSS-TTS / peak / mutation markers. CLI: `python -m analysis.motives.deepcis_visualize --input SCAN_CSV [--peaks PEAKS_CSV] [--output DIR] [--genes ...] [--tfs ...] [--format png] [--random-subset] [--no-title] [--[no-]highlight-padding] [--[no-]show-tss-tts] [--tss-position 1000] [--tts-position 2020] [--[no-]highlight-peaks] [--peak-signals ...] [--[no-]show-difference] [--difference-direction ...]`; saves `OUTPUT/<gene>/<gene>_<tf>[_<signal>].png`.

- `PEAK_SIGNAL_TYPES` — ("reference", "optimized", "difference").
- `DIFFERENCE_DIRECTIONS` — ("optimized_minus_reference", "reference_minus_optimized").
- `PEAK_BACKGROUND_COLORS` — signal type → shading color.
- `PEAK_REQUIRED_COLUMNS` — {gene, tf, signal_type, peak_start, peak_end}.
- `MUTATION_MARKER_COLOR` (`"#d62728"`), `MUTATION_MARKER_LINESTYLE`, `MUTATION_MARKER_LINEWIDTH` — style of introduced-mutation markers (imported by paper_plots).
- `load_scan_results(scan_path: str) -> pd.DataFrame` — read scan CSV (existence check only).
- `load_peak_results(peak_path: str) -> pd.DataFrame` — read peak CSV, checks `PEAK_REQUIRED_COLUMNS`.
- `get_tf_columns(df: pd.DataFrame) -> List[str]` — sorted non-metadata columns, excluding any containing `"__"`.
- `get_centers(data: pd.DataFrame) -> np.ndarray` — window centers `(window_start + window_end + 1)/2` (1-based convention).
- `extract_plot_data(df_subset: pd.DataFrame, tf_col: str, difference_direction: str = "optimized_minus_reference") -> Tuple[ref_x, ref_y, mut_x, mut_y, diff_x, diff_y, x_min, x_max]` — splits one gene's rows into reference/optimized curves at window centers plus signed difference on inner-joined windows.
- `_plot_gene_tf(df_subset, gene, tf_col, ax, include_title=True, highlight_padding=False, show_tss_tts=True, tss_position=1000.0, tts_position=2020.0, highlight_peaks=False, peak_signal_types=None, peak_df=None, show_difference=True, difference_direction="optimized_minus_reference", mutation_positions: Optional[Sequence[float]] = None, mutation_marker_label="Introduced mutations") -> None` — private name but the main reusable ax-level plot (imported by `workflows/paper_plots/fig1_miniatures.py`, `compose_plots_mpl.py`). Draws ref (black), optimized (blue), difference (red) curves, ylim [0,1] or [−1,1], optional markers/peak shading (peak bounds shifted +125 to center coords).
- `_get_padding_edge_positions(df_subset) -> List[float]` — private but imported by `workflows/one_off_visualizations/vis.py`; centers of first/last padding-containing windows.
- `visualize_scan_results(input: Union[str, pd.DataFrame], genes=None, tfs=None, output_dir=None, output_format="png", include_title=True, highlight_padding=True, show_tss_tts=True, tss_position=1000.0, tts_position=2020.0, highlight_peaks=False, peak_signal_types=None, show_difference=True, difference_direction="optimized_minus_reference", random_subset=False, *, peak_input=None) -> None` — batch-save one 12×6 figure per gene × TF (separate image per peak signal type when >1); default output dir `deepcis_scan_plots/` next to input.
- `parse_arguments(args=None)` / `run_visualization(args) -> int` — CLI parsing and runner.

## analyze_mapping.py  [script]
Old MOCA-blue EPM motif-mapping analysis (EPM counts per sequence, initial vs. final, per-mutation fitness effect per EPM). Not imported elsewhere. CLI: `python -m analysis.motives.analyze_mapping (-i NEW_CSV | -io OLD_CSV) [-o OUT] [-d DATA_PATH] [--difference_epm] [--compare_distribution] [-m] [--all]`.

- `get_epm_tfbs_mapping_old(input_path: str) -> pd.DataFrame` — unique (`epm`, `tf_name`) pairs from an old-pipeline CSV.
- `get_epm_tfbs_mapping_new(input_path: str) -> pd.DataFrame` — **[broken]** explicitly "NOT PROPERLY IMPLEMENTED"; fills `tf_name` with `"not_implemented"`. Used by `main` for new-format input.
- `add_epm(data: pd.DataFrame) -> None` — in place; derives `epm` from `motif` by stripping trailing `[RF]_<n>`.
- `get_epm_count_table(input_path: str, new: bool = True) -> pd.DataFrame` — sequence × EPM count table of forward-strand motifs (`p\d+m\d+F_\d+$`); side effect: writes `<input>_epm_counts.csv`.
- `add_original_sequence_and_number_mutations(df: pd.DataFrame) -> None` — in place; parses `<orig>_mutations_<n>...` from `sequence`.
- `get_min_max_dfs(input_df: pd.DataFrame, remembered_issues: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]` — rows with 0 and max mutations per original sequence; **raises ValueError unless `remembered_issues=True`** (reminder that some 0-mutation rows are missing).
- `get_difference_number_epm(input_path: str, output_folder: str = ".", new: bool = True)` — total EPM count at 0 vs. max mutations per sequence, difference + log2 ratio + sum row; writes CSV.
- `statistics_epm_before_after(epm_counts: pd.DataFrame) -> Dict[str, float]` — per `epm_*` column, paired test 0 vs. max mutations: `normaltest` both p>0.05 → `ttest_rel`, else `wilcoxon`; returns p-values.
- `add_tf_names(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame` — merges mapping on index = `epm`.
- `compare_initial_and_final_distribution(input_path: str, mapping: pd.DataFrame, output_folder: str = ".", new: bool = True)` — per-EPM summed counts first vs. last sequence per gene, log2 ratio, p-values; writes CSV. First/last chosen by **lexicographic** sort of sequence names and `startswith(gene)` — (unclear) whether that equals 0/max mutations.
- `add_fitness(df: pd.DataFrame, data_path: str) -> None` — in place; looks up fitness per row from `data_path/<run>/<gene>/saved_populations/[<gene>_]pareto_front.json` (sequence name `run__gene_mutations_n`), plus per-mutation fitness diffs.
- `find_diff_tf(epm_cols: List[str], diff_epms: pd.Series) -> Tuple[Optional[str], int]` — returns the single EPM column that changed by exactly ±1, else None.
- `add_diff_tf(df: pd.DataFrame) -> pd.DataFrame` — **[possibly buggy]** iterates positionally (`iloc[i]`) but writes by label (`df.loc[i]`); `analyze_mutations_effect` calls it on a sorted frame without `reset_index`, so results can land on wrong rows.
- `analyze_mutations_effect(input_path: str, output_folder: str, mapping: pd.DataFrame, data_path: str, epm_of_interest: str = ..., new: bool = True) -> pd.DataFrame` — per (EPM, +/−) group, mean per-mutation fitness change vs. all other groups (normaltest → t-test else Mann-Whitney U); writes `effect_per_mutation_epm.csv`. `epm_of_interest` unused.

## annotate_individual_mutation.py  [one-off]
**[broken]** Work-in-progress: `raise NotImplementedError` at import time, so nothing here is importable. Contains drafts of `compare_sequences_df` (ref vs. optimized per-window TF deltas), `annotate_mutations_in_windows`, and `add_genomic_coordinates` (window → genomic coords via `analysis.mutations.genomic_annotation.AnnotatedMutatedSequence.parse_sequence_name`).

Note: CLAUDE.md lists `deepcis_annotation.py`; no such file exists (the corresponding `test/analysis/motives/test_deepcis_annotation.py` is almost fully commented out).
