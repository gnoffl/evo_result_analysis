# src/analysis/starrseq/

Exploratory analysis of a plant STARR-seq experiment that tested WRKY/bHLH candidate-hub variants (reference, binding and non-binding designs). Covers enrichment of binding vs non-binding variants (Mann-Whitney, t-test, permutation), deepCIS predictions on the tested sequences, core/flank annotation via evolution's EPM position finders, flank ANOVA/chi², and a linear model of enrichment on deepCIS TF scores. All paths are hardcoded (relative `data/...` and absolute ARCitect paths); most functions print results or save plots to the CWD. Nothing else in the repo imports it.

## starrseq_analysis.py  [script]
Exploratory script; `python -m analysis.starrseq.starrseq_analysis` runs `starrseq_analysis_deep()`, and other analyses are switched on by uncommenting lines in `__main__`. Needs tensorflow, pyfaidx, sklearn, deepCRE and evolution.

- `reverse_complement(seq: str) -> str` — maps ACGT/acgt only; other characters are kept and reversed.
- `load_df() -> pd.DataFrame` — reads `data/plantstarr-seq_main_dark_simon_gernot.csv` (relative to the CWD); adds `reference` (id contains "reference"), `binding` (id lacks "non_binding") and `base_sequence` (text between the first two "_").
- `simplest_stats() -> None` — per TF (WRKY/bHLH), prints means and a two-sided Mann-Whitney test of binding vs non-binding enrichment.
- `visualize_simple(ax=None) -> None` — per-TF boxplot of binding vs non-binding (`enrichment_<tf>.pdf`); with `ax`, both TFs are drawn on top of each other.
- `visualize_simple_2(ax=None) -> None` — grouped boxplot of TF × binding status (`enrichment_WRKY_bHLH.png`).
- `calc_averages() -> None` — prints mean enrichment of references, binding and non-binding variants.
- `describe_data() -> None` — prints variants per base_sequence and the binding/non-binding split.
- `analyze_differences_permutation() -> None` — permutation test of mean |binding − non-binding|, shuffling labels within each base_sequence (10 000 permutations). **[bug]** the return tuple is unpacked in the wrong order, so the printed "P-value" is actually the mean of the null distribution.
- `analyze(exclude_references=True) -> None` — compares |enrichment − reference enrichment| between binding and non-binding, choosing t-test or Mann-Whitney by normaltest. **[bug]** the choice is inverted: it uses the t-test when both normaltest p < 0.05, i.e. when normality is rejected.
- `visualize_differences(ax=None) -> None` — histograms of |Δ enrichment| for binding vs non-binding (`<tf>_enrichment_differences_hist.png`).
- `starrseq_sequences_for_deepCIS_predictions() -> None` — **[broken]** `make_sequences` returns None, so the `"\n".join(...)` write fails.
- `make_sequences(ids, source_sequences: Fasta, submitted_sequences: Fasta)` — builds FASTA lines extended by 40 bp of source flank on each side and checks that references match the source. **[broken]** it has no return statement.
- `make_deepCIS_predictions() -> (wrky_df, bHLH_df)` — one-hot encodes (wraps deepCRE.utils.one_hot_encode) and predicts with a keras deepCIS model at a hardcoded path (`.../Evolution/models/deepCIS_model_chrom_1_model.h5`) on the extended sequences. Columns are TF names from the hub metadata CSV; writes `data/deepCIS_predictions_<TF>.csv`.
- `get_match_score(sequence: str, motif: str) -> int` — number of case-insensitive position-wise matches over the zipped length.
- `get_better_match(sequence, forward_motif, reverse_motif) -> str` — the higher-scoring motif (ties go to forward).
- `find_motifs() -> (wrky_data, bhlh_data)` — runs the predictions, then find_motif_epm for WRKY (GTCAA/TTGAC, length 5) and bHLH (CACGTG, length 6).
- `find_motif_epm(target, motif_length, data, forward_motif, reverse_motif, output_path, position_finder: Callable) -> pd.DataFrame` — adds the reference sequence, core start/end, clean reference core and orientation for each variant. The position comes from `position_finder` (wraps evolution.write_run_script.find_correct_WRKY_position / find_correct_bHLH_position).
- `add_starrseq_results() -> (wrky_data, bhlh_data)` — joins enrichment onto the motif tables and writes CSVs.
- `binding_boxplots(data: Dict[str, pd.DataFrame], name_thing: str, ax=None)` — enrichment boxplots by category × binding status with n labels (`binding_status_<name>_boxplots.pdf`).
- `binding_vs_non_binding(data, name) -> None` — splits by whether the core and/or flanks are mutated; boxplots plus a Mann-Whitney test per split.
- `apply_flanking(row, sequence_col="sequence")` / `apply_core(row, sequence_col="sequence")` — the sequence outside / inside [start:end].
- `apply_leading(row, sequence_col="sequence")` / `apply_trailing(row, sequence_col="sequence")` — the flank base next to the core, orientation-aware (for reverse orientation, the opposite flank reverse-complemented).
- `calculate_combined_frequencies(leading: pd.Series, trailing: pd.Series) -> pd.Series` — expected counts of two-flank combinations under independence.
- `flank_combination_chi_squared(data) -> None` — chi² test of observed vs expected flank combinations.
- `flank_combinations(data) -> None` — one-way ANOVA of enrichment across flank combinations, plus the chi² test.
- `leading_trailing_anova_enrichment(data) -> None` — ANOVA by leading and by trailing flank.
- `tf_linear_model(data) -> None` — sklearn LinearRegression of enrichment on the first 46 columns (assumed to be deepCIS TF scores); prints R² and coefficients.
- `starrseq_analysis_deep() -> None` — runs the deep analysis for WRKY, bHLH and both combined.
