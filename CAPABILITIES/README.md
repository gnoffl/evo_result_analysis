# Capability index

Read this before writing new code. Goal: reuse what exists instead of reimplementing it.
Look for your task in **By task** below, then open the linked detail file for exact signatures and caveats.
Detail files list public functions only, with `[library]` / `[script]` / `[one-off]` tags and **[broken]** / **[bug]** flags.
Also check `DUPLICATES.md`: if what you need exists in several copies, it names the one to use.

External packages `evolution` and `deepCRE` (not in this repo) hold coordinate/extraction logic, `one_hot_encode`,
model loading and run-config helpers — prefer those over reimplementing.

## Keeping this index current (mandatory)

When you add, rename, delete or change the signature/behaviour of a public function, class or constant in `src/`
or `analysis_scripts/`, update the matching detail file in the same change, using the same format. A new folder
gets a new detail file plus a row in the table below. Add a line under **By task** only for helpers that are
genuinely reusable (not workflow glue). Private `_helpers` are not indexed.

## Conventions worth knowing

- `pareto_front.json` entries are `[sequence, fitness, n_mutations]`; stored order is front[0] = optimized endpoint,
  front[-1] = reference. Fronts can have gaps and ties in mutation count (41 % of starrseq_v2 fronts have gaps) —
  helpers that assume a gap-free monotone front are marked.
- Extraction frame: 3020 bp, TSS at 1000, TTS at 2020, central 20 bp N-padding at 1500–1520.
- deepCIS scan/peak coordinates: `window_start` 0-based; peaks are in window-start space (see analysis-motives.md).

## Folders

| Folder | What it does | Detail |
|---|---|---|
| analysis/motives | deepCIS sliding-window scan, peak calling, peak summary, signal-track plots; legacy EPM mapping | [analysis-motives.md](analysis-motives.md) |
| analysis/blamm | PWM TFBS counting with the external `blamm` binary, per-motif Wilcoxon+BH significance | [analysis-blamm.md](analysis-blamm.md) |
| analysis/overview | Per-run stats (start/final fitness, half-max mutations), average fronts, convergence loss; method comparison | [analysis-overview.md](analysis-overview.md) |
| analysis/mutations | `MutatedSequence`/`MutationsGene` data model + JSON; positional/conservation/distance stats; genomic coords | [analysis-mutations.md](analysis-mutations.md) |
| analysis/starrseq | Exploratory STARR-seq v1 WRKY/bHLH enrichment analysis (hardcoded, several bugs) | [analysis-starrseq.md](analysis-starrseq.md) |
| analysis/utils | Canonical significance recipe (Wilcoxon vs 0 → BH → stars); console progress printing | [analysis-utils.md](analysis-utils.md) |
| workflows/paper_plots | Publication style module + main/supplementary figure composition (ax-injected panels) | [workflows-paper_plots.md](workflows-paper_plots.md) |
| workflows/evo_alg_pooled_plots | Cross-run analyses: TF comparison heatmaps, runtime benchmarks, natural vs unconstrained, example-gene fronts | [workflows-evo_alg_pooled_plots.md](workflows-evo_alg_pooled_plots.md) |
| workflows/overlap_analysis | STARR-seq × deepCRE agreement (fragment → 3020 bp window, correlation by position/model/species) | [workflows-overlap_analysis.md](workflows-overlap_analysis.md) |
| workflows/starrseq_v2 | STARR-seq v2: gene/window selection for runs, then front pooling → candidate selection → oligo library | [workflows-starrseq_v2.md](workflows-starrseq_v2.md) |
| workflows/mutation_distribution_analysis | Null model: random baseline sequences with matched SNP counts (conditional-Poisson sampler), scored by deepCRE | [workflows-mutation_distribution_analysis.md](workflows-mutation_distribution_analysis.md) |
| workflows/mutation_distance_analysis | Inter-mutation distances vs random null (KS, Wasserstein) | [workflows-mutation_distance_analysis.md](workflows-mutation_distance_analysis.md) |
| workflows/adversarial | Rescore fronts with sibling MSR models (adversarial check) + plots | [workflows-adversarial.md](workflows-adversarial.md) |
| workflows/compare_reruns | Compare two runs on shared genes (fitness, mutation overlap, paired Wilcoxon) | [workflows-compare_reruns.md](workflows-compare_reruns.md) |
| workflows/hoffie | Hoffie-lab maize plasmid constructs (mostly hardcoded scripts) | [workflows-hoffie.md](workflows-hoffie.md) |
| workflows/deepCRE_TPM_correlation | deepCRE prediction vs leaf TPM correlation | [workflows-deepCRE_TPM_correlation.md](workflows-deepCRE_TPM_correlation.md) |
| workflows/one_off_visualizations | Scratch plots | [workflows-one_off_visualizations.md](workflows-one_off_visualizations.md) |
| workflows/*.py (top level) | GOF/LOF candidate selection + restriction-site screening; SVG figure composition; Excel update | [workflows-toplevel.md](workflows-toplevel.md) |
| analysis_scripts/ | Bash pipelines: deepCIS scan → peaks → plots; stats → mutation summary → mutation analysis | [analysis_scripts.md](analysis_scripts.md) |

## By task

Paths are relative to `src/`. ⚠ marks a caveat explained in the detail file.

### Run folders, Pareto fronts, gene IDs
- One gene folder (front + params + full reference): `analysis/motives/deepcis_scanner.GeneRunData.load_gene_run_data`; list gene folders: `deepcis_scanner.find_gene_folders`
- Front helpers: `analysis/overview/simple_result_stats.get_pareto_front`, `deduplicate_pareto_front`, `expand_pareto_front`, `normalize_front`
- Per-run stats cached as `stats_<name>.json` (start/final fitness, half-max mutations): `simple_result_stats.get_stats_per_gene`, `calculate_half_max_mutations`
- All generations as mutation objects: `analysis/mutations/summarize_mutations.MutationsGene`, `summarize_mutations_all_folders`, `load_mutations_from_json`
- Pick entry at max / exact mutation count: `analysis/blamm/blamm_tfbs_counts.select_pareto_entry`; ⚠ `workflows/candidate_selection.get_data_at_mutation_count` assumes gap-free fronts
- Front → DataFrame with gap/tie handling: `workflows/starrseq_v2/selection/pool_front_table.front_entries`
- One representative per mutation count: `workflows/adversarial/reevaluate.select_front_representatives`
- All saved generations / rejected individuals of one gene: `workflows/evo_alg_pooled_plots/example_gene_pareto/example_gene_pareto.all_population_files`, `load_final_front_and_rejected`
- Gene-ID substring matching against folder names: `analysis/motives/gene_matching.resolve_gene_ids`
- Gene ID from names (⚠ different conventions): `adversarial/reevaluate.clean_gene_id` (→ `AT1G01720`), `compare_reruns/compare_runs.extract_gene_key` (→ `1_AT1G01720`), `starrseq_v2/selection/build_library.parse_record_name`, external `evolution.write_run_script.gene_id_from_record`

### Sequences and coordinates
- Splice mutable region into full reference: `deepcis_scanner.get_full_sequence`; splice fragment into window + one-hot: `workflows/overlap_analysis/_common.build_sequences`
- Sliding windows / N-padding detection: `deepcis_scanner.slide_windows`, `find_padding_region`
- Assign position to promoter/UTR/... region: `evo_alg_pooled_plots/natural_unconstrained_comparison/region_mutation_breakdown.assign_region`, `REGIONS`
- Reverse complement, Hamming distance: `overlap_analysis/_common.reverse_complement`, `compare_sequences`
- Restriction sites: `workflows/candidate_selection.get_restriction_regexes`, `get_sequences_without_restriction_sites` (generic); `starrseq_v2/selection/build_library.has_restriction_site` (BsaI incl. cloning suffix)
- Extract gene frames from a genome: `starrseq_v2/run/build_starrseq_v2_library.extract_genes` (wraps `evolution.extract_sequences.find_genes` / `extract_string`), `splice`
- VCF positions: `workflows/mutation_distribution_analysis/natural_variation_in_window.read_vcf_positions`
- FASTA writing: `analysis/blamm/blamm_tfbs_counts.write_fasta` (list of tuples); reading: use `pyfaidx.Fasta`
- Sequence position → genome coordinate: ⚠ `analysis/mutations/genomic_annotation.AnnotatedMutatedSequence.from_mutated_sequence` (unverified; prefer evolution/deepCRE)

### Model prediction
- deepCIS: `deepcis_scanner.load_deepcis_model`, `predict_windows`, `scan_single_gene_folder`, `scan_all_genes`; output names `TF_FAMILY_NAMES`
- deepCRE on sequences: `overlap_analysis/_common.make_deepcre_predictions`; `workflows/adversarial/reevaluate.load_models`, `predict_sequences`; FASTA → CSV: `mutation_distribution_analysis/evaluate_sequences.evaluate_sequences` (⚠ module path constants)
- Model registries (per species / model group): `overlap_analysis/model_species_comparison/_models.list_species_models`, `_model_groups.MODEL_GROUPS`, `list_group_models`

### TF binding / motifs
- Peak calling on a window signal: `analysis/motives/peak_annotation.PeakAnnotator`; over a whole scan: `peak_scanner.DeepCISPeakScanner`, `peak_scanner.run`
- Peak summaries / overlap with mutable region: `analysis/motives/analyze_peaks.summarize_peaks`, `filter_overlapping_peaks`, `analyze_wrky_peak_overlaps`
- Per-TF change matrices + significance across runs: `evo_alg_pooled_plots/tf_comparison/tf_comparison_calc.load_run_diffs`, `build_matrix`, `paired_tf_significance`, `single_run_tf_significance`; heatmap: `tf_comparison_plot.plot_heatmap`
- PWM scanning: `analysis/blamm/meme_to_jaspar.convert_meme_databases`, `blamm_tfbs_counts.analyse_run`, `blamm_significance.motif_significance`

### Mutation analysis
- Diff two sequences into mutation list / compact string: `summarize_mutations.MutatedSequence`; set of (pos, base): `compare_reruns/compare_runs.mutation_set`
- Positional / net nucleotide change, conservation, gaps, span: `analysis/mutations/analyze_mutations.calculate_positional_nucleotide_change`, `calculate_net_nucleotide_change`, `calculate_conservation_statistic`, `calculate_mutation_distances_single_gene`, `analyze_range_single_gene`
- Mutation spans across runs: `evo_alg_pooled_plots/mutation_range/mutation_span.collect_run_spans`, `select_sequences_with_mutation_count`
- Mutations in VCF-allowed positions / per region: `evo_alg_pooled_plots/natural_unconstrained_comparison/` (see detail file)
- PCA of mutation-presence vectors: `analysis/overview/compare_methods.pca_transform_all_methods`

### Sampling / null models
- k distinct weighted items, exact inclusion probabilities: `mutation_distribution_analysis/conditional_poisson.PositionSampler`, `sample_positions`, `inclusion_probabilities`
- Pooled SNP pool + matched random baselines: `mutation_pool.MutationPool`, `baseline_sampler.GeneMutationSampler`, `generate_baselines_fasta`
- Random inter-mutation distances: `workflows/mutation_distance_analysis/mutation_distance_analysis.compute_random_distances`

### Statistics
- **Canonical** Wilcoxon vs 0 → BH → stars: `analysis/utils/statistics.wilcoxon_pvalue_vs_zero`, `benjamini_hochberg_qvalues`, `qvalue_to_stars`
- Paired Wilcoxon + rank-biserial: `natural_unconstrained_comparison/natural_unconstrained_comparison.wilcoxon_paired_test`, `rank_biserial_effect_size`
- Pooled percent with gene-level cluster-bootstrap CI: `natural_unconstrained_comparison/actual_mutation_region_breakdown.pooled_percent_with_bootstrap_ci`, `build_pooled_ci_dataframe`
- Logistic fit: `simple_result_stats.fit_logit_linear`; log-log power-law fit: `evo_alg_pooled_plots/benchmarks/runtime_benchmark_calc.fit_log_log_scaling`
- Correlation by position / sliding window: `overlap_analysis/_common.compute_correlation_by_position`, `compute_overlay_correlation_data`

### Plotting and figures
- Publication style: `workflows/paper_plots/style.publication_style`, `figure_size_inches`, `mm_to_inch`, `panel_label`, `sync_axis_limits`, `broken_y_axes`, `save_publication_figure` (⚠ `.svg` only, writes PNG twin)
- Figure composition: paper_plots `compose_plots_mpl` / `supplementary_plots_mpl` (inject `ax` into existing plot functions); SVG stitching: `workflows/figure_composition_svg.compose_figures` (SVG inputs only)
- Save a figure: `analysis/motives/analyze_peaks.save_figure` (see DUPLICATES.md)
- deepCIS signal track on an ax: `analysis/motives/deepcis_visualize._plot_gene_tf` (private name, used by paper_plots)
- Average normalized front, half-max histogram, start-vs-final scatter (ax-injectable): `simple_result_stats.show_average_pareto_front`, `hist_half_max_mutations`, `draw_visualize_start_vs_max_fitness_by_mutations`
- Nucleotide colours / stacked bars / rolling position plot: `analyze_mutations.COLORS`, `plot_dict_as_stacked_bars`, `make_line_plot_rolling_window` (⚠ TTS tick at 2000)
- Box outline recolouring: `evo_alg_pooled_plots/benchmarks/runtime_benchmark_plot.color_box_outlines`; gradient colours: `example_gene_pareto.sample_gradient_colors`

### STARR-seq
- Fragment → deepCRE window mapping and deltas: `overlap_analysis/_common.map_starrseq_to_deepcre`, `merge_with_starrseq_results`, `calculate_deltas`
- Enrichment tables: `overlap_analysis/starrseq_deepcre_correlation_WRKY.prepare_wrky_enrichment_df`, `starrseq_deepcre_correlation_bHLH.prepare_bhlh_enrichment_df`
- v2 library pipeline: `starrseq_v2/selection/pool_front_table` → `select_candidates.select_candidates` → `build_library` (`deduplicate`, `write_upload_sheet`)

### Provenance and misc
- Checksums, git commit, input record: `workflows/adversarial/reevaluate.compute_file_checksum`, `get_analysis_commit`, `write_input_record`; `analysis/blamm/blamm_tfbs_counts.collect_provenance`
- Progress printing for long workflows only: `analysis/utils/io.py`
- Run a tool inside a conda env: `starrseq_v2/run/flowering_data/find_nicotiana_orthologs.conda_run`
