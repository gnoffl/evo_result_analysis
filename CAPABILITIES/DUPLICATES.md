# Duplicated functionality

Found while building the index. Nothing here has been consolidated. When you need one of these, use the
**preferred** copy and don't add another. Paths are relative to `src/`.

## Exact or near-exact copies

- `save_figure(fig, filename, output_dir, fmt)`: 7 definitions. **Preferred:** `analysis/motives/analyze_peaks.save_figure` (returns the Path). The copies are in mutation_distribution_analysis/analyse_results, natural_unconstrained_comparison, hoffie/mutation_assessment/assess_mutations, overlap_analysis/correct_construct/analyze_in_true_background, and compare_reruns/compare_runs. The one in example_gene_pareto has a different signature.
- `reverse_complement`: 3 copies. overlap_analysis/_common and analysis/starrseq/starrseq_analysis are identical (case-preserving `str.maketrans`). workflows/candidate_selection uses a dict over ACGTN and raises on lowercase.
- `load_json`: identical in starrseq_v2/selection/pool_front_table and build_library.
- `find_gene_folders`: analysis/motives/deepcis_scanner requires both `parameters.json` and `pareto_front.json`; adversarial/reevaluate requires only `parameters.json`.
- `load_mutation_data` in analysis/mutations/analyze_mutations duplicates `summarize_mutations.load_mutations_from_json`. The function of the same name in compare_methods does something else and is broken.
- Significance stars:
  - **Preferred:** `analysis/utils/statistics.qvalue_to_stars`.
  - `tf_comparison_plot.q_to_stars` is just an alias of it.
  - `natural_unconstrained_comparison.p_value_to_stars` uses its own thresholds and returns "ns".
  - paper_plots also has private star helpers.
- Wilcoxon against 0: **preferred** `analysis/utils/statistics.wilcoxon_pvalue_vs_zero`; overlap_analysis/binding_vs_nonbinding has a private copy.
- Paired Wilcoxon: `natural_unconstrained_comparison.wilcoxon_paired_test` vs the private one in compare_runs.
- Rank-biserial effect size: `natural_unconstrained_comparison.rank_biserial_effect_size` vs the private one in binding_vs_nonbinding.
- `DEEPCRE_PATH` constant: defined in both overlap_analysis/_common and analyze_in_true_background.
- Box outline recolouring: `runtime_benchmark_plot.color_box_outlines` vs private helpers in region_mutation_breakdown.

## Same job, different implementations

- Loading and traversing Pareto fronts: reimplemented in about 16 modules. **Preferred:** `deepcis_scanner.GeneRunData` (one gene folder) and `simple_result_stats` helpers. ⚠ `candidate_selection.get_data_at_mutation_count` and `simple_result_stats.get_min_mutation_count_for_fitness` assume gap-free monotone fronts.
- Gene ID from folder or record names: `clean_gene_id`, `extract_gene_key`, `core_gene_id`, `candidate_selection.get_gene_name_from_path` (splits on `_`, which breaks on chromosome names that contain underscores), `build_library.parse_record_name`, `genomic_annotation.parse_sequence_name` (buggy), and 3 parsers in hoffie. **Preferred:** external `evolution.write_run_script.gene_id_from_record`, or `parse_record_name`.
- VCF parsing: hand-written line loops in 4 functions of evo_alg_pooled_plots/natural_unconstrained_comparison and natural_gof_lof_top_deltas. A pandas version is `mutation_distribution_analysis/natural_variation_in_window.read_vcf_positions`.
- FASTA I/O: `write_fasta` exists 3 times with different input types (dict, DataFrame, list of tuples), and `read_fasta` is hand-rolled twice. Elsewhere the code uses `pyfaidx.Fasta`.
- "Load .h5 model → one-hot → predict" is repeated in about 7 places. The one-hot encoder comes from `deepCRE.utils.one_hot_encode` in some and `evolution.sequences.one_hot_encode` in others.
- Applying (pos, src, new) SNPs to a sequence: private `_apply_mutations` in both baseline_sampler and summarize_mutations.
- Restriction-site screening: `candidate_selection` (generic, driven by a FASTA of sites) vs `build_library.has_restriction_site` (BsaI plus a cloning-suffix rule).
- Linear fit on a scatter: done three ways, in overlap_analysis/_common (polyfit + Spearman), analyze_in_true_background (linregress + Pearson) and deepCRE_TPM_correlation/correlate.
- Front "area" (Σ fitness of the expanded front): computed separately in compare_methods `calculate_differences_between_fronts`, `get_best_area` and `get_area`.
- `adversarial/plot.expand_gene_front` is a row-level version of `simple_result_stats.expand_pareto_front`.
- Two different functions are both called `load_starrseq_data`: one in overlap_analysis/_common (parses FASTA headers), one in analyze_in_true_background (reads a CSV).
- TF column detection and the optimized − reference difference signal are each computed twice in analysis/motives (peak_scanner vs deepcis_visualize).
- Peak coordinates are converted out of window-start space ad hoc: analyze_peaks adds +250 to the end, deepcis_visualize adds +125 to both ends.
