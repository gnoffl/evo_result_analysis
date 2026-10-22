# src/workflows/hoffie/

Collaboration workflow (Hoffie lab, maize/Ubi1 plasmid constructs): post-processing of EA pareto-front FASTAs into lab-ready sequences — keep reference/max-mutation/chimeric variants, strip restriction sites, splice optimised promoters into the pIK74 plasmid — plus SSR-vs-MSR prediction comparisons and a check of lab-introduced mutations. Mostly hard-coded one-off scripts run as `__main__`; inputs live in `data/` (GOF_LOF pareto-front FASTAs, `ubi/`, `zea/`, `restriction_sites.fa`, pIK74 GenBank), outputs written back into `data/` or `mutation_assessment/`. Pareto-front FASTA headers follow `<run>__<chr>_<gene>_gene:<start>-<end>_<ts>_fitness_<pred>_mutations_<NNN>[_<v>]`.

## compare_predictions_SSR_MSR.py  [script | library]
Re-predict GOF/LOF pareto-front FASTA records (`data/GOF_LOF/*.fa`) with a hard-coded maize SSR model (`Zmay_S0X0...ssr...h5`, also applied to the Arabidopsis AT* genes) and compare to the MSR fitness stored in the header; uses `deepCRE.utils.one_hot_encode`. Run directly (main plots 6 hard-coded AT genes for `GOF_single` runs). `parse_single_name` is imported by `select_best_candidate_and_merge_into_plasmid.py`.

- `plot_single_gene_scatter(ssr_predictions: np.ndarray, msr_predictions: np.ndarray, gene_name: str, output_path: str, mutations: List[int])` — scatter SSR and MSR prediction vs. mutation count; saves `{gene_name}_ssr_msr_comparison.png` (opens a new figure each call, never closed).
- `plot_single_gene_diff(ssr_predictions: np.ndarray, msr_predictions: np.ndarray, gene_name: str, output_path: str, mutations: List[int])` — scatter SSR−MSR vs. mutation count; saves `{gene_name}_ssr_msr_difference.png`.
- `plot_single_run(data: pd.DataFrame, output_path: str)` — both plots per `Base_Name` (cols `Base_Name, SSR_Predictions, MSR_Predictions, Mutations`).
- `plot_goi_pareto_comparison(genes_of_interest: List[str], output_path: str, runs_of_interest: List[str])` — for `data/GOF_LOF` FASTAs whose name contains a run substring: parse, SSR-predict, plot per gene.
- `parse_single_name(name: str) -> Tuple[str, float, int]` — parse a pareto FASTA header into (base_name, fitness, n_mutations); handles optional trailing version suffix. Note returned base_name still ends in `_fitness`.
- `parse_fasta(fasta_path: str, genes: List[str]) -> Tuple[List[float], List[str], List[int], List[str]]` — (header fitnesses, sequences, mutation counts, base names) for records whose name contains any of `genes` (all records if `genes` empty).
- `compare_predictions(genes: List[str])` — SSR-predict all `data/GOF_LOF` records in 512-chunks, print min/max/mean of SSR−MSR and log(SSR/MSR).

## create_filtered_sequences.py  [script]
**[broken]** imports `get_sequences_without_restriction_sites` from `analysis.candidate_selection`, which does not exist (function lives in `workflows/candidate_selection.py`) — module fails at import. CLI: `-d/--data-dir` (default `./data`), `-c/--create_chimeras`, `-r/--remove_restriction_sites` with `-s/--sequence_path` and `-rp/--restriction_site_path`.

- `extract_gene_and_mutation_count(header)` — regex gene id (`__<single digit>_<gene>_gene`) and `mutations_(\d+)` from a pareto header; (None, None) if no match.
- `group_by_gene(fasta)` — pyfaidx Fasta -> {gene: [(n_mutations, header, seq), ...]} sorted by mutation count, first record per (gene, count) kept.
- `create_chimeric_sequence(ref_seq, max_seq)` — first half of `max_seq` + second half of `ref_seq` (split at len//2).
- `create_filtered_file(input_file, output_file)` — per gene write reference (0 mutations), max-mutation, and chimeric (`mutations_chimeric_NNN`) record, 80-char lines; genes without a 0-mutation record skipped.
- `filter_restriction_sites(sequence_path: str, restriction_site_path: str) -> None` — for a file or every .fa/.fasta in a dir, write `*_no_restriction_sites.fa` via `get_sequences_without_restriction_sites`.
- `create_chimeras(data_dir: Path)` — run `create_filtered_file` on every `*.fa` in `data_dir` -> `<stem>_filtered.fa`.

## select_best_candidate_and_merge_into_plasmid.py  [one-off]
Hard-coded: sum |front[0] − front[-1]| fitness per Ubi transcript (T01–T03) over `data/ubi/run_results/*`, and splice optimised promoters (trailing N stripped) into pIK74 (`data/{ubi,zea}/pIK74.fa`, record `PLASMID`, replacing bp 1900:3302 resp. 1900:3890) to write `merged_plasmids.fa`. Imports `parse_single_name` but never uses it.

## visualizations.py  [one-off]
Two bar plots of hard-coded predicted expression values (Ubi1 T03 runs; ZmEXPA6 reference/max_nat/max) saved into `data/ubi/` and `data/zea/`.

## mutation_assessment/assess_mutations.py  [script]
One-off check whether mutations found by the lab (poly-C insertion) change Zmay SSR deepCRE predictions: rebuilds the 3020 bp input (first 1500 bp of the optimised T03 `_mutations_045` record + bp 1500: of `T03_full_ref.fasta`), predicts original vs. lab variant, writes CSV + bar plot. Uses `evolution.sequences.one_hot_encode`. Run directly; `--plot-only` re-plots from `prediction_results.csv`. Tests: `test/workflows/hoffie/test_assess_mutations.py`.

- `MODEL_PATH`, `SEQ_DIR`, `BACKGROUND_PATH`, `OUT_CSV`, `OUTPUT_PATH`, `PLOT_OUTPUT_DIR` — hard-coded paths (Zmay SSR .h5, `data/ubi/sequences`, outputs in `mutation_assessment/`).
- `CONDITIONS` — condition -> (pareto FASTA filename, 5' flank of poly-C run or None).
- `get_t03_seq(fasta_file: str) -> str` — uppercase sequence of the first `_T03_` header ending `mutations_045` in `SEQ_DIR/fasta_file`.
- `insert_poly_c(promoter: str, flank: str) -> str` — insert one C right after the first occurrence of `flank`, truncate to 1500 bp.
- `save_generated_sequences(names: List[str], variants: List[str], seqs: List[str]) -> None` — write `>{name}_{variant}` FASTA to `OUTPUT_PATH`.
- `load_prediction_results(csv_path: str) -> pd.DataFrame` — `pd.read_csv` wrapper.
- `save_figure(fig: plt.Figure, filename: str, output_dir: str, fmt: str = "png") -> None` — save at dpi 150 (duplicate of other `save_figure`s).
- `plot_prediction_comparison(df: pd.DataFrame, output_dir: str, fmt: str = "png") -> None` — grouped original vs. lab_variant bars per hard-coded condition list, value and Δ annotations; saves `prediction_comparison.{fmt}`.
