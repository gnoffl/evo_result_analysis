# src/workflows/starrseq_v2/

The STARR-seq v2 round, start to finish. `run/` picks the candidate genes for Arabidopsis and N. tabacum (STARR-seq v1, FLOR-ID flowering, GOF/LOF, random), extracts 3020 bp gene frames, splices 170 bp windows into the construct backbone and writes the gene metadata CSV that the config grid in `evolution.write_run_script` reads. `probe_analysis/` checks whether the pilot runs saturate. `selection/` turns the finished full-run Pareto fronts into an ordered oligo library in three steps (pool fronts -> select entries -> build library), writing to `selection/output/` (CSV, FASTA, .ods upload sheet). Paths and parameters are hardcoded module constants; nothing takes a CLI.

Caveats: the run command in `README.md` (`-m src.workflows.starrseq_v2.build_starrseq_v2_library`) is stale, because the module now lives in `run/`. Modules import as `workflows.…`, so `src/` must be on the path.

## probe_analysis/probe_analysis.py  [script]
Compares the best fitness of the probe runs at generation 1000 and 2000, at ≤20 and ≤60 mutations, normalised per gene. Run as `python -m workflows.starrseq_v2.probe_analysis.probe_analysis`. **[stale path]** `OUTPUT_DIR = Path(__file__).parent / "probe_analysis"` now resolves to a nested `probe_analysis/probe_analysis/`, because the file was moved out of `run/`.

- `PROBE_DIR` — the probe run tree (`…/Evolution_runs/dataset/starrseq_v2/probe`).
- `best_fitness(front: list, mutation_limit: int, maximize: bool) -> float` — the max (or min) fitness among front entries with `n_mutations <= mutation_limit`. Front entries are `(seq, fitness, n_mut)`.
- `collect_run(run_dir: Path, maximize: bool) -> list` — one row dict per (generation, mutation limit 0/20/60) for one run. Reads `parameters.json:sequence_name` and `saved_populations/pareto_front_gen_01000.json` / `pareto_front.json`, and gives NaN when a checkpoint is missing.
- `normalize_per_gene(results: pd.DataFrame) -> pd.DataFrame` — rescales fitness per gene so that 0 = unmutated and 1 = gen 2000 at ≤60 mutations, then drops the limit-0 rows.
- `plot_direction(results: pd.DataFrame, direction: str, output_path: Path) -> None` — bar plot of the mean ± SD normalised gain per (generation, limit), with n and missing counts in the labels. Saves a PNG.

## run/build_starrseq_v2_library.py  [script]
The whole v2 library build: gene selection, frame extraction, arm/grid subsets, directions, off-site windows, construct splicing and the metadata CSV. Run `main()` (no CLI). Imports `wrky_mapping_results` and `bhlh_mapping_results` from `workflows.overlap_analysis.starrseq_deepcre_correlation_{WRKY,bHLH}`. Frame extraction wraps `evolution.extract_sequences.find_genes` / `extract_string`. The output checks are still a TODO at the end of `main()`.

- `DATA`, `ARC` — the Evolution `data/starrseq_v2` directory and the genRE ARC Gene_Data root.
- `ARA_GENOME`, `NATB_GENOME` (sic), `ARA_ANNOTATION`, `NTAB_ANNOTATION`, `ARA_VCF` — the TAIR10 / N. tabacum genome, the GTF files and the combined 1001-genomes SNP VCF.
- `INSERT_START=780`, `INSERT_END=950`, `INSERT_LENGTH`, `BACKGROUND_PATH` — the 170 bp insert placeholder in the extracted construct (`overlap_analysis/correct_construct/construct_1_extracted.fa`).
- `STARRSEQ_WINDOW_LENGTH=170`, `RANDOM_WINDOW_STARTS` (`[0,1330) ∪ [1520,2850)`), `INTRAGENIC=500`, `EXTRAGENIC=1000` — frame geometry (3020 bp, TSS at 1000, N padding at 1500–1520).
- `GENE_METADATA_PATH`, `*_FASTA`, `*_CONSTRUCT_PATH`, `*_OFF_TARGET_PATH`, `STARRSEQ_MAPPING_PATH` — output paths under `DATA`.
- `draw_random_genes(annotation_path: str, exclude_genes: List[str], n_genes: int, seed: int) -> List[str]` — seeded sample of gene ids from a GTF (via `find_genes`, with the `_gene` suffix stripped) that excludes `exclude_genes`. Raises if too few are left.
- `extract_genes(genome_path: str, annotation_path: str, gene_ids: List[str], output_path: str, vcf_paths: Optional[List[str]] = None) -> None` — writes 3020 bp frames for the genes to FASTA. Wraps `evolution.extract_sequences.extract_string` with `CENTRAL_PADDING`.
- `gene_ids_from_fasta(fasta_path: str) -> List[str]` — gene ids from the headers of an extracted-frame FASTA (`<chrom>_<gene>_gene:…`). Assumes the chromosome name has no `_`.
- `add_to_metadata(metadata: pd.DataFrame, gene_ids: List[str], species: str, reason: str) -> pd.DataFrame` — appends rows of `gene_id, species, reason`.
- `draw_subset_genes(eligible: pd.DataFrame, fraction: float, min_genes: int, seed: int) -> List[str]` — per species, draws `max(min_genes, round(fraction·n))` unique genes. Raises if a species has `<= min_genes` genes.
- `select_arm_subset(metadata: pd.DataFrame, seed: int) -> pd.DataFrame` — adds a boolean `arm_subset` column, drawn with the `starrseq_v1` rows excluded. Every row of a drawn gene is flagged.
- `select_full_grid_subset(metadata: pd.DataFrame, seed: int) -> pd.DataFrame` — adds `full_grid_subset`, drawn from within the arm subset.
- `add_v1_windows(metadata: pd.DataFrame, starrseq_v1_mapping: pd.DataFrame) -> pd.DataFrame` — left-merges the v1 fragment overlap windows in as `alternative_start/_end` (one row per window).
- `add_random_windows(metadata: pd.DataFrame, frames: Dict[str, str], seed: int) -> pd.DataFrame` — for arm genes without a window, draws one random 170 bp window start from `RANDOM_WINDOW_STARTS` whose window contains no N. The columns become `Int64`.
- `extract_ntab(metadata)`, `extract_ara(metadata)`, `extract_starrseq_v1(metadata)`, `extract_all_candidate_genes() -> Tuple[pd.DataFrame, pd.DataFrame]` — per-species gene selection plus extraction. These are untyped workflow glue.
- `splice(background: str, insert: str) -> str` — replaces the `N`×170 placeholder in the backbone with the insert. Raises if the insert is not 170 bp.
- `splice_correct_windows(gene_fasta_path: str, output_path: str) -> None` — writes one construct per frame, with frame `[780,950)` spliced into the backbone.
- `assign_directions(metadata: pd.DataFrame, seed: int) -> pd.DataFrame` — adds `direction`: GOF→maximize, LOF→minimize, and a balanced random draw for everything else within each (subset stratum × species) cell. Each gene gets a single direction.
- `flag_full_length_windows(metadata: pd.DataFrame) -> pd.DataFrame` — adds `full_length_window` (the alternative window is exactly 170 bp).
- `map_genes_to_frames(frame_keys: List[str], gene_ids: List[str]) -> dict` — gene id → FASTA header, found by suffix match, which is safe when chromosome names contain `_` (N. tabacum).
- `splice_off_target_windows(metadata: pd.DataFrame, gene_fasta_path: str, output_path: str, reason: Optional[str] = None) -> None` — splices each full-length alternative window into the backbone. Headers are `<frame>_off_<start>`.
- `summarize_subsets(metadata: pd.DataFrame, frames: Dict[str, str]) -> pd.DataFrame` — non-v1 gene counts per subset × species: all genes, with a frame, and with an N-free correct window.

## run/flowering_data/download_data.py  [script]
Scrapes the FLOR-ID flowering-time gene table (regex HTML parsing) to `florid_flowering_genes.csv` and writes the gene-id list `ara_flowering_genes_florid.json`. Run `main()`.

- `URL`, `OUTPUT`, `COLUMNS` — the FLOR-ID page, the output CSV and its nine column names.
- `cell_text(cell: str) -> str` — HTML cell → plain text (tags stripped, entities unescaped, whitespace collapsed).
- `parse_gene_table(page: str) -> List[List[str]]` — the rows of the first `<tbody>` as lists of cell texts.
- `download() -> None` — fetches and writes the CSV.
- `check()` — prints a summary and asserts the column names.
- `create_input_json()` — writes the unique non-null `gene_id`s to JSON.

## run/flowering_data/find_nicotiana_orthologs.py  [script]
Maps the Arabidopsis FLOR-ID genes to N. tabacum best hits: gffread translation of both genomes, longest protein per gene, then DIAMOND blastp best hit (evalue ≤ 1e-10, identity ≥ 30 %). Writes `ara_ntab_flowering_orthologs.csv` and `ntab_flowering_genes.json`. Runs the external tools through `conda run -n deepcre_tools`.

- `ARA_GENOME`, `ARA_ANNOTATION`, `NTAB_GENOME`, `NTAB_ANNOTATION` — the ARC genome and GTF paths. `MAX_EVALUE`, `MIN_IDENTITY`, `THREADS` — the hit cutoffs.
- `conda_run(*command: str) -> None` — `subprocess.run` inside the `deepcre_tools` env, with `check=True`.
- `read_fasta(path: Path) -> Dict[str, str]` — minimal FASTA reader keyed by the first word of each header.
- `longest_per_gene(proteins: Dict[str, str], genes: Optional[List[str]] = None) -> Dict[str, str]` — the longest isoform per gene (strips the last `.suffix` from transcript ids), optionally restricted to `genes`.
- `write_fasta(sequences: Dict[str, str], path: Path) -> None` — writes a dict as single-line FASTA.
- `best_hits(tsv: str) -> Dict[str, str]` — the first hit per query that passes the identity and evalue cutoffs, from DIAMOND outfmt-6 text (`qseqid sseqid pident evalue …`).
- `build_query_fasta() -> None`, `build_database() -> None`, `run_diamond() -> str` — the three pipeline stages (gffread → FASTA, diamond makedb, diamond blastp).

## selection/PLAN.md  [doc]
Design and measured facts for the three selection steps. Points relevant to reuse: fronts have gaps in their mutation counts and ties at a count, so key entries by `(run_dir, front_index)` and never by mutation count. 852 entries contain N.

## selection/pool_front_table.py  [script]
Step 1: pools every final Pareto front of the v2 `full_run` into `output/run_overview.csv` (one row per run), `output/front_table.csv.gz` (one row per front entry, keyed by `(run_dir, front_index)`) and `output/fixed_parameters.json`. Factors are parsed from the config folder name. Uses `evolution.write_run_script` (`BASELINE`, `FACTORS`, `STARRSEQ_V2_INSERT_START/END`, `gene_id_from_record`).

- `RUNS`, `ANALYSIS`, `METADATA`, `OUTPUT` — the full-run tree, the per-config `stats_<config>.json` tree, the gene metadata CSV and the output dir.
- `CONFIG_LEVELS` — `FACTORS` plus `v1_genes` and the direction. `FIXED_PARAMETERS` — the GA parameter names that are identical in every run.
- `load_json(path: str) -> Any` — reads one JSON file.
- `config_factors() -> Dict[str, Dict[str, str]]` — every possible config name → its factor-level dict (Cartesian product of `CONFIG_LEVELS`).
- `group_of(factors: Dict[str, str]) -> str` — `starrseq_v1` / `baseline` / `arms` (1 factor off baseline) / `grid` (gene_sp ignored).
- `source_window(factors: Dict[str, str], parameters: Dict[str, Any]) -> Tuple[int, int]` — the `[start, end)` of the optimised 170 bp within the 3020 bp frame. For construct+offsite it is parsed from the `_off_<start>` record-name suffix; for natural+offsite it is `mutation_start/end`.
- `run_record(run_path: str, factors: Dict[str, str], stats: Dict[str, Any], gene_ids: Set[str]) -> Dict[str, Any]` — per-run columns: record name, gene id, factors, source window, generations, models, VCF flag, start/final fitness, max mutations and half-max count (from stats).
- `front_entries(front_path: str) -> pd.DataFrame` — one front as `front_index, n_mutations, fitness, n_count` (N count) plus `tied_at_count`. Sequences are not kept.
- `collapse_metadata(metadata: pd.DataFrame) -> pd.DataFrame` — one row per gene: its species and all of its reasons joined with `|`.
- `add_entry_flags(front_table: pd.DataFrame) -> pd.DataFrame` — adds `is_reference`, `is_half_max`, `is_max_mutations` and a signed `fitness_delta_vs_reference` (positive = the optimisation worked, in either direction).
- `pool_fronts(runs_root: str, analysis_root: str, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]` — walks config dirs → run dirs and returns `(run_overview, front_table)`.

## selection/select_candidates.py  [script]
Step 2: picks `TARGET=57_000` entries from `front_table.csv.gz` in three masked stages — each run's three anchors (reference, half-max, max), then `TARGET_BEST=5_600` globally ranked best picks, then a round-robin fill with draws uniform on the mutation-count axis. Entries containing N are excluded. Writes `output/selected_keys.csv` (`run_dir, front_index, selection_reason`, reasons joined with `|`).

- `TARGET`, `TARGET_BEST`, `SEED`, `ANCHOR_FLAGS`, `BEST_MUTATION_COUNTS` — the total entry count, the best-pick share of it, the RNG seed, the anchor → flag column map and the counts `(1,2,3,5,7,10,15)` the fixed-count ranking uses.
- `anchor_reasons(front: pd.DataFrame) -> Dict[int, List[str]]` — front index → the anchor names it satisfies. On ties the lower front index wins.
- `split_evenly(total: int, parts: int) -> List[int]` — sizes differing by at most one, remainder to the earlier parts.
- `rank_by_score(entries: pd.DataFrame, score: pd.Series) -> List[Tuple[str, int]]` — `(run_dir, front_index)` keys, best score first; ties go to the lower `run_dir` then the lower index.
- `take_best(ranking: Sequence[Tuple[str, int]], reasons: Dict[str, Dict[int, List[str]]], reason: str, quota: int) -> int` — walks the ranking labelling entries until `quota` **previously unselected** ones were added; an already-selected entry gains the reason for free. Extends `reasons` in place, returns the shortfall.
- `select_best(usable: pd.DataFrame, reasons: Dict[str, Dict[int, List[str]]], target_best: int, mutation_counts: Sequence[int]) -> int` — half the budget by `fitness_delta_vs_reference` at each count in `mutation_counts` (equal share each), half by gain per mutation over all mutated entries. Neither ranking caps a run's share, so a strong run contributes a mutation ladder.
- `draw_one(n_mutations: np.ndarray, front_indices: np.ndarray, available: np.ndarray, max_count: int, rng: np.random.Generator) -> Optional[int]` — draws `u ~ U[1, max_count-1]` and snaps to the nearest available entry (ties go to the lower index). Clears `available` in place.
- `fill_to_target(fronts: Dict[str, pd.DataFrame], reasons: Dict[str, Dict[int, List[str]]], target: int, rng: np.random.Generator) -> int` — round-robin draws (one per run per round, in random run order) until the target is reached. Extends `reasons` in place and returns the shortfall.
- `select_candidates(front_table: pd.DataFrame, target: int, seed: int, target_best: int = 0, mutation_counts: Sequence[int] = BEST_MUTATION_COUNTS) -> pd.DataFrame` — the three stages in order. Raises if the target is below what the earlier stages took, if a ranking cannot supply its quota of new entries, or if the fronts cannot reach the target.

## selection/build_library.py  [script]
Step 3: fetches the sequences of the selected entries and screens for BsaI. A run whose reference has a site loses all its entries. Survivors are numbered, deduplicated on the exact sequence and given reference pointers, then validated. Writes `library_table.csv`, `library_upload.ods` (appended to the autoTRAIT template via `odfpy`), `library.fasta` and `configs_to_copy.txt`.

- `RUNS`, `CONFIGS`, `OUTPUT`, `TEMPLATE` — the run tree, the Evolution `configs/starrseq_v2`, the output dir and the `.ods` template.
- `RESTRICTION_SITE` — regex for BsaI `GGTCTC` and its reverse complement `GAGACC`, which also matches `GGTCT`/`GAGAC` at the end of the sequence (the cloning suffix supplies the final C).
- `GENOME_BY_SPECIES` — `{"arabidopsis": "TAIR10", "ntab": "Ntab-TN90"}`. `SEQUENCE_LENGTH=170`, `AUTHOR`, `CONFIG_PREFIX`, `CONFIG_FOLDERS`, `SORT_KEY`.
- `load_json(path: str) -> Any` — reads one JSON file (duplicate of the one in `pool_front_table`).
- `flatten_field(value: str) -> str` — `_`→`-` and `|`→`+`, for use in identifier fields.
- `parse_record_name(record_name: str, gene_id: str) -> Tuple[str, str]` — `(chromosome, "start-end" span)` from `<chrom>_<gene>_gene:<s>-<e>[_off_<n>]`. It splits on the known gene id, so it is safe with `_` in chromosome names. A reversed span means the minus strand.
- `build_identifier(entry: Dict[str, Any], serial: int) -> str` — the five-field upload identifier `<genome>_<chrom>:<span>_<gene.config.fNN.mNN.reason>_GS_<serial:06d>`.
- `attach_sequences(selected: pd.DataFrame, runs_root: str) -> pd.DataFrame` — adds the upper-cased `sequence` by reading each `pareto_front.json` once and indexing by `front_index`.
- `check_references_selected(table: pd.DataFrame) -> None` — raises unless every run has exactly one `is_reference` row.
- `has_restriction_site(sequence: str) -> bool` — whether `RESTRICTION_SITE` matches.
- `drop_restriction_sites(table: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]` — drops site-carrying entries and whole runs whose reference carries a site. Returns the drop counts.
- `assign_numbers(table: pd.DataFrame) -> pd.DataFrame` — stable sort by `SORT_KEY`, `sequence_number` from 1, and `identifier`.
- `deduplicate(table: pd.DataFrame) -> pd.DataFrame` — adds `representative_number`, `representative_id` and `is_representative` (the lowest number per distinct sequence).
- `add_reference_pointers(table: pd.DataFrame) -> pd.DataFrame` — adds `reference_id` and `reference_representative_id` per run.
- `config_source_paths(config_names: Iterable[str], configs_root: str) -> Dict[str, str]` — config name → JSON path under `CONFIG_FOLDERS`, skipping the `bins/` copies. Raises if a config is missing.
- `validate(table: pd.DataFrame, representatives: pd.DataFrame) -> None` — raises on wrong length, non-ACGT bases, non-unique numbers/ids/representative sequences, or dangling reference pointers.
- `spreadsheet_row(values: List[str]) -> TableRow` — an odfpy row of string cells in the "Default" style.
- `write_upload_sheet(representatives: pd.DataFrame, template_path: str, destination: str) -> None` — appends rows (n, id, reference id, sequence, config path) to the template's first sheet.
- `write_fasta(representatives: pd.DataFrame, path: str) -> None` — one `>identifier` record per representative.
- `build_library(selected: pd.DataFrame, front_table: pd.DataFrame, runs_root: str) -> Tuple[pd.DataFrame, Dict[str, int]]` — the whole step 3, from merge through reference pointers and `path_to_config`.

## selection/verify_pool_front_table_untested.py  [one-off]
Independent cross-check of `pool_front_table` outputs against every raw source (folder names, parameters.json, fronts, stats, metadata, FASTAs). Prints the failed checks.

## selection/verify_build_library_untested.py  [one-off]
Independent cross-check of `build_library` outputs (screen, numbering, dedup, reference pointers, FASTA and .ods) against the raw fronts.
