# src/analysis/blamm/ (+ src/workflows/blamm_tfbs_analysis/run_blamm_analysis.sh)

PWM-based TFBS counting with the external `blamm` binary: converts MEME motif databases to one JASPAR file, scans the reference vs. one optimized pareto-front sequence per gene (single blamm call, shared background model, both strands), and counts motif hits per gene and in aggregate. A per-motif significance test (Wilcoxon signed-rank of per-gene count differences vs 0, BH across motifs) is written alongside. Inputs: evolutionary run folder + `.jaspar` + motif metadata CSV; outputs: `motif_counts_per_gene.csv`, `motif_counts_aggregate.csv`, `motif_significance.csv`, `selected_entries.csv`, `skipped_genes.csv`, `run_parameters.json`.

## blamm_tfbs_counts.py  [script]
Run blamm motif counting for one run: `python -m analysis.blamm.blamm_tfbs_counts -r <run> -m <x.jaspar> -t <meta.csv> -o <out> [-mc MAX|int] [-p 1e-4] [--empirical-histograms] [--blamm path]`.

- `MAX_MUTATIONS = "MAX"` — sentinel selecting the pareto entry with the most mutations.
- `DEFAULT_P_VALUE = 1e-4` — `blamm scan -pt` threshold.
- `SEQUENCE_ID_SEPARATOR = "__"`, `VARIANT_REFERENCE = "reference"`, `VARIANT_MUTATED = "mutated"` — FASTA id scheme `<gene>__<variant>`.
- `LOG2_PSEUDOCOUNT = 1` — pseudocount added to both sides of log2 fold changes.
- `OCCURRENCE_COLUMNS` — column names of blamm's GTF-like `occurrences.txt`.
- `select_pareto_entry(pareto_front: Sequence[ParetoEntry], mutation_count: int | "MAX") -> Tuple[int, ParetoEntry]` — first entry with the max (or exact) mutation count (count rounded to int); ValueError if the front is empty or nothing matches.
- `build_sequence_records(gene_folders: Sequence[str], mutation_count) -> (records, selections_df, skipped_df)` — loads each gene via `motives.deepcis_scanner.GeneRunData.load_gene_run_data`; builds `(id, full_seq)` for reference and optimized (mutable region spliced into the full reference via `deepcis_scanner.get_full_sequence`); skips genes on ValueError/KeyError/OSError.
- `write_fasta(records: Sequence[Tuple[str, str]], fasta_path: str) -> None` — writes FASTA, one line per sequence; rejects duplicate or whitespace-containing ids.
- `write_manifest(fasta_path: str, manifest_path: str) -> None` — writes the blamm `.mf` manifest (single group "evo").
- `run_blamm(blamm_executable, motifs_path, manifest_path, work_dir, p_value, empirical_histograms) -> str` — runs `blamm dict`, `hist [-e]`, `scan -rc -pt` via subprocess; returns the path to `occurrences.txt`.
- `parse_occurrences(occurrences_path: str) -> pd.DataFrame` — reads blamm occurrences; columns gene, variant, motif_id, start, end, score, strand.
- `count_motif_hits(occurrences: pd.DataFrame, motif_metadata: pd.DataFrame) -> pd.DataFrame` — per (gene, motif): reference/mutated counts, `diff`, `log2_fold`; only pairs with ≥1 hit.
- `aggregate_over_genes(per_gene_counts: pd.DataFrame, n_genes_analysed: int) -> pd.DataFrame` — per-motif totals, `n_genes_with_hit`, `diff_per_gene` (= total_diff / n_genes_analysed), pooled log2 fold; sorted by total_diff, descending.
- `collect_provenance(...) -> Dict[str, object]` — JSON-able record of inputs: paths, SHA256 of the motif files, blamm version, git commit, gene counts.
- `analyse_run(run_folder, motifs_path, motif_metadata_path, output_folder, mutation_count="MAX", p_value=1e-4, empirical_histograms=False, blamm_executable="blamm") -> pd.DataFrame` — full pipeline including significance; writes all output files and returns the aggregate table.
- `parse_arguments(args=None) -> argparse.Namespace` — CLI parsing and validation (the motif file must end in `.jaspar`).

## blamm_significance.py  [library]
Per-motif, within-run significance of TFBS gain/loss. Also a CLI to (re)compute it for existing output folders: `python -m analysis.blamm.blamm_significance -o <folder> [<folder> ...]`.

- `PER_GENE_COUNTS_FILE`, `SELECTED_ENTRIES_FILE`, `SIGNIFICANCE_FILE` — file names in a blamm output folder.
- `SIGNIFICANCE_COLUMNS` — columns of `motif_significance.csv` (n_genes_gained/lost/unchanged, median/mean diff, direction, p_intra, q_intra, significance_stars).
- `build_diff_matrix(per_gene_counts: pd.DataFrame, genes: Sequence[str]) -> pd.DataFrame` — genes × motifs matrix of `diff`; missing pairs filled with 0; validates the gene list.
- `motif_significance(per_gene_counts: pd.DataFrame, genes: Sequence[str]) -> pd.DataFrame` — per motif: two-sided Wilcoxon vs 0 over genes, BH q across motifs, stars; direction from the sign of the mean diff.
- `analyse_output_folder(output_folder: str) -> pd.DataFrame` — reads per-gene counts and selected genes from a folder and writes `motif_significance.csv` (no rescan).
- `summarise_significance(significance: pd.DataFrame) -> str` — one-line count of significant enriched/removed motifs.
- `parse_arguments(args=None) -> argparse.Namespace` — CLI parsing.

## meme_to_jaspar.py  [library]
Convert MEME motif DBs (PPM + nsites) into one JASPAR PFM file plus a metadata CSV for blamm: `python -m analysis.blamm.meme_to_jaspar --meme-files a.meme b.meme --output-jaspar x.jaspar --output-metadata meta.csv`.

- `NUCLEOTIDES = "ACGT"`; `DEFAULT_NSITES = 100` — used, with a warning, when a motif has no `nsites=`.
- `Motif(motif_id, motif_name, source_db, nsites, counts)` — NamedTuple; `counts` = {base: list[int]} per position.
  - `.width -> int` — motif length.
- `parse_meme_file(meme_path: str) -> List[Motif]` — reads `letter-probability matrix` blocks only; counts = round(p · nsites).
- `collect_motifs(meme_paths: Sequence[str]) -> List[Motif]` — parses several files; ValueError if a motif id appears in more than one DB.
- `write_jaspar(motifs: Sequence[Motif], output_path: str) -> None` — JASPAR format (`>id\tname` + A/C/G/T rows).
- `write_metadata(motifs: Sequence[Motif], output_path: str) -> pd.DataFrame` — CSV of motif_id, motif_name, source_db, width, nsites.
- `convert_meme_databases(meme_paths, output_jaspar, output_metadata) -> pd.DataFrame` — collects motifs and writes both files.
- `parse_arguments(args=None) -> argparse.Namespace` — CLI parsing.

## run_blamm_analysis.sh  [script]  (src/workflows/blamm_tfbs_analysis/)
Paper driver; `bash run_blamm_analysis.sh` takes no arguments. Builds `motifs/plant_tf_motifs.jaspar` and its metadata from three hardcoded MEME DBs in the moca-pipeline motif_databases dir (JASPAR2024 CORE plants non-redundant v2, ArabidopsisDAPv1, ArabidopsisPBM_20140210). Then runs `blamm_tfbs_counts` with p=1e-4 on 4 hardcoded runs (GOF and LOF single_mutation; arabidopsis_toolkit msr_max and msr_min single) × mutation counts {MAX, 5}. Output goes to `<run_name>__mutations_<MAX|5>/` next to the script, with the console log tee'd. Sets OPENBLAS/OMP/MKL_NUM_THREADS=1.
