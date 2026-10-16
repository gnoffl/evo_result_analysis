# src/analysis/mutations/

Mutation-level analysis of evolutionary runs. `summarize_mutations.py` turns per-gene pareto fronts (all saved generations) into `MutatedSequence`/`MutationsGene` objects and a compact JSON (`all_mutated_sequences_<name>.json`; each sequence is encoded like `"12AC45GT|0.83"`, 0-based positions in the 3020 bp mutation window). `analyze_mutations.py` reads that JSON and computes per-position nucleotide changes, a conservation statistic, gaps between mutations and mutation-span quantiles, with plots. `genomic_annotation.py` maps sequence positions to genomic coordinates; feature annotation (GFF) is not implemented.

## summarize_mutations.py  [library]
Core mutation data model, imported widely (paper_plots, mutation_distribution_analysis, evo_alg_pooled_plots, compare_methods). CLI: `python -m analysis.mutations.summarize_mutations -r <run> -n <name> -o <out> [-f 1999] [-g gen]` (prints `OUTPUT_FILE=<path>`).

- `MutatedSequence(reference_sequence: str, mutated_sequence: str, fitness: float)` — diffs two equal-length strings into `mutations: List[(pos 0-based, ref, mut)]`.
  - `.get_mutation_number() -> int`.
  - `.from_string(reference_sequence: str, mutation_string: str) -> MutatedSequence` (classmethod) — parses `"<pos><ref><mut>...|<fitness>"` and applies it to the reference, checking ref bases.
  - `repr()` returns the compact string encoding; `__eq__` compares all fields.
- `MutationsGene(gene_folder_path: str, final_generation: int = 1999, generation: Optional[int] = None)` — reference = record index 1 of `reference_sequence.fa` (= `reference_sequence_mutation_window_0_3020`, the mutable window, not the full sequence). `generation_dict: {gen: [MutatedSequence]}` comes from all `pareto_front_gen_*.json` files plus `pareto_front.json`, which is stored under `final_generation`. `generation` restricts loading to one generation.
  - `.to_dict() -> Dict` / `.from_dict(data: Dict) -> MutationsGene` (classmethod; note: pops 'reference_sequence' from the dict passed in, mutating it).
  - `.get_init_and_optimal_fitness_generation(generation: int) -> Tuple[float, float]` — (fitness of the entry with fewest mutations, fitness of the entry with most).
  - `.get_equal_or_next_closest_fitness(generation: int, fitness: float) -> MutatedSequence` — binary search on the front sorted by mutation count; returns an exact match or the next entry with more mutations.
  - `.search_mutation_count(mutation_count: int, generation: Optional[int] = None) -> Dict[int, List[MutatedSequence]]`.
  - `.search_mutation_fitness(fitness: float, generation: Optional[int] = None) -> Dict[int, List[MutatedSequence]]` — exact float equality.
  - `.get_all_mutations_generation(generation: int) -> List[Tuple[int, str, str]]` — union of mutations over the front.
- `load_mutations_from_json(json_file: str) -> Dict[str, MutationsGene]` — loads a summarized JSON.
- `summarize_mutations_all_folders(base_folder_path: str, name: str, final_generation: int, generation: Optional[int] = None, output_folder: str = ".") -> Tuple[Dict[str, MutationsGene], str]` — builds and caches `all_mutated_sequences_<name>[_gen<g>].json`, reusing the file if it already exists; genes that error are skipped with a warning.

## analyze_mutations.py  [library]
Mutation statistics and plots from the summarized JSON; imported by paper_plots, mutation_distance_analysis and compare_methods. "Max-fitness sequence" below means the final generation's entry with the most mutations, found via get_equal_or_next_closest_fitness. CLI: `python -m analysis.mutations.analyze_mutations -m <json> -n <name> -o <out> [--all | --plot_half_max_stacked --plot_mutations_location --plot_stacked_only --plot_rolling_only --plot_mutation_conservation --plot_mutation_distances --plot_net_nucleotide_change] [-w 31] [-g 1999] [-mp 3000] [-f png] [--no_titles]`.

- `COLORS = {"A": "green", "C": "blue", "G": "orange", "T": "red", "Sum": "black"}` — nucleotide colour map.
- `count_mutations_single_gene(pareto_front: List[MutatedSequence]) -> Tuple[List[int], List[int]]` — (occurrence count of each distinct (pos, alt) mutation across the front, per-individual mutation counts sorted descending).
- `count_mutations_all_genes(mutation_data_path: str, generation: int = 1999) -> Dict[str, Tuple[List[int], List[int]]]`.
- `calculate_mutation_stats(mutation_data_path, generation=1999) -> (sorted_by_count, mean_max, std_max, mean_n_individuals, std_n_individuals, ratios)` — per gene, how often its most frequent single mutation recurs, and the front size; ratio = max / (n − 1).
- `print_mutation_stats(mutation_data_path, generation=1999) -> None`.
- `create_ideal_distribution(individual_mutation_counts: List[int], target_length: int) -> np.ndarray` — mutation-count profile if all individuals overlapped maximally (nested).
- `create_worst_case_distribution(mutation_counts, mutable_positions) -> np.ndarray` — total mutations spread evenly over 3·mutable_positions (pos, alt) slots.
- `calculate_conservation_statistic(mutation_counts: List[int], individual_mutation_counts: List[int], mutable_positions: int) -> float` — 1 − L1(ideal, observed sorted counts) / L1(ideal, worst case); 1 means mutations are shared maximally.
- `calc_conservation_stat_stats(conservation_stats: Dict[str, float]) -> (min, max, min_gene, max_gene, mean, std)`.
- `calculate_conservation_statistics(mutation_data_path, name, generation=1999, mutable_positions=3000, output_folder=".") -> Dict[str, float]` — per gene; writes `conservation_statistics_<name>_gen_<g>.json`.
- `plot_dict_as_stacked_bars(data_dict, title, xlabel, ylabel, file_path=None, titles=True, ax=None)` — stacked A/C/G/T bars per key.
- `make_line_plot_rolling_window(data_dict: {pos: {base: count}}, name, output_format, window_size=11, output_folder=".", titles=True, ax=None, plot_sum=True)` — rolling mean per base over a hardcoded 3020 bp frame, with a NaN gap over the central 20 bp padding. The title comes from the name suffix `_to/_from/_diff`. Ticks label TSS at 1000 and TTS at 2000; per the extraction frame the TTS is at 2020, so that label is about 20 bp off.
- `plot_hist_half_max_mutations_stacked(mutation_data_path, name, output_format, output_folder=".", titles=True)` — stacked histograms of removed/introduced bases in the half-max sequence, keyed by its mutation count.
- `plot_mutations_location(mutation_data_path, name, output_format, window_size=11, plot_stacked=True, plot_rolling=True, output_folder=".", titles=True)` — from/to/diff stacked bars and rolling-mean lines.
- `calculate_positional_nucleotide_change(mutation_data_path: str) -> (from_dict, to_dict, diff_dict)` — per position: counts of each base removed, introduced and net, using each gene's max-fitness sequence.
- `plot_hist_mutation_conservation(mutation_data_path, name, output_format, generation=1999, mutable_positions=3000, output_folder=".", titles=True, ax=None) -> None` — histogram of the conservation statistic (reuses the cached JSON if present).
- `calculate_mutation_distances_single_gene(mutations: List[int]) -> Counter` — counts of gaps between consecutive sorted positions.
- `load_mutation_data(mutation_data_path: str) -> Dict[str, MutationsGene]` — duplicates summarize_mutations.load_mutations_from_json.
- `analyze_mutation_distances(mutation_data, output_folder, name) -> None` — writes `mutation_ranges_<name>.txt` (first/last mutation and full/90 %/50 % span stats).
- `calculate_mutation_distances(mutation_data: Dict[str, MutationsGene]) -> Counter` — gap counts pooled over genes (max-fitness sequences).
- `get_random_mutation_distributions(mutable_positions=3000, num_samples=10000) -> Counter` — null gap distribution for 90 positions drawn uniformly; not used by default.
- `analyze_range_single_gene(mutations: List[int]) -> (min, max, start_90, end_90, start_50, end_50)` — repeatedly drops the outermost mutation on the side with the larger gap until 90 % and then 50 % of mutations remain.
- `plot_mutation_distances(mutation_data_path, name, output_format, output_folder, titles=True) -> None` — ranges txt plus gap histograms (all gaps, and gaps ≤200).
- `plot_dist_hist(name, output_folder, distances, counts, output_format, random_distribution=None, titles=True, ax=None)`.
- `calculate_net_nucleotide_change(mutation_data_path: str) -> Dict[str, int]` — total introduced minus removed per base (max-fitness sequences).
- `plot_net_nucleotide_change(net_change: Dict[str, int], name, output_format, output_folder=".", titles=True, ax=None) -> None` — bar plot.

## genomic_annotation.py  [library]
Maps mutation positions in the extracted promoter+terminator sequence to genomic coordinates. Only `AnnotatedMutatedSequence.parse_sequence_name` is used elsewhere (motives/annotate_individual_mutation.py). CLI: `-m <mutations_json>` prints genomic positions.

- `GenomicFeature(chromosome, source, feature_type, start, end, score, strand, phase, attributes)` — dataclass for a GFF record (1-based, inclusive). `.overlaps(position: int) -> bool`. Unused; there is no GFF parser.
- `MutationAnnotation(seq_position, genomic_position, ref_base, mut_base, feature_type, feature_id=None, gene_id=None)` — dataclass; feature_type is always 'unknown' in practice.
- `AnnotatedMutatedSequence(mutated_sequence, gene_id, chromosome, gene_start, gene_end, strand, annotations, generation=None)` — dataclass.
  - `.to_dict() -> Dict`.
  - `.parse_sequence_name(seq_name: str) -> Tuple[str, str, int, int]` (staticmethod) — parses `{index}_{gene_id}_gene:{start}-{end}_...`. **[bug]** returns gene_id in the chromosome slot, and gene_id includes the leading index (e.g. "3_Zm00001eb130940").
  - `.from_mutated_sequence(seq_name, mutated_sequence, intragenic=500, extragenic=1000, central_padding=20, generation=None)` (classmethod) — hand-rolled mapping from sequence position to genomic position. Strand is inferred from start < end; the promoter is [0,1500) and the terminator starts at 1520. Whether the coordinate conventions are right (1-based, off-by-one on each strand) is unverified — (unclear). Prefer reusing the evolution/deepCRE coordinate logic.
  - `.from_mutations_gene(seq_name, mutations_gene, generation=None, intragenic=500, extragenic=1000, central_padding=20) -> Dict[int, List[AnnotatedMutatedSequence]]` (classmethod).
- `annotate_mutations(...)`, `annotate_all_genes(...)` — **[broken]** stubs that return {}.
- `annotate_from_json(json_path: str, gff_file: str, generation=None) -> Dict` — coordinates only; `gff_file` is ignored.
