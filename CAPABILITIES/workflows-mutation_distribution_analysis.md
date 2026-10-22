# src/workflows/mutation_distribution_analysis/

Null model for the EA's SNPs: pool every (position, source_base, new_base) SNP of the pareto-tip sequences across ~1000 genes, then draw random "baseline" sequences per gene with the same SNP count from that pool, and score them with the MSR deepCRE model. Pipeline: summarized-mutations JSON -> `mutation_pool.py` (MutationPool JSON in `mutation_pools/`) -> `baseline_sampler.py` (FASTA in `mutated_sequences/`) -> `evaluate_sequences.py` (prediction CSV) -> `analyse_results.py` (bar plot before/random/after). `conditional_poisson.py` is a generic maximum-entropy sampler of k distinct weighted items, also used by `mutation_distance_analysis`. README.md and `plans/*.md` document the design (incl. why successive sampling was replaced). No `__init__.py` (namespace package); tests in `test/workflows/mutation_distribution_analysis/`.

## mutation_pool.py  [library | script]
Step 1: flatten a summarized-mutations JSON into a MutationPool artifact; imports `MutatedSequence`, `MutationsGene`, `load_mutations_from_json` from `analysis.mutations.summarize_mutations`. CLI: `python -m workflows.mutation_distribution_analysis.mutation_pool --input <all_mutated_sequences_*.json> --output <pool.json> [--generation 1999]`.

- `VALID_BASES` — frozenset {A,C,G,T}; mutations with any other source/new base (N padding) are dropped.
- `MUTATION_COLUMNS` — `["gene_id", "position", "source_base", "new_base"]`.
- `GENE_STATS_COLUMNS` — `["gene_id", "n_mutations", "initial_fitness", "final_fitness"]`.
- `MutationPool(mutations: pd.DataFrame, gene_stats: pd.DataFrame, references: Dict[str, str])` — dataclass: long-format SNP table (0-based positions into the wildtype), per-gene stats, gene_id -> wildtype sequence. Also imported by `paper_plots/compose_plots_mpl.py`.
  - `.from_summarized_json(path: str, generation: int = 1999) -> MutationPool` — classmethod; per gene takes the max-fitness sequence of `generation_dict[generation]` (assumed = most mutated), keeps ACGT-only SNPs; `n_mutations` = kept count; fitnesses via `MutationsGene.get_init_and_optimal_fitness_generation(generation)`. Raises ValueError if a gene lacks that generation.
  - `.load(path: str) -> MutationPool` — classmethod; read JSON written by `.save`.
  - `.save(path: str) -> None` — write JSON with keys `mutations`, `gene_stats`, `references` (parent dir must exist).

## conditional_poisson.py  [library]
Generic maximum-entropy (conditional Bernoulli / conditional Poisson, Chen-Dempster-Liu 1994) sampler of k distinct indices with exact target inclusion probabilities. Distribution: independent coins with odds w_i, conditioned on exactly k heads, so P(subset S) ∝ ∏_{i∈S} w_i; w is calibrated so P(i ∈ S) = π_i = k·counts_i/Σcounts (capped at 1, iterated "take-all"). Pure numpy, no repo imports.

- `DEFAULT_TOLERANCE` (1e-10), `DEFAULT_MAX_ITERATIONS` (500), `DEFAULT_MAX_ATTEMPTS` (10000) — calibration / rejection caps.
- `inclusion_probabilities(counts: np.ndarray, sample_size: int) -> np.ndarray` — π_i = k·c_i/Σc with iterated capping at 1 (like R `sampling::inclusionprobabilities`); zero counts -> 0; sums to k.
- `log_normalising_constants(working_odds: np.ndarray, sample_size: int) -> tuple[np.ndarray, float]` — returns (log R(k-1, S\{j}) for all j, log R(k, S)) via row-rescaled forward/backward Gail-Lubin-Rubinstein recursion, O(n·k); requires 1 ≤ k < n.
- `calibrate_working_odds(target: np.ndarray, tolerance: float = ..., max_iterations: int = ...) -> np.ndarray` — Chen-Liu (1997) eq. 11 fixed-point inversion: find odds whose conditional-Bernoulli inclusion probs equal `target` (entries strictly in (0,1), sum integer k<n); log-mean-centred result; RuntimeError if unconverged.
- `scale_odds_to_sample_size(working_odds: np.ndarray, sample_size: int) -> np.ndarray` — rescale odds by global c (bisection) so Σ c·w/(1+c·w) = k; design unchanged, only rejective acceptance rate (~1/sqrt(2πk)).
- `sample_positions(working_odds: np.ndarray, sample_size: int, rng: np.random.Generator, max_attempts: int = ...) -> np.ndarray` — one rejective draw: flip Bernoulli(w/(1+w)) per index, accept iff exactly k heads; returns ascending indices.
- `PositionSampler(counts: np.ndarray, sample_size: int, tolerance: float = ..., max_iterations: int = ...)` — calibrate once in `__init__` (~0.2 s at n≈2700, k≈88): computes target π, forces π=1 indices into every draw, calibrates the rest. Attributes `counts, sample_size, target, working_odds, forced_indices, free_indices`.
  - `.draw(rng: np.random.Generator, max_attempts: int = ...) -> np.ndarray` — ascending array of `sample_size` distinct indices into `counts`; index i included with probability exactly `target[i]`.

## baseline_sampler.py  [library | script]
Steps 2+3: draw random baseline sequences per gene from a MutationPool; imports `PositionSampler` (conditional_poisson) and `MutationPool` (mutation_pool), `print_status` from `analysis.utils.io`. CLI: `python -m workflows.mutation_distribution_analysis.baseline_sampler -i <pool.json> -o <out.fa> [-n n_per_gene=1] [-s seed=42]`.
Sampled distribution: for target gene with wildtype W, keep only pool rows (from ALL genes, including itself) with `source_base == W[position]` ("applicable"). Let n_i = #applicable rows at position i, N = total. Positions: k distinct positions under the max-entropy design with inclusion prob exactly k·n_i/N (capped at 1). Substitution at each drawn position: uniform over its applicable rows (i.e. P(new_base=b | i) = n_{i,b}/n_i). k = the gene's own `gene_stats.n_mutations`. No two SNPs at one site.

- `GeneMutationSampler(unique_positions, group_starts, group_sizes, source_bases, sorted_new_bases, position_sampler)` — frozen dataclass; per-gene calibrated sampler (build via private `_build_gene_sampler(applicable, k)`).
  - `.draw(rng: np.random.Generator) -> List[Tuple[int, str, str]]` — one set of k `(position, source_base, new_base)` tuples, ascending, distinct positions.
- `sample_baseline_sequence(pool: MutationPool, gene_id: str, n_mutations: int, rng: np.random.Generator) -> str` — one mutated baseline sequence for a gene (re-calibrates each call, ~0.3 s; KeyError if unknown gene, ValueError if < k applicable positions).
- `generate_baselines_fasta(pool: MutationPool, output_path: str, n_per_gene: int = 1, rng: Optional[np.random.Generator] = None) -> None` — for every gene draw `n_per_gene` baselines with k from gene_stats; writes FASTA headers `>{gene_id}_baseline_{i:03d}`; genes with too few applicable positions skipped with warning.
- `parse_args() -> argparse.Namespace` — CLI parsing (validates input file exists).

## evaluate_sequences.py  [script]
Step 4: one-hot (`deepCRE.utils.one_hot_encode`) + Keras `.h5` predict of every record in a FASTA, writes CSV `sequence_name,prediction`. Run directly, hard-wired paths, no CLI. **[possibly broken]** `ZEA_MSR_PATH` points at the `..._A_thaliana_msr_train_models...h5` and `ARA_MSR_PATH` at the `..._Z_mays_msr...h5` model, whereas `Evolution/src/evolution/write_run_script.py` pairs A_thaliana model with Arabidopsis runs and Z_mays with maize — species/model look swapped (unverified).

- `evaluate_sequences(model_path: str, fasta_path: str, output_path: str) -> None` — predict all FASTA records with one model, save CSV.

## analyse_results.py  [script]
Compares mean model score before optimisation / after random baseline mutations / after optimisation per species (grouped seaborn bar plot). Run directly; reads `mutated_sequences/*_predictions.csv` and hard-coded `summary_{ara,zea}_msr_max_single.txt` under the ARCitect dataset. Labels say "deepCIS" but the random-mutation values come from evaluate_sequences.py (deepCRE MSR models).

- `average_predications_after_random_mutations()` — mean of `prediction` column in the two CSVs; prints and writes to `AVERAGE_PREDICTIONS_PATH` (hard-coded paths).
- `load_random_mutation_averages(filepath: str) -> dict[str, float]` — parse "Arabidopsis:"/"Maize:" lines of that text file.
- `load_evolution_summary(filepath: str, run_key: str) -> dict[str, float]` — parse `start_fitness_mean` / `final_fitness_mean` from the "Summary for {run_key}:" block of a `summary_*.txt` (written by `analysis/overview/simple_result_stats.py`).
- `build_plot_dataframe(ara_summary_path: str, zea_summary_path: str, random_mutations_path: str) -> pd.DataFrame` — tidy (species, condition, score), run keys hard-coded to `ara/zea_msr_max_single`.
- `plot_optimization_vs_random(data: pd.DataFrame, output_dir: str | None = None, fmt: str = "png", ax: plt.Axes | None = None) -> None` — grouped bar plot; saves only when `ax` is None.
- `save_figure(fig: plt.Figure, filename: str, output_dir: str, fmt: str = "png") -> None` — save to `output_dir/filename.fmt` at dpi 150 (duplicate of several other `save_figure`s in the repo).

## natural_variation_in_window.py  [script]
Prints how many natural SNPs (from re-extracted per-gene VCFs, GOF/LOF Arabidopsis) fall in the 170 bp plasmid test window (-220..-51 rel. TSS = 1-based 781..950 in the 3020 bp deepCRE frame) vs. the rest of the promoter half (1..1500). Run: `python -m workflows.mutation_distribution_analysis.natural_variation_in_window`.

- `VCF_DIRECTORIES` — {"GOF"/"LOF": `Evolution/data/Arabidopsis_{GOF,LOF}_reextracted_vcfs`}.
- `WINDOW_START`, `WINDOW_END`, `WINDOW_LENGTH` — 781, 950, 170 (1-based inclusive, deepCRE construct coords).
- `PROMOTER_START`, `PROMOTER_END`, `PROMOTER_LENGTH` — 1, 1500, 1500.
- `read_vcf_positions(vcf_path: str) -> pd.DataFrame` — CHROM/POS columns of a VCF as `gene_id`, `position` (1-based).
- `load_variation(vcf_directory: str) -> pd.DataFrame` — concatenate all `*.vcf` in a directory (ValueError if none).
- `summarize(dataset_name: str, variation: pd.DataFrame) -> None` — print per-gene window counts and SNP density window vs. promoter-outside-window (genes = only those with ≥1 SNP anywhere).

## plans/  [docs]
`plan.md` (original PMF plan, superseded), `mutation_pool_plan.md`, `baseline_sampler_plan.md`, `sampler_inclusion_probability_fix.md` (why successive sampling flattens inclusion probabilities), `conditional_poisson_implementation_plan.md`.

## plans/sampler_fix_scratch/  [one-off]
Exploration scripts for the sampler fix; several import private `baseline_sampler._filter_applicable` or each other via `sys.path` hacks.
- `accept_baseline.py` — predicted Sampford acceptance rate on per-gene applicable pools.
- `bench_cdl.py` — timing of CDL calibration vs. odds-ratio variant; **[broken]** imports `verify_cdl` via a session-specific /tmp scratchpad path (likely gone).
- `callsites.py` — facts (distinct k values, per-gene pool variation) to scope the fix at both call sites.
- `check_bias.py` — quantify successive-sampling vs πps bias in the distance null (pool path as argv[1]).
- `compare_updates.py` — compare CDL update vs. odds-scale update convergence on a toy pool.
- `cond_poisson_explained.py` — exact conditional-Poisson inclusion probabilities by enumeration (toy).
- `condpoisson_viability.py` — acceptance rate of rejective conditional Poisson on real pools.
- `feasibility.py` — is max k·p < 1 on per-gene applicable pools (Sampford feasibility).
- `pivotal.py` — Deville-Tillé pivotal-method sampler prototype.
- `pre_distort.py` — showing input weights for successive sampling must be pre-distorted to hit π = k·p.
- `recover_pmf.py` — Monte Carlo test of whether position-blocked sampling recovers the input positional PMF.
- `sampford.py` — Sampford's method prototype vs. successive sampling.
- `sampford_real.py` — Sampford feasibility/cost on the real ara pool.
- `second_order.py` — do Sampford and pivotal give the same distance null (second-order inclusion).
- `toy.py` — exact successive-sampling inclusion probabilities, k=2, 4 positions.
- `trace.py` — trace odds-ratio calibration iterations on a toy pool.
- `verify_cdl.py` — brute-force check of CDL eq. 11 and the scaled prefix/suffix R recursion.
- `vs_real.py` — real vs. successive-sampling null inter-mutation distances (KS/Wasserstein).
- `why_rejection.py` — where Sampford's ~2% acceptance comes from (collision analysis).
