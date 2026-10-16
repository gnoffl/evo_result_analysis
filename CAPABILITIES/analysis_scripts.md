# analysis_scripts/

Two bash entry points that chain the Python analysis modules over one evolutionary-run results folder. Both take the same three positional arguments (`<results_folder> <analysis_name> <output_folder>`) and pass optional flags straight through to the underlying modules, omitting any flag that was not given so the Python defaults apply. Both call bare `python -m src.…`, so the `deepCREshap` env has to be active already. `run_deepcis_peak_pipeline.sh` cds to the repo root first; `run_full_analysis.sh` does not, so it must be launched from the repo root.

## run_deepcis_peak_pipeline.sh  [script]
`./analysis_scripts/run_deepcis_peak_pipeline.sh <results_folder> <analysis_name> <output_folder> [options]` — deepCIS sliding-window scan, peak calling, peak summary and plots. `set -euo pipefail`, so any failing stage aborts the run.

Stages:
1. `python -m src.analysis.motives.deepcis_scanner --run-folder --output --name --model [--window-size --step --batch-size --overwrite --mutation-count --genes]` -> `<output>/deepcis_scan/deepcis_window_scan_<analysis_name>.csv`.
2. `python -m src.analysis.motives.peak_scanner <scan_csv> --output-path --signal-type --log-level [--annotator-window-size --annotator-step-size --annotator-threshold-peak --annotator-sigma --annotator-lambda-weight --genes --tfs]` -> `<analysis_name>_annotated_peaks_<signal_label>_<timestamp>.csv` in the same folder.
3. `python -m src.analysis.motives.analyze_peaks <peak_csv> --summarize-peaks --output-folder --base_name` -> `<analysis_name>_peak_summary.csv`.
4. `python -m src.analysis.motives.deepcis_visualize --input <scan_csv> --peaks <peak_csv> --output --format [--random-subset --no-highlight-padding --no-highlight-peaks --genes --tfs]` -> `deepcis_scan/deepcis_scan_plots/<gene>/`. The plot folder is **deleted and recreated** on every run.

Key options and behaviour:
- `--model <path>` — default `models/deepcis/deepCIS_model_chrom_1_model.h5`, resolved relative to the repo root when not absolute. Existence is checked before anything runs.
- `--signal-type reference|optimized|difference|all` (default `all`) — validated up front; `all` is labelled `reference_optimized_difference` in the peak file name.
- `--mutation-count <int>` — selects the optimized sequence at that exact mutation count instead of the max-mutation entry, and appends `_mut<N>` to `analysis_name`, so every downstream filename carries it. Must be a non-negative integer.
- `--genes "g1 g2"` / `--tfs "tf_0 tf_1"` — comma- or space-separated; the gene filter also reaches the scanner. `--random-subset` is only added to the visualization when neither filter is given.
- `--overwrite`, `--deepcis-{window-size,step,batch-size}`, `--annotator-*`, `--log-level`, `--viz-format`, `--no-highlight-padding`, `--no-highlight-peaks`.
- After stages 1 and 2 the expected output file is checked and the script exits 1 if it is missing.

## run_full_analysis.sh  [script]
`./analysis_scripts/run_full_analysis.sh <results_folder> <analysis_name> <output_folder> [options]` — Pareto-front statistics, mutation summarization and mutation analysis. It has **no `set -e`** and builds the commands as strings run through `eval`, so a path with spaces breaks it; each stage's failure is recorded in `OVERALL_SUCCESS` and the script exits 1 at the end (stage 3 is skipped if stage 2 produced no file).

Stages:
1. `python -m src.analysis.overview.simple_result_stats --results_folder --name --output_folder --all [--last_generation --output_format --overwrite --no_titles]`.
2. `python -m src.analysis.mutations.summarize_mutations --results_folder --name --output_folder [--final_generation --generation]`. Its stdout is captured and the mutation data path is grepped out of the `OUTPUT_FILE=` line it prints.
3. `python -m src.analysis.mutations.analyze_mutations --mutation_data <from step 2> --name --output_folder --all [--window_size --mutable_positions --output_format --generation --no_titles]`.

Key options: `-l/--last-generation` (stage 1 only), `-f/--final-generation` and `-g/--generation` (stage 2; `-g` also reaches stage 3), `-w/--window-size` and `-m/--mutable-positions` (stage 3), `-o/--output-format` and `--no-titles` (stages 1 and 3), `--overwrite` (stage 1 only).
