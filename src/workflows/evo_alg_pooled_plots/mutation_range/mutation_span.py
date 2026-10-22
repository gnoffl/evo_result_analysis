"""Span and border positions of the 5-mutation Pareto-front sequences.

For each gene of a run we take the Pareto-front entry that carries exactly
``n_mutations`` mutations and record where those mutations sit inside the
reference window: the position of the first mutation, the position of the last
mutation, and the span between them (``last - first``). The per-run summary is
a ``DataFrame.describe()`` over those three quantities.
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from analysis.mutations.summarize_mutations import (
    MutatedSequence,
    load_mutations_from_json,
)

DATASET_ROOT = (
    "/home/gernot/ARCitect/ARCs/dream/assays/Evo_run_analysis/dataset"
)

RUN_JSON_FILES: Dict[str, str] = {
    "ara_msr_max": os.path.join(
        DATASET_ROOT,
        "paper_runs/single_mutation/ara_msr_max_single",
        "all_mutated_sequences_ara_msr_max_single_gen1999.json",
    ),
    "ara_msr_min": os.path.join(
        DATASET_ROOT,
        "paper_runs/single_mutation/ara_msr_min_single",
        "all_mutated_sequences_ara_msr_min_single_gen1999.json",
    ),
    "zea_msr_max": os.path.join(
        DATASET_ROOT,
        "paper_runs/single_mutation/zea_msr_max_single",
        "all_mutated_sequences_zea_msr_max_single_gen1999.json",
    ),
    "zea_msr_min": os.path.join(
        DATASET_ROOT,
        "paper_runs/single_mutation/zea_msr_min_single",
        "all_mutated_sequences_zea_msr_min_single_gen1999.json",
    ),
    "GOF": os.path.join(
        DATASET_ROOT,
        "GOF_LOF/GOF/GOF_single",
        "all_mutated_sequences_GOF_single_gen19999.json",
    ),
    "LOF": os.path.join(
        DATASET_ROOT,
        "GOF_LOF/LOF/LOF_single",
        "all_mutated_sequences_LOF_single_gen19999.json",
    ),
}

SPAN_COLUMNS = ["first_position", "last_position", "span"]

RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def select_sequences_with_mutation_count(
    sequences: List[MutatedSequence], n_mutations: int
) -> List[MutatedSequence]:
    """Return the sequences carrying exactly ``n_mutations`` mutations.

    Args:
        sequences: Pareto-front sequences of a single gene.
        n_mutations: Required number of mutations.

    Returns:
        All matching sequences, in input order (usually zero or one entry).
    """
    return [
        sequence
        for sequence in sequences
        if sequence.get_mutation_number() == n_mutations
    ]


def collect_run_spans(json_file: str, n_mutations: int = 5) -> pd.DataFrame:
    """Collect first/last mutation position and span per gene of one run.

    Genes without a Pareto-front entry of exactly ``n_mutations`` mutations are
    skipped. If a gene has several such entries, the highest-fitness one is
    used.

    Args:
        json_file: Path to an ``all_mutated_sequences_*.json`` file.
        n_mutations: Number of mutations the selected sequence must carry.

    Returns:
        DataFrame with columns ``gene_id``, ``generation``, ``fitness``,
        ``first_position``, ``last_position``, ``span``, ``reference_length``.

    Raises:
        FileNotFoundError: If ``json_file`` does not exist.
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Mutation summary file not found: {json_file}")

    genes = load_mutations_from_json(json_file)
    rows = []
    for gene_id, gene in genes.items():
        for generation, sequences in gene.generation_dict.items():
            candidates = select_sequences_with_mutation_count(sequences, n_mutations)
            if not candidates:
                continue
            selected = max(candidates, key=lambda sequence: sequence.fitness)
            positions = sorted(position for position, _, _ in selected.mutations)
            rows.append(
                {
                    "gene_id": gene_id,
                    "generation": generation,
                    "fitness": selected.fitness,
                    "first_position": positions[0],
                    "last_position": positions[-1],
                    "span": positions[-1] - positions[0],
                    "reference_length": len(gene.reference_sequence),
                }
            )
    return pd.DataFrame(rows, columns=[
        "gene_id",
        "generation",
        "fitness",
        "first_position",
        "last_position",
        "span",
        "reference_length",
    ])


def describe_spans(span_table: pd.DataFrame) -> pd.DataFrame:
    """Describe the positional columns of a span table.

    Args:
        span_table: Output of :func:`collect_run_spans`.

    Returns:
        ``describe()`` output restricted to ``first_position``,
        ``last_position`` and ``span``.
    """
    return span_table[SPAN_COLUMNS].describe()


def collect_all_runs(
    run_json_files: Dict[str, str] | None = None, n_mutations: int = 5
) -> pd.DataFrame:
    """Collect span tables for all runs into one long-format table.

    Args:
        run_json_files: Mapping run name -> mutation summary JSON path.
            Defaults to :data:`RUN_JSON_FILES`.
        n_mutations: Number of mutations the selected sequences must carry.

    Returns:
        Concatenated per-gene table with an additional ``run`` column.
    """
    if run_json_files is None:
        run_json_files = RUN_JSON_FILES
    tables = []
    for run_name, json_file in run_json_files.items():
        table = collect_run_spans(json_file, n_mutations=n_mutations)
        table.insert(0, "run", run_name)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def describe_all_runs(all_runs_table: pd.DataFrame) -> pd.DataFrame:
    """Stack the per-run ``describe()`` outputs into a single table.

    Args:
        all_runs_table: Output of :func:`collect_all_runs`.

    Returns:
        DataFrame indexed by ``(run, statistic)`` with the columns of
        :data:`SPAN_COLUMNS`.
    """
    summaries = {}
    for run_name, run_table in all_runs_table.groupby("run", sort=False):
        summaries[run_name] = describe_spans(run_table)
    summary = pd.concat(summaries, names=["run", "statistic"])
    return summary


def main() -> None:
    """Write per-gene spans and per-run summaries to ``results/``."""
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    all_runs_table = collect_all_runs()
    summary = describe_all_runs(all_runs_table)
    pooled = describe_spans(all_runs_table)

    per_gene_path = os.path.join(RESULTS_FOLDER, "mutation_span_per_gene_5mut.csv")
    summary_path = os.path.join(RESULTS_FOLDER, "mutation_span_describe_5mut.csv")
    all_runs_table.to_csv(per_gene_path, index=False)
    summary.to_csv(summary_path)
    pooled_path = os.path.join(RESULTS_FOLDER, "mutation_span_describe_5mut_pooled.csv")
    pooled.to_csv(pooled_path)

    with pd.option_context("display.width", 120, "display.float_format", "{:.2f}".format):
        for run_name, run_table in all_runs_table.groupby("run", sort=False):
            print(f"\n=== {run_name} (n genes = {len(run_table)}, "
                  f"reference length = {sorted(run_table['reference_length'].unique())}) ===")
            print(describe_spans(run_table))

        print(f"\n=== all experiments pooled (n genes = {len(all_runs_table)}) ===")
        print(pooled)

    print(f"\nWrote {per_gene_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {pooled_path}")


if __name__ == "__main__":
    main()
