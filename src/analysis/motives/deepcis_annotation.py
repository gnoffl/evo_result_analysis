"""Mutation annotation logic for deepCIS window scan results."""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from analysis.motives.deepcis_scanner import (
    N_TF_FAMILIES,
    TF_FAMILY_NAMES,
    DEFAULT_DELTA_THRESHOLD,
    DEFAULT_EXTRAGENIC,
    DEFAULT_INTRAGENIC,
    DEFAULT_CENTRAL_PADDING,
    GeneRunData,
    _get_max_mutation_entry,
    get_full_sequence,
    _get_zero_mutation_entry,
)
from evolution.sequences import one_hot_encode, one_hot_decode, compare_sequences

def _mutations_from_ohe(
    ref_ohe: np.ndarray,
    max_ohe: np.ndarray,
) -> List[Tuple[int, str, str]]:
    diff_positions = compare_sequences(ref_ohe, max_ohe)
    mutations: List[Tuple[int, str, str]] = []
    for pos in diff_positions:
        pos = int(pos)
        ref_base = one_hot_decode(ref_ohe[pos : pos + 1])
        mut_base = one_hot_decode(max_ohe[pos : pos + 1])
        mutations.append((pos, ref_base, mut_base))
    return mutations

def compare_sequences_df(
    df: pd.DataFrame,
    threshold: float = DEFAULT_DELTA_THRESHOLD,
) -> pd.DataFrame:
    ref_df = df[df["sequence_type"] == "reference"].drop(columns="sequence_type")
    max_df = df[df["sequence_type"] == "max_mutated"].drop(columns="sequence_type")

    merge_keys = ["gene", "window_start", "window_end", "contains_padding"]
    merged = ref_df.merge(max_df, on=merge_keys, suffixes=("_ref", "_max_mut"))

    delta_cols: List[str] = []
    for tf_name in TF_FAMILY_NAMES:
        col_ref = f"{tf_name}_ref"
        col_max = f"{tf_name}_max_mut"
        delta_col = f"delta_{tf_name}"
        merged[delta_col] = merged[col_max] - merged[col_ref]
        delta_cols.append(delta_col)

    merged["max_delta"] = merged[delta_cols].abs().max(axis=1)
    merged["binding_changed"] = merged["max_delta"] >= threshold

    return merged

def annotate_mutations_in_windows(
    changed_df: pd.DataFrame,
    genes_data: Dict[str, GeneRunData],
) -> pd.DataFrame:
    mutation_lists: List[List[Tuple[int, str, str]]] = []

    for _, row in changed_df.iterrows():
        gene_name = row["gene"]
        win_start = int(row["window_start"])
        win_end = int(row["window_end"])

        gd = genes_data.get(gene_name)
        if gd is None:
            mutation_lists.append([])
            continue

        max_entry = _get_max_mutation_entry(gd.pareto_front)
        zero_entry = _get_zero_mutation_entry(gd.pareto_front)
        
        ref_full = get_full_sequence(
            zero_entry[0],
            gd.reference_sequence_full,
            gd.mutation_start,
            gd.mutation_end,
        )
        max_full = get_full_sequence(
            max_entry[0],
            gd.reference_sequence_full,
            gd.mutation_start,
            gd.mutation_end,
        )
        ref_ohe = one_hot_encode(ref_full)
        max_ohe = one_hot_encode(max_full)
        all_mutations = _mutations_from_ohe(ref_ohe, max_ohe)
        in_window = [
            (pos, ref, mut)
            for pos, ref, mut in all_mutations
            if win_start <= pos < win_end
        ]
        mutation_lists.append(in_window)

    result = changed_df.copy()
    result["mutations_in_window"] = mutation_lists
    result["n_mutations_in_window"] = result["mutations_in_window"].apply(len)
    return result

def add_genomic_coordinates(
    df: pd.DataFrame,
    extragenic: int = DEFAULT_EXTRAGENIC,
    intragenic: int = DEFAULT_INTRAGENIC,
    central_padding: int = DEFAULT_CENTRAL_PADDING,
) -> pd.DataFrame:
    from analysis.mutations.genomic_annotation import AnnotatedMutatedSequence

    promoter_length = extragenic + intragenic
    terminator_start_pos = promoter_length + central_padding

    def _map_pos(seq_pos: int, gene_start: int, gene_end: int, strand: str) -> int:
        if strand == "+":
            if seq_pos < promoter_length:
                return gene_start - extragenic + seq_pos + 1
            else:
                return gene_end - intragenic + (seq_pos - terminator_start_pos) + 1
        else:
            if seq_pos < promoter_length:
                return gene_start + extragenic - seq_pos
            else:
                return gene_end + intragenic - (seq_pos - terminator_start_pos)

    records: List[dict] = []
    for _, row in df.iterrows():
        try:
            gene_id, chromosome, gene_start, gene_end = (
                AnnotatedMutatedSequence.parse_sequence_name(row["gene"])
            )
            strand = "+" if gene_start < gene_end else "-"
            gw_start = _map_pos(int(row["window_start"]), gene_start, gene_end, strand)
            gw_end = _map_pos(int(row["window_end"]) - 1, gene_start, gene_end, strand)
        except Exception:
            gene_id, chromosome, strand = row["gene"], "unknown", "unknown"
            gw_start, gw_end = None, None

        records.append({
            "gene_id": gene_id,
            "chromosome": chromosome,
            "strand": strand,
            "genomic_window_start": gw_start,
            "genomic_window_end": gw_end,
        })

    coord_df = pd.DataFrame(records, index=df.index)
    return pd.concat([df, coord_df], axis=1)

