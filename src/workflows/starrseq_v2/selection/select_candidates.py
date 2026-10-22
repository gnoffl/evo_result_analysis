"""Select the STARR-seq v2 library candidates from the pooled front table.

Selection runs in three stages, each masked out of the next, so TARGET is spent in
order of how specific the claim on a slot is:

1. **Anchors** — every run contributes its reference, half-max and max-mutation entries.
2. **Best picks** — TARGET_BEST entries ranked globally across all runs, half of them by
   fitness gain at each of BEST_MUTATION_COUNTS, half by fitness gain per mutation.
3. **Uniform fill** — the rest of TARGET, one entry per run per round, drawn uniformly
   along the mutation-count axis. Drawing on the axis rather than over the entries
   themselves keeps the sparse high-mutation stretch of a front from being
   underrepresented just because it has fewer entries.

An entry already taken by an earlier stage gains the later stage's reason but does not
consume its budget, so a stage always contributes exactly its quota of new entries.

TARGET counts selected entries. Step 3 deduplicates on the exact sequence and will land
a little below it — by about 1400 at the measured scale, almost all of them references.
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

OUTPUT = os.path.join(os.path.dirname(__file__), "output")
# entries to select in total; the three anchors per run are always taken, so this
# cannot go below their count (9764 runs x anchors at the measured scale)
TARGET = 64_000
# of those, the globally ranked best picks, split evenly over the two rankings
TARGET_BEST = 6_400
SEED = 0

ANCHOR_FLAGS = {"reference": "is_reference", "half_max": "is_half_max", "max": "is_max_mutations"}
# the counts at which the strongest entries are picked; a run strong enough to win at
# several of them contributes a mutation ladder, which is intended
BEST_MUTATION_COUNTS = (1, 2, 3, 5, 7, 10, 15)


def anchor_reasons(front: pd.DataFrame) -> Dict[int, List[str]]:
    """Front index of every anchor entry of one run, mapped to the anchors it satisfies.

    One entry can be several anchors at once on a short front. Where two entries share a
    mutation count, the lower front index wins — fronts run from optimised to reference.
    """
    reasons: Dict[int, List[str]] = {}
    for reason, flag in ANCHOR_FLAGS.items():
        flagged = front.loc[front[flag], "front_index"]
        if not flagged.empty:
            reasons.setdefault(int(flagged.min()), []).append(reason)
    return reasons


def draw_one(
    n_mutations: np.ndarray,
    front_indices: np.ndarray,
    available: np.ndarray,
    max_count: int,
    rng: np.random.Generator,
) -> Optional[int]:
    """Draw one front index uniformly on the mutation-count axis, snapped to the nearest free entry.

    Args:
        available: Which entries may still be drawn; the drawn one is cleared in place.
        max_count: The run's highest mutation count; the draw falls in [1, max_count - 1],
            since counts 0 and max_count are already anchors.

    Returns:
        The drawn front index, or None when the run has nothing left.
    """
    if not available.any():
        return None
    draw = rng.uniform(1.0, max(1.0, max_count - 1.0))
    distance = np.where(available, np.abs(n_mutations - draw), np.inf)
    # equal distance goes to the lower front index, the more optimised entry
    nearest = int(np.lexsort((front_indices, distance))[0])
    available[nearest] = False
    return int(front_indices[nearest])


def split_evenly(total: int, parts: int) -> List[int]:
    """Split total into parts differing by at most one; the earlier parts take the remainder."""
    base, remainder = divmod(total, parts)
    return [base + int(part < remainder) for part in range(parts)]


def rank_by_score(entries: pd.DataFrame, score: pd.Series) -> List[Tuple[str, int]]:
    """The (run_dir, front_index) keys of entries, best score first.

    Args:
        score: Aligned with the index of entries. Ties go to the lower run_dir, then the
            lower front index, so the ranking does not depend on row order.
    """
    order = pd.DataFrame({
        "score": score,
        "run_dir": entries["run_dir"],
        "front_index": entries["front_index"],
    }).sort_values(by=["score", "run_dir", "front_index"], ascending=[False, True, True])
    return [(str(run_dir), int(index)) for run_dir, index in zip(order["run_dir"], order["front_index"])]


def take_best(
    ranking: Sequence[Tuple[str, int]],
    reasons: Dict[str, Dict[int, List[str]]],
    reason: str,
    quota: int,
) -> int:
    """Label entries down the ranking until quota previously unselected ones have been added.

    An entry an earlier stage already took gains the reason but does not consume the
    quota, so the stage adds exactly quota new entries wherever the ranking is long
    enough. `reasons` is extended in place.

    Returns:
        The shortfall: slots left unfilled because the ranking ran out.
    """
    added = 0
    for run_dir, front_index in ranking:
        if added == quota:
            break
        selected = reasons[run_dir]
        if front_index in selected:
            selected[front_index].append(reason)
        else:
            selected[front_index] = [reason]
            added += 1
    return quota - added


def select_best(
    usable: pd.DataFrame,
    reasons: Dict[str, Dict[int, List[str]]],
    target_best: int,
    mutation_counts: Sequence[int],
) -> int:
    """Add the globally strongest entries to reasons in place, half per ranking.

    The fixed-count ranking goes first and takes an equal share at every count in
    mutation_counts, so the gain-per-mutation ranking spends its half on what is left.
    Both rank across all runs at once and neither caps how much one run may contribute.

    Args:
        usable: Front entries that may be selected, with fitness_delta_vs_reference.

    Returns:
        The shortfall summed over every ranking.
    """
    by_fitness, by_efficiency = split_evenly(target_best, 2)
    shortfall = 0
    for count, quota in zip(mutation_counts, split_evenly(by_fitness, len(mutation_counts))):
        at_count = pd.DataFrame(usable[usable["n_mutations"] == count])
        ranking = rank_by_score(at_count, pd.Series(at_count["fitness_delta_vs_reference"]))
        shortfall += take_best(ranking, reasons, f"best_at_{count}", quota)
    # references gain nothing by definition and would divide by zero
    mutated = pd.DataFrame(usable[usable["n_mutations"] > 0])
    efficiency = mutated["fitness_delta_vs_reference"] / mutated["n_mutations"]
    shortfall += take_best(rank_by_score(mutated, efficiency), reasons, "best_efficiency", by_efficiency)
    return shortfall


def fill_to_target(
    fronts: Dict[str, pd.DataFrame],
    reasons: Dict[str, Dict[int, List[str]]],
    target: int,
    rng: np.random.Generator,
) -> int:
    """Draw entries round by round until target is reached or every front is exhausted.

    One draw per run per round, so the runs stay within one entry of each other until a
    front runs dry. Runs are visited in a random order, so the partial final round does
    not always favour the same runs. `reasons` is extended in place.

    Returns:
        The shortfall: entries still wanted when every front was exhausted, normally 0.
    """
    pools = {}
    for run_dir, front in fronts.items():
        front_indices = front["front_index"].to_numpy()
        pools[run_dir] = (
            front["n_mutations"].to_numpy(),
            front_indices,
            ~np.isin(front_indices, list(reasons[run_dir])),
            int(front["max_mutations_reached"].iloc[0]),
        )
    already_taken = sum(len(entries) for entries in reasons.values())
    if target < already_taken:
        raise ValueError(f"target {target} is below the {already_taken} entries the earlier stages already took")
    budget = target - already_taken
    order = [str(run_dir) for run_dir in rng.permutation(list(fronts))]
    while budget > 0:
        drawn_in_round = 0
        for run_dir in order:
            if budget == 0:
                break
            index = draw_one(*pools[run_dir], rng=rng)
            if index is not None:
                reasons[run_dir][index] = ["random"]
                budget -= 1
                drawn_in_round += 1
        if drawn_in_round == 0:
            break
    return max(budget, 0)


def select_candidates(
    front_table: pd.DataFrame,
    target: int,
    seed: int,
    target_best: int = 0,
    mutation_counts: Sequence[int] = BEST_MUTATION_COUNTS,
) -> pd.DataFrame:
    """Select exactly target entries across all runs, skipping entries that contain N.

    Args:
        target_best: How many of target go to the globally ranked best picks, drawn
            after the anchors and before the uniform fill.

    Raises:
        ValueError: If target is below what the earlier stages take, if a best-pick
            ranking runs out, or if the fronts cannot supply target.

    Returns:
        Columns run_dir, front_index, selection_reason, the reasons joined with '|'.
    """
    usable = pd.DataFrame(front_table[front_table["n_count"] == 0])
    rng = np.random.default_rng(seed)
    fronts = {
        str(run_dir): pd.DataFrame(front).sort_values(by="front_index")
        for run_dir, front in usable.groupby("run_dir", sort=True)
    }
    reasons = {run_dir: anchor_reasons(front) for run_dir, front in fronts.items()}
    best_shortfall = select_best(usable, reasons, target_best, mutation_counts)
    if best_shortfall:
        raise ValueError(f"{best_shortfall} best picks short, the rankings are exhausted")
    shortfall = fill_to_target(fronts, reasons, target, rng)
    if shortfall:
        raise ValueError(f"{shortfall} entries short of the target, every front is exhausted")
    selected = []
    for run_dir in sorted(reasons):
        indices = sorted(reasons[run_dir])
        selected.append(pd.DataFrame({
            "run_dir": run_dir,
            "front_index": indices,
            "selection_reason": ["|".join(reasons[run_dir][index]) for index in indices],
        }))
    return pd.concat(selected, ignore_index=True)


def main() -> None:
    """Write selected_keys.csv next to the pooled front table."""
    # the default float parser is not exact; round_trip reads back the written value
    front_table = pd.read_csv(  # pyright: ignore[reportCallIssue]
        os.path.join(OUTPUT, "front_table.csv.gz"),
        float_precision="round_trip",  # pyright: ignore[reportArgumentType]
    )
    selected = select_candidates(front_table, TARGET, SEED, TARGET_BEST)
    selected.to_csv(os.path.join(OUTPUT, "selected_keys.csv"), index=False)
    per_run = selected.groupby("run_dir").size()
    print(f"selected {len(selected)} entries from {len(per_run)} runs (target {TARGET})")
    print(f"entries per run: min {per_run.min()}, max {per_run.max()}")
    # an entry can carry several reasons, so these count labels, not entries
    print(selected["selection_reason"].str.split("|").explode().value_counts().to_string())


if __name__ == "__main__":
    main()
