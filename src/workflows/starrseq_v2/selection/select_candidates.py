"""Select the STARR-seq v2 library candidates from the pooled front table.

Every run contributes its reference, half-max and max-mutation entries. The rest of the
TARGET is filled by drawing one further entry per run in rounds, uniformly along the
mutation-count axis. Drawing on the axis rather than over the entries themselves keeps
the sparse high-mutation stretch of a front from being underrepresented just because it
has fewer entries.

TARGET counts selected entries. Step 3 deduplicates on the exact sequence and will land
a little below it — by about 1400 at the measured scale, almost all of them references.
"""
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

OUTPUT = os.path.join(os.path.dirname(__file__), "output")
# entries to select in total; the three anchors per run are always taken, so this
# cannot go below their count (9764 runs x anchors at the measured scale)
TARGET = 57_000
SEED = 0

ANCHOR_FLAGS = {"reference": "is_reference", "half_max": "is_half_max", "max": "is_max_mutations"}


def anchor_reasons(front: pd.DataFrame) -> Dict[int, str]:
    """Front index of every anchor entry of one run, mapped to the anchors it satisfies.

    One entry can be several anchors at once on a short front. Where two entries share a
    mutation count, the lower front index wins — fronts run from optimised to reference.
    """
    reasons: Dict[int, List[str]] = {}
    for reason, flag in ANCHOR_FLAGS.items():
        flagged = front.loc[front[flag], "front_index"]
        if not flagged.empty:
            reasons.setdefault(int(flagged.min()), []).append(reason)
    return {index: "|".join(names) for index, names in reasons.items()}


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


def fill_to_target(
    fronts: Dict[str, pd.DataFrame],
    reasons: Dict[str, Dict[int, str]],
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
    anchor_count = sum(len(entries) for entries in reasons.values())
    if target < anchor_count:
        raise ValueError(f"target {target} is below the {anchor_count} anchor entries that are always taken")
    budget = target - anchor_count
    order = [str(run_dir) for run_dir in rng.permutation(list(fronts))]
    while budget > 0:
        drawn_in_round = 0
        for run_dir in order:
            if budget == 0:
                break
            index = draw_one(*pools[run_dir], rng=rng)
            if index is not None:
                reasons[run_dir][index] = "random"
                budget -= 1
                drawn_in_round += 1
        if drawn_in_round == 0:
            break
    return max(budget, 0)


def select_candidates(front_table: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """Select exactly target entries across all runs, skipping entries that contain N.

    Raises:
        ValueError: If target is below the anchor count or exceeds the usable entries.

    Returns:
        Columns run_dir, front_index, selection_reason.
    """
    usable = front_table[front_table["n_count"] == 0]
    rng = np.random.default_rng(seed)
    fronts = {
        str(run_dir): pd.DataFrame(front).sort_values(by="front_index")
        for run_dir, front in usable.groupby("run_dir", sort=True)
    }
    reasons = {run_dir: anchor_reasons(front) for run_dir, front in fronts.items()}
    shortfall = fill_to_target(fronts, reasons, target, rng)
    if shortfall:
        raise ValueError(f"{shortfall} entries short of the target, every front is exhausted")
    selected = []
    for run_dir in sorted(reasons):
        indices = sorted(reasons[run_dir])
        selected.append(pd.DataFrame({
            "run_dir": run_dir,
            "front_index": indices,
            "selection_reason": [reasons[run_dir][index] for index in indices],
        }))
    return pd.concat(selected, ignore_index=True)


def main() -> None:
    """Write selected_keys.csv next to the pooled front table."""
    # the default float parser is not exact; round_trip reads back the written value
    front_table = pd.read_csv(  # pyright: ignore[reportCallIssue]
        os.path.join(OUTPUT, "front_table.csv.gz"),
        float_precision="round_trip",  # pyright: ignore[reportArgumentType]
    )
    selected = select_candidates(front_table, TARGET, SEED)
    selected.to_csv(os.path.join(OUTPUT, "selected_keys.csv"), index=False)
    per_run = selected.groupby("run_dir").size()
    print(f"selected {len(selected)} entries from {len(per_run)} runs (target {TARGET})")
    print(f"entries per run: min {per_run.min()}, max {per_run.max()}")
    print(selected["selection_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
