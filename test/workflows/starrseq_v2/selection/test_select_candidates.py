import unittest
from typing import List, Optional

import numpy as np
import pandas as pd

from workflows.starrseq_v2.selection import select_candidates as sc


def make_front(
    n_mutations: List[int],
    half_max: int,
    run_dir: str = "config/run",
    n_count: Optional[List[int]] = None,
) -> pd.DataFrame:
    """One run's front table rows, in front order (index 0 is the most optimised entry).

    Args:
        n_mutations: Mutation count per entry; the maximum becomes max_mutations_reached.
    """
    counts = pd.Series(n_mutations)
    return pd.DataFrame({
        "run_dir": run_dir,
        "front_index": range(len(n_mutations)),
        "n_mutations": counts,
        "n_count": n_count if n_count is not None else 0,
        "max_mutations_reached": counts.max(),
        "is_reference": counts == 0,
        "is_half_max": counts == half_max,
        "is_max_mutations": counts == counts.max(),
    })


def many_fronts(n_runs: int, counts: List[int], half_max: int) -> pd.DataFrame:
    return pd.concat(make_front(counts, half_max, run_dir=f"config/run{i}") for i in range(n_runs))


def pool_of(front: pd.DataFrame):
    """The sampling arrays of one front, with its anchors already removed."""
    reasons = sc.anchor_reasons(front)
    front_indices = front["front_index"].to_numpy()
    return (
        front["n_mutations"].to_numpy(),
        front_indices,
        ~np.isin(front_indices, list(reasons)),
        int(front["max_mutations_reached"].iloc[0]),
    )


class TestAnchorReasons(unittest.TestCase):
    def test_three_distinct_anchors(self):
        front = make_front([10, 5, 0], half_max=5)
        self.assertEqual(sc.anchor_reasons(front), {0: "max", 1: "half_max", 2: "reference"})

    def test_one_entry_is_every_anchor(self):
        """A front of length 1 — nothing but the reference survived."""
        self.assertEqual(sc.anchor_reasons(make_front([0], half_max=0)), {0: "reference|half_max|max"})

    def test_half_max_coincides_with_max(self):
        front = make_front([1, 0], half_max=1)
        self.assertEqual(sc.anchor_reasons(front), {0: "half_max|max", 1: "reference"})

    def test_tie_at_a_count_takes_the_lower_front_index(self):
        """Two non-dominated entries at the max count: the more optimised one wins."""
        front = make_front([8, 8, 4, 0], half_max=4)
        self.assertEqual(sc.anchor_reasons(front)[0], "max")
        self.assertNotIn(1, sc.anchor_reasons(front))


class TestDrawOne(unittest.TestCase):
    def test_snaps_to_the_nearest_free_entry(self):
        """Counts 0-10; a draw is answered by the entry closest to it on the count axis."""
        n_mutations, front_indices, available, max_count = pool_of(make_front(list(range(10, -1, -1)), half_max=5))
        rng = np.random.default_rng(0)
        drawn = sc.draw_one(n_mutations, front_indices, available, max_count, rng)
        self.assertIsNotNone(drawn)
        # the drawn entry is gone from the pool, and no anchor was ever in it
        self.assertFalse(available[np.where(front_indices == drawn)[0][0]])
        self.assertNotIn(int(n_mutations[np.where(front_indices == drawn)[0][0]]), (0, 5, 10))

    def test_returns_none_once_exhausted(self):
        # counts 4, 2 and 0 are the anchors, so only 3 and 1 can be drawn
        n_mutations, front_indices, available, max_count = pool_of(make_front([4, 3, 2, 1, 0], half_max=2))
        rng = np.random.default_rng(0)
        self.assertIsNotNone(sc.draw_one(n_mutations, front_indices, available, max_count, rng))
        self.assertIsNotNone(sc.draw_one(n_mutations, front_indices, available, max_count, rng))
        self.assertIsNone(sc.draw_one(n_mutations, front_indices, available, max_count, rng))

    def test_never_returns_the_same_entry_twice(self):
        front = make_front(list(range(20, -1, -1)), half_max=10)
        n_mutations, front_indices, available, max_count = pool_of(front)
        rng = np.random.default_rng(3)
        drawn = [sc.draw_one(n_mutations, front_indices, available, max_count, rng) for _ in range(18)]
        self.assertEqual(len(set(drawn)), 18)
        self.assertNotIn(None, drawn)


class TestAxisUniformity(unittest.TestCase):
    def test_isolated_entry_beyond_a_gap_is_favoured(self):
        """The whole point of snapping: a lone entry above a gap must inherit the gap.

        Counts are 0-10 plus an isolated 45 and the max at 50. After the three anchors the
        pool holds ten entries, nine of them at counts 1-10 and one at 45. Sampling over
        entries would draw count 45 one time in ten; sampling the axis gives it everything
        above the midpoint 27.5, i.e. 21.5 of the range [1, 49] -> about 45%.
        """
        counts = [50, 45] + list(range(10, -1, -1))
        front = make_front(counts, half_max=5)
        isolated = 0
        for seed in range(300):
            n_mutations, front_indices, available, max_count = pool_of(front)
            drawn = sc.draw_one(n_mutations, front_indices, available, max_count, np.random.default_rng(seed))
            assert drawn is not None
            isolated += int(counts[drawn] == 45)
        self.assertGreater(isolated / 300, 0.30)
        self.assertLess(isolated / 300, 0.60)


class TestFillToTarget(unittest.TestCase):
    def test_hits_the_target_exactly(self):
        table = many_fronts(10, list(range(20, -1, -1)), half_max=10)
        selected = sc.select_candidates(table, target=100, seed=0)
        self.assertEqual(len(selected), 100)

    def test_partial_final_round_differs_by_one(self):
        """75 entries over 10 runs: 3 anchors each, 4 full rounds, then 5 of the 10 runs."""
        table = many_fronts(10, list(range(20, -1, -1)), half_max=10)
        per_run = sc.select_candidates(table, target=75, seed=0).groupby("run_dir").size()
        self.assertEqual(set(per_run), {7, 8})
        self.assertEqual(sorted(per_run).count(8), 5)
        self.assertEqual(per_run.sum(), 75)

    def test_exhausted_fronts_hand_their_budget_to_the_others(self):
        """One long front and three of length 3; the long one absorbs the rest of the target."""
        table = pd.concat(
            [make_front(list(range(30, -1, -1)), half_max=15, run_dir="config/long")]
            + [make_front([4, 2, 0], half_max=2, run_dir=f"config/short{i}") for i in range(3)]
        )
        per_run = sc.select_candidates(table, target=30, seed=0).groupby("run_dir").size()
        # every short front gives all 3 of its entries and nothing more
        self.assertEqual(list(per_run[["config/short0", "config/short1", "config/short2"]]), [3, 3, 3])
        self.assertEqual(per_run["config/long"], 21)

    def test_shortfall_when_every_front_is_exhausted_raises(self):
        table = many_fronts(3, [4, 2, 0], half_max=2)
        with self.assertRaisesRegex(ValueError, "short of the target"):
            sc.select_candidates(table, target=500, seed=0)

    def test_target_below_the_anchor_count_raises(self):
        table = many_fronts(5, list(range(10, -1, -1)), half_max=5)
        with self.assertRaisesRegex(ValueError, "below the 15 anchor"):
            sc.select_candidates(table, target=2, seed=0)

    def test_selection_is_reproducible(self):
        table = many_fronts(6, list(range(20, -1, -1)), half_max=10)
        first = sc.select_candidates(table, target=80, seed=7)
        second = sc.select_candidates(table, target=80, seed=7)
        pd.testing.assert_frame_equal(first, second)


class TestSelectCandidates(unittest.TestCase):
    def test_runs_whose_entries_contain_n_are_dropped(self):
        clean = make_front([4, 2, 0], half_max=2, run_dir="config/clean")
        with_n = make_front([4, 2, 0], half_max=2, run_dir="config/with_n", n_count=[3, 3, 3])
        selected = sc.select_candidates(pd.concat([clean, with_n]), target=3, seed=0)
        self.assertEqual(list(selected["run_dir"].unique()), ["config/clean"])

    def test_columns_and_key_uniqueness(self):
        selected = sc.select_candidates(many_fronts(3, list(range(10, -1, -1)), half_max=5), target=21, seed=0)
        self.assertEqual(list(selected.columns), ["run_dir", "front_index", "selection_reason"])
        self.assertFalse(selected.duplicated(subset=["run_dir", "front_index"]).any())

    def test_every_run_keeps_its_three_anchors(self):
        selected = sc.select_candidates(many_fronts(4, list(range(15, -1, -1)), half_max=7), target=40, seed=1)
        anchors = selected[selected["selection_reason"] != "random"]
        self.assertEqual(len(anchors), 12)
        self.assertEqual(anchors.groupby("run_dir").size().tolist(), [3, 3, 3, 3])


if __name__ == "__main__":
    unittest.main()
