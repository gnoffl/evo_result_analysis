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
    gain_per_mutation: float = 0.01,
) -> pd.DataFrame:
    """One run's front table rows, in front order (index 0 is the most optimised entry).

    Args:
        n_mutations: Mutation count per entry; the maximum becomes max_mutations_reached.
        gain_per_mutation: Scales fitness_delta_vs_reference, which is count * this, so a
            larger value makes the whole run rank above a smaller one at every count.
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
        "fitness_delta_vs_reference": counts * gain_per_mutation,
    })


def many_fronts(n_runs: int, counts: List[int], half_max: int) -> pd.DataFrame:
    return pd.concat(make_front(counts, half_max, run_dir=f"config/run{i}") for i in range(n_runs))


def graded_fronts(n_runs: int, counts: List[int], half_max: int) -> pd.DataFrame:
    """Identical fronts whose fitness gain falls with the run number, so run0 is the strongest.

    The gain is linear in the mutation count, so gain per mutation is constant within a
    run and the runs rank the same way under both best-pick rankings.
    """
    return pd.concat(
        make_front(counts, half_max, run_dir=f"config/run{i}", gain_per_mutation=1.0 - i * 0.01)
        for i in range(n_runs)
    )


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
        self.assertEqual(sc.anchor_reasons(front), {0: ["max"], 1: ["half_max"], 2: ["reference"]})

    def test_one_entry_is_every_anchor(self):
        """A front of length 1 — nothing but the reference survived."""
        self.assertEqual(sc.anchor_reasons(make_front([0], half_max=0)), {0: ["reference", "half_max", "max"]})

    def test_half_max_coincides_with_max(self):
        front = make_front([1, 0], half_max=1)
        self.assertEqual(sc.anchor_reasons(front), {0: ["half_max", "max"], 1: ["reference"]})

    def test_tie_at_a_count_takes_the_lower_front_index(self):
        """Two non-dominated entries at the max count: the more optimised one wins."""
        front = make_front([8, 8, 4, 0], half_max=4)
        self.assertEqual(sc.anchor_reasons(front)[0], ["max"])
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
        with self.assertRaisesRegex(ValueError, "below the 15 entries"):
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


class TestSplitEvenly(unittest.TestCase):
    def test_exact_division(self):
        self.assertEqual(sc.split_evenly(10, 5), [2, 2, 2, 2, 2])

    def test_remainder_goes_to_the_earlier_parts(self):
        self.assertEqual(sc.split_evenly(11, 3), [4, 4, 3])

    def test_total_is_preserved(self):
        self.assertEqual(sum(sc.split_evenly(2800, 7)), 2800)


class TestRankByScore(unittest.TestCase):
    def test_orders_by_descending_score(self):
        front = make_front([5, 4, 3, 0], half_max=2)
        ranking = sc.rank_by_score(front, pd.Series(front["fitness_delta_vs_reference"]))
        self.assertEqual([index for _, index in ranking], [0, 1, 2, 3])

    def test_ties_go_to_the_lower_run_then_the_lower_index(self):
        table = pd.concat([
            make_front([1, 1], half_max=1, run_dir="config/b"),
            make_front([1, 1], half_max=1, run_dir="config/a"),
        ])
        self.assertEqual(
            sc.rank_by_score(table, pd.Series(table["fitness_delta_vs_reference"])),
            [("config/a", 0), ("config/a", 1), ("config/b", 0), ("config/b", 1)],
        )

    def test_full_function(self):
        table = pd.concat([
            make_front([1, 0], half_max=1, run_dir="config/b"),
            make_front([1, 0], half_max=1, run_dir="config/a"),
        ])
        self.assertEqual(
            sc.rank_by_score(table, pd.Series(table["fitness_delta_vs_reference"])),
            [("config/a", 0), ("config/b", 0), ("config/a", 1), ("config/b", 1)],
        )


class TestTakeBest(unittest.TestCase):
    def test_adds_exactly_the_quota(self):
        reasons = {"config/run": {}}
        shortfall = sc.take_best([("config/run", i) for i in range(5)], reasons, "best", 3)
        self.assertEqual(shortfall, 0)
        self.assertEqual(reasons["config/run"], {0: ["best"], 1: ["best"], 2: ["best"]})

    def test_an_already_selected_entry_gains_the_reason_without_spending_the_quota(self):
        reasons = {"config/run": {0: ["half_max"]}}
        ranking = [("config/run", 0), ("config/run", 1), ("config/run", 2)]
        self.assertEqual(sc.take_best(ranking, reasons, "best", 2), 0)
        self.assertEqual(reasons["config/run"], {0: ["half_max", "best"], 1: ["best"], 2: ["best"]})

    def test_a_short_ranking_reports_the_shortfall(self):
        self.assertEqual(sc.take_best([("config/run", 0)], {"config/run": {}}, "best", 3), 2)


class TestSelectBest(unittest.TestCase):
    def empty_reasons(self, table: pd.DataFrame):
        return {str(run_dir): {} for run_dir in table["run_dir"].unique()}

    def test_the_budget_is_split_evenly_over_the_rankings(self):
        table = graded_fronts(10, list(range(6, -1, -1)), half_max=3)
        reasons = self.empty_reasons(table)
        self.assertEqual(sc.select_best(table, reasons, 8, mutation_counts=(1, 2)), 0)
        labels = [label for run in reasons.values() for entry in run.values() for label in entry]
        self.assertEqual(labels.count("best_at_1"), 2)
        self.assertEqual(labels.count("best_at_2"), 2)
        self.assertEqual(labels.count("best_efficiency"), 4)

    def test_a_strong_run_wins_at_several_counts(self):
        """The deliberate ladder: run0 leads at every count, so it enters once per count."""
        table = graded_fronts(5, list(range(6, -1, -1)), half_max=3)
        reasons = self.empty_reasons(table)
        sc.select_best(table, reasons, 6, mutation_counts=(1, 2, 3))
        at_counts = {
            label for entry in reasons["config/run0"].values()
            for label in entry if label.startswith("best_at_")
        }
        self.assertEqual(at_counts, {"best_at_1", "best_at_2", "best_at_3"})
        # only 1 candidate per mutation count can be selected and the first run also has highest efficiency at every count, so no other run will be selected
        represented_runs = {run for run in reasons if reasons[run]}
        self.assertEqual(represented_runs, {"config/run0"})

    def test_references_are_never_an_efficiency_pick(self):
        """Index 2 is the 0-mutation entry; it has no gain per mutation to rank on."""
        table = graded_fronts(3, [4, 2, 0], half_max=2)
        reasons = self.empty_reasons(table)
        sc.select_best(table, reasons, 4, mutation_counts=(2,))
        for entries in reasons.values():
            self.assertNotIn(2, entries)

    def test_an_exhausted_ranking_reports_the_shortfall(self):
        """Both rankings get 10 and can offer only 2 new entries each."""
        table = graded_fronts(2, [4, 2, 0], half_max=2)
        self.assertEqual(sc.select_best(table, self.empty_reasons(table), 20, mutation_counts=(2,)), 16)


class TestBestPicksInSelection(unittest.TestCase):
    # half_max avoids BEST_MUTATION_COUNTS, so the anchors do not saturate a target count
    FRONT = list(range(20, -1, -1))

    def test_target_is_still_exact(self):
        table = graded_fronts(10, self.FRONT, half_max=9)
        self.assertEqual(len(sc.select_candidates(table, target=100, seed=0, target_best=20)), 100)

    def test_a_best_pick_is_never_drawn_again_by_the_uniform_fill(self):
        table = graded_fronts(10, self.FRONT, half_max=9)
        selected = sc.select_candidates(table, target=100, seed=0, target_best=20)
        self.assertFalse(selected.duplicated(subset=["run_dir", "front_index"]).any())
        reason = selected["selection_reason"]
        self.assertFalse((reason.str.contains("best_") & reason.str.contains("random")).any())

    def test_an_anchor_can_also_be_a_best_pick(self):
        """run0's half-max sits at count 2, a target count, so that entry carries both reasons."""
        table = pd.concat([
            make_front([6, 4, 2, 0], half_max=2, run_dir="config/run0", gain_per_mutation=1.0),
            make_front([6, 4, 2, 0], half_max=4, run_dir="config/run1", gain_per_mutation=0.5),
            make_front([6, 4, 2, 0], half_max=4, run_dir="config/run2", gain_per_mutation=0.4),
        ])
        selected = sc.select_candidates(table, target=12, seed=0, target_best=2, mutation_counts=(2,))
        reason_of = dict(zip(zip(selected["run_dir"], selected["front_index"]), selected["selection_reason"]))
        self.assertEqual(reason_of[("config/run0", 2)], "half_max|best_at_2")

    def test_a_target_count_that_is_an_anchor_in_every_run_raises(self):
        """Nothing new is left to take at count 2, even though the ranking is not empty."""
        table = graded_fronts(4, [4, 2, 0], half_max=2)
        with self.assertRaisesRegex(ValueError, "best picks short"):
            sc.select_candidates(table, target=12, seed=0, target_best=2, mutation_counts=(2,))

    def test_best_picks_are_reproducible(self):
        table = graded_fronts(8, self.FRONT, half_max=9)
        first = sc.select_candidates(table, target=90, seed=3, target_best=20)
        second = sc.select_candidates(table, target=90, seed=3, target_best=20)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
