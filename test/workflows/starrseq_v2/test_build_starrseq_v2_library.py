import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from workflows.starrseq_v2 import build_starrseq_v2_library as sw

FRAME_LENGTH = 3020

# Real id shapes from both genomes. The Arabidopsis chromosome is a bare digit,
# the N. tabacum scaffold carries underscores and dots and recurs inside its own
# gene ids -- that asymmetry is what broke the header parsing once.
ARA_GENES = ["AT1G01010", "AT2G45660", "AT5G61850", "AT4G24540"]
ARA_CHROMOSOMES = ["1", "2", "5", "4"]
NTAB_GENES = [
    "Nicotiana_tabacum_NW_015787655.1_000001",
    "Nicotiana_tabacum_NW_015788138.1_000002",
    "Nicotiana_tabacum_NC_001879.2_000039",
    "Nicotiana_tabacum_NW_015792055.1_000004",
]
NTAB_CHROMOSOMES = ["NW_015787655.1", "NW_015788138.1", "NC_001879.2", "NW_015792055.1"]


def frame_header(chromosome: str, gene_id: str, start: int = 1918, end: int = 5863) -> str:
    """Build an extraction header: extract_string writes '<chrom>_<gene_id>_gene:<start>-<end>'."""
    return f"{chromosome}_{gene_id}_gene:{start}-{end}"


def ara_headers(count: int) -> list:
    """Extraction headers for the first ``count`` Arabidopsis genes."""
    return [frame_header(c, g) for c, g in list(zip(ARA_CHROMOSOMES, ARA_GENES))[:count]]


def ntab_headers(count: int) -> list:
    """Extraction headers for the first ``count`` N. tabacum genes."""
    return [frame_header(c, g) for c, g in list(zip(NTAB_CHROMOSOMES, NTAB_GENES))[:count]]


def ara_gene_series(count: int) -> list:
    """``count`` distinct Arabidopsis-shaped gene ids."""
    return [f"AT1G{index:05d}" for index in range(1, count + 1)]


def ntab_gene_series(count: int) -> list:
    """``count`` distinct N. tabacum-shaped gene ids."""
    return [
        f"Nicotiana_tabacum_NW_0157876{index % 100:02d}.1_{index:06d}"
        for index in range(1, count + 1)
    ]


def write_fasta(path: str, records: dict) -> None:
    """Write name -> sequence records to a FASTA file."""
    with open(path, "w") as fasta_file:
        for name, sequence in records.items():
            fasta_file.write(f">{name}\n{sequence}\n")


def read_fasta(path: str) -> dict:
    """Read a two-line-per-record FASTA into name -> sequence."""
    with open(path) as fasta_file:
        lines = fasta_file.read().split()
    return {lines[i][1:]: lines[i + 1] for i in range(0, len(lines), 2)}


def make_frame(marker: str) -> str:
    """Frame whose positions are distinguishable: a marker char every 10 bp."""
    return "".join(marker if i % 10 == 0 else "A" for i in range(FRAME_LENGTH))


class TestDrawRandomGenes(unittest.TestCase):
    def patch_find_genes(self, gene_ids: list) -> None:
        """Stand in for find_genes, which suffixes every id with the feature type."""
        genes = pd.DataFrame({"gene_id": [f"{gene}_gene" for gene in gene_ids]})
        patcher = patch.object(sw, "find_genes", return_value=genes)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_zero_genes_returns_empty(self):
        self.patch_find_genes(ARA_GENES)
        self.assertEqual(sw.draw_random_genes("ann", [], 0, 1), [])

    def test_excludes_and_strips_suffix_ara(self):
        self.patch_find_genes(ARA_GENES)
        drawn = sw.draw_random_genes("ann", ARA_GENES[:2], 2, 1)
        self.assertEqual(sorted(drawn), sorted(ARA_GENES[2:]))

    def test_excludes_and_strips_suffix_ntab(self):
        # ntab ids contain underscores, so the "_gene" suffix must be stripped
        # from the end rather than split on
        self.patch_find_genes(NTAB_GENES)
        drawn = sw.draw_random_genes("ann", NTAB_GENES[:2], 2, 1)
        self.assertEqual(sorted(drawn), sorted(NTAB_GENES[2:]))

    def test_deduplicates(self):
        self.patch_find_genes(ARA_GENES + [ARA_GENES[0]])
        self.assertEqual(sorted(sw.draw_random_genes("ann", [], 4, 1)), sorted(ARA_GENES))

    def test_deterministic_for_seed(self):
        self.patch_find_genes(NTAB_GENES)
        self.assertEqual(
            sw.draw_random_genes("ann", [], 2, 5), sw.draw_random_genes("ann", [], 2, 5)
        )

    def test_too_few_candidates_raises(self):
        self.patch_find_genes(ARA_GENES)
        with self.assertRaises(ValueError):
            sw.draw_random_genes("ann", ARA_GENES[:1], 4, 1)


class TestGeneIdsFromFasta(unittest.TestCase):
    def test_parses_ara_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "frames.fa")
            write_fasta(path, {header: "ACGT" for header in ara_headers(3)})
            self.assertEqual(sw.gene_ids_from_fasta(path), ARA_GENES[:3])

    def test_ntab_ids_are_not_recoverable(self):
        # Documents a known limit: with an underscored scaffold name the header
        # alone is ambiguous, so this helper is only valid on the Arabidopsis
        # GOF/LOF files it is called on. map_genes_to_frames handles the rest.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "frames.fa")
            write_fasta(path, {header: "ACGT" for header in ntab_headers(1)})
            self.assertNotEqual(sw.gene_ids_from_fasta(path), NTAB_GENES[:1])


class TestMapGenesToFrames(unittest.TestCase):
    def test_ara_chromosome_without_underscore(self):
        headers = ara_headers(2)
        self.assertEqual(
            sw.map_genes_to_frames(headers, ARA_GENES[:2]),
            dict(zip(ARA_GENES[:2], headers)),
        )

    def test_ntab_chromosome_with_underscores(self):
        # the scaffold name recurs inside the gene id, so the split point cannot
        # be guessed from the header alone
        headers = ntab_headers(4)
        self.assertEqual(
            sw.map_genes_to_frames(headers, NTAB_GENES),
            dict(zip(NTAB_GENES, headers)),
        )

    def test_mixed_species_fasta(self):
        headers = ara_headers(2) + ntab_headers(2)
        genes = ARA_GENES[:2] + NTAB_GENES[:2]
        self.assertEqual(sw.map_genes_to_frames(headers, genes), dict(zip(genes, headers)))

    def test_unknown_gene_absent(self):
        self.assertEqual(sw.map_genes_to_frames(ara_headers(1), NTAB_GENES[:1]), {})


class TestAddToMetadata(unittest.TestCase):
    def test_appends_rows(self):
        metadata = pd.DataFrame({
            "gene_id": [NTAB_GENES[0]], "species": ["ntab"], "reason": ["random"],
        })
        result = sw.add_to_metadata(metadata, ARA_GENES[:2], "arabidopsis", "GOF")
        self.assertEqual(result["gene_id"].tolist(), [NTAB_GENES[0]] + ARA_GENES[:2])
        self.assertEqual(result["reason"].tolist(), ["random", "GOF", "GOF"])
        self.assertEqual(result["species"].tolist(), ["ntab", "arabidopsis", "arabidopsis"])
        self.assertEqual(list(result.index), [0, 1, 2])


class TestDrawSubsetGenes(unittest.TestCase):
    def test_size_per_species(self):
        eligible = pd.DataFrame({
            "gene_id": ara_gene_series(100) + ntab_gene_series(10),
            "species": ["arabidopsis"] * 100 + ["ntab"] * 10,
        })
        drawn = sw.draw_subset_genes(eligible, fraction=0.2, min_genes=5, seed=1)
        self.assertEqual(sum(gene.startswith("AT") for gene in drawn), 20)
        # min_genes applies where the fraction yields fewer
        self.assertEqual(sum(gene.startswith("Nicotiana") for gene in drawn), 5)

    def test_capped_at_available_and_unique(self):
        genes = [NTAB_GENES[0], NTAB_GENES[0], NTAB_GENES[1]]
        eligible = pd.DataFrame({"gene_id": genes, "species": ["ntab"] * 3})
        with self.assertRaises(ValueError):
            sw.draw_subset_genes(eligible, fraction=0.5, min_genes=10, seed=1)


class TestSubsetSelection(unittest.TestCase):
    def make_metadata(self) -> pd.DataFrame:
        # 400 arabidopsis genes put the 20% fraction above the 50-gene floor,
        # 200 ntab genes keep it below, so both branches are covered
        ara_random = ara_gene_series(400)
        ntab_random = ntab_gene_series(200)
        v1_genes = ARA_GENES[:3]
        return pd.DataFrame({
            "gene_id": v1_genes + ara_random + ntab_random,
            "species": ["arabidopsis"] * (3 + 400) + ["ntab"] * 200,
            "reason": ["starrseq_v1"] * 3 + ["random"] * 600,
        })

    def test_arm_subset_excludes_v1(self):
        metadata = sw.select_arm_subset(self.make_metadata(), seed=1)
        self.assertFalse(metadata.loc[metadata["reason"] == "starrseq_v1", "arm_subset"].any())
        # 80 by fraction (arabidopsis) + 50 by floor (ntab)
        self.assertEqual(metadata["arm_subset"].sum(), 130)

    def test_arm_subset_flags_all_rows_of_gene(self):
        # this is not necessarily intended behaviour, but it makes no difference in the downstream analysis based on
        # the usage in write_run_script in Evolution
        metadata = self.make_metadata()
        # the first v1 gene also appears as a random gene; if drawn, both rows flag
        metadata.loc[0, "gene_id"] = metadata.loc[3, "gene_id"]
        metadata = sw.select_arm_subset(metadata, seed=1)
        flags = metadata.loc[metadata["gene_id"] == metadata.loc[3, "gene_id"], "arm_subset"]
        self.assertEqual(flags.nunique(), 1)

    def test_full_grid_within_arm(self):
        metadata = sw.select_arm_subset(self.make_metadata(), seed=1)
        metadata = sw.select_full_grid_subset(metadata, seed=1)
        self.assertTrue((metadata["arm_subset"] | ~metadata["full_grid_subset"]).all())
        # min_genes floor applies to both species' 40-gene arm subsets
        self.assertEqual(metadata["full_grid_subset"].sum(), 2 * sw.FULL_GRID_SUBSET_MIN_GENES)

    def test_species_drawn_separately(self):
        metadata = sw.select_arm_subset(self.make_metadata(), seed=1)
        drawn = metadata[metadata["arm_subset"]]
        self.assertEqual((drawn["species"] == "arabidopsis").sum(), 80)
        self.assertEqual((drawn["species"] == "ntab").sum(), 50)


class TestAddV1Windows(unittest.TestCase):
    def test_merges_only_v1_rows(self):
        metadata = pd.DataFrame({
            "gene_id": [ARA_GENES[0], ARA_GENES[0], ARA_GENES[1]],
            "species": ["arabidopsis"] * 3,
            "reason": ["starrseq_v1", "GOF", "starrseq_v1"],
        })
        mapping = pd.DataFrame({
            "gene": [ARA_GENES[0]] * 3,
            "overlap_start": [10, 10, 50],
            "overlap_end": [180, 180, 220],
            "fragment": ["frag_1", "frag_1", "frag_2"],
        })
        result = sw.add_v1_windows(metadata, mapping)
        v1_rows = result[
            (result["gene_id"] == ARA_GENES[0]) & (result["reason"] == "starrseq_v1")
        ]
        self.assertEqual(sorted(v1_rows["alternative_start"]), [10, 50])
        self.assertTrue(result.loc[result["reason"] == "GOF", "alternative_start"].isna().all())
        self.assertTrue(
            result.loc[result["gene_id"] == ARA_GENES[1], "alternative_start"].isna().all()
        )
        self.assertEqual(len(result), 4)


class TestAddRandomWindows(unittest.TestCase):
    def test_fills_missing_arm_windows(self):
        metadata = pd.DataFrame({
            "gene_id": [NTAB_GENES[0], NTAB_GENES[0], ARA_GENES[0], ARA_GENES[1]],
            "arm_subset": [True, True, True, False],
            "alternative_start": [float("nan"), float("nan"), 10.0, float("nan")],
            "alternative_end": [float("nan"), float("nan"), 180.0, float("nan")],
        })
        result = sw.add_random_windows(metadata, {NTAB_GENES[0]: make_frame("C")}, seed=1)
        ntab_starts = result.loc[result["gene_id"] == NTAB_GENES[0], "alternative_start"]
        self.assertEqual(ntab_starts.nunique(), 1)
        self.assertIn(ntab_starts.iloc[0], sw.RANDOM_WINDOW_STARTS)
        window_lengths = result["alternative_end"] - result["alternative_start"]
        self.assertEqual(window_lengths.iloc[0], sw.STARRSEQ_WINDOW_LENGTH)
        self.assertEqual(result.loc[2, "alternative_start"], 10)
        self.assertTrue(pd.isna(result.loc[3, "alternative_start"]))
        self.assertEqual(str(result["alternative_start"].dtype), "Int64")

    def test_windows_avoid_n_and_need_a_frame(self):
        # only the start 0 window [0, 170) is free of N
        frame = "A" * 170 + "N" * (FRAME_LENGTH - 170)
        metadata = pd.DataFrame({
            "gene_id": [NTAB_GENES[0], NTAB_GENES[1]],
            "arm_subset": [True, True],
            "alternative_start": [float("nan"), float("nan")],
            "alternative_end": [float("nan"), float("nan")],
        })
        result = sw.add_random_windows(metadata, {NTAB_GENES[0]: frame}, seed=1)
        self.assertEqual(result.loc[0, "alternative_start"], 0)
        self.assertTrue(pd.isna(result.loc[1, "alternative_start"]))

    def test_windows_avoid_central_region(self):
        self.assertTrue(all(
            start + sw.STARRSEQ_WINDOW_LENGTH <= 1500 or start >= 1520
            for start in sw.RANDOM_WINDOW_STARTS
        ))


class TestSummarizeSubsets(unittest.TestCase):
    def test_counts(self):
        gap_frame = make_frame("C")[:sw.INSERT_START] + "N" * sw.INSERT_LENGTH + make_frame("C")[sw.INSERT_END:]
        metadata = pd.DataFrame({
            # NTAB_GENES[0] has two rows and is counted once; the v1 row is ignored
            "gene_id": [NTAB_GENES[0], NTAB_GENES[0], NTAB_GENES[1], NTAB_GENES[2], ARA_GENES[0]],
            "species": ["ntab", "ntab", "ntab", "ntab", "arabidopsis"],
            "reason": ["flowering", "random", "random", "random", "starrseq_v1"],
            "arm_subset": [True, True, True, False, False],
            "full_grid_subset": [True, True, False, False, False],
        })
        frames = {NTAB_GENES[0]: make_frame("C"), NTAB_GENES[1]: gap_frame, ARA_GENES[0]: make_frame("C")}
        result = sw.summarize_subsets(metadata, frames).set_index("subset")
        self.assertEqual(list(result["species"]), ["ntab"] * 3)
        self.assertEqual(result.loc["baseline", ["genes", "with_frame", "n_free_correct_window"]].tolist(), [3, 2, 1])
        self.assertEqual(result.loc["arm_subset", ["genes", "with_frame", "n_free_correct_window"]].tolist(), [2, 2, 1])
        self.assertEqual(result.loc["full_grid_subset", ["genes", "with_frame", "n_free_correct_window"]].tolist(), [1, 1, 1])


class TestAssignDirections(unittest.TestCase):
    def test_fixed_and_balanced(self):
        random_genes = ara_gene_series(20)
        metadata = pd.DataFrame({
            "gene_id": [ARA_GENES[0], ARA_GENES[1]] + random_genes,
            "species": ["arabidopsis"] * 22,
            "reason": ["GOF", "LOF"] + ["random"] * 20,
            "arm_subset": [False, False] + [True] * 10 + [False] * 10,
            "full_grid_subset": [False, False] + [True] * 4 + [False] * 16,
        })
        result = sw.assign_directions(metadata, seed=1)
        self.assertEqual(result.loc[0, "direction"], "maximize")
        self.assertEqual(result.loc[1, "direction"], "minimize")
        for stratum in [slice(2, 6), slice(6, 12), slice(12, 22)]:
            counts = result.iloc[stratum]["direction"].value_counts()
            self.assertEqual(counts["maximize"], counts["minimize"])

    def test_balanced_within_each_species(self):
        metadata = pd.DataFrame({
            "gene_id": ara_gene_series(10) + ntab_gene_series(10),
            "species": ["arabidopsis"] * 10 + ["ntab"] * 10,
            "reason": ["random"] * 20,
            "arm_subset": [False] * 20,
            "full_grid_subset": [False] * 20,
        })
        result = sw.assign_directions(metadata, seed=1)
        for species in ["arabidopsis", "ntab"]:
            counts = result[result["species"] == species]["direction"].value_counts()
            self.assertEqual(counts["maximize"], 5)
            self.assertEqual(counts["minimize"], 5)

    def test_same_gene_same_direction(self):
        metadata = pd.DataFrame({
            "gene_id": [NTAB_GENES[0], NTAB_GENES[0], NTAB_GENES[1]],
            "species": ["ntab"] * 3,
            "reason": ["flowering", "random", "random"],
            "arm_subset": [False] * 3,
            "full_grid_subset": [False] * 3,
        })
        result = sw.assign_directions(metadata, seed=1)
        self.assertEqual(result.loc[0, "direction"], result.loc[1, "direction"])
        self.assertNotEqual(result.loc[0, "direction"], result.loc[2, "direction"])


class TestFlagFullLengthWindows(unittest.TestCase):
    def test_flags(self):
        metadata = pd.DataFrame({
            "gene_id": [ARA_GENES[0], NTAB_GENES[0], NTAB_GENES[1]],
            "alternative_start": pd.array([0, 10, pd.NA], dtype="Int64"),
            "alternative_end": pd.array([170, 100, pd.NA], dtype="Int64"),
        })
        result = sw.flag_full_length_windows(metadata)
        self.assertEqual(result["full_length_window"].tolist(), [True, False, False])


class TestSplicing(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.background_path = os.path.join(self.tmp, "background.fa")
        write_fasta(self.background_path, {"bg": "GGG" + "N" * sw.INSERT_LENGTH + "TTT"})
        patcher = patch.object(sw, "BACKGROUND_PATH", self.background_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.output_path = os.path.join(self.tmp, "out.fa")
        self.ara_header, self.ntab_header = ara_headers(1)[0], ntab_headers(1)[0]
        self.frames = {self.ara_header: make_frame("C"), self.ntab_header: make_frame("G")}
        self.frame_path = os.path.join(self.tmp, "frames.fa")
        write_fasta(self.frame_path, self.frames)

    def test_splice_correct_windows(self):
        sw.splice_correct_windows(self.frame_path, self.output_path)
        result = read_fasta(self.output_path)
        self.assertEqual(list(result), list(self.frames))
        for header, frame in self.frames.items():
            expected = "GGG" + frame[sw.INSERT_START:sw.INSERT_END] + "TTT"
            self.assertEqual(result[header], expected)

    def test_splice_off_target_windows_both_species(self):
        metadata = pd.DataFrame({
            "gene_id": [ARA_GENES[0], NTAB_GENES[0], NTAB_GENES[1]],
            "reason": ["starrseq_v1", "random", "random"],
            "alternative_start": pd.array([5, 1600, 0], dtype="Int64"),
            "alternative_end": pd.array([175, 1770, 170], dtype="Int64"),
            "full_length_window": [True, True, True],
        })
        sw.splice_off_target_windows(metadata, self.frame_path, self.output_path)
        result = read_fasta(self.output_path)
        # the third gene has no frame in the FASTA and is skipped
        self.assertEqual(
            list(result), [f"{self.ara_header}_off_5", f"{self.ntab_header}_off_1600"]
        )
        self.assertEqual(
            result[f"{self.ntab_header}_off_1600"],
            "GGG" + self.frames[self.ntab_header][1600:1770] + "TTT",
        )

    def test_splice_off_target_windows_filters_by_reason(self):
        metadata = pd.DataFrame({
            "gene_id": [NTAB_GENES[0], NTAB_GENES[0], ARA_GENES[0]],
            "reason": ["starrseq_v1", "GOF", "starrseq_v1"],
            "alternative_start": pd.array([5, 300, 400], dtype="Int64"),
            "alternative_end": pd.array([175, 470, 570], dtype="Int64"),
            "full_length_window": [True, True, True],
        })
        sw.splice_off_target_windows(
            metadata, self.frame_path, self.output_path, reason="starrseq_v1"
        )
        # the GOF row of the ntab gene is dropped, both starrseq_v1 rows are kept
        self.assertEqual(
            list(read_fasta(self.output_path)),
            [f"{self.ntab_header}_off_5", f"{self.ara_header}_off_400"],
        )

    def test_window_past_frame_end_raises(self):
        metadata = pd.DataFrame({
            "gene_id": [ARA_GENES[0]],
            "reason": ["random"],
            "alternative_start": pd.array([FRAME_LENGTH - 100], dtype="Int64"),
            "alternative_end": pd.array([FRAME_LENGTH + 70], dtype="Int64"),
            "full_length_window": [True],
        })
        with self.assertRaises(ValueError):
            sw.splice_off_target_windows(metadata, self.frame_path, self.output_path)

    def test_short_window_skipped(self):
        metadata = pd.DataFrame({
            "gene_id": [ARA_GENES[0], NTAB_GENES[0]],
            "reason": ["random", "random"],
            "alternative_start": pd.array([5, 300], dtype="Int64"),
            "alternative_end": pd.array([175, 400], dtype="Int64"),
            "full_length_window": [True, False],
        })
        sw.splice_off_target_windows(metadata, self.frame_path, self.output_path)
        self.assertEqual(list(read_fasta(self.output_path)), [f"{self.ara_header}_off_5"])


class ExtractionTestCase(unittest.TestCase):
    """Shared patching of the two side-effecting calls of the extract_* functions."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.drawn_genes = []
        self.draw_calls = []
        self.extract_calls = []
        for name, stub in [
            ("draw_random_genes", self.fake_draw_random_genes),
            ("extract_genes", self.fake_extract_genes),
        ]:
            patcher = patch.object(sw, name, stub)
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_draw_random_genes(self, annotation_path, exclude_genes, n_genes, seed):
        self.draw_calls.append((annotation_path, list(exclude_genes), n_genes, seed))
        return list(self.drawn_genes)

    def fake_extract_genes(self, genome_path, annotation_path, gene_ids, output_path,
                           vcf_paths=None):
        self.extract_calls.append(
            (genome_path, annotation_path, list(gene_ids), output_path, vcf_paths)
        )

    def write_json(self, name: str, content: list) -> str:
        """Write a gene-id list to a JSON file and return its path."""
        path = os.path.join(self.tmp, name)
        with open(path, "w") as json_file:
            json.dump(content, json_file)
        return path

    def patch_path(self, constant: str, value: str) -> None:
        patcher = patch.object(sw, constant, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def empty_metadata(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["gene_id", "species", "reason"])


class TestExtractNtab(ExtractionTestCase):
    def test_flowering_and_random_genes(self):
        flowering = ntab_gene_series(3)
        self.patch_path(
            "FLOWERING_GENES_NTAB_PATH", self.write_json("ntab_flowering.json", flowering)
        )
        self.drawn_genes = ntab_gene_series(10)[-2:]

        result = sw.extract_ntab(self.empty_metadata())

        self.assertEqual(result["gene_id"].tolist(), flowering + self.drawn_genes)
        self.assertEqual(result["species"].unique().tolist(), ["ntab"])
        self.assertEqual(result["reason"].tolist(), ["flowering"] * 3 + ["random"] * 2)

    def test_random_draw_fills_up_to_target(self):
        flowering = ntab_gene_series(3)
        self.patch_path(
            "FLOWERING_GENES_NTAB_PATH", self.write_json("ntab_flowering.json", flowering)
        )
        sw.extract_ntab(self.empty_metadata())

        annotation_path, exclude_genes, n_genes, _ = self.draw_calls[0]
        self.assertEqual(annotation_path, sw.NTAB_ANNOTATION)
        self.assertEqual(exclude_genes, flowering)
        self.assertEqual(n_genes, sw.TARGET_GENE_COUNT_NTAB - 3)

    def test_missing_flowering_file_draws_full_target(self):
        self.patch_path("FLOWERING_GENES_NTAB_PATH", os.path.join(self.tmp, "absent.json"))
        self.drawn_genes = ntab_gene_series(2)

        result = sw.extract_ntab(self.empty_metadata())

        self.assertEqual(self.draw_calls[0][2], sw.TARGET_GENE_COUNT_NTAB)
        self.assertEqual(result["reason"].tolist(), ["random"] * 2)

    def test_extracts_without_vcf(self):
        self.patch_path("FLOWERING_GENES_NTAB_PATH", self.write_json("f.json", ["gene_a"]))
        self.drawn_genes = ["gene_b"]

        sw.extract_ntab(self.empty_metadata())

        genome, annotation, gene_ids, output_path, vcf_paths = self.extract_calls[0]
        self.assertEqual((genome, annotation), (sw.NATB_GENOME, sw.NTAB_ANNOTATION))
        self.assertEqual(gene_ids, ["gene_a", "gene_b"])
        self.assertEqual(output_path, sw.NTAB_FLOWERING_FASTA)
        self.assertIsNone(vcf_paths)


class TestExtractAra(ExtractionTestCase):
    def setUp(self):
        super().setUp()
        self.patch_path(
            "FLOWERING_GENES_ARA_PATH", self.write_json("ara_flowering.json", ARA_GENES[:2])
        )
        self.gof_genes, self.lof_genes = [ARA_GENES[2]], [ARA_GENES[3]]
        patcher = patch.object(
            sw, "gene_ids_from_fasta",
            lambda path: self.gof_genes if path == sw.GOF_GENES_ARA_PATH else self.lof_genes,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_reasons_of_the_four_groups(self):
        self.drawn_genes = ara_gene_series(2)

        result = sw.extract_ara(self.empty_metadata())

        self.assertEqual(result["species"].unique().tolist(), ["arabidopsis"])
        self.assertEqual(
            result["reason"].tolist(),
            ["flowering"] * 2 + ["GOF", "LOF"] + ["random"] * 2,
        )
        self.assertEqual(result["gene_id"].tolist(), ARA_GENES + self.drawn_genes)

    def test_gene_in_two_groups_counts_once_towards_the_target(self):
        # the same gene is both a flowering gene and a GOF gene
        self.gof_genes = [ARA_GENES[0]]

        sw.extract_ara(self.empty_metadata())

        # 2 flowering + 1 LOF distinct genes, the GOF row duplicates a flowering one
        self.assertEqual(self.draw_calls[0][2], sw.TARGET_GENE_COUNT_ARA - 3)
        self.assertEqual(self.extract_calls[0][2], ARA_GENES[:2] + [ARA_GENES[3]])

    def test_prior_metadata_is_excluded_from_the_draw(self):
        prior = sw.add_to_metadata(
            self.empty_metadata(), ["prior_gene"], "arabidopsis", "starrseq_v1"
        )

        sw.extract_ara(prior)

        annotation_path, exclude_genes, n_genes, _ = self.draw_calls[0]
        self.assertEqual(annotation_path, sw.ARA_ANNOTATION)
        self.assertEqual(exclude_genes, ["prior_gene"] + ARA_GENES)
        # the prior gene is excluded from the draw but not counted against the target
        self.assertEqual(n_genes, sw.TARGET_GENE_COUNT_ARA - 4)

    def test_extracts_with_vcf(self):
        sw.extract_ara(self.empty_metadata())
        genome, annotation, _, output_path, vcf_paths = self.extract_calls[0]
        self.assertEqual((genome, annotation), (sw.ARA_GENOME, sw.ARA_ANNOTATION))
        self.assertEqual(output_path, sw.ARA_FLOWERING_GOF_FASTA)
        self.assertEqual(vcf_paths, [sw.ARA_VCF])


class TestExtractStarrseqV1(ExtractionTestCase):
    def patch_mappings(self, wrky: list, bhlh: list) -> None:
        for name, results in [("wrky_mapping_results", wrky), ("bhlh_mapping_results", bhlh)]:
            patcher = patch.object(sw, name, lambda results=results: results)
            patcher.start()
            self.addCleanup(patcher.stop)

    def mapping_row(self, gene_id: str, start: int) -> dict:
        return {"gene": gene_id, "overlap_start": start, "overlap_end": start + 170}

    def test_concatenates_both_families_and_deduplicates_genes(self):
        # the second WRKY fragment maps onto the same gene as the first
        self.patch_mappings(
            [self.mapping_row(ARA_GENES[0], 10), self.mapping_row(ARA_GENES[0], 400)],
            [self.mapping_row(ARA_GENES[1], 20)],
        )

        metadata, mapping = sw.extract_starrseq_v1(self.empty_metadata())

        self.assertEqual(len(mapping), 3)
        self.assertEqual(metadata["gene_id"].tolist(), ARA_GENES[:2])
        self.assertEqual(metadata["reason"].unique().tolist(), ["starrseq_v1"])
        self.assertEqual(metadata["species"].unique().tolist(), ["arabidopsis"])

    def test_extracts_with_vcf(self):
        self.patch_mappings([self.mapping_row(ARA_GENES[0], 10)], [])

        sw.extract_starrseq_v1(self.empty_metadata())

        genome, annotation, gene_ids, output_path, vcf_paths = self.extract_calls[0]
        self.assertEqual((genome, annotation), (sw.ARA_GENOME, sw.ARA_ANNOTATION))
        self.assertEqual(gene_ids, [ARA_GENES[0]])
        self.assertEqual(output_path, sw.STARRSEQ_V1_FASTA)
        self.assertEqual(vcf_paths, [sw.ARA_VCF])


if __name__ == "__main__":
    unittest.main()
