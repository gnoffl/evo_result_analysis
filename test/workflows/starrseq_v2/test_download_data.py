"""Unit tests for the FLOR-ID gene table download."""

import unittest

from src.workflows.starrseq_v2 import download_data

HTML = """
<table><thead><tr><th>Name</th></tr></thead>
<tbody>
<tr><td width="15%"><a href="http://host/details?gene=AT2G13540">ABA HYPERSENSITIVE 1</a></td>\
<td width="4%"><a href="http://host/details?gene=AT2G13540">ABH1, CBP80</a></td>\
<td width="4%">General</td>\
<td width="4%"><span class="fa fa-minus-square"></span> Negative</td>\
<td width="4%">SD and LD</td>\
<td width="20%"><b>Single mutant</b>:</br> <em>cbp80</em> is early\nflowering. &amp; so on</td>\
<td width="8%"><a href="http://host/details?gene=AT2G13540">AT2G13540</a></td>\
<td width="20%"><a href='http://host/n/a'>microRNA biosynthesis</a></br><a href='http://host/n/b'>FLC regulation</a></td>\
<td width="20%">Kuhn J M et al., 2007, Plant J.</td></tr>
<tr><td width="15%"><a href="http://host/details?gene=AT1G69120">APETALA1</a></td>\
<td width="4%"><a href="http://host/details?gene=AT1G69120">AP1</a></td>\
<td width="4%">Flower development</td>\
<td width="4%"></td>\
<td width="4%">None</td>\
<td width="20%">abnormal flower</td>\
<td width="8%"><a href="http://host/details?gene=AT1G69120">AT1G69120</a></td>\
<td width="20%"></td>\
<td width="20%"></td></tr>
</tbody></table>
"""


class TestCellText(unittest.TestCase):
    def test_strips_tags_unescapes_and_collapses_whitespace(self):
        cell = "<b>Single mutant</b>:</br> <em>cbp80</em> is early\nflowering. &amp; so on"

        text = download_data.cell_text(cell)

        self.assertEqual(text, "Single mutant : cbp80 is early flowering. & so on")

    def test_adjacent_tags_do_not_glue_words_together(self):
        cell = "<a href='x'>microRNA biosynthesis</a></br><a href='y'>FLC regulation</a>"

        text = download_data.cell_text(cell)

        self.assertEqual(text, "microRNA biosynthesis FLC regulation")


class TestParseGeneTable(unittest.TestCase):
    def test_returns_one_row_per_gene(self):
        rows = download_data.parse_gene_table(HTML)

        self.assertEqual(len(rows), 2)

    def test_keeps_all_nine_columns(self):
        rows = download_data.parse_gene_table(HTML)

        self.assertEqual(len(rows[0]), len(download_data.COLUMNS))

    def test_parses_a_full_row(self):
        rows = download_data.parse_gene_table(HTML)

        self.assertEqual(
            rows[0],
            [
                "ABA HYPERSENSITIVE 1",
                "ABH1, CBP80",
                "General",
                "Negative",
                "SD and LD",
                "Single mutant : cbp80 is early flowering. & so on",
                "AT2G13540",
                "microRNA biosynthesis FLC regulation",
                "Kuhn J M et al., 2007, Plant J.",
            ],
        )

    def test_empty_cells_become_empty_strings(self):
        rows = download_data.parse_gene_table(HTML)

        self.assertEqual([rows[1][3], rows[1][7], rows[1][8]], ["", "", ""])


if __name__ == "__main__":
    unittest.main()
