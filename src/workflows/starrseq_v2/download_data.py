"""Download the FLOR-ID flowering-time gene table as a CSV."""

import json
import os

import pandas as pd
import csv
import html
import re
import urllib.request
from pathlib import Path
from typing import List

URL = "https://vps-bfdba49c.vps.ovh.net/databases/gene_list/flowering"
OUTPUT = Path(__file__).parent / "flowering_data" / "florid_flowering_genes.csv"

COLUMNS = [
    "name",
    "short_name",
    "keyword",
    "effect_on_flowering_time",
    "conditions_for_effect",
    "phenotype",
    "gene_id",
    "appears_in",
    "key_articles",
]

CELL_PATTERN = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
TAG_PATTERN = re.compile(r"<[^>]+>")


def cell_text(cell: str) -> str:
    """Plain text of one table cell, tags replaced by a space and whitespace collapsed."""
    return " ".join(html.unescape(TAG_PATTERN.sub(" ", cell)).split())


def parse_gene_table(page: str) -> List[List[str]]:
    """The nine text columns of every row of the FLOR-ID gene table."""
    body = page.split("<tbody>")[1].split("</tbody>")[0]
    return [[cell_text(cell) for cell in CELL_PATTERN.findall(row)] for row in body.split("<tr>")[1:]]


def download() -> None:
    with urllib.request.urlopen(URL) as response:
        rows = parse_gene_table(response.read().decode("utf-8"))
    os.makedirs(OUTPUT.parent, exist_ok=True)
    with open(OUTPUT, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    print(f"wrote {len(rows)} genes to {OUTPUT}")


def check():
    """Check that the downloaded CSV has the expected columns."""
    df = pd.read_csv(OUTPUT)
    print(df.describe())
    print(df.head())
    print(df.columns)
    assert list(df.columns) == COLUMNS, f"unexpected columns in {OUTPUT}"#


def create_input_json():
    """Create a JSON file with the gene IDs of all flowering-time genes."""
    df = pd.read_csv(OUTPUT)
    gene_ids = df["gene_id"].dropna().drop_duplicates().to_list()
    input_json_path = Path(__file__).parent / "flowering_data" / "ara_flowering_genes_florid.json"
    with open(input_json_path, "w") as f:
        json.dump(gene_ids, f, indent=2)
    print(f"wrote {len(gene_ids)} gene IDs to {input_json_path}")


def main():
    download()
    check()
    create_input_json()
    print("\n======================================\n")
    print("REMEMBER CITING https://academic.oup.com/nar/article/44/D1/D1167/2502597 !!!!!")
    print("\n======================================")



if __name__ == "__main__":
    main()
