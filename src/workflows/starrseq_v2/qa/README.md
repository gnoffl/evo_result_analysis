# STARR-seq v2 library — ordered sequences and metadata

The submitted sequences of all contributors were pooled, checked against the library
requirements, and deduplicated. Rows failing a check were excluded from the order, and
identical sequences are included only once. These files document the result.

| file | content |
| --- | --- |
| `metadata.csv` | every submitted sequence, with its verdict |
| `candidates.fa` | the unique sequences to actually be used in starrseq |

## `metadata.csv`

One row per submitted sequence, in pooled order. Nothing is deleted: sequences that failed a
check stay in the table and are marked as such. The first five columns come from the upload
template that every contributor fills in; the last three were added during pooling.

### Columns from the upload template

**`Number`** — the row's position in the pooled table, running 1..N without gaps.

> This is **renumbered during pooling** and no longer matches the numbering in the
> contributor's own sheet. Match rows back on `id`, never on `Number`.

**`id`** — the identifier of the sequence, unique across the whole pool; no two rows share
one.

Identifiers are chosen by the contributing author and follow no common scheme across the
library. Most encode design information — genome, coordinates, gene, variant — but the
conventions differ per contributor and are not interchangeable.
`path_to_config` is the field with consistent meaning across all contributors.

**`reference`** — the `id` of the unmutated sequence this row's design started from. Every
row points at one, and a reference row points at itself. This is the row to compare against
when computing an effect size; it is guaranteed to be present in the kept set (see
`reference_dropped` below).

**`product`** — the 170 bp sequence, uppercase, `ACGT` only.

**`path_to_config`** — the configuration file of the run that produced the sequence,
relative to the configs root (configs/ in sciebo), e.g. `configs_gernot/<config name>.json`. Every path was
verified to resolve to an existing file.

### Columns added during pooling

**`source_file`** — the submitted sheet the row came from, without its extension. Filter on
this to isolate one contributor's rows.

**`drop_reasons`** — empty for a sequence that passed every check, otherwise the check it
failed. Only three of the library checks rejected anything in this round; 5447 rows were dropped
in total, each for a single reason. Several reasons are `|`-separated where a row fails more
than one.

| reason | rows | meaning |
| --- | --- | --- |
| `restriction_site` | 4191 | `product` carries a BsaI site (`GGTCTC` or its reverse complement `GAGACC`), counting one that the cloning suffix would complete at the 3′ end |
| `reference_dropped` | 1148 | the row passed every check itself, but the row it references was dropped |
| `not_acgt` | 108 | `product` contains characters other than A, C, G, T |

BsaI sites were matched with the regular expression

```
GGTCT(C|$)|GAGAC(C|$)
```

which rejects the recognition site `GGTCTC` and its reverse complement `GAGACC` anywhere in
the sequence, and additionally `GGTCT` or `GAGAC` ending the sequence, where the adjoining
cloning suffix supplies the final `C` and so completes a site across the insertion border.

**`representative_id`** — the `id` of the row whose oligo carries this row's sequence; empty
for dropped rows.

Identical sequences dont need to be tested multiple times. Rows sharing a `product` therefore form a group, and only the
lowest-numbered of them is the group's representative and gets included in the fasta. A row with
`representative_id == id` is a representative and is on the order; any other kept row is a
duplicate of one, and is measured by its representative's oligo.

## `candidates.fa`

The sequences on the order: one record per representative, so exactly the rows of
`metadata.csv` where `drop_reasons` is empty and `representative_id == id`. The header is
the row's `id` and the sequence is its `product`. Dropped rows and duplicates are not
present.

## Using the table once results arrive

Measurements come back keyed by the identifiers in `candidates.fa`, which are fewer than the
kept rows in `metadata.csv`. The difference is the duplicates. Assuming results come back keyed on the ID in the fasta:

- **Join results on `representative_id`, not on `id`.** Joining on `id` silently drops every
  duplicate row.
- **Then look references up on `id`.** After that join every kept row carries a value,
  duplicates included, so a row's `reference` resolves with a single lookup on `id` in the
  merged table.
- **A duplicate group is one measurement, not several.** Rows sharing a representative are
  not independent replicates — the same oligo in the same well produced all of their values.
  Restrict to representatives before any statistical test, or n is inflated and the variance
  understated.
