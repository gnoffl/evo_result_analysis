"""Substring-based matching of requested gene IDs to full gene names.

Gene folders (and therefore the ``gene`` column of the scan CSV) carry full,
run-specific names such as ``"AT1G12345_control_run3"``.  Users, however, want
to restrict an analysis by passing only the bare gene ID (e.g. ``"AT1G12345"``).
This module resolves such IDs against the available full names by checking
whether each requested ID appears *anywhere* within a name, so callers never
have to reproduce the full folder name.
"""

from typing import Iterable, List, Tuple


def resolve_gene_ids(
    requested_ids: Iterable[str],
    available_names: Iterable[str],
) -> Tuple[List[str], List[str]]:
    """Resolve requested gene IDs to full gene names by substring match.

    A requested ID matches an available name when the ID occurs as a substring
    of that name.  This lets callers pass a bare gene ID (e.g. ``"AT1G12345"``)
    instead of the full gene-folder name it is embedded in (e.g.
    ``"AT1G12345_control_run3"``).  A single ID may match several names; all
    matches are returned.

    Args:
        requested_ids: Gene IDs (or ID substrings) to look for.
        available_names: Full gene names available in the data.

    Returns:
        Tuple of:
        * Sorted, de-duplicated list of matched full names.
        * Sorted, de-duplicated list of requested IDs that matched no name.

    Examples:
        >>> resolve_gene_ids(["AT1G12345"], ["AT1G12345_run3", "AT2G00010_run3"])
        (['AT1G12345_run3'], [])
        >>> resolve_gene_ids(["AT9G99999"], ["AT1G12345_run3"])
        ([], ['AT9G99999'])
    """
    available_list = list(available_names)
    matched_names: set[str] = set()
    unmatched_ids: set[str] = set()
    for gene_id in requested_ids:
        hits = [name for name in available_list if gene_id in name]
        if hits:
            matched_names.update(hits)
        else:
            unmatched_ids.add(gene_id)
    return sorted(matched_names), sorted(unmatched_ids)
