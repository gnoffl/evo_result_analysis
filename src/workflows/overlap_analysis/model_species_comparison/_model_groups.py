"""The four deepCRE model groups compared against the STARR-seq measurement.

:mod:`_models` compares two single-species-regression (SSR) sets that share one
training recipe. This module widens the comparison to four groups that do *not*
share a recipe:

* ``MSR`` - the multi-species-regression set (``M0X0.75dP7K25f``), one model per
  held-out training species.
* ``Ara_ssr`` / ``Slyc_ssr`` - the two SSR sets of :mod:`_models`
  (``S0X0.75dC7K25f``), one model per held-out chromosome.
* ``Ntab_ssr`` - the tobacco SSR set, one model per held-out pseudo-chromosome.

Because recipe, training species and set size all differ between groups, a
difference in agreement with the measurement cannot be attributed to the
training species alone. Set size matters in particular for the group medians:
the median of 12 models suppresses model-specific noise more than the median of
5, which favours the larger groups independently of what they were trained on.

The group name is reused as the ``species`` field of :class:`SpeciesModel` and
the member name as ``held_out_chromosome``, so that the existing per-model
summary helpers in :mod:`_summary` apply unchanged. For the two SSR groups the
member names are the same accessions :mod:`_models` produces, which lets the
prediction cache written by :mod:`construct_by_species` be reused as is.
"""
import os
import re
from typing import List, NamedTuple

from workflows.overlap_analysis.model_species_comparison._models import (
    MODEL_ROOT,
    SpeciesModel,
)


class ModelGroup(NamedTuple):
    """One set of deepCRE models that is summarised as a single predictor.

    Attributes:
        name: Group name, used as the ``species`` field of its models and as the
            panel label.
        directory: Absolute path to the directory holding the group's ``.h5``
            files.
        member_pattern: Regular expression with exactly one capturing group,
            matched against each file name to name the model within its group.
    """

    name: str
    directory: str
    member_pattern: str


MODEL_GROUPS = [
    ModelGroup(
        name="MSR",
        directory=os.path.join(MODEL_ROOT, "MSR_models_M0X0.75"),
        member_pattern=r"K25f_([A-Z]_[a-z]+)_msr_train",
    ),
    ModelGroup(
        name="Ara",
        directory=os.path.join(MODEL_ROOT, "Ara_ssr"),
        member_pattern=r"_(NC_[0-9.]+)_ssr_train",
    ),
    ModelGroup(
        name="Slyc",
        directory=os.path.join(MODEL_ROOT, "Slyc_ssr"),
        member_pattern=r"_(NC_[0-9.]+)_ssr_train",
    ),
    ModelGroup(
        name="Ntab",
        directory=os.path.join(MODEL_ROOT, "ntab"),
        member_pattern=r"nicotiana_tabacum_([^_]+)_ssr_train",
    ),
]


def parse_member_name(file_name: str, member_pattern: str) -> str:
    """Extract a model's name within its group from the model file name.

    Args:
        file_name: Base name of the ``.h5`` model file.
        member_pattern: Regular expression with one capturing group.

    Returns:
        The captured name, e.g. ``"NC_003070.9"``, ``"A_thaliana"`` or
        ``"NtabPC1"``.

    Raises:
        ValueError: If the pattern does not match the file name.
    """
    match = re.search(member_pattern, file_name)
    if match is None:
        raise ValueError(
            f"Cannot parse a model name from {file_name!r} with {member_pattern!r}"
        )
    return match.group(1)


def list_group_models(group: ModelGroup) -> List[SpeciesModel]:
    """List every model of one group, sorted by member name.

    Args:
        group: The group to list.

    Returns:
        One :class:`SpeciesModel` per ``.h5`` file, with ``species`` set to the
        group name and ``held_out_chromosome`` to the member name.

    Raises:
        FileNotFoundError: If the group directory is missing or holds no models.
        ValueError: If a file name does not match the group's member pattern.
    """
    if not os.path.isdir(group.directory):
        raise FileNotFoundError(
            f"Model directory for group {group.name} not found: {group.directory}"
        )
    file_names = sorted(f for f in os.listdir(group.directory) if f.endswith(".h5"))
    if not file_names:
        raise FileNotFoundError(f"No .h5 models in {group.directory}")
    models = [
        SpeciesModel(
            species=group.name,
            held_out_chromosome=parse_member_name(file_name, group.member_pattern),
            path=os.path.join(group.directory, file_name),
        )
        for file_name in file_names
    ]
    return sorted(models, key=lambda model: model.held_out_chromosome)


def list_all_group_models() -> List[SpeciesModel]:
    """List the models of every group in :data:`MODEL_GROUPS`.

    Returns:
        The concatenated per-group listings, in group declaration order.

    Raises:
        FileNotFoundError: If any group directory is missing or empty.
    """
    models: List[SpeciesModel] = []
    for group in MODEL_GROUPS:
        models.extend(list_group_models(group))
    return models
