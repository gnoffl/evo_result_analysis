"""The two species-specific deepCRE cross-validation model sets.

Each set contains one single-species-regression (SSR) model per held-out
chromosome of the training species: 5 for Arabidopsis, 12 for tomato. Both sets
share the same training configuration (``S0X0.75dC7K25f``), so a difference in
agreement with the STARR-seq measurement between the two sets is attributable to
the training species rather than to the training recipe.

Note that this configuration differs from the model used by the original
analyses (``Atha_S0X0.75dP7K25g``, ``dP`` rather than ``dC``), so results here
are not directly comparable to the numbers those produced.
"""
import os
from typing import List, NamedTuple

MODEL_ROOT = "/home/gernot/Code/PhD_Code/Evolution/models"
SPECIES_DIRS = {
    "Ara": os.path.join(MODEL_ROOT, "Ara_ssr"),
    "Slyc": os.path.join(MODEL_ROOT, "Slyc_ssr"),
}


class SpeciesModel(NamedTuple):
    """One cross-validation model of one training species.

    Attributes:
        species: ``"Ara"`` or ``"Slyc"``.
        held_out_chromosome: RefSeq accession of the chromosome held out during
            training, parsed from the file name; identifies the model within its
            species set.
        path: Absolute path to the ``.h5`` model file.
    """

    species: str
    held_out_chromosome: str
    path: str


def _parse_held_out_chromosome(file_name: str) -> str:
    """Extract the held-out chromosome accession from an SSR model file name.

    File names look like
    ``Atha_S0X0.75dC7K25f_NC_003070.9_ssr_train_models_250617_232757.h5``, where
    the accession is the two underscore-separated fields following the training
    configuration field.

    Args:
        file_name: Base name of the model file.

    Returns:
        The accession, e.g. ``"NC_003070.9"``.

    Raises:
        ValueError: If the name does not contain a parsable accession.
    """
    fields = file_name.split("_")
    for index, field in enumerate(fields):
        if field.startswith("NC") and index + 1 < len(fields):
            return f"{field}_{fields[index + 1]}"
    raise ValueError(f"Cannot parse held-out chromosome from {file_name!r}")


def list_species_models() -> List[SpeciesModel]:
    """List every cross-validation model of both species, Arabidopsis first.

    Returns:
        One :class:`SpeciesModel` per ``.h5`` file, sorted by species then by
        held-out chromosome accession, so the ordering is stable across runs.

    Raises:
        FileNotFoundError: If a species model directory is missing or empty.
    """
    models: List[SpeciesModel] = []
    for species, directory in SPECIES_DIRS.items():
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Model directory for {species} not found: {directory}"
            )
        file_names = sorted(f for f in os.listdir(directory) if f.endswith(".h5"))
        if not file_names:
            raise FileNotFoundError(f"No .h5 models in {directory}")
        models.extend(
            SpeciesModel(
                species=species,
                held_out_chromosome=_parse_held_out_chromosome(file_name),
                path=os.path.join(directory, file_name),
            )
            for file_name in file_names
        )
    return models
