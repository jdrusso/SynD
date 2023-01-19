"""Functions for interacting with SynD models."""
from synd.models.base import BaseSynDModel


def load_model(filename: str):
    """Load a SynD model from a file.

    Parameters
    ----------
    filename : str
        Path to SynD model file.

    Returns
    -------
    BaseSynDModel

    """
    return BaseSynDModel.load(filename)
