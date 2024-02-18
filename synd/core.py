"""Functions for interacting with SynD models."""
import pickle
from numpy.random import default_rng

def load_model(filename: str, randomize: bool = True):
    """
    Load a SynD model from a file.

    Parameters
    ----------
    filename
        Path to SynD model file

    Returns
    -------
        SynD model
    """

    with open(filename, 'rb') as infile:
        model = pickle.load(infile)

    if randomize:
        model.rng = default_rng(seed=None)

    return model
