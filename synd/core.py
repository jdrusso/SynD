"""Functions for interacting with SynD models."""
import pickle
import numpy as np
from numpy.random import default_rng
try:
    import packaging
except ModuleNotFoundError:
    from pkg_resources import packaging


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

    model.numpy_version_greater = packaging.version.Version(np.__version__) >= packaging.version.Version('1.25.0')

    return model
