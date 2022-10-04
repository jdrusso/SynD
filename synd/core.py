"""Functions for interacting with SynD models."""
import pickle


def load_model(filename: str):
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

    return model
