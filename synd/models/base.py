"""Abstract base SynD model class."""
from abc import ABC
import logging
from rich.logging import RichHandler
import pickle
try:
    from packaging.version import parse
except ModuleNotFoundError:
    from pkg_resources import parse_version as parse

logger = logging.getLogger(__name__)

rich_handler = RichHandler()
rich_handler.setLevel(logging.DEBUG)
rich_handler.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(rich_handler)


class BaseSynDModel(ABC):
    """
    Abstract base SynD model.
    """

    def __init__(self):

        self.logger = logger
        self.numpy_version_greater = parse(numpy.__version__) >= parse('1.25.0')

    def serialize(self):
        """
        Get the serialized representation of the SynD model.

        Returns
        -------
        Serialized representation of the model.
        """

        return pickle.dumps(self)

    def save(self, outfile: str):
        """
        Saves a SynD model to a file on disk.

        Parameters
        ----------
        outfile :
            Name of the file to save the model to.
        """

        with open(outfile, 'wb') as of:
            pickle.dump(self, of)
