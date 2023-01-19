"""Abstract base SynD model class."""
from abc import ABC
import logging
from rich.logging import RichHandler
import pickle

logger = logging.getLogger(__name__)

rich_handler = RichHandler()
rich_handler.setLevel(logging.DEBUG)
rich_handler.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(rich_handler)


class BaseSynDModel(ABC):
    """Abstract base SynD model."""

    def __init__(self):
        self.logger = logger

    def serialize(self):
        """Get the serialized representation of the SynD model.

        Returns
        -------
        bytes
            Serialized representation of the model.

        """
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data: bytes):
        """Deserialize a SynD model.

        Parameters
        ----------
        data : bytes
            A serialized SynD model.

        Returns
        -------
        BaseSynDModel

        """
        obj = pickle.loads(data)
        if not isinstance(obj, cls):
            raise TypeError(f'object must be an instance of {cls}')
        return obj

    def save(self, file: str):
        """Save a SynD model to a file on disk.

        Parameters
        ----------
        file : str
            Name of the file to save the model to.

        """
        with open(file, 'wb') as f:
            f.write(self.serialize())

    @classmethod
    def load(cls, file: str):
        """Load a SynD model from a file.

        Parameters
        ----------
        file : str
            Path to SynD model file.

        Returns
        -------
        BaseSynDModel

        """
        with open(file, 'rb') as f:
            return cls.deserialize(f.read())
