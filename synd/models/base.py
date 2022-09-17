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

    def __init__(self):

        self.logger = logger

    def serialize(self):

        return pickle.dumps(self)

    def save(self, outfile: str):

        with open(outfile, 'wb') as of:
            pickle.dump(self, of)
