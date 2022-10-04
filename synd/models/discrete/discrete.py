from synd.models.base import BaseSynDModel
from abc import abstractmethod
import numpy.typing as npt


class DiscreteGenerator(BaseSynDModel):
    """
    Abstract base class for discrete SynD models.
    """

    def __init__(self):

        super().__init__()

    @abstractmethod
    def backmap(self, discrete_index: int):
        pass

    @abstractmethod
    def generate_trajectory(self, initial_distribution: npt.ArrayLike, n_steps: int):
        pass
