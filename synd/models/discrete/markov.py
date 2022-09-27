from synd.models.discrete.discrete import DiscreteGenerator
import numpy as np
import numpy.typing as npt
from typing import Callable
from scipy import sparse
import pickle


class MarkovGenerator(DiscreteGenerator):

    def __init__(self, transition_matrix: npt.ArrayLike, backmapper: Callable[[int], npt.ArrayLike], seed: int = None):

        super().__init__()

        if sparse.issparse(transition_matrix):
            transition_matrix = transition_matrix.toarray()
        self.transition_matrix = transition_matrix

        self.n_states = self.transition_matrix.shape[0]
        self._backmapper = backmapper

        self.rng = np.random.default_rng(seed=seed)

        self.logger.info(f"Discrete Markov model created with {self.n_states} states successfully created")

    def backmap(self, discrete_index: int) -> npt.ArrayLike:

        return self._backmapper(discrete_index)

    def generate_trajectory(self, initial_distribution: npt.ArrayLike, n_steps: int) -> npt.ArrayLike:

        self.logger.debug(f"Propagating {initial_distribution} for {n_steps} steps...")

        trajectories = np.full(
            shape=(initial_distribution.shape[0], n_steps),
            dtype=int,
            fill_value=-1
        )

        n_trajectories = initial_distribution.shape[0]

        trajectories[:, 0] = initial_distribution

        for i in range(1, n_steps):

            previous_states = trajectories[:, i-1]
            probabilities = self.transition_matrix[previous_states]

            for j in range(n_trajectories):

                trajectories[j, i] = self.rng.choice(
                    a=np.arange(self.n_states),
                    p=probabilities[j]
                )

        return trajectories

    @staticmethod
    def validate_transition_matrix(transition_matrix: npt.ArrayLike):

        assert transition_matrix.ndim == 2, "Transition matrix is not 2-dimensional"

        assert np.isclose(transition_matrix.sum(axis=1), 1.0), "Transition matrix is not row-normalized"

    def __setstate__(self, state):
        """
        Make the transition matrix dense when unpickling
        """

        if sparse.issparse(state['transition_matrix']):
            state['transition_matrix'] = state['transition_matrix'].toarray()

        self.__dict__ = state

    def __getstate__(self):
        """
        When pickling, make the transition matrix sparse
        """

        self.__dict__['transition_matrix'] = sparse.csr_matrix(self.__dict__['transition_matrix'])

        return self.__dict__