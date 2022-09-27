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

        self.cumulative_probabilities = np.cumsum(self.transition_matrix, axis=1)

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

        probabilities = self.rng.random(size=(n_trajectories, n_steps - 1))

        for istep in range(1, n_steps):

            current_states = trajectories[:, istep - 1]

            next_states = np.argmin(
                self.cumulative_probabilities[current_states].T
                < probabilities[:, istep - 1],
                axis=0,
            )

            trajectories[:, istep] = next_states

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

        # Recalculate the cumulative probabilities
        state['cumulative_probabilities'] = np.cumsum(state['transition_matrix'], axis=1)

        self.__dict__ = state

    def __getstate__(self):
        """
        When pickling, make the transition matrix sparse
        """

        sparse_dict = self.__dict__.copy()

        # Store a sparse representation of this
        sparse_dict['transition_matrix'] = sparse.csr_matrix(self.__dict__['transition_matrix'])

        # Don't pickle this, because 1) it can be calculated and 2) it's large and not sparse
        sparse_dict.pop('cumulative_probabilities')

        return sparse_dict