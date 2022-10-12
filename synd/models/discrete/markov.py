from __future__ import annotations  # Sets PEP563, necessary for autodoc type aliases
from synd.models.discrete.discrete import DiscreteGenerator
import numpy as np
from numpy.typing import ArrayLike
from typing import Callable, Union
from scipy import sparse


class MarkovGenerator(DiscreteGenerator):
    """
    Generator using discrete Markov dynamics.
    """

    def __init__(self, transition_matrix: ArrayLike, backmapper: Callable[[int], ArrayLike], seed: int = None):
        """
        Parameters
        ----------
        transition_matrix
            A valid discrete transition matrix
        backmapper
            Callable mapping a discrete state index to a full-coordinate representation
        seed
            The seed for random number generator
        """

        super().__init__()

        if sparse.issparse(transition_matrix):
            transition_matrix = transition_matrix.toarray()

        self.transition_matrix = transition_matrix

        self.n_states = self.transition_matrix.shape[0]
        self._backmappers = {'default': backmapper}

        self.rng = np.random.default_rng(seed=seed)

        self.cumulative_probabilities = np.cumsum(self.transition_matrix, axis=1)

        self.logger.info(f"Discrete Markov model created with {self.n_states} states successfully created")

    def add_backmapper(self, backmapper: Callable[[int], ArrayLike], name: str):
        """
        Define a new backmapper.

        Parameters
        ----------
        backmapper :
            A callable defining a new backmapper.
        name :
            The name to associate with the new backmapper.
        """

        if name in self._backmappers:
            raise KeyError(f'A backmapper named {name} is already defined for this model.')

        self._backmappers[name] = backmapper

    def _vectorized_backmapper(self, mapper='default'):
        backmapper = self._backmappers.get(mapper)

        # TODO: This might be sketchy -- is 0 guaranteed to be mappable?
        returned_shape = backmapper(0).shape
        if len(returned_shape) == 1:
            returned_shape = f"({returned_shape[0]})"

        vectorized = np.vectorize(backmapper, signature=f"()->{returned_shape}")

        return vectorized

    def backmap(self,
                discrete_index: Union[int, ArrayLike],
                mapper: str = 'default'
                ) -> ArrayLike:
        """
        Return the full-coordinate representation of a discrete state.

        Parameters
        ----------
        discrete_index :
            Discrete state index
        mapper :
            Optional string to specify a backmapper for this model.

        Returns
        -------
        Array of coordinates
        """

        backmap = self._vectorized_backmapper(mapper)
        return backmap(discrete_index)

    def generate_trajectory(self, initial_states: ArrayLike, n_steps: int) -> ArrayLike:
        """

        Parameters
        ----------
        initial_states :
            Array of initial discrete states to propagate trajectories from
        n_steps :
            Number of steps forward to propagate from each initial state. Total trajectory length will be n_steps

        Returns
        -------

        """

        self.logger.debug(f"Propagating {initial_states} for {n_steps} steps...")

        trajectories = np.full(
            shape=(initial_states.shape[0], n_steps),
            dtype=int,
            fill_value=-1
        )

        n_trajectories = initial_states.shape[0]

        trajectories[:, 0] = initial_states

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
    def validate_transition_matrix(transition_matrix: ArrayLike):
        """
        Validate that a transition matrix is valid. Raises an :code:`AssertionError` if it is not.

        Parameters
        ----------
        transition_matrix
            A transition matrix

        """

        assert transition_matrix.ndim == 2, "Transition matrix is not 2-dimensional"

        assert np.isclose(transition_matrix.sum(axis=1), 1.0), "Transition matrix is not row-normalized"

    def __setstate__(self, state):
        """
        Makes the transition matrix dense when unpickling
        """

        if sparse.issparse(state['transition_matrix']):
            state['transition_matrix'] = state['transition_matrix'].toarray()

        # Recalculate the cumulative probabilities
        state['cumulative_probabilities'] = np.cumsum(state['transition_matrix'], axis=1)

        self.__dict__ = state

    def __getstate__(self):
        """
        When pickling, makes the transition matrix sparse
        """

        sparse_dict = self.__dict__.copy()

        # Store a sparse representation of this
        sparse_dict['transition_matrix'] = sparse.csr_matrix(self.__dict__['transition_matrix'])

        # Don't pickle this, because 1) it can be calculated and 2) it's large and not sparse
        sparse_dict.pop('cumulative_probabilities')

        return sparse_dict
