from __future__ import annotations  # Sets PEP563, necessary for autodoc type aliases

import numpy as np

from collections.abc import Iterable, Iterator
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from synd.models.base import SynDModel
from typing import Any, Callable, Optional, Set, Union


class MarkovGenerator(SynDModel):
    """A SynD model governed by a Markov chain.

    Parameters
    ----------
    transition_matrix : ArrayLike | sparse.csr_matrix
        A row stochastic matrix specifying the interstate transition
        probabilities.
    default_backmapper : Callable[[NDArray[int]], Any]
        A function that maps a discrete state trajectory (consisting of
        zero-based state indices) to a full-coordinate representation.
    seed : int, optional
        Seed for the random number generator.

    """
    def __init__(
            self,
            transition_matrix: Union[ArrayLike, sparse.spmatrix],
            default_backmapper: Callable[[NDArray[int]], Any],
            seed: Optional[int] = None,
    ):
        super().__init__(default_backmapper)
        self.transition_matrix = _ensure_transition_matrix(transition_matrix)
        self.rng = np.random.default_rng(seed=seed)
        self._preprocess_transition_matrix()
        self.logger.info(f'Created Markov generator with {self.n_states} states.')

    def _preprocess_transition_matrix(self):
        matrix = self.transition_matrix
        self.accessible_states = [row.indices for row in matrix]
        self.cumulative_probabilities = [np.cumsum(row.data) for row in matrix]

    @property
    def n_states(self) -> int:
        """int: Number of states of the underlying Markov chain."""
        return self.transition_matrix.shape[0]

    def generate_unmapped_trajectories(
            self,
            length: int,
            initial_states: Iterable[int],
            target: Optional[Set[int]] = None,
    ) -> Iterator[NDArray[int]]:
        """Generate trajectories of the underlying Markov chain.

        Parameters
        ----------
        length : int
        initial_states : Iterable[int]
        target : Set[int], optional

        Returns
        -------
        Iterator[NDArray[int]]

        """
        for initial_state in initial_states:
            yield self._generate_unmapped_trajectory(length, initial_state, target)

    def _generate_unmapped_trajectory(
            self,
            length: int,
            initial_state: int,
            target: Optional[Set[int]],
    ) -> NDArray[int]:
        traj = np.full(length, -1)
        traj[0] = initial_state
        random = self.rng.random(length - 1)
        for n in range(length - 1):
            start = traj[n]
            index = np.searchsorted(self.cumulative_probabilities[start], random[n])
            traj[n + 1] = self.accessible_states[start][index]
            if target is not None and traj[n + 1] in target:
                return traj[:n + 2]
        return traj

    def __getstate__(self):
        state = self.__dict__.copy()
        # Delete attributes that can be recomputed from the transition matrix:
        state.pop('accessible_states')
        state.pop('cumulative_probabilities')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._preprocess_transition_matrix()


def _ensure_transition_matrix(matrix):
    if not isinstance(matrix, sparse.csr_matrix):
        matrix = sparse.csr_matrix(matrix)
    if matrix.shape[1] != matrix.shape[0]:
        raise ValueError('transition matrix must be a square matrix')
    if not np.allclose([row.sum() for row in matrix], 1.0):
        raise ValueError('transition matrix must be row-normalized')
    return matrix
