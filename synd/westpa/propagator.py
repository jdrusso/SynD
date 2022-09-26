from westpa.core.propagators import WESTPropagator
from numpy.random import Generator, PCG64
import numpy as np
import scipy.sparse as sparse
import westpa
import pickle


def get_segment_parent_index(segment):
    """
    For a given segment, identify the discrete index of its parent
    """

    sim_manager = westpa.rc.get_sim_manager()
    data_manager = westpa.rc.get_data_manager()

    # If the parent id is >= 0, then the parent was a segment
    if segment.parent_id >= 0:
        parent_map = sim_manager.we_driver._parent_map

        try:
            parent_state_index = parent_map[segment.parent_id].data["state_indices"][-1]
        except KeyError as e:
            print(f"Parent map is currently {parent_map}")
            print(
                f"Parent map doesn't contain an entry for segment {segment} with parent ID {segment.parent_id}"
            )
            raise e

        # TODO: Why does this happen..?
        if type(parent_state_index) is np.ndarray:
            parent_state_index = parent_state_index.item()

    # Otherwise, that means the segment was a bstate/istate
    else:
        parent_istate = data_manager.get_segment_initial_states([segment])[0]

        parent_bstate_id = parent_istate.basis_state_id
        parent_state_index = int(sim_manager.next_iter_bstates[parent_bstate_id].auxref)

    return parent_state_index


def copy_segment_data():
    """
    In order to avoid any disk IO, we store each segment's discrete state as an attribute on the Segment object.
    Between iterations, we copy the parent's state index to the segment.

    Note that in this function, we're augmenting segments FOR THE NEXT ITERATION with the discrete state indices
    of segments that were run IN THE CURRENT ITERATION.
    That means, the "parent" segments here are the segments that just finished running, or the segments associated
     with cur_iter_* properties.
    In other words, this means that cur_iter_istates are NOT the istates for `next_iter_segments`!
    """

    sim_manager = westpa.rc.get_sim_manager()

    for segment in sim_manager.we_driver.next_iter_segments:
        segment.data["parent_final_state_index"] = get_segment_parent_index(
            segment
        )


class SynMDPropagator(WESTPropagator):
    def __init__(self, rc=None, transition_matrix=None, n_steps: int = None, pcoord_map: dict = None):
        # TODO: The goal of these arguments is to take, optionally, parameters provided through westpa.rc or
        #       provided individually as arguments.
        #       I don't think this is a particularly clean way of doing this, and I should give some more thought
        #       to how this handles edge cases. (What if arguments are specified in both? Which one gets priority?)

        super(SynMDPropagator, self).__init__(rc)

        rc_parameters = rc.config.get(['west', 'propagation', 'parameters'])

        if n_steps is None:
            # TODO: Would be nice to decouple this from pcoord len, so you can run dynamics at a higher resolution
            #  but only save every N
            n_steps = rc.config.get(['west', 'system', 'system_options', 'pcoord_len'])
            print(f"SynD propagator inferring {n_steps} steps per iteration "
                  f"from west.system.system_options.pcoord_len")

        if transition_matrix is None:
            transition_matrix = rc_parameters['transition_matrix']

        # Allow either a string, which will be a path, or a dict, which is the mapping itself
        if pcoord_map is None:
            pcoord_map_path = rc_parameters['pcoord_map']

            with open(pcoord_map_path, 'rb') as inf:
                pcoord_map = pickle.load(inf)

        sim_manager = rc.get_sim_manager()

        sim_manager.register_callback(
            sim_manager.finalize_iteration, copy_segment_data, 1
        )

        # AKA the number of steps to take
        self.coord_len = n_steps
        self.coord_dtype = int

        try:
            self.transition_matrix = sparse.load_npz(transition_matrix)
        except ValueError:  # .npz file doesn't contain a sparse matrix
            with np.load(transition_matrix) as npzfile:
                self.transition_matrix = npzfile[npzfile.files[0]]

        # We explicitly make the matrix dense. This isn't ideal, in general, but appears to be necessary to use the
        #   rows as a probability distribution to rng.choice. See the note in propagate()
        if sparse.issparse(self.transition_matrix):
            self.transition_matrix = self.transition_matrix.toarray()

        self.cumulative_probabilities = np.cumsum(self.transition_matrix, axis=1)

        self.rng = Generator(PCG64())

        self.pcoord_map = pcoord_map

    def get_pcoord(self, state):
        """Get the progress coordinate of the given basis or initial state."""

        state_index = int(state.auxref)
        state.pcoord = self.pcoord_map[state_index]

    def gen_istate(self, basis_state, initial_state):

        basis_state_index = int(basis_state.auxref)
        initial_state.pcoord = self.pcoord_map[basis_state_index]

    def propagate(self, segments):

        # Populate the segment initial positions
        n_segs = len(segments)

        coords = np.empty((n_segs, self.coord_len), dtype=self.coord_dtype)

        for iseg, segment in enumerate(segments):

            try:
                coords[iseg, 0] = segment.data["parent_final_state_index"]
            except KeyError:
                # If we're in the first iteration, no states have been written to auxdata yet.
                # If that's the case, we need to get this state index directly from the bstate that it
                #       was generated from.

                parent_id = segment.parent_id
                assert parent_id < 0, "Parent is not a bstate, but also doesn't have state indices written for SynD"

                bstates = westpa.rc.get_sim_manager().current_iter_bstates
                bstate = bstates[parent_id]
                coords[iseg, 0] = bstate.auxref

                # TODO: Can do this in one shot using T^{1..N} (not sure if more efficient)
        #       Precompute T^N (for N in 1..n_steps) at the beginning when the propagator is created, then reuse
        probabilities = self.rng.random(size=(n_segs, self.coord_len - 1))
        for istep in range(1, self.coord_len):
            current_states = coords[:, istep - 1]

            next_states = np.argmin(
                self.cumulative_probabilities[current_states].T
                < probabilities[:, istep - 1],
                axis=0,
            )

            coords[:, istep] = next_states

        for iseg, segment in enumerate(segments):
            segment.data["state_indices"] = coords[iseg, :]

            segment.pcoord = np.array(
                [self.pcoord_map[x] for x in segment.data["state_indices"]]
            ).reshape(self.coord_len, -1)

            segment.status = segment.SEG_STATUS_COMPLETE

        return segments
