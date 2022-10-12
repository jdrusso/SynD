from westpa.core.propagators import WESTPropagator
import numpy as np
import scipy.sparse as sparse
import westpa
import pickle
from westpa.core.states import InitialState
import mdtraj as md
from copy import deepcopy

import synd.core
from synd.models.discrete.markov import MarkovGenerator


def get_segment_index(segment):

    data_manager = westpa.rc.data_manager
    sim_manager = westpa.rc.sim_manager

    iter_group = data_manager.get_iter_group(sim_manager.n_iter)

    # If this segment doesn't have a state index, it was just created, so get it from its bstate
    if segment.parent_id < 0:

        segment_state_index = get_segment_ibstate_discrete_index(segment)

    else:

        # TODO: Avoid reading directly from the H5 file, if possible. However, it's not guaranteed that this data
        #   will exist anywhere else when this function is called.
        segment_state_index = data_manager.we_h5file[
            f"{iter_group.name}/auxdata/state_indices"
        ][segment.seg_id][-1]

    return int(segment_state_index)


def get_segment_parent_index(segment):
    """
    For a given segment, identify the discrete index of its parent
    """

    sim_manager = westpa.rc.get_sim_manager()

    # If the parent id is >= 0, then the parent was a segment, and we can get its index directly.
    #   Otherwise, we have to get it from the ibstate auxdata
    parent_was_ibstate = segment.parent_id < 0

    if not parent_was_ibstate:

        parent_map = sim_manager.we_driver._parent_map
        parent_state_index = parent_map[segment.parent_id].data["state_indices"][-1]

        # # TODO: Why does this happen..? DOES this happen any more?
        # if type(parent_state_index) is np.ndarray:
        #     parent_state_index = parent_state_index.item()

    # Otherwise, that means the segment was a bstate/istate
    else:

        parent_state_index = get_segment_ibstate_discrete_index(segment)

    return int(parent_state_index)


def get_segment_ibstate_discrete_index(segment):
    """
    Given an ibstate, returns the discrete index of the SynD state that generated it
    """
    sim_manager = westpa.rc.get_sim_manager()
    data_manager = westpa.rc.get_data_manager()

    istate = data_manager.get_segment_initial_states([segment])[0]

    if istate.istate_type is InitialState.ISTATE_TYPE_BASIS:
        bstate_id = -(segment.parent_id + 1)
        parent_state_index = sim_manager.current_iter_bstates[bstate_id].auxref

    elif istate.istate_type is InitialState.ISTATE_TYPE_GENERATED:
        bstate_id = istate.basis_state_id
        parent_state_index = sim_manager.current_iter_bstates[bstate_id].auxref

    elif istate.istate_type is InitialState.ISTATE_TYPE_START:
        parent_state_index = istate.basis_auxref

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
    """
    A WESTPA propagator for SynD models.
    """
    def __init__(self, rc: westpa.rc = None):
        """
        The keys loaded from WESTPA configuration are:

        - :code:`west.system.system_options.pcoord_len`: The number of steps to propagate

        EITHER

        - :code:`west.propagation.parameters.pcoord_map`: The path to either a pickled dictionary, mapping discrete states to progress coordinates, or to an arbitrary pickled callable that takes a discrete state and returns a progress coordinate.

        - :code:`west.propagation.parameters.transition_matrix`: The path to a transition matrix to construct the SynD propagator from.

        OR

        - :code:`west.propagation.parameters.synd_model`: The path to a saved SynD model.

        Parameters
        ----------
        rc :
            A :code:`westpa.rc` configuration object, containing :code:`west.propagation.parameters.transition_matrix`/:code:`pcoord_map`

        TODO
        ----
        Instead of creating the synD model from a transition matrix and states, just load in a SynD model.
        Then users can provide arbitrary models, discrete or not.
        However, may need to make some changes to generalize the latent space representation in the auxdata (won't be
        an integer any more).
        """

        super(SynMDPropagator, self).__init__(rc)

        rc_parameters = rc.config.get(['west', 'propagation', 'parameters'])
        self.topology = md.load(rc_parameters['topology'])

        if 'synd_model' in rc_parameters.keys():
            model_path = rc_parameters['synd_model']
            self.synd_model = synd.core.load_model(model_path)
        else:
            pcoord_map_path = rc_parameters['pcoord_map']
            with open(pcoord_map_path, 'rb') as inf:
                pcoord_map = pickle.load(inf)
            if type(pcoord_map) is dict:
                backmapper = pcoord_map.get
            else:
                backmapper = pcoord_map

            transition_matrix = rc_parameters['transition_matrix']
            try:
                self.transition_matrix = sparse.load_npz(transition_matrix)
            except ValueError:  # .npz file doesn't contain a sparse matrix
                with np.load(transition_matrix) as npzfile:
                    self.transition_matrix = npzfile[npzfile.files[0]]

            self.synd_model = MarkovGenerator(
                transition_matrix=self.transition_matrix,
                backmapper=backmapper,
                seed=None
            )

        # Our dynamics are propagated in the discrete space, which is recorded only in auxdata. After completing an
        #   iteration, we write the final discrete indices to the initial point auxdata of the next segments.
        # All discrete information is stored exclusively in auxdata, so that as far as the WE is concerned, it's all
        #   continuous.
        sim_manager = rc.get_sim_manager()
        sim_manager.register_callback(
            sim_manager.finalize_iteration, copy_segment_data, 1
        )

        # TODO: Would be nice to decouple this from pcoord len, so you can run dynamics at a higher resolution
        #  but only save every N
        n_steps = rc.config.get(['west', 'system', 'system_options', 'pcoord_len'])
        print(f"SynD propagator inferring {n_steps} steps per iteration from west.system.system_options.pcoord_len")

        # AKA the number of steps to take
        self.coord_len = n_steps
        self.coord_dtype = int

    def get_pcoord(self, state):
        """Get the progress coordinate of the given basis or initial state."""

        state_index = int(state.auxref)
        state.pcoord = self.synd_model.backmap(state_index)

    def gen_istate(self, basis_state, initial_state):

        basis_state_index = int(basis_state.auxref)
        initial_state.pcoord = self.synd_model.backmap(basis_state_index)

    def propagate(self, segments):

        # Populate the segment initial positions
        n_segs = len(segments)

        initial_points = np.empty(n_segs, dtype=self.coord_dtype)

        for iseg, segment in enumerate(segments):

            initial_points[iseg] = get_segment_parent_index(segment)

        new_trajectories = self.synd_model.generate_trajectory(
            initial_states=initial_points,
            n_steps=self.coord_len
        )

        for iseg, segment in enumerate(segments):
            segment.data["state_indices"] = new_trajectories[iseg, :]

            segment.pcoord = np.array([
                self.synd_model.backmap(x) for x in segment.data["state_indices"]
            ]).reshape(self.coord_len, -1)

            # For H5 plugin
            if westpa.rc.get_data_manager().store_h5:

                # TODO: how to handle restart data?
                #       I think we just don't for SynD, but let's silence that warning.
                segment.data['iterh5/restart'] = None

                full_coordinate_trajectory = np.array([
                    self.synd_model.backmap(x, 'full_coordinates')
                    for x in segment.data["state_indices"]
                ])

                # To mimic the behavior of a saved MD trajectory, we omit the first point.
                # I don't love this, but it's consistent with the OpenMM propagator.
                # TODO: Change this -- it's inconsistent with how pcoords are saved, and isn't quite right
                #   for the haMSM plugin.
                self.topology.xyz = full_coordinate_trajectory[1:]
                self.topology.time = np.arange(self.coord_len-1) + \
                                     (self.coord_len-1) * westpa.rc.get_sim_manager().n_iter

                # TODO: Avoid this copy! Probably super slow
                segment.data['iterh5/trajectory'] = deepcopy(self.topology)

            segment.status = segment.SEG_STATUS_COMPLETE

        return segments
