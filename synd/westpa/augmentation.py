import westpa
import pickle
from .propagator import get_segment_parent_index


class SynDAugmentationDriver:
    """
    WESTPA plugin to automatically handle coordinate augmentation.

    After each iteration, appends coordinates to iter_XXX/auxdata/coord, for later usage with haMSM construction.
    """

    def __init__(self, sim_manager, plugin_config):
        westpa.rc.pstatus("Initializing coordinate augmentation plugin")

        if not sim_manager.work_manager.is_master:
            westpa.rc.pstatus("Not running on the master process, skipping")
            return

        self.data_manager = sim_manager.data_manager
        self.sim_manager = sim_manager

        self.plugin_config = plugin_config

        coord_map_path = plugin_config.get('coord_map')
        with open(coord_map_path, 'rb') as infile:
            self.coord_map = pickle.load(infile)

        # Big number is low priority -- this should run before anything else
        self.priority = plugin_config.get('priority', 1)

        sim_manager.register_callback(sim_manager.post_propagation, self.augment_coordinates, self.priority)

    def augment_coordinates(self):
        """
        After propagation completes in a WE iteration, this populates auxdata/coord with the coordinates.
        """

        iter_group = self.data_manager.get_iter_group(self.sim_manager.n_iter)
        n_iter = self.sim_manager.n_iter
        segments = self.data_manager.get_segments(
            n_iter=self.sim_manager.n_iter, load_pcoords=True
        )

        feature_shape = self.coord_map[0].squeeze().shape
        n_walkers = len(segments)

        # Create auxdata/coord for the current iteration
        self.data_manager.we_h5file.create_dataset(
            # f"{iter_path}/auxdata/coord",
            f"{iter_group.name}/auxdata/coord",
            shape=(n_walkers, 2, *feature_shape),
        )

        self.data_manager.flush_backing()

        auxcoord_dataset = self.data_manager.we_h5file[f"{iter_group.name}/auxdata/coord"]

        for i, segment in enumerate(segments):

            # If this segment doesn't have a state index, it was just created, so get it from its bstate
            if segment.parent_id < 0:

                istate = self.data_manager.get_segment_initial_states([segment])[0]
                bstate_id = istate.basis_state_id
                segment_state_index = int(self.sim_manager.next_iter_bstates[bstate_id].auxref)

            else:

                segment_state_index = self.data_manager.we_h5file[
                    f"{iter_group.name}/auxdata/state_indices"
                ][segment.seg_id][-1]

            parent_state_index = get_segment_parent_index(segment)

            auxcoord_dataset[segment.seg_id, 0] = self.coord_map[parent_state_index]
            auxcoord_dataset[segment.seg_id, 1] = self.coord_map[segment_state_index]