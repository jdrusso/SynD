import westpa
import pickle
from .propagator import get_segment_parent_index, get_segment_index


class SynDAugmentationDriver:
    """
    WESTPA plugin to automatically handle coordinate augmentation.

    After each iteration, appends coordinates to iter_XXX/auxdata/coord, for later usage with haMSM construction.

    Can be used by including the following entries in your west.cfg::

        west:
            plugins:
            - plugin: synd.westpa.augmentation_driver.SynDAugmentationDriver
                  coord_map: Pickled dictionary mapping discrete state indices to an array of coordinates
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
        segments = self.data_manager.get_segments(
            n_iter=self.sim_manager.n_iter, load_pcoords=True
        )

        feature_shape = self.coord_map[0].squeeze().shape
        n_walkers = len(segments)

        # Create auxdata/coord for the current iteration
        self.data_manager.we_h5file.require_dataset(
            f"{iter_group.name}/auxdata/coord",
            shape=(n_walkers, 2, *feature_shape),
            dtype=self.coord_map[0].dtype
        )

        self.data_manager.flush_backing()

        auxcoord_dataset = self.data_manager.we_h5file[f"{iter_group.name}/auxdata/coord"]

        for i, segment in enumerate(segments):

            segment_state_index = get_segment_index(segment)
            parent_state_index = get_segment_parent_index(segment)

            auxcoord_dataset[segment.seg_id, 0] = self.coord_map[parent_state_index]
            auxcoord_dataset[segment.seg_id, 1] = self.coord_map[segment_state_index]
