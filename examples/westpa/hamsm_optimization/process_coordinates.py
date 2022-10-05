import MDAnalysis as mda
from MDAnalysis.analysis import distances
import numpy as np

ref_file = '/home/jd/openeye/synmd_runs/2JOF.pdb'

def processCoordinates(self, coords):
    
    u_ref = mda.Universe(ref_file)
    u_check = mda.Universe(ref_file)
    
    dist_out = []
    
    u_check.load_new(coords)

    for frame in u_check.trajectory:

        dists = distances.dist(
            u_check.select_atoms('backbone'),
            u_ref.select_atoms('backbone')
        )[2]

        dist_out.append(dists)

    dist_out = np.array(dist_out)
    
    return dist_out
    
