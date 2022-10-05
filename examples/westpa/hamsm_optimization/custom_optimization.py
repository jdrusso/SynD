from msm_we import optimization
import westpa
import numpy as np


def optimize_bins(hamsm):

    discrepancy, variance = optimization.solve_discrepancy(
        tmatrix=hamsm.Tmatrix,
        pi=hamsm.pSS,
        B=hamsm.indTargets
    )

    # Get the original bin target counts, i.e. where the counts are nonzero
    we_driver = westpa.rc.get_we_driver()
    n_active_bins = np.count_nonzero(we_driver.bin_target_counts)

    microstate_assignments = optimization.get_clustered_mfpt_bins(
        variance, discrepancy,
        hamsm.pSS,
        n_active_bins
    )

    return microstate_assignments


def optimize_allocation(hamsm):

    original_allocation = westpa.rc.get_we_driver().bin_target_counts

    return original_allocation
