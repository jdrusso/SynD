# WESTPA Hyperparameter Optimization with SynD Propagator: Trp-cage unfolding

This example demonstrates running WESTPA using SynD for dynamics with
- Coordinate augmentation
  After each propagation step, auxiliary coords needed for haMSM construction will be written to `west.h5`
- haMSM construction
  After a round of WE completes, an haMSM is constructed
- Binning/Allocation optimization + PCoord Extension
  After haMSM construction, the progress coordinate is extended with the MSM features, and 

The generator is a 10,500 state discrete Markov model from Trp-cage MD simulations, and the progress coordinate is RMSD from the initial unfolded state.

Arbitrary bin optimization + allocation optimization schemes can be defined, as shown in `west.cfg`.
These functions are passed the constructed haMSM (a `msm_we.msm_we.modelWE` object), and can additionally access
WESTPA internal state through `westpa.rc`.

To run the example:
1. Install SynD (i.e., clone the repo and then `pip install <path_to_synd>`)
2. `cd synd/examples/westpa`
3. `w_init --bstate 'basis,1,1871' --tstate 'target,10'`
4. `w_run`
