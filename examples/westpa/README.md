# WESTPA with SynD Propagator: Trp-cage unfolding

This example demonstrates running WESTPA using SynD for dynamics.

The generator is a 10,500 state discrete Markov model from Trp-cage MD simulations, and the progress coordinate is RMSD from the initial unfolded state.

To run the example:
1. Install SynD (i.e., clone the repo and then `pip install <path_to_synd>`)
2. `cd synd/examples/westpa`
3. `w_init --bstate 'basis,1,1871' --tstate 'target,10'`
4. `w_run`
