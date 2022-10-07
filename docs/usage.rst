==================
Installation/Usage
==================


To install :code:`synd`:

.. code-block:: bash

    cd </path/to/synd>

    conda env create -f environment.yml

or, to install it into an existing conda environment:

.. code-block:: bash

    cd </path/to/synd>

    conda env update --name <your environment> --file environment.yml



Model building and trajectory generation
----------------------------------------

.. literalinclude :: ../examples/generate_markov_trajectory.py
   :language: python


Model building and saving
-------------------------

.. literalinclude :: ../examples/create_model.py
   :language: python


Using a saved model file
------------------------

.. literalinclude :: ../examples/use_saved_model.py
   :language: python


WESTPA Integration
-------------------

See the :code:`examples/westpa` directory for an example of WESTPA with a SynD propagator.