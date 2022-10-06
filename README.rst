====
SynD
====

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/jdrusso/SynD/HEAD?labpath=examples%2Fsynd_demo.ipynb
 

.. image:: https://github.com/jdrusso/synd/actions/workflows/build_docs.yml/badge.svg
  :target: https://github.com/jdrusso/synd/actions/workflows/build_docs.yml
 
Library for creating and using synthetic dynamics generators.


* Free software: MIT license

* Documentation: https://synd.readthedocs.io .


Installation
------------

SynD can be installed with
``pip install SynD``

Model Hosting
-------------
Serialized models can be stored/retrieved from an S3 object store using the
functionality in ``synd.hosted`` module.

To use hosted models, set the ``MINIO_ACCESSKEY`` and ``MINIO_SECRETKEY`` environment variables, or create an ``.env``
from ``.env-template``.

Usage
-----
Some example scripts are provided in ``examples/``.


Sample Scripts
==============
* ``examples/data/simple_model.py`` parameterizes a SynD generator of a simple 3-state Markov system.
* ``generate_markov_trajectory.py`` Creates a ``MarkovGenerator``, and generates some trajectories.
* ``create_model.py`` Creates a ``MarkovGenerator`` from ``simple_model.py``, and serializes it.
* ``use_saved_model.py`` Loads the ``MarkovGenerator`` created by ``create_model.py`` and generates some short trajectories.

* ``examples/westpa`` Example of simulating Trp-cage unfolding, using WESTPA with a 10,500 state discrete Markov SynD propagator.

Features
--------

* Trajectory generation from synthetic Markov models
* Model serialization/storage/retrieval via S3


Todo
----

* Add MD-specific Markov model, that backmaps MDTraj structures
* Make Markov propagation more efficient
* (?) Implement Markov models using deeptime
* Add progress bar to trajectory generation, using Rich

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
