====
SynD
====


Library for creating and using synthetic dynamic models.


* Free software: MIT license


Model Hosting
-------------
Serialized models can be stored/retrieved from an S3 object store using the
functionality in `synd.hosted`_ module.


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
