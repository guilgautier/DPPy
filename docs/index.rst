.. DPPy documentation master file, created by
	 sphinx-quickstart on Tue Jun  5 07:45:55 2018.
	 You can adapt this file completely to your liking, but it should at least
	 contain the root `toctree` directive.

.. |siren| replace:: 🚨
.. |fire| replace:: 🔥
.. |explosion| replace:: 💥
.. |palm_tree| replace:: 🌴
.. |tree| replace:: 🌿
.. |numbers| replace:: 🔢
.. |histogram| replace:: 📶

Welcome to DPPy's documentation!
################################

**Determinantal point processes** (DPPs) are specific probability distributions over clouds of points, which have been popular as models or computational tools across physics, probability, statistics, random matrices, and more recently machine learning. DPPs are often used to induce diversity or repulsiveness among the points of a sample.

**Sampling from DPPs** is more tractable than sampling generic point processes with interaction, but it remains a nontrivial matter and a research area of its own.

As a contraction of **DPPs and Python, DPPy** is an effort to gather:

- all **exact and approximate samplers** for :ref:`finite DPPs <finite_dpps>` |explosion|
- :ref:`full <full_matrix_models>` and :ref:`banded <banded_matrix_models>` (tri/quindiagonal) matrix models for :math:`\beta`-:ref:`Ensembles <beta_ensembles>` |fire|
- exact samplers for more :ref:`exotic_dpps` |palm_tree|

  * :ref:`uniform spanning trees <UST>` |tree|
  * :ref:`descent processes <stationary_1-dependent_process>` |numbers|
  * the :ref:`Poissonized Plancherel <poissonized_plancherel_measure>` |histogram|

The purpose of this **documentation** is to both provide a **quick survey of DPPs and relate each mathematical property with its implementation in DPPy**.
The documentation can thus be read in different ways:

	- if you read the sections in the order they appear, they will first take you through mathematical definitions and quick illustrations of how these definitions are encoded in DPPy.
	- for more a traditional library documentation please refer to the corresponding API sections documenting the methods of each object, along with pointers to the mathematical definitions if needed.
	- you can also directly jump to the Jupyter `notebooks <https://github.com/guilgautier/DPPy/tree/master/notebooks>`_, which showcase the use of some DPPy objects in more detail.

For another entry point to DPPy see the `companion paper <https://arxiv.org/abs/1809.07258>`_ :cite:`GaBaVa18`.

.. image:: _images/original_graph.png
   :width: 25%
.. image:: _images/kernel.png
   :width: 37.5%
.. image:: _images/sample_Wilson.png
   :width: 25%

As an :ref:`exotic <exotic_dpps>` example, the uniform measure on the spanning trees of a connected graph is a DPP!

Installation instructions
=========================

See `README <https://github.com/guilgautier/DPPy/blob/master/README.rst>`_

Documentation contents
======================

.. toctree::
  :maxdepth: 3

  finite_dpps/index
  continuous_dpps/index
  exotic_dpps/index
  bibliography/index
