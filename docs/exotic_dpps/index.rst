.. _exotic_dpps:

Exotic DPPs
###########

.. _carries_process:

Carries process
***************

.. plot:: plots/ex_plot_carries_process.py

..	seealso::

	:cite:`BoDiFu09`


.. _UST:

Uniform spanning trees
**********************

.. code-block:: python

	from exotic_dpps import *

	# Build graph
	g = nx.Graph()
	edges = [(0,2), (0,3), (1,2), (1,4), (2,3), (2,4), (3,4)]
	g.add_edges_from(edges)

	# Initialize UST object
	ust = UST(g)
	# Display original graph
	ust.plot_graph()
	# Display some samples
	for md in ("Aldous-Broder", "Wilson", "DPP_exact"):
	    ust.sample(md); ust.plot_sample()
	# Display underlyin kernel i.e. transfer current matrix
	ust.plot_kernel()

.. image:: ../_images/original_graph.png
   :width: 45%
.. image:: ../_images/kernel.png
   :width: 45%

.. image:: ../_images/sample_Wilson.png
   :width: 30%
.. image:: ../_images/sample_Aldous-Broder.png
   :width: 30%
.. image:: ../_images/sample_DPP_exact.png
   :width: 30%

.. image:: ../_images/ust_histo.png

.. seealso::

 :ref:`exotic_dpps_api`


.. _RSK:

RSK
***

.. _non_intersecting_RW:

Non intersecting random walks
*****************************

cf Dyson Brownian motion

.. _exotic_dpps_api:

API
***

.. important::

	Check out the Notebook `Exotic DPPs <https://github.com/guilgautier/DPPy/blob/master/notebooks/exotic_dpps.ipynb>`_ 

.. currentmodule:: exotic_dpps

.. autoclass:: UST
	:members: