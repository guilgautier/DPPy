.. _exotic_dpps:

Exotic DPPs
###########

.. _carries_process:

Carries process
***************

Recording the carries appearing when computing the cumulative sum of a sequence independent random integers

.. code-block:: python

	b, N = 10, 100 # base, length of the sequence

	A = np.random.randint(0, b-1, N) # Draw random digits in [0, b-1]
	B = np.mod(np.cumsum(A), b) # Record first digit in cumulative sum 

	# Record the carries using the digit descent process
	X = np.zeros(N, dtype=bool)
	X[1:] = B[1:] < B[:-1] 

	carries = np.arange(0, N)[X] # in

	bernoullis = np.random.rand(N) < 0.5*(1-1/p)

.. plot:: plots/ex_plot_carries_process.py

..	seealso::

	:cite:`BoDiFu09`


.. _UST:

Uniform Spanning Trees
**********************

The Uniform measure on Spanning Trees (UST) of a directed connected graph corresponds to a projection DPP with kernel the transfer current matrix of the graph.
The later is actually the orthogonal projection matrix onto the row span of the vertex-edge incidence matrix.
In fact, one can discard any row of the vertex-edge incidence matrix say :math:`A` to compute :math:`\mathbf{K}=A^{\top}[AA^{\top}]^{-1}A`.

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
	for md in ("Wilson", "Aldous-Broder", "DPP_exact"):
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