.. _exotic_dpps:

Exotic DPPs
###########

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
	    ust.sample(md); ust.plot()
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

.. todo::

	Add references Wilson, Aldous-Broder


.. _carries_process:

Carries process
***************

The sequence of carries appearing when computing the cumulative sum (in base :math:`b`) of a sequence of i.i.d. digits forms a DPP on :math:`\mathbb{N}` with non symmetric kernel.

.. plot:: plots/ex_plot_carries_process.py

..	seealso::

	:cite:`BoDiFu10`



.. _poissonized_plancherel_measure:

Poissonized Plancherel measure
******************************

The poissonized Plancherel measure is a measure on partitions :math:`\lambda=(\lambda_1 \geq \lambda_2 \geq \cdots \geq 0)\in \mathbb{N}^{\mathbb{N}^*}`.
Samples from this measure can be obtained by:

- Sampling :math:`N \sim \mathcal{P}(\theta)`
- Sampling a uniform permutation :math:`\sigma\in \mathfrak{S}_N`
- Computing the sorting tableau :math:`P` associated to the RSK (`Robinson-Schensted-Knuth correspondence <https://en.wikipedia.org/wiki/Robinson%E2%80%93Schensted%E2%80%93Knuth_correspondence>`_) applied to :math:`\sigma`
- Considering only the shape :math:`\lambda` of :math:`P`.

Finally, the point process formed by :math:`\{\lambda_i - i + \frac12\}_{i\geq 1}` is a DPP on :math:`\mathbb{Z}+\frac12`.

.. plot:: plots/ex_plot_poissonized_plancherel.py

.. seealso::

	- :cite:`Bor09` Section 6

.. _exotic_dpps_api:

API
***

.. important::

	Check out the Notebook `Exotic DPPs <https://github.com/guilgautier/DPPy/blob/master/notebooks/exotic_dpps.ipynb>`_ 

.. currentmodule:: exotic_dpps

.. autoclass:: UST
	:members:

.. autoclass:: CarriesProcess
	:members:

.. autoclass:: PoissonizedPlancherel
	:members: