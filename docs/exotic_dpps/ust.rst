.. _UST:

Uniform Spanning Trees
**********************

The Uniform measure on Spanning Trees (UST) of a directed connected graph corresponds to a projection DPP with kernel the transfer current matrix of the graph.
The later is actually the orthogonal projection matrix onto the row span of the vertex-edge incidence matrix.
In fact, one can discard any row of the vertex-edge incidence matrix - note  :math:`A` the resulting matrix - to compute :math:`\mathbf{K}=A^{\top}[AA^{\top}]^{-1}A`.

.. code-block:: python


    from exotic_dpps import UST
    import networkx as nx

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

    - :py:class:`~dppy.exotic_dpps.UST`
    - Wilson algorithm :cite:`PrWi98`
    - Aldous-Broder :cite:`Ald90`
    - :cite:`Lyo02`
