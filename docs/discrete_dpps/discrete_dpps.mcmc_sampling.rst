MCMC sampling
-------------

:cite:`AnGhRe16`, :cite:`LiJeSr16c` and :cite:`LiJeSr16d` derived variants of a Monte Carlo Markov Chain based on MH sampler having for stationnary distribution DPPs. 

Correlated moves

Basis exchange
~~~~~~~~~~~~~~

Pure (basis) exchange moves

.. math::

	B' \leftrightarrow B \setminus s \cup t

Add-Delete
~~~~~~~~~~

Pure addition/deletion moves

.. math::

	S' \leftrightarrow S \setminus s \quad \text{Delete}\\
	S' \leftrightarrow S \cup t \quad \text{Add}

Add-Exchange-Delete
~~~~~~~~~~~~~~~~~~~

Mix of exchange and add-delete moves

.. math::
	
	S' &\leftrightarrow S \setminus s \quad \text{Delete}\\
	S' &\leftrightarrow S \setminus s \cup t \quad \text{Exchange}\\
	S' &\leftrightarrow S \cup t \quad \text{Add}

Zonotope
~~~~~~~~

Embedding of a discrete MH sampler in a continuous domain :cite:`GaBaVa17`