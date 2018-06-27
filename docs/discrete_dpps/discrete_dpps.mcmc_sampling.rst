.. _disctrete_dpps_mcmc_sampling:

MCMC sampling
-------------

:cite:`AnGhRe16`, :cite:`LiJeSr16c` and :cite:`LiJeSr16d` derived variants of a Monte Carlo Markov Chain based on MH sampler having for stationnary distribution :math:`\operatorname{DPP}(\mathbf{L})` :eq:`marginal_proba`.

.. math::
	
	P[S' | S] 
		\propto \min \left( 1, \frac{\det \mathbf{L}_S'}{\det \mathbf{L}_S} \right)
		1_{|S' \Delta S|\leq 1}

.. hint::
	
	Because :math:`|S' \Delta S|\leq 1`, transitions are very local inducing correlated moves.

Basis exchange
~~~~~~~~~~~~~~

Pure (basis) exchange moves

.. math::

	S' \leftrightarrow S \setminus s \cup t

Add-Delete
~~~~~~~~~~

Pure addition/deletion moves

	- Delete :math:`S' \leftrightarrow S \setminus s`
	- Add :math:`S' \leftrightarrow S \cup t`

Add-Exchange-Delete
~~~~~~~~~~~~~~~~~~~

Mix of exchange and add-delete moves

	- Delete :math:`S' \leftrightarrow S \setminus s`
	- Exchange :math:`S' \leftrightarrow S \setminus s \cup t`
	- Add :math:`S' \leftrightarrow S \cup t`

Zonotope
~~~~~~~~

Embedding of a discrete MH sampler in a continuous domain :cite:`GaBaVa17` targeting a *projection* :math:`\operatorname{DPP}(\mathbf{K})` where

.. math::

	\mathbf{K} = \Phi^{\top} [\Phi \Phi^{\top}]^{-1} \Phi

with :math:`\Phi` the underlying :math:`r\times N` feature matrix under the assumption :math:`\operatorname{rank}(\Phi)=r`.