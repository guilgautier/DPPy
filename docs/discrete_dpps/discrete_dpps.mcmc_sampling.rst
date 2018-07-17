.. _discrete_dpps_mcmc_sampling:

MCMC sampling
*************

.. seealso::
	.. currentmodule:: discrete_dpps

	:func:`Discrete_DPP.sample_mcmc <Discrete_DPP.sample_mcmc>`

.. _discrete_dpps_mcmc_sampling_add_exchange_delete:

Add/exchange/delete
===================

:cite:`AnGhRe16`, :cite:`LiJeSr16a`, :cite:`LiJeSr16c` and :cite:`LiJeSr16d` derived variants of a Monte Carlo Markov Chain based on Metropolis Hastings (MH) sampler having for stationnary distribution :math:`\operatorname{DPP}(\mathbf{L})` :eq:`marginal_proba`.

The transition probability takes the following generic form

.. math::
	:label: AED_transition_probas
	
	\mathbb{P}[S' | S] 
		\propto \min \left( 1, \frac{\det \mathbf{L}_S'}{\det \mathbf{L}_S} \right)
		1_{|S' \Delta S|\leq 1}

.. hint::
	
	Because :math:`|S' \Delta S|\leq 1`, transitions are very local inducing correlated moves.

.. _discrete_dpps_mcmc_sampling_E:

Exchange
--------

Pure exchange moves

.. math::

	S' \leftrightarrow S \setminus s \cup t

.. _discrete_dpps_mcmc_sampling_AD:

Add-Delete
----------

Pure addition/deletion moves

	- Delete :math:`S' \leftrightarrow S \setminus s`
	- Add :math:`S' \leftrightarrow S \cup t`

.. _discrete_dpps_mcmc_sampling_AED:

Add-Exchange-Delete
-------------------

Mix of exchange and add-delete moves

	- Delete :math:`S' \leftrightarrow S \setminus s`
	- Exchange :math:`S' \leftrightarrow S \setminus s \cup t`
	- Add :math:`S' \leftrightarrow S \cup t`

.. seealso::

	- :cite:`AnGhRe16`, :cite:`LiJeSr16a`, :cite:`LiJeSr16c` and :cite:`LiJeSr16d`

.. _discrete_dpps_mcmc_sampling_zonotope:

Zonotope
========

:cite:`GaBaVa17` target a *projection* :math:`\operatorname{DPP}(\mathbf{K})` with

.. math::

	\mathbf{K} = \Phi^{\top} [\Phi \Phi^{\top}]^{-1} \Phi

where :math:`\Phi` is the underlying :math:`r\times N` feature matrix satisfying :math:`\operatorname{rank}(\Phi)=\operatorname{rank}(\mathbf{K})=r`.

In this setting the :ref:`discrete_dpps_nb_points` is almost surely equal to :math:`r` and we have

.. math::
	:label: zonotope_marginal

	\mathbb{P}[\mathcal{X}=S] 
		= \det \mathbf{K}_S 1_{|S|=r}
		= \frac{\det^2\Phi_{:S}}{\det\Phi \Phi^{\top}} 1_{|S|=r}
		= \frac{\operatorname{Vol}^2 \{\phi_s\}_{s\in S}}
					{\det\Phi \Phi^{\top}} 1_{|S|=r}

Then, the original discrete ground set is embedded in a continuous domain called a zonotope.

.. math::
	
	\mathcal{Z}(\Phi) = \Phi [0,1]^N

This zonotope is a polytope with a very singular feature; it admits a tiling made of non-degenerate parallelograms spanned by the feature vectors :math:`\{\phi_s\}_{s\in S}` i.e. :math:`\operatorname{Vol}^2 \{\phi_s\}_{s\in S} \neq 0`.
Any sample of :math:`\operatorname{DPP}(\mathbf{K})` is now represented by a tile, so that the corresponding MCMC jumps from one tile to another.

The underlying continuous structure of the zonotope is exploited through the hit-and-run kernel.
The associated Markov chain is used to move across the zonotope and visit the different tiles.
Finally, to recover the discrete DPP samples one needs to identify the tile in which the successive points lie, this is done by solving a linear program (LP).


.. hint::

	At the current point hit-and-run takes a uniform direction generating a line crossing the zonotope passing through the current state.
	This defines a segment onto which the next point is proposed uniformly at random and then accepted with a specified ratio.

	In this setting:

	- From the current state any point in the zonotope is accessible
	- The portions of the random segment intersecting the different tiles is *positively correlated* to the volume of the tile
	- To target distributions proportional to :math:`\operatorname{Vol}^{\alpha}` the acceptance ratio takes the form

		.. math::

			\left[\frac{\operatorname{Vol} \{\phi_s\}_{s\in S'}}
								{\operatorname{Vol} \{\phi_s\}_{s\in S}} \right]^{\alpha - 1}


.. caution::

	On the one hand, the :ref:`discrete_dpps_mcmc_sampling_zonotope` perspective on sampling *projection* DPPs yields a better exploration of the state space at the cost of solving 3 LPs at each step (1 for the identification of the tile and 2 very similar to find the endpoints of the segment).
	On the other hand, the :ref:`discrete_dpps_mcmc_sampling_add_exchange_delete` view allows to perform cheap but very local moves.

.. seealso::

	:cite:`GaBaVa17`