.. _finite_dpps_mcmc_sampling:

MCMC sampling
*************

.. _finite_dpps_mcmc_sampling_add_exchange_delete:

Add/exchange/delete
===================

:cite:`AnGhRe16`, :cite:`LiJeSr16c` and :cite:`LiJeSr16d` derived variants of a Metropolis sampler having for stationary distribution :math:`\operatorname{DPP}(\mathbf{L})` :eq:`likelihood`.
The proposal mechanism works as follows.

At state :math:`S\subset [N]`, propose :math:`S'` different from :math:`S` by at most 2 elements by picking

.. math::

  s \sim \mathcal{U}_{S}
  \quad \text{and} \quad
  t \sim \mathcal{U}_{[N]\setminus S}

Then perform

.. _finite_dpps_mcmc_sampling_E:

Exchange
--------

Pure exchange moves

.. math::

  S' \leftrightarrow S \setminus s \cup t

.. _finite_dpps_mcmc_sampling_AD:

Add-Delete
----------

Pure addition/deletion moves

  - Add :math:`S' \leftrightarrow S \cup t`
  - Delete :math:`S' \leftrightarrow S \setminus s`

.. _finite_dpps_mcmc_sampling_AED:

Add-Exchange-Delete
-------------------

Mix of exchange and add-delete moves

  - Delete :math:`S' \leftrightarrow S \setminus s`
  - Exchange :math:`S' \leftrightarrow S \setminus s \cup t`
  - Add :math:`S' \leftrightarrow S \cup t`

.. hint::

  Because moves are allowed between subsets having at most 2 different elements, transitions are very local inducing correlation, however *fast* mixing was proved.

.. testcode::

  import numpy as np
  from dppy.finite_dpps import FiniteDPP

  rng = np.random.RandomState(413121)

  r, N = 4, 10

  # Random feature vectors
  Phi = rng.randn(r, N)
  L = Phi.T.dot(Phi)
  DPP = FiniteDPP('likelihood', **{'L': L})

  DPP.sample_mcmc('AED', random_state=rng)
  print(DPP.list_of_samples)

.. testoutput::

  [[[0, 2, 3, 6], [0, 2, 3, 6], [0, 2, 3, 6], [0, 2, 3, 6], [0, 2, 3, 6], [0, 2, 3, 6], [0, 2, 6, 9], [0, 2, 6, 9], [2, 6, 9], [2, 6, 9]]]

.. seealso::

  .. currentmodule:: dppy.finite_dpps

  - :py:meth:`~FiniteDPP.sample_mcmc`
  - :cite:`AnGhRe16`, :cite:`LiJeSr16c` and :cite:`LiJeSr16d`


.. _finite_dpps_mcmc_sampling_zonotope:

Zonotope
========

:cite:`GaBaVa17` target a *projection* :math:`\operatorname{DPP}(\mathbf{K})` with

.. math::

  \mathbf{K} = \Phi^{\top} [\Phi \Phi^{\top}]^{-1} \Phi

where :math:`\Phi` is the underlying :math:`r\times N` feature matrix satisfying :math:`\operatorname{rank}(\Phi)=\operatorname{rank}(\mathbf{K})=r`.

In this setting the :ref:`finite_dpps_nb_points` is almost surely equal to :math:`r` and we have

.. math::
  :label: zonotope_marginal

  \mathbb{P}[\mathcal{X}=S]
  = \det \mathbf{K}_S 1_{|S|=r}
  = \frac{\det^2\Phi_{:S}}{\det\Phi \Phi^{\top}} 1_{|S|=r}
  = \frac{\operatorname{Vol}^2 \{\phi_s\}_{s\in S}}
      {\det\Phi \Phi^{\top}} 1_{|S|=r}

The original finite ground set is embedded into a continuous domain called a zonotope.
The hit-and-run procedure is used to move across this polytope and visit the different tiles.
To recover the finite DPP samples one needs to identify the tile in which the successive points lie, this is done by solving linear programs (LPs).

.. hint::

  Sampling from a *projection* DPP boils down to solving randomized LPs.

.. testcode::

  from numpy.random import RandomState
  from dppy.finite_dpps import FiniteDPP

  rng = RandomState(413121)

  r, N = 4, 10
  A = rng.randn(r, N)

  DPP = FiniteDPP('correlation', projection=True, **{'A_zono': A})

  DPP.sample_mcmc('zonotope', random_state=rng)
  print(DPP.list_of_samples)

.. testoutput::

  [array([[2, 4, 5, 7],
         [2, 4, 5, 7],
         [2, 4, 5, 7],
         [1, 4, 5, 7],
         [1, 4, 5, 7],
         [1, 4, 5, 7],
         [0, 4, 7, 8],
         [0, 2, 7, 9],
         [0, 2, 7, 9],
         [2, 4, 5, 7]])]

.. note::

  On the one hand, the :ref:`finite_dpps_mcmc_sampling_zonotope` perspective on sampling *projection* DPPs yields a better exploration of the state space.
  Using hit-and-run, moving to any other state is possible but at the cost of solving LPs at each step.
  On the other hand, the :ref:`finite_dpps_mcmc_sampling_add_exchange_delete` view allows to perform cheap but local moves.

.. seealso::

  .. currentmodule:: dppy.finite_dpps

  - :py:meth:`~FiniteDPP.sample_mcmc`
  - :cite:`GaBaVa17`

.. _finite_dpps_mcmc_sampling_k_dpps:

k-DPPs
======

To preserve the size :math:`k` of the sample, only :ref:`finite_dpps_mcmc_sampling_E` moves can be performed.

.. caution::

  :math:`k` must satisfy :math:`k \leq \operatorname{rank}(L)`

.. testcode::

  from numpy.random import RandomState
  from dppy.finite_dpps import FiniteDPP

  rng = RandomState(123)

  r, N = 5, 10

  # Random feature vectors
  Phi = rng.randn(r, N)
  L = Phi.T.dot(Phi)
  DPP = FiniteDPP('likelihood', **{'L': L})

  k = 3
  DPP.sample_mcmc_k_dpp(size=k, random_state=rng)
  print(DPP.list_of_samples)

.. testoutput::

  [[[7, 2, 5], [7, 2, 5], [7, 2, 9], [7, 8, 9], [7, 8, 9], [7, 8, 2], [7, 8, 2], [6, 8, 2], [1, 8, 2], [1, 8, 2]]]

.. seealso::

  .. currentmodule:: dppy.finite_dpps

  - :py:meth:`~FiniteDPP.sample_mcmc_k_dpp`
  - :cite:`KuTa12` Section 5
  - :cite:`LiJeSr16a`