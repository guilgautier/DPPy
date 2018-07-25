.. _discrete_dpps_mcmc_sampling:

MCMC sampling
*************

.. _discrete_dpps_mcmc_sampling_add_exchange_delete:

Add/exchange/delete
===================

:cite:`AnGhRe16`, :cite:`LiJeSr16a`, :cite:`LiJeSr16c` and :cite:`LiJeSr16d` derived variants of a Metropolis sampler having for stationary distribution :math:`\operatorname{DPP}(\mathbf{L})` :eq:`marginal_proba`.
The proposal mechanism works as follows.

At state :math:`S\subset [N]`, propose :math:`S'` different from :math:`S` by at most 2 elements by picking

.. math::

  s \sim \mathcal{U}_{S}
    \quad \text{and} \quad 
  t \sim \mathcal{U}_{[N]\setminus S}

Then perform

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

.. hint::
  
  Because moves are allowed between subsets having at most 2 different elements, transitions are very local inducing correlation.

.. testcode::

  from discrete_dpps import *
  np.random.seed(413121)

  r, N = 4, 10
  A = np.random.randn(r, N)
  L = A.T@A
  DPP = Discrete_DPP("marginal", **{"L":L})

  DPP.sample_mcmc("AED")
  print(DPP.list_of_samples)

.. testoutput::

  L (marginal) kernel available
  [[[0, 2, 3, 6], [0, 2, 3, 6], [0, 2, 3, 6], [0, 2, 3, 6], [0, 2, 3, 6], [0, 2, 3, 6], [0, 2, 6, 9], [0, 2, 6, 9], [2, 6, 9], [2, 6, 9]]]

.. seealso::

  .. currentmodule:: discrete_dpps

  - :func:`Discrete_DPP.sample_mcmc <Discrete_DPP.sample_mcmc>`
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

The original discrete ground set is embedded in a continuous domain called a zonotope.
Hit-and-run procedure is used to move across this polytope and visit the different tiles.
To recover the discrete DPP samples one needs to identify the tile in which the successive points lie, this is done by solving linear programs (LPs).

.. hint::

  Sampling from a *projection* DPP boils down to solving randomized LPs.

.. testcode::

  from discrete_dpps import *
  np.random.seed(1234)

  r, N = 4, 10
  A = np.random.randn(r, N)

  DPP = Discrete_DPP("inclusion", projection=True, **{"A_zono":A})

  DPP.sample_mcmc("zonotope")
  print(DPP.list_of_samples)

.. testoutput::

  [array([[2, 4, 7, 8],
         [3, 4, 7, 8],
         [0, 7, 8, 9],
         [3, 4, 6, 9],
         [3, 5, 7, 8],
         [3, 5, 7, 8],
         [1, 5, 8, 9],
         [0, 2, 4, 9],
         [4, 6, 8, 9],
         [4, 5, 8, 9]])]

.. note::

  On the one hand, the :ref:`discrete_dpps_mcmc_sampling_zonotope` perspective on sampling *projection* DPPs yields a better exploration of the state space.
  Using hit-and-run from a given given all other states become accessible but at the cost of solving LPs at each step.
  On the other hand, the :ref:`discrete_dpps_mcmc_sampling_add_exchange_delete` view allows to perform cheap but local moves.

.. seealso::

  .. currentmodule:: discrete_dpps

  - :func:`Discrete_DPP.sample_mcmc <Discrete_DPP.sample_mcmc>`
  - :cite:`GaBaVa17`
  