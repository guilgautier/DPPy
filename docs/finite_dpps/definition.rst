.. currentmodule:: dppy.finite_dpps

.. _finite_dpps_definition:

Definition
**********

A finite point process :math:`\mathcal{X}` on :math:`[N] \triangleq \{1,\dots,N\}` can be understood as a random subset.
It is defined either via its:

- inclusion probabilities (also called correlation functions)

	.. math::

		\mathbb{P}[S\subset \mathcal{X}], \text{ for } S\subset [N],

- likelihood

  	.. math::

		\mathbb{P}[\mathcal{X}=S], \text{ for } S\subset [N].

.. hint::

	The *determinantal* feature of DPPs stems from the fact that such inclusion, resp. marginal probabilities are given by the principal minors of the corresponding correlation kernel :math:`\mathbf{K}` (resp. likelihood kernel :math:`\mathbf{L}`).

Inclusion probabilities
=======================

We say that :math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{K})` with correlation kernel a complex matrix :math:`\mathbf{K}` if

.. math::
	:label: eq:inclusion_proba_DPP_K

	\mathbb{P}[S\subset \mathcal{X}] = \det \mathbf{K}_S,
	\quad \forall S\subset [N],

where :math:`\mathbf{K}_S = [\mathbf{K}_{ij}]_{i,j\in S}` i.e. the square submatrix of :math:`\mathbf{K}` obtained by keeping only rows and columns indexed by :math:`S`.

Likelihood
==========

We say that :math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{L})` with likelihood kernel a complex matrix :math:`\mathbf{L}` if

.. math::
	:label: eq:likelihood_DPP_L

	\mathbb{P}[\mathcal{X}=S] = \frac{\det \mathbf{L}_S}{\det [I+\mathbf{L}]},
	\quad \forall S\subset [N].

Existence
=========

Some common sufficient conditions to guarantee existence are:

.. math::
	:label: eq:suff_cond_K

	\mathbf{K} = \mathbf{K}^{\dagger}
	\quad \text{and} \quad
	0_N \preceq \mathbf{K} \preceq I_N,

.. math::
	:label: eq:suff_cond_L

	\mathbf{L} = \mathbf{L}^{\dagger}
	\quad \text{and} \quad
	\mathbf{L} \succeq 0_N,

where the dagger :math:`\dagger` symbol means *conjugate transpose*.

.. note::

	In the following, unless otherwise specified:

	- we work under the sufficient conditions :eq:`eq:suff_cond_K` and :eq:`eq:suff_cond_K`,
	- :math:`\left(\lambda_{1}, \dots, \lambda_{N} \right)` denote the eigenvalues of :math:`\mathbf{K}`,
	- :math:`\left(\gamma_{1}, \dots, \gamma_{N} \right)` denote the eigenvalues of :math:`\mathbf{L}`.

.. :ref:`Fig. <correlation_kernel_plot>`

.. _correlation_kernel_plot:

.. plot:: plots/ex_plot_correlation_K_kernel.py

	Correlation :math:`\mathbf{K}` kernel

.. _finite_dpps_definition_projection_dpps:

Projection DPPs
===============

.. important::

	:math:`\operatorname{DPP}(\mathbf{K})` defined by an *orthogonal projection* correlation kernel :math:`\mathbf{K}` are called *projection* DPPs.

	Recall that `orthogonal projection matrices <https://en.wikipedia.org/wiki/Projection_(linear_algebra)#Projection_matrix>`_ are notably characterized by

	a. :math:`\mathbf{K}^2=\mathbf{K}` and :math:`\mathbf{K}^{\dagger}=\mathbf{K}`,
	b. or equivalently by :math:`\mathbf{K}=U U^{\dagger}` with :math:`U^{\dagger} U=I_r` where :math:`r=\operatorname{rank}(\mathbf{K})`.

	They are indeed valid kernels since they meet the above sufficient conditions: they are Hermitian with eigenvalues :math:`0` or :math:`1`.

	.. code-block:: python

		from numpy import ones
		from numpy.random import randn
		from scipy.linalg import qr
		from dppy.finite.dpp import FiniteDPP

		r, N = 4, 10

		eig_vals = ones(r)
		A = randn(r, N)
		eig_vecs, _ = qr(A.T, mode='economic')

		proj_DPP = FiniteDPP('correlation', projection=True,
		                     **{'K_eig_dec': (eig_vals, eig_vecs)})
		# or
		# proj_DPP = FiniteDPP('correlation', projection=True, **{'A_zono': A})
		# K = eig_vecs.dot(eig_vecs.T)
		# proj_DPP = FiniteDPP('correlation', projection=True, **{'K': K})

.. _finite_dpps_definition_k_dpps:

k-DPPs
======

A :math:`k\!\operatorname{-DPP}` can be defined as :math:`\operatorname{DPP(\mathbf{L})}` :eq:`eq:likelihood_DPP_L` conditioned to a fixed sample size :math:`|\mathcal{X}|=k`, we denote it :math:`k\!\operatorname{-DPP}(\mathbf{L})`.

It is naturally defined through its joint probabilities

.. math::
	:label: eq:likelihood_kDPP_L

	\mathbb{P}_{k\!\operatorname{-DPP}}[\mathcal{X}=S]
	= \frac{1}{e_k(L)} \det \mathbf{L}_S 1_{|S|=k},

.. \mathbb{P}_{k\!\operatorname{-DPP}}[\mathcal{X}=S]
.. 	= \frac{1}{e_k(L)} \det \mathbf{L}_S ~ 1_{|S|=k},

where the normalizing constant :math:`e_k(L)` corresponds to the `elementary symmetric polynomial <https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial>`_ of order :math:`k` evaluated in the eigenvalues of :math:`\mathbf{L}`,

.. math::

	e_k(\mathbf{L})
		\triangleq e_k(\gamma_1, \dots, \gamma_N)
		= \sum_{\substack{S \subset [N]\\|S|=k}} \prod_{s\in S} \gamma_{s}
		= \sum_{\substack{S \subset [N]\\|S|=k}} \det L_S.

.. note::

  	Obviously, one must take :math:`k \leq \operatorname{rank}(L)` otherwise :math:`\det \mathbf{L}_S = 0` for :math:`|S| = k > \operatorname{rank}(L)`.

.. warning::

	k-DPPs are not DPPs in general.
	Viewed as a :math:`\operatorname{DPP}` conditioned to a fixed sample size :math:`|\mathcal{X}|=k`, the only case where they coincide is when the original DPP is a *projection* :math:`\operatorname{DPP}(\mathbf{K})`, and :math:`k=\operatorname{rank}(\mathbf{K})`, see :eq:`eq:likelihood_projection_K`.

.. seealso::

	- :ref:`Exact sampling of k-DPPs <finite_dpps_exact_sampling_k_dpps>`
	- :class:`FiniteDPP <FiniteDPP>`
	- :cite:`KuTa12` Section 2 for DPPs
	- :cite:`KuTa12` Section 5 for :math:`k`-DPPs
