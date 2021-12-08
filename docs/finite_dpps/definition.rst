.. currentmodule:: dppy.finite.dpp

.. _finite_dpps_definition:

Definition
**********

A finite point process :math:`\mathcal{X}` on :math:`[N] \triangleq \{1,\dots,N\}` can be understood as a random subset.
It can either be defined by

- the inclusion probabilities, also called marginal probabilities or correlation functions

	.. math::
		:label: eq:inclusion_proba

		\mathbb{P}[S\subset \mathcal{X}], \text{ for } S\subset [N],

- or the joint probabilities, also called the likelihood

  	.. math::
		:label: eq:likelihood_proba

		\mathbb{P}[\mathcal{X}=S], \text{ for } S\subset [N].

.. hint::

	The *determinantal* feature of Determinantal Point Processes (DPPs) stems from the fact that many of their statistical properties can be expressed by means of determinants and minors of kernel matrices.

.. _finite_dpps_inclusion_probabilities:

Inclusion probabilities
=======================

We say that :math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{K})` with correlation kernel a complex matrix :math:`\mathbf{K}` if

.. math::
	:label: eq:inclusion_proba_DPP_K

	\mathbb{P}[S\subset \mathcal{X}] = \det \mathbf{K}_S,
	\quad \forall S\subset [N],

where :math:`\mathbf{K}_S = [\mathbf{K}_{ij}]_{i,j\in S}` i.e. the square submatrix of :math:`\mathbf{K}` obtained by keeping only rows and columns indexed by :math:`S`.

.. _finite_dpps_likelihood:

Likelihood
==========

We say that :math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{L})` with likelihood kernel a complex matrix :math:`\mathbf{L}` if

.. math::
	:label: eq:likelihood_DPP_L

	\mathbb{P}[\mathcal{X}=S] = \frac{\det \mathbf{L}_S}{\det [I+\mathbf{L}]},
	\quad \forall S\subset [N].

.. _finite_dpps_likelihood_from_correlation_kernel:

Alternative expression of the likelihood
----------------------------------------

Assuming :math:`\operatorname{DPP}(\mathbf{K})` is well defined, see, e.g., :ref:`finite_dpps_existence`, the likelihood :eq:`eq:likelihood_DPP_L` also reads

.. math::
	:label: eq:likelihood_from_correlation_kernel

	\mathbb{P}[\mathcal{X}=X]
		= (-1)^{|X^{c}|} \det [K - I^{X^{c}}]
		= \left\lvert \det [K - I^{X^{c}}] \right\rvert,

where :math:`I^{A}` denotes the indicator matrix of the subset A, i.e., :math:`[I^{A}]_{ij} = 1_{i\in A} 1_{j\in A}`.

In particular, :math:`\mathbb{P}[\mathcal{X}=\emptyset] = \det[I-K]`.

.. seealso::

	:cite:`KuTa12`, Section 3.5.

.. _finite_dpps_existence:

Existence
=========

Some necessary conditions on the leading principal minors of the respective kernels can be derived directly from the definitions above

- for :math:`\mathbf{K}` in :ref:`finite_dpps_inclusion_probabilities` must satisfy :math:`0 \leq \det \mathbf{K}_S \leq 1`,

- for :math:`\mathbf{L}` in :ref:`finite_dpps_likelihood` must satisfy :math:`\det \mathbf{L}_S \geq 0`,

for all subsets :math:`S\subset [N]`.

In the **real symmetric** case, `Sylvester's criterion <https://en.wikipedia.org/wiki/Sylvester%27s_criterion>`_ allows to convert these conditions on the leading principal minors to equivalent positive semi-definiteness conditions on the corresponding kernel.

In fact

- for :math:`\mathbf{K} = \mathbf{K}^{\dagger}`, :math:`\operatorname{DPP}(\mathbf{K})` exists if and only if

	.. math::
		:label: eq:suff_cond_K

		0_N \preceq \mathbf{K} \preceq I_N.

	.. code-block:: python

		import numpy as np
		import scipy.linalg as la
		from dppy.finite.dpp import FiniteDPP

		r, N = 4, 10

		A = np.random.randn(N, r)
		eig_vecs, _ = la.qr(A, mode='economic')
		eig_vals = np.random.rand(r)
		K = (eig_vecs * eig_vals).dot(eig_vecs.T)

		dpp = FiniteDPP('correlation', projection=False, hermitian=True, K_eig_dec=(eig_vals, eig_vecs))
		# or
		# dpp = FiniteDPP('correlation', projection=False, hermitian=True, K=K)

- for :math:`\mathbf{L} = \mathbf{L}^{\dagger}`, :math:`\operatorname{DPP}(\mathbf{L})` exists if and only if

	.. math::
		:label: eq:suff_cond_L

		\mathbf{L} \succeq 0_N.

	.. code-block:: python

		import numpy as np
		from dppy.finite.dpp import FiniteDPP

		r, N = 4, 10

		Phi = np.random.randn(N, r)
		L = np.dot(Phi.T, Phi)

		dpp = FiniteDPP('likelihood', projection=False, hermitian=True, L=L)
		# or
		# dpp = FiniteDPP('likelihood', projection=False, hermitian=True, L_gram_factor=Phi)

.. important::

	In the following, unless otherwise specified:

    - we work with hermitian kernels satisfying the sufficient conditions :eq:`eq:suff_cond_K` and :eq:`eq:suff_cond_L`,
    - :math:`\left(\lambda_{1}, \dots, \lambda_{N} \right)` denote the eigenvalues of :math:`\mathbf{K} = U \Lambda U^{\dagger}`,
    - :math:`\left(\gamma_{1}, \dots, \gamma_{N} \right)` denote the eigenvalues of :math:`\mathbf{L}=V \Gamma V^{\dagger}`.

.. _finite_dpps_definition_projection_dpps:

Projection DPPs
===============

:math:`\operatorname{DPP}(\mathbf{K})` defined by a *projection* correlation kernel, i.e., :math:`\mathbf{K}^2=\mathbf{K}` are called *projection* DPPs.

If in addition :math:`\mathbf{K}` is hermitian, i.e.,  :math:`\mathbf{K}^{\dagger}=\mathbf{K}`, then it is called an `orthogonal projection matrix <https://en.wikipedia.org/wiki/Projection_(linear_algebra)#Projection_matrix>`_.
In this case, the existence conditions :eq:`eq:suff_cond_K` are satisfied since :math:`\mathbf{K}` has eigenvalues equal to :math:`0` or :math:`1`.
The corresponding :math:`\operatorname{DPP}(\mathbf{K})` is called an orthogonal projection DPP are simply a projection DPP for brevity.

	.. code-block:: python

		import numpy as np
		import scipy.linalg as la
		from dppy.finite.dpp import FiniteDPP

		r, N = 4, 10

		A = np.random.randn(r, N)
		eig_vecs, _ = la.qr(A.T, mode='economic')
		eig_vals = np.ones(r)
		K = eig_vecs.dot(eig_vecs.T)  # = A^T (A A^T)^-1 A

		dpp = FiniteDPP('correlation', projection=True, hermitian=True, K_eig_dec=(eig_vals, eig_vecs))
		# or
		# dpp = FiniteDPP('correlation', projection=True, hermitian=True, A_zono=A)
		# dpp = FiniteDPP('correlation', projection=True, hermitian=True, K=K)

.. _finite_dpps_definition_k_dpps:

k-DPPs
======

A :math:`k\!\operatorname{-DPP}` can be defined as a :math:`\operatorname{DPP(\mathbf{L})}` :eq:`eq:likelihood_DPP_L` conditioned to a fixed sample size :math:`|\mathcal{X}|=k`, hence the notation :math:`k\!\operatorname{-DPP}(\mathbf{L})`.

:math:`k`-DPPs are naturally defined through their joint probabilities or likelihood

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
		= \sum_{\substack{S \subset [N]\\|S|=k}} \det \mathbf{L}_S.

.. note::

  	Obviously, one must take :math:`k = |S| = \leq \operatorname{rank}(\mathbf{L})` otherwise :math:`\det \mathbf{L}_S = 0`.

.. warning::

	k-DPPs are not DPPs in general.
	Viewed as a :math:`\operatorname{DPP}` conditioned to a fixed sample size :math:`|\mathcal{X}|=k`, the only case where they coincide is when the original DPP is a *projection* :math:`\operatorname{DPP}(\mathbf{K})`, and :math:`k=\operatorname{rank}(\mathbf{K})`, see :eq:`eq:likelihood_projection_K`.

.. seealso::

	- :ref:`Exact sampling of k-DPPs <finite_dpps_exact_sampling_k_dpps>`
	- :class:`FiniteDPP <FiniteDPP>`
	- :cite:`KuTa12` Section 2 for DPPs
	- :cite:`KuTa12` Section 5 for :math:`k`-DPPs
