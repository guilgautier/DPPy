.. _finite_dpps_definition:

Definition
**********

A finite point process :math:`\mathcal{X}` on :math:`[N] \triangleq \{1,\dots,N\}` can be understood as a random subset.
It is defined either via its:

- inclusion probabilities (also called correlation functions)

	.. math::

		\mathbb{P}[S\subset \mathcal{X}], \text{ for } S\subset [N]

- marginal probabilities

	.. math::

		\mathbb{P}[\mathcal{X}=S], \text{ for } S\subset [N]

.. hint::

	The *determinantal* feature of DPPs stems from the fact that such inclusion, resp. marginal probabilities are given by the principal minors of the corresponding correlation kernel :math:`\mathbf{K}` (resp. likelihood kernel :math:`\mathbf{L}`).

Inclusion probabilities
=======================

We say that :math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{K})` with correlation kernel a complex matrix :math:`\mathbf{K}` if

	.. math::
		:label: inclusion_proba

		\mathbb{P}[S\subset \mathcal{X}] = \det \mathbf{K}_S,
		\quad \forall S\subset [N]

where :math:`\mathbf{K}_S = [\mathbf{K}_{ij}]_{i,j\in S}` i.e. the square submatrix of :math:`\mathbf{K}` obtained by keeping only rows and columns indexed by :math:`S`.

Marginal probabilities
======================

We say that :math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{L})` with likelihood kernel a complex matrix :math:`\mathbf{L}` if

	.. math::
		:label: marginal_proba

		\mathbb{P}[\mathcal{X}=S] = \frac{\det \mathbf{L}_S}{\det [I+\mathbf{L}]},
		\quad \forall S\subset [N]

Existence
=========

Some common sufficient conditions to guarantee existence are:

	.. math::
		:label: suff_cond_K

		\mathbf{K} = \mathbf{K}^{\dagger}
		\quad \text{and} \quad
		0_N \preceq \mathbf{K} \preceq I_N

	.. math::
		:label: suff_cond_L

		\mathbf{L} = \mathbf{L}^{\dagger}
		\quad \text{and} \quad
		\mathbf{L} \succeq 0_N

where the dagger :math:`\dagger` symbol means *conjugate transpose*.

.. note::

	In the following, unless otherwise specified, we work under these sufficient conditions.


.. plot:: plots/ex_plot_K_kernel.py
  :include-source:

.. important::

	DPPs defined by an *orthogonal projection* correlation kernel :math:`\mathbf{K}` are called *projection* DPPs.
	They are indeed valid kernels since they meet the above sufficient conditions: they are Hermitian with eigenvalues :math:`0` or :math:`1`.

	.. code-block:: python

		from numpy import ones
		from numpy.random import randn
		from scipy.linalg import qr
		from dppy.finite_dpps import FiniteDPP

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

:math:`\operatorname{k-DPPs}` can be defined as :math:`\operatorname{DPP(\mathbf{L})s}` conditioned to a fixed sample size :math:`|\mathcal{X}|=k`.
Thus, they are defined through the joint probabilities

.. math::

	\mathbb{P}_{\operatorname{k-DPP}}[\mathcal{X}=S]
		= \frac{1}{e_k(L)} \det \mathbf{L}_S ~~ 1_{|S|=k}

where :math:`e_k(L)` corresponds to the `elementary symmetric polymial <https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial>`_ of order :math:`k` evaluated in the eigenvalues of :math:`\mathbf{L}`,

.. math::

	e_k(\mathbf{L})
		\triangleq e_k(\delta_1, \dots, \delta_N)
		= \sum_{S\subset [N]: |S|=k} \prod_{n \in S} \delta_i
		= \sum_{S\subset [N]: |S|=k} \det L_S

.. caution::

  - :math:`k` must satisfy :math:`k \leq \operatorname{rank}(L)`

.. warning::

	k-DPPs are not DPPs in general.
	Viewed as :math:`\operatorname{DPPs}` conditioned to a fixed sample size :math:`|\mathcal{X}|=k`, the only case where they coincide is when the original DPP is a *projection* :math:`\operatorname{DPP}(\mathbf{K})`, and :math:`k=\operatorname{rank}(\mathbf{K})`, see :eq:`marginal_projection_K`.

.. seealso::

	.. currentmodule:: dppy.finite_dpps

	- :class:`FiniteDPP <FiniteDPP>`
	- :cite:`KuTa12` Section 2 for :math:`\operatorname{DPPs}`
	- :cite:`KuTa12` Section 5 for :math:`\operatorname{k-DPPs}`
