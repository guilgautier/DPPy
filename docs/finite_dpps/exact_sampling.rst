.. _finite_dpps_exact_sampling:

Exact sampling
**************

Given the correlation kernel :math:`\mathbf{K}` :eq:`eq:inclusion_proba` or likelihood kernel :math:`\mathbf{L}` :eq:`eq:likelihood` of a DPP, there exist three main types of exact sampling procedures:

1. The spectral method requires the eigendecomposition of the correlation kernel :math:`\mathbf{K}` or the likelihood kernel :math:`\mathbf{L}`). It is based on the fact that :ref:`generic DPPs are mixtures of projection DPPs <finite_dpps_mixture>` together with the application of the chain rule to sample projection DPPs. We present it in Section :ref:`finite_dpps_exact_sampling_spectral_method`.

2. a Cholesky-based procedure which requires :math:`\mathbf{K}` and also applies to non symmetric kernels. It boilds down to applying the chain rule on sets; where each item in turn is decided to be excluded or included in the sample. We present it in Section :ref:`finite_dpps_exact_sampling_cholesky_method`.

	- In the general case, each sample costs :math:`\mathcal{O}(N^3)`.

3. rencently, :cite:`DeCaVa19` have also proposed an alternative method to get exact samples: first sample an intermediate distribution and correct the bias by thinning the intermediate sample using a carefully designed DPP.

   - In certain regimes, this procedure may be more practical with an overall :math:`\mathcal{O}(N \text{poly}(\mathbb{E}\left[|\mathcal{X}|\right]) \text{polylog}(N))` cost.

.. note::

	- There exist specific samplers for special DPPs, like the ones presented in Section :ref:`exotic_dpps`.

.. _finite_dpps_exact_sampling_chain_rule:

Projection DPPs: the chain rule
-------------------------------

Recall that for a projection :math:`\operatorname{DPP}(\mathbf{K})` with :math:`r=\operatorname{rank}(\mathbf{K})=\operatorname{trace}(\mathbf{K})`, the likelihood :eq:`eq:likelihood_projection_K` of :math:`S=\{s_1, \dots, s_r\}`  reads

.. math::

	\mathbb{P}[\mathcal{X}=S]
	= \det \mathbf{K}_S.

.. caution::

	Next, we describe the Algorithm 18 of :cite:`HKPV06` which is **designed and valid only for projection DPPs**.

Using the invariance by permutation of the derterminant it is sufficient to apply the chain rule to sample :math:`(s_1, \dots, s_r)` with joint distribution

.. math::

	\mathbb{P}[s_1, \dots, s_r]
	= \frac{1}{r!} \mathbb{P}[\mathcal{X}=\{s_1, \dots, s_r\}]
	= \frac{1}{r!} \det \mathbf{K}_S,

and forget about the sequential feature of the chain rule to get a valid sample :math:`\{s_1, \dots, s_r\} \sim \operatorname{DPP}(\mathbf{K})`.

Considering :math:`S=\{s_1, \dots, s_r\}` such that :math:`\mathbb{P}[\mathcal{X}=S] = \det \mathbf{K}_S > 0`, the following generic formulation of the chain rule

.. math::
	:label: eq:chain_rule_genericformulation

	\mathbb{P}[s_1, \dots, s_r]
	= \mathbb{P}[s_1]
		\prod_{i=2}^{r}
			\mathbb{P}[s_{i} | s_{1:i-1}]

can be expressed as a telescopic ratio of determinants

.. math::
	:label: eq:chain_rule_K

	\mathbb{P}[s_1, \dots, s_r]
	= \dfrac{\mathbf{K}_{s_1,s_1}}{r}
		\prod_{i=2}^{r}
			\dfrac{1}{r-(i-1)}
		\frac{\det \mathbf{K}_{\{s_{1:i}\}}}
			 {\det \mathbf{K}_{\{s_{1:i-1}\}}}.

Using `Woodbury's formula <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_ the ratios of determinants in :eq:`eq:chain_rule_K` can be expanded into

.. math::
	:label: eq:chain_rule_schur

	\mathbb{P}[s_1, \dots, s_r]
	= \dfrac{\mathbf{K}_{s_1,s_1}}{r}
		\prod_{i=2}^{r}
			\dfrac{
				K_{s_i s_i} - \mathbf{K}_{s_i, s_{1:i-1}} {\mathbf{K}_{\{s_{1:i}\}}}^{-1} \mathbf{K}_{s_{1:i-1}, s_1}
				}{r-(i-1)}

.. hint::

	Since :math:`\mathbf{K}` is an **orthogonal projection** matrix,
	the following Gram factorizations provide an insightful geometrical interpretation of the chain rule mechanism :eq:`eq:chain_rule_K`:

	1. Using :math:`\mathbf{K} = \mathbf{K}^2`
	and :math:`\mathbf{K}^{\dagger}=\mathbf{K}`, we have

	.. math::
		:label: eq:correlation_kernel_factorization_KK.T

		\mathbf{K}
		= \mathbf{K} \mathbf{K}^{\dagger},

	so that the chain rule becomes

	.. math::
		:label: eq:chain_rule_dist2_K

		\mathbb{P}[s_1, \dots, s_r]
		&=	\frac{1}{r!}
			\operatorname{Volume}^2(
			\mathbf{K}_{s_{1},:}, \dots, \mathbf{K}_{s_{r},:}
			)\\
		&=	\dfrac{\left\| \mathbf{K}_{s_1,:} \right\|^2}{r}
			\prod_{i=2}^{r}
			\dfrac{
				\operatorname{distance}^2
				(\mathbf{K}_{s_{i},:},
				\operatorname{Span}
					\left\{
					\mathbf{K}_{s_{1},:}, \dots, \mathbf{K}_{s_{i-1},:}
					\right\}
			}{r-(i-1)}

	2. Using the eigendecomposition, we have

	.. math::
		:label: eq:correlation_kernel_factorization_UU.T

		\mathbf{K}
		= U U^{\dagger},
		\quad \text{with } U^{\dagger} U = I_r,

	so that :eq:`eq:chain_rule_K` becomes

	.. math::
		:label: eq:chain_rule_dist2_U

		\mathbb{P}[s_1, \dots, s_r]
		&=	\frac{1}{r!}
			\operatorname{Volume}^2(
			U_{s_{1},:}, \dots, U_{s_{r},:}
			)\\
		&= \dfrac{\left\| U_{s_1,:} \right\|^2}{r}
			\prod_{i=2}^{r}
			\dfrac{
				\operatorname{distance}^2
				(U_{s_{i},:},
				\operatorname{Span}
					\left\{
					U_{s_{1},:}, \dots, U_{s_{i-1},:}
					\right\}
			}{r-(i-1)}

	In other words, the chain rule formulated as :eq:`eq:chain_rule_dist2_K` and :eq:`eq:chain_rule_dist2_U` is akin to do Gram-Schmidt orthogonalization of the "feature vectors" :math:`\mathbf{K}_{i,:}` or :math:`\mathbf{U}_{i,:}`.
	In the end, projection DPPs favors sets of size :math:`\operatorname{rank}(\mathbf{K})` associated to feature vectors that span large volumes; the chain rule can be understand the simple base :math:`\times` height formula.


.. _finite_dpps_exact_sampling_spectral_method:

Spectral method
---------------

Main idea
=========

The procedure stems from Theorem 7 of :cite:`HKPV06`, i.e., the fact that :ref:`generic DPPs are mixtures of projection DPPs <finite_dpps_mixture>`, suggesting the following two steps algorithm.
Given the spectral decomposition of the correlation kernel :math:`\mathbf{K}`

.. math::

	\mathbf{K}
	= U \Lambda U^{\dagger}
	= \sum_{n=1}^{N} \lambda_n u_n u_n^{\dagger}

.. _finite_dpps_exact_sampling_generic_dpps_step_1:

1. Draw independent :math:`\operatorname{\mathcal{B}er}(\lambda_n)` and collect :math:`\mathcal{B}=\left\{ n ~;~ B_n=1 \right\}`

.. _finite_dpps_exact_sampling_generic_dpps_step_2:

2. Sample from the **projection** DPP with correlation kernel :math:`U_{:\mathcal{B}} {U_{:\mathcal{B}}}^{\dagger} = \sum_{n\in \mathcal{B}} u_n u_n^{\dagger}`.

.. note::

	1. selects a component of the mixture
	2. requires sampling from the corresponding **projection** DPP, cf.

In practice
===========

- Sampling a *projection* :math:`\operatorname{DPP}(\mathbf{K})` from :math:`\mathbf{K}=U U^{\dagger}` with :math:`U^{\dagger}U = I_{\operatorname{rank}(\mathbf{K})}`) can be done in :math:`\mathcal{O}(N\operatorname{rank}(\mathbf{K})^2)`

.. testcode::

	import numpy as np
	from scipy.linalg import qr
	from dppy.finite_dpps import FiniteDPP

	rng = np.random.RandomState(1)

	r, N = 4, 10
	eig_vals = np.ones(r)  # For projection DPP
	eig_vecs, _ = qr(rng.randn(N, r), mode='economic')

	DPP = FiniteDPP(kernel_type='correlation',
					projection=True,
					**{'K_eig_dec': (eig_vals, eig_vecs)})

	for _ in range(10):
		# mode='GS': Gram-Schmidt (default)
		DPP.sample_exact(mode='GS', random_state=rng)

	print(list(map(list, DPP.list_of_samples)))

.. testoutput::

	[[0, 4, 8, 2], [1, 8, 2, 0], [8, 3, 6, 1], [6, 7, 1, 9], [9, 3, 0, 4], [9, 4, 0, 8], [9, 6, 1, 8], [0, 1, 2, 7], [1, 2, 8, 9], [8, 2, 9, 4]]

- Sampling a general :math:`\operatorname{DPP}(\mathbf{K})` from :math:`0_N \preceq\mathbf{K}) \preceq I_N` or :math:`\operatorname{DPP}(\mathbf{L})` from :math:`\preceq\mathbf{L}) \succeq 0_N`, requires an initial :math:`\mathcal{O}(N^3)` eigendecompose either kernel. Then, the average cost to get a sample is of order :math:`\mathcal{O}(N \mathbb{E}\left[|\mathcal{X}|\right]^2)`, where :math:`\mathbb{E}\left[|\mathcal{X}|\right]=\operatorname{trace}(\mathbf{K})`.

.. testcode::

	import numpy as np
	from scipy.linalg import qr
	from dppy.finite_dpps import FiniteDPP

	rng = np.random.RandomState(1)

	r, N = 4, 10
	eig_vals = rng.rand(r)  # General case
	eig_vecs, _ = qr(rng.randn(N, r), mode='economic')

	DPP = FiniteDPP(kernel_type='correlation',
					projection=False,
					**{'K_eig_dec': (eig_vals, eig_vecs)})

	for _ in range(10):
		# mode='GS': Gram-Schmidt (default)
		DPP.sample_exact(mode='GS', random_state=rng)

	print(list(map(list, DPP.list_of_samples)))

.. testoutput::

	[[7], [4], [3, 4], [4, 2, 3], [9, 3], [0], [1], [4, 7], [0, 6], [4]]

.. seealso::

	.. currentmodule:: dppy.finite_dpps

	- :py:meth:`~FiniteDPP.sample_exact`
	- :cite:`HKPV06` Theorem 7, Algorithm 18 and Proposition 19, for the original idea
	- :cite:`KuTa12` Algorithm 1, for a first interpretation of :cite:`HKPV06` algorithm running in :math:`\mathcal{O}(N r^3)`
	- :cite:`Gil14` Algorithm 2, for the :math:`\mathcal{O}(N r^2)` implementation
	- :cite:`TrBaAm18` Algorithm 3, for a technical report on DPP sampling


Caution
=======

.. attention::

	For the chain rule as described in :eq:`eq:chain_rule_K` to be valid, it is **crucial** that :math:`\mathbf{K}` is a *projection* kernel.
	It is the very reason why the normalization constants of the conditionals  are independent of the previous points and that :math:`S=\{s_1, \dots, s_r\}` is a valid sample of :math:`\operatorname{DPP}(\mathbf{K})`.

	To see this, consider :math:`\mathbf{K}` satisfying :eq:`eq:suff_cond_K` with Gram factorization :math:`\mathbf{K} = VV^{\dagger}` and denote :math:`Y=\{s_1, \dots, s_{j-1}\}`.
	Without prior asumption on :math:`V`, the Schur complement formula allows to express the ratio of determinants appearing in the conditionals as

	.. math::

		\frac{\det \mathbf{K}_{Y+i}}{\det \mathbf{K}_{Y}}
		&= \mathbf{K}_{ii}
		- \mathbf{K}_{iY} \left[\mathbf{\mathbf{K}}_{Y}\right]^{-1} \mathbf{K}_{Yi}\\
		&= \mathbf{K}_{ii}
		- V_{i:}V_{Y:}^{\dagger}
		\left[V_{Y:} V_{Y:}^{\dagger}\right]^{-1}
		V_{Y:} V_{i:}^{\dagger} \\
		&= \mathbf{K}_{ii}
		- V_{i:} \Pi_{V_{Y:}} V_{i:}^{\dagger}

	where :math:`\Pi_{V_{Y:}}` is the orthogonal projection onto the span of the (independent) rows of :math:`V_{Y:}`.

	Now, let's compute the normalizing constant.
	The first term :math:`\operatorname{Tr}(\mathbf{K})` is independent of :math:`Y`, contrary to the second term if no additional assumption is made on the Gram factor :math:`V`.
	Indeed,

	.. math::

		\sum_{i=1}^N
			\frac{\det \mathbf{K}_{Y+i}}{\det \mathbf{K}_{Y}}
		&= \sum_{i=1}^N \mathbf{K}_{ii}
		  - V_{i:} \Pi_{V_{Y:}} V_{i:}^{\dagger}\\
		&= \operatorname{Tr}(\mathbf{K})
		  - \operatorname{Tr}(V \Pi_{V_{Y:}} V^{\dagger})\\
		&= \operatorname{Tr}(\mathbf{K})
		  - \operatorname{Tr}(\Pi_{V_{Y:}}V^{\dagger}V)\\

	The first term :math:`\operatorname{Tr}(\mathbf{K})` is independent of :math:`Y`, but this is no longer true for the second term without additional assumption on the Gram factor V.

	However, for :math:`V = \mathbf{K}` or :math:`U`, we have

	.. math::

		&\qquad\operatorname{Tr}(\mathbf{K})
		&\qquad\operatorname{Tr}(\mathbf{K})
			- \operatorname{Tr}(\Pi_{\mathbf{K}_{Y:}}\mathbf{K}\mathbf{K}^{\dagger})
		&\qquad
		\operatorname{Tr}(\mathbf{K})
			- \operatorname{Tr}(\Pi_{U_{Y:}}U^{\dagger}U)
			\\
		&\qquad= \operatorname{rank}(\mathbf{K})
		&\qquad= r - \operatorname{Tr}(\Pi_{\mathbf{K}_{Y:}}\mathbf{K})
		&\qquad= r - \operatorname{Tr}(\Pi_{U_{Y:}}I_r)
			\\
		&\qquad= r
		&\qquad= r - \operatorname{Tr}(\Pi_{\mathbf{K}_{Y:}})
		&\qquad= r - \operatorname{Tr}(\Pi_{U_{Y:}})
			\\
		&
		&\qquad= r - |Y|
		&\qquad= r - |Y|

.. _finite_dpps_exact_sampling_generic_dpps:

Generic DPPs
============

When considering non-projection DPPs, the eigendecomposition of the underlying kernel is required; adding an initial extra :math:`\mathcal{O}(N^3)` cost to sampling a *projection DPP*

.. tip::

	If the likelihood kernel was constructed as :math:`\mathbf{L}=\Phi^{\dagger}\Phi` where :math:`\Phi` is a :math:`d\times N` feature matrix, it may be judicious to exploit the lower dimensional structure of the *dual* kernel :math:`\tilde{\mathbf{L}} = \Phi \Phi^{\dagger}`.
	Indeed, when :math:`d<N` computing the eigendecomposition of :math:`\tilde{\mathbf{L}}` costs :math:`\mathcal{O}(d^3)` compared to :math:`\mathcal{O}(N^3)` for :math:`\mathbf{L}`.

.. note::

	Noting the respective spectral decompositions

	.. math::

		\mathbf{K} = U \Lambda U^{\top},
		\quad \mathbf{L} = V \Delta V^{\top}
		\quad \text{and} \quad
		\tilde{\mathbf{L}} = W \Gamma W^{\top}

	where

	.. math::

		\Lambda = \Delta (I+\Delta)^{-1}
		\quad \text{and} \quad
		U = V

	and with an abuse of notation, considering only the non-zero eigenvalues (and corresponding eigenvectors)

	.. math::

		\Delta = \Gamma
		\quad \text{and} \quad
		U = V = \Phi^{\top} W \Gamma^{-1/2}

In the generic setting, the exact sampling scheme works as a two steps algorithm based on the property that :ref:`generic DPPs are mixtures of projection ones <finite_dpps_mixture>`.

.. hint::

	- :ref:`Phase 1 <finite_dpps_exact_sampling_generic_dpps_phase_1>` selects a component of the mixture
	- :ref:`Phase 2 <finite_dpps_exact_sampling_generic_dpps_phase_2>` samples from this *projection* DPP component

In practice, sampling is performed in the following way:

.. _finite_dpps_exact_sampling_generic_dpps_phase_1:

**Phase 1** Draw independent Bernoulli variables :math:`(B_n)` with parameters the eigenvalues of :math:`\mathbf{K}`:

	.. math::

		\lambda_n
		= \frac{\delta_n}{1+\delta_n}
		= \frac{\gamma_n}{1+\gamma_n}

.. _finite_dpps_exact_sampling_generic_dpps_phase_2:

**Phase 2** Conditionally on :math:`(B_n)` set :math:`\mathcal{B} = \{ n ~;~ B_n = 1 \}` and apply the chain rule ref Eq phase 2 with

	.. math::

		r = |\mathcal{B}|
		\quad \text{and} \quad
		U =
			U_{:\mathcal{B}}, \
			V_{:\mathcal{B}}, \
			\Phi^{\top} W_{:\mathcal{B}} \Gamma_{:\mathcal{B}}^{-1/2} \
		\text{respectively}

.. testcode::

	from numpy.random import RandomState
	from scipy.linalg import qr
	from dppy.finite_dpps import FiniteDPP

	rng = RandomState(1)

	r, N = 5, 10
	eig_vals = rng.rand(r)
	eig_vecs, _ = qr(rng.randn(N, r), mode='economic')

	DPP = FiniteDPP('correlation', **{'K_eig_dec': (eig_vals, eig_vecs)})

	for _ in range(10):
		DPP.sample_exact(random_state=rng)

	print(list(map(list, DPP.list_of_samples)))

.. testoutput::

	[[7], [4], [3, 4], [4, 2, 3], [9, 3], [0], [1], [4, 7], [0, 6], [4]]

.. seealso::

	.. currentmodule:: dppy.finite_dpps

	:py:meth:`~FiniteDPP.sample_exact`


.. _finite_dpps_exact_sampling_cholesky_method:

Cholesky-based method
---------------------

This method requires acces to the correlation kernel :math:`\mathbf{K}` applying the chain rule on sets; where each item in turn is decided to be excluded or included in the sample. We present it in Section :ref:`finite_dpps_exact_sampling_cholesky_method`.

Main idea
=========

In practice
-----------

The method is fully generic since it applies to any (valid), even non hermitian, correlation kernel :math:`\mathbf{K}`.
The simplicty of the implementation is even more surprising, see the pseudo-code below.

.. code-block::
	# Poulson (2019, Algorithm 1)

	sample = []
	A = K.copy()

	for j in range(n):

		if Bernoulli(A[j, j]) == 1:
			sample.append(j)
		else:
			A[j, j] −= 1

        A[j+1:, j] /= A[j, j]
        A[j+1:, j+1:] -= A[j+1:, j] @ A[j, j+1:]  # outer product

	return sample, A


.. seealso::

	- :cite:`Pou19`
	- :cite:`LaGaDe18`

.. _finite_dpps_exact_sampling_intermediate_sampling_method:

Intermediate sampling method
----------------------------

Main idea
=========


.. todo::

	TBC

.. seealso::

	:cite:`DeCaVa19`


.. _finite_dpps_exact_sampling_k_dpps:

k-DPPs
------

A :math:`\operatorname{k-DPP}` viewed as a :math:`\operatorname{DPP}(\mathbf{L})` constrained to a fixed cardinality :math:`k` (see :ref:`définition <finite_dpps_definition_k_dpps>`),  can be sampled using a rejection mechanism i.e. sample :math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{L})` and consider only realizations with cardinality :math:`|X| = k`.

.. caution::

	- :math:`k` must satisfy :math:`k \leq \operatorname{rank}(L)`

In practice, the 2 steps algorithm for :ref:`sampling generic DPPs <finite_dpps_exact_sampling_generic_dpps>` can be adapted to generate fixed cardinality samples.

More specifically,

- :ref:`Phase 1 <finite_dpps_exact_sampling_generic_dpps_phase_1>` is replaced by :cite:`KuTa12` Algorithm 8. It requires the evaluation of the elementary symmetric polynomials in the eigenvalues of :math:`\mathbf{L}` ; :math:`[E[l, n]]_{l=1, n=1}^{k, N}` with :math:`E[l, n]:=e_l(\lambda_1, \dots, \delta_n)`.

.. code-block:: python

	# This is a pseudo code, in particular Python indexing is not respected everywhere
	B = set({})
	l = k

	for n in range(N, 0, -1):

	  if Unif(0,1) < delta[n] * E[l-1, n-1] / E[l, n]:
	    l -= 1
	    B.union({n})

	    if l == 0:
	      break

- :ref:`Phase 2 <finite_dpps_exact_sampling_generic_dpps_phase_1>` is unchanged

.. testcode::

	import numpy as np
	from dppy.finite_dpps import FiniteDPP

	rng = np.random.RandomState(1)

	r, N = 5, 10
	# Random feature vectors
	Phi = rng.randn(r, N)
	DPP = FiniteDPP('likelihood', **{'L': Phi.T.dot(Phi)})

	k = 4
	for _ in range(10):
	    DPP.sample_exact_k_dpp(size=k, random_state=rng)

	print(list(map(list, DPP.list_of_samples)))

.. testoutput::

	[[1, 8, 5, 7], [3, 8, 5, 9], [5, 3, 1, 8], [5, 8, 2, 9], [1, 2, 9, 6], [1, 0, 2, 3], [7, 0, 3, 5], [8, 3, 7, 6], [0, 2, 3, 7], [1, 3, 7, 5]]

.. seealso::

	.. currentmodule:: dppy.finite_dpps

	- :py:meth:`~FiniteDPP.sample_exact_k_dpp`
	- :cite:`KuTa12` Algorithm 7 for the recursive evaluation of the elementary symmetric polynomials :math:`[e_l(\lambda_1, \dots, \delta_n)]_{l=1, n=1}^{k, N}` in the eigenvalues of :math:`\mathbf{L}`
