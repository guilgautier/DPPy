.. currentmodule:: dppy.finite_dpps

.. _finite_dpps_exact_sampling:

Exact sampling
**************

Consider a finite DPP defined by its correlation kernel :math:`\mathbf{K}` :eq:`eq:inclusion_proba` or likelihood kernel :math:`\mathbf{L}` :eq:`eq:likelihood`.
There exist three main types of exact sampling procedures:

1. The spectral method requires the eigendecomposition of the correlation kernel :math:`\mathbf{K}` or the likelihood kernel :math:`\mathbf{L}`. It is based on the fact that :ref:`generic DPPs are mixtures of projection DPPs <finite_dpps_mixture>` together with the application of the chain rule to sample projection DPPs. It is presented in Section :ref:`finite_dpps_exact_sampling_spectral_method`.

2. a Cholesky-based procedure which requires the correlation kernel :math:`\mathbf{K}` (even non-Hermitian!). It boilds down to applying the chain rule on sets; where each item in turn is decided to be excluded or included in the sample. It is presented in Section :ref:`finite_dpps_exact_sampling_cholesky_method`.

3. rencently, :cite:`DeCaVa19` have also proposed an alternative method to get exact samples: first sample an intermediate distribution and correct the bias by thinning the intermediate sample using a carefully designed DPP. It is presented in Section :ref:`finite_dpps_exact_sampling_intermediate_sampling_method`.

.. note::

	- There exist specific samplers for special DPPs, like the ones presented in Section :ref:`exotic_dpps`.

.. important::

	In the next section, we describe the Algorithm 18 of :cite:`HKPV06`, based on the chain rule, which was originally designed to sample continuous projection DPPs.
	Obviously, it has found natural a application in the finite setting for sampling projection :math:`\operatorname{DPP}(\mathbf{K})`.
	However, **we insist on the fact that this chain rule mechanism is specific to orthogonal projection kernels**.
	In particular, it cannot be applied blindly to sample general :math:`k-\operatorname{DPP}(\mathbf{L})` but it remains valid **only** when :math:`\operatorname{DPP}(\mathbf{L})` is an orthogonal projection kernel.

	This crucial point is developed in the following :ref:`finite_kdpps_exact_sampling_chain_rule_projection_kernel_caution` section.


.. _finite_dpps_exact_sampling_projection_dpp_chain_rule:

Projection DPPs: the chain rule
-------------------------------

Recall that projection :math:`\operatorname{DPP}(\mathbf{K})` generated subsets :math:`S=\{s_1, \dots, s_r\}` with fixed cardinality :math:`r=\operatorname{rank}(\mathbf{K})=\operatorname{trace}(\mathbf{K})`, almost surely.
So that the likelihood :eq:`eq:likelihood_projection_K` of :math:`S=\{s_1, \dots, s_r\}` reads

.. math::

	\mathbb{P}[\mathcal{X}=S]
	= \det \mathbf{K}_S.

Using the invariance by permutation of the derterminant it is sufficient to apply the chain rule to sample :math:`(s_1, \dots, s_r)` with joint distribution

.. math::

	\mathbb{P}[s_1, \dots, s_r]
	= \frac{1}{r!} \mathbb{P}[\mathcal{X}=\{s_1, \dots, s_r\}]
	= \frac{1}{r!} \det \mathbf{K}_S,

and forget about the sequential feature of the chain rule to get a valid sample :math:`\{s_1, \dots, s_r\} \sim \operatorname{DPP}(\mathbf{K})`.

Considering :math:`S=\{s_1, \dots, s_r\}` such that :math:`\mathbb{P}[\mathcal{X}=S] = \det \mathbf{K}_S > 0`, the following generic formulation of the chain rule

.. math::

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
	These formulations can also be understood as an application of the base :math:`\times` height formula.
	In the end, projection DPPs favors sets of :math:`r=\operatorname{rank}(\mathbf{K})` of items are associated to feature vectors that span large volumes.

.. _finite_dpps_exact_sampling_projection_dpp_chain_rule_in_practice:

In practice
===========

The cost of getting one sample from a **projection** DPP is of order :math:`\mathcal{O}(N\operatorname{rank}(\mathbf{K})^2)`, whenever :math:`\operatorname{DPP}(\mathbf{K})` is defined through

- :math:`\mathbf{K}` itself; sampling relies on formulations :eq:`eq:chain_rule_dist2_K` or :eq:`eq:chain_rule_schur`

	.. testcode::

		import numpy as np
		from scipy.linalg import qr
		from dppy.finite_dpps import FiniteDPP

		seed = 0
		rng = np.random.RandomState(seed)

		r, N = 4, 10
		eig_vals = np.ones(r)  # For projection DPP
		eig_vecs, _ = qr(rng.randn(N, r), mode='economic')

		DPP = FiniteDPP(kernel_type='correlation',
		                projection=True,
		                **{'K': (eig_vecs * eig_vals).dot(eig_vecs.T)})

		for mode in ('GS', 'Schur', 'Chol'):  # default: GS

		    rng = np.random.RandomState(seed)
		    DPP.flush_samples()

		    for _ in range(10):
		        DPP.sample_exact(mode=mode, random_state=rng)

		    print(DPP.sampling_mode)
		    print(list(map(list, DPP.list_of_samples)))

	.. testoutput::

		GS
		[[5, 7, 2, 1], [4, 6, 2, 9], [9, 2, 6, 4], [5, 9, 0, 1], [0, 8, 6, 7], [9, 6, 2, 7], [0, 6, 2, 9], [5, 2, 1, 8], [5, 4, 0, 8], [5, 6, 9, 1]]
		Schur
		[[5, 7, 2, 1], [4, 6, 2, 9], [9, 2, 6, 4], [5, 9, 0, 1], [0, 8, 6, 7], [9, 6, 2, 7], [0, 6, 2, 9], [5, 2, 1, 8], [5, 4, 0, 8], [5, 6, 9, 1]]
		Chol
		[[5, 7, 6, 0], [4, 6, 5, 7], [9, 5, 0, 1], [5, 9, 2, 4], [0, 8, 1, 7], [9, 0, 5, 1], [0, 6, 5, 9], [5, 0, 1, 9], [5, 0, 2, 8], [5, 6, 9, 1]]

	.. seealso::

		- :py:meth:`~FiniteDPP.sample_exact`
		- :cite:`HKPV06` Theorem 7, Algorithm 18 and Proposition 19, for the original idea
		- :cite:`Pou19` Algorithm 3, for the equivalent Cholesky-based perspective with cost of order :math:`\mathcal{O}(N r^2)`

- its eigenvectors :math:`U`, i.e., :math:`\mathbf{K}=U U^{\dagger}` with :math:`U^{\dagger}U = I_{\operatorname{rank}(\mathbf{K})}`; sampling relies on :eq:`eq:chain_rule_dist2_U`

	.. testcode::

		import numpy as np
		from scipy.linalg import qr
		from dppy.finite_dpps import FiniteDPP

		seed = 0
		rng = np.random.RandomState(seed)

		r, N = 4, 10
		eig_vals = np.ones(r)  # For projection DPP
		eig_vecs, _ = qr(rng.randn(N, r), mode='economic')

		DPP = FiniteDPP(kernel_type='correlation',
						projection=True,
						**{'K_eig_dec': (eig_vals, eig_vecs)})

		rng = np.random.RandomState(seed)

		for _ in range(10):
			# mode='GS': Gram-Schmidt (default)
			DPP.sample_exact(mode='GS', random_state=rng)

		print(list(map(list, DPP.list_of_samples)))

	.. testoutput::

		[[5, 7, 2, 1], [4, 6, 2, 9], [9, 2, 6, 4], [5, 9, 0, 1], [0, 8, 6, 7], [9, 6, 2, 7], [0, 6, 2, 9], [5, 2, 1, 8], [5, 4, 0, 8], [5, 6, 9, 1]]

	.. seealso::

		- :py:meth:`~FiniteDPP.sample_exact`
		- :cite:`HKPV06` Theorem 7, Algorithm 18 and Proposition 19, for the original idea
		- :cite:`KuTa12` Algorithm 1, for a first interpretation of the spectral counterpart of :cite:`HKPV06` Algorithm 18 running in :math:`\mathcal{O}(N r^3)`
		- :cite:`Gil14` Algorithm 2, for the :math:`\mathcal{O}(N r^2)` implementation
		- :cite:`TrBaAm18` Algorithm 3, for a technical report on DPP sampling

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

.. _finite_dpps_exact_sampling_spectral_method_step_1:

**Step** 1. Draw independent Bernoulli random variables :math:`B_n \sim \operatorname{\mathcal{B}er}(\lambda_n)` for :math:`n=1,\dots, N` and collect :math:`\mathcal{B}=\left\{ n ~;~ B_n=1 \right\}`

.. _finite_dpps_exact_sampling_spectral_method_step_2:

**Step** 2. Sample from the **projection** DPP with correlation kernel :math:`U_{:\mathcal{B}} {U_{:\mathcal{B}}}^{\dagger} = \sum_{n\in \mathcal{B}} u_n u_n^{\dagger}`.

.. note::

	**Step** :ref:`1. <finite_dpps_exact_sampling_spectral_method_step_1>` selects a component of the mixture

	**Step** :ref:`2. <finite_dpps_exact_sampling_spectral_method_step_2>` requires sampling from the corresponding **projection** DPP, cf.

In practice
===========

- Sampling *projection* :math:`\operatorname{DPP}(\mathbf{K})` from the eigendecomposition of :math:`\mathbf{K}=U U^{\dagger}` with :math:`U^{\dagger}U = I_{\operatorname{rank}(\mathbf{K})}`) can be done by applying

  	**Step** :ref:`2. <finite_dpps_exact_sampling_spectral_method_step_2>` with a cost of order :math:`\mathcal{O}(N\operatorname{rank}(\mathbf{K})^2)`, see :ref:`the section above <finite_dpps_exact_sampling_projection_dpp_chain_rule_in_practice>`

- Sampling :math:`\operatorname{DPP}(\mathbf{K})` from :math:`0_N \preceq\mathbf{K} \preceq I_N` can be done by following

  	**Step** 0. compute the eigendecomposition of :math:`\mathbf{L} = U \Lambda U^{\dagger}` in :math:`\mathcal{O}(N^3)`.

 	**Step** :ref:`1. <finite_dpps_exact_sampling_spectral_method_step_1>`  draw independent Bernoulli random variables :math:`B_n \sim \operatorname{\mathcal{B}er}(\lambda_n)` for :math:`n=1,\dots, N` and collect :math:`\mathcal{B}=\left\{ n ~;~ B_n=1 \right\}`

	**Step** :ref:`2. <finite_dpps_exact_sampling_spectral_method_step_2>` with an average cost of order :math:`\mathcal{O}(N \mathbb{E}\left[|\mathcal{X}|\right]^2)`, where :math:`\mathbb{E}\left[|\mathcal{X}|\right]=\operatorname{trace}(\mathbf{K})=\sum_{n=1}^{N} \lambda_n`.

	.. testcode::

		from numpy.random import RandomState
		from scipy.linalg import qr
		from dppy.finite_dpps import FiniteDPP

		rng = RandomState(0)

		r, N = 4, 10
		eig_vals = rng.rand(r)  # For projection DPP
		eig_vecs, _ = qr(rng.randn(N, r), mode='economic')

		DPP = FiniteDPP(kernel_type='correlation',
						projection=False,
						**{'K': (eig_vecs*eig_vals).dot(eig_vecs.T)})

		for _ in range(10):
			# mode='GS': Gram-Schmidt (default)
			DPP.sample_exact(mode='GS', random_state=rng)

		print(list(map(list, DPP.list_of_samples)))

	.. testoutput::

		[[7, 0, 1, 4], [6], [0, 9], [0, 9], [8, 5], [9], [6, 5, 9], [9], [3, 0], [5, 1, 6]]

- Sampling :math:`\operatorname{DPP}(\mathbf{L})` from :math:`\mathbf{L} \succeq 0_N` can be done by following

  	**Step** 0. compute the eigendecomposition of :math:`\mathbf{L} = V \Delta V^{\dagger}` in :math:`\mathcal{O}(N^3)`.

 	**Step** :ref:`1. <finite_dpps_exact_sampling_spectral_method_step_1>`  draw independent Bernoulli random variables :math:`B_n \sim \operatorname{\mathcal{B}er}(\frac{\delta_n}{1+\delta_n})` for :math:`n=1,\dots, N` and collect :math:`\mathcal{B}=\left\{ n ~;~ B_n=1 \right\}`

	**Step** :ref:`2. <finite_dpps_exact_sampling_spectral_method_step_2>`
	sample from the **projection** DPP with correlation kernel defined by its eigenvectors :math:`V_{:,\mathcal{B}}`.

	.. important::

		Step 0. must be performed once and for all in :math:`\mathcal{O}(N^3)`, the average cost of getting one sample is :math:`\mathcal{O}(N \mathbb{E}\left[|\mathcal{X}|\right]^2)`, where :math:`\mathbb{E}\left[|\mathcal{X}|\right]=\operatorname{trace}(\mathbf{L(I+L)^{-1}})=\sum_{n=1}^{N} \frac{\delta_n}{1+\delta_n}`

	.. testcode::

		from numpy.random import RandomState
		from scipy.linalg import qr
		from dppy.finite_dpps import FiniteDPP

		rng = RandomState(0)

		r, N = 4, 10
		phi = rng.randn(r, N)

		DPP = FiniteDPP(kernel_type='likelihood',
						projection=False,
						**{'L': phi.T.dot(phi)})

		for _ in range(10):
			# mode='GS': Gram-Schmidt (default)
			DPP.sample_exact(mode='GS', random_state=rng)

		print(list(map(list, DPP.list_of_samples)))

	.. testoutput::

		[[3, 1, 0, 4], [9, 6], [4, 1, 3, 0], [7, 0, 6, 4], [5, 0, 7], [4, 0, 2], [5, 3, 8, 4], [0, 5, 2], [7, 0, 2], [6, 0, 3]]

- Sampling a :math:`\operatorname{DPP}(\mathbf{L})` for which each item is represented by a :math:`d\leq N` dimensional feature vector, all stored in a _feature_ matrix :math:`\Phi \in \mathbb{R}^{d\times N}`, so that :math:`\mathbf{L}=\Phi^{\top} \Phi \succeq 0_N`, can be done by following

  	**Step** 0. compute the so-called *dual* kernel :math:`\tilde{L}=\Phi \Phi^{\dagger}\in \mathbb{R}^{d\times}` and eigendecompose it :math:`\tilde{\mathbf{L}} = W \Gamma W^{\top}`.
  	This corresponds to a cost of order :math:`\mathcal{O}(Nd^2 + d^3)`.

 	**Step** :ref:`1. <finite_dpps_exact_sampling_spectral_method_step_1>`  draw independent Bernoulli random variables :math:`B_i \sim \operatorname{\mathcal{B}er}(\gamma_i)` for :math:`i=1,\dots, d` and collect :math:`\mathcal{B}=\left\{ i ~;~ B_i=1 \right\}`

	**Step** :ref:`2. <finite_dpps_exact_sampling_spectral_method_step_2>`
	sample from the **projection** DPP with correlation kernel defined by its eigenvectors :math:`\Phi^{\top} W_{:,\mathcal{B}} \Gamma_{\mathcal{B}}^{-1/2}`.

	.. important::

		Step 0. must be performed once and for all in :math:`\mathcal{O}(Nd^2 + d^3)`.
		Then, the average cost of getting one sample is :math:`\mathcal{O}(N \mathbb{E}\left[|\mathcal{X}|\right]^2)`, where :math:`\mathbb{E}\left[|\mathcal{X}|\right]=\operatorname{trace}(\mathbf{L(I+L)^{-1}})=\sum_{i=1}^{d} \frac{\gamma_i}{1+\gamma_i}`

	.. testcode::

		from numpy.random import RandomState
		from scipy.linalg import qr
		from dppy.finite_dpps import FiniteDPP

		rng = RandomState(0)

		r, N = 4, 10
		phi = rng.randn(r, N)  # L = phi.T phi, L_dual = phi phi.T

		DPP = FiniteDPP(kernel_type='likelihood',
						projection=False,
						**{'L_gram_factor': phi})

		for _ in range(10):
			# mode='GS': Gram-Schmidt (default)
			DPP.sample_exact(mode='GS', random_state=rng)

		print(list(map(list, DPP.list_of_samples)))

	.. testoutput::

		L_dual = Phi Phi.T was computed: Phi (dxN) with d<N
		[[9, 0, 2, 3], [0, 1, 5, 2], [7, 0, 9, 4], [2, 0, 3], [6, 4, 0, 3], [5, 0, 6, 3], [0, 6, 3, 9], [4, 0, 9], [7, 3, 9, 4], [9, 4, 3]]

.. _finite_dpps_exact_sampling_cholesky_method:

Cholesky-based method
---------------------

Main idea
=========

This method requires acces to the correlation kernel :math:`\mathbf{K}` to perform a bottom-up chain rule on sets: starting from the empty set, each item in turn is decided to be added or excluded from the sample.
This can be summarized as the exploration of the binary probability tree displayed in :numref:`fig:cholesky_chain_rule_sets`.

.. figure:: ../_images/cholesky_chain_rule_sets.png
   :width: 80%
   :align: center
   :name: fig:cholesky_chain_rule_sets

   Probability tree corresponding to the chain rule on sets

**Example:** for :math:`N=5`, if :math:`\left\{ 1, 4 \right\}` was sampled, the path in the probability tree would correspond to

.. math::

	\mathbb{P}\!\left[\mathcal{X} = \left\{ 1, 4 \right\}\right]
	=
	&\mathbb{P}\!\left[
					1\in \mathcal{X}
				\right]\\
	&\times\mathbb{P}\!\left[
					2\notin \mathcal{X}
					\mid 1\in \mathcal{X}
				\right]\\
	&\times\mathbb{P}\!\left[
					3\notin \mathcal{X}
					\mid 1\in \mathcal{X}, 2\notin \mathcal{X}
				\right]\\
	&\times\mathbb{P}\!\left[
					4\in \mathcal{X}
					\mid 1\in \mathcal{X},
					\left\{ 2, 3 \right\} \cap \mathcal{X} = \emptyset
				\right]\\
	&\times\mathbb{P}\!\left[
					5\notin \mathcal{X}
					\mid \left\{ 1, 4 \right\} \subset \mathcal{X},
					\left\{ 2, 3 \right\} \cap \mathcal{X} = \emptyset
				\right],

where each conditional probability has closed formed expression given by :eq:`eq:conditioned_on_S_in_X` and :eq:`eq:conditioned_on_S_notin_X`, namely

.. math::

	\mathbb{P}[T \subset \mathcal{X} \mid S \subset \mathcal{X}]
        &= \det\left[\mathbf{K}_T - \mathbf{K}_{TS} \mathbf{K}_S^{-1} \mathbf{K}_{ST}\right]\\
	\mathbb{P}[T \subset \mathcal{X} \mid S \cap \mathcal{X} = \emptyset]
    	&= \det\left[\mathbf{K}_T - \mathbf{K}_{TS} (\mathbf{K}_S - I)^{-1} \mathbf{K}_{ST}\right].

.. important::

	This quantities can be computed efficiently as they appear in the computation of the Cholesky-type :math:`LDL^{\dagger}` or :math:`LU` factorization of the correlation :math:`\mathbf{K}` kernel, in the Hermitian or non-Hermitian case, respectively.
	See :cite:`Pou19` for the details.

.. note::

	The sparsity of :math:`\mathbf{K}` can be leveraged to get derive faster samples using the correspondence between the chain rule on sets and Cholesky-type factorizations, see e.g., :cite:`Pou19` Section 4.

In practice
===========

.. important::

	- The method is fully generic since it applies to any (valid), even non-Hermitian, correlation kernel :math:`\mathbf{K}`.
	- Each sample costs :math:`\mathcal{O}(N^3)`.
	- Nevertheless, the connexion between the chain rule on sets and Cholesky-type factorization is nicely supported by the surprising simplicty to implement the corresponding sampler.

	.. code-block:: python

		# Poulson (2019, Algorithm 1) pseudo-code

		sample = []
		A = K.copy()

		for j in range(N):

			if np.random.rand() < A[j, j]:  # Bernoulli(A_jj)
				sample.append(j)
			else:
				A[j, j] -= 1

			A[j+1:, j] /= A[j, j]
			A[j+1:, j+1:] -= np.outer(A[j+1:, j], A[j, j+1:])

		return sample, A

.. testcode::

	from numpy.random import RandomState
	from scipy.linalg import qr
	from dppy.finite_dpps import FiniteDPP

	rng = RandomState(1)

	r, N = 4, 10
	eig_vals = rng.rand(r)  # For projection DPP
	eig_vecs, _ = qr(rng.randn(N, r), mode='economic')

	DPP = FiniteDPP(kernel_type='correlation',
					projection=False,
					**{'K': (eig_vecs*eig_vals).dot(eig_vecs.T)})

	for _ in range(10):
		DPP.sample_exact(mode='Chol', random_state=rng)

	print(list(map(list, DPP.list_of_samples)))

.. testoutput::

	[[2, 9], [0], [2], [6], [4, 9], [2, 7, 9], [0], [1, 9], [0, 1, 2], [2]]

.. seealso::

	- :cite:`Pou19`
	- :cite:`LaGaDe18`

.. _finite_dpps_exact_sampling_intermediate_sampling_method:

Intermediate sampling method
----------------------------

Main idea
=========

First sample an intermediate distribution and correct the bias by thinning the intermediate sample using a carefully designed DPP.

.. seealso::

	:cite:`DeCaVa19`

In practice
===========

   - In certain regimes, this procedure may be more practical with an overall :math:`\mathcal{O}(N \text{poly}(\mathbb{E}\left[|\mathcal{X}|\right]) \text{polylog}(N))` cost.

.. todo::

	TBC

.. _finite_dpps_exact_sampling_k_dpps:

k-DPPs
------

Main idea
=========

A :math:`\operatorname{k-DPP}` viewed as a :math:`\operatorname{DPP}(\mathbf{L})` constrained to a fixed cardinality :math:`k` (see :ref:`définition <finite_dpps_definition_k_dpps>`),  can be sampled using a rejection mechanism i.e. sample :math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{L})` and consider only realizations with cardinality :math:`|X| = k`.

.. caution::

	- :math:`k` must satisfy :math:`k \leq \operatorname{rank}(L)`

In practice
===========

In practice, the 2 steps algorithm for :ref:`sampling generic DPPs <finite_dpps_exact_sampling_spectral_method>` can be adapted to generate fixed cardinality samples.

More specifically,

**Step** :ref:`1. <finite_dpps_exact_sampling_spectral_method_step_1>` is replaced by :cite:`KuTa12` Algorithm 8. It requires the evaluation of the `elementary symmetric polynomials <https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial>`_ in the eigenvalues of :math:`\mathbf{L}` ; :math:`[E[l, n]]_{l=1, n=1}^{k, N}` with :math:`E[l, n]:=e_l(\lambda_1, \dots, \delta_n)`.

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

**Step** :ref:`2. <finite_dpps_exact_sampling_spectral_method_step_2>` is unchanged

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

	- :py:meth:`~FiniteDPP.sample_exact_k_dpp`
	- :cite:`KuTa12` Algorithm 7 for the recursive evaluation of the elementary symmetric polynomials :math:`[e_l(\lambda_1, \dots, \delta_n)]_{l=1, n=1}^{k, N}` in the eigenvalues of :math:`\mathbf{L}`

.. _finite_kdpps_exact_sampling_chain_rule_projection_kernel_caution:

Caution
=======

.. attention::

	As mentioned earlier, Algorithm 18 of :cite:`HKPV06` was designed only for projection DPPs.
	That is to say, the chain rule :eq:`eq:chain_rule_schur` can not be blindly applied to generic correlation kernels :math:`0_N \preceq \mathbf{K} \preceq I_N`, it is **only** valid for orthogonal projection kernels, which can be characterized equivalently by

	a. :math:`\mathbf{K}^2=\mathbf{K}` and :math:`\mathbf{K}^{\dagger}=\mathbf{K}`
	b. :math:`\mathbf{K}=U U^{\dagger}` with :math:`U^{\dagger} U=I_r` where :math:`r=\operatorname{rank}(\mathbf{K})`.

	Next, we explain why the chain rule :eq:`eq:chain_rule_schur` only applies to projection DPPs.

	Recall that a DPP can be understood as a random **subset** :math:`\mathcal{X}\sim \operatorname{DPP}(\mathbf{K})` whose size is also random in the general
	The first questions that should come

	To do this, consider a valid correlation kernel: :math:`0_N \preceq \mathbf{K} \preceq I_N`.
	In that case we

Using the invariance by permutation of the derterminant it is sufficient to apply the chain rule to sample :math:`(s_1, \dots, s_r)` with joint distribution

.. math::

	\mathbb{P}[s_1, \dots, s_r]
	= \frac{1}{r!} \mathbb{P}[\mathcal{X}=\{s_1, \dots, s_r\}]
	= \frac{1}{r!} \det \mathbf{K}_S,

	Now consider the following factorization :math:`\mathbf{K} = VV^{\dagger}`.

	and denote :math:`Y=\{s_1, \dots, s_{j-1}\}`.
	Without prior asumption on :math:`V`, the Schur complement formula allows to express the ratio of determinants appearing in the conditionals as

	.. math::

		\frac{\det \mathbf{K}_{Y+i}}{\det \mathbf{K}_{Y}}
		&= \mathbf{K}_{ii}
			- \mathbf{K}_{iY} \left[\mathbf{\mathbf{K}}_{Y}\right]^{-1} \mathbf{K}_{Yi}\\
		&= \mathbf{K}_{ii}
			- V_{i:}V_{Y:}^{\dagger}
			\left[V_{Y:} {V_{Y:}}^{\dagger}\right]^{-1}
			V_{Y:} V_{i:}^{\dagger} \\
		&= \mathbf{K}_{ii}
			- V_{i:} \Pi_{V_{Y:}} {V_{i:}}^{\dagger}

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



	In particular, a. implies that :math:`\mathbf{K}=\mathbf{K}\mathbf{K}^{\dagger}`.

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
