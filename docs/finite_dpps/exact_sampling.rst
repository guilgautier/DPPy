.. currentmodule:: dppy.finite_dpps

.. _finite_dpps_exact_sampling:

Exact sampling
**************

Consider a finite DPP defined by its correlation kernel :math:`\mathbf{K}` :eq:`eq:inclusion_proba_DPP_K` or likelihood kernel :math:`\mathbf{L}` :eq:`eq:likelihood_DPP_L`.
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
	In particular, it cannot be applied blindly to sample general :math:`\operatorname{k-DPP}(\mathbf{L})` but it is valid when :math:`\mathbf{L}` is an orthogonal projection kernel.

	This crucial point is developed in the following :ref:`finite_kdpps_exact_sampling_chain_rule_projection_kernel_caution` section.


.. _finite_dpps_exact_sampling_projection_dpp_chain_rule:

Projection DPPs: the chain rule
-------------------------------

Recall that the :ref:`number of points of a projection <finite_dpps_properties_number_of_points_dpp_K_projection>` :math:`r=\operatorname{DPP}(\mathbf{K})` is, almost surely, equal to :math:`\operatorname{rank}(K)`, so that the likelihood :eq:`eq:likelihood_projection_K` of :math:`S=\{s_1, \dots, s_r\}` reads

.. math::

	\mathbb{P}[\mathcal{X}=S]
	= \det \mathbf{K}_S.

Using the invariance by permutation of the derterminant it is sufficient to apply the chain rule to sample :math:`(s_1, \dots, s_r)` with joint distribution

.. math::

	\mathbb{P}[(s_1, \dots, s_r)]
	= \frac{1}{r!} \mathbb{P}[\mathcal{X}=\{s_1, \dots, s_r\}]
	= \frac{1}{r!} \det \mathbf{K}_S,

and forget about the sequential feature of the chain rule to get a valid sample :math:`\{s_1, \dots, s_r\} \sim \operatorname{DPP}(\mathbf{K})`.

Considering :math:`S=\{s_1, \dots, s_r\}` such that :math:`\mathbb{P}[\mathcal{X}=S] = \det \mathbf{K}_S > 0`, the following generic formulation of the chain rule

.. math::

	\mathbb{P}[(s_1, \dots, s_r)]
	= \mathbb{P}[s_1]
		\prod_{i=2}^{r}
			\mathbb{P}[s_{i} | s_{1:i-1}]

can be expressed as a telescopic ratio of determinants

.. math::
	:label: eq:chain_rule_K

	\mathbb{P}[(s_1, \dots, s_r)]
	= \dfrac{\mathbf{K}_{s_1,s_1}}{r}
		\prod_{i=2}^{r}
			\dfrac{1}{r-(i-1)}
		\frac{\det \mathbf{K}_{\{s_{1:i}\}}}
			 {\det \mathbf{K}_{\{s_{1:i-1}\}}}.

Using `Woodbury's formula <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_ the ratios of determinants in :eq:`eq:chain_rule_K` can be expanded into

.. math::
	:label: eq:chain_rule_schur

	\mathbb{P}[(s_1, \dots, s_r)]
	= \dfrac{\mathbf{K}_{s_1,s_1}}{r}
		\prod_{i=2}^{r}
			\dfrac{
				\mathbf{K}_{s_i s_i} - \mathbf{K}_{s_i, s_{1:i-1}} {\mathbf{K}_{\{s_{1:i-1}\}}}^{-1} \mathbf{K}_{s_{1:i-1}, s_1}
				}{r-(i-1)},

.. hint::

	MLers will recognize in :eq:`eq:chain_rule_schur` the incremental posterior variance of the Gaussian Process (GP) associated to :math:`\mathbf{K}`, see :cite:`RaWi06` Equation 2.26.

	.. caution::

		The connexion between the chain rule :eq:`eq:chain_rule_schur` and Gaussian Processes is valid in the case where the GP kernel is an **orthogonal projection kernel**, see also :ref:`finite_kdpps_exact_sampling_chain_rule_projection_kernel_caution`.

.. _finite_dpps_exact_sampling_projection_dpp_chain_rule_geometrical_interpretation:

Geometrical interpretation
==========================

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

		\mathbb{P}[(s_1, \dots, s_r)]
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

		\mathbb{P}[(s_1, \dots, s_r)]
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

**Step 1.** Draw independent Bernoulli random variables :math:`B_n \sim \operatorname{\mathcal{B}er}(\lambda_n)` for :math:`n=1,\dots, N` and collect :math:`\mathcal{B}=\left\{ n ~;~ B_n=1 \right\}`

.. _finite_dpps_exact_sampling_spectral_method_step_2:

**Step 2.** Sample from the **projection** DPP with correlation kernel :math:`U_{:\mathcal{B}} {U_{:\mathcal{B}}}^{\dagger} = \sum_{n\in \mathcal{B}} u_n u_n^{\dagger}`, see :ref:`the section above <finite_dpps_exact_sampling_projection_dpp_chain_rule_in_practice>`

.. note::

	**Step 1. ** selects a component of the mixture while
	**Step 2.** requires sampling from the corresponding **projection** DPP, cf. :ref:`finite_dpps_exact_sampling_projection_dpp_chain_rule`

In practice
===========

- Sampling *projection* :math:`\operatorname{DPP}(\mathbf{K})` from the eigendecomposition of :math:`\mathbf{K}=U U^{\dagger}` with :math:`U^{\dagger}U = I_{\operatorname{rank}(\mathbf{K})}`) was presented in :ref:`the section above <finite_dpps_exact_sampling_projection_dpp_chain_rule_in_practice>`

- Sampling :math:`\operatorname{DPP}(\mathbf{K})` from :math:`0_N \preceq\mathbf{K} \preceq I_N` can be done by following

  	**Step** 0. compute the eigendecomposition of :math:`\mathbf{K} = U \Lambda ^{\dagger}` in :math:`\mathcal{O}(N^3)`.

 	**Step** :ref:`1. <finite_dpps_exact_sampling_spectral_method_step_1>` draw independent Bernoulli random variables :math:`B_n \sim \operatorname{\mathcal{B}er}(\lambda_n)` for :math:`n=1,\dots, N` and collect :math:`\mathcal{B}=\left\{ n ~;~ B_n=1 \right\}`

	**Step** :ref:`2. <finite_dpps_exact_sampling_spectral_method_step_2>`
	sample from the **projection** DPP with correlation kernel defined by its eigenvectors :math:`U_{:, \mathcal{B}}`

	.. important::

		Step 0. must be performed once and for all in :math:`\mathcal{O}(N^3)`.
		Then the average cost of getting one sample by applying Steps 1. and 2. is :math:`\mathcal{O}(N \mathbb{E}\left[|\mathcal{X}|\right]^2)`, where :math:`\mathbb{E}\left[|\mathcal{X}|\right]=\operatorname{trace}(\mathbf{K})=\sum_{n=1}^{N} \lambda_n`.

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

  	**Step** 0. compute the eigendecomposition of :math:`\mathbf{L} = V \Gamma V^{\dagger}` in :math:`\mathcal{O}(N^3)`.

 	**Step** :ref:`1. <finite_dpps_exact_sampling_spectral_method_step_1>` is adapted to: draw independent Bernoulli random variables :math:`B_n \sim \operatorname{\mathcal{B}er}(\frac{\gamma_n}{1+\gamma_n})` for :math:`n=1,\dots, N` and collect :math:`\mathcal{B}=\left\{ n ~;~ B_n=1 \right\}`

	**Step** :ref:`2. <finite_dpps_exact_sampling_spectral_method_step_2>` is adapted to: sample from the **projection** DPP with correlation kernel defined by its eigenvectors :math:`V_{:,\mathcal{B}}`.

	.. important::

		Step 0. must be performed once and for all in :math:`\mathcal{O}(N^3)`.
		Then the average cost of getting one sample by applying Steps 1. and 2. is :math:`\mathcal{O}(N \mathbb{E}\left[|\mathcal{X}|\right]^2)`, where :math:`\mathbb{E}\left[|\mathcal{X}|\right]=\operatorname{trace}(\mathbf{L(I+L)^{-1}})=\sum_{n=1}^{N} \frac{\gamma_n}{1+\gamma_n}`

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

  	**Step** 0. compute the so-called *dual* kernel :math:`\tilde{L}=\Phi \Phi^{\dagger}\in \mathbb{R}^{d\times}` and eigendecompose it :math:`\tilde{\mathbf{L}} = W \Delta W^{\top}`.
  	This corresponds to a cost of order :math:`\mathcal{O}(Nd^2 + d^3)`.

 	**Step** :ref:`1. <finite_dpps_exact_sampling_spectral_method_step_1>` is adapted to: draw independent Bernoulli random variables :math:`B_i \sim \operatorname{\mathcal{B}er}(\delta_i)` for :math:`i=1,\dots, d` and collect :math:`\mathcal{B}=\left\{ i ~;~ B_i=1 \right\}`

	**Step** :ref:`2. <finite_dpps_exact_sampling_spectral_method_step_2>` is adapted to: sample from the **projection** DPP with correlation kernel defined by its eigenvectors :math:`\Phi^{\top} W_{:,\mathcal{B}} \Delta_{\mathcal{B}}^{-1/2}`.

	.. important::

		Step 0. must be performed once and for all in :math:`\mathcal{O}(Nd^2 + d^3)`.
		Then the average cost of getting one sample by applying Steps 1. and 2. is :math:`\mathcal{O}(N \mathbb{E}\left[|\mathcal{X}|\right]^2)`, where :math:`\mathbb{E}\left[|\mathcal{X}|\right]=\operatorname{trace}(\mathbf{L(I+L)^{-1}})=\sum_{i=1}^{d} \frac{\delta_i}{1+\delta_i}`

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

This method requires access to the correlation kernel :math:`\mathbf{K}` to perform a bottom-up chain rule on sets: starting from the empty set, each item in turn is decided to be added or excluded from the sample.
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

where each conditional probability can be written in closed formed using :eq:`eq:conditioned_on_S_in_X` and :eq:`eq:conditioned_on_S_notin_X`, namely

.. math::

	\mathbb{P}[T \subset \mathcal{X} \mid S \subset \mathcal{X}]
        &= \det\left[\mathbf{K}_T - \mathbf{K}_{TS} \mathbf{K}_S^{-1} \mathbf{K}_{ST}\right]\\
	\mathbb{P}[T \subset \mathcal{X} \mid S \cap \mathcal{X} = \emptyset]
    	&= \det\left[\mathbf{K}_T - \mathbf{K}_{TS} (\mathbf{K}_S - I)^{-1} \mathbf{K}_{ST}\right].

.. important::

	This quantities can be computed efficiently as they appear in the computation of the Cholesky-type :math:`LDL^{\dagger}` or :math:`LU` factorization of the correlation :math:`\mathbf{K}` kernel, in the Hermitian or non-Hermitian case, respectively.
	See :cite:`Pou19` for the details.

.. note::

	The sparsity of :math:`\mathbf{K}` can be leveraged to get derive faster samplers using the correspondence between the chain rule on sets and Cholesky-type factorizations, see e.g., :cite:`Pou19` Section 4.

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

:ref:`Recall <finite_dpps_definition_k_dpps>` :eq:`eq:likelihood_kDPP_L` that :math:`\operatorname{k-DPP}(\mathbf{L})` can be viewed as a :math:`\operatorname{DPP}(\mathbf{L})` constrained to a have fixed cardinality :math:`k \leq \operatorname{rank}(L)`.

To generate a sample of :math:`\operatorname{k-DPP}(\mathbf{L})`, one natural solution would be to use a rejection mechanism: draw :math:`S \sim \operatorname{DPP}(\mathbf{L})` and keep it only if :math:`|X| = k`.
However, the rejection constant may be pretty bad depending on the choice of :math:`k` regarding the distribution of the number of points :eq:`eq:number_of_points`.

The alternative solution was found by :cite:`KuTa12` Section 5.2.2.
The procedure relies on a slight modification of :ref:`Step 1. <finite_dpps_exact_sampling_spectral_method_step_1>` of the :ref:`finite_dpps_exact_sampling_spectral_method` which requires the computation of the `elementary symmetric polynomials <https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial>`_.

In practice
===========

Sampling :math:`\operatorname{k-DPP}(\mathbf{L})` from :math:`\mathbf{L} \succeq 0_N` can be done by following

	**Step 0.**
		a) compute the eigendecomposition of :math:`\mathbf{L} = V \Gamma V^{\dagger}` in :math:`\mathcal{O}(N^3)`
		b) evaluate the `elementary symmetric polynomials <https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial>`_ in the eigenvalues of :math:`\mathbf{L}`: :math:`E[l, n]:=e_l(\gamma, \dots, \gamma_n)` for :math:`l=0,\dots,k` and :math:`n=0,\dots,N`. These computations can done recursively using :cite:`KuTa12` Algorithm 8 in :math:`\mathcal{O}(N k^2)`.

	**Step** :ref:`1. <finite_dpps_exact_sampling_spectral_method_step_1>` is replaced by :cite:`KuTa12` Algorithm 8.

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

	**Step** :ref:`2. <finite_dpps_exact_sampling_spectral_method_step_2>` is adapted to: sample from the **projection** DPP with correlation kernel defined by its eigenvectors :math:`V_{:,\mathcal{B}}`, with a cost of order :math:`\mathcal{O}(N k^2)`.

.. important::

	Step 0. must be performed once and for all in :math:`\mathcal{O}(N^3 + Nk^2)`.
	Then the cost of getting one sample by applying Steps 1. and 2. is :math:`\mathcal{O}(N k^2)`

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
	- Step 0. requires :cite:`KuTa12` Algorithm 7 for the recursive evaluation of the elementary symmetric polynomials :math:`[e_l(\gamma_1, \dots, \gamma_n)]_{l=1, n=1}^{k, N}` in the eigenvalues of :math:`\mathbf{L}`
	- Step 1. calls :cite:`KuTa12` Algorithm 8 for selecting the eigenvectors for Step 2.

.. _finite_kdpps_exact_sampling_chain_rule_projection_kernel_caution:

Caution
=======

.. attention::

	Since the number of points :math:`k` is fixed, like for :ref:`projection DPPs <finite_dpps_properties_number_of_points_dpp_K_projection>`, it might be tempting to apply a chain rule similar to :eq:`eq:chain_rule_schur` for :math:`k` steps in order to sample :math:`\operatorname{k-DPP}(\mathbf{L})`.
	However, this is a misleading impression: the chain rule it cannot be applied blindly to sample general :math:`\operatorname{k-DPP}(\mathbf{L})`.
	Nevertheless, it is valid when :math:`\mathbf{L}` is an orthogonal projection kernel!

**Here are the reasons why**

1. First keep in mind that, the ultimate goal is to draw a **subset** :math:`S=\{ s_{1}, \dots, s_{k} \} \sim \operatorname{k-DPP}(\mathbf{L})` with probability :eq:`eq:likelihood_kDPP_L`

.. math::
	:label: eq:caution_likelihood_kDPP_L

	\mathbb{P}[\mathcal{X}=S]
	= \frac{1}{e_k(\mathbf{L})} \det \mathbf{L}_S ~ 1_{|S|=k}

2. Now, if we were to use a chain rule of the form :eq:`eq:chain_rule_schur` this would correspond to sampling sequentially the items :math:`s_1, \dots, s_{k}`, so that the resulting **vector** :math:`(s_{1}, \dots, s_{k})` has probability

.. math::
	:label: eq:chain_rule_caution_vector

	\mathbb{Q}[(s_{1}, \dots, s_{k})]
	&= \dfrac{\mathbf{L}_{s_1,s_1}}{Z_1}
		\prod_{i=2}^{r}
			\dfrac{
				\mathbf{L}_{s_i s_i} - \mathbf{L}_{s_i, s_{1:i-1}} {\mathbf{L}_{\{s_{1:i-1}\}}}^{-1} \mathbf{L}_{s_{1:i-1}, s_1}
				}{Z_i(s_{1}, \dots, s_{i-1})}\\
	&= \frac{1}{Z(s_{1}, \dots, s_{k})} \det \mathbf{L}_S.

Contrary to :math:`Z_1=\operatorname{trace}(\mathbf{L})`, the normalizations :math:`Z_i(s_{1}, \dots, s_{i-1})` of the successive conditionals depend, *a priori*, on the order :math:`s_{1}, \dots, s_{k}` were selected. For this reason we denote the global normalization constant :math:`Z(s_{1}, \dots, s_{k})`.

.. warning::

	Equation :eq:`eq:chain_rule_caution_vector` suggests that, the sequential feature of the chain rule matters, *a priori*; the distribution of :math:`\left(s_{1}, \dots, s_{k} \right)` is not `exchangeable <https://en.wikipedia.org/wiki/Exchangeable_random_variables>`_ *a priori*, i.e., it is not invariant to permutations of its coordinates.
	This fact, would only come from the normalization :math:`Z(s_{1}, \dots, s_{k})`, since :math:`\mathbf{L}_S` is invariant by permutation.

	.. note::

		To see this, let's compute the normalization constant :math:`Z_i(s_1, \dots, s_{i-1})` in :eq:`eq:chain_rule_caution_vector` for a generic :math:`\mathbf{L}\succeq 0_N` factored as :math:`\mathbf{L} = VV^{\dagger}`, with no specific assumption on :math:`V`.
		Further denote :math:`S_{i-1}=\{s_{1}, \dots, s_{i-1}\}`, so that

		.. math::
			:label: eq:chain_rule_caution_normalization_constant_conditional

			Z_i(s_1, \dots, s_{i-1})
			&= \sum_{i=1}^N \mathbf{L}_{ii}
				- \mathbf{L}_{iS_{i-1}} \left[\mathbf{\mathbf{L}}_{S_{i-1}}\right]^{-1} \mathbf{L}_{S_{i-1}i}\\
			&= \operatorname{trace}(
			    \mathbf{L}
			    - \mathbf{L}_{:, S_{i-1}}
			      \left[\mathbf{\mathbf{L}}_{S_{i-1}}\right]^{-1} \mathbf{L}_{S_{i-1}, :}
			    )\\
			&=  \operatorname{trace}\left(
			    \mathbf{L}
			    - V {V^{\dagger}}_{:,S_{i-1}}
			      \left[V_{S_{i-1},:} {V^{\dagger}}_{:,S_{i-1}}\right]^{-1}
			      V_{S_{i-1},:} V^{\dagger}
			    \right)\\
			&= 	\operatorname{trace}
				\big(
				\mathbf{L}_{ii}
				-
				\underbrace{{V_{S_{i-1}:}}^{\dagger}
				 			\left[V_{S_{i-1}:} {V_{S_{i-1}:}}^{\dagger}\right]^{-1}
				 			V_{S_{i-1}:}}_{\Pi_{V_{S_{i-1}:}}}
				V^{\dagger}V
				\big)\\
			&= \operatorname{trace}(\mathbf{L})
			  - \operatorname{trace}(\Pi_{V_{S_{i-1}:}}V^{\dagger}V)

		where :math:`\Pi_{V_{S_{i-1}:}}` denotes the `orthogonal projection <https://en.wikipedia.org/wiki/Proofs_involving_the_Moore–Penrose_inverse#Projectors_and_subspaces>`_ onto :math:`\operatorname{Span}\{V_{s_1,:}, \dots, V_{s_i-1, :}\}`, the supspace spanned the feature vectors associated to :math:`s_{1}, \dots, s_{i-1}`.

Then, summing :eq:`eq:chain_rule_caution_vector` over the :math:`k!` permutations of :math:`1, \dots, k`, yields the probability of drawing the **subset** :math:`S=\left\{ s_{1}, \dots, s_{k} \right\}`, namely

.. math::
	:label: eq:chain_rule_caution_set

	\mathbb{Q}[\{ s_{1}, \dots, s_{k} \}]
	= \sum_{\sigma \in \mathfrak{S}_k}
		\mathbb{Q}[(s_{\sigma(1)}, \dots, s_{\sigma(k)})]
 	= \det\mathbf{L}_S
		\underbrace{
			\sum_{\sigma \in \mathfrak{S}_k}
			\frac{1}{Z(s_{\sigma(1)}, \dots, s_{\sigma(k)})}
			}_{
			1/Z_S
			}.

3. For the chain rule :eq:`eq:chain_rule_caution_vector` to be a valid procedure for sampling :math:`\operatorname{k-DPP}(\mathbf{L})`, we must be able to identify :eq:`eq:caution_likelihood_kDPP_L` and :eq:`eq:chain_rule_caution_set`, i.e., :math:`\mathbb{Q}[S] = \mathbb{P}[S]` for all :math:`|S|=k`, or equivalently :math:`Z_S = e_k(L)` for all :math:`|S|=k`.

.. important::

	A sufficient condition (very likely to be necessary) is that the joint distribution of :math:`(s_{1}, \dots, s_{k})`, generated by the chain rule mechanism :eq:`eq:chain_rule_caution_vector` is `exchangeable <https://en.wikipedia.org/wiki/Exchangeable_random_variables>`_ (invariant to permutations of the coordinates).
	In that case, the normalization in :eq:`eq:chain_rule_caution_vector` would then be constant :math:`Z(s_{1}, \dots, s_{k})=Z` .
	So that :math:`Z_S` would in fact play the role of the normalization constant of :eq:`eq:chain_rule_caution_set`, since it would be constant as well and equal to :math:`Z_S = \frac{Z}{k!}`.
	Finally, :math:`Z_S = e_k(L)` by identification of :eq:`eq:caution_likelihood_kDPP_L` and :eq:`eq:chain_rule_caution_set`.

**This is what we can prove in the particular case where** :math:`\mathbf{L}` **is an orthogonal projection matrix.**

To do this, denote :math:`r=\operatorname{rank}(\mathbf{L})` and recall that in this case :math:`\mathbf{L}` satisfies :math:`\mathbf{L}^2=\mathbf{L}` and :math:`\mathbf{L}^{\dagger}=\mathbf{L}`, so that it can be factored as :math:`\mathbf{L}=\Pi_{\mathbf{L}}=\mathbf{L}^{\dagger}\mathbf{L}=\mathbf{L}\mathbf{L}^{\dagger}`

Finally, we can plug :math:`V=\mathbf{L}` in :eq:`eq:chain_rule_caution_normalization_constant_conditional` to obtain

.. math::

	Z_i(s_1, \dots, s_{i-1})
	&= \operatorname{trace}(\mathbf{L})
	  - \operatorname{trace}(\Pi_{\mathbf{L}_{S_{i-1}:}}\mathbf{L}^{\dagger}\mathbf{L})\\
	&= \operatorname{trace}(\Pi_{\mathbf{L}})
	  - \operatorname{trace}(\Pi_{\mathbf{L}_{S_{i-1}:}}\Pi_{\mathbf{L}})\\
	&= \operatorname{trace}(\Pi_{\mathbf{L}})
	  - \operatorname{trace}(\Pi_{\mathbf{L}_{S_{i-1}:}})\\
	&= \operatorname{rank}(\Pi_{\mathbf{L}})
	  - \operatorname{rank}(\Pi_{\mathbf{L}_{S_{i-1}:}})\\
	&= r - (i - 1) := Z_i

Thus, the normalization :math:`Z(s_1, \dots, s_k)` in :eq:`eq:chain_rule_caution_normalization_constant_conditional` is constant as well equal to

.. math::

	Z(s_1, \dots, s_k)
	= \prod_{i=1}^{k} Z_i
	= \prod_{i=1}^{k} r - (i - 1)
	= \frac{r!}{(r-k)!}
	= k! {r \choose k}
	= k! e_k(\mathbf{L})
	:= Z

where the last equality is a simple computation of the `elementary symmetric polynomial <https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial>`_

.. math::

	e_k(\mathbf{L})
	= e_k(\gamma_{1:r}=1, \gamma_{r+1:N}=0)
	= \sum_{\substack{S \subset [N]\\|S|=k}} \prod_{s\in S} \gamma_{s}
	= {r \choose k}

.. important::

	This shows that, when :math:`\mathbf{L}` is an orthogonal projection matrix, the order the items :math:`s_1, \dots, s_r` we selected by the chain rule :eq:`eq:chain_rule_caution_vector` can be forgotten, so that :math:`\{s_1, \dots, s_r\}` can be considered as valid sample of :math:`\operatorname{k-DPP}(\mathbf{L})`.

.. code-block:: python

	# For our toy example, this sub-optimized implementation is enough
	# to illustrate that the chain rule applied to sample k-DPP(L)
	# draws s_1, ..., s_k sequentially, with joint probability
	# P[(s_1, ..., s_k)] = det L_S / Z(s_1, ..., s_k)
	#
	# 1. is exchangeable when L is an orthogonal projection matrix
	#    P[(s1, s2)] = P[(s_2, s_1)]
	# 2. is a priori NOT exchangeable for a generic L >= 0
	#    P[(s1, s2)] /= P[(s_2, s_1)]

	import numpy as np
	import scipy.linalg as LA
	from itertools import combinations, permutations

	k, N = 2, 4
	potential_samples = list(combinations(range(N), k))

	rank_L = 3

	rng = np.random.RandomState(1)

	eig_vecs, _ = LA.qr(rng.randn(N, rank_L), mode='economic')

	for projection in [True, False]:

	    eig_vals = 1.0 + (0.0 if projection else 2 * rng.rand(rank_L))
	    L = (eig_vecs * eig_vals).dot(eig_vecs.T)

	    proba = np.zeros((N, N))
	    Z_1 = np.trace(L)

	    for S in potential_samples:

	        for s in permutations(S):

	            proba[s] = LA.det(L[np.ix_(s, s)])

	            Z_2_s0 = np.trace(L - L[:, s[:1]].dot(LA.inv(L[np.ix_(s[:1], s[:1])])).dot(L[s[:1], :]))

	            proba[s] /= Z_1 * Z_2_s0

	    print('L is {}projection'.format('' if projection else 'NOT '))

	    print('P[s0, s1]', proba, sep='\n')
	    print('P[s0]', proba.sum(axis=0), sep='\n')
	    print('P[s1]', proba.sum(axis=1), sep='\n')

	    print(proba.sum(), '\n' if projection else '')

.. code-block:: python

    L is projection
    P[s0, s1]
    [[0.         0.09085976 0.01298634 0.10338529]
     [0.09085976 0.         0.06328138 0.15368033]
     [0.01298634 0.06328138 0.         0.07580691]
     [0.10338529 0.15368033 0.07580691 0.        ]]
    P[s0]
    [0.20723139 0.30782147 0.15207463 0.33287252]
    P[s1]
    [0.20723139 0.30782147 0.15207463 0.33287252]
    1.0000000000000002

    L is NOT projection
    P[s0, s1]
    [[0.         0.09986722 0.01463696 0.08942385]
     [0.11660371 0.         0.08062998 0.20535251]
     [0.01222959 0.05769901 0.         0.04170435]
     [0.07995922 0.15726273 0.04463087 0.        ]]
    P[s0]
    [0.20879253 0.31482896 0.13989781 0.33648071]
    P[s1]
    [0.20392803 0.4025862  0.11163295 0.28185282]
    1.0
