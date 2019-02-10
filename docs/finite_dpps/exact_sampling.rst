.. _finite_dpps_exact_sampling:

Exact sampling
**************

The procedure stems from the fact that :ref:`generic DPPs are mixtures of projection DPPs <finite_dpps_mixture>`, suggesting the following two steps algorithm given the spectral decomposition of the inclusion kernel :math:`\mathbf{K}`

.. math::

	\mathbf{K} = \sum_{n=1}^{N} \lambda_n u_n u_n^{\dagger}

1. Draw independent :math:`\operatorname{\mathcal{B}er}(\lambda_n)` for each eigenvector :math:`u_n` and store the selected ones in :math:`\tilde{U}`.
2. Sample from the corresponding *projection* :math:`\operatorname{DPP}(\tilde{U}\tilde{U}^{\top})`.

:cite:`HKPV06` Algorithm 18 gives the procedure for sampling *projection* DPPs. It is based on the chain rule and the geometrical interpretations are reflected through the conditionals.

In the general case, the average cost of the exact sampling scheme is :math:`\mathcal{O}(N\mathbb{E}[|\mathcal{X}|]^2)` with an initial :math:`\mathcal{O}(N^3)` cost to access the spectral decomposition of the underlying inclusion kernel.

.. important::

	Sampling from a *projection* :math:`\operatorname{DPP}(\mathbf{K})` can be done in :math:`\mathcal{O}(Nr^2)` with :math:`r=\operatorname{rank}(\mathbf{K})`. It is worth mentioning that to sample from a *projection* DPP:

	As we shall see next in :ref:`finite_dpps_exact_sampling_projection_dpps`:

	- Given the projection kernel :math:`\mathbf{K}` there is no need to compute its eigenvectors
	- Given some orthonormal vectors stacked in :math:`\tilde{U}` there is no need to compute :math:`\mathbf{K}=\tilde{U}\tilde{U}^{\top}`

.. _finite_dpps_exact_sampling_projection_dpps:

Projection DPPs
===============

	.. important::

		*Orthogonal projection* inclusion kernel :math:`\mathbf{K}` admit the following Gram matrix factorizations

		1. Using :math:`\mathbf{K} = \mathbf{K}^2` 
		and :math:`\mathbf{K}^{\dagger}=\mathbf{K}`

			.. math::
				:label: inclusion_kernel_factorization_K.TK

				\mathbf{K} 
				= \mathbf{K} \mathbf{K}^{\dagger}
				= \mathbf{K}^{\dagger} \mathbf{K},

		2. Using the spectral decomposition

			.. math::
				:label: inclusion_kernel_factorization_UU.T

				\mathbf{K} 
				= \mathbf{U} \mathbf{U}^{\dagger}, 
				\quad \text{where } \mathbf{U}^{\dagger} \mathbf{U} = I_r

		In this setting, the rows (equiv. columns) of the inclusion kernel :math:`\mathbf{K}` or alternatively the rows of the eigenvectors :math:`\mathbf{U}` play the role of feature vectors.

		Finally, we will see that the chain rule is akin to Gram-Schmidt orthogonalization of these somewhat artificial feature vectors.

	.. _finite_dpps_exact_sampling_chain_rule:

Chain rule
----------

	Let :math:`S=\{s_1, \dots, s_r\}` with :math:`r=\operatorname{rank}(\mathbf{K})`, equation :eq:`number_points_projection_K` yields 

	.. math::

		\mathbb{P}[\mathcal{X}=S] 
		= \det \mathbf{K}_S
		
	The invariance by transposition and permutation of the determinant allows to express the joint probability distribution of :math:`(s_1, \dots, s_r)` as 

	.. math:: 

		\mathbb{P}[s_1, \dots, s_r] 
		= \frac{1}{r!} \mathbb{P}[\mathcal{X}=S] 
		= \frac{1}{r!} \det \mathbf{K}_S

	As announced, the exact sampling scheme relies on the chain rule

	.. math:: 
		:label: chain_rule
	
		\mathbb{P}[s_1, \dots, s_r] 
		= \mathbb{P}[s_1] \prod_{j=2}^{r} \mathbb{P}[s_{j} | s_{1:j-1}]

	.. note::

		Once the chain rule performed, one can forget the order the points :math:`s_1,\dots,s_r` were drawn and take :math:`S=\{s_1, \dots, s_r\}` as a valid sample of :math:`\operatorname{DPP}(\mathbf{K})`.

	To proceed further, we need the conditionals involved in :eq:`chain_rule`

	.. math::
		:label: chain_rule_K

		\mathbb{P}[s_1] 
		&= \dfrac{1}{r} \mathbf{K}_{s_1s_1}\\
		\mathbb{P}[s_{j} | s_{1:j-1}]
		&= \dfrac{1}{r-(j-1)} 
		\frac{\det \mathbf{K}_{\{s_{1:j}\}}}{\det \mathbf{K}_{\{s_{1:j-1}\}}}, 
		\qquad \forall 2\leq j \leq r

	.. hint::

		The geometrical perspective may ease the interpretation as well as practical implementation of such conditionals. Note that Woodbury's formula can also be applied to evaluate recursively the ratio of determinants.

		In fact, the chain rule boils down to applying Gram-Schmidt on the rows of either :math:`\mathbf{K}` or :math:`\mathbf{U}`.

		- Given the *orthogonal projection* kernel :eq:`inclusion_kernel_factorization_K.TK` the sampling scheme writes

			.. math::

				\mathbb{P}[s_1] 
				&= \dfrac{1}{r} \mathbf{K}_{s_1s_1}\\
				\mathbb{P}[s_{j} | s_{1:j-1}]
				&= \dfrac{1}{r-(j-1)} 
				\operatorname{dist}^2 
				(\mathbf{K}_{s_{j}:} ~;~ \operatorname{Span} \mathbf{K}_{s_{1:j-1}:})

		- Given the eigendecomposition :math:`\mathbf{K}=\mathbf{U}\mathbf{U}^{\dagger}` of the *orthogonal projection* kernel :eq:`inclusion_kernel_factorization_UU.T` the sampling scheme writes

			.. math::
				:label: phase_2_eig_vec

				\mathbb{P}[s_1] 
				&= \dfrac{1}{r} \| \mathbf{U}_{s_1:} \|^2\\
				\mathbb{P}[s_{j} | s_{1:j-1}]
				&= \dfrac{1}{r-(j-1)} 
				\operatorname{dist}^2 
				(\mathbf{U}_{s_{j}:} ~;~ \operatorname{Span} \mathbf{U}_{s_{1:j-1}:})


		Finally, sampling from a projection :math:`\operatorname{DPP}(\mathbf{K})` can be performed in :math:`\mathcal{O}(N r^2)`.

	.. testcode::

		from numpy import ones
		from numpy.random import seed, randn
		from scipy.linalg import qr
		from dppy.finite_dpps import FiniteDPP

		seed(1)

		r, N = 4, 10
		eig_vals = ones(r)
		eig_vecs, _ = qr(randn(N, r), mode='economic')

		DPP = FiniteDPP('inclusion', **{'K_eig_dec':(eig_vals, eig_vecs)})

		for _ in range(10):
			DPP.sample_exact()

		print(list(map(list, DPP.list_of_samples)))
	
	.. testoutput::

		[[0, 4, 8, 2], [1, 8, 2, 0], [8, 3, 6, 1], [6, 7, 1, 9], [9, 3, 0, 4], [9, 4, 0, 8], [9, 6, 1, 8], [0, 1, 2, 7], [1, 2, 8, 9], [8, 2, 9, 4]]

	.. seealso::

		.. currentmodule:: dppy.finite_dpps

		- :py:meth:`~FiniteDPP.sample_exact`
		- :cite:`HKPV06` Algorithm 18 and Proposition 19, for the original idea
		- :cite:`KuTa12` Algorithm 1, for a first interpretation of :cite:`HKPV06` algorithm running in :math:`\mathcal{O}(N r^3)`
		- :cite:`Gil14` Algorithm 2, for the :math:`\mathcal{O}(N r^2)` implementation
		- :cite:`TrBaAm18` Algorithm 3, for a technical report on DPP sampling
		- :cite:`LaGaDe18` for a different perspective on exact sampling using Cholesky decomposition instead of the spectral decomposition
		- :ref:`UST`

	.. _finite_dpps_exact_sampling_caution:

Caution
-------

	.. attention::

		For the chain rule as described in :eq:`chain_rule_K` to be valid, it is **crucial** that :math:`\mathbf{K}` is a *projection* kernel.
		It is the very reason why the normalization constants of the conditionals  are independent of the previous points and that :math:`S=\{s_1, \dots, s_r\}` is a valid sample of :math:`\operatorname{DPP}(\mathbf{K})`.

		To see this, consider :math:`\mathbf{K}` satisfying :eq:`suff_cond_K` with Gram factorization :math:`\mathbf{K} = VV^{\dagger}` and denote :math:`Y=\{s_1, \dots, s_{j-1}\}`.
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

		However, for :math:`V = \mathbf{K}` or :math:`\mathbf{U}`, we have

		.. math::

			&\qquad\operatorname{Tr}(\mathbf{K})
			&\qquad\operatorname{Tr}(\mathbf{K}) 
				- \operatorname{Tr}(\Pi_{\mathbf{K}_{Y:}}\mathbf{K}\mathbf{K}^{\dagger})
			&\qquad 
			\operatorname{Tr}(\mathbf{K}) 
				- \operatorname{Tr}(\Pi_{\mathbf{U}_{Y:}}\mathbf{U}^{\dagger}\mathbf{U})
				\\
			&\qquad= \operatorname{rank}(\mathbf{K}) 
			&\qquad= r - \operatorname{Tr}(\Pi_{\mathbf{K}_{Y:}}\mathbf{K})
			&\qquad= r - \operatorname{Tr}(\Pi_{\mathbf{U}_{Y:}}I_r)
				\\
			&\qquad= r
			&\qquad= r - \operatorname{Tr}(\Pi_{\mathbf{K}_{Y:}})
			&\qquad= r - \operatorname{Tr}(\Pi_{\mathbf{U}_{Y:}})
				\\
			&
			&\qquad= r - |Y|
			&\qquad= r - |Y|

.. _finite_dpps_exact_sampling_generic_dpps:

Generic DPPs
============

	When considering non-projection DPPs, the eigendecomposition of the underlying kernel is required; adding an initial extra :math:`\mathcal{O}(N^3)` cost to sampling a *projection DPP*

	.. tip::

		If the marginal kernel was constructed as :math:`\mathbf{L}=\Phi^{\dagger}\Phi` where :math:`\Phi` is a :math:`d\times N` feature matrix, it may be judicious to exploit the lower dimensional structure of the *dual* kernel :math:`\tilde{\mathbf{L}} = \Phi \Phi^{\dagger}`.
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

	**Phase 2** Conditionally on :math:`(B_n)` set :math:`\mathcal{B} = \{ n ~;~ B_n = 1 \}` and apply the chain rule :eq:`phase_2_eig_vec` with 

		.. math::

			r = |\mathcal{B}|
			\quad \text{and} \quad
			U =
				U_{:\mathcal{B}}, \ 
				V_{:\mathcal{B}}, \
				\Phi^{\top} W_{:\mathcal{B}} \Gamma_{:\mathcal{B}}^{-1/2} \ 
			\text{respectively}

	.. testcode::

		from numpy.random import seed, rand, randn
		from scipy.linalg import qr
		from dppy.finite_dpps import FiniteDPP

		seed(1)

		r, N = 5, 10
		eig_vals = rand(r)
		eig_vecs, _ = qr(randn(N, r), mode='economic')

		DPP = FiniteDPP('inclusion', **{'K_eig_dec':(eig_vals, eig_vecs)})

		for _ in range(10):
			DPP.sample_exact()

		print(list(map(list, DPP.list_of_samples)))
	
	.. testoutput::

		[[7], [4], [3, 4], [4, 2, 3], [9, 3], [0], [1], [4, 7], [0, 6], [4]]

.. seealso::

	.. currentmodule:: dppy.finite_dpps

	:py:meth:`~FiniteDPP.sample_exact`


.. _finite_dpps_exact_sampling_k_dpps:

k-DPPs
======

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

	from numpy.random import seed, rand, randn
	from scipy.linalg import qr
	from dppy.finite_dpps import FiniteDPP

	seed(1)

	r, N = 5, 10
	# Random feature vectors
	Phi = randn(r, N)
	DPP = FiniteDPP('marginal', **{'L': Phi.T.dot(Phi)})

	k = 4
	for _ in range(10):
	    DPP.sample_exact_k_dpp(size=k)

	print(list(map(list, DPP.list_of_samples)))

.. testoutput::

	[[1, 8, 5, 7], [3, 8, 5, 9], [5, 3, 1, 8], [5, 8, 2, 9], [1, 2, 9, 6], [1, 0, 2, 3], [7, 0, 3, 5], [8, 3, 7, 6], [0, 2, 3, 7], [1, 3, 7, 5]]

.. seealso::

	.. currentmodule:: dppy.finite_dpps

	- :py:meth:`~FiniteDPP.sample_exact_k_dpps`
	- :cite:`KuTa12` Algorithm 7 for the recursive evaluation of the elementary symmetric polynomials :math:`[e_l(\lambda_1, \dots, \delta_n)]_{l=1, n=1}^{k, N}` in the eigenvalues of :math:`\mathbf{L}`
