.. _discrete_dpps_exact_sampling:

Exact sampling
**************

The procedure stems from the fact that :ref:`discrete_dpps_mixture`, suggesting the following two steps algorithm given the spectral decomposition of the inclusion kernel :math:`\mathbf{K}`

.. math::

	\mathbf{K} = \sum_{n=1}^{N} \lambda_n u_n u_n^{\dagger}

1. Subsample the set of eigenvectors by drawing independent Bernoulli variables :math:`\mathcal{B}(\lambda_n)` and store the selected vectors in :math:`\tilde{U}`.

2. Sample from the corresponding *projection* :math:`\operatorname{DPP}(\tilde{U}\tilde{U}^{\top})`.

:cite:`HKPV06` Algorithm 18 gives the procedure for sampling *projection* DPPs. It is based on the chain rule and the geometrical interpretations are reflected through the conditionals.

In the general case, the average cost of the exact sampling scheme is :math:`\mathcal{O}(N\mathbb{E}[|\mathcal{X}|]^2)` with an initial :math:`\mathcal{O}(N^3)` cost to access the spectral decomposition of the underlying inclusion kernel.

.. important::

	Sampling from a *projection* :math:`\operatorname{DPP}(\mathbf{K})` can be done in :math:`\mathcal{O}(Nr^2)` with :math:`r=\operatorname{rank}(\mathbf{K})`. It is worth mentioning that to sample from a *projection* DPP:

	- Given the projection kernel :math:`\mathbf{K}` there is no need to compute its eigenvectors
	- Given some orthonormal vectors stacked in :math:`\tilde{U}` there is no need to compute :math:`\mathbf{K}=\tilde{U}\tilde{U}^{\top}`

.. _discrete_dpps_exact_sampling_projection_dpps:

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

	**Chain rule**

	Let :math:`S=\{s_1, \dots, s_r\}` with :math:`r=\operatorname{rank}(K)`, equation :eq:`number_points_projection_K` yields 

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

	.. attention::

		The fact that :math:`\mathbf{K}` is a *projection* kernel is **crucial**.
		It is the very reason why the normalization constants of the conditionals in :eq:`chain_rule` are independent of the previous points and that :math:`S=\{s_1, \dots, s_r\}` is a sample of :math:`\operatorname{DPP}(\mathbf{K})`.

		Consider :math:`\mathbf{K}` satisfying :eq:`suff_cond_K` with Gram factorization :math:`\mathbf{K} = VV^{\dagger}` and set :math:`Y=\{s_1, \dots, s_{j-1}\}`.
		Without prior asumption on :math:`V`, the Schur complement formula allows to express the conditionals in :eq:`chain_rule_K` as

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


	.. seealso::

		- :cite:`HKPV06` Algorithm 18 and Proposition 19, for the original idea
		- :cite:`KuTa12` Algorithm 1, for a first interpretation of :cite:`HKPV06` algorithm running in :math:`\mathcal{O}(N r^3)`
		- :cite:`Gil14` Algorithm 2, for the :math:`\mathcal{O}(N r^2)` implementation
		- :cite:`TrBaAm18` Algorithm 3, for a technical report on DPP sampling

		.. todo::

			- Refer to code also
			- Equivalence with Cholesky updates? 

.. _discrete_dpps_exact_sampling_generic_dpps:

Generic DPPs
============

	**Generic DPPs are mixtures of projection DPPs**

	When considering non-projection DPPs, the eigendecomposition of the underlying kernel is required; adding an initial extra :math:`\mathcal{O}(N^3)` cost to sampling a *projection DPP*

	.. tip::

		If the marginal kernel was constructed as :math:`\mathbf{L}=\Phi^{\dagger}\Phi` where :math:`\Phi` is a :math:`d\times N` feature matrix, it may be judicious to exploit the lower dimensional structure of the *dual* kernel :math:`\tilde{\mathbf{L}} = \Phi \Phi^{\dagger}`.

	.. note::

		Noting the respective spectral decompositions

		.. math::

			\mathbf{K} = U \Lambda U^{\top},
			\quad \mathbf{L} = V \Delta V^{\top}
			\quad \text{and} \quad
			\tilde{\mathbf{L}} = W \Gamma W^{\top}

		we have,

		.. math::

			\Lambda = \Delta (I+\Delta)^{-1}
			\quad \text{and} \quad
			U = V

		and with an abuse of notation, considering only the non-zero eigenvalues (and corresponding eigenvectors)

		.. math::

			\Delta = \Gamma
			\quad \text{and} \quad
			U = V = \Phi^{\top} W \Gamma^{-1/2}

	In the generic setting, the exact sampling scheme works as a two steps algorithm:

	**Phase 1** Draw independent Bernoulli variables :math:`(B_n)` with parameters the eigenvalues:

		- :math:`(\lambda_n)_{1:N}` of the inclusion kernel :math:`\mathbf{K}`,
		- :math:`(\delta_n)_{1:N}` of the marginal kernel :math:`\mathbf{L}`,
		- :math:`(\gamma_n)_{1:d}` of the (marginal) dual :math:`\tilde{\mathbf{L}}`, respectively.

	**Phase 2** Conditionally on :math:`(B_n)` set :math:`\mathcal{B} = \{ n ~;~ B_n = 1 \}` and apply :eq:`phase_2_eig_vec` with 

		- :math:`r=|\mathcal{B}|`

	and 

		- :math:`U=U_{:\mathcal{B}}`,
		- :math:`U=V_{:\mathcal{B}}`,
		- :math:`\Phi^{\top} W_{:\mathcal{B}} \Gamma_{:\mathcal{B}}^{-1/2}`, respectively.

.. seealso::
	.. currentmodule:: discrete_dpps

	:func:`Discrete_DPP.sample_exact <Discrete_DPP.sample_exact>`