.. _discrete_dpps_properties:

Properties
==========

Geometrical insights
~~~~~~~~~~~~~~~~~~~~

	Kernels satisfying the sufficient conditions :eq:`suff_cond_K` and :eq:`suff_cond_L` can be expressed as

	.. math::

		K_{ij} = \langle \phi_i, \phi_j \rangle
		\quad \text{and} \quad
		L_{ij} = \langle \psi_i, \psi_j \rangle,

	where each item is represented by a feature vector :math:`\phi_i` (resp. :math:`\psi_i`).

	The geometrical view is then straightforward.

	a. The inclusion probabilities interpret as

		.. math::

			\mathbb{P}[S\subset \mathcal{X}] 
			= \det \mathbf{K}_S
			= \operatorname{Vol}^2 \{\phi_s\}_{s\in S}

	b. The inclusion probabilities interpret as

		.. math::

			\mathbb{P}[\mathcal{X} = S] 
			\propto \det \mathbf{L}_S
			= \operatorname{Vol}^2 \{\psi_s\}_{s\in S}
		
	That is to say, DPPs favor subsets :math:`S` whose corresponding feature vectors span a large volume i.e. *DPPs sample softened orthogonal bases*.

Diversity
~~~~~~~~~

	The *determinantal* structure of DPPs encodes the notion of diversity.
	Deriving the pair inclusion probability, also called the 2-point correlation function using :eq:`inclusion_proba`, we obtain
	
	.. math::
		
		\mathbb{P}[\{i, j\} \subset \mathcal{X}]
	  &= \begin{vmatrix}
	    \mathbb{P}[i \in \mathcal{X}]	& \mathbf{K}_{i j}\\
	    \overline{\mathbf{K}_{i j}}		& \mathbb{P}[j \in \mathcal{X}]
	  \end{vmatrix}\\
	  &= \mathbb{P}[i \in \mathcal{X}] \mathbb{P}[j \in \mathcal{X}] 
	  	- |\mathbf{K}_{i j}|^2

	That is, the greater the similarity :math:`|\mathbf{K}_{i j}|` between items :math:`i` and :math:`j`, the less likely they co-occur in the samples.

Relation between inclusion and marginal kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	.. math::
		:label: relation_K_L

		\mathbf{K} = \mathbf{L}(I+\mathbf{L})^{—1} 
			\qquad \text{and} \qquad 
		\mathbf{L} = \mathbf{K}(I-\mathbf{K})^{—1}

	.. warning::
		
		For DPPs with *projection* inclusion kernel :math:`K`, the marginal kernel :math:`\mathbf{L}` cannot be computed via  :eq:`relation_K_L` with :math:`\mathbf{L} = \mathbf{K}(I-\mathbf{K})^{—1}`, since :math:`\mathbf{K}` has at least one eigenvalue equal to :math:`1` (:math:`K^2=K`).
		However, the marginal kernel :math:`\mathbf{L}` coincides with :math:`\mathbf{K}`.

		.. math::

			\mathbb{P}[\mathcal{X}=S] = 
				\det \mathbf{K}_S 1_{|S|=\operatorname{rank}\mathbf{K}}
				\quad \forall S\subset [N]

	Thus, except for inclusion kernels :math:`\mathbf{K}` with some eigenvalues equal to :math:`1`, both :math:`\mathbf{K}` and :math:`\mathbf{L}` are diagonalizable in the same basis

	.. math::

		\mathbf{K} = U \Lambda^{\mathbf{K}} U^{\top}
			\qquad \text{and} \qquad
		\mathbf{L} = U \Lambda^{\mathbf{L}} U^{\top}

Number of points 
~~~~~~~~~~~~~~~~

	.. math::
		:label: number_points

		|\mathcal{X}|
			= \sum_{n=1}^N 
				\operatorname{\mathcal{B}er}
				\left(
					\lambda_n^{\mathbf{K}}
				\right)
			= \sum_{n=1}^N 
				\operatorname{\mathcal{B}er}
				\left(
					\frac{\lambda_n^{\mathbf{L}}}{1+\lambda_n^{\mathbf{L}}}
				\right)

	.. hint::

		Keep in mind the first key equality :math:`|\mathcal{X}| = \sum_{n=1}^N \operatorname{\mathcal{B}er} \left(\lambda_n^{\mathbf{K}}\right)` to understand the :ref:`discrete_dpps_exact_sampling`.
	
	a. Expectation

	.. math::
		:label: expect_number_points

		\mathbb{E}[|\mathcal{X}|] 
			= \operatorname{Tr} \mathbf{K}
			= \sum_{n=1}^N \lambda_n^{\mathbf{K}}
			= \sum_{n=1}^N \frac{\lambda_n^{\mathbf{L}}}{1+\lambda_n^{\mathbf{L}}}

	b. Variance

	.. math::
		:label: var_number_points

		\operatorname{\mathbb{V}ar}[|\mathcal{X}|] 
			= \operatorname{Tr} \mathbf{K} - \operatorname{Tr} \mathbf{K}^2
			= \sum_{n=1}^N \lambda_n^{\mathbf{K}}(1-\lambda_n^{\mathbf{K}})
			= \sum_{n=1}^N \frac{\lambda_n^{\mathbf{L}}}{(1+\lambda_n^{\mathbf{L}})^2}

	.. important::

		Realizations of projection DPPs have fixed cardinality.

		.. math::
			:label: number_points_projection_K

			|\mathcal{X}| 
				\overset{a.s.}{=} 
					\operatorname{Tr} \mathbf{K} 
				= \operatorname{rank} \mathbf{K}

		Indeed, since :math:`\mathbf{K}^2=\mathbf{K}`, :eq:`var_number_points` becomes

		.. math::

			\mathbb{V}ar[|\mathcal{X}|] 
			= \operatorname{Tr} \mathbf{K} - \operatorname{Tr} \mathbf{K}^2
			= 0

		and :eq:`expect_number_points` gives

		.. math::

			\mathbb{E}[|\mathcal{X}|] 
			= \operatorname{Tr} \mathbf{K} 
			= \operatorname{rank} \mathbf{K}