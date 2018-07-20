.. _discrete_dpps_properties:

Properties
**********

Throughout this section, we assume :math:`\mathbf{K}` and :math:`\mathbf{L}` satisfy the sufficient conditions :eq:`suff_cond_K` and eq:`suff_cond_L` respectively.

.. _discrete_dpps_mixture:

Generic DPPs as mixtures of projection DPPs
===========================================

*Projection* DPPs are the building blocks of the model in the sense that generic DPPs are mixtures of *projection* DPPs.

.. important::

	More precisely, noting the spectral decomposition :math:`\mathbf{K}
	= \sum_{n=1}^N \lambda_n^{\mathbf{K}} u_n u_n^{\dagger}` then we have

	.. math::

		\operatorname{DPP}(\mathbf{K})\sim\operatorname{DPP}(\mathbf{K}^B)
	
	where :math:`\mathbf{K}^B` is the **random** *projection* kernel defined by

	.. math::

		\mathbf{K}^B
		= \sum_{n=1}^N 
		B_n
		u_n u_n^{\dagger}

	with :math:`(B_n)_{n=1}^N` independent Bernoulli variables with respective parameter the :math:`\lambda_n^{\mathbf{K}})`.

.. seealso::

	- Theorem 7 in :cite:`HKPV06`
	- :ref:`discrete_dpps_exact_sampling`

.. _discrete_dpps_nb_points:

Number of points
================

	Based on :ref:`discrete_dpps_mixture`, we have

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

		Realizations of *projection* DPPs have fixed cardinality.

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

		Thus,

		.. math::
			:label: marginal_projection_K

			\mathbb{P}[\mathcal{X}=S] 
				= \det \mathbf{K}_S 1_{|S|=\operatorname{rank} \mathbf{K}}


.. _discrete_dpps_geometry:

Geometrical insights
====================

	Kernels satisfying the sufficient conditions :eq:`suff_cond_K` and :eq:`suff_cond_L` can be expressed as

	.. math::

		\mathbf{K}_{ij} = \langle \phi_i, \phi_j \rangle
		\quad \text{and} \quad
		\mathbf{K}_{ij} = \langle \psi_i, \psi_j \rangle,

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

.. _discrete_dpps_diversity:

Diversity
=========

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

	That is, the greater the similarity :math:`|\mathbf{K}_{i j}|` between items :math:`i` and :math:`j`, the less likely they co-occur.

.. _discrete_dpps_relation_kernels:

Relation between inclusion and marginal kernels
===============================================

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

		\mathbf{K} = U \Lambda^{\mathbf{K}} U^{\dagger}
			\qquad \text{and} \qquad
		\mathbf{L} = U \Lambda^{\mathbf{L}} U^{\dagger}