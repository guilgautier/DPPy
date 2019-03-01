.. _finite_dpps_properties:

Properties
**********

Throughout this section, we assume :math:`\mathbf{K}` and :math:`\mathbf{L}` satisfy the sufficient conditions :eq:`suff_cond_K` and :eq:`suff_cond_L` respectively.

.. _finite_dpps_mixture:

Generic DPPs as mixtures of projection DPPs
===========================================

*Projection* DPPs are the building blocks of the model in the sense that generic DPPs are mixtures of *projection* DPPs.

.. important::

	Given the spectral decomposition of the kernel

	.. math::

		\mathbf{K} = \sum_{n=1}^N \lambda_n^{\mathbf{K}} u_n u_n^{\dagger}`

	define the **random** orthogonal projection kernel
	:math:`\mathbf{K}^B = \sum_{n=1}^N B_n u_n u_n^{\dagger}``

	with independent :math:`B_i\sim\mathcal{B}(\lambda_n^{\mathbf{K}})`.
	Then we have

	.. math::

		\operatorname{DPP}(\mathbf{K})\sim\operatorname{DPP}(\mathbf{K}^B)

.. seealso::

	- Theorem 7 in :cite:`HKPV06`
	- :ref:`finite_dpps_exact_sampling`
	- Continuous case of :ref:`continuous_dpps_mixture`

.. _finite_dpps_nb_points:

Number of points
================

Based on :ref:`finite_dpps_mixture`, we have

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

Expectation
-----------

.. math::
	:label: expect_number_points

	\mathbb{E}[|\mathcal{X}|] 
		= \operatorname{Tr} \mathbf{K}
		= \sum_{n=1}^N \lambda_n^{\mathbf{K}}
		= \sum_{n=1}^N \frac{\lambda_n^{\mathbf{L}}}{1+\lambda_n^{\mathbf{L}}}

Variance
--------

.. math::
	:label: var_number_points

	\operatorname{\mathbb{V}ar}[|\mathcal{X}|] 
		= \operatorname{Tr} \mathbf{K} - \operatorname{Tr} \mathbf{K}^2
		= \sum_{n=1}^N \lambda_n^{\mathbf{K}}(1-\lambda_n^{\mathbf{K}})
		= \sum_{n=1}^N \frac{\lambda_n^{\mathbf{L}}}{(1+\lambda_n^{\mathbf{L}})^2}

.. seealso::

	Expectation and variance of :ref:`continuous_dpps_linear_statistics`

.. testcode::

	from numpy import array
	from numpy.random import seed, randn, rand
	from scipy.linalg import qr
	from dppy.finite_dpps import FiniteDPP

	seed(1)

	r, N = 5, 10
	eig_vals = rand(r) # 0< <1
	eig_vecs, _ = qr(randn(N, r), mode='economic')

	DPP = FiniteDPP('correlation', projection=False,
	                **{'K_eig_dec':(eig_vals, eig_vecs)})

	nb_samples = 2000
	for _ in range(nb_samples):
	    DPP.sample_exact()

	sizes = array([s.size for s in DPP.list_of_samples])
	print('E[|X|]:\n theo={:.3f}, emp={:.3f}'
	      .format(sizes.mean(), eig_vals.sum()))
	print('Var[|X|]:\n theo={:.3f}, emp={:.3f}'
	      .format(sizes.var(), (eig_vals*(1-eig_vals)).sum()))

.. testoutput::

	E[|X|]:
	 theo=1.581, emp=1.587
	Var[|X|]:
	 theo=0.795, emp=0.781


.. important::

	Realizations of *projection* DPPs i.e. have fixed cardinality.

	.. math::
		:label: number_points_projection_K

		|\mathcal{X}| 
			\overset{a.s.}{=} 
				\operatorname{Tr} \mathbf{K} 
			= \operatorname{rank} \mathbf{K}

	Since :math:`\mathbf{K}^2=\mathbf{K}`, :eq:`expect_number_points` and :eq:`var_number_points` yield

	.. math::

		\mathbb{E}[|\mathcal{X}|] 
			= \operatorname{Tr} \mathbf{K} 
			= \operatorname{rank} \mathbf{K}
		\quad \text{and} \quad
		\mathbb{V}ar[|\mathcal{X}|] 
			= \operatorname{Tr} \mathbf{K} - \operatorname{Tr} \mathbf{K}^2
			= 0

	Thus,

	.. math::
		:label: marginal_projection_K

		\mathbb{P}[\mathcal{X}=S] 
			= \det \mathbf{K}_S 1_{|S|=\operatorname{rank} \mathbf{K}}

	.. testcode::

		from numpy import ones
		from numpy.random import seed, randn
		from scipy.linalg import qr
		from dppy.finite_dpps import FiniteDPP

		seed(1)

		r, N = 4, 10
		eig_vals = ones(r)
		eig_vecs, _ = qr(randn(N, r), mode='economic')

		DPP = FiniteDPP('correlation', projection=True,
		                **{'K_eig_dec':(eig_vals, eig_vecs)})

		for _ in range(10):
		    DPP.sample_exact()

		print(list(map(list, DPP.list_of_samples)))
	
	.. testoutput::

		[[0, 4, 8, 2], [1, 8, 2, 0], [8, 3, 6, 1], [6, 7, 1, 9], [9, 3, 0, 4], [9, 4, 0, 8], [9, 6, 1, 8], [0, 1, 2, 7], [1, 2, 8, 9], [8, 2, 9, 4]]

.. _finite_dpps_geometry:

Geometrical insights
====================

Kernels satisfying the sufficient conditions :eq:`suff_cond_K` and :eq:`suff_cond_L` can be expressed as

.. math::

	\mathbf{K}_{ij} = \langle \phi_i, \phi_j \rangle
	\quad \text{and} \quad
	\mathbf{L}_{ij} = \langle \psi_i, \psi_j \rangle,

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

.. _finite_dpps_diversity:

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

.. _finite_dpps_relation_kernels:

Relation between correlation and likelihood kernels
===================================================

.. math::
	:label: relation_K_L

	\mathbf{K} = \mathbf{L}(I+\mathbf{L})^{—1} 
		\qquad \text{and} \qquad 
	\mathbf{L} = \mathbf{K}(I-\mathbf{K})^{—1}

.. warning::
	
	For DPPs with *projection* correlation kernel :math:`K`, the likelihood kernel :math:`\mathbf{L}` cannot be computed via  :eq:`relation_K_L` with :math:`\mathbf{L} = \mathbf{K}(I-\mathbf{K})^{—1}`, since :math:`\mathbf{K}` has at least one eigenvalue equal to :math:`1` (:math:`\mathbf{K}^2=\mathbf{K}`).

	However, the likelihood kernel :math:`\mathbf{L}` coincides with :math:`\mathbf{K}`.

	.. math::

		\mathbb{P}[\mathcal{X}=S] = 
			\det \mathbf{K}_S 1_{|S|=\operatorname{rank}\mathbf{K}}
			\quad \forall S\subset [N]

Thus, except for correlation kernels :math:`\mathbf{K}` with some eigenvalues equal to :math:`1`, both :math:`\mathbf{K}` and :math:`\mathbf{L}` are diagonalizable in the same basis

.. math::

	\mathbf{K} = U \Lambda U^{\dagger}, \quad
	\mathbf{L} = U \Gamma U^{\dagger}
	\qquad \text{with} \qquad
	\lambda_n = \frac{\gamma_n}{1+\gamma_n}

.. code-block:: python

	from numpy.random import randn, rand
	from scipy.linalg import qr
	from dppy.finite_dpps import FiniteDPP

	r, N = 4, 10
	eig_vals = rand(r)  # 0< <1
	eig_vecs, _ = qr(randn(N, r), mode='economic')

	DPP = FiniteDPP('correlation', **{'K_eig_dec': (eig_vals, eig_vecs)})
	DPP.compute_L()

	# - L (likelihood) kernel computed via:
	# - eig_L = eig_K/(1-eig_K)
	# - U diag(eig_L) U.T

.. seealso::

	.. currentmodule:: dppy.finite_dpps

	- :py:meth:`~FiniteDPP.compute_K`
	- :py:meth:`~FiniteDPP.compute_L`
