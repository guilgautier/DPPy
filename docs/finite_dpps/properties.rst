.. _finite_dpps_properties:

Properties
**********

Throughout this section, we assume :math:`\mathbf{K}` and :math:`\mathbf{L}` satisfy the sufficient conditions :eq:`eq:suff_cond_K` and :eq:`eq:suff_cond_L` respectively.

.. _finite_dpps_relation_kernels:

Relation between correlation and likelihood kernels
===================================================

1. Considering the DPP defined by :math:`\mathbf{L} \succeq 0_N`, the associated correlation kernel :math:`\mathbf{K}` :eq:`eq:inclusion_proba_DPP_K` can be derived as

	.. math::
		:label: eq:compute_K_from_L

		\mathbf{K} = \mathbf{L}(I+\mathbf{L})^{—1} = I - (I+\mathbf{L})^{—1}.

	.. seealso::

		Theorem 2.2 :cite:`KuTa12`.

2. Considering the DPP defined by :math:`0_N \preceq \mathbf{K} \prec I_N`, the associated likelihood kernel :math:`\mathbf{L}` :eq:`eq:likelihood_DPP_L` can be derived as

	.. math::
		:label: eq:compute_L_from_K

		\mathbf{L} = \mathbf{K}(I-\mathbf{K})^{—1} = -I + (I-\mathbf{K})^{—1}.

	.. seealso::

		Equation 25 :cite:`KuTa12`.

.. important::

	Thus, except for correlation kernels :math:`\mathbf{K}` with some eigenvalues equal to :math:`1`, both :math:`\mathbf{K}` and :math:`\mathbf{L}` are diagonalizable in the same basis

	.. math::
		:label: eq:eigendecomposition_K_L

		\mathbf{K} = U \Lambda U^{\dagger}, \quad
		\mathbf{L} = U \Gamma U^{\dagger}
		\qquad \text{with} \qquad
		\lambda_n = \frac{\gamma_n}{1+\gamma_n}.

.. note::

	For DPPs with *projection* correlation kernel :math:`\mathbf{K}`, the likelihood kernel :math:`\mathbf{L}` cannot be computed via  :eq:`eq:compute_L_from_K`, since :math:`\mathbf{K}` has at least one eigenvalue equal to :math:`1` (:math:`\mathbf{K}^2=\mathbf{K}`).

	Nevertheless, if you recall that the :ref:`number of points of a projection DPP <finite_dpps_properties_number_of_points_dpp_K_projection>`, then its likelihood reads

	.. math::

		\mathbb{P}[\mathcal{X}=S] =
			\det \mathbf{K}_S 1_{|S|=\operatorname{rank}(\mathbf{K})}
			\quad \forall S\subset [N].

.. code-block:: python

	from numpy.random import randn, rand
	from scipy.linalg import qr
	from dppy.finite_dpps.finite_dpps  import FiniteDPP

	r, N = 4, 10
	eig_vals = rand(r)  # 0< <1
	eig_vecs, _ = qr(randn(N, r), mode='economic')

	DPP = FiniteDPP('correlation', **{'K_eig_dec': (eig_vals, eig_vecs)})
	DPP.compute_L()

	# - L (likelihood) kernel computed via:
	# - eig_L = eig_K/(1-eig_K)
	# - U diag(eig_L) U.T

.. seealso::

	.. currentmodule:: dppy.finite_dpps.finite_dpps

	- :py:meth:`~FiniteDPP.compute_K`
	- :py:meth:`~FiniteDPP.compute_L`

.. _finite_dpps_mixture:

Generic DPPs as mixtures of projection DPPs
===========================================

*Projection* DPPs are the building blocks of the model in the sense that generic DPPs are mixtures of *projection* DPPs.

.. important::

	Consider :math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{K})` and write the spectral decomposition of the corresponding kernel as

	.. math::

		\mathbf{K} = \sum_{n=1}^N \lambda_n u_n u_n^{\dagger}.

	Then, denote :math:`\mathcal{X}^B \sim \operatorname{DPP}(\mathbf{K}^B)` with

	.. math::

		\mathbf{K}^B = \sum_{n=1}^N B_n u_n u_n^{\dagger},
		\quad
		\text{where}
		\quad
		B_n \overset{\text{i.i.d.}}{\sim} \mathcal{B}er(\lambda_n),

	where :math:`\mathcal{X}^B` is obtained by first choosing :math:`B_1, \dots, B_N` independently and then sampling from :math:`\operatorname{DPP}(\mathbf{K}^B)` the DPP with orthogonal projection kernel :math:`\mathbf{K}^B`.

	Finally, we have :math:`\mathcal{X} \overset{d}{=} \mathcal{X}^B`.

.. seealso::

	- Theorem 7 in :cite:`HKPV06`
	- :ref:`finite_dpps_exact_sampling`
	- Continuous case of :ref:`continuous_dpps_mixture`

.. _finite_dpps_number_of_points:

Number of points
================

For projection DPPs, i.e., when :math:`\mathbf{K}` is an orthogonal projection matrix, one can show that :math:`|\mathcal{X}|=\operatorname{rank}(\mathbf{K})=\operatorname{Trace}(\mathbf{K})` almost surely (see, e.g., Lemma 17 of :cite:`HKPV06` or Lemma 2.7 of :cite:`KuTa12`).

In the general case, based on the fact that :ref:`generic DPPs are mixtures of projection DPPs <finite_dpps_mixture>`, we have

.. math::
	:label: eq:number_of_points

	|\mathcal{X}|
		= \sum_{n=1}^N
			\operatorname{\mathcal{B}er}
			\left(
				\lambda_n
			\right)
		= \sum_{n=1}^N
			\operatorname{\mathcal{B}er}
			\left(
				\frac{\gamma_n}{1+\gamma_n}
			\right).

.. note::

	From :eq:`eq:number_of_points` it is clear that :math:`|\mathcal{X}|\leq \operatorname{rank}(\mathbf{K})=\operatorname{rank}(\mathbf{L})`.

Expectation
-----------

.. math::
	:label: eq:expect_number_points

	\mathbb{E}[|\mathcal{X}|]
		= \operatorname{trace} \mathbf{K}
		= \sum_{n=1}^N \lambda_n
		= \sum_{n=1}^N \frac{\gamma_n}{1+\gamma_n}.

The expected size of a DPP with likelihood matrix :math:`\mathbf{L}` is also related to the effective dimension :math:`d_{\text{eff}}(\mathbf{L}) = \operatorname{trace} (\mathbf{L}(\mathbf{L}+\mathbf{I})^{-1})= \operatorname{trace} \mathbf{K} = \mathbb{E}[|\mathcal{X}|]` of :math:`\mathbf{L}`, a quantity with many applications in randomized numerical linear algebra and statistical learning theory (see e.g., :cite:`DeCaVa19`).

Variance
--------

.. math::
	:label: eq:var_number_points

	\operatorname{\mathbb{V}ar}[|\mathcal{X}|]
		= \operatorname{trace} \mathbf{K} - \operatorname{trace} \mathbf{K}^2
		= \sum_{n=1}^N \lambda_n(1-\lambda_n)
		= \sum_{n=1}^N \frac{\gamma_n}{(1+\gamma_n)^2}.

.. seealso::

	Expectation and variance of :ref:`continuous_dpps_linear_statistics`.

.. testcode::

	import numpy as np
	from scipy.linalg import qr
	from dppy.finite_dpps.finite_dpps  import FiniteDPP

	rng = np.random.RandomState(1)

	r, N = 5, 10
	eig_vals = rng.rand(r) # 0< <1
	eig_vecs, _ = qr(rng.randn(N, r), mode='economic')

	dpp_K = FiniteDPP('correlation', projection=False,
	                **{'K_eig_dec': (eig_vals, eig_vecs)})

	nb_samples = 2000
	for _ in range(nb_samples):
	    dpp_K.sample_exact(random_state=rng)

	sizes = list(map(len, dpp_K.list_of_samples))
	print('E[|X|]:\n emp={:.3f}, theo={:.3f}'
	      .format(np.mean(sizes), np.sum(eig_vals)))
	print('Var[|X|]:\n emp={:.3f}, theo={:.3f}'
	      .format(np.var(sizes), np.sum(eig_vals*(1-eig_vals))))

.. testoutput::

	E[|X|]:
	 emp=1.581, theo=1.587
	Var[|X|]:
	 emp=0.795, theo=0.781

Special cases
-------------

.. _finite_dpps_properties_number_of_points_dpp_K_projection:

1. When the correlation kernel :math:`\mathbf{K}` :eq:`eq:inclusion_proba_DPP_K` of :math:`\operatorname{DPP}(\mathbf{K})` is an orthogonal projection kernel, i.e., :math:`\operatorname{DPP}(\mathbf{K})` is a :ref:`projection DPP <finite_dpps_definition_projection_dpps>`, we have

   	.. math::
   		:label: number_of_points_dpp_K_projection

   		|\mathcal{X}| = \operatorname{rank}(\mathbf{K}) = \operatorname{trace}(\mathbf{K}), \quad \text{almost surely}.

   	.. testcode::

		import numpy as np
		from scipy.linalg import qr
		from dppy.finite_dpps.finite_dpps  import FiniteDPP

		r, N = 4, 10
		eig_vals = np.ones(r)
		eig_vecs, _ = qr(rng.randn(N, r), mode='economic')

		DPP = FiniteDPP('correlation', projection=True,
		                **{'K_eig_dec': (eig_vals, eig_vecs)})

		for _ in range(1000):
		    DPP.sample_exact()

		sizes = list(map(len, DPP.list_of_samples))
		# np.array(DPP.list_of_samples).shape = (1000, 4)

		assert([np.mean(sizes), np.var(sizes)] == [r, 0])

	.. important::

		Since :math:`|\mathcal{X}|=\operatorname{rank}(\mathbf{K})` points, almost surely, the likelihood of the projection :math:`\operatorname{DPP}(\mathbf{K})` reads

		.. math::
			:label: eq:likelihood_projection_K

			\mathbb{P}[\mathcal{X}=S]
				= \det \mathbf{K}_S 1_{|S|=\operatorname{rank} \mathbf{K}}.

		In other words, the projection DPP having for **correlation** kernel the orthogonal projection matrix :math:`\mathbf{K}` coincides with the :ref:`k-DPP <finite_dpps_definition_k_dpps>` having **likelihood** kernel  :math:`\mathbf{K}` when :math:`k=\operatorname{rank}(\mathbf{K})`.

2. When the likelihood kernel :math:`\mathbf{L}` of :math:`\operatorname{DPP}(\mathbf{L})` :eq:`eq:likelihood_DPP_L` is an orthogonal projection kernel we have

   	.. math::
   		:label: number_of_points_dpp_L_projection

   		|\mathcal{X}| \sim \operatorname{Binomial}(\operatorname{rank}(\mathbf{L}), 1/2).

	.. :ref:`Fig. <nb_points_DPP_L_projectin_plot>`

	.. _nb_points_DPP_L_projectin_plot:

	.. plot:: plots/ex_plot_number_of_points_finite_dpp_L_projection.py

		Distribution of the numbe of points of :math:`\operatorname{DPP}(\mathbf{L})` with orthogonal projection kernel :math:`\mathbf{L}` with rank :math:`5`.

.. _finite_dpps_geometry:

Geometrical insights
====================

Kernels satisfying the sufficient conditions :eq:`eq:suff_cond_K` and :eq:`eq:suff_cond_L` can be expressed as

.. math::

	\mathbf{K}_{ij} = \langle \phi_i, \phi_j \rangle
	\quad \text{and} \quad
	\mathbf{L}_{ij} = \langle \psi_i, \psi_j \rangle,

where each item is represented by a feature vector :math:`\phi_i` (resp. :math:`\psi_i`).

The geometrical view is then straightforward.

a. The inclusion probabilities read

	.. math::

		\mathbb{P}[S\subset \mathcal{X}]
		= \det \mathbf{K}_S
		= \operatorname{Vol}^2 \{\phi_s\}_{s\in S}.

b. The likelihood reads

	.. math::

		\mathbb{P}[\mathcal{X} = S]
		\propto \det \mathbf{L}_S
		= \operatorname{Vol}^2 \{\psi_s\}_{s\in S}.

That is to say, DPPs favor subsets :math:`S` whose corresponding feature vectors span a large volume i.e. *DPPs sample softened orthogonal bases*.

.. seealso::

	:ref:`Geometric interpretation of the chain rule for projection DPPs <finite_dpps_exact_sampling_projection_dpp_chain_rule_geometrical_interpretation>`

.. _finite_dpps_diversity:

Diversity
=========

The *determinantal* structure of DPPs encodes the notion of diversity.
Deriving the pair inclusion probability, also called the 2-point correlation function using :eq:`eq:inclusion_proba_DPP_K`, we obtain

.. math::

	\mathbb{P}[\{i, j\} \subset \mathcal{X}]
	&= \begin{vmatrix}
		\mathbb{P}[i \in \mathcal{X}]	& \mathbf{K}_{i j}\\
		\overline{\mathbf{K}_{i j}}		& \mathbb{P}[j \in \mathcal{X}]
	\end{vmatrix}\\
	&= \mathbb{P}[i \in \mathcal{X}] \mathbb{P}[j \in \mathcal{X}]
		- |\mathbf{K}_{i j}|^2,

so that, the larger :math:`|\mathbf{K}_{i j}|` less likely items :math:`i` and :math:`j` co-occur. If :math:`K_{ij}` models the :ref:`similarity <finite_dpps_geometry>` between items :math:`i` and :math:`j`, DPPs are thus random diverse sets of elements.

.. _finite_dpps_conditioning:

Conditioning
============

Like many other statistics of DPPs, the conditional probabilities can be expressed my means of a determinant and involve the correlation kernel :math:`\mathbf{K}` :eq:`eq:inclusion_proba_DPP_K`.

For any disjoint subsets :math:`S, T \subset [N]`, i.e., such that :math:`S\cap T = \emptyset` we have

.. math::
	:label: eq:conditioned_on_S_in_X

	\mathbb{P}[T \subset \mathcal{X} \mid S \subset \mathcal{X}]
        = \det\left[\mathbf{K}_T - \mathbf{K}_{TS} \mathbf{K}_S^{-1} \mathbf{K}_{ST}\right],

.. math::
	:label: eq:conditioned_on_S_notin_X

	\mathbb{P}[T \subset \mathcal{X} \mid S \cap \mathcal{X} = \emptyset]
    	= \det\left[\mathbf{K}_T - \mathbf{K}_{TS} (\mathbf{K}_S - I)^{-1} \mathbf{K}_{ST}\right].

.. seealso::

	- Propositions 3 and 5 of :cite:`Pou19` for the proofs
	- Equations :eq:`eq:conditioned_on_S_in_X` and :eq:`eq:conditioned_on_S_in_X` are key to derive the :ref:`Cholesky-based exact sampler <finite_dpps_exact_sampling_cholesky_method>` which makes use of the chain rule on sets.
