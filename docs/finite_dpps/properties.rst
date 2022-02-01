.. _finite_dpps_properties:

Properties
**********

Throughout this section, we assume :math:`\mathbf{K}` and :math:`\mathbf{L}` satisfy the sufficient conditions :eq:`eq:suff_cond_K` and :eq:`eq:suff_cond_L` respectively.

.. _finite_dpps_relation_kernels:

Relation between correlation and likelihood kernels
===================================================

1. Considering :math:`\operatorname{DPP}(\mathbf{L})` :eq:`eq:likelihood_DPP_L` defined by :math:`\mathbf{L} \succeq 0_N`, the associated correlation kernel :math:`\mathbf{K}` :eq:`eq:inclusion_proba_DPP_K` can be derived as

    .. math::
        :label: eq:compute_K_from_L

        \mathbf{K} = \mathbf{L}(I+\mathbf{L})^{—1} = I - (I+\mathbf{L})^{—1}.

    .. seealso::

        Theorem 2.2 :cite:`KuTa12`.

2. Considering :math:`\operatorname{DPP}(\mathbf{K})` :eq:`eq:inclusion_proba_DPP_K` defined by :math:`0_N \preceq \mathbf{K} \prec I_N`, the associated likelihood kernel :math:`\mathbf{L}` :eq:`eq:likelihood_DPP_L` can be derived as

    .. math::
        :label: eq:compute_L_from_K

        \mathbf{L} = \mathbf{K}(I-\mathbf{K})^{—1} = -I + (I-\mathbf{K})^{—1}.

    .. seealso::

        Equation 25 :cite:`KuTa12`.

.. important::

    For hermitian kernels, apart from correlation kernels :math:`\mathbf{K}` with some eigenvalues equal to :math:`1`, both :math:`\mathbf{K}` and :math:`\mathbf{L}` are diagonalizable in the same basis

    .. math::
        :label: eq:eigendecomposition_K_L

        \mathbf{K} = U \Lambda U^{*}, \quad
        \mathbf{L} = U \Gamma U^{*}
        \qquad \text{with} \qquad
        \lambda_n = \frac{\gamma_n}{1+\gamma_n}, \gamma_n = \frac{\lambda_n}{1-\lambda_n}.

.. note::

    For *projection* :math:`\operatorname{DPP}(\mathbf{K})`, the likelihood kernel :math:`\mathbf{L}` cannot be computed via  :eq:`eq:compute_L_from_K`, since :math:`\mathbf{K}` has at least one eigenvalue equal to :math:`1` (:math:`\mathbf{K}^2=\mathbf{K}`).

    Nevertheless, in this case, for :math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{K})` we have :math:`|\mathcal{X}|=\operatorname{rank}(\mathbf{K})`, see :ref:`number of points of a projection DPP <finite_dpps_number_of_points_projection_case>`.
    Hence, the likelihood reads

    .. math::

        \mathbb{P}[\mathcal{X}=S] =
            \det \mathbf{K}_S 1_{|S|=\operatorname{rank}(\mathbf{K})}
            \quad \forall S\subset [N].

.. code-block:: python

    import numpy as np
    import scipy.linalg as la
    from dppy.finite.dpp import FiniteDPP

    r, N = 4, 10
    eig_vals = np.random.rand(r)  # 0< <1
    eig_vecs, _ = la.qr(np.random.randn(N, r), mode="economic")

    dpp = FiniteDPP(
        "correlation", projection=False, hermitian=True, K_eig_dec=(eig_vals, eig_vecs)
    )

.. seealso::

    - :py:meth:`~dppy.finite.dpp.FiniteDPP.compute_K`
    - :py:meth:`~dppy.finite.dpp.FiniteDPP.compute_L`

.. _finite_dpps_mixture:

Generic DPPs as mixtures of projection DPPs
===========================================

*Projection* DPPs are the building blocks of the model in the sense that generic DPPs are mixtures of *projection* DPPs.

.. important::

    Consider :math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{K})` and write the spectral decomposition of the corresponding kernel as

    .. math::

        \mathbf{K} = \sum_{n=1}^N \lambda_n u_n u_n^{*}.

    Then, denote :math:`\mathcal{X}^B \sim \operatorname{DPP}(\mathbf{K}^B)` with

    .. math::

        \mathbf{K}^B = \sum_{n=1}^N B_n u_n u_n^{*},
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

.. _finite_dpps_number_of_points_general_case:

General case
------------

.. _finite_dpps_number_of_points_general_case_expectation:

Expectation
^^^^^^^^^^^

.. math::
    :label: eq:expect_number_points_general

    \mathbb{E}[|\mathcal{X}|]
        = \operatorname{trace} \mathbf{K}.

.. _finite_dpps_number_of_points_general_case_variance:

Variance
^^^^^^^^

.. math::
    :label: eq:var_number_points_general

    \operatorname{\mathbb{V}ar}[|\mathcal{X}|]
        = \operatorname{trace} \mathbf{K} - \operatorname{trace} \mathbf{K}^2.

.. seealso::

    These formulas are particular cases of the continuous case:

    - :ref:`continuous_dpps_linear_statistics`
    - :ref:`continuous_dpps_number_of_points`

.. _finite_dpps_number_of_points_projection_case:

Projection case
---------------

For projection :math:`\operatorname{DPP}(\mathbf{K})`, i.e., :math:`\mathbf{K}^2 = \mathbf{K}`, :eq:`eq:expect_number_points_general` and :eq:`eq:var_number_points_general` yield

.. math::
    :label: number_of_points_dpp_K_projection

    |\mathcal{X}|
        = \operatorname{trace}(\mathbf{K})
        = \operatorname{rank}(\mathbf{K}),
    \quad \text{almost surely}.

.. seealso::

    - :cite:`HKPV06`, Lemma 17
    - :cite:`KuTa12`, Lemma 2.7

Example

.. testcode::

    import numpy as np
    from scipy.linalg import qr
    from dppy.finite.dpp import FiniteDPP

    r, N = 4, 10
    eig_vals = np.ones(r)
    eig_vecs, _ = qr(rng.randn(N, r), mode="economic")

    dpp = FiniteDPP(
        "correlation", projection=True, hermitian=True, K_eig_dec=(eig_vals, eig_vecs)
    )

    for _ in range(1000):
        dpp.sample_exact(method="spectral")

    sizes = [len(X) for X in DPP.list_of_samples]

    assert [np.mean(sizes), np.var(sizes)] == [r, 0]

.. note::

    Since :math:`|\mathcal{X}|=\operatorname{rank}(\mathbf{K})` almost surely,, the likelihood of projection :math:`\operatorname{DPP}(\mathbf{K})` reads

    .. math::
        :label: eq:likelihood_projection_K

        \mathbb{P}[\mathcal{X}=S]
            = \det \mathbf{K}_S 1_{|S|=\operatorname{rank} \mathbf{K}}.


    In other words, projection :math:`\operatorname{DPP}(\mathbf{K})` coincides with :math:`k\!\operatorname{-DPP}(\mathbf{L})` when :math:`\mathbf{L}=\mathbf{K}` and :math:`k=\operatorname{rank}(\mathbf{K})`.

.. _finite_dpps_number_of_points_hermitian_case:

Hermitian case
--------------

For hermitian DPPs, based on the fact that :ref:`generic DPPs are mixtures of projection DPPs <finite_dpps_mixture>`, we have

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

.. _finite_dpps_number_of_points_hermitian_case_expectation:

Expectation
^^^^^^^^^^^

.. math::
    :label: eq:expect_number_points

    \mathbb{E}[|\mathcal{X}|]
        = \operatorname{trace} \mathbf{K}
        = \sum_{n=1}^N \lambda_n
        = \sum_{n=1}^N \frac{\gamma_n}{1+\gamma_n}.

.. note::

    The expected size of a DPP with likelihood matrix :math:`\mathbf{L}` is also related to the effective dimension :math:`d_{\text{eff}}(\mathbf{L}) = \operatorname{trace} (\mathbf{L}(\mathbf{L}+\mathbf{I})^{-1})= \operatorname{trace} \mathbf{K} = \mathbb{E}[|\mathcal{X}|]` of :math:`\mathbf{L}`, a quantity with many applications in randomized numerical linear algebra and statistical learning theory (see e.g., :cite:`DeCaVa19`).

.. _finite_dpps_number_of_points_hermitian_case_variance:

Variance
^^^^^^^^

.. math::
    :label: eq:var_number_points

    \operatorname{\mathbb{V}ar}[|\mathcal{X}|]
        = \operatorname{trace} \mathbf{K} - \operatorname{trace} \mathbf{K}^2
        = \sum_{n=1}^N \lambda_n(1-\lambda_n)
        = \sum_{n=1}^N \frac{\gamma_n}{(1+\gamma_n)^2}.


.. testcode::

    import numpy as np

    from dppy.finite.dpp import FiniteDPP
    from scipy.linalg import qr

    rng = np.random.RandomState(1)

    r, N = 5, 10
    eig_vals = rng.rand(r)  # 0< <1
    eig_vecs, _ = qr(rng.randn(N, r), mode="economic")

    dpp_K = FiniteDPP(
        "correlation", projection=False, **{"K_eig_dec": (eig_vals, eig_vecs)}
    )

    nb_samples = 2000
    for _ in range(nb_samples):
        dpp_K.sample_exact(random_state=rng)

    sizes = list(map(len, dpp_K.list_of_samples))
    print("E[|X|]:\n emp={:.3f}, theo={:.3f}".format(np.mean(sizes), np.sum(eig_vals)))
    print(
        "Var[|X|]:\n emp={:.3f}, theo={:.3f}".format(
            np.var(sizes), np.sum(eig_vals * (1 - eig_vals))
        )
    )

.. testoutput::

    E[|X|]:
    emp=1.581, theo=1.587
    Var[|X|]:
    emp=0.795, theo=0.781

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

That is to say, Hermitian DPPs favor subsets :math:`S` whose corresponding feature vectors span a large volume i.e. *DPPs sample softened orthogonal bases*.

.. seealso::

    :ref:`Geometric interpretation of the chain rule for projection DPPs <finite_dpps_exact_sampling_projection_dpp_geometrical_interpretation>`

.. _finite_dpps_diversity:

Diversity
=========

For hermitian DPPs, i.e., defined by hermitian kernels :eq:`eq:suff_cond_K` or :eq:`eq:suff_cond_L`, the notion of diversity or negative correlation is encoded by the *determinantal* structure.
For example, using :eq:`eq:inclusion_proba_DPP_K`, the pair inclusion probability provides

.. math::

    \mathbb{P}[\{i, j\} \subset \mathcal{X}]
    &= \begin{vmatrix}
        \mathbb{P}[i \in \mathcal{X}]	& \mathbf{K}_{i j}\\
        \overline{\mathbf{K}_{i j}}		& \mathbb{P}[j \in \mathcal{X}]
    \end{vmatrix}\\
    &= \mathbb{P}[i \in \mathcal{X}] \mathbb{P}[j \in \mathcal{X}]
        - |\mathbf{K}_{i j}|^2.

In other words, the larger :math:`|\mathbf{K}_{i j}|` less likely items :math:`i` and :math:`j` co-occur.
If :math:`K_{ij}` models the :ref:`similarity <finite_dpps_geometry>` between items :math:`i` and :math:`j`, DPPs are thus random diverse sets of elements.

.. _finite_dpps_complementary_process:

Complementary process
=====================

Let :math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{K})` and :math:`\mathcal{X}^{c} \triangleq \left\{1, \dots, N\right\} \setminus \mathcal{X}`, then

.. math::
    :label: eq:finite_dpp_complementary_process

    \mathcal{X}^{c} \sim \operatorname{DPP}(I-\mathbf{K}).

In particular, this means that

.. math::
    :label: eq:complementary

    \mathbb{P}[ \mathcal{X}\cap B = \emptyset]
    = \mathbb{P}[ B \subset \mathcal{X}^c]
    = \det [I-\mathbf{K}]_B.


.. _finite_dpps_inclusion_exclusion_principle:

Inclusion-exclusion principle
=============================

For disjoint subsets :math:`A, B`, we have

.. math::
    :label: eq:inclusion-exclusion_principle_dpp

    \mathbb{P}[A\subset \mathcal{X},  \mathcal{X}\cap B = \emptyset]
        % &= \sum_{S: S\subset B}
            % (-1)^{|S|} \mathbb{P}[A\cup S \subset  \mathcal{X}]\\
        = \det [I^A\mathbf{K}  + I^{A^{c}} (I-\mathbf{K} )]_{A\sqcup B},

where :math:`I^{A}` denotes the indicator matrix of the subset :math:`A`, i.e., :math:`[I^{A}]_{ij} = 1_{i\in A} 1_{j\in A}`.

In particular,

.. math::
    :label: eq:inclusion-exclusion_principle_dpp_B_not_in_X

    \mathbb{P}[A\subset \mathcal{X},  \mathcal{X}\cap B = \emptyset]
        =
        % &= \sum_{S: S\subset B}
            % (-1)^{|S|} \mathbb{P}[A\cup S \subset  \mathcal{X}]\\
        \begin{cases}
            \det [I-\mathbf{K} ]_B
            \det [\mathbf{K} + \mathbf{K} _{:B}[I-\mathbf{K} ]_B^{-1}\mathbf{K} _{B:}]_{A},
                &\text{ if }\mathbb{P}[ \mathcal{X}\cap B = \emptyset]>0,\\
            \det [\mathbf{K} ]_A
            \det [I - (\mathbf{K} -\mathbf{K} _{:A}\mathbf{K} _A^{-1}\mathbf{K} _{A:})]_{B},
                &\text{ if }\mathbb{P}[A\subset \mathcal{X}]>0.\\
        \end{cases}

.. _finite_dpps_conditioning:

Conditioning
============

Let :math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{K})`, we have

.. math::
    :label: eq:finite_dpp_conditioning_inclusion_dpp

    \mathcal{X} \mid B \subset \mathcal{X}
        \sim \operatorname{DPP}(I^{B} + \mathbf{K} - \mathbf{K}_{:B} \mathbf{K}_B^{-1} \mathbf{K}_{B:}),

and

.. math::
    :label: eq:finite_dpp_conditioning_exclusion_dpp

    \mathcal{X} \mid B \cap \mathcal{X} = \emptyset
        \sim
        \operatorname{DPP}(
            I^{B^{c}}
            [\mathbf{K} + \mathbf{K}_{:B} [I-\mathbf{K}]_B^{-1} \mathbf{K}_{B:}]
            I^{B^{c}}
        ).

In particular,

.. math::
	:label: eq:finite_dpp_conditioning_inclusion_proba

	\mathbb{P}[A \subset \mathcal{X} \mid B \subset \mathcal{X}]
        = \det\left[\mathbf{K}_A - \mathbf{K}_{AB} \mathbf{K}_B^{-1} \mathbf{K}_{BA}\right],

.. math::
	:label: eq:finite_dpp_conditioning_exclusion_proba

	\mathbb{P}[A \subset \mathcal{X} \mid B \cap \mathcal{X} = \emptyset]
    	= \det\left[\mathbf{K}_A - \mathbf{K}_{AB} (\mathbf{K}_B - I)^{-1} \mathbf{K}_{BA}\right].

.. seealso::

    - Propositions 3 and 5 of :cite:`Pou19` for the proofs,
    - :ref:`Cholesky-based exact sampler <finite_dpps_exact_sampling_sequential_methods>`.

.. _finite_dpps_dpp_union_bernoulli_process:

Union with a Bernoulli process
==============================

Consider the following independent point processes :math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{K})` and :math:`\mathcal{Y} \sim \operatorname{DPP}(\mathbf{D})`, where :math:`\mathbf{D} = \operatorname{diag}(d_1, \dots, d_n)`, also called a Bernoulli process.

Then,

.. math::
    :label: eq:finite_dpps_dpp_union_bernoulli_process

    \mathcal{X} \cup \mathcal{Y}
    \sim
    \operatorname{DPP}(
        \mathbf{D} + (I-\mathbf{D})^{1/2}\mathbf{K}(I-\mathbf{D})^{1/2}
    )
