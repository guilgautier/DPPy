.. _continuous_dpps_sampling:

Sampling
********

Exact sampling
==============

The procedure stems from the fact that :ref:`generic DPPs are mixtures of projection DPPs <continuous_dpps_mixture>`, suggesting the following two steps algorithm.
Given the spectral decomposition of the kernel

.. math::
    :label: eq:continuous_dpps_sampling_eigendecomposition_K

	K(x,y)=\sum_{i=1}^{\infty} \lambda_i \phi_i(x) \overline{\phi_i(y)},

.. _continuous_dpps_exact_sampling_spectral_method_step_1:

**Step 1.** Draw :math:`B_i\sim\operatorname{\mathcal{B}er}(\lambda_i)` independently and note :math:`\{i_1,\dots,i_{N}\} = \{i~;~B_i=1\}`,

.. _continuous_dpps_exact_sampling_spectral_method_step_2:

**Step 2.** Sample from the *projection* DPP with kernel :math:`\tilde{K}(x,y) = \sum_{n=1}^{N}\phi_{i_n}(x) \overline{\phi_{i_n}(y)}`.

.. important::

    - Step :ref:`1. <continuous_dpps_exact_sampling_spectral_method_step_1>` selects a component of the mixture, see :cite:`LaMoRu15` Section 2.4.1
    - Step :ref:`2. <continuous_dpps_exact_sampling_spectral_method_step_2>` generates a sample from the projection :math:`\operatorname{DPP}(\tilde{K})`. This can be done using Algorithm 18 of :cite:`HKPV06` who provide a generic projection DPP sampler that we describe in the :ref:`next section <continuous_dpps_exact_sampling_projection_dpp_chain_rule>`.

    .. attention::

        The strong requirement of the procedure is that the eigendecomposition of the continuous kernel in :eq:`eq:continuous_dpps_sampling_eigendecomposition_K` must be available.

.. seealso::

    :ref:`finite_dpps_exact_sampling_spectral_method` for sampling finite DPPs.

.. _continuous_dpps_exact_sampling_projection_dpp_chain_rule:

Projection DPPs: the chain rule
-------------------------------

In this section, we describe a generic procedure to perform Step :ref:`2. <continuous_dpps_exact_sampling_spectral_method_step_2>`, i.e., for sampling projection DPPs.
It was originally proposed, in an abstract form, by :cite:`HKPV06` Algorithm 18.

For simplicity, consider a projection :math:`\operatorname{DPP}(K)` with a real-valued orthogonal projection kernel.
We note :math:`r=\operatorname{rank}(K)` and write

.. math::

    K(x,y)
    = \sum_{i=1}^{r} \phi_i(x) \phi_i(y)
    = \Phi(x)^{\top} \Phi(y),

where :math:`\Phi(x) \triangleq (\phi_{1}(x), \dots, \phi_{r}(x))` denotes the *feature vector* associated to :math:`x\in \mathbb{X}`.

.. important::

    The eigendecomposition of the kernel is not mandatory for sampling **projection** :math:`\operatorname{DPP}(K)`.

Recall that the :ref:`number of points <continuous_dpps_number_of_points>` of this projection :math:`\operatorname{DPP}(K)` is :math:`\mu`-almost surely equal to :math:`r=\operatorname{rank}(K)`.

Using the invariance by permutation of the determinant and the fact that :math:`K` is an orthogonal projection kernel, it is sufficient to apply the chain rule to sample :math:`(x_1, \dots, x_r)` with joint distribution

.. math::
    :label: eq:continuous_dpps_exact_sampling_projection_DPP_joint_distribution

    \mathbb{P}[(x_1, \dots, x_r)]
    &= \frac{1}{r!} \det [K(x_p,x_q)]_{p,q=1}^r \mu^{\otimes r}(d x_{1:r})\\
    &= \frac{1}{r!}
        \det [\Phi(x_p)^{\top} \Phi(x_q))]_{p,q=1}^r
        \mu^{\otimes r}(d x_{1:r})\\
    &= \frac{1}{r!}
        \operatorname{Volume}^2 \{\Phi(x_1), \dots \Phi(x_r)\}
        \mu^{\otimes r}(d x_{1:r}),

and forget about the order the points were selected, to obtain a valid sample :math:`X=\{x_{1}, \dots, x_{r}\} \sim \operatorname{DPP}(K)`.

.. hint::

    In the end, the joint distribution :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_joint_distribution` shows that projection DPPs favors sets of :math:`r=\operatorname{rank}(\mathbf{K})` of items are associated to feature vectors that span large volumes.
    This is another way of understanding :ref:`repulsiveness <finite_dpps_diversity>`.

The chain rule can be interpreted from a geometrical perspective

.. math::
    :label: eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_dist2_K

    \mathbb{P}[(x_1, \dots, x_r)]
    &= \dfrac{K(x_1,x_1)}{r}
        \mu(d x_{1})
        \prod_{i=2}^{r}
            \dfrac{1}{r-(i-1)}
        \frac{\det \mathbf{K}_{i}}
             {\det \mathbf{K}_{i-1}}
            \mu(d x_{i})\\
    &= \dfrac{\left\| \Phi(x_1) \right\|^2}{r} \mu(d x_{1})
        \prod_{i=2}^{r}
        \dfrac{
                \operatorname{distance}^2
                \left(\Phi(x_i),
                \operatorname{Span}
                    \{
                    \Phi(x_1), \dots, \Phi(x_{i-1})
                    \}\right)
            }
            {r-(i-1)}
        \mu(d x_{i}),

where :math:`\mathbf{K}_{i-1} = [K(x_p,x_q)]_{p,q=1}^{i-1}`.

Using `Woodbury's formula <https://en.wikipedia.org/wiki/Woodbury_matrix_identity>`_ the ratios of determinants in :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_dist2_K` can be expanded into

.. math::
    :label: eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_schur

    \mathbb{P}[(x_1, \dots, x_r)]
    = \dfrac{K(x_1,x_1)}{r} \mu(d x_{1})
        \prod_{i=2}^{r}
            \dfrac{
                K(x_i, x_i)
                - \mathbf{K}_{i-1}(x_i)^{\top}
                    \mathbf{K}_{i-1}^{-1}
                    \mathbf{K}_{i-1}(x_i)
                }
                {r-(i-1)}
                \mu(d x_{i}),

where :math:`\mathbf{K}_{i-1} = [K(x_p,x_q)]_{p,q=1}^{i-1}` and :math:`\mathbf{K}_{i-1}(x) = (K(x,x_1), \dots, K(x,x_{i-1}))^{\top}`.

.. hint::

    a) The chain rule :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_dist2_K` can be understood as an application of the base :math:`\times` height formula.
    b) The overall procedure is akin to a sequential Gram-Schmidt orthogonalization of :math:`\\Phi(x_{1}), \\dots, \\Phi(x_{N})`.
    c) MLers will recognize in :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_schur` the incremental posterior variance of the Gaussian Process (GP) associated to :math:`K`, see :cite:`RaWi06` Equation 2.26.

    .. caution::

        The connexion between the chain rule :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_schur` and Gaussian Processes is valid in the case where the GP kernel is an **orthogonal projection kernel**, see also :ref:`finite_kdpps_exact_sampling_chain_rule_projection_kernel_caution`.

.. important::

    a) The expression :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_joint_distribution` indeed defines a probability distribution, with normalization constant :math:`r!`. In particular this distribution is `exchangeable <https://en.wikipedia.org/wiki/Exchangeable_random_variables>`_
    b) The successive ratios that appear in :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_dist2_K` and :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_schur` are the normalized conditional densities (w.r.t.\, :math:`\mu`) that drive the chain rule. The associated normalizing constants :math:`r-(i-1)` are independent of the previous points.

    .. caution::

      The main differences with the :ref:`finite case <finite_dpps_exact_sampling_projection_dpp_chain_rule>` is that we need to be able to sample from the conditionals that appear in :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_schur`.
      This can be done using a rejection sampling mechanism but finding the right proposal density is a challenging but achievable problem, see, e.g., :ref:`Multivariate Jacobi Ensemble <multivariate_jacobi_ope>`.

.. seealso::

    - Algorithm 18 :cite:`HKPV06`
    - :ref:`finite_dpps_exact_sampling_projection_dpp_chain_rule` in the finite case
    - :ref:`beta_ensembles_definition_OPE` and their specific :ref:`samplers <beta_ensembles_sampling>`
    - :ref:`Multivariate Jacobi Ensemble <multivariate_jacobi_ope>` whose :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.sample` method relies on the chain rule described by :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_schur`

Perfect sampling
----------------

	:cite:`DFL13` uses Coupling From The Past
	(`CFTP <https://pdfs.semanticscholar.org/622e/a9c9c665002670ff26119d1aad5c3c5e0be8.pdf_>`_).

Approximate sampling
====================

.. seealso::

	Approximation of :math:`K(x,y)=K(x-y)` by Fourier series :cite:`LaMoRu15` Section 4
