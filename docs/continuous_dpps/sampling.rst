.. _continuous_dpps_sampling:

Sampling
********

In contrast to the :ref:`finite case <finite_dpps_exact_sampling>` where the ML community has put efforts in improving the efficiency and tractability of the sampling routine, much less has been done in the continuous setting.

.. _continuous_dpps_exact_sampling:

Exact sampling
==============

In this section, we describe the main techniques for sampling exactly continuous DPPs.

As for :ref:`finite DPPs <finite_dpps_exact_sampling>` the most prominent one relies on the fact that :ref:`generic DPPs are mixtures of projection DPPs<continuous_dpps_mixture>`.

.. _continuous_dpps_exact_sampling_projection_dpp_chain_rule:

Projection DPPs: the chain rule
-------------------------------

Let's focus on sampling from projection :math:`\operatorname{DPP}(K)` with a real-valued orthogonal projection kernel :math:`K:\mathbb{X}\times \mathbb{X}\to \mathbb{R}` and reference measure :math:`\mu`, that is

.. math::

    K(x,y)=K(y,x)
    \quad\text{and}\quad
    \int_{\mathbb{X}} K(x, z) K(z, y) \mu(d z) = K(x, y)

In this setting, recall that the :ref:`number of points <continuous_dpps_number_of_points>` is :math:`\mu`-almost surely equal to :math:`r=\operatorname{rank}(K)`.

To generate a valid sample :math:`X=\{x_{1}, \dots, x_{r}\} \sim \operatorname{DPP}(K)`, :cite:`HKPV06` Proposition 19 showed that it is sufficient to apply the chain rule to sample :math:`(x_1, \dots, x_r)` with joint distribution

.. math::
    :label: eq:continuous_dpps_exact_sampling_projection_DPP_joint_distribution

    \mathbb{P}[(x_1, \dots, x_r)]
    = \frac{1}{r!}
        \det [K(x_p, x_q)]_{p,q=1}^r
        \mu^{\otimes r}(d x_{1:r})

and forget the order the points were selected.

The original projection DPP sampler of :cite:`HKPV06` Algorithm 18, was given in an abstract form, which can be implemented using the following strategy.
Write the determinant in :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_joint_distribution` as a telescopic product of ratios of determinants and use `Schur complements <https://en.wikipedia.org/wiki/Schur_complement>`_ to get

.. math::
    :label: eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_schur

    \mathbb{P}[(x_1, \dots, x_r)]
    &= \dfrac{K(x_1,x_1)}{r}
        \mu(d x_{1})
        \prod_{i=2}^{r}
            \dfrac{1}{r-(i-1)}
        \frac{\det \mathbf{K}_{i}}
             {\det \mathbf{K}_{i-1}}
            \mu(d x_{i})\\
    &= \dfrac{K(x_1,x_1)}{r} \mu(d x_{1})
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

.. important::

    a) The expression :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_joint_distribution` indeed defines a probability distribution, with normalization constant :math:`r!`. In particular this distribution is `exchangeable <https://en.wikipedia.org/wiki/Exchangeable_random_variables>`_, i.e., invariant by permutation of the coordinates.
    b) The successive ratios that appear in  :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_schur` are the normalized conditional densities (w.r.t. :math:`\mu`) that drive the chain rule. The associated normalizing constants :math:`r-(i-1)` are independent of the previous points.
    c) Sampling **projection** DPPs does not require the eigendecomposition of the kernel!

.. hint::

    MLers will recognize :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_schur` as the incremental posterior variance of a noise-free Gaussian Process (GP) model with kernel :math:`K`, see :cite:`RaWi06` Equation 2.26.

    .. caution::

        The connexion between the chain rule :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_schur` and Gaussian Processes is valid in the case where the GP kernel is an **orthogonal projection kernel**, see also :ref:`finite_kdpps_exact_sampling_chain_rule_projection_kernel_caution`.

.. _continuous_dpps_exact_sampling_projection_dpp_chain_rule_geometrical_interpretation:

Geometrical interpretation
^^^^^^^^^^^^^^^^^^^^^^^^^^

When the eigendecomposition of the kernel is available, the chain rule can be interpreted and implemented from a geometrical perspective, see, e.g., :cite:`LaMoRu15` Algorithm 1.

Writing the Gram formulation of the kernel as

.. math::

    K(x,y)
    = \sum_{i=1}^{r} \phi_i(x) \phi_i(y)
    = \Phi(x)^{\top} \Phi(y),

where :math:`\Phi(x) \triangleq (\phi_{1}(x), \dots, \phi_{r}(x))` denotes the *feature vector* associated to :math:`x\in \mathbb{X}`.

The joint distribution :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_joint_distribution` reads

.. math::
    :label: eq:continuous_dpps_exact_sampling_projection_DPP_joint_distribution_volume

    \mathbb{P}[(x_1, \dots, x_r)]
    &= \frac{1}{r!}
        \det [\Phi(x_p)^{\top} \Phi(x_q))]_{p,q=1}^r
        \mu^{\otimes r}(d x_{1:r})\\
    &= \frac{1}{r!}
        \operatorname{Volume}^2 \{\Phi(x_1), \dots, \Phi(x_r)\}
        \mu^{\otimes r}(d x_{1:r}),

.. hint::

    The joint distribution :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_joint_distribution_volume` characterizes the fact that **projection** :math:`\operatorname{DPP}(K)` favor sets of :math:`r=\operatorname{rank}(\mathbf{K})` points :math:`\left(x_{1}, \dots, x_{r} \right)` whose feature vectors :math:`\Phi(x_1), \dots \Phi(x_r)` span a large volume.
    This is another way of understanding the repulsive/diversity feature of DPPs.

Then, the previous telescopic product of ratios of determinants in :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_schur` can be understood as the base :math:`\times` height formula applied to compute :math:`\operatorname{Volume}^2 \{\Phi(x_1), \dots, \Phi(x_r)\}`, so that

.. math::
    :label: eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_geometric

    \mathbb{P}[(x_1, \dots, x_r)]
    &= \dfrac{\left\langle \Phi(x_1)^{\top} \Phi(x_1) \right\rangle}{r}
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

where :math:`\mathbf{K}_{i-1} = \left[\left\langle \Phi(x_p)^{\top} \Phi(x_q) \right\rangle\right]_{p,q=1}^{i-1}`.

.. hint::

    The overall procedure is akin to a sequential Gram-Schmidt orthogonalization of :math:`\Phi(x_{1}), \dots, \Phi(x_{N})`.

.. attention::

    In contrast to the :ref:`finite case <finite_dpps_exact_sampling_projection_dpp>` where the conditionals are simply probability vectors, the chain rule formulations :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_schur` and :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_geometric` require sampling from a continuous distribution.
    This can be done using a rejection sampling mechanism but finding a good proposal density with tight rejection bounds is a challenging problem :cite:`LaMoRu15` Section 2.4.
    But it is achievable in some specific cases, see, e.g., :ref:`Multivariate Jacobi Ensemble <multivariate_jacobi_ope>`.

.. seealso::

    - Algorithm 18 :cite:`HKPV06` for the original abstract **projection** DPP sampler
    - :ref:`finite_dpps_exact_sampling_projection_dpp` in the finite case
    - Some :ref:`beta_ensembles_definition_OPE` (specific instances of projection DPPs) can be :ref:`sampled <beta_ensembles_sampling>` in :math:`\mathcal{O}(r^2)` by computing the eigenvalues of properly randomised tridiagonal matrices.
    - The :ref:`multivariate Jacobi ensemble <multivariate_jacobi_ope>` whose :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.sample` method relies on the chain rule described by :eq:`eq:continuous_dpps_exact_sampling_projection_DPP_chain_rule_geometric`.

.. _continuous_dpps_exact_sampling_spectral_method:

Generic DPPs: the spectral method
---------------------------------

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
    - Step :ref:`2. <continuous_dpps_exact_sampling_spectral_method_step_2>` generates a sample from the projection :math:`\operatorname{DPP}(\tilde{K})`, cf. :ref:`previous section <continuous_dpps_exact_sampling_projection_dpp_chain_rule>`.

    .. attention::

        Contrary to projection DPPs, the general case requires the eigendecomposition of the kernel :eq:`eq:continuous_dpps_sampling_eigendecomposition_K`.

.. seealso::

    :ref:`finite_dpps_exact_sampling_spectral_method` for sampling finite DPPs.

Perfect sampling
----------------

	:cite:`DFL13` uses Coupling From The Past
	(`CFTP <https://pdfs.semanticscholar.org/622e/a9c9c665002670ff26119d1aad5c3c5e0be8.pdf_>`_).

Approximate sampling
====================

.. seealso::

	- Approximation of :math:`K(x,y)=K(x-y)` by Fourier series :cite:`LaMoRu15` Section 4
