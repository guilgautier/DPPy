.. _continuous_dpps_sampling:

Sampling
********

Exact sampling
==============

Chain rule
----------

The procedure stems from the fact that :ref:`generic DPPs are mixtures of projection DPPs <continuous_dpps_mixture>`, suggesting the following two steps algorithm given the spectral decomposition of the kernel

.. math::

	K(x,y)=\sum_{i=1}^{\infty} \lambda_i \phi_i(x) \overline{\phi_i(y)}

1. Draw :math:`B_i\sim\operatorname{\mathcal{B}er}(\lambda_i)` independently and note :math:`\{i_1,\dots,i_{N}\} = \{i~;~B_i=1\}`.
2. Sample from the *projection* DPP with kernel :math:`\tilde{K}(x,y) = \sum_{n=1}^{N}\phi_{i_n}(x) \overline{\phi_{i_n}(y)}`.


The remaining question of sampling from *projection* DPPs is addressed by Algorithm 18 :cite:`HKPV06`.
It based on the chain rule and the fact that \textit{projection} :math:`\operatorname{DPP}(\tilde{K})` generates configurations of
:math:`N=\operatorname{Tr} \tilde{K}` :ref:`points almost surely <continuous_dpps_number_of_points>`.
In the first phase each point :math:`x\in \mathbb{X}` is associated to the random feature vector :math:`\Phi(x)=(\phi_{i_1}(x),\dots,\phi_{i_N}(x))`, therefore :math:`\tilde{K}(x,y) = \Phi(y)^{\dagger} \Phi(x)`.

In this setting, the joint distribution of :math:`(X_1,\dots,X_N)` reads


.. math::

  \frac{1}{N!} \det \left[K(x_m,x_n)\right]_{m,n=1}^N \prod_{n=1}^N\mu(d x_n)
    = \frac{1}{N!} \operatorname{Vol}^2\{\Phi(x_1),\dots,\Phi(x_n)\} \prod_{n=1}^N\mu(d x_n)

so that the conditional densities appearing in the chain rule are ratios of 2 determinants that can be expressed as

.. math::

  g_1(x)           &= \frac{1}{N} \|\phi(x)\|^2
                   = \frac{1}{N} K(x,x) \\
  g_{n | 1:n-1}(x) &= \frac{1}{N-(n-1)}
                      \| \Pi_{H_{n-1}^{\perp}} \phi(x) \|^2 \\
                   &= \frac{1}{N-(n-1)}
                      \left[
	                      K(x,x) 
	                      - \overline{K(x,x_{1:n-1})}
	                      \left[\left[K(x_k,x_l)\right]_{k,l=1}^{n-1}\right]^{-1} 
	                      K(x_{1:n-1},x)
                      \right]

.. caution::

  As in the finite case, the eigendecomposition of the kernel is required. The main difference is that we now have to resort to rejection sampling to sample the conditionals. Finding the right proposal density is a challenging problem, though for some specific kernels, there are natural choices :cite:`BaHa16`.

.. seealso::

	- Algorithm 18 :cite:`HKPV06`
	- :ref:`beta_ensembles_definition_OPE`

Perfect sampling
----------------

	:cite:`DFL13` uses Coupling From The Past 
	(`CFTP <https://pdfs.semanticscholar.org/622e/a9c9c665002670ff26119d1aad5c3c5e0be8.pdf_>`_).

Approximate sampling
====================

.. seealso::

	Approximation of :math:`K(x,y)=K(x-y)` by Fourier series :cite:`LaMoRu15` Section 4
