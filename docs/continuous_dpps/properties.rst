.. _continuous_dpps_properties:

Properties
**********

.. _continuous_dpps_mixture:

Generic DPPs as mixtures of projection DPPs
===========================================

*Projection* DPPs are the building blocks of the model in the sense that generic DPPs are mixtures of *projection* DPPs.


	Consider :math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{K})` and write the spectral decomposition of the corresponding kernel as

	.. math::

		\mathbf{K} = \sum_{n=1}^N \lambda_n^{\mathbf{K}} u_n u_n^{\dagger}.

	Then, denote :math:`\mathcal{X}^B \sim \operatorname{DPP}(\mathbf{K}^B)` with

	.. math::

		\mathbf{K}^B = \sum_{n=1}^N B_n u_n u_n^{\dagger},
		\quad
		\text{where}
		\quad
		B_n \overset{\text{i.i.d.}}{\sim} \mathcal{B}er(\lambda_n^{\mathbf{K}})

	where :math:`\mathcal{X}^B` is obtained by first choosing :math:`B_1, \dots, B_N` independently and then sampling from :math:`\operatorname{DPP}(\mathbf{K}^B)` the DPP with orthogonal projection kernel :math:`\mathbf{K}^B`.

	Finally, we have :math:`\mathcal{X} \overset{d}{=} \mathcal{X}^B`.

.. important::

	Given the spectral decomposition of the kernel

	.. math::

	  K(x,y)=\sum_{i=1}^{\infty} \lambda_i \phi_i(x) \overline{\phi_i(y)},
	  \quad \text{with }
	  \int \phi_i(x) \overline{\phi_j(x)} \mu(dx) = \delta_{ij},

	define the random orthogonal projection kernel
	:math:`K^B(x,y) = \sum_{i=1}^{\infty} B_i \phi_i(x) \overline{\phi_i(y)}`
	, with independent :math:`B_i\sim\mathcal{B}(\lambda_i)`.
	Then we have

	.. math::

	  \operatorname{DPP}(K) \sim \operatorname{DPP}(K^B)

.. seealso::

	- Theorem 7 in :cite:`HKPV06`
	- :ref:`continuous_dpps_sampling`
	- Finite case of :ref:`finite_dpps_mixture`

.. _continuous_dpps_linear_statistics:

Linear statistics
=================

Expectation
-----------

.. math::

	\mathbb{E}\left[ \sum_{X \in \mathcal{X}} f(X) \right]
		= \int f(x) K(x,x) \mu(dx)
		= \operatorname{Tr}(Kf)
		= \operatorname{Tr}(fK)

Variance
--------

.. math::

	\operatorname{\mathbb{V}ar}\left[ \sum_{X \in \mathcal{X}} f(X) \right]
		&= \mathbb{E}\left[ \sum_{X \neq Y \in \mathcal{X}} f(X) f(Y)
							+ \sum_{X \in \mathcal{X}} f(X)^2 \right]
			- \mathbb{E}\left[ \sum_{X \in \mathcal{X}} f(X) \right]^2\\
		&= \iint f(x)f(y) [K(x,x)K(y,y)-K(x,y)K(y,x)] \mu(dx) \mu(dy)\\
			&\quad + \int f(x)^2 K(x,x) \mu(dx)
			- \left[\int f(x) K(x,x) \mu(dx)\right]^2 \\
		&= \int f(x)^2 K(x,x) \mu(dx)
			 - \iint f(x)f(y) K(x,y)K(y,x) \mu(dx) \mu(dy)\\
		&= \operatorname{Tr}(f^2K) - \operatorname{Tr}(fKfK)

a. Hermitian kernel i.e. :math:`K(x,y)=\overline{K(y,x)}`

	.. math::

		\operatorname{\mathbb{V}ar}\left[ \sum_{X \in \mathcal{X}} f(X) \right]
		= \int f(x)^2 K(x,x) \mu(dx) - \iint f(x)f(y) |K(x,y)|^2 \mu(dx) \mu(dy)

b. Orthogonal projection case i.e. :math:`K^2 = K = K^*`

	Using
	:math:`K(x,x) = \int K(x,y) K(y,x) \mu(dy) = \int |K(x,y)|^2 \mu(dy)`,

	.. math::

		\operatorname{\mathbb{V}ar}\left[ \sum_{X \in \mathcal{X}} f(X) \right]
		= \frac12 \iint [f(x) - f(y)]^2 |K(x,y)|^2 \mu(dy) \mu(dx)

.. _continuous_dpps_number_of_points:

Number of points
================

Based on :ref:`continuous_dpps_mixture`, we have

.. math::

	|\mathcal{X}|
		= \sum_{i=1}^{\infty}
			\operatorname{\mathcal{B}er}(\lambda_i)

.. important::

	- For any Borel set :math:`B`, instanciating :math:`f=1_{B}` yields nice expressions for the expectation and variance of the number of points falling in :math:`B`.
	- *Projection* DPPs have almost surely :math:`|\mathcal{X}| = \operatorname{Tr}(K) = \operatorname{rank}(K)` points (take :math:`f=1`).

.. seealso::

	:ref:`finite_dpps_nb_points` in the finite case

.. _continuous_dpps_thinning:

Thinning
========

Let :math:`\lambda > 1`.
The configuration of points :math:`\mathcal{X}^{\lambda}` obtained after subsampling the points of a configuration :math:`\mathcal{X}\sim \operatorname{DPP}(\mathbf{K})` with i.i.d. :math:`\operatorname{\mathcal{B}er}\left(\frac{1}{\lambda}\right)` is still a DPP with kernel :math:`\frac{1}{\lambda} \mathbf{K}`.

	.. math::

		\mathbb{E}\left[ \sum_{\substack{(x_1,\dots,x_k) \\ x_i \neq x_j \in \mathcal{X}^{\lambda}} } f(x_1,\dots,x_k) \right]
		&= \mathbb{E}\left[
				\mathbb{E}\left[
				\sum_{\substack{(x_1,\dots,x_k) \\ x_i \neq x_j \in \mathcal{X} } }
				f(x_1,\dots,x_k)
				\prod_{i=1}^k 1_{\{x_i \in \mathcal{X}^{\lambda} \}}
				\Bigg| \mathcal{X}\right]
				\right]\\
		&= \mathbb{E}\left[
						\sum_{\substack{(x_1,\dots,x_k) \\ x_i \neq x_j \in \mathcal{X} } }
						f(x_1,\dots,x_k)
						\mathbb{E}\left[ \prod_{i=1}^k B_i \Bigg| \mathcal{X} \right]
				\right]\\
		&= \mathbb{E}\left[
						\sum_{\substack{(x_1,\dots,x_k) \\ x_i \neq x_j \in \mathcal{X} } }
								f(x_1,\dots,x_k)
						\frac{1}{\lambda^k}
				\right]\\
		&= \int
				f(x_1,\dots,x_k)
				\det \left[ \frac{1}{\lambda} K(x_i,x_j) \right]_{1\leq i,j\leq k}
				\mu^{\otimes k}(dx) \\