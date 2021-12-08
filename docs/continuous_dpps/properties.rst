.. _continuous_dpps_properties:

Properties
**********

.. _continuous_dpps_mixture:

Generic DPPs as mixtures of projection DPPs
===========================================

*Projection* DPPs are the building blocks of the model in the sense that generic DPPs are mixtures of *projection* DPPs.

Consider :math:`\mathcal{X} \sim \operatorname{DPP}(K)` and write the spectral decomposition of the corresponding kernel as

.. math::

	K = \sum_{n=1}^{\infty} \lambda_n \phi(x) \overline{\phi(y)}.

Then, denote :math:`\mathcal{X}^B \sim \operatorname{DPP}(K^B)` with

.. math::

	K = \sum_{n=1}^{\infty} B_n \phi(x) \overline{\phi(y)},
	\quad
	\text{where}
	\quad
	B_n \sim \mathcal{B}er(\lambda_n) \text{ are independent},

:math:`\mathcal{X}^B` is obtained by first sampling :math:`B_1, \dots, B_N` independently and then sampling conditionally from :math:`\operatorname{DPP}(K^B)`, the DPP with orthogonal projection kernel :math:`K^B`.

Finally, we have :math:`\mathcal{X} \overset{d}{=} \mathcal{X}^B`.

.. seealso::

	- Theorem 7 in :cite:`HKPV06`
	- Finite case of :ref:`finite_dpps_mixture`
	- :ref:`continuous_dpps_sampling`

.. _continuous_dpps_linear_statistics:

Linear statistics
=================

Expectation
-----------

.. math::

	\mathbb{E}\left[ \sum_{X \in \mathcal{X}} f(X) \right]
		= \int f(x) K(x,x) \mu(dx)
		= \operatorname{trace}(Kf)
		= \operatorname{trace}(fK).

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
		&= \operatorname{trace}(f^2K) - \operatorname{trace}(fKfK).

a. Hermitian kernel i.e. :math:`K(x,y)=\overline{K(y,x)}`

	.. math::

		\operatorname{\mathbb{V}ar}\left[ \sum_{X \in \mathcal{X}} f(X) \right]
		= \int f(x)^2 K(x,x) \mu(dx) - \iint f(x)f(y) |K(x,y)|^2 \mu(dx) \mu(dy).

b. Orthogonal projection case i.e. :math:`K^2 = K = K^*`

	Using
	:math:`K(x,x) = \int K(x,y) K(y,x) \mu(dy) = \int |K(x,y)|^2 \mu(dy)`,

	.. math::

		\operatorname{\mathbb{V}ar}\left[ \sum_{X \in \mathcal{X}} f(X) \right]
		= \frac12 \iint [f(x) - f(y)]^2 |K(x,y)|^2 \mu(dy) \mu(dx).

.. _continuous_dpps_number_of_points:

Number of points
================

For projection DPPs, i.e., when :math:`K` is the kernel associated to an orthogonal projector, one can show that :math:`|\mathcal{X}|=\operatorname{rank}(K)=\operatorname{trace}(K)` almost surely (see, e.g., :cite:`HKPV06` Lemma 17).

In the general case, based on the fact that :ref:`generic DPPs are mixtures of projection DPPs <finite_dpps_mixture>`, we have

.. math::

	|\mathcal{X}|
		= \sum_{i=1}^{\infty}
			\operatorname{\mathcal{B}er}(\lambda_i).

.. note::

	- For any Borel set :math:`B`, instantiating :math:`f=1_{B}` yields nice expressions for the expectation and variance of the number of points falling in :math:`B`.

.. seealso::

	:ref:`continuous_dpps_number_of_points` in the finite case

.. _continuous_dpps_thinning:

Thinning
========

.. important::

	The class of DPPs is closed under independent thinning.

Let :math:`\lambda > 1`.
The configuration of points :math:`\mathcal{X}^{\lambda}` obtained after subsampling the points of a configuration :math:`\mathcal{X}\sim \operatorname{DPP}(K)` with i.i.d. :math:`\operatorname{\mathcal{B}er}\left(\frac{1}{\lambda}\right)` is still a DPP with kernel :math:`\frac{1}{\lambda} K`.
To see this, let's compute the correlation functions of the thinned process

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
			\mu^{\otimes k}(dx).
