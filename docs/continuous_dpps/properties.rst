.. _continuous_dpps_properties:

Properties
**********

.. important::

	The :ref:`finite_dpps_properties` presented in the finite case regarding see :ref:`finite_dpps_mixture` and :ref:`finite_dpps_nb_points` are still valid!

Thinning
========

Let :math:`\lambda > 1`.
The configuration of points :math:`\gamma^{\lambda}` obtained after subsampling the points of a configuration :math:`\gamma^{\lambda}\sim \operatorname{DPP}(\mathbf{K})` with i.i.d. :math:`\operatorname{\mathcal{B}er}\left(\frac{1}{\lambda}\right)` is still a DPP with kernel :math:`\frac{1}{\lambda} \mathbf{K}`.

	.. math::
	
		\mathbb{E}\left[ \sum_{\substack{(x_1,\dots,x_k) \\ x_i \neq x_j \in \gamma^{\lambda}} } f(x_1,\dots,x_k) \right]
		&= \mathbb{E}\left[ 
				\mathbb{E}\left[ 
				\sum_{\substack{(x_1,\dots,x_k) \\ x_i \neq x_j \in \gamma } } 
				f(x_1,\dots,x_k) 
				\prod_{i=1}^k 1_{\{x_i \in \gamma^{\lambda} \}}
				\Bigg| \gamma\right]
				\right]\\
		&= \mathbb{E}\left[ 
						\sum_{\substack{(x_1,\dots,x_k) \\ x_i \neq x_j \in \gamma } } 
						f(x_1,\dots,x_k) 
						\mathbb{E}\left[ \prod_{i=1}^k B_i \Bigg| \gamma \right]
				\right]\\
		&= \mathbb{E}\left[ 
						\sum_{\substack{(x_1,\dots,x_k) \\ x_i \neq x_j \in \gamma } } 
								f(x_1,\dots,x_k)
						\frac{1}{\lambda^k}
				\right]\\
		&= \int
				f(x_1,\dots,x_k)
				\det \left[ \frac{1}{\lambda} K(x_i,x_j) \right]_{1\leq i,j\leq k}  
				\mu^{\otimes k}(dx) \\

Linear statistics
=================

1. Expectation 

	.. math::

		\mathbb{E}\left[ \sum_{X \in \gamma} f(X) \right] 
			= \int f(x) K(x,x) \mu(dx)
			= \operatorname{Tr}(Kf)
			= \operatorname{Tr}(fK)

2. Variance

	.. math::

		\operatorname{\mathbb{V}ar}\left[ \sum_{X \in \gamma} f(X) \right] 
			&= \int f(x)^2 K(x,x) \mu(dx) 
			- \int f(x) K(x,x) \mu(dx) \\
			+& \iint f(x)f(y) [K(x,x)K(y,y)-K(x,y)K(y,x)] \mu(dx) \mu(dy)\\
			&= \int f(x)^2 K(x,x) \mu(dx) 
				 - \iint f(x)f(y) K(x,y)K(y,x) \mu(dx) \mu(dy)\\
			&= \operatorname{Tr}(f^2K) - \operatorname{Tr}(fKfK)

	a. Hermitian kernel i.e. :math:`K(x,y)=\overline{K(y,x)}`

		.. math::

			\operatorname{\mathbb{V}ar}\left[ \sum_{X \in \gamma} f(X) \right] 
			= \int f(x)^2 K(x,x) \mu(dx) - \iint f(x)f(y) |K(x,y)|^2 \mu(dx) \mu(dy)

	b. Orthogonal projection case

		Using 
		:math:`K(x,x) = \int K(x,y) K(y,x) \mu(dy) = \int |K(x,y)|^2 \mu(dy)`,

		.. math::

			\operatorname{\mathbb{V}ar}\left[ \sum_{X \in \gamma} f(X) \right]
			= \frac12 \iint [f(x) - f(y)]^2 |K(x,y)|^2 \mu(dy) \mu(dx)

.. note::

	Taking :math:`f = 1_{B}` allows to express the expectation and variance of the number of points in e.g. a bounded set :math:`B`.