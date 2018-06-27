.. _continuous_dpps_properties:

Properties
----------

Number of points
~~~~~~~~~~~~~~~~



Linear statistics
~~~~~~~~~~~~~~~~~

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