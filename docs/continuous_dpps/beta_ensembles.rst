.. _beta_ensembles:

:math:`\beta`-Ensembles
#######################

.. _beta_ensembles_definition:

Definition
----------

Let :math:`\beta>0`, the joint distribution of the :math:`\beta`-Ensemble associated to the reference measure :math:`\mu` writes

.. math::
	:label: joint_beta_ensemble

	(x_1,\dots,x_N) 
	\sim 
		\frac{1}{Z_{N,\beta}}
		\left|\Delta(x_1,\dots,x_N)\right|^{\beta}
		\prod_{i= 1}^N 
			\mu(d x_i)

.. hint::
	
	- :math:`|\Delta(x_1,\dots,x_N)| = \prod_{i<j} (x_i - x_j)` is the  determinant of the Vandermonde matrix, 

	.. math::

		\Delta(x_1,\dots,x_N)
		= \begin{bmatrix}
			1 				& \dots 	& 1				\\
			x_1 			& \dots 	& x_N			\\
			\vdots		& 				& \vdots	\\
			x_1^{N-1}	&					&x_N^{N-1}
		\end{bmatrix}

	  encoding the repulsive interaction.
	  The *closer* the points are the lower the density.

	- :math:`\beta` is the inverse temperature parameter quantifying the strength of the repulsion between the points.

.. todo::

	The cases :math:`\beta=1,2,4` take roots in random matrix theory.......

	.. seealso::

		:ref:`beta_ensembles_sampling`

Orthogonal Polynomial Ensembles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The case :math:`\beta=2` corresponds a specific type of *projection* DPPs also called Orthogonal Polynomial Ensembles (OPEs) with associated kernel

.. math::

	K_N(x, y) = \sum_{n=0}^{N-1} p_n(x) p_n(y)

where :math:`(p_n)` are the orthonormal polynomials w.r.t. :math:`\mu` i.e. :math:`\langle p_k, p_l \rangle_{L^2(\mu)}=\delta_{kl}`.

.. note::

	OPEs (with :math:`N` points) correspond to *projection* DPPs onto

	.. math::

		\operatorname{Span}\{p_n\}_{n=0}^{N-1} = \mathbb{R}^{N-1}[X]

.. hint::

	First, linear combinations of the rows of :math:`\Delta(x_1,\dots,x_N)` allow to make appear the orthonormal polynomials :math:`(p_n)` so that

	.. math::

		|\Delta(x_1,\dots,x_N)|
			\propto 
			\begin{vmatrix}
				p_0(x_1) 			& \dots 	& p_0(x_N) 		\\
				p_1(x_1) 			& \dots 	& p_1(x_N) 		\\
				\vdots				& 				& \vdots      \\ 
				p_{N-1}(x_1)	&					& p_{N-1}(x_N)
			\end{vmatrix}

	Then,

	.. math::

		|\Delta|^2 
			= | \Delta^{\top} \Delta |
			\propto \det \left[ K_N(x_i, x_j)\right]_{i,j=1}^N

	Finally, the joint distribution of :math:`(x_1, \dots, x_N)` reads
 
	.. math::
		:label: joint_OPE

		(x_1,\dots,x_N) 
		\sim 
			\frac{1}{N!}
			\det \left[ K_N(x_i, x_j)\right]_{i,j=1}^N
			\prod_{i= 1}^N 
				\mu(d x_i)

.. seealso::

	.. todo::

		cite Johansson, Konig

.. _beta_ensembles_sampling:

Sampling
--------

Full matrix models
~~~~~~~~~~~~~~~~~~

beta = 1, 2, 4

Diagonalization of full random


Banded models
~~~~~~~~~~~~~

:cite:`DuEd02`, :cite:`KiNe04`
