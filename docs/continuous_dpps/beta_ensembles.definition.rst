.. _beta_ensembles_definition:

Definition
==========

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
	
	- :math:`|\Delta(x_1,\dots,x_N)| = \prod_{i<j} |x_i - x_j|` is the absolute value of the determinant of the Vandermonde matrix, 

		.. math::
			:label: abs_vandermonde_det

			\Delta(x_1,\dots,x_N)
			= \det \begin{bmatrix}
				1 				& \dots 	& 1				\\
				x_1 			& \dots 	& x_N			\\
				\vdots		& 				& \vdots	\\
				x_1^{N-1}	&					&x_N^{N-1}
			\end{bmatrix}

		encoding the repulsive interaction.
		The *closer* the points are the lower the density.

	- :math:`\beta` is the inverse temperature parameter quantifying the strength of the repulsion between the points.

.. important:: 

	For Gaussian, Gamma and Beta reference measures, the :math:`\beta=1,2` and :math:`4` cases received a very special attention in the random matrix literature, e.g. :cite:`DuEd02`.

	The associated ensembles actually correspond to the eigenvalues of random matrices whose distribution is invariant to the action of the orthogonal (:math:`\beta=1`), unitary (:math:`\beta=2`) and symplectic (:math:`\beta=4`) group respectively.

	+---------------+---------------------+----------------------+---------------------------------------+
	| :math:`\mu`   | :math:`\mathcal{N}` | :math:`\Gamma`       | :math:`\operatorname{\mathcal{B}eta}` |
	+===============+=====================+======================+=======================================+
	| Ensemble name | Hermite             | Laguerre             | Jacobi                                |
	+---------------+---------------------+----------------------+---------------------------------------+
	| support       | :math:`\mathbb{R}`  | :math:`\mathbb{R}^+` | :math:`[0,1]`                         |
	+---------------+---------------------+----------------------+---------------------------------------+

.. note::

	The study of the distribution of the eigenvalues of random orthogonal, unitary and symplectic matrices lying on the unit circle is also very thorough :cite:`KiNe04`.

.. _beta_ensembles_definition_OPE:

Orthogonal Polynomial Ensembles
-------------------------------

The case :math:`\beta=2` corresponds a specific type of *projection* DPPs also called Orthogonal Polynomial Ensembles (OPEs) :cite:`Kon05` with associated kernel

.. math::

	K_N(x, y) = \sum_{n=0}^{N-1} p_n(x) p_n(y)

where :math:`(p_n)` are the orthonormal polynomials w.r.t. :math:`\mu` i.e. :math:`\operatorname{deg}(p_n)=n` and :math:`\langle p_k, p_l \rangle_{L^2(\mu)}=\delta_{kl}`.

.. note::

	OPEs (with :math:`N` points) correspond to *projection* DPPs onto 
	:math:`\operatorname{Span}\{p_n\}_{n=0}^{N-1} = \mathbb{R}^{N-1}[X]``

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

	:cite:`Kon05`, :cite:`Joh06`