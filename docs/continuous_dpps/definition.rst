.. _continuous_dpps_definition:

Definition
**********

Point Process
=============

Let :math:`\mathbb{X} = \mathbb{R}^d, \mathbb{C}^d \text{ or } \mathbb{S}^{d-1}` be the ambient space, we endow it with the corresponding Borel :math:`\sigma`-algebra :math:`\mathcal{B}(\mathbb{X})` together with a reference measure :math:`\mu`.

For our purpose, we consider point processes as locally finite random subsets :math:`\mathcal{X} \subset \mathbb{X}` i.e.

	.. math::

		\forall C \subset \mathbb{X} \text{ compact},
			\quad \#(\mathcal{X} \cap C) < \infty.

.. hint::

	A point process is a random subset of points
	:math:`\mathcal{X} \triangleq\{X_1, \dots, X_N\} \subset \mathbb{X}`
	with :math:`N` being random.

.. seealso::

	More formal definitions can be found in :cite:`MoWa04` Section 2 and :cite:`Joh06` Section 2 and bibliography therein.

To understand the interaction between the points of a point process, one focuses on the interaction of each cloud of :math:`k` points (for all :math:`k`).
The corresponding :math:`k`-correlation functions characterize the underlying point process.

.. _continuous_dpps_correlation_functions:

Correlation functions
=====================

For :math:`k\geq 0`, the :math:`k`-correlation function :math:`\rho_k` is defined by:

:math:`\forall f : \mathbb{X}^k \to \mathbb{C}` bounded measurable

.. math::

  \mathbb{E}
  \left[ \sum_{
    \substack{
    	(X_1,\dots,X_k) \\
    	X_1 \neq \dots \neq X_k \in \mathcal{X}} }
    f(X_1,\dots,X_k)
  \right]
	  = \int_{\mathbb{X}^k}
	  	f(x_1,\dots,x_k) \rho_k(x_1,\dots,x_k)
	  	\prod_{i=1}^k \mu(dx_i).

.. hint::

	The :math:`k`-correlation function does not always exists, but but when they do, they have a meaningful interpretation.

	.. math::

		"
		\rho_k(x_1,\dots,x_k)
		\mu(dx_{1}), \dots, \mu(dx_{N})
		= \mathbb{P}
		\left[
		\begin{array}{c}
			\text{there is 1 point in each}\\
			B(x_1, d x_1), \dots, B(x_n, d x_n)
		\end{array}
		\right]
		",

    where :math:`B(x, dx)` denotes the ball centered at :math:`x` with radius :math:`dx`.

A Determinant Point Process (DPP) is a point process on :math:`(\mathbb{X}, \mathcal{B}(\mathbb{X}), \mu)` parametrized by a kernel :math:`K` associated to the reference measure :math:`\mu`.
The :math:`k`-correlation functions read

.. math::

	\forall k\geq 1, \quad
	\rho_k(x_1,\dots,x_k)
		= \det [K(x_i, x_j)]_{i,j=1}^k.

.. seealso::

	:cite:`Mac75`
	:cite:`Sos00`
	:cite:`Joh06`
	:cite:`HKPV06`

.. _continuous_dpps_existence:

Existence
=========

One can view :math:`K` as an integral operator on :math:`L^2(\mu)`

.. math::

	\forall x \in \mathbb{X},
	Kf(x) = \int_{\mathbb{X}} K(x,y) f(y) \mu(dy).

To access spectral properties of the kernel, it is common practice to assume :math:`K`

1. `Hilbert Schmidt <https://en.wikipedia.org/wiki/Hilbert%E2%80%93Schmidt_integral_operator>`_

	.. math::

		\iint_{\mathbb{X}\times \mathbb{X}}
			|K(x,y)|^2
			\mu(dx) \mu(dy)
		< \infty,

2. Self-adjoint equiv. Hermitian

	.. math::

		K(x,y) = \overline{K(y,x)},

3. Locally trace class

	.. math::

		\forall B\subset \mathbb{X} \text{ compact}, \quad
		\int_B K(x,x) \mu(dx) < \infty.

.. hint::

	- 1. implies :math:`K` to be a continuous compact operator.

	- 2. with 1. allows to apply the spectral theorem, providing

		.. math::

			K(x,y) = \sum_{n=0}^{\infty} \lambda_n \phi_{n}(x)\phi_{n}(y), \quad \text{where } K\phi_{n} = \lambda_n \phi_{n}.

	- 3. makes sure there is no accumulation of points: :math:`|\mathcal{X}\cap B| = \int_B K(x,x) \mu(dx) \leq \infty`, see also :ref:`continuous_dpps_number_of_points`

.. warning::

	These are only sufficient conditions, there indeed exist DPPs with non symmetric kernels, see, e.g., :ref:`carries_process`.

.. important::

	Under assumptions 1, 2, and 3

	.. math::

		\operatorname{DPP}(K) \text{ exists}
		\Longleftrightarrow
			0\leq \lambda_n \leq 1, \quad \forall n \in \mathbb{N}

.. seealso::

	- Remarks 1-2 and Theorem 3 :cite:`Sos00`
	- Theorem 22 :cite:`HKPV06`

.. _continuous_dpps_projection_dpps:

Projection DPPs
===============

:math:`\operatorname{DPP}(K)` is said to be a **projection** DPP with reference measure :math:`\mu` when :math:`K:\mathbb{X}\times \mathbb{X}\to \mathbb{C}` is a orthogonal projection kernel, that is

.. math::

    K(x,y)=\overline{K(y,x)}
    \quad\text{and}\quad
    \int_{\mathbb{X}} K(x, z) K(z, y) \mu(d z) = K(x, y)

.. _continuous_dpps_construction:

Construction
============

A canonical way to construct DPPs generating configurations of at most :math:`N` points is the following.

Consider :math:`N` orthonormal functions :math:`\phi_{0},...,\phi_{N−1} \in L^2(\mu)`

.. math::

	\int \phi_{k}(x)\phi_{l}(x)\mu(dx) = \delta_{kl},

and attach :math:`[0,1]`-valued coefficients :math:`\lambda_n` such that

.. math::

	K(x, y) = \sum_{n=0}^{N-1} \lambda_n \phi_{n}(x)\phi_{n}(y).

The special case where :math:`\lambda_0=\cdots=\lambda_{N-1}=1` corresponds to the construction of a projection DPP with :math:`N` points.

.. seealso::

	- :ref:`continuous_dpps_number_of_points`
	- Lemma 21 :cite:`HKPV06`
	- Proposition 2.11 :cite:`Joh06` biorthogonal families
