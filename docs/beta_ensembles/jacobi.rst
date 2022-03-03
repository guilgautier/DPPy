.. _beta_ensembles_jacobi:

Jacobi Ensemble
^^^^^^^^^^^^^^^

Take for reference measure :math:`\mu=\operatorname{Beta}(a,b)`

.. math::

	(x_1,\dots,x_N)
	\sim
		\left|\Delta(x_1,\dots,x_N)\right|^{\beta}
			%
		\prod_{i= 1}^N
					x_i^{a-1}
					(1-x_i)^{b-1}
			% \indic_{\bbR}(x_i)
		\ d x_i.


.. _jacobi_full_matrix_model:

Full model
^^^^^^^^^^^^^^^

Take for reference measure
:math:`\mu=\operatorname{Beta}\left(\frac{\beta}{2}(M_1-N+1), \frac{\beta}{2}(M_2-N+1)\right)`,
the pdf of the corresponding :math:`\beta`-Ensemble reads

.. math::

	(x_1,\dots,x_N)
	\sim
		\left|\Delta(x_1,\dots,x_N)\right|^{\beta}
	    %
		\prod_{i= 1}^N
	        x_i^{\frac{\beta}{2}(M_1-N+1)-1}
	        (1-x_i)^{\frac{\beta}{2}(M_2-N+1)-1}
			% \indic_{\bbR}(x_i)
		\ d x_i,

	where from the definition in :eq:`eq:abs_vandermonde_det` we have :math:`\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|`.

.. hint::

	The Jacobi ensemble (whose name comes from the fact that Jacobi polynomials are orthogonal w.r.t the Beta distribution) is associated with the multivariate analysis of variance (MANOVA) model.

- :math:`\beta=1`

.. math::

	\begin{cases}
		X \sim \mathcal{N}_{N, M_1}(0,1)\\
		Y \sim \mathcal{N}_{N, M_2}(0,1)\\
	\end{cases}
	\qquad
	A = XX^{\top}\left(XX^{\top} + YY^{\top}\right)^{-1}.

- :math:`\beta=2`

.. math::

	\begin{cases}
		X \sim \mathcal{N}_{N, M_1}(0,1) + i~ \mathcal{N}_{N, M_1}(0,1)\\
		Y \sim \mathcal{N}_{N, M_2}(0,1) + i~ \mathcal{N}_{N, M_2}(0,1)\\
	\end{cases}
	\qquad
	A = XX^{\dagger}\left(XX^{\dagger} + YY^{\dagger}\right)^{-1}.

- :math:`\beta=4`

.. math::

	\begin{cases}
		X_1 \sim \mathcal{N}_{N, M_1}(0,1) + i~ \mathcal{N}_{N, M_1}(0,1)\\
        X_2 \sim \mathcal{N}_{N, M_1}(0,1) + i~ \mathcal{N}_{N, M_1}(0,1)\\
        Y_1 \sim \mathcal{N}_{N, M_2}(0,1) + i~ \mathcal{N}_{N, M_2}(0,1)\\
		Y_2 \sim \mathcal{N}_{N, M_2}(0,1) + i~ \mathcal{N}_{N, M_2}(0,1)
	\end{cases}
	\qquad
    \begin{cases}
        X = \begin{bmatrix}
                X_1    & X_2   \\
                -X_2^* & X_1^*
            \end{bmatrix}\\
        Y = \begin{bmatrix}
                Y_1    & Y_2   \\
                -Y_2^* & Y_1^*
            \end{bmatrix}
    \end{cases}
    \qquad
	A = XX^{\dagger}\left(XX^{\dagger} + YY^{\dagger}\right)^{-1}.

Concentrates as Wachter law

.. math::

	\frac{(a+b)\sqrt{(\sigma_+-x)(x-\sigma_-)}}{2\pi x(1-x)}dx,

where

.. math::

	a = \frac{M_1}{N},
	b = \frac{M_2}{N}
	\quad\text{and}\quad
	\sigma_{\pm} = \left(\frac{\sqrt{a(a+b-1)} \pm \sqrt{b}}{a+b}\right)^2,

itself tending to the arcsine law in the limit.

.. :ref:`Fig. <jacobi_full_matrix_model_plot>`

.. _jacobi_full_matrix_model_plot:

.. plot:: plots/ex_plot_jacobi_full_matrix_model.py

	Full matrix model for the Jacobi ensemble

.. seealso::

	- :ref:`Banded matrix model <jacobi_banded_matrix_model>` for Jacobi ensemble
	- :py:mod:`~dppy.beta_ensembles.jacobi` in API
	- :ref:`multivariate_jacobi_ope`
	- :py:class:`~dppy.continuous.jacobi.JacobiProjectionDPP` in API


.. _jacobi_banded_matrix_model:

Banded model
^^^^^^^^^^^^^^^^^^^

.. note::

	Recall that from the definition in :eq:`eq:abs_vandermonde_det`

	.. math::

		\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|.

The equivalent tridiagonal model reads

.. math::

	\begin{bmatrix}
			\alpha_1    & \sqrt{\beta_2}& 0           &    0      & 0         \\
			\sqrt{\beta_2}  & \alpha_2    & \sqrt{\beta_3}&    0      & 0         \\
					0       & \ddots      & \ddots        & \ddots  & 0         \\
					0       & 0       & \sqrt{\beta_{N-1}} & \alpha_{N- 1}   & \sqrt{\beta_{N}} \\
					0       & 0        & 0            & \sqrt{\beta_N}  & \alpha_{N}
	\end{bmatrix}.

.. math::

	\begin{aligned}
			\alpha_1 &= \xi_1                   \quad & \\
			\alpha_k &= \xi_{2k-2}+\xi_{2k-1}    \quad &\beta_{k+1} &= \xi_{2k-1}\xi_{2k}
	\end{aligned}

	\begin{aligned}
			\xi_1 &= c_1             \quad &\gamma_1 &= 1-c_1 \\
			\xi_k &= (1-c_{k-1})c_k    \quad &\gamma_k &= c_{k-1}(1-c_k)
	\end{aligned},

with

.. math::

	c_{2i-1}
			\sim
			\operatorname{Beta}
			\left(
					\frac{\beta}{2}(N-i) + a,
					\frac{\beta}{2}(N-i) + b
			\right)
			\quad
	\text{and}
			\quad
	c_{2i}
			\sim
			\operatorname{Beta}
			\left(
					\frac{\beta}{2} (N-i),
					\frac{\beta}{2} (N-i-1) + a + b
			\right).

To recover the full matrix model for :ref:`laguerre_full_matrix_model`, recall that :math:`\Gamma(\frac{k}{2}, 2)\equiv \chi_k^2` and take

.. math::

	a = \frac{\beta}{2}(M_1-N+1)
	\quad \text{and} \quad
	b = \frac{\beta}{2}(M_2-N+1).

That is to say,

.. math::

	c_{2i-1}
			\sim
			\operatorname{Beta}
			\left(
					\frac{\beta}{2}(M_1-i+1),
					\frac{\beta}{2}(M_2-i+1)
			\right)
			\quad
	\text{and}
			\quad
	c_{2i}
			\sim
			\operatorname{Beta}
			\left(
					\frac{\beta}{2} (N-i),
					\frac{\beta}{2} (M_1+M_2-N-i+1)
			\right).

.. :ref:`Fig. <jacobi_banded_model_plot>`

.. _jacobi_banded_model_plot:

.. plot:: plots/ex_plot_jacobi_banded_matrix_model.py

	Tridiagonal matrix model for the Jacobi ensemble

.. seealso::

	- :cite:`KiNe04` Theorem 2
	- :ref:`Full matrix model <jacobi_full_matrix_model>` for Jacobi ensemble
	- :py:mod:`~dppy.beta_ensembles.jacobi` in API
	- :ref:`multivariate_jacobi_ope`
	- :py:class:`~dppy.continuous.jacobi.JacobiProjectionDPP` in API
