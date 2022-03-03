
.. _beta_ensembles_laguerre:

Laguerre Ensemble
^^^^^^^^^^^^^^^^^

Take for reference measure :math:`\mu=\Gamma(k,\theta)`

.. math::

	(x_1,\dots,x_N)
	\sim
		\left|\Delta(x_1,\dots,x_N)\right|^{\beta}
			%
		\prod_{i= 1}^N
					x_i^{k-1}
			e^{- \frac{x_i}{\theta}}
			% \indic_{\bbR}(x_i)
		\ d x_i.


.. _laguerre_full_matrix_model:

Full model
^^^^^^^^^^^^^^^^^

Take for reference measure
:math:`\mu=\Gamma\left(\frac{\beta}{2}(M-N+1), 2\right)=\chi_{\beta(M-N+1)}^2`,
the pdf of the corresponding :math:`\beta`-Ensemble reads

.. math::

	(x_1,\dots,x_N)
	\sim
		\left|\Delta(x_1,\dots,x_N)\right|^{\beta}
	    %
		\prod_{i= 1}^N
	        x_i^{\frac{\beta}{2}(M-N+1)-1}
			e^{- \frac12 x_i}
			% \indic_{\bbR}(x_i)
		\ d x_i,

	where from the definition in :eq:`eq:abs_vandermonde_det` we have :math:`\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|`.

.. hint::

	The Laguerre ensemble (whose name comes from the fact that Laguerre polynomials are orthogonal w.r.t the Gamma distribution) refers to the eigenvalue distribution of empirical covariance matrices of i.i.d. Gaussian vectors.

- :math:`\beta=1`

.. math::

	X \sim \mathcal{N}_{N, M}(0,1)
	\qquad
	A = XX^{\top}.

- :math:`\beta=2`

.. math::

	X \sim \mathcal{N}_{N, M}(0,1) + i~ \mathcal{N}_{N, M}(0,1)
	\qquad
	A = XX^{\dagger}.

- :math:`\beta=4`

.. math::

	\begin{cases}
        X \sim \mathcal{N}_{N, M}(0,1) + i~ \mathcal{N}_{N, M}(0,1)\\
        Y \sim \mathcal{N}_{N, M}(0,1) + i~ \mathcal{N}_{N, M}(0,1)
    \end{cases}
    \qquad
	A = \begin{bmatrix}
            X    & Y   \\
            -Y^* & X^*
        \end{bmatrix}
    \quad A = A A^{\dagger}.

Normalization :math:`\beta M` to concentrate as Marcenko-Pastur law.

.. math::

	\frac{1}{2\pi}
	\frac{\sqrt{(\lambda_+-x)(x-\lambda_-)}}{cx}
	1_{[\lambda_-,\lambda_+]}
	dx,

where

.. math::

	c = \frac{M}{N}
	\quad \text{and} \quad
	\lambda_\pm = (1\pm\sqrt{c})^2.

.. :ref:`Fig. <laguerre_full_matrix_model_plot>`

.. _laguerre_full_matrix_model_plot:

.. plot:: plots/ex_plot_laguerre_full_matrix_model.py

	Full matrix model for the Laguerre ensemble

.. seealso::

	- :ref:`Banded matrix model <laguerre_banded_matrix_model>` for Laguerre ensemble
	- :py:mod:`~dppy.beta_ensembles.laguerre` in API


.. _laguerre_banded_matrix_model:

Banded matrix model
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
	\end{bmatrix}
	=
	\begin{bmatrix}
			\sqrt{\xi_1}  &         &             &         \\
			\sqrt{\xi_2}    & \sqrt{\xi_3}  &           &         \\
								& \ddots    & \ddots    &       \\
									&           & \sqrt{\xi_{2N-2}} & \sqrt{\xi_{2N-1}}
	\end{bmatrix}
	\begin{bmatrix}
			\sqrt{\xi_1}  & \sqrt{\xi_2}  &           &                  \\
								& \sqrt{\xi_3}  & \ddots    &                  \\
								&         & \ddots  & \sqrt{\xi_{2N-2}} \\
									&           &     & \sqrt{\xi_{2N-1}}
	\end{bmatrix},

with

.. math::

	\xi_{2i-1}
			\sim \Gamma\left(\frac{\beta}{2}(N - i) + k, \theta \right)
			\quad
	\text{and}
			\quad
	\xi_{2i}
			\sim \Gamma\left(\frac{\beta}{2}(N - i), \theta \right).

To recover the full matrix model for :ref:`laguerre_full_matrix_model`, recall that :math:`\Gamma(\frac{k}{2}, 2)\equiv \chi_k^2` and take

.. math::

	k = \frac{\beta}{2}(M-N+1)
	\quad \text{and} \quad
	\theta = 2.

That is to say,

.. math::

	\xi_{2i-1}
			\sim \chi_{\beta(M - i + 1)}^2
			\quad
	\text{and}
			\quad
	\xi_{2i}
			\sim \chi_{\beta(N - i)}^2.

.. :ref:`Fig. <laguerre_banded_model_plot>`

.. _laguerre_banded_model_plot:

.. plot:: plots/ex_plot_laguerre_banded_matrix_model.py

	Tridiagonal matrix model for the Laguerre ensemble

.. seealso::

	- :cite:`DuEd02` III-B
	- :ref:`Full matrix model <laguerre_full_matrix_model>` for Laguerre ensemble
	- :py:mod:`~dppy.beta_ensembles.laguerre` in API
