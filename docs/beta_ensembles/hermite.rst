.. _beta_ensembles_hermite:

Hermite Ensemble
----------------

Take for reference measure :math:`\mu=\mathcal{N}(\mu, \sigma)`

.. math::

	(x_1,\dots,x_N)
	\sim
		\left|\Delta(x_1,\dots,x_N)\right|^{\beta}
		\prod_{i= 1}^N
			e^{- \frac{(x_i-\mu)^2}{2\sigma^2}}
			% \indic_{\bbR}(x_i)
		\ d x_i.

.. note::

	Recall that from the definition in :eq:`eq:abs_vandermonde_det`

	.. math::

		\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|.


.. _hermite_full_matrix_model:

Full model
^^^^^^^^^^^^^^^^

Take for reference measure :math:`\mu=\mathcal{N}(0, 2)`, the pdf of the corresponding :math:`\beta`-Ensemble reads

.. math::

	(x_1,\dots,x_N)
	\sim
		\left|\Delta(x_1,\dots,x_N)\right|^{\beta}
		\prod_{i= 1}^N
			e^{- \frac{1}{2}\frac{x_i^2}{2}}
			% \indic_{\bbR}(x_i)
		\ d x_i,

	where from the definition in :eq:`eq:abs_vandermonde_det` we have :math:`\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|`.

.. hint::

	The Hermite ensemble (whose name comes from the fact that Hermite polynomials are orthogonal w.r.t the Gaussian distribution) refers to the eigenvalue distribution of random matrices formed by i.i.d. Gaussian vectors.

- :math:`\beta=1`

.. math::

	X \sim \mathcal{N}_{N, N}(0,1)
	\qquad
	A = \frac{X+X^{\top}}{\sqrt{2}}.

- :math:`\beta=2`

.. math::

	X \sim \mathcal{N}_{N, N}(0,1) + i~ \mathcal{N}_{N, N}(0,1)
	\qquad
	A = \frac{X+X^{\dagger}}{\sqrt{2}}.

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
    \quad A = \frac{X+X^{\dagger}}{\sqrt{2}}.

Normalization :math:`\sqrt{\beta N}` to concentrate as the semi-circle law.

.. math::

	\frac{\sqrt{4-x^2}}{2\pi} 1_{[-2,2]} dx.

.. :ref:`Fig. <hermite_full_matrix_model_plot>`

.. _hermite_full_matrix_model_plot:

.. plot:: plots/ex_plot_hermite_full_matrix_model.py

	Full matrix model for the Hermite ensemble

.. seealso::

	- :ref:`Banded matrix model <hermite_banded_matrix_model>` for Hermite ensemble
	- :py:mod:`~dppy.beta_ensembles.hermite` in API


.. _hermite_banded_model:

Banded model
^^^^^^^^^^^^^^^^^^^^

The equivalent tridiagonal model reads

.. math::

	\begin{bmatrix}
			\alpha_1    & \sqrt{\beta_2}& 0           &    0      & 0         \\
			\sqrt{\beta_2}  & \alpha_2    & \sqrt{\beta_3}&    0      & 0         \\
					0       & \ddots      & \ddots        & \ddots  & 0         \\
					0       & 0       & \sqrt{\beta_{N-1}} & \alpha_{N- 1}   & \sqrt{\beta_{N}} \\
					0       & 0        & 0            & \sqrt{\beta_N}  & \alpha_{N}
	\end{bmatrix},

with

.. math::

	\alpha_{i}
			\sim \mathcal{N}(\mu, \sigma^2)
			\quad
	\text{and}
			\quad
	\beta_{i+1}
			\sim \Gamma\left(\frac{\beta}{2}(N - i), \sigma^2\right).

To recover the full matrix model for :ref:`hermite_full_matrix_model`, recall that :math:`\Gamma(\frac{k}{2}, 2)\equiv \chi_k^2` and take

.. math::

	\mu = 0
	\quad \text{and} \quad
	\sigma^2 = 2.

That is to say,

.. math::

	\alpha_{i}
			\sim \mathcal{N}(0, 2)
			\quad
	\text{and}
			\quad
	\beta_{i+1}
			\sim \chi_{\beta(N - i)}^2.

.. :ref:`Fig. <hermite_banded_model_plot>`

.. _hermite_banded_model_plot:

.. plot:: plots/ex_plot_hermite_banded_matrix_model.py

	Tridiagonal matrix model for the Hermite ensemble

.. seealso::

	- :cite:`DuEd02` II-C
	- :ref:`Full matrix model <hermite_full_matrix_model>` for Hermite ensemble
	- :py:mod:`~dppy.beta_ensembles.hermite` in API
