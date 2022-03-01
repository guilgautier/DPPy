.. _banded_matrix_models:

Banded matrix models
--------------------

Computing the eigenvalues of a full :math:`N\times N` random matrix is :math:`\mathcal{O}(N^3)`, and can thus become prohibitive for large :math:`N`.
A way to circumvent the problem is to adopt the equivalent banded models i.e. diagonalize banded matrices.

The first tridiagonal models for the :ref:`hermite_banded_matrix_model` and :ref:`laguerre_banded_matrix_model` were revealed by :cite:`DuEd02`, who left the :ref:`jacobi_banded_matrix_model` as an open question, addressed by :cite:`KiNe04`.
Such tridiagonal formulations made sampling possible at cost :math:`\mathcal{O}(N^2)` but also unlocked sampling for generic :math:`\beta>0`!

Note that :cite:`KiNe04` also derived a quindiagonal model for the :ref:`circular_banded_matrix_model`.

.. _hermite_banded_matrix_model:

Hermite Ensemble
^^^^^^^^^^^^^^^^

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
	- :py:class:`~dppy.beta_ensembles.HermiteEnsemble` in API

.. _laguerre_banded_matrix_model:

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
	- :py:class:`~dppy.beta_ensembles.LaguerreEnsemble` in API

.. _jacobi_banded_matrix_model:

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
	- :py:class:`~dppy.beta_ensembles.JacobiEnsemble` in API
	- :ref:`multivariate_jacobi_ope`
	- :py:class:`~dppy.continuous.jacobi.JacobiProjectionDPP` in API

.. _circular_banded_matrix_model:

Circular Ensemble
^^^^^^^^^^^^^^^^^

.. math::

	\left|\Delta(e^{i \theta_1},\dots, e^{i \theta_N})\right|^{\beta}
		\prod_{j = 1}^N \frac{1}{2\pi} \mathbf{1}_{[0,2\pi]} (\theta_j) d\theta_j.

.. note::

	Recall that from the definition in :eq:`eq:abs_vandermonde_det`

	.. math::

		\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|.

.. important::

	Consider the distribution :math:`\Theta_{\nu}` that for integers :math:`\nu\geq2` is defined as follows:

	Draw :math:`v` uniformly at random from the unit sphere :math:`\mathbb{S}^{\nu} \in \mathbb{R}^{\nu+1}`, then :math:`v_1 + iv_2\sim \Theta_{\nu}`

Now, given :math:`\beta\in \mathbb{N}^*`, let

- :math:`\alpha_k\sim \Theta_{\beta(N-k-1)+1}` independent variables
- for :math:`0\leq k\leq N-1` set :math:`\rho_k = \sqrt{1-|\alpha_k|^2}`.

Then, the equivalent quindiagonal model corresponds to the eigenvalues of either :math:`LM` or :math:`ML` with

.. math::

	L = \operatorname{diag}[\Xi_0,\Xi_2,\dots]
	\quad \text{and} \quad
	M = \operatorname{diag}[\Xi_{-1},\Xi_1,\Xi_3\dots],

and where

.. math::

	\Xi_k =
	\begin{bmatrix}
	\overline{\alpha}_k & \rho_k\\
	\rho_k   & -\alpha_k
	\end{bmatrix}
	, \quad 0\leq k\leq N-2
	, \quad \text{with} \quad
	\Xi_{-1} = [1]
	\quad \text{and} \quad
	\Xi_{N-1} = [\overline{\alpha}_{N-1}].

.. hint::

	The effect of increasing the :math:`\beta` parameter can be nicely visualized on this :ref:`circular_banded_matrix_model`.
	Viewing :math:`\beta` as the inverse temperature, the configuration of the eigenvalues crystallizes with :math:`\beta`, see the figure below.

.. :ref:`Fig. <circular_banded_model_plot>`

.. _circular_banded_model_plot:

.. plot:: plots/ex_plot_circular_banded_matrix_model.py

	Quindiagonal matrix model for the Circular ensemble

.. seealso::

	- :cite:`KiNe04` Theorem 1
	- :ref:`Full matrix model <circular_full_matrix_model>` for Circular ensemble
	- :py:class:`~dppy.beta_ensembles.CircularEnsemble` in API
