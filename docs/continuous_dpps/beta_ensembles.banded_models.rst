.. _banded_matrix_models:

Banded matrix models
--------------------

Computing the eigenvalues of a full :math:`N\times N` random matrix is :math:`\mathcal{O}(N^3)`, and can thus become prohibitive for large :math:`N`.
A way to circumvent the problem is to adopt the equivalent banded models i.e. diagonalize banded matrices.

The first tridiagonal models for the :ref:`hermite_ensemble_banded` and :ref:`laguerre_ensemble_banded` were revealed by :cite:`DuEd02`, who left the :ref:`jacobi_ensemble_banded` as an open question, addressed by :cite:`KiNe04`.
Such tridiagonal formulations made sampling possible at cost :math:`\mathcal{O}(N^2)` but also unlocked sampling for generic :math:`\beta>0`!

Note that :cite:`KiNe04` also derived a quindiagonal model for the :ref:`circular_ensemble_banded`.

.. _hermite_ensemble_banded:

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
		\ d x_i

.. note::

	Recall that from the definition in :eq:`abs_vandermonde_det`
	
	.. math::

		\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|


The equivalent tridiagonal model reads

.. math::

	\begin{bmatrix}
			\alpha_1    & \sqrt{\beta_2}& 0           &    0      & 0         \\
			\sqrt{\beta_2}  & \alpha_2    & \sqrt{\beta_3}&    0      & 0         \\
					0       & \ddots      & \ddots        & \ddots  & 0         \\
					0       & 0       & \sqrt{\beta_{N-1}} & \alpha_{N- 1}   & \sqrt{\beta_{N}} \\
					0       & 0        & 0            & \sqrt{\beta_N}  & \alpha_{N}
	\end{bmatrix}

with

.. math::

	\alpha_{i}
			\sim \mathcal{N}(\mu, \sigma^2)
			\quad
	\text{and}
			\quad
	\beta_{i+1}
			\sim \Gamma\left(\frac{\beta}{2}(N - i), \sigma^2\right)
	

To recover the full matrix model for :ref:`hermite_ensemble_full`, recall that :math:`\Gamma(\frac{k}{2}, 2)\equiv \chi_k^2` and take

.. math::

	\mu = 0
	\quad \text{and} \quad
	\sigma^2 = 2

That is to say,

.. math::

	\alpha_{i}
			\sim \mathcal{N}(0, 2)
			\quad
	\text{and}
			\quad
	\beta_{i+1}
			\sim \chi_{\beta(N - i)}^2
	

.. plot:: plots/ex_plot_hermite_banded.py
	:include-source:

.. seealso::

	- :cite:`DuEd02` II-C
	- :ref:`Full matrix model <hermite_ensemble_full>` for Hermite ensemble

.. _laguerre_ensemble_banded:

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
		\ d x_i

.. note::

	Recall that from the definition in :eq:`abs_vandermonde_det`
	
	.. math::

		\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|


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
	\end{bmatrix}


with 

.. math::

	\xi_{2i-1}
			\sim \Gamma\left(\frac{\beta}{2}(N - i) + k, \theta \right)
			\quad
	\text{and}
			\quad
	\xi_{2i}
			\sim \Gamma\left(\frac{\beta}{2}(N - i), \theta \right)


To recover the full matrix model for :ref:`laguerre_ensemble_full`, recall that :math:`\Gamma(\frac{k}{2}, 2)\equiv \chi_k^2` and take 

.. math::

	k = \frac{\beta}{2}(M-N+1)
	\quad \text{and} \quad
	\theta = 2

That is to say,

.. math::

	\xi_{2i-1}
			\sim \chi_{\beta(M - i + 1)}^2
			\quad
	\text{and}
			\quad
	\xi_{2i}
			\sim \chi_{\beta(N - i)}^2

.. plot:: plots/ex_plot_laguerre_banded.py
	:include-source:

.. seealso::

	- :cite:`DuEd02` III-B
	- :ref:`Full matrix model <laguerre_ensemble_full>` for Laguerre ensemble

.. _jacobi_ensemble_banded:

Jacobi Ensemble
^^^^^^^^^^^^^^^

Take for reference measure :math:`\mu=\operatorname{\mathcal{B}eta}(a,b)`

.. math::

	(x_1,\dots,x_N) 
	\sim 
		\left|\Delta(x_1,\dots,x_N)\right|^{\beta}
			%
		\prod_{i= 1}^N 
					x_i^{a-1}
					(1-x_i)^{b-1}
			% \indic_{\bbR}(x_i)
		\ d x_i

.. note::

	Recall that from the definition in :eq:`abs_vandermonde_det`
	
	.. math::

		\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|


The equivalent tridiagonal model reads

.. math::

	\begin{bmatrix}
			\alpha_1    & \sqrt{\beta_2}& 0           &    0      & 0         \\
			\sqrt{\beta_2}  & \alpha_2    & \sqrt{\beta_3}&    0      & 0         \\
					0       & \ddots      & \ddots        & \ddots  & 0         \\
					0       & 0       & \sqrt{\beta_{N-1}} & \alpha_{N- 1}   & \sqrt{\beta_{N}} \\
					0       & 0        & 0            & \sqrt{\beta_N}  & \alpha_{N}
	\end{bmatrix}


.. math::

	\begin{aligned}
			\alpha_1 &= \xi_1                   \quad & \\
			\alpha_k &= \xi_{2k-2}+\xi_{2k-1}    \quad &\beta_{k+1} &= \xi_{2k-1}\xi_{2k}
	\end{aligned}

	\begin{aligned}
			\xi_1 &= c_1             \quad &\gamma_1 &= 1-c_1 \\
			\xi_k &= (1-c_{k-1})c_k    \quad &\gamma_k &= c_{k-1}(1-c_k)
	\end{aligned}


with,

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
			\right)


To recover the full matrix model for :ref:`laguerre_ensemble_full`, recall that :math:`\Gamma(\frac{k}{2}, 2)\equiv \chi_k^2` and take 

.. math::

	a = \frac{\beta}{2}(M_1-N+1)
	\quad \text{and} \quad
	b = \frac{\beta}{2}(M_2-N+1)

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
			\right)

.. plot:: plots/ex_plot_jacobi_banded.py
	:include-source:

.. seealso::

	- :cite:`KiNe04` Theorem 2
	- :ref:`Full matrix model <jacobi_ensemble_full>` for Jacobi ensemble

.. _circular_ensemble_banded:

Circular Ensemble
^^^^^^^^^^^^^^^^^

.. math::

	\left|\Delta(e^{i \theta_1},\dots, e^{i \theta_N})\right|^{\beta}
		\prod_{j = 1}^N \frac{1}{2\pi} \mathbf{1}_{[0,2\pi]} (\theta_j) d\theta_j

.. note::

	Recall that from the definition in :eq:`abs_vandermonde_det`
	
	.. math::

		\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|


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
	M = \operatorname{diag}[\Xi_{-1},\Xi_1,\Xi_3\dots]

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
	\Xi_{N-1} = [\overline{\alpha}_{N-1}]

.. hint::

	The effect of increasing the :math:`\beta` parameter can be nicely vizualized on this :ref:`circular_ensemble_banded`
	Viewing :math:`\beta` as the inverse temperature, the configuration of the eigenvalues cristallizes with :math:`\beta`.
	The following pictures display realizations for :math:`\beta=1, 10, 20` respectively.

.. plot:: plots/ex_plot_circular_banded.py
	:include-source:

.. seealso::

	- :cite:`KiNe04` Theorem 1
	- :ref:`Full matrix model <circular_ensemble_full>` for Circular ensemble
