.. _beta_ensemble_circular:

Circular Ensemble
^^^^^^^^^^^^^^^^^

.. math::

	\left|\Delta(e^{i \theta_1 },\dots, e^{i \theta_N})\right|^{\beta}
    \prod_{j = 1}^N \frac{1}{2\pi} \mathbf{1}_{[0,2\pi]} (\theta_j) d\theta_j,

   where from the definition in :eq:`eq:abs_vandermonde_det` we have :math:`\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|`.


.. _circular_full_matrix_model:

Full model
^^^^^^^^^^

.. hint::

	Eigenvalues of orthogonal (resp. unitary and  self-dual unitary) matrices drawn uniformly i.e. Haar measure on the respective groups.
	The eigenvalues lie on the unit circle i.e. :math:`\lambda_n = e^{i \theta_n}`.
	The distribution of the angles :math:`\theta_n` converges to the uniform measure on :math:`[0, 2\pi[` as :math:`N` grows.

- :math:`\beta=1`

	Uniform measure i.e. Haar measure on orthogonal matrices :math:`\mathbb{O}_N`:  :math:`U^{\top}U = I_N`

	1. Via QR algorithm, see :cite:`Mez06` Section 5

	.. code-block:: python

		import numpy as np
		from numpy.random import randn
		import scipy.linalg as la

		A = randn(N, N)
		Q, R = la.qr(A)
		d = np.diagonal(R)
		U = np.multiply(Q, d/np.abs(d), Q)
		la.eigvals(U)


	2. The Hermite way

	.. math::

		X \sim \mathcal{N}_{N, N}(0,1)\\
		A = X+X^{\top}
	      = U^{\top}\Lambda U\\
	    eigvals(U).

- :math:`\beta=2`

	Uniform measure i.e. Haar measure on unitary matrices :math:`\mathbb{U}_N`: :math:`U^{\dagger}U = I_N`

	1. Via QR algorithm, see :cite:`Mez06` Section 5

	.. code-block:: python

		import numpy as np
		from numpy.random import randn
		import scipy.linalg as la

		A = randn(N, N) + 1j*randn(N, N)
		Q, R = la.qr(A)
		d = np.diagonal(R)
		U = np.multiply(Q, d / np.abs(d), Q)
		la.eigvals(U)

	.. :ref:`Fig. <circular_full_matrix_model_qr_plot>`

	.. _circular_full_matrix_model_qr_plot:

	.. plot:: plots/ex_plot_circular_full_matrix_model_qr.py

		Full matrix model for the Circular ensemble from QR on random Gaussian matrix

	2. The Hermite way

	.. math::

		X \sim \mathcal{N}_{N, N}(0,1) + i~ \mathcal{N}_{N, N}(0,1)\\
	    A = X+X^{\dagger}
	      = U^{\dagger}\Lambda U\\
	    eigvals(U).

	.. :ref:`Fig. <circular_full_matrix_model_hermite_plot>`

	.. _circular_full_matrix_model_hermite_plot:

	.. plot:: plots/ex_plot_circular_full_matrix_model_hermite.py

		Full matrix model for the Circular ensemble from Hermite matrix

- :math:`\beta=4`

  Uniform measure i.e. Haar measure on self-dual unitary matrices :math:`\mathbb{U}\operatorname{Sp}_{2N}`:   :math:`U^{\dagger}U = I_{2N}`

	.. math::

		\begin{cases}
	        X \sim \mathcal{N}_{N, M}(0,1) + i~ \mathcal{N}_{N, M}(0,1)\\
	        Y \sim \mathcal{N}_{N, M}(0,1) + i~ \mathcal{N}_{N, M}(0,1)
	    \end{cases}\\
		A = \begin{bmatrix}
	            X    & Y   \\
	            -Y^* & X^*
	        \end{bmatrix}
	    \quad A = X+X^{\dagger}
	            = U^{\dagger} \Lambda U\\
	    eigvals(U).

.. seealso::

	- :ref:`Banded matrix model <circular_banded_matrix_model>` for Circular ensemble
	- :py:mod:`~dppy.beta_ensembles.circular` in API


.. _circular_banded_matrix_model:

Quindiagonal model
^^^^^^^^^^^^^^^^^^

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
	- :py:mod:`~dppy.beta_ensembles.circular` in API
