.. _full_matrix_models:

Full matrix models
------------------

For specific reference measures the :math:`\beta = 1, 2, 4` cases are very singular in the sense that the corresponding ensembles coincide with the eigenvalues of random matrices.

This is a highway for sampling exactly such ensembles in :math:`\mathcal{O}(N^3)`!

.. _hermite_ensemble_full:

Hermite Ensemble
^^^^^^^^^^^^^^^^

Take for reference measure :math:`\mu=\mathcal{N}(0, 2)`, the pdf of the corresponding :math:`\beta`-Ensemble reads

.. math::

	(x_1,\dots,x_N) 
	\sim 
		\left|\Delta(x_1,\dots,x_N)\right|^{\beta}
		\prod_{i= 1}^N 
			e^{- \frac{1}{2}\frac{x_i^2}{2}} 
			% \indic_{\bbR}(x_i)
		\ d x_i

.. note::

	Recall that from the definition in :eq:`abs_vandermonde_det`

	.. math::

		\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|

.. hint::

	The Hermite ensemble (whose name comes from the fact that Hermite polynomials are orthogonal w.r.t the Gaussian distribution) refers to the eigenvalue distribution of random matrices formed by i.i.d. Gaussian vectors.


- :math:`\beta=1`

.. math::

	X \sim \mathcal{N}_{N, N}(0,1)
	\qquad
	A = \frac{X+X^{\top}}{\sqrt{2}}

- :math:`\beta=2`

.. math::

	X \sim \mathcal{N}_{N, N}(0,1) + i~ \mathcal{N}_{N, N}(0,1)
	\qquad
	A = \frac{X+X^{\dagger}}{\sqrt{2}}

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
    \quad A = \frac{X+X^{\dagger}}{\sqrt{2}}


Normalization :math:`\sqrt{\beta N}` to concentrate as the semi-circle law.

.. math::
	
	\frac{\sqrt{4-x^2}}{2\pi} 1_{[-2,2]} dx

.. plot:: plots/ex_plot_hermite_full_mat.py
  :include-source:


.. seealso::

	- :ref:`Banded matrix model <hermite_ensemble_banded>` for Hermite ensemble

.. _laguerre_ensemble_full:

Laguerre Ensemble
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
		\ d x_i

.. note::

	Recall that from the definition in :eq:`abs_vandermonde_det`
	
	.. math::

		\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|


.. hint::

	The Laguerre ensemble (whose name comes from the fact that Laguerre polynomials are orthogonal w.r.t the Gamma distribution) refers to the eigenvalue distribution of empirical covariance matrices of i.i.d. Gaussian vectors.

- :math:`\beta=1`

.. math::

	X \sim \mathcal{N}_{N, M}(0,1)
	\qquad
	A = XX^{\top}

- :math:`\beta=2`

.. math::

	X \sim \mathcal{N}_{N, M}(0,1) + i~ \mathcal{N}_{N, M}(0,1)
	\qquad
	A = XX^{\dagger}

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
    \quad A = A A^{\dagger}


Normalization :math:`\beta M` to concentrate as Marcenko-Pastur law

.. math::

	\frac{1}{2\pi}
	\frac{\sqrt{(\lambda_+-x)(x-\lambda_-)}}{cx} 
	1_{[\lambda_-,\lambda_+]}
	dx

where 

.. math::

	c = \frac{M}{N}
	\quad \text{and} \quad 
	\lambda_\pm = (1\pm\sqrt{c})^2

.. plot:: plots/ex_plot_laguerre_full_mat.py
  :include-source:

.. seealso::

	- :ref:`Banded matrix model <laguerre_ensemble_banded>` for Laguerre ensemble

.. _jacobi_ensemble_full:

Jacobi Ensemble
^^^^^^^^^^^^^^^

Take for reference measure 
:math:`\mu=\operatorname{\mathcal{B}eta}\left(\frac{\beta}{2}(M_1-N+1), \frac{\beta}{2}(M_2-N+1)\right)`, 
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
		\ d x_i

.. note::

	Recall that from the definition in :eq:`abs_vandermonde_det`
	
	.. math::

		\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|


.. hint::

	The Jacobi ensemble (whose name comes from the fact that Jacobi polynomials are orthogonal w.r.t the Beta distribution) is associated with the multivariate analysis of variance (MANOVA) model.

- :math:`\beta=1`

.. math::

	\begin{cases}
		X \sim \mathcal{N}_{N, M_1}(0,1)\\
		Y \sim \mathcal{N}_{N, M_2}(0,1)\\
	\end{cases}
	\qquad
	A = XX^{\top}\left(XX^{\top} + YY^{\top}\right)^{-1}

- :math:`\beta=2`

.. math::

	\begin{cases}
		X \sim \mathcal{N}_{N, M_1}(0,1) + i~ \mathcal{N}_{N, M_1}(0,1)\\
		Y \sim \mathcal{N}_{N, M_2}(0,1) + i~ \mathcal{N}_{N, M_2}(0,1)\\
	\end{cases}
	\qquad
	A = XX^{\dagger}\left(XX^{\dagger} + YY^{\dagger}\right)^{-1}

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
	A = XX^{\dagger}\left(XX^{\dagger} + YY^{\dagger}\right)^{-1}

Concentrates as Wachter law

.. math::

	\frac{(a+b)\sqrt{(\sigma_+-x)(x-\sigma_-)}}{2\pi x(1-x)}dx

where 

.. math::

	a = \frac{M_1}{N}, 
	b = \frac{M_2}{N}
	\quad\text{and}\quad
	\sigma_{\pm} = \left(\frac{\sqrt{a(a+b-1)} \pm \sqrt{b}}{a+b}\right)^2

itself tending to the arcsine law in the limit

.. plot:: plots/ex_plot_jacobi_full_mat.py
  :include-source:

.. seealso::

	- :ref:`Banded matrix model <jacobi_ensemble_banded>` for Jacobi ensemble

.. _circular_ensemble_full:

Circular Ensemble
^^^^^^^^^^^^^^^^^

.. math::

	\left|\Delta(e^{i \theta_1 },\dots, e^{i \theta_N})\right|^{\beta}
    \prod_{j = 1}^N \frac{1}{2\pi} \mathbf{1}_{[0,2\pi]} (\theta_j) d\theta_j

.. note::

	Recall that from the definition in :eq:`abs_vandermonde_det`
	
	.. math::

		\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|


.. hint::

	Eigenvalues of orthogonal (resp. unitary and  self-dual unitary) matrices drawn uniformly i.e. Haar measure on the respective groups.
	The eigenvalues lie on the unit circle i.e. :math:`\lambda_n = e^{i \theta_n}`.
	The distribution of the angles :math:`\theta_n` converges to the uniform measure on :math:`[0, 2\pi[` as :math:`N` grows.

- :math:`\beta=1`

	Uniform measure i.e. Haar measure on orthogonal matrices :math:`\mathbb{O}_N`:  :math:`U^{\top}U = I_N`

	1. Via QR algorithm, see :cite:`Mez06` Section 5

	.. code-block:: python

		A = np.random.randn(N, N)
		Q, R = la.qr(A)
		d = np.diagonal(R)
		U = np.multiply(Q, d/np.abs(d), Q)
		return eigvals(U)


	2. The Hermite way
	
	.. math::

		X \sim \mathcal{N}_{N, N}(0,1)\\
		A = X+X^{\top} 
	      = U^{\top}\Lambda U\\
	    eigvals(U)

- :math:`\beta=2`

	Uniform measure i.e. Haar measure on unitary matrices :math:`\mathbb{U}_N`: :math:`U^{\dagger}U = I_N`

	1. Via QR algorithm, see :cite:`Mez06` Section 5

	.. code-block:: python

		A = np.random.randn(N, N) + 1j*np.random.randn(N, N)
		A /= np.sqrt(2.0)
		Q, R = la.qr(A)
		d = np.diagonal(R)
		U = np.multiply(Q, d/np.abs(d), Q)
		return eigvals(U)

	.. plot:: plots/ex_plot_circular_full_mat_QR.py
		:include-source:

	2. The Hermite way

	.. math::

		X \sim \mathcal{N}_{N, N}(0,1) + i~ \mathcal{N}_{N, N}(0,1)\\
	    A = X+X^{\dagger}
	      = U^{\dagger}\Lambda U\\
	    eigvals(U)

	.. plot:: plots/ex_plot_circular_full_mat_hermite.py
		:include-source:


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
	    eigvals(U)

.. seealso::

	- :ref:`Banded matrix model <circular_ensemble_banded>` for Circular ensemble

.. _ginibre_ensemble_full:

Ginibre Ensemble
^^^^^^^^^^^^^^^^

.. math::

	\left|\Delta(z_1,\dots,z_N)\right|^{2}
	\prod_{i = 1}^N e^{ - \frac{1}{2}|z_i|^2 }
	d z_i

.. note::

	Recall that from the definition in :eq:`abs_vandermonde_det`
	
	.. math::

		\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|


.. math::
	
	A \sim 
	\frac{1}{\sqrt{2}} 
	\left( \mathcal{N}_{N,N}(0,1) + i~ \mathcal{N}_{N, N}(0,1) \right)

Nomalization :math:`\sqrt{N}` to concentrate in the unit circle

.. plot:: plots/ex_plot_ginibre_full_mat.py
  :include-source: