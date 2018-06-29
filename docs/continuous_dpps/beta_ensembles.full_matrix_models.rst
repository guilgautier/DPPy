.. _full_matrix_models:

Full matrix models
~~~~~~~~~~~~~~~~~~

As mentioned earlier, for specific reference measures the :math:`\beta = 1, 2, 4` cases are very singular in the sense that the corresponding ensembles coincide with the eigenvalues of random matrices.

This is a highway for sampling exactly such ensembles in :math:`\mathcal{O}(N^3)`!

.. _hermite_ensemble:

Hermite Ensemble
++++++++++++++++

.. hint::

	Random symmetric matrices

.. math::

	(x_1,\dots,x_N) 
	\sim 
		\left|\Delta(x_1,\dots,x_N)\right|^{\beta}
		\prod_{i= 1}^N 
			e^{- \frac{1}{2}\frac{x_i^2}{2}} 
			% \indic_{\bbR}(x_i)
		\ d x_i

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

.. image:: ../images/Hermite.pdf
	:width: 40pt

.. _laguerre_ensemble:

Laguerre Ensemble
+++++++++++++++++

.. hint::

	Random covariance matrices

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
	\lambda_\pm = (1\pm\sqrt{c})^2`

.. image:: ../images/Laguerre.pdf
	:width: 40pt


.. _jacobi_ensemble:

Jacobi Ensemble
+++++++++++++++

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

.. image:: ../images/Jacobi.pdf
	:width: 40pt

.. _circular_ensemble:

Circular Ensemble
+++++++++++++++++

.. math::

	\left|\Delta(e^{i \theta_1 },\dots, e^{i \theta_N})\right|^{\beta}
    \prod_{j = 1}^N \frac{1}{2\pi} \mathbf{1}_{[0,2\pi]} (\theta_j) d\theta_j

- :math:`\beta=1`

	.. hint::

		Uniform measure i.e. Haar measure on orthogonal matrices $\mathbb{O}_N$:  $U^{\top}U = I_N$

	1. Via QR algorithm, see :cite:`Mez06` Section 5

	.. code-block:: python

		A = np.random.randn(N, N)
		Q, R = np.linalg.qr(A)
		d = np.diagonal(R)
		U = np.multiply(Q, d/np.abs(d), Q)
		return eigvals(U)


	2. The Hermite way
	
	.. math::

		X \sim \mathcal{N}_{N, N}(0,1)\\
		A = X+X^{\top} 
	      = U^{\top}\Lambda U\\
	    eigvals(U)

- :math:\beta=2`

	.. hint::

		Uniform measure i.e. Haar measure on unitary matrices $\mathbb{U}_N$:   $U^{\dagger}U = I_N$

	1. Via QR algorithm, see :cite:`Mez06` Section 5

	.. code-block:: python

		A = np.random.randn(N, N) + 1j*np.random.randn(N, N)
		A /= np.sqrt(2.0)
		Q, R = np.linalg.qr(A)
		d = np.diagonal(R)
		U = np.multiply(Q, d/np.abs(d), Q)
		return eigvals(U)


	2. The Hermite way

	.. math::

		X \sim \mathcal{N}_{N, N}(0,1) + i~ \mathcal{N}_{N, N}(0,1)\\
	    A = X+X^{\dagger}
	      = U^{\dagger}\Lambda U\\
	    eigvals(U)


- :math:`\beta=4`
  
  .. hint::

  	Uniform measure i.e. Haar measure  onsymplectic matrices $\mathbb{U}\operatorname{Sp}_{2N}$:   $U^{\dagger}U = I_N$

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

.. todo::

	add figure

.. _ginibre_ensemble:

Ginibre Ensemble
++++++++++++++++

.. math::

	\left|\Delta(z_1,\dots,z_N)\right|^{2}
	\prod_{i = 1}^N e^{ - \frac{1}{2}|z_i|^2 }
	d z_i

.. math::
	
	A \sim 
	\frac{1}{\sqrt{2}} 
	\left( \mathcal{N}_{N,N}(0,1) + i~ \mathcal{N}_{N, N}(0,1) \right)

Nomalization $\sqrt{N}$ to concentrate in the unit circle

.. todo::

	add figure