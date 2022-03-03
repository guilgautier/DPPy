.. _beta_ensembles_ginibre:

Ginibre Ensemble
^^^^^^^^^^^^^^^^

.. math::

	\left|\Delta(z_1,\dots,z_N)\right|^{2}
	\prod_{i = 1}^N e^{ - \frac{1}{2}|z_i|^2 }
	d z_i,

where from the definition in :eq:`eq:abs_vandermonde_det` we have :math:`\left|\Delta(x_1,\dots,x_N)\right| = \prod_{i<j} |x_i - x_j|`.

.. math::

	A \sim
	\frac{1}{\sqrt{2}}
	\left( \mathcal{N}_{N,N}(0,1) + i~ \mathcal{N}_{N, N}(0,1) \right).

Nomalization :math:`\sqrt{N}` to concentrate in the unit circle.

.. :ref:`Fig. <ginibre_full_matrix_model_plot>`

.. _ginibre_full_matrix_model_plot:

.. plot:: plots/ex_plot_ginibre_full_matrix_model.py

	Full matrix model for the Ginibre ensemble

.. seealso::

	- :py:mod:`~dppy.beta_ensembles.ginibre` in API
