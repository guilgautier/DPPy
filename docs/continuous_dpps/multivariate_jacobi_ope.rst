.. _multivariate_jacobi_ope:

Multivariate Jacobi ensemble
----------------------------

.. important::

    For the details please refer to:

    a) the extensive documentation of :py:class:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE` below
    b) the associated `Jupyter notebook <https://github.com/guilgautier/DPPy/blob/master/notebooks/>`_ which showcases :py:class:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE`
    c) our NeurIPS'19 paper :cite:`GaBaVa19` *On two ways to use determinantal point processes for Monte Carlo integration*
    d) our `ICML'19 workshop paper <https://guilgautier.github.io/publications/>`_

The figures below display a sample of a :math:`d=2` dimensional Jacobi ensemble :py:class:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE` with :math:`N=200` points.
The red and green dashed curves correspond to the normalized base densities proportional to :math:`(1-x)^{a^1} (1+x)^{b^1}` and :math:`(1-y)^{a^1} (1+y)^{b^1}`, respectively.

.. plot:: plots/ex_plot_multivariate_jacobi_ensemble.py
    :width: 50%

- In the first plot, observe that the empirical marginal density converges to the arcsine density :math:`\frac{1}{\pi\sqrt{1-x^2}}`, displayed in orange.
- In the second plot, we take the same sample and attach a weight :math:`\frac{1}{K(x,x)}` to each of the points. This illustrates the choice of the weights defining the estimator of :cite:`BaHa16` as a proxy for the reference measure.

----

.. automodule:: dppy.multivariate_jacobi_ope
    :members:
    :inherited-members:
    :show-inheritance:
