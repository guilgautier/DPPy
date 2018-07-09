.. _continuous_dpps_sampling:

Sampling
********

.. note::



Exact sampling
==============

Chain rule
----------

.. important::

	Alike the discrete case, the spectral decomposition of the kernel is required for generic continuous DPPs.

This similar to the discrete :ref:`discrete_dpps_exact_sampling` i.e. the chain rule still applies except that now conditionals have a density and it becomes tricky to sample from them.

One could use rejection sampling but a good proposal is difficult to tailor, the acceptance rate that is difficult to control.

.. seealso::

	- Algorithm 18 :cite:`HKPV06`

Perfect sampling
----------------

	- :cite:`DFL13`


Approximate sampling
====================

Fourier approximation of the kernel

.. seealso::

	:cite:`LaMoRu15`
