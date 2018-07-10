.. _continuous_dpps_sampling:

Sampling
********

Exact sampling
==============

Chain rule
----------

.. important::

	In the same vein as the discrete case, the spectral decomposition of the kernel is required for applying the chain rule to sample generic continuous DPPs.

The chain rule still applies except that now conditionals have a density and it becomes tricky to sample from them.
One could use rejection sampling but a good proposal is difficult to tailor, the acceptance rate that is difficult to control.

.. seealso::

	Algorithm 18 :cite:`HKPV06`

Perfect sampling
----------------

	:cite:`DFL13` uses Coupling From The Past 
	(`CFTP <https://pdfs.semanticscholar.org/622e/a9c9c665002670ff26119d1aad5c3c5e0be8.pdf_>`_).

Approximate sampling
====================

Fourier approximation of the kernel.

.. seealso::

	:cite:`LaMoRu15` Section 4