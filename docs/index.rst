.. DPPy documentation master file, created by
   sphinx-quickstart on Tue Jun  5 07:45:55 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DPPy's documentation!
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Exact Sampling

.. automodule:: exact_sampling

.. autofunction:: dpp_sampler_exact

Projection :math:`\operatorname{DPP}`
-------------------------------------

    .. math::

        \mathbb{P}[i~|~Y+j] 
            &= \frac{\det K_{Y+j+i}}{\det K_{Y+j}} \\
            &= \|\Pi_{\operatorname{Span}K_{:Y+j}^{\perp}} K_{:i}\|^2 \\
            &= \|\Pi_{\operatorname{Span}K_{:Y}^{\perp}} K_{:i}\|^2 - \|\Pi_{\operatorname{Span}K_{:j}} K_{:i}\|^2


Generic :math:`\operatorname{DPP}`
-------------------------------------



:math:`\operatorname{k-DPP}`
----------------------------

Approximate Sampling

References
==========

.. bibliography:: biblio.bib
    :encoding: latex+latin
    :style: alpha
    :cited:

.. :style: alpha, plain , unsrt, and unsrtalpha

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
