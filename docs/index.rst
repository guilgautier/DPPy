.. automodule:: exact_sampling
.. automodule:: approximate_sampling

.. automodule:: discrete_dpps
.. automodule:: discrete_k_dpps

Exact Sampling

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