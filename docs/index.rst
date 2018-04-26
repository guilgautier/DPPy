.. automodule:: exact_sampling

Exact Sampling
==============

    .. autofunction:: dpp_sampler
    
    .. autofunction:: dpp_sampler

Projection :math:`\operatorname{DPP}`
-------------------------------------

    .. autofunction:: projection_dpp_sampler_GS

    .. math::

        \mathbb{P}[i~|~Y+j] 
            &= \frac{\det K_{Y+j+i}}{\det K_{Y+j}} \\
            &= \|\Pi_{\operatorname{Span}K_{:Y+j}^{\perp}} K_{:i}\|^2 \\
            &= \|\Pi_{\operatorname{Span}K_{:Y}^{\perp}} K_{:i}\|^2 - \|\Pi_{\operatorname{Span}K_{:j}} K_{:i}\|^2

    .. autofunction:: projection_dpp_sampler_Schur

Generic :math:`\operatorname{DPP}`
-------------------------------------

    .. autofunction:: dpp_sampler_eig_GS
    .. autofunction:: dpp_sampler_eig_Cholesky
    .. autofunction:: dpp_sampler_KuTa12

:math:`\operatorname{k-DPP}`
----------------------------

    .. autofunction:: k_dpp_sampler

    .. autofunction:: elem_symm_poly

    .. autofunction:: select_eig_vec


References
==========

.. bibliography:: biblio.bib
    :encoding: latex+latin
    :style: alpha
    :cited:

.. :style: alpha, plain , unsrt, and unsrtalpha