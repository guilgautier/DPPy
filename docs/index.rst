.. DPPy documentation master file, created by
	 sphinx-quickstart on Tue Jun  5 07:45:55 2018.
	 You can adapt this file completely to your liking, but it should at least
	 contain the root `toctree` directive.

Welcome to DPPy's documentation!
================================

Discrete DPPs
=============

Definition
----------

A discrete point process :math:`\mathcal{X}` on :math:`[N] \triangleq \{1,\dots,N\}` can be defined via its:

- inclusion probabilities (also called correlation functions) 

	.. math::

		\mathbb{P}[S\in \mathcal{X}], \text{ for } S\subset [N]

- marginal probabilities

	.. math::

		\mathbb{P}[\mathcal{X}=S], \text{ for } S\subset [N]

The *determinantal* feature of DPPs stems from the fact that such inclusion, resp. marginal probabilities are given by the principal minors of the corresponding inclusion, resp. marginal kernels :math:`\mathbf{K}`, resp. :math:`\mathbf{L}`.

Inclusion probabilities
~~~~~~~~~~~~~~~~~~~~~~~
:math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{K})` with inclusion kernel :math:`\mathbf{K}` if it satisfies

	.. math::
		:label: eq:inclusion_proba

		\mathbb{P}[S\in \mathcal{X}] = \det \mathbf{K}_S, 
		\quad \forall S\subset [N]

Marginal probabilities
~~~~~~~~~~~~~~~~~~~~~~
:math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{L})` with marginal! kernel :math:`\mathbf{L}` if it satisfies

	.. math::
		:label: eq:marginal_proba

		\mathbb{P}[\mathcal{X}=S] = \frac{\det \mathbf{L}_S}{\det [I+\mathbf{L}]}, 
		\quad \forall S\subset [N]

Existence
~~~~~~~~~

Sufficient condition:

	-  :math:`\mathbf{K} = \mathbf{K}^{\dagger}` with :math:`0_N \preceq \mathbf{K} \preceq I_N`

	- :math:`\mathbf{L} = \mathbf{L}^{\dagger}` with :math:`\mathbf{L} \succeq 0_N`

.. note::

	This is only a sufficient condition, there indeed exist DPPs with non symmetric kernels such as the carries process.
	For now, DPPy only treats real symmetric case

.. todo::
	
	Put reference to carries process

Properties
~~~~~~~~~~

- Number of points :math:`|\mathcal{X}| = \sum_{n=1}^N 1_{n \in \mathcal{X}}`
	
	1. Expectation

	.. math::
		:label: eq:expect_number_points_projection_K

		\mathbb{E}[|\mathcal{X}|] 
		= \operatorname{Tr} \mathbf{K}

	2. Variance

	.. math::
		:label: eq:var_number_points_projection_K

		\mathbb{V}ar[|\mathcal{X}|] 
		= \operatorname{Tr} \mathbf{K} - \operatorname{Tr} \mathbf{K}^2

.. note::
	For projection :math:`\mathbf{K}` i.e. :math:`\mathbf{K}^2=\mathbf{K}`

	.. math::

		\mathbb{E}[|\mathcal{X}|] 
		& = \operatorname{Tr} \mathbf{K} 
			= \operatorname{rank} \mathbf{K}\\
		\mathbb{V}ar[|\mathcal{X}|] 
		& = \operatorname{Tr} \mathbf{K} - \operatorname{Tr} \mathbf{K}^2
			= 0

	That is to say DPPs with projection :math:`\mathbf{K}` yield fixed size samples. Indeed, :eq:`eq:expect_number_points_projection_K` and :eq:`eq:var_number_points_projection_K` yield

	.. math::
		:label: eq:number_points_projection_K

		|\mathcal{X}| 
			\overset{a.s.}{=} 
				\operatorname{Tr} \mathbf{K} 
			= \operatorname{rank} \mathbf{K}

- Relation between :math:`\mathbf{K}` and :math:`\mathbf{L}`

	.. math::
		:label: eq:relation_K_L_kernels

		\mathbf{K} = \mathbf{L}(I+\mathbf{L})^{—1} 
			\qquad \text{and} \qquad 
		\mathbf{L} = \mathbf{K}(I-\mathbf{K})^{—1}

	.. warning::
		
		Recall that :math:`\operatorname{DPP}(\mathbf{K})` with *projection* inclusion kernel :math:`\mathbf{K}` yields fixed size samples (see :eq:`eq:number_points_projection_K`).
		Thus, the marginal kernel :math:`\mathbf{L}` cannot be computed via :eq:`eq:relation_K_L_kernels` with :math:`\mathbf{L} = \mathbf{K}(I-\mathbf{K})^{—1}`, since :math:`\mathbf{K}` has at least one eigenvalue equal to :math:`1`.
		However, the marginal kernel :math:`\mathbf{L}` coincides with :math:`\mathbf{K}`

		.. math::

			\mathbb{P}[\mathcal{X}=S] = 
				\det \mathbf{K}_S 1_{|S|=\operatorname{rank}\mathbf{K}}
				\quad \forall S\subset [N]

- Negative association
	
	.. math::
		:label: eq:negative_association
		
		\mathbb{P}[\{i, j\} \subset \mathcal{X}]
	  &= \begin{vmatrix}
	    \mathbb{P}[i \in \mathcal{X}]	& \mathbf{K}_{i j}\\
	    \overline{\mathbf{K}_{i j}}		& \mathbb{P}[j \in \mathcal{X}]
	  \end{vmatrix}\\
	  &= \mathbb{P}[i \in \mathcal{X}] \mathbb{P}[j \in \mathcal{X}] 
	  	- |\mathbf{K}_{i j}|^2


Exact sampling
--------------

Projection kernels
~~~~~~~~~~~~~~~~~~

Generic kernels
~~~~~~~~~~~~~~~~~~



MCMC sampling
-------------

Basis exchange
~~~~~~~~~~~~~~

Add-Delete
~~~~~~~~~~

Add-Exchange-Delete
~~~~~~~~~~~~~~~~~~~

Zonotope
~~~~~~~~



Continuous DPPs
===============

Definition
----------


Exotic DPPs
===============








Old shit
========

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
