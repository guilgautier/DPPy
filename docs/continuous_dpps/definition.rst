.. _continuous_dpps_definition:

Definition
----------

Denote :math:`\mathbb{X}` the ambient space. 
It is assumed metric, `locally compact <https://en.wikipedia.org/wiki/Locally_compact_space>`_ and we equip it with its Borel :math:`\sigma`-algebra e.g. :math:`\mathbb{R}^d, \mathbb{C}^d, \mathbb{S}^{d-1}`.

A configuration of points :math:`\gamma` is a discrete locally finite subset of :math:`\mathbb{X}`

- Discrete:

  .. math::

    \forall x \in \gamma, \exists r >0, \quad B(x,r) \cap \gamma = \{x\}

- Locally finite: 

	.. math::
		
		\forall K \subset \mathbb{X} \text{ compact}, 
			\quad \#(\gamma \cap K) < \infty

More specifically, a probability measure on the space of configurations of points endowed with the *canonical* :math:`\sigma`-algebra making the application :math:`\gamma \mapsto \# (\gamma \cap K)` measurable for all compact :math:`K`.

From now on, we denote :math:`\mu` the reference measure set on :math:`\mathbb{X}`.

.. hint::

	A point process is a random configuration of points

	.. math::

		\{X_1, \dots, X_N\} \subset \mathbb{X}

	with N being random.

	To understand the interaction between the points of a point process, one focuses on the interaction of each cloud of :math:`k` points (for all :math:`k`). 
	The corresponding :math:`k`-correlation functions characterize the underlying point process.
 
Correlation functions
~~~~~~~~~~~~~~~~~~~~~

For :math:`k\geq 0`, the :math:`k`-correlation function :math:`\rho_k` is defined by:

:math:`\forall f : \mathbb{X}^k \to \mathbb{C}` measurable with compact support

.. math::

  \mathbb{E}
  \left[ \sum_{  
    \substack{
    	(X_1,\dots,X_k) \\ 
    	X_1 \neq \dots \neq X_k \in \gamma} } 
    f(X_1,\dots,X_k) 
  \right]
	  = \int_{\mathbb{X}^k} 
	  	f(x_1,\dots,x_k) \rho_k(x_1,\dots,x_k) 
	  	\prod_{i=1}^k \mu(dx_i)