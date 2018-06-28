.. _discrete_dpps_definition:

Definition
----------

A discrete point process :math:`\mathcal{X}` on :math:`[N] \triangleq \{1,\dots,N\}` can be understood as a random subset.
It is defined via its:

- inclusion probabilities (also called correlation functions) 

	.. math::

		\mathbb{P}[S\in \mathcal{X}], \text{ for } S\subset [N]

- marginal probabilities

	.. math::

		\mathbb{P}[\mathcal{X}=S], \text{ for } S\subset [N]

The *determinantal* feature of DPPs stems from the fact that such inclusion, resp. marginal probabilities are given by the principal minors of the corresponding inclusion kernel :math:`\mathbf{K}` (resp. marginal kernel :math:`\mathbf{L}`).

Inclusion probabilities
~~~~~~~~~~~~~~~~~~~~~~~
:math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{K})` with inclusion kernel :math:`\mathbf{K}` if it satisfies

	.. math::
		:label: inclusion_proba

		\mathbb{P}[S\in \mathcal{X}] = \det \mathbf{K}_S, 
		\quad \forall S\subset [N]

Marginal probabilities
~~~~~~~~~~~~~~~~~~~~~~
:math:`\mathcal{X} \sim \operatorname{DPP}(\mathbf{L})` with marginal kernel :math:`\mathbf{L}` if it satisfies

	.. math::
		:label: marginal_proba

		\mathbb{P}[\mathcal{X}=S] = \frac{\det \mathbf{L}_S}{\det [I+\mathbf{L}]}, 
		\quad \forall S\subset [N]

Existence
~~~~~~~~~

Necessary condition:

	.. math::

		\mathbf{K} \succeq 0_N
			\quad \text{i.e.} \quad
			\forall x\in \mathbb{C}^N, 
				x^{\dagger} \mathbf{K} x \geq 0

Sufficient conditions:

	.. math::
		:label: suff_cond_K

		\mathbf{K} = \mathbf{K}^{\dagger}
		\quad \text{and} \quad 
		0_N \preceq \mathbf{K} \preceq I_N

	.. math:: 
		:label: suff_cond_L

		\mathbf{L} = \mathbf{L}^{\dagger}
		\quad \text{and} \quad
		\mathbf{L} \succeq 0_N

.. note::

	This is only a sufficient condition, there indeed exist DPPs with non symmetric kernels such as the carries process.

	In the following, DPPs defined by an *orthogonal projection* inclusion kernel :math:`\mathbf{K}` are called *projection* DPPs.
	They are indeed valid kernels since they meet the above sufficient conditions: they are hermitian with eigenvalues :math:`0` or :math:`1`.

	.. todo::
		
		Put reference to carries process

.. seealso::

	:cite:`KuTa12`
