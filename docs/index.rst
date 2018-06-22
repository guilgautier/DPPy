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
	They are indeed valid kernels since they meet the above sufficient conditions: their eigenvalues are either :math:`0` or :math:`1`.

	.. todo::
		
		Put reference to carries process

Properties
~~~~~~~~~~

1. Geometrical insights

	Kernels satisfying the sufficient conditions :eq:`suff_cond_K` and :eq:`suff_cond_L` can be expressed as

	.. math::

		K_{ij} = \langle \phi_i, \phi_j \rangle
		\quad \text{and} \quad
		L_{ij} = \langle \psi_i, \psi_j \rangle,

	where each item is represented by a feature vector :math:`\phi_i` (resp. :math:`\psi_i`).

	The geometrical view is then straightforward.

	a. The inclusion probabilities interpret as

		.. math::

			\mathbb{P}[S\subset \mathcal{X}] 
			= \det \mathbf{K}_S
			= \operatorname{Vol}^2 \{\phi_s\}_{s\in S}

	b. The inclusion probabilities interpret as

		.. math::

			\mathbb{P}[\mathcal{X} = S] 
			\propto \det \mathbf{L}_S
			= \operatorname{Vol}^2 \{\psi_s\}_{s\in S}
		
	That is to say, DPPs favor subsets :math:`S` whose corresponding feature vectors span a large volume i.e. *DPPs sample softened orthogonal bases*.

2. Diversity

	The *determinantal* structure of DPPs encodes the notion of diversity.
	Deriving the pair inclusion probability, also called the 2-point correlation function using :eq:`inclusion_proba`, we obtain
	
	.. math::
		
		\mathbb{P}[\{i, j\} \subset \mathcal{X}]
	  &= \begin{vmatrix}
	    \mathbb{P}[i \in \mathcal{X}]	& \mathbf{K}_{i j}\\
	    \overline{\mathbf{K}_{i j}}		& \mathbb{P}[j \in \mathcal{X}]
	  \end{vmatrix}\\
	  &= \mathbb{P}[i \in \mathcal{X}] \mathbb{P}[j \in \mathcal{X}] 
	  	- |\mathbf{K}_{i j}|^2

	That is, the greater the similarity :math:`|\mathbf{K}_{i j}|` between items :math:`i` and :math:`j`, the less likely they co-occur in the samples.

3. Relation between :math:`\mathbf{K}` (inclusion :eq:`inclusion_proba`) and :math:`\mathbf{L}` (marginal :eq:`inclusion_proba`) kernel

	.. math::
		:label: relation_K_L

		\mathbf{K} = \mathbf{L}(I+\mathbf{L})^{—1} 
			\qquad \text{and} \qquad 
		\mathbf{L} = \mathbf{K}(I-\mathbf{K})^{—1}

	.. warning::
		
		For DPPs with *projection* inclusion kernel :math:`K`, the marginal kernel :math:`\mathbf{L}` cannot be computed via  :eq:`relation_K_L` with :math:`\mathbf{L} = \mathbf{K}(I-\mathbf{K})^{—1}`, since :math:`\mathbf{K}` has at least one eigenvalue equal to :math:`1` (:math:`K^2=K`).
		However, the marginal kernel :math:`\mathbf{L}` coincides with :math:`\mathbf{K}`.

		.. math::

			\mathbb{P}[\mathcal{X}=S] = 
				\det \mathbf{K}_S 1_{|S|=\operatorname{rank}\mathbf{K}}
				\quad \forall S\subset [N]

	Thus, except for inclusion kernels :math:`\mathbf{K}` with some eigenvalues equal to :math:`1`, both :math:`\mathbf{K}` and :math:`\mathbf{L}` are diagonalizable in the same basis

	.. math::

		\mathbf{K} = U \Lambda^{\mathbf{K}} U^{\top}
			\qquad \text{and} \qquad
		\mathbf{L} = U \Lambda^{\mathbf{L}} U^{\top}

4. Number of points 
	.. math::
		:label: number_points

		|\mathcal{X}|
			= \sum_{n=1}^N 
				\operatorname{\mathcal{B}er}
				\left(
					\lambda_n^{\mathbf{K}}
				\right)
			= \sum_{n=1}^N 
				\operatorname{\mathcal{B}er}
				\left(
					\frac{\lambda_n^{\mathbf{L}}}{1+\lambda_n^{\mathbf{L}}}
				\right)

	.. hint::

		The first equality :math:`|\mathcal{X}| = \sum_{n=1}^N \operatorname{\mathcal{B}er} \left(\lambda_n^{\mathbf{K}}\right)` is key for the :ref:`sub_exact_sampling` scheme.
	
	a. Expectation

	.. math::
		:label: expect_number_points

		\mathbb{E}[|\mathcal{X}|] 
			= \operatorname{Tr} \mathbf{K}
			= \sum_{n=1}^N \lambda_n^{\mathbf{K}}
			= \sum_{n=1}^N \frac{\lambda_n^{\mathbf{L}}}{1+\lambda_n^{\mathbf{L}}}

	b. Variance

	.. math::
		:label: var_number_points

		\operatorname{\mathbb{V}ar}[|\mathcal{X}|] 
			= \operatorname{Tr} \mathbf{K} - \operatorname{Tr} \mathbf{K}^2
			= \sum_{n=1}^N \lambda_n^{\mathbf{K}}(1-\lambda_n^{\mathbf{K}})
			= \sum_{n=1}^N \frac{\lambda_n^{\mathbf{L}}}{(1+\lambda_n^{\mathbf{L}})^2}

	.. important::

		Realizations of projection DPPs have fixed cardinality.

		.. math::
			:label: number_points_projection_K

			|\mathcal{X}| 
				\overset{a.s.}{=} 
					\operatorname{Tr} \mathbf{K} 
				= \operatorname{rank} \mathbf{K}

		Indeed, since :math:`\mathbf{K}^2=\mathbf{K}`, :eq:`var_number_points` becomes

		.. math::

			\mathbb{V}ar[|\mathcal{X}|] 
			= \operatorname{Tr} \mathbf{K} - \operatorname{Tr} \mathbf{K}^2
			= 0

		and :eq:`expect_number_points` gives

		.. math::

			\mathbb{E}[|\mathcal{X}|] 
			= \operatorname{Tr} \mathbf{K} 
			= \operatorname{rank} \mathbf{K}

.. _sub_exact_sampling:

Exact sampling
--------------

The exact sampling scheme derived in Alg. 18 :cite:`HKPV06` is based on the chain rule and the geometrical interpretations are reflected through the conditionals.
The spectral decomposition of the inclusion kernel :math:`\mathbf{K}` (or equivalently the marginal kernel :math:`\mathbf{L}`) is required except for the case of a *projection* inclusion kernel.

.. important::

	Sampling *projection* DPPs is the building block of sampling generic DPPs
	since they are mixtures of *projection* DPPs, see Theorem 7 in :cite:`HKPV06`.

	More precisely, if the spectral decomposition writes :math:`\mathbf{K}
	= \sum_{n=1}^N \lambda_n^{\mathbf{K}} u_n u_n^{\top}` then we have

	.. math::

		\operatorname{DPP}(\mathbf{K})\sim\operatorname{DPP}(\mathbf{K}^B)
	
	where :math:`\mathbf{K}^B` is the **random** *projection* kernel defined by

	.. math::

		\mathbf{K}^B
		= \sum_{n=1}^N 
		\operatorname{\mathcal{B}er}(\lambda_n^{\mathbf{K}}) 
		u_i u_i^{\top}


Projection DPPs
~~~~~~~~~~~~~~~

	.. important::

		*Orthogonal projection* inclusion kernel :math:`\mathbf{K}` admit the following Gram matrix factorizations

		1. Using :math:`\mathbf{K} = \mathbf{K}^2` 
		and :math:`\mathbf{K}^{\dagger}=\mathbf{K}`

			.. math::
				:label: inclusion_kernel_factorization_K.TK

				\mathbf{K} 
				= \mathbf{K}^{\dagger} \mathbf{K} = \mathbf{K} \mathbf{K}^{\dagger},

		2. Using the spectral decomposition

			.. math::
				:label: inclusion_kernel_factorization_UU.T

				\mathbf{K} 
				= \mathbf{U} \mathbf{U}^{\dagger}, 
				\quad \text{where } \mathbf{U}^{\dagger} \mathbf{U} = I_r

		In this light, the rows (equiv. columns) of the inclusion kernel :math:`\mathbf{K}` or alternatively the rows of the eigenvectors :math:`\mathbf{U}` play the role of feature vectors.

		Finally, we will see that the chain rule is akin to Gram-Schmidt orthogonalization of these somewhat artificial feature vectors.

	Let :math:`S=\{s_1, \dots, s_r\}` with :math:`r=\operatorname{rank}(K)`, equation :eq:`number_points_projection_K` yields 

	.. math::

		\mathbb{P}[\mathcal{X}=S] 
		= \det \mathbf{K}_S
		
	As announced, the exact sampling scheme relies on the chain rule.

	.. math:: 
		:label: chain_rule
	
		\mathbb{P}[s_1, \dots, s_r] 
		= \mathbb{P}[s_1] \prod_{j=2}^r \mathbb{P}[s_j | s_{1:j-1}]

	The invariance by transposition and permutation of the determinant allows to express the joint probability distribution of :math:`(s_1, \dots, s_r)` as 

	.. math:: 

		\mathbb{P}[s_1, \dots, s_r] 
		= \frac{1}{r!} \mathbb{P}[\mathcal{X}=S] 
		= \frac{1}{r!} \det \mathbf{K}_S

	.. note::

		Once the chain rule performed, one can forget the order the points :math:`s_1,\dots,s_r` where drawn and take :math:`S=\{s_1, \dots, s_r\}` as a valid sample of :math:`\operatorname{DPP}(\mathbf{K})`.

	To proceed further, we need the conditionals involved in :eq:`chain_rule`

	**Chain rule**

	.. math::
		:label: chain_rule_K

		\mathbb{P}[s_1] 
		&= \dfrac{1}{r} \mathbf{K}_{s_1s_1}\\
		\mathbb{P}[s_j | s_{1:j-1}]
		&= \dfrac{1}{r-(j-1)} 
		\frac{\det \mathbf{K}_{\{s_{1:j}\}}}{\det \mathbf{K}_{\{s_{1:j-1}\}}}

	.. hint::

		The geometrical perspective may ease the interpretation as well as the computation of such conditionals.

		- Via :eq:`inclusion_kernel_factorization_K.TK` the sampling scheme takes the form

			.. math::

				\mathbb{P}[s_1] 
				&= \dfrac{1}{r} \mathbf{K}_{s_1s_1}\\
				\mathbb{P}[s_j | s_{1:j-1}]
				&= \dfrac{1}{r-(k-1)} 
				\operatorname{dist}^2 
				(\mathbf{K}_{s_j:} ~;~ \operatorname{Span} \mathbf{K}_{s_{1:j-1}:})

		- Via :eq:`inclusion_kernel_factorization_UU.T` it takes the form

			.. math::

				\mathbb{P}[s_1] 
				&= \dfrac{1}{r} \| \mathbf{U}_{s_1:} \|^2\\
				\mathbb{P}[s_j | s_{1:j-1}]
				&= \dfrac{1}{r-(k-1)} 
				\operatorname{dist}^2 
				(\mathbf{U}_{s_j:} ~;~ \operatorname{Span} \mathbf{U}_{s_{1:j-1}:})

	.. important::

		As mentioned earlier, the derivation of the chain rule boils down to applying Gram-Schmidt on the rows of either :math:`\mathbf{K}` or :math:`\mathbf{U}`.

		Finally, sampling from a projection :math:`\operatorname{DPP}(\mathbf{K})` can be performed in :math:`\mathcal{O}(N r^2)`.

	.. attention::

		The fact that :math:`\mathbf{K}` is a projection kernel is **crucial**.
		It is the very reason why the normalization constants of the conditionals in :eq:`chain_rule` are independent of the previous points and that :math:`S=\{s_1, \dots, s_r\}` is a sample of :math:`\operatorname{DPP}(\mathbf{K})`.

		Indeed, consider a kernel :math:`\mathbf{K} = V^{\dagger}V` satisfying :eq:`suff_cond_K` (with no *apriori* on :math:`V`).

		Setting :math:`Y=\{s_1, \dots, s_{j-1}\}`, the Schur complement formula provides

		.. math::

			\frac{\det \mathbf{K}_{Y+i}}{\det \mathbf{K}_{Y}}
			&= K_{ii} 
			- K_{Yi}^{\dagger} \left[\mathbf{K}_{Y}\right]^{-1} K_{Yi}\\
			&= K_{ii} 
			- V_{:i}^{\dagger} V_{:Y}
			\left[V_{:Y}^{\dagger} V_{:Y}\right]^{-1} 
			V_{Y:}^{\dagger} V_{:i}\\
			&= K_{ii} 
			- V_{:i}^{\dagger} \Pi_{V_{:Y}} V_{:i}

		where :math:`\Pi_{V_{:Y}}` corresponds to the orthogonal projection onto the span of the columns of :math:`V_{:Y}`.

		Thus,

		- at the initial step :math:`Y=\emptyset`

			.. math::
				
				\sum_{i=1}^N \mathbf{K}_{ii} = \operatorname{Tr}(K)
		
		- then

			.. math::		
				
				\sum_{i=1}^N
					\frac{\det \mathbf{K}_{Y+i}}{\det \mathbf{K}_{Y}}
				&= \sum_{i=1}^N K_{ii} - V_{:i}^{\dagger} \Pi_{V_{:Y}} V_{:i}\\
				&= \operatorname{Tr}(K) - \operatorname{Tr}(V^{\dagger}\Pi_{V_{:Y}}V)\\
				&= \operatorname{Tr}(K) - \operatorname{Tr}(\Pi_{V_{:Y}}VV^{\dagger})\\



	.. seealso::

		- :cite:`HKPV06` Algorithm 18 and Proposition 19, for the original idea
		- :cite:`KuTa12` Algorithm 1, for a first interpretation of :cite:`HKPV06` algorithm running in :math:`\mathcal{O}(N r^3)`
		- :cite:`Gil14` Algorithm 2, for the :math:`\mathcal{O}(N r^2)` implementation
		- :cite:`TrBaAm18` Algorithm 3, for a technical report on DPP sampling

		.. todo::

			- Refer to code also
			- Equivalence with Cholesky updates? 


Generic DPPs
~~~~~~~~~~~~

For ge





MCMC sampling
-------------

:cite:`AnGhRe16`
:cite:`LiJeSr16c`
:cite:`LiJeSr16d`
:cite:`GaBaVa17`


Basis exchange
~~~~~~~~~~~~~~

.. math::

	B' \leftrightarrow B \setminus s \cup t

Add-Delete
~~~~~~~~~~

.. math::

	S' \leftrightarrow S \setminus s \quad \text{Delete}\\
	S' \leftrightarrow S \cup t \quad \text{Add}

Add-Exchange-Delete
~~~~~~~~~~~~~~~~~~~

.. math::
	
	S' &\leftrightarrow S \setminus s \quad \text{Delete}\\
	S' &\leftrightarrow S \setminus s \cup t \quad \text{Exchange}\\
	S' &\leftrightarrow S \cup t \quad \text{Add}

Zonotope
~~~~~~~~

.. todo::

	Add random projection, low rank approximation of the kernel.






Continuous DPPs
===============

Definition
----------







Exotic DPPs
===============

Uniform spanning trees
----------------------

Carries process
---------------

RSK
---

Non intersecting random walks
-----------------------------





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
