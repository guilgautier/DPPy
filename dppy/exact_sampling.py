# coding: utf-8

import numpy as np
from numpy.core.umath_tests import inner1d
import scipy.linalg as la

########################################
################# DPPs #################
########################################

def dpp_sampler(K, ortho_proj_K=False, update_rule="GS"):
	""" Sample from :math:`\operatorname{DPP}(K)`, where :math:`K` is real symmetric with eigen-values in :math:`[0,1]`.

	:param K: Real symmetric kernel with eigen-values in :math:`[0,1]`
	:type K:
		array_type

	:param ortho_proj_K: 
		Indicate :math:`K` is an orthogonal projection kernel. 
		If ``ortho_proj_K=True``, diagonalization of :math:`K` is not necessary, thus not performed.
	:type ortho_proj_K:
		bool, default 'False'

	:param update_rule: 
		Indicate how the conditional probabilities i.e. the ratio of 2 determinants must be updated.

		If ``ortho_proj_K=True``:
			- "GS" (default): Gram-Schmidt on the columns of :math:`K` equiv Cholesky updates
			- "Schur": Schur complement updates
		
		If ``ortho_proj_K=False``:
			- 'GS' (default): Gram-Schmidt on the columns of :math:`K` equiv Cholesky updates
			- "KuTa12": Algorithm 1 in :cite:`KuTa12`
	:type update_rule:
		string, default 'GS'
		
	:return:
		A sample from :math:`\operatorname{DPP}(K)`.
	:rtype: 
		list

	.. seealso::

		Projection :math:`\operatorname{DPP}` samplers
			- :func:`projection_dpp_sampler_GS <projection_dpp_sampler_GS>`
			- :func:`projection_dpp_sampler_Schur <projection_dpp_sampler_Schur>`
		
		Generic :math:`\operatorname{DPP}` samplers
			- :func:`dpp_sampler_eig_GS <dpp_sampler_eig_GS>`
			- :func:`dpp_sampler_KuTa12 <dpp_sampler_KuTa12>`
	"""

	# Check the symmetry of the Kernel
	if not np.allclose(K, K.T):
		raise ValueError("Kernel doesn't seem to be symmetric")

	if ortho_proj_K == True:

		# Cheap test to check the reproducing property
		if not np.allclose(np.inner(K[0,:],K[0,:]), K[0,0]):
			raise ValueError("Kernel doesn't seem to be a projection")

		# Size of the sample = Trace of the Kernel
		# r = int(np.round(np.trace(K)))

		if update_rule == "GS":
			Y = projection_dpp_sampler_GS(K)

		elif update_rule == "Schur":
			Y = projection_dpp_sampler_Schur(K)

		else:
			raise ValueError("Invalid update rule for orthogonal projection kernels, choose among:\n\
				- 'GS' (default),\n\
				- 'Schur'")
	else:
		### Phase 1: 
		# Eigen-decompose the kernel (the symmetry was checked earlier)
		eig_vals, eig_vecs = la.eigh(K)

		# Check that the eigen-values lie in [0,1]
		if not np.all((0 <= eig_vals) & (eig_vals <= 1)):
			raise ValueError("Invalid Kernel: eigen-values are not in [0,1]")

		### Phase 2: 
		# 1) Select eigen-vectors by drawing Bernoulli variables with parameter the eigen-values
		# 2) Apply the chain rule on the underlying projection kernel
		#       K = eig_vecs_kept.dot(eig_vecs_kept.T) 
		#    without explicitely building it.
		if update_rule == "GS":
			# - Gram-Schmidt
			dpp_sampler_eig_GS(eig_vals, eig_vecs)

		if update_rule == "Cholesky":
			# - Cholesky
			dpp_sampler_eig_Cholesky(eig_vals, eig_vecs)

		elif update_rule == "KuTa12":
			# - Kulesza
			dpp_sampler_KuTa12(eig_vals, eig_vecs)

		else:
			raise ValueError("Invalid update rule for generic kernels, \
							choose among:\n\
							- 'GS' (default),\n\
							- 'Cholesky',\n\
							- 'KuTa12'")

	return Y

#########################
#### PROJECTION DPPs ####
#########################

def projection_dpp_sampler_GS(K, k=None):
	""" Sample from :math:`\operatorname{DPP}(K)` where the similarity kernel :math:`K` 
	is an orthogonal projection matrix.
	It performs sequential Gram-Schmidt orthogonalization or equivalently Cholesky decomposition updates of K.

	:param K: 
		Orthogonal projection kernel.
	:type K: 
		array_like
	:param k: 
		Size of the sample.
		Default is :math:`k=\operatorname{Tr}(K)=\operatorname{rank}(K)`.
	:type k: 
		int

	:return:
		A sample from :math:`\operatorname{DPP}(K)`.
	:rtype: 
		list

	.. seealso::
		- :func:`projection_dpp_sampler_Schur <projection_dpp_sampler_Schur>`
	"""

	# Size of the ground set
	N = K.shape[0]
	# Maximal size of the sample: Tr(K)=rank(K)
	r = int(np.round(np.trace(K)))
	# Size of the sample = k
	if k is None: # Full projection DPP
		k=r
	# else k-DPP(K) with K orthogonal projection

	# Initialization
	ground_set, rem_set = np.arange(N), np.full(N, True)
	# Sample 
	Y = np.zeros(k, dtype=int)

	c = np.zeros((N,k))
	d_2 = K.diagonal().copy()

	for it in range(k):

		j = np.random.choice(ground_set[rem_set], 
												size=1, 
												p=np.fabs(d_2[rem_set])/(r-it))[0]
		# Add the item to the sample
		rem_set[j], Y[it] = False, j

		###### Update the Cholesky factor
		c[rem_set, it] = (K[rem_set,j] - c[rem_set,:it].dot(c[j,:it]))\
										/np.sqrt(d_2[j])
		d_2[rem_set] -= c[rem_set,it]**2

	return Y
 
def projection_dpp_sampler_Schur(K, k=None):

	""" Sample from :math:`\operatorname{k-DPP}(K)` where the similarity kernel :math:`K` 
	is an orthogonal projection matrix.
	It sequentially updates the Schur complement by updating the inverse of the matrix involved.

	:param K: 
		Orthogonal projection kernel.
	:type K: 
		array_type

	:param k: 
		Size of the sample.
		Default is :math:`k=\operatorname{Tr}(K)=\operatorname{rank}(K)`.
	:type k: 
		int
			
	:return: 
		If ``k`` is not provided (None), 
			A sample from :math:`\operatorname{DPP}(K)`.
		If ``k`` is provided,
			A sample from :math:`\operatorname{k-DPP}(K)`.
	:rtype: 
		list

	.. seealso::

		- :func:`projection_dpp_sampler_GS <projection_dpp_sampler_GS>`
	"""

	# Size of the ground set
	N = K.shape[0]
	# Maximal size of the sample: Tr(K)=rank(K)
	r = int(np.round(np.trace(K)))
	# Size of the sample = k
	if k is None: # Full projection DPP
		k=r
	# else k-DPP(K) with K orthogonal projection

	# Initialization
	ground_set, rem_set = np.arange(N), np.full(N, True)
	# Sample 
	Y = np.zeros(k, dtype=int)

	K_diag = K.diagonal() # Used to compute the first term of Schur complement
	schur_comp = K_diag.copy() # Initialize the f

	for it in range(k):
		
		# Pick a new item
		j = np.random.choice(ground_set[rem_set], 
												size=1, 
												p=np.fabs(schur_comp[rem_set])/(r-it))[0]

		#### Update Schur complements K_ii - K_iY (K_Y)^-1 K_Yi for Y <- Y+j
		#
		# 1) use Woodbury identity to update K[Y,Y]^-1 to K[Y+j,Y+j]^-1
		# K[Y+j,Y+j]^-1 =
		# [ K[Y,Y]^-1 + (K[Y,Y]^-1 K[Y,j] K[j,Y] K[Y,Y]^-1)/schur_j  -K[Y,Y]^-1 K[Y,j]/schur_j] 
		# [ -K[j,Y] K[Y,Y]^-1/schur_j                                1/schur_j                ]

		if it == 0:
			K_inv=1.0/K[j,j]
		elif i == 1:
			K_inv=np.array([[K[j,j], -K[j,Y]], \
							[-K[j,Y], K[Y,Y]]] \
							)/(K[Y,Y]*K[j,j]-K[j,Y]**2)
		else:
			schur_j = K[j,j] - K[j,Y].dot(K_inv.dot(K[Y,j]))
			temp = K_inv.dot(K[Y,j])

			K_inv = np.lib.pad(K_inv, 
							(0,1), 
							'constant', 
							constant_values=1.0/schur_j)

			K_inv[:-1,:-1] += np.outer(temp, temp/schur_j)
			K_inv[:-1,-1] *= -temp
			K_inv[-1,:-1] = K_inv[:-1,-1]
			# K_inv[-1,-1] = 1.0/schur_j

		# Add the item to the sample
		rem_set[j], Y[it] = False, j
		
		# 2) update Schur complements
		# K_ii - K_iY (K_Y)^-1 K_Yi for Y <- Y+j
		schur_comp[rem_set] = K_diag[rem_set]\
													- inner1d(K[np.ix_(rem_set,Y)],
																		K[np.ix_(rem_set,Y)].dot(K_inv))
		
	return Y





######################
#### GENERIC DPPs ####
######################

# Using Gram-Schmidt orthogonalization via Cholesky updates
def dpp_sampler_eig_Cholesky(eig_vals, eig_vecs):
	""" Sample from :math:`\operatorname{DPP}(K)` using the eigen-decomposition of the similarity kernel :math:`K`. 
	It performs sequential updates of Cholesky decomposition of K.

	:param eig_vals: 
		Collection of eigen values of the similarity kernel :math:`K`.
	:type eig_vals: 
		list

	:param eig_vecs: 
		Eigen-vectors of the similarity kernel :math:`K`.
	:type eig_vecs: 
		array_type
		
	:return: 
		A sample from :math:`\operatorname{DPP}(K)`.
	:rtype: 
		list
  
	:Example:

		.. testcode::

				from exact_sampling import *
				np.random.seed(1234)

				r, n = 3, 10
				A = np.random.randn(r,n)
				K = A.T.dot(np.linalg.inv(A.dot(A.T)).dot(A))

				eig_vals, eig_vecs = np.linalg.eigh(K)

				print(dpp_sampler_eig_Cholesky(eig_vals, eig_vecs))

		.. testoutput::
				
			[1, 7, 9]



	.. seealso::
	
		- :func:`dpp_sampler_KuTa12 <dpp_sampler_KuTa12>`
		- :func:`dpp_sampler_eig_GS <dpp_sampler_eig_GS>`

	"""

	##### Phase 1: Select eigen-vectors with Bernoulli variables with parameter the eigen-values
	ind_bool = np.random.rand(len(eig_vals)) < eig_vals
	# if L-ensemble eig_vals = eig_vals/(1+eig_vals)

	# Feature vectors are the rows of V
	V = eig_vecs[:,ind_bool]

	# Size of the ground set, size of the sample (=rank=Trace of VV.T)
	N, k = V.shape 
	ground_set, rem_set = np.arange(N), np.full(N, True)
	# Sample
	Y = np.zeros(k, dtype=int)

	##### Phase 2: Chain rule
	# To compute the squared volume of the parallelepiped spanned by the feature vectors defining the sample
	# use Gram-Schmidt recursion aka Base x Height formula.

	# Initially this corresponds to the squared norm of the feature vectors
	c = np.zeros((N,k))
	norms_2 = inner1d(V, V)

	for it in range(k):

		# Pick an item \propto this squred distance
		j = np.random.choice(ground_set[rem_set], 
												size=1, 
												p=norms_2[rem_set]/(k-it))[0]

		# Add the item just picked    
		rem_set[j], Y[it] = False, j

		# Cancel the contribution of V_j to the remaining feature vectors
		c[rem_set, it] = V[rem_set,:].dot(V[j,:]) - c[rem_set,:it].dot(c[j,:it])
		c[rem_set, it] /= np.sqrt(norms_2[j])

		# Compute the square distance of the feature vectors to Span(V_Y:)
		norms_2[rem_set] -= c[rem_set,it]**2

	return Y.tolist()

# Using Gram-Schmidt orthogonalization
def dpp_sampler_eig_GS(eig_vals, eig_vecs):
	""" Sample from :math:`\operatorname{DPP}(K)` using the eigen-decomposition of the similarity kernel :math:`K`. 
	It performs sequential Gram-Schmidt orthogonalization of the rows of the selected eigen-vectors.
	It is equivalent to Cholesky decomposition updates of K.

	:param eig_vals: 
		Collection of eigen values of the similarity kernel :math:`K`.
	:type eig_vals: 
		list

	:param eig_vecs: 
		Eigen-vectors of the similarity kernel :math:`K`.
	:type eig_vecs: 
		array_type
		
	:return: 
		A sample from :math:`\operatorname{DPP}(K)`.
	:rtype: 
		list
  
	:Example:

		.. testcode::

				from exact_sampling import *
				np.random.seed(1234)

				r, n = 3, 10
				A = np.random.randn(r,n)
				K = A.T.dot(np.linalg.inv(A.dot(A.T)).dot(A))

				eig_vals, eig_vecs = np.linalg.eigh(K)

				print(dpp_sampler_eig_GS(eig_vals, eig_vecs))

		.. testoutput::
				
			[1, 7, 9]

	.. seealso::
	
		- :func:`dpp_sampler_eig_Cholesky <dpp_sampler_eig_Cholesky>`
		- :func:`dpp_sampler_KuTa12 <dpp_sampler_KuTa12>`
	"""

	##### Phase 1: Select eigen-vectors with Bernoulli variables with parameter the eigen-values
	# if L-ensemble eig_vals = eig_vals/(1+eig_vals)
	ind_bool = np.random.rand(len(eig_vals)) < eig_vals

	# Feature vectors are the rows of V
	V = eig_vecs[:,ind_bool]

	# Size of the ground set, size of the sample (=rank=Trace of VV.T)
	N, k = V.shape 
	ground_set, rem_set = np.arange(N), np.full(N, True)
	# Sample
	Y = []

	##### Phase 2: Chain rule
	# To compute the squared volume of the parallelepiped spanned by the feature vectors defining the sample
	# use Gram-Schmidt recursion aka Base x Height formula.

	### Matrix of the contribution of remaining vectors V_i onto the orthonormal basis {e_j}_Y of V_Y
	# <V_i,P_{V_Y}^{orthog} V_j>
	contrib = np.zeros((N,k))

	### Residual square norm 
	# ||P_{V_Y}^{orthog} V_j||^2
	norms_2 = inner1d(V, V)

	for it in range(k):
		
		# Pick an item proportionally to the residual square norm 
		# ||P_{V_Y}^{orthog} V_j||^2
		j = np.random.choice(ground_set[rem_set], 
												size=1, 
												p=np.fabs(norms_2[rem_set])/(k-it))[0] 
		
		### Update the residual square norm
		#
		# |P_{V_Y+j}^{orthog} V_i|^2        
		#                                    <V_i,P_{V_Y}^{orthog} V_j>^2
		#     =  |P_{V_Y}^{orthog} V_i|^2 -  ----------------------------
		#                                      |P_{V_Y}^{orthog} V_j|^2
		

		## 1) Orthogonalize V_j w.r.t. orthonormal basis of Span(V_Y)
		#    V'_j = P_{V_Y}^{orthog} V_j
		#         = V_j - <V_j,∑_Y V'_k>V"_k
		# Note V'_j is not normalized
		V[j,:] -= contrib[j,:it].dot(V[Y,:])

		# Make the item selected unavailable
		rem_set[j] = False
		
		## 2) Compute <V_i,V'_j> = <V_i,P_{V_Y}^{orthog} V_j>
		contrib[rem_set,it] = V[rem_set,:].dot(V[j,:])
		
		## 3) Normalize V'_j with norm^2 and not norm
		#              V'_j         P_{V_Y}^{orthog} V_j
		#    V"_j  =  -------  =  --------------------------
		#             |V'j|^2      |P_{V_Y}^{orthog} V_j|^2
		V[j,:] /= norms_2[j]
		# for next orthogonalization in 1) 
		#                          	<V_i,P_{V_Y}^{orthog} V_j> P_{V_Y}^{orthog} V_j
		#  V_i - <V_i,V'_j>V"_j = V_i - -----------------------------------------
		#                                           |P_{V_Y}^{orthog} V_j|^2


		## 4) Update the residual square norm by cancelling the contribution of V_i onto V_j
		#                            
		# |P_{V_Y+j}^{orthog} V_i|^2 
		#		= |P_{V_Y}^{orthog} V_i|^2 - <V_i,V'_j>^2 / |V'j|^2    
		#
		#  																	<V_i,P_{V_Y}^{orthog} V_j>^2
		#   =  |P_{V_Y}^{orthog} V_i|^2 -		----------------------------
		#                                     |P_{V_Y}^{orthog} V_j|^2

		norms_2[rem_set] -= (contrib[rem_set,it]**2)/norms_2[j]
		
		# Add the item
		Y.append(j)

	return Y

def dpp_sampler_KuTa12(eig_vals, eig_vecs):
	""" Sample from :math:`\operatorname{DPP}(K)` using the eigen-decomposition of the similarity kernel :math:`K`. 
	It is based on the orthogonalization of the selected eigen-vectors.

	:param eig_vals: 
		Collection of eigen values of the similarity kernel :math:`K`.
	:type eig_vals: 
		list

	:param eig_vecs: 
		Eigen-vectors of the similarity kernel :math:`K`.
	:type eig_vecs: 
		array_type
		
	:return: 
		A sample from :math:`\operatorname{DPP}(K)`.
	:rtype: 
		list

	.. seealso::

		Algorithm 1 in :cite:`KuTa12`

	"""

	# Phase 1: 
	# Select eigen vectors \propto eig_vals
	ind_bool = np.random.rand(len(eig_vals)) < eig_vals
	# Stack the selected eigen-vectors
	V = eig_vecs[:,ind_bool]
	# N = size of the ground set, n = size of the sample
	N, k = V.shape 

	#### Phase 2: Chain rule
	# Initialize the sample
	norms_2 = inner1d(V,V)
	# Pick an item
	i = np.random.choice(N, size=1, p=np.fabs(norms_2)/k)[0] 
	# Add the item just picked
	Y = np.zeros(k,dtype=int)
	Y[0]=i
	
	# Following [Algo 1, KuTa12], the aim is to compute the orhto complement of the subspace spanned by the selected eigen-vectors to the canonical vectors \{e_i ; i \in Y\}. We proceed recursively.
	for it in range(1,k):
		
		# Cancel the contribution of e_i to the remaining vectors that is, find the subspace of V that is orthogonal to \{e_i ; i \in Y\}

		# Take the index of a vector that has a non null contribution along e_i
		j = np.where(V[i,:]!=0)[0][0]
		# Cancel the contribution of the remaining vectors along e_i, but stay in the subspace spanned by V i.e. get the subspace of V orthogonal to \{e_i ; i \in Y\}
		V -= np.outer(V[:,j]/V[i,j], V[i,:])
		# V_:j is set to 0 so we delete it and we can derive an orthononormal basis of the subspace under consideration
		V, _ = np.linalg.qr(np.delete(V, j, axis=1)) 

		norms_2 = inner1d(V,V) 
		# Pick an item
		i = np.random.choice(N, size=1, p=np.fabs(norms_2)/(r-it))[0]
		# Add the item just picked
		Y[it] = i

	return Y





















##########################################
################# k-DPPs #################
##########################################

def k_dpp_sampler(K, k=1, ortho_proj_K=False, update_rule="GS"):
	""" Sample from :math:`\operatorname{DPP}(K)`, where :math:`K` is real symmetric with eigen-values in :math:`[0,1]`.

	:param K: Real symmetric kernel with eigen-values in :math:`[0,1]`
	:type K:
		array_type

	:param k: Size of the sample
	:type k:
		int, default '1'

	:param ortho_proj_K: 
		Indicate :math:`K` is an orthogonal projection kernel. 
		If ``ortho_proj_K=True``, diagonalization of :math:`K` is not necessary, thus not performed.
	:type ortho_proj_K:
		bool, default 'False'

	:param update_rule: 
		Indicate how the conditional probabilities i.e. the ratio of 2 determinants must be updated.

		If ``ortho_proj_K=True``:
			- "GS" (default): Gram-Schmidt on the columns of :math:`K` equiv Cholesky updates
			- "Schur": Schur complement updates
		
		If ``ortho_proj_K=False``:
			- 'GS' (default): Gram-Schmidt on the columns of :math:`K` equiv Cholesky updates
			- "KuTa12": Algorithm 1 in :cite:`KuTa12`
	:type update_rule:
		string, default 'GS'
		
	:return:
		A sample from :math:`\operatorname{DPP}(K)`.
	:rtype: 
		list

	.. seealso::

		Projection :math:`\operatorname{DPP}` samplers
			- :func:`projection_dpp_sampler_GS <projection_dpp_sampler_GS>`
			- :func:`projection_dpp_sampler_Schur <projection_dpp_sampler_Schur>`
		
		Generic :math:`\operatorname{DPP}` samplers
			- :func:`dpp_sampler_eig_GS <dpp_sampler_eig_GS>`
			- :func:`dpp_sampler_eig_Cholesky <dpp_sampler_eig_Cholesky>`
			- :func:`dpp_sampler_KuTa12 <dpp_sampler_KuTa12>`
	"""

	# Check the symmetry of the Kernel
	if not np.allclose(K, K.T):
		raise ValueError("Kernel doesn't seem to be symmetric")

	# Check k>0 integer for k-DPP
	if (k <= 0) & (not isinstance(k, int)):
		raise ValueError("k parameter must be a positive integer")



	if ortho_proj_K == True:

		# Cheap test to check the reproducing property
		if not np.allclose(np.inner(K[0,:],K[0,:]), K[0,0]):
			raise ValueError("Kernel doesn't seem to be a projection")

		# Size of the sample = Trace of the Kernel
		# r = int(np.round(np.trace(K)))

		if update_rule == "GS":
			Y = projection_dpp_sampler_GS(K, k)

		elif update_rule == "Schur":
			Y = projection_dpp_sampler_Schur(K, k)

		else:
			str_list = ["Invalid update_rule for orthogonal projection kernels, choose among:",
				"- 'GS' (default)",
				"- 'Schur'",
				"Given 'update_rule' = {}"]
			raise ValueError('\n'.join(str_list).format(update_rule))
	else:
		### Phase 1: 
		# Eigen-decompose the kernel (the symmetry was checked earlier)
		eig_vals, eig_vecs = np.linalg.eigh(K)

		# Check that the eigen-values lie in [0,1]
		if not np.all((0 <= eig_vals) & (eig_vals <= 1)):
			raise ValueError("Invalid Kernel: eigen-values are not in [0,1]")

		eig_vals = select_eig_vec(eig_val, k)

		### Phase 2: 
		# 1) Select eigen-vectors by drawing Bernoulli variables with parameter the eigen-values
		# 2) Apply the chain rule on the underlying projection kernel
		#       K = eig_vecs_kept.dot(eig_vecs_kept.T) 
		#    without explicitely building it.
		if update_rule == "GS":
			# - Gram-Schmidt
			dpp_sampler_eig_GS(eig_vals, eig_vecs)

		if update_rule == "Cholesky":
			# - Cholesky
			dpp_sampler_eig_Cholesky(eig_vals, eig_vecs)

		elif update_rule == "KuTa12":
			# - Kulesza
			dpp_sampler_KuTa12(eig_vals, eig_vecs)

		else:
			raise ValueError("Invalid update rule for generic kernels, \
							choose among:\n\
							- 'GS' (default),\n\
							- 'Cholesky',\n\
							- 'KuTa12'")

	return Y



#######################################################
# From the eigen decomposition of the kernel :math:`K`

# Evaluate the elementary symmetric polynomials
def elem_symm_poly(eig_vals, k):
	""" Evaluate the elementary symmetric polynomials in the eigen-values of the similarity kernel :math:`K`.

	:param eig_vals: 
		Collection of eigen values of the similarity kernel :math:`K`.
	:type eig_vals: 
		list
	
	:param k: 
		Maximum degree of elementary symmetric polynomial.
	:type k: 
		int
			
	:return: 
		poly(k,N) = :math:`e_k(\lambda_1, \cdots, \lambda_N)`
	:rtype: 
		array_type

	.. seealso::

		Algorithm 7 in :cite:`KuTa12`

		- :func:`k_dpp_KuTa12 <k_dpp_KuTa12>`
	"""

	# Number of variables for the elementary symmetric polynomials to be evaluated
	N = len(eig_vals)
	# Initialize output array
	poly = np.zeros((k+1, N+1)) 
	poly[0, :] = 1

	# Recursive evaluation
	for l in range(1, k+1):
		for n in range(1, N+1):
			poly[l, n] = poly[l, n-1] + eig_vals[n-1] * poly[l-1, n-1]

	return poly

def select_eig_vec(eig_vals, k):
	""" Select the :math:`k` eigenvectors to sample for :math:`\operatorname{k-DPP}(K)`.

	:param eig_vals: 
		Collection of eigen values of the similarity kernel :math:`K`.
	:type eig_vals: 
		list
	
	:param k: 
		Size of the sample.
	:type k: 
		int

	:return: 
		The  :math:`\operatorname{k-DPP}(K)`.
	:rtype: 
		list
			
	.. seealso::

		Algorithm 8 in :cite:`KuTa12` 
	"""

	E = elem_symm_poly(eig_vals, k) # Evaluate the elem symm polys in the eigenvalues 
	N = len(eig_vals) # Size of the ground set

	# Eigenvectors to be kept
	S = [] # Indices
	l = k # Number i.e. size of the sample

	for n in range(N,0,-1):
		if l == 0:
			break
		if np.random.rand() < eig_vals[n-1]*(E[l-1,n-1]/E[l,n]):
			S.append(n-1)
			l-=1

	eig_v = np.zeros(N)
	eig_v[S] = 1.0

	return eig_v