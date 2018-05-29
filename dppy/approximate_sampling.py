import time
import numpy as np
import scipy.linalg as la

from cvxopt import matrix, spmatrix, solvers
solvers.options['show_progress'] = False
solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')

def det_kernel_ST(kernel, S, T=None):

	if T is None:
		if S:
			return la.det(kernel[np.ix_(S, S)]) # det K_SS = det K_S
		else:
			return 1 # = det K_emptyset = 1

	elif S and T:
		return la.det(kernel[np.ix_(S, T)]) # det K_ST

	elif not S and not T:
		return 1

	else:
		raise ValueError("Big problem")

###############################################################
############## Approximate samplers for projection DPPs #######
###############################################################

def dpp_sampler_approx(kernel, sampling_mode="AED", **params):

	s_init = params.get("s_init", None)
	nb_it_max = params.get("nb_it_max", 10)
	T_max = params.get("T_max", None)
	size = params.get("size", None) # For projection K-ensemble size = Tr(K)

	if sampling_mode == "AED":
		if s_init is None:
			s_init = initialize_AED_sampler(kernel)
		sampl = add_exchange_delete_sampler(kernel, s_init, nb_it_max, T_max)

	elif sampling_mode == "AD":
		if s_init is None:
			s_init = initialize_AD_and_E_sampler(kernel)
		sampl = add_delete_sampler(kernel, s_init, nb_it_max, T_max)

	elif sampling_mode == "E":
		if s_init is None:
			s_init = initialize_AD_and_E_sampler(kernel, size)
		sampl = basis_exchange_sampler(kernel, s_init, nb_it_max, T_max)

	return sampl

def initialize_AED_sampler(kernel):

	N = kernel.shape[0]
	ground_set = np.arange(N)

	S0, det_S0 = [], 0.0
	nb_it_max = 100
	tol = 1e-9

	for _ in range(nb_it_max):
		if det_S0 > tol:
			break
		else:
			T = np.random.choice(2*N, size=N, replace=False)
			S0 = np.intersect1d(T, ground_set).tolist()
			det_S0 = det_kernel_ST(kernel, S0)
	else:
		raise ValueError("Initialization problem!")

	return S0

def initialize_AD_and_E_sampler(kernel, size=None):

	N = kernel.shape[0]
	ground_set = np.arange(N)

	S0, det_S0 = [], 0.0
	nb_it_max = 100
	tol = 1e-9
	
	for _ in range(nb_it_max):
		if det_S0 > tol:
			break
		else:
			sz = size if size else np.random.choice(ground_set[1:], size=1)[0]
			S0 = np.random.choice(ground_set, size=sz, replace=False).tolist()
			det_S0 = det_kernel_ST(kernel, S0)
	else:
		raise ValueError("Initialization problem!")

	return S0

def add_exchange_delete_sampler(kernel, s_init=None, nb_it_max=10, T_max=None):
	""" MCMC sampler for generic DPPs, it is a mix of add/delete and basis exchange MCMC samplers.

	:param kernel:
			Kernel martrix
	:type kernel:
			array_type

	:param s_init:
			Initial sample.
	:type s_init: 
			list

	:param nb_it_max:
		Maximum number of iterations performed by the the algorithm.
		Default is 10.
	:type nb_it_max: 
		int

	:param T_max: 
		Maximum running time of the algorithm (in seconds).
	:type T_max: 
			float

	:return:
		list of `nb_it_max` approximate sample of DPP(kernel)
	:rtype:
		array_type

	.. seealso::
			
			Algorithm 3 in :cite:`LiJeSr16c`
	"""

	N = kernel.shape[0]
	ground_set = np.arange(N)

	# Initialization
	S0, det_S0 = s_init, det_kernel_ST(kernel, s_init)
	sampl_size = len(S0) # Size of the current sample
	samples = [S0] # Initialize the collection (list) of sample
	
	# Evaluate running time...
	flag = True
	it = 1
	t_start = time.time() if T_max else 0

	while flag:

		S1 = S0.copy() # S1 = S0
		# Uniform s in S_0 by index
		s = np.random.choice(sampl_size, size=1)[0] 
		# Unif t in [N]-S_0
		t = np.random.choice(np.delete(ground_set, S0), size=1)[0] 

		unif_01 = np.random.rand() 
		ratio = sampl_size/N # Proportion of items in current sample

		# Add: S1 = S0 + t
		if unif_01 < 0.5*(1-ratio)**2:
			S1.append(t) # S1 = S0 + t
			# Accept_reject the move
			det_S1 = det_kernel_ST(kernel, S1) # det K_S1
			if np.random.rand() < (det_S1/det_S0 * (sampl_size+1)/(N-sampl_size)):
				S0, det_S0 = S1, det_S1
				samples.append(S1)
				sampl_size += 1

		# Exchange: S1 = S0 - s + t
		elif (0.5*(1-ratio)**2 <= unif_01) & (unif_01 < 0.5*(1-ratio)):
			del S1[s] # S1 = S0 - s
			S1.append(t) # S1 = S1 + t = S0 - s + t
			# Accept_reject the move
			det_S1 = det_kernel_ST(kernel, S1) # det K_S1
			if np.random.rand() < (det_S1/det_S0):
				S0, det_S0 = S1, det_S1
				samples.append(S1)
				# sampl_size stays the same

		# Delete: S1 = S0 - s
		elif (0.5*(1-ratio) <= unif_01) & (unif_01 < 0.5*(ratio**2+(1-ratio))):
			del S1[s] # S0 - s
			# Accept_reject the move
			det_S1 = det_kernel_ST(kernel, S1) # det K_S1
			if np.random.rand() < (det_S1/det_S0 * sampl_size/(N-(sampl_size-1))):
				S0, det_S0 = S1, det_S1
				samples.append(S1)
				sampl_size -= 1

		else:
			samples.append(S0)
		
		it += 1
		flag = (it < nb_it_max) if not T_max else ((time.time()-t_start) < T_max)

	return samples

def add_delete_sampler(kernel, s_init, nb_it_max=10, T_max=10):
	""" MCMC sampler for generic DPP(kernel), it performs local moves by removing/adding one element at a time.

	:param kernel:
			Kernel martrix
	:type kernel:
			array_type

	:param s_init:
			Initial sample.
	:type s_init: 
			list

	:param nb_it_max:
			Maximum number of iterations performed by the the algorithm.
			Default is 10.
	:type nb_it_max: 
			int

	:param T_max: 
			Maximum running time of the algorithm (in seconds).
			Default is 10s.
	:type T_max: 
			float

	:return:
		list of `nb_it_max` approximate sample of DPP(kernel)
	:rtype:
		array_type

	.. seealso::
			
			Algorithm 1 in :cite:`LiJeSr16c`
	"""

	# Initialization
	N = kernel.shape[0] # Number of elements

	# Initialization
	S0, det_S0 = s_init, det_kernel_ST(kernel, s_init)
	samples = [S0] # Initialize the collection (list) of sample
	
	# Evaluate running time...
	flag = True
	it = 1
	t_start = time.time() if T_max else 0

	while flag:

		# With proba 1/2 try to add/delete an element
		if np.random.rand() < 0.5: 

			# Perform the potential add/delete move S1 = S0 +/- s
			S1 = S0.copy() # S1 = S0
			s = np.random.choice(N, size=1)[0] # Uniform item in [N]
			if s in S0: 
				S1.remove(s) # S1 = S0 - s
			else: 
				S1.append(s) # S1 = SO + s

			# Accept_reject the move
			det_S1 = det_kernel_ST(kernel, S1) # det K_S1
			if np.random.rand() < det_S1/det_S0:
				S0, det_S0 = S1, det_S1
				samples.append(S1)

			else:
				samples.append(S0)

		else:
			samples.append(S0)
		
		it += 1
		flag = (it < nb_it_max) if not T_max else ((time.time()-t_start) < T_max)

	return samples

def basis_exchange_sampler(kernel, s_init, nb_it_max=10, T_max=10):
	""" MCMC sampler for projection DPPs, based on the basis exchange property.

	:param kernel:
			Feature vector matrix, feature vectors are stacked columnwise.
			It is assumed to be full row rank.
	:type kernel:
			array_type

	:param s_init:
			Initial sample.
	:type s_init: 
			list

	:param nb_it_max:
			Maximum number of iterations performed by the the algorithm.
			Default is 10.
	:type nb_it_max: 
			int

	:param T_max: 
			Maximum running time of the algorithm (in seconds).
			Default is 10s.
	:type T_max: 
			float

	:return:
			MCMC chain of approximate sample (stacked row_wise i.e. nb_it_max rows).
	:rtype: 
			array_type

	.. seealso::
			
			Algorithm 2 in :cite:`LiJeSr16c`
	"""

	# Initialization
	N = kernel.shape[0] # Number of elements
	ground_set = np.arange(N) # Ground set

	size = len(s_init) # Size of the sample (cardinality is fixed)
	# Initialization
	S0, det_S0 = s_init, det_kernel_ST(kernel, s_init)
	samples = [S0] # Initialize the collection (list) of sample
	
	# Evaluate running time...
	flag = True
	it = 1
	t_start = time.time() if T_max else 0

	while flag:

		# With proba 1/2 try to swap 2 elements
		if np.random.rand() < 0.5: 

			# Perform the potential exchange move S1 = S0 - s + t
			S1 = S0.copy() # S1 = S0
			# Pick one element in S_0 by index uniformly at random
			rnd_ind = np.random.choice(size, size=1)[0]
			# Pick one element t in [N]\S_0 uniformly at random
			t = np.random.choice(np.delete(ground_set, S0), size=1)[0]
			S1[rnd_ind] = t # S_1 = S_0 - S_0[rnd_ind] + t
			
			det_S1 = det_kernel_ST(kernel, S1) # det K_S1

			# Accept_reject the move w. proba
			if np.random.rand() < det_S1/det_S0:
				S0, det_S0 = S1, det_S1
				samples.append(S1)

			else: # if reject, stay in the same state
				samples.append(S0)	

		else:
			samples.append(S0)

		it += 1
		flag = (it < nb_it_max) if not T_max else ((time.time()-t_start) < T_max)

	return samples



######################
###### ZONOTOPE ######
######################

def extract_basis(y_sol, eps=1e-5):
	""" Subroutine of zono_sampling to extract the tile of the zonotope 
	in which a point lies. It extracts the indices of entries of the solution 
	of LP :eq:`Px` that are in (0,1).

	:param y_sol: 
			Optimal solution of LP :eq:`Px`
	:type y_sol: 
			list
					
	:param eps: 
			Tolerance :math:`y_i^* \in (\epsilon, 1-\epsilon), \quad \epsilon \geq 0`
	:eps type: 
			float

	:return: 
			Indices of the feature vectors spanning the tile in which the point is lies. 
			:math:`B_{x} = \left\{ i \, ; \, y_i^* \in (0,1) \\right\}`
	:rtype: 
			list         
					
	.. seealso::

			Algorithm 3 in :cite:`GaBaVa17`

			- :func:`zono_sampling <zono_sampling>`
	"""

	basis = np.where((eps<y_sol)&(y_sol<1-eps))[0]
	
	return basis.tolist()

def zonotope_sampler(A_zono, **params):
	""" MCMC based sampler for projection DPPs.
	The similarity matrix is the orthogonal projection matrix onto 
	the row span of the feature vector matrix.
	Samples are of size equal to the ransampl_size of the projection matrix 
	also equal to the rank of the feature matrix (assumed to be full row rank).

	:param A_zono:
			Feature vector matrix, feature vectors are stacked columnwise.
			It is assumed to be full row rank.
	:type A_zono:
			array_type

	:param c: Linear objective of the linear program used to identify 
			the tile in which a point lies.
	:type c: list

	:param nb_it_max:
			Maximum number of iterations performed by the the algorithm.
			Default is 10.
	:type nb_it_max: 
			int

	:param T_max: 
			Maximum running time of the algorithm (in seconds).
			Default is 10s.
	:type T_max: 
			float

	:return:
			MCMC chain of approximate sample (stacked row_wise i.e. nb_it_max rows).
	:rtype: 
			array_type    

	.. seealso::

			Algorithm 5 in :cite:`GaBaVa17`

			- :func:`extract_basis <extract_basis>`
			- :func:`basis_exchange_sampler <basis_exchange_sampler>`
	"""

	r, N = A_zono.shape # Sizes of r=samples=rank(A_zono), N=ground set
	# Linear objective
	c = params.get("lin_obj", matrix(np.random.randn(N)))
	# Initial point x0 = A*u, u~U[0,1]^n
	x0 = params.get("x0", matrix(A_zono.dot(np.random.rand(N))))
	nb_it_max = params.get("nb_it_max", 10)
	T_max = params.get("T_max", None)
	#################################################
	############### Linear problems #################
	#################################################
	# Canonical form
	# min		c.T*x 				min		c.T*x
	# s.t.	G*x <= h	<=> s.t.	G*x + s = h
	#				A*x = b 						A*x = b
	#														s >= 0
	# CVXOPT
	# =====> solvers.lp(c, G, h, A, b, solver='glpk')
	#################################################

	### To access the tile Z(B_x) 
	# Solve P_x(A,c)
	#########################################################
	# y^* =
	# argmin  c.T*y       argmin  c.T*y
	# s.t.		A*y = x 		<=> s.t.    	A  *y  =	x
	#					0 <= y <= 1						[ I_n] *y <=	[1^n]
	#																[-I_n]				[0^n]
	#########################################################
	# Then B_x = \{ i ; y_i^* \in ]0,1[ \}

	A = spmatrix(0.0, [], [], (r, N))
	A[:,:] = A_zono

	G = spmatrix(0.0, [], [], (2*N, N))
	G[:N,:] = spmatrix(1.0, range(N), range(N))
	G[N:,:] = spmatrix(-1.0, range(N), range(N))

	### Endpoints of segment 
	# D_x \cap Z(A) = [x+alpha_m*d, x-alpha_M*d]
	#############################################################################
	# alpha_m/_M = 
	# argmin  +/-alpha 									argmin  [+/-1 0^N].T * [alpha,lambda]
	# s.t.    x + alpha d = A lambda  <=>   s.t.  [-d A] * [alpha,lambda] = x
	#					0 <= lambda <= 1            [ 0^N I_N ] *[alpha,lambda] <=  [1^N]
	#																			[ 0^N -I_N]             				[0^N]
	#############################################################################

	c_mM = matrix(0.0, (N+1,1))
	c_mM[0] = 1.0

	A_mM = spmatrix(0.0, [], [], (r, N+1))
	A_mM[:,1:] = A

	G_mM = spmatrix(0.0, [], [], (2*N, N+1))
	G_mM[:,1:] = G

	# Common h to both kind of LP
	# cf. 0 <= y <= 1 and 0 <= lambda <= 1
	h = matrix(0.0, (2*N, 1))
	h[:N,:] = 1.0

	########################
	#### Initialization ####
	########################
	B_x0 = []
	while len(B_x0) != r:
		# Initial tile B_x0
				# Solve P_x0(A,c)
		y_star = solvers.lp(c, G, h, A, x0, solver='glpk')['x']
				# Get the tile
		B_x0 = extract_basis(np.asarray(y_star))

	# Initialize sequence of sample
	Bases = [B_x0]

	# Compute the det of the tile (Vol(B)=abs(det(B)))
	det_B_x0 = la.det(A_zono[:,B_x0])

	it, t_start = 1, time.time()
	flag = it < nb_it_max
	while flag:

		# Take uniform direction d defining D_x0
		d = matrix(np.random.randn(r, 1))

		# Define D_x0 \cap Z(A) = [x0 + alpha_m*d, x0 - alpha_M*d]
			# Update the constraint [-d A] * [alpha,lambda] = x
		A_mM[:,0] = -d
				# Find alpha_m/M
		alpha_m = solvers.lp(c_mM, G_mM, h, A_mM, x0, solver='glpk')['x'][0]
		alpha_M = solvers.lp(-c_mM, G_mM, h, A_mM, x0, solver='glpk')['x'][0]

		# Propose x1 ~ U_{[x0+alpha_m*d, x0-alpha_M*d]}
		x1 = x0 + (alpha_m + (alpha_M-alpha_m)*np.random.rand())*d
		# Proposed tile B_x1
			# Solve P_x1(A,c)
		y_star = solvers.lp(c, G, h, A, x1, solver='glpk')['x']
				# Get the tile
		B_x1 = extract_basis(np.asarray(y_star))

		# Accept/Reject the move with proba Vol(B1)/Vol(B0)
		if len(B_x1) != r: # In case extract_basis returned smtg ill conditioned
			Bases.append(B_x0)
		else:
			det_B_x1 = la.det(A_zono[:, B_x1])
			if np.random.rand() < abs(det_B_x1/det_B_x0):
					x0, B_x0, det_B_x0 = x1, B_x1, det_B_x1
					Bases.append(B_x1)
			else:
					Bases.append(B_x0)

		it += 1
		flag = (it < nb_it_max) if not T_max else ((time.time()-t_start) < T_max)

	return np.array(Bases)