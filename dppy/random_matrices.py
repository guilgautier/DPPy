# coding: utf-8

import numpy as np
import scipy.linalg as la 

###############
### Hermite ###
###############

# Hermite, full matrix model
def hermite_sampler_full(N, beta=2):

	size_sym_mat = int(N*(N-1)/2)

	if beta==1:
		A = np.random.randn(N, N)

	elif beta==2:
		A = np.random.randn(N, N) + 1j*np.random.randn(N, N)
		
	elif beta==4:
		X = np.random.randn(N, N) + 1j*np.random.randn(N, N)
		Y = np.random.randn(N, N) + 1j*np.random.randn(N, N)

		A = np.block([  [X,            Y       ],\
						[-Y.conj(),    X.conj()]])

	else:
		raise ValueError("Invalid beta parameter.\n"
						"beta coefficient must be equal to 1, 2 or 4"
						"Given beta={}".format(beta))

	# return la.eigvalsh(A+A.conj().T)
	return la.eigvalsh(A+A.conj().T)/np.sqrt(2.0)

## Hermite tridiag
def hermite_sampler_tridiag(N, beta=2):

	alpha_coef = np.sqrt(2)*np.random.randn(N)
	beta_coef = np.random.chisquare(beta*np.arange(N-1, 0, step=-1))

	return la.eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))

# Semi-circle law
def semi_circle_law(x, R=2.0):
	# [Dubbs & Edelman, Table 1] https://arxiv.org/pdf/1502.04931.pdf
	# https://en.wikipedia.org/wiki/Wigner_semicircle_distribution
	return 2/(np.pi*R**2) * np.sqrt(R**2 - x**2)

## mu_ref == normal
def muref_normal_sampler_tridiag(loc=0.0, scale=1.0, beta=2, size=10):

	# beta/2*[N-1, N-2, ..., 1]
	b_2_Ni = 0.5*beta*np.arange(size-1, 0, step=-1)

	alpha_coef = np.random.normal(loc=loc, scale=scale, size=size)
	beta_coef = np.random.gamma(shape=b_2_Ni, scale=scale**2)

	return la.eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))




################
### Laguerre ###
################

# Laguerre, full matrix model
def laguerre_sampler_full(M, N, beta=2):
	
	if beta==1:
		A = np.random.randn(N, M)

	elif beta==2:
		A = np.random.randn(N, M) + 1j*np.random.randn(N, M)

	elif beta==4:
		X = np.random.randn(N, M) + 1j*np.random.randn(N, M)
		Y = np.random.randn(N, M) + 1j*np.random.randn(N, M)
		A = np.block([  [X,         Y       ],\
						[-Y.conj(), X.conj()]])

	else:
		raise ValueError("Invalid beta parameter.\n"
						"beta coefficient must be equal to 1, 2 or 4"
						"Given beta={}".format(beta))

	return la.eigvalsh(A.dot(A.conj().T))

## Laguerre, tridiagonal model
def laguerre_sampler_tridiag(M, N, beta=2):
	# M=>N

	# xi_odd = xi_1, ... , xi_2N-1
	xi_odd = np.random.chisquare(beta*np.arange(M, M-N, step=-1)) # odd

	# xi_even = xi_0=0, xi_2, ... ,xi_2N-2
	xi_even = np.zeros(N)
	xi_even[1:] = np.random.chisquare(beta*np.arange(N-1, 0, step=-1)) # even

	# alpha_i = xi_2i-2 + xi_2i-1
	# alpha_1 = xi_0 + xi_1 = xi_1
	alpha_coef = xi_even + xi_odd
	# beta_i+1 = xi_2i-1 * xi_2i
	beta_coef = xi_odd[:-1] * xi_even[1:]
	
	return la.eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))

# Marcenko Pastur law
def marcenko_pastur_law(x, M, N, sigma=1.0):
	# https://en.wikipedia.org/wiki/Marchenko-Pastur_distribution
	# M>=N
	c = N/M
	Lm, Lp = (sigma*(1-np.sqrt(c)))**2, (sigma*(1+np.sqrt(c)))**2

	return 1.0/(2*np.pi*sigma**2) * 1.0/(c*x) *np.sqrt(np.maximum((Lp-x)*(x-Lm),0))

## mu_ref == Gamma 
def mu_ref_gamma_sampler_tridiag(shape=1.0, scale=1.0, beta=2, size=10):

	# beta/2*[N-1, N-2, ..., 1, 0]
	b_2_Ni = 0.5*beta*np.arange(size-1,-1,step=-1)

	# xi_odd = xi_1, ... , xi_2N-1
	xi_odd = np.random.gamma(shape=b_2_Ni + shape, scale=scale) # odd

	# xi_even = xi_0=0, xi_2, ... ,xi_2N-2
	xi_even = np.zeros(size)
	xi_even[1:] = np.random.gamma(shape=b_2_Ni[:-1], scale=scale) # even

	# alpha_i = xi_2i-2 + xi_2i-1
	# alpha_1 = xi_0 + xi_1 = xi_1
	alpha_coef = xi_even + xi_odd
	# beta_i+1 = xi_2i-1 * xi_2i
	beta_coef = xi_odd[:-1] * xi_even[1:]
	
	return la.eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))






##############
### Jacobi ###
##############

# Jacobi, full matrix model
def jacobi_sampler_full(M_1, M_2, N, beta=2):
	
	if beta==1:
		X = np.random.randn(N, M_1)
		Y = np.random.randn(N, M_2)

	elif beta==2:
		X = np.random.randn(N, M_1) + 1j*np.random.randn(N, M_1)
		Y = np.random.randn(N, M_2) + 1j*np.random.randn(N, M_2)

	elif beta==4:
		X_1 = np.random.randn(N, M_1) + 1j*np.random.randn(N, M_1)
		X_2 = np.random.randn(N, M_1) + 1j*np.random.randn(N, M_1)

		Y_1 = np.random.randn(N, M_2) + 1j*np.random.randn(N, M_2)
		Y_2 = np.random.randn(N, M_2) + 1j*np.random.randn(N, M_2)

		X = np.block([  [X_1,         X_2         ],\
										[-X_2.conj(), X_1.conj()]])
		Y = np.block([  [Y_1,         Y_2         ],\
										[-Y_2.conj(), Y_1.conj()]])

	else:
		raise ValueError("Invalid beta parameter.\n"
										"beta coefficient must be equal to 1, 2 or 4"
										"Given beta={}".format(beta))

	X_tmp = X.dot(X.conj().T)
	Y_tmp = Y.dot(Y.conj().T)

	return la.eigvals(X_tmp.dot(np.linalg.inv(X_tmp + Y_tmp))).real

## Jacobi, tridiagonal model
def jacobi_sampler_tridiag(M_1, M_2, N, beta=2):

	# c_odd = c_1, c_2, ..., c_2N-1
	c_odd = np.random.beta(
				0.5*beta*np.arange(M_1, M_1-N, step=-1),
				0.5*beta*np.arange(M_2, M_2-N, step=-1))

	# c_even = c_0, c_2, c_2N-2
	c_even = np.zeros(N)
	c_even[1:] = np.random.beta(
					0.5*beta*np.arange(N-1, 0, step=-1),
					0.5*beta*np.arange(M_1+M_2-N, M_1+M_2-2*N+1,step=-1))

	# xi_odd = xi_2i-1 = (1-c_2i-2) c_2i-1
	xi_odd = (1-c_even)*c_odd
	
	# xi_even = xi_0=0, xi_2, xi_2N-2
	# xi_2i = (1-c_2i-1)*c_2i
	xi_even = np.zeros(N)
	xi_even[1:] = (1-c_odd[:-1])*c_even[1:]
	
	# alpha_i = xi_2i-2 + xi_2i-1
	# alpha_1 = xi_0 + xi_1 = xi_1
	alpha_coef = xi_even + xi_odd
	# beta_i+1 = xi_2i-1 * xi_2i
	beta_coef = xi_odd[:-1] * xi_even[1:]

	return la.eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))

# Wachter law
def wachter_law(x, M_1, M_2, N):
	# M_1, M_2>=N
	# [Dubbs & Edelman, Table 1] https://arxiv.org/pdf/1502.04931.pdf
	a, b = M_1/N, M_2/N

	Lm = ((np.sqrt(a*(a+b-1)) - np.sqrt(b))/(a+b))**2
	Lp = ((np.sqrt(a*(a+b-1)) + np.sqrt(b))/(a+b))**2

	return (a+b)/(2*np.pi) * 1/(x*(1-x)) * np.sqrt(np.maximum((Lp-x)*(x-Lm),0))

	# Lm = ((np.sqrt(M_1*(M_1+M_2-N)) - np.sqrt(M_2*N))/(M_1+M_2))**2
	# Lp = ((np.sqrt(M_1*(M_1+M_2-N)) + np.sqrt(M_2*N))/(M_1+M_2))**2

	# return 1/(2*np.pi) * (M_1+M_2)/N * 1/(x*(1-x)) * np.sqrt(np.maximum((Lp-x)*(x-Lm),0))  

def mu_ref_beta_sampler_tridiag(a, b, beta=2, size=10):

	# beta/2*[N-1, N-2, ..., 1, 0]
	b_2_Ni = 0.5*beta*np.arange(size-1,-1,step=-1)

	# c_odd = c_1, c_2, ..., c_2N-1
	c_odd = np.random.beta(
				b_2_Ni + a,
				b_2_Ni + b)

	# c_even = c_0, c_2, c_2N-2
	c_even = np.zeros(size)
	c_even[1:] = np.random.beta(
					b_2_Ni[:-1],
					b_2_Ni[1:] + a + b)

	# xi_odd = xi_2i-1 = (1-c_2i-2) c_2i-1
	xi_odd = (1-c_even)*c_odd
	
	# xi_even = xi_0=0, xi_2, xi_2N-2
	# xi_2i = (1-c_2i-1)*c_2i
	xi_even = np.zeros(size)
	xi_even[1:] = (1-c_odd[:-1])*c_even[1:]
	
	# alpha_i = xi_2i-2 + xi_2i-1
	# alpha_1 = xi_0 + xi_1 = xi_1
	alpha_coef = xi_even + xi_odd
	# beta_i+1 = xi_2i-1 * xi_2i
	beta_coef = xi_odd[:-1] * xi_even[1:]

	return la.eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))




#########################
### Circular ensemble ###
#########################

# Full matrix model
def circular_sampler_full(N, beta=2, mode="QR"):
	
	if mode == "hermite":
		size_sym_mat = int(N*(N-1)/2)

		if beta==1:#COE
			A = np.random.randn(N, N)

		elif beta==2:#CUE
			A = np.random.randn(N, N) + 1j*np.random.randn(N, N)
			
		elif beta==4:
			X = np.random.randn(N, N) + 1j*np.random.randn(N, N)
			Y = np.random.randn(N, N) + 1j*np.random.randn(N, N)

			A = np.block([  [X,            Y       ],\
											[-Y.conj(),    X.conj()]])

		else:
			raise ValueError("Invalid beta parameter, in 'hermite' mode."
							"Only beta = 1, 2, 4 are available.\n"
							"Given {}".format(beta))

		_, U = la.eigh(A+A.conj().T)

	elif mode == "QR": 
	#[Mezzadri, Sec 5] https://arxiv.org/pdf/math-ph/0609050.pdf	

		if beta == 1: #COE
			A = np.random.randn(N, N)

		elif beta == 2: #CUE
			A = np.random.randn(N, N) + 1j*np.random.randn(N, N)
			A /= np.sqrt(2.0)

		# elif beta==4:
		else:
			raise ValueError("Invalid beta parameter, in 'QR' mode."
							"Only beta = 1, 2 are available.\n"
							"Given {}".format(beta))
			
		#U, _ = np.linalg.qr(A)
		Q, R = np.linalg.qr(A)
		d = np.diagonal(R)
		U = np.multiply(Q, d/np.abs(d), Q)
		
	else:
		raise ValueError("mode = 'hermite' or 'QR'")
	
	return la.eigvals(U)

# Circular, quindiagonal model
def block_diag(arrs):
	# adapted from semi_circleipy.linalg.block_diag
	# https://docs.semi_circleipy.org/doc/semi_circleipy-0.14.0/
	# reference/generated/semi_circleipy.linalg.block_diag.html

	shapes = np.array([a.shape for a in arrs])
	out = np.zeros(np.sum(shapes, axis=0), dtype=np.complex_)

	r, c = 0, 0
	for (rr, cc), blck in zip(shapes, arrs):
		out[r:r + rr, c:c + cc] = blck
		r += rr
		c += cc

	return out

def mu_ref_unif_unit_circle_sampler_quindiag(beta=2, size=10):
	""" TODO: 
	LM and ML have same eigenvalues thus L.dot(M) + M.dot(L) is symmetric
	could use `eigvals_banded`.
	
	adapted from semi_circleipy.linalg.block_diag
	https://docs.semi_circleipy.org/doc/semi_circleipy-0.14.0/
	reference/generated/semi_circleipy.linalg.block_diag.html

	see also [Killip Nenciu, Theorem 1] https://arxiv.org/abs/math/0410034
	"""

	if not ((beta > 0) & isinstance(beta, int)):
		raise ValueError("beta must be positive integer")

	# Xi_-1 = [1]
	xi_1 = np.array([1], ndmin=2, dtype=np.complex_)
	# xi_0,1,...,N-2
	xi_list = np.zeros((size-1, 2, 2), dtype=np.complex_) 
	# xi_N-1 = [alpha_N-1.conj()] i.e. 
	# conjugate of a point uniformly distributed on the unit circle
	vec_N_1 = np.random.randn(2)
	vec_N_1 /= np.linalg.norm(vec_N_1)
	xi_N_1 = np.array([vec_N_1[0]-1j*vec_N_1[1]], ndmin=2, dtype=np.complex_)

	nu_s = beta*np.arange(size-1, 0, step=-1, dtype=int) + 1

	for ind, nu in enumerate(nu_s):

		# Pick a point on the unit sphere S^nu in R^nu+1
		vec = np.random.randn(nu+1)
		vec /= np.linalg.norm(vec)

		alpha = vec[0] + 1j* vec[1]
		rho = np.sqrt(1-np.abs(alpha)**2)

		xi_list[ind,:,:] = [[alpha.conj(),	rho		],
												[rho,					 -alpha]]
		
	# L = diag(xi_0,xi_2,\dots)
	L = block_diag(xi_list[::2,:,:])
	# M = diag(xi_-1,xi_1,\xi_3\dots)
	M = block_diag(xi_list[1::2,:,:])
	M = block_diag([xi_1, M])

	if size%2==1:
		L = block_diag([L, xi_N_1])
	else:
		M = block_diag([M, xi_N_1])

	return la.eigvals(L.dot(M))


###############
### Ginibre ###
###############

def ginibre_sampler_full(N):

	# if beta == 1:
	#     A = np.random.randn(N, N)

	A = np.random.randn(N, N) + 1j*np.random.randn(N, N)

	return la.eigvals(A)/np.sqrt(2.0)