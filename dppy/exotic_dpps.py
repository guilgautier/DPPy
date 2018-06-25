# coding: utf-8

import numpy as np

# Carries process
def carries_process(N, b=10):

		A = np.random.randint(0,b-1,N)
		B = np.mod(np.cumsum(A),b)

		X = np.zeros((N,), dtype=int)
		X[1:] = B[1:]<B[:-1]

		return X		

### Uniform permutation

def unif_permutation(N):

		N=10
		tmp = np.arange(N)
		for i in range(N-1,1,-1):
				j = np.random.randint(0, i+1)
				tmp[j], tmp[i] = tmp[i], tmp[j]

		return tmp

#sigma = [2,2,4,3,8,7,3,2]

### RSK
def RSK(sigma):
	"""Perform Robinson-Schensted-Knuth correspondence on a sequence of reals, e.g. a permutation
	"""	

	P, Q = [], [] # Insertion/Recording tableaux

	# Enumerate the sequence
	for it, x in enumerate(sigma): 
			
			# Iterate along the rows of the tableau P
			# to find a place for the bouncing x and
			# record the position where it is inserted
			for row_P, row_Q in zip(P,Q): 
					
					# In case x finds a place at the end of a row of P
					# Add it and record its position to the row of Q
					if x >= row_P[-1]:
							row_P.append(x); row_Q.append(it+1)
							break

					# Otherwise find the place where x must be added
					# to keep the row ordered
					ind_insert = bisect_right(row_P, x)
					# Swap x with 
					x, row_P[ind_insert] = row_P[ind_insert], x
					
			# In case the bouncing x cannot find a place at the end of a row of P
			# Create a new row and save 
			else:
					P.append([x]); Q.append([it+1])

	return P, Q, len(P[0])



def r_hahn(N, a=0, b=0):
		
		if (N<=0) | (a<-1) | (b<-1):
				raise ValueError("Arguments(s) out of range: N>0, a,b>-1")
		
		alpha_beta_coef = np.zeros((N+1,2))
		ind_0_Np1 = np.arange(1,N+2)
		alpha_beta_coef[0,1] = np.prod(1+(a+b+1)/ind_0_Np1)
		
		if (a+b)==0:
				aux = ind_0_Np1
				alpha_beta_coef[:,0] = ((2*aux+a+b-1)*N+(b-a)*aux+a)/(2*(2*aux-1))
				aux = ind_0_Np1[:-1]
				alpha_beta_coef[1:,1] = .25*((N+1)**2)*(1+a/aux)*(1+b/aux)*(1-(aux/(N+1))**2) \
											 /(4-(1/aux)**2);
		elif (a+b+1)==0:
				aux = ind_0_Np1
				alpha_beta_coef[:,0] = ((2*(aux-1)**2+b)*N+(2*b+1)*(aux-1)**2)/(4*(aux-1)**2-1)
				
				aux = ind_0_Np1[:-1]
				alpha_beta_coef[1:,1] = .25*((N+1)**2)*(1+a/aux)*(1+b/aux)*(1-aux/(N+1))* \
												(1+(aux-1)/(N+1))/(4-(1/aux)**2);
		else:
				aux = ind_0_Np1
				alpha_beta_coef[:,0]=((aux+a+b)*(aux+a)*(N-aux+1)/(2*aux+a+b)\
									 +(aux-1)*(aux+b-1)*(N+aux+a+b)/(2*aux+a+b-2))\
								/(2*aux+a+b-1)
				
				aux = ind_0_Np1[:-1]
				alpha_beta_coef[1:,1]=((N+1)**2)*(1+a/aux)*(1+b/aux)*(1+(a+b)/aux)* \
											 (1-aux/(N+1))*(1+(aux+a+b)/(N+1))\
									 /(((2+(a+b)/aux)**2)*((2+(a+b)/aux)**2-(1/aux)**2))
								
		return alpha_beta_coef

#### Kravchuk
M, p = 10, 0.5
mu = [binom(M,p).pmf(k) for k in range(M+1)]

def alpha_coef(M, N, p): 

		return (1-2*p)*np.arange(N) + p*M

def beta_coef(M, N, p):
		# beta_0=1.0 b_k = p*(1-p)*k*(N-k+1)
		tmp = np.arange(N)
		
		coef = p*(1-p)*tmp*(M-tmp+1)
		coef[0]=1.0
		
		return coef

######################
### Spanning trees ###
######################

# Aldous

# Wilson

# Gaudillere