# coding: utf-8

import numpy as np
from scipy.linalg import eig, eigh, eigvals, eigvalsh, eigvalsh_tridiagonal, block_diag

##########################
### Full matrix models ###
##########################

#############
### Real line
#############

# Semi-circle law
def sc_law(x, R=1):
    return 2/(np.pi*R**2) * np.sqrt(R**2 - x**2)

# Marcenko Pastur law
def MP_law(x, M, N, sigma=1.0):
	# M>=N
    c = N/M
    Lm, Lp = (sigma*(1-np.sqrt(c)))**2, (sigma*(1+np.sqrt(c)))**2

    return np.sqrt(np.maximum((Lp - x)*(x-Lm),0)) / (2*np.pi*c*sigma**2*x) 

###### Hermite
def beta_12_Hermite(N, beta=2):
    
    # if beta==1:
    #     A = np.random.randn(N,N)

    # elif beta==2:
    #     A = np.random.randn(N,N) + 1j*np.random.randn(N,N)

    if beta==1:
        A = np.zeros((N,N))
        random_var = np.random.randn(int(N*(N-1)/2))

        A[np.tril_indices_from(A, k=-1)] = random_var
        A[np.triu_indices_from(A, k=+1)] = random_var
        A[np.diag_indices_from(A)] = np.random.randn(N)

    elif beta==2:
        A = np.zeros((N,N), dtype=np.complex_)
        random_var = np.random.randn(int(N*(N-1)/2)) + 1j*np.random.randn(int(N*(N-1)/2))

        A[np.tril_indices_from(A, k=-1)] = random_var
        A[np.triu_indices_from(A, k=+1)] = random_var.conj()
        A[np.diag_indices_from(A)] = np.random.randn(N)
        
    elif beta==4:
        X = np.random.randn(N,N) + 1j*np.random.randn(N,N)
        Y = np.random.randn(N,N) + 1j*np.random.randn(N,N)
        A = np.block([\
            [X,           Y         ],\
            [-Y.conj(), X.conj()]\
            ])
        
    else:
        raise ValueError("beta coefficient must be equal to 1 or 2")

    return eigvalsh(A)

###### Laguerre
def beta_12_Laguerre(M, N, beta=2):
    
    if beta==1:
        A = np.random.randn(N,M)

    elif beta==2:
        A = np.random.randn(N,M) + 1j*np.random.randn(N,M)

    elif beta==4:
        X = np.random.randn(N,M) + 1j*np.random.randn(N,M)
        Y = np.random.randn(N,M) + 1j*np.random.randn(N,M)
        A = np.block([\
            [X,           Y     ],\
            [-Y.conj(), X.conj()]\
            ])
        
    else:
        raise ValueError("beta coefficient must be equal to 1 or 2")

    return eigvalsh(A.dot(A.conj().T))


###### Jacobi
def beta_12_Jacobi(M_1, M_2, N, beta=2):
    
    if beta==1:
        X = np.random.randn(N,M_1)
        Y = np.random.randn(N,M_2)

        X_tmp = X.dot(X.T)
        Y_tmp = Y.dot(Y.T)

    elif beta==2:
        X = np.random.randn(N,M_1) + 1j*np.random.randn(N,M_1)
        Y = np.random.randn(N,M_2) + 1j*np.random.randn(N,M_2)

        X_tmp = X.dot(X.conj().T)
        Y_tmp = Y.dot(Y.conj().T)

    elif beta==4:
        X_1 = np.random.randn(N,M_1) + 1j*np.random.randn(N,M_1)
        X_2 = np.random.randn(N,M_1) + 1j*np.random.randn(N,M_1)

        Y_1 = np.random.randn(N,M_2) + 1j*np.random.randn(N,M_2)
        Y_2 = np.random.randn(N,M_2) + 1j*np.random.randn(N,M_2)

        X = np.block([\
            [X_1,           X_2         ],\
            [-X_2.conj(), X_1.conj()]\
            ])
        Y = np.block([\
            [Y_1,           Y_2         ],\
            [-Y_2.conj(), Y_1.conj()]\
            ])

        X_tmp = X.dot(X.conj().T)
        Y_tmp = Y.dot(Y.conj().T)

    else:
        raise ValueError("beta coefficient must be equal to 1 or 2")

    return eigvalsh(X_tmp.dot(np.linalg.inv(X_tmp + Y_tmp)))


###############
### Disk/Circle
###############
def Ginibre(N=10):

    A = np.random.randn(N,N) + 1j*np.random.randn(N,N)

    return eigvals(A)/np.sqrt(2)

def beta_12_Circular(N=10, beta=2, gen_from="Ginibre"):
    
    if gen_from == "Ginibre":
        # https://arxiv.org/pdf/math-ph/0609050.pdf
        
        if beta == 1: #COE
            A = np.random.randn(N,N)
            
        elif beta == 2: #CUE
            A = (np.random.randn(N,N) + 1j*np.random.randn(N,N))/np.sqrt(2.0)

        #U, _ = np.linalg.qr(A)
        Q, R = np.linalg.qr(A)
        d = np.diagonal(R)
        U = np.multiply(Q, d/np.abs(d), Q)
        
    elif gen_from == "Hermite":

        if beta == 1:#COE
            A = np.zeros((N,N))
            random_var = np.random.randn(int(N*(N-1)/2))
            
            A[np.tril_indices_from(A, k=-1)] = random_var
            A[np.triu_indices_from(A, k=+1)] = random_var
            A[np.diag_indices_from(A)] = np.random.randn(N)
            
        elif beta == 2:#CUE
            A = np.zeros((N,N), dtype=np.complex_)
            random_var = np.random.randn(int(N*(N-1)/2)) + 1j*np.random.randn(int(N*(N-1)/2))

            A[np.tril_indices_from(A, k=-1)] = random_var
            A[np.triu_indices_from(A, k=+1)] = random_var.conj()
            A[np.diag_indices_from(A)] = np.random.randn(N)

        _, U = eigh(A)
        
    else:
        raise ValueError("gen_from = 'Ginibre' or 'Hermite'")
    
    return eigvals(U)










##########################
### Tridiagonal models ###
##########################

###### Hermite
def beta_triadiag_Hermite(N, beta=2):

    alpha_coef = np.random.randn(N)
    beta_coef = 0.5*np.random.chisquare(beta*np.arange(N-1,0,step=-1))

    return eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))

###### Laguerre
def beta_tridiag_Laguerre(M, N, beta=2):

    # xi_odd = xi_1, ... , xi_2i-1
    xi_odd = np.random.chisquare(beta*np.arange(M,M-N,step=-1)) # odd

    # xi_even = xi_0=0, xi_2, ... ,xi_2N-2
    xi_even = np.zeros(N)
    xi_even[1:] = np.random.chisquare(beta*np.arange(N-1,0,step=-1)) # even

    # alpha_i = xi_2i-2 + xi_2i-1
    # alpha_1 = xi_0 + xi_1 = xi_1
    alpha_coef = xi_even + xi_odd
    # beta_i+1 = xi_2i-1 * xi_2i
    beta_coef = xi_odd[:-1] * xi_even[1:]

    eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))
    
    return eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))

###### Jacobi
def beta_tridiag_Jacobi(a, b, N, beta=2):

    # beta/2*[N-1, N-2, ..., 1, 0]
    b_2_Ni = beta/2*np.arange(N-1,-1,step=-1)

    # c_odd = c_1, c_2, ..., c_2N-1
    c_odd = np.random.beta(
                b_2_Ni + a,
                b_2_Ni + b)

    # c_even = c_0, c_2, c_2N-2
    c_even = np.zeros(N)
    c_even[1:] = np.random.beta(
                    b_2_Ni[:-1],
                    b_2_Ni[1:] + a + b)

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

    return eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))




###########################
### Quindiagonal models ###
###########################

def beta_quindiag_Circular(N, beta=2):
    # Killip Nenciu https://arxiv.org/abs/math/0410034

    if not ((beta > 0) & isinstance(beta,int)):
        raise ValueError("beta must be positive integer")

    # Xi_-1 = [1]
    Xi_1 = np.array([1], ndmin=2, dtype=np.complex_)
    # Xi_0,1,...,N-2
    Xi_list = np.zeros((N-1,2,2), dtype=np.complex_) 
    # Xi_N-1 = [alpha_N-1.conj()]
    vec_N_1 = np.random.randn(2)
    vec_N_1 /= np.linalg.norm(vec_N_1)
    Xi_N_1 = np.array([vec_N_1[0]-1j*vec_N_1[1]], ndmin=2, dtype=np.complex_)

    nu_s = beta*np.arange(N-1, 0, step=-1, dtype=int) + 1

    for ind, nu in enumerate(nu_s):

        vec = np.random.randn(nu+1)
        vec /= np.linalg.norm(vec)

        alpha = vec[0] + 1j* vec[1]
        rho = np.sqrt(1-np.abs(alpha)**2)

        Xi_list[ind,:,:] = [[alpha.conj(), rho],
                            [rho         , -alpha]]
        
    # L = diag(Xi_0,Xi_2,\dots)
    L = block_diag(Xi_list[::2,:,:])
    # M = diag(Xi_-1,Xi_1,\Xi_3\dots)
    M = block_diag(Xi_list[1::2,:,:])
    M = block_diag([Xi_1, M])

    if N%2==1:
        L = block_diag([L, Xi_N_1])
    else:
        M = block_diag([M, Xi_N_1])
        
    return eigvals(L.dot(M))