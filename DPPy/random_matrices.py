# coding: utf-8

import numpy as np
from scipy.linalg import eig, eigh, eigvals, eigvalsh, eigvalsh_tridiagonal

##########################
### Full matrix models ###
##########################

#############
### Real line
#############

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
        A = np.zeros((N,N), dtype=complex_)
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


# Semi-circle law
def sc_law(x, R=1):
    return 2/(pi*R**2) * np.sqrt(R**2 - x**2)

# Marcenko Pastur law
def MP_law(x, M, N, sigma=1.0):
    c = N/M
    Lm, Lp = (sigma*(1-np.sqrt(c)))**2, (sigma*(1+np.sqrt(c)))**2

    return np.sqrt(np.maximum((Lp - x)*(x-Lm),0)) / (2*np.pi*c*sigma**2*x) 














###############
### Disk/Circle
###############
def Ginibre(N=10):

    A = np.random.randn(N,N) + 1j*np.random.randn(N,N)

    return eigvals(A)/np.sqrt(2)

def beta_12_Circular(N=10, beta=2, gen_from="Ginibre"):
    
    if gen_from == "Ginibre":
        # https://arxiv.org/pdf/math-ph/0609050.pdf
        # From Ginibre
        
        if beta == 1: #COE
            A = np.random.randn(N,N)
            
        elif beta == 2: #CUE
            A = (np.random.randn(N,N) + 1j*np.random.randn(N,N))/sqrt(2.0)

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
            A = np.zeros((N,N), dtype=complex_)
            random_var = np.random.randn(int(N*(N-1)/2)) + 1j*np.random.randn(int(N*(N-1)/2))

            A[np.tril_indices_from(A, k=-1)] = random_var
            A[np.triu_indices_from(A, k=+1)] = random_var.conj()
            A[np.diag_indices_from(A)] = np.random.randn(N)

        _, U = eigh(A)
        
    else:
        raise ValueError("gen_from = 'Ginibre' or 'Hermite'")
    
    return eigvals(U)