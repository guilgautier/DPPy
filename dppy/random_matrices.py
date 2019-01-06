# coding: utf-8

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

###########
# Hermite #
###########


# Hermite, full matrix model
def hermite_sampler_full(N, beta=2):

    # size_sym_mat = int(N * (N-1) / 2)

    if beta == 1:
        A = np.random.randn(N, N)

    elif beta == 2:
        A = np.random.randn(N, N) + 1j * np.random.randn(N, N)

    elif beta == 4:
        X = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        Y = np.random.randn(N, N) + 1j * np.random.randn(N, N)

        A = np.block([[X, Y], [-Y.conj(), X.conj()]])

    else:
        err_print = ('`beta` parameter must be 1, 2 or 4.\
                     Given: {}'.format(beta))
        raise ValueError(err_print)

    # return la.eigvalsh(A+A.conj().T)
    return la.eigvalsh(A + A.conj().T) / np.sqrt(2.0)


# Hermite tridiag
def hermite_sampler_tridiag(N, beta=2):
    """
    .. seealso::

        :cite:`DuEd02` II-C
    """

    if not (beta > 0):
        raise ValueError('`beta` must be positive. Given: {}'.format(beta))

    alpha_coef = np.sqrt(2) * np.random.randn(N)
    beta_coef = np.random.chisquare(beta * np.arange(N - 1, 0, step=-1))

    return la.eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))


# Semi-circle law
def semi_circle_law(x, R=2.0):
    # :cite:`DuEd15` Table 1
    # https://en.wikipedia.org/wiki/Wigner_semicircle_distribution
    return 2 / (np.pi * R**2) * np.sqrt(R**2 - x**2)


# mu_ref == normal
def mu_ref_normal_sampler_tridiag(loc=0.0, scale=1.0, beta=2, size=10):
    """
    .. seealso::

        :cite:`DuEd02` II-C
    """

    if not (beta > 0):
        raise ValueError('`beta` must be positive. Given: {}'.format(beta))

    # beta/2*[N-1, N-2, ..., 1]
    b_2_Ni = 0.5 * beta * np.arange(size - 1, 0, step=-1)

    alpha_coef = np.random.normal(loc=loc, scale=scale, size=size)
    beta_coef = np.random.gamma(shape=b_2_Ni, scale=scale**2)

    return la.eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))

############
# Laguerre #
############


# Laguerre, full matrix model
def laguerre_sampler_full(M, N, beta=2):

    if beta == 1:
        A = np.random.randn(N, M)

    elif beta == 2:
        A = np.random.randn(N, M) + 1j * np.random.randn(N, M)

    elif beta == 4:
        X = np.random.randn(N, M) + 1j * np.random.randn(N, M)
        Y = np.random.randn(N, M) + 1j * np.random.randn(N, M)
        A = np.block([[X, Y], [-Y.conj(), X.conj()]])

    else:
        err_print = ('`beta` parameter must be 1, 2 or 4.\
                     Given: {}'.format(beta))
        raise ValueError(err_print)

    return la.eigvalsh(A.dot(A.conj().T))


# Laguerre, tridiagonal model
def laguerre_sampler_tridiag(M, N, beta=2):
    """
    .. seealso::

        :cite:`DuEd02` III-B
    """

    if not (beta > 0):
        raise ValueError('`beta` must be positive. Given: {}'.format(beta))
    # M=>N

    # xi_odd = xi_1, ... , xi_2N-1
    xi_odd = np.random.chisquare(beta * np.arange(M, M - N, step=-1))

    # xi_even = xi_0=0, xi_2, ... ,xi_2N-2
    xi_even = np.zeros(N)
    xi_even[1:] = np.random.chisquare(beta * np.arange(N - 1, 0, step=-1))

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
    c = N / M
    Lm, Lp = (sigma * (1 - np.sqrt(c)))**2, (sigma * (1 + np.sqrt(c)))**2

    return np.sqrt(np.maximum((Lp-x)*(x-Lm),0)) / (c*x) / (2*np.pi*sigma**2) 


# mu_ref == Gamma
def mu_ref_gamma_sampler_tridiag(shape=1.0, scale=1.0, beta=2, size=10):
    """
    .. seealso::

        :cite:`DuEd02` III-B
    """

    if not (beta > 0):
        raise ValueError('`beta` must be positive. Given: {}'.format(beta))

    # beta/2*[N-1, N-2, ..., 1, 0]
    b_2_Ni = 0.5 * beta * np.arange(size - 1, -1, step=-1)

    # xi_odd = xi_1, ... , xi_2N-1
    xi_odd = np.random.gamma(shape=b_2_Ni + shape, scale=scale)  # odd

    # xi_even = xi_0=0, xi_2, ... ,xi_2N-2
    xi_even = np.zeros(size)
    xi_even[1:] = np.random.gamma(shape=b_2_Ni[:-1], scale=scale)  # even

    # alpha_i = xi_2i-2 + xi_2i-1
    # alpha_1 = xi_0 + xi_1 = xi_1
    alpha_coef = xi_even + xi_odd
    # beta_i+1 = xi_2i-1 * xi_2i
    beta_coef = xi_odd[:-1] * xi_even[1:]

    return la.eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))

##########
# Jacobi #
##########


# Jacobi, full matrix model
def jacobi_sampler_full(M_1, M_2, N, beta=2):

    if beta == 1:
        X = np.random.randn(N, M_1)
        Y = np.random.randn(N, M_2)

    elif beta == 2:
        X = np.random.randn(N, M_1) + 1j * np.random.randn(N, M_1)
        Y = np.random.randn(N, M_2) + 1j * np.random.randn(N, M_2)

    elif beta == 4:
        X_1 = np.random.randn(N, M_1) + 1j * np.random.randn(N, M_1)
        X_2 = np.random.randn(N, M_1) + 1j * np.random.randn(N, M_1)

        Y_1 = np.random.randn(N, M_2) + 1j * np.random.randn(N, M_2)
        Y_2 = np.random.randn(N, M_2) + 1j * np.random.randn(N, M_2)

        X = np.block([[X_1, X_2], [-X_2.conj(), X_1.conj()]])
        Y = np.block([[Y_1, Y_2], [-Y_2.conj(), Y_1.conj()]])

    else:
        err_print = ('`beta` parameter must be 1, 2 or 4.\
                     Given: {}'.format(beta))
        raise ValueError(err_print)

    X_tmp = X.dot(X.conj().T)
    Y_tmp = Y.dot(Y.conj().T)

    return la.eigvals(X_tmp.dot(la.inv(X_tmp + Y_tmp))).real


# Jacobi, tridiagonal model
def jacobi_sampler_tridiag(M_1, M_2, N, beta=2):
    """
    .. seealso::

        :cite:`KiNe04` Theorem 2
    """

    if not (beta > 0):
        raise ValueError('`beta` must be positive. Given: {}'.format(beta))

    # c_odd = c_1, c_2, ..., c_2N-1
    c_odd = np.random.beta(a=0.5 * beta * np.arange(M_1, M_1 - N, step=-1),
                           b=0.5 * beta * np.arange(M_2, M_2 - N, step=-1))

    # c_even = c_0, c_2, c_2N-2
    c_even = np.zeros(N)
    c_even[1:] = np.random.beta(a=0.5 * beta * np.arange(N - 1, 0, step=-1),
                                b=0.5 * beta * np.arange(M_1 + M_2 - N,
                                                         M_1 + M_2 - 2 * N + 1,
                                                         step=-1))

    # xi_odd = xi_2i-1 = (1-c_2i-2) c_2i-1
    xi_odd = (1 - c_even) * c_odd

    # xi_even = xi_0=0, xi_2, xi_2N-2
    # xi_2i = (1-c_2i-1)*c_2i
    xi_even = np.zeros(N)
    xi_even[1:] = (1 - c_odd[:-1]) * c_even[1:]

    # alpha_i = xi_2i-2 + xi_2i-1
    # alpha_1 = xi_0 + xi_1 = xi_1
    alpha_coef = xi_even + xi_odd
    # beta_i+1 = xi_2i-1 * xi_2i
    beta_coef = xi_odd[:-1] * xi_even[1:]

    return la.eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))


# Wachter law
def wachter_law(x, M_1, M_2, N):
    # M_1, M_2>=N
    # :cite:`DuEd15` Table 1
    a, b = M_1 / N, M_2 / N

    Lm = ((np.sqrt(a * (a + b - 1)) - np.sqrt(b)) / (a + b))**2
    Lp = ((np.sqrt(a * (a + b - 1)) + np.sqrt(b)) / (a + b))**2

    return (a+b)/(2*np.pi) * 1/(x*(1-x)) * np.sqrt(np.maximum((Lp-x)*(x-Lm),0))


def mu_ref_beta_sampler_tridiag(a, b, beta=2, size=10):

    """
    .. seealso::

        :cite:`KiNe04` Theorem 2
    """

    if not (beta > 0):
        raise ValueError('`beta` must be positive. Given: {}'.format(beta))

    # beta/2*[N-1, N-2, ..., 1, 0]
    b_2_Ni = 0.5 * beta * np.arange(size - 1, -1, step=-1)

    # c_odd = c_1, c_2, ..., c_2N-1
    c_odd = np.random.beta(b_2_Ni + a, b_2_Ni + b)

    # c_even = c_0, c_2, c_2N-2
    c_even = np.zeros(size)
    c_even[1:] = np.random.beta(b_2_Ni[:-1], b_2_Ni[1:] + a + b)

    # xi_odd = xi_2i-1 = (1-c_2i-2) c_2i-1
    xi_odd = (1 - c_even) * c_odd

    # xi_even = xi_0=0, xi_2, xi_2N-2
    # xi_2i = (1-c_2i-1)*c_2i
    xi_even = np.zeros(size)
    xi_even[1:] = (1 - c_odd[:-1]) * c_even[1:]

    # alpha_i = xi_2i-2 + xi_2i-1
    # alpha_1 = xi_0 + xi_1 = xi_1
    alpha_coef = xi_even + xi_odd
    # beta_i+1 = xi_2i-1 * xi_2i
    beta_coef = xi_odd[:-1] * xi_even[1:]

    return la.eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))

#####################
# Circular ensemble #
#####################


# Full matrix model
def circular_sampler_full(N, beta=2, haar_mode='QR'):
    """
    .. seealso::

        :cite:`Mez06` Section 5
    """

    if haar_mode == 'Hermite':
        # size_sym_mat = int(N*(N-1)/2)

        if beta == 1:  # COE
            A = np.random.randn(N, N)

        elif beta == 2:  # CUE
            A = np.random.randn(N, N) + 1j * np.random.randn(N, N)

        elif beta == 4:
            X = np.random.randn(N, N) + 1j * np.random.randn(N, N)
            Y = np.random.randn(N, N) + 1j * np.random.randn(N, N)

            A = np.block([[X, Y], [-Y.conj(), X.conj()]])

        else:
            err_print = ('For `haar_mode="hermite"`, `beta` = 1, 2 or 4.\
                         Given: {}'.format(beta))
            raise ValueError(err_print)

        _, U = la.eigh(A + A.conj().T)

    elif haar_mode == 'QR':

        if beta == 1:  # COE
            A = np.random.randn(N, N)

        elif beta == 2:  # CUE
            A = np.random.randn(N, N) + 1j * np.random.randn(N, N)

        # elif beta==4:
        else:
            err_print = ('With `haar_mode="QR", `beta` = 1 or 2.\
                         Given: {}'.format(beta))
            raise ValueError(err_print)

        # U, _ = la.qr(A)
        Q, R = la.qr(A)
        d = np.diagonal(R)
        U = np.multiply(Q, d / np.abs(d), Q)

    else:
        err_print = ('Invalid `haar_mode`.\
                     Choose from `haar_mode="Hermite" or "QR".\
                     Given: {}'.format(haar_mode))
        raise ValueError(err_print)

    return la.eigvals(U)


# Circular, quindiagonal model
def mu_ref_unif_unit_circle_sampler_quindiag(beta=2, size=10):
    """
    .. see also::

        :cite:`KiNe04` Theorem 1
    """

    if not ((beta > 0) & isinstance(beta, int)):
        raise ValueError('`beta` must be positive integer.\
                         Given: {}'.format(beta))

    alpha = np.zeros(size, dtype=np.complex_)

    # nu = 1 + beta*(N-1, N-2, ..., 0)
    for i, nu in enumerate(1 + beta * np.arange(size - 1, -1, step=-1)):
        gauss_vec = np.random.randn(nu + 1)
        alpha[i] = (gauss_vec[0] + 1j * gauss_vec[1]) / la.norm(gauss_vec)

    rho = np.sqrt(1 - np.abs(alpha[:-1])**2)

    xi = np.zeros((size - 1, 2, 2), dtype=np.complex_)  # xi[0,..,N-1]
    xi[:, 0, 0], xi[:, 0, 1] = alpha[:-1].conj(), rho
    xi[:, 1, 0], xi[:, 1, 1] = rho, -alpha[:-1]

    # xi[N-1] = alpha[N-1].conj()
    # L = diag(xi_0, xi_2, ...)
    # M = diag(1, xi_1, x_3, ...)
    if size % 2 == 0:  # even
        L = sp.block_diag(xi[::2, :, :],
                          dtype=np.complex_)
        M = sp.block_diag([1.0, *xi[1::2, :, :], alpha[-1].conj()],
                          dtype=np.complex_)
    else:  # odd
        L = sp.block_diag([*xi[::2, :, :], alpha[-1].conj()],
                          dtype=np.complex_)
        M = sp.block_diag([1.0, *xi[1::2, :, :]],
                          dtype=np.complex_)

    return la.eigvals(L.dot(M).toarray())


###########
# Ginibre #
###########


def ginibre_sampler_full(N):

    # if beta == 1:
    #     A = np.random.randn(N, N)

    A = np.random.randn(N, N) + 1j * np.random.randn(N, N)

    return la.eigvals(A) / np.sqrt(2.0)
