import numpy as np
from numpy.linalg import det, matrix_rank
from numpy.core.umath_tests import inner1d


def det_ST(matrix, S, T=None):
    """ Compute :math:`\det M_{S, T} = \det [M_{ij}]_{i\inS, j\in T}`

    :param matrix:
        Matrix
    :type M:
        array_like

    :param S:
        collection of indices
    :type M:
        1D list, array_like

    :param T:
        collection of indices
    :type M:
        1D list, array_like, default None

    :return:
        - if `T is None` return :math:`\det M_{S, S}`
        - else return :math:`\det M_{S, T}`. If S=T=[], numpy convention = 1.0
    :rtype:
        float
    """

    if T is None:  # det M_SS = det M_S
        return det(matrix[np.ix_(S, S)])

    else:  # det M_ST, numpy deals with det M_[][] = 1.0
        return det(matrix[np.ix_(S, T)])


def is_symmetric(matrix):
    # Cheap test to check symmetry M^T = M

    if matrix is None:
        return None

    indx = np.arange(min(20, matrix.shape[0]))
    if np.allclose(matrix[indx, indx].T, matrix[indx, indx]):
        return matrix
    else:
        raise ValueError('matrix not symmetric')


def is_projection(matrix):
    # Cheap test to check reproducing property: M^2 = M

    if matrix is None:
        return None

    indx = np.arange(min(5, matrix.shape[0]))
    M_i_ = matrix[indx, :]
    M_ii = matrix[indx, indx]

    if np.allclose(inner1d(M_i_, M_i_), M_ii):
        return matrix
    else:
        raise ValueError('matrix not seem to be a projection: M^2 != M')


def is_orthonormal(matrix):
    # Cheap test for checking orthonormality matrix columns: M.T M = I

    if matrix is None:
        return None

    indx = np.arange(np.min([5, matrix.shape[1]]))
    U = matrix[:, indx]

    if np.allclose(U.T.dot(U), np.eye(indx.size)):
        return matrix
    else:
        raise ValueError('matrix does not seem orthonormal: M.T M != I')


def is_equal_to_O_or_1(matrix, tol=1e-8):
    # Check if entries are **all** in {0, 1}, for a given tolerance

    if matrix is None:
        return None

    equal_0 = np.abs(matrix) <= tol
    equal_1 = np.abs(1 - matrix) <= tol
    equal_0_or_1 = equal_0 ^ equal_1  # ^ = xor

    if np.all(equal_0_or_1):
        return matrix
    else:
        raise ValueError('matrix with entries not all in {0,1}')


def is_in_01(matrix, tol=1e-8):
    # Check if entries are **all** in [0, 1], for a given tolerance

    if matrix is None:
        return None
    elif np.all((-tol <= matrix) & (matrix <= 1.0 + tol)):
        return matrix
    else:
        raise ValueError('matrix with entries not all in [0,1]')


def is_geq_0(matrix, tol=1e-8):
    # Check if entries are **all** >= 0, for a given tolerance

    if matrix is None:
        return None
    elif np.all(matrix >= -tol):
        return matrix
    else:
        raise ValueError('matrix with entries not all >= 0')


def is_full_row_rank(matrix):
    # Check rank(M) = #rows

    if matrix is None:
        return None

    d, N = matrix.shape
    err_print = 'matrix (size = dxN) is not full row rank'

    if d > N:
        raise ValueError(err_print + 'd(={}) > N(={})'.format(d, N))
    else:
        rank = matrix_rank(matrix)
        if rank == d:
            return matrix
        else:
            raise ValueError(err_print + 'd(={}) != rank(={})'.format(d, rank))
