from string import ascii_lowercase
import numpy as np
from numpy.linalg import det, matrix_rank


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    .. seealso::

        `Scikit learn source code <https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/utils/validation.py#L763>`_
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def inner1d(arr1, arr2=None, axis=0):
    """ Efficient equivalent to ``(arr1**2).sum(axis)`` or ``(arr1*arr2).sum(axis)`` for ``arr1.shape == arr2.shape``.
    Expected to be used with arrays of same shape and mainly with 1D or 2D arrays but works for upto 26D arrays...

    If ``arr1.shape == arr2.shape``, then ``inner1d(arr1, arr2, arr1.ndim)`` replaces ``numpy.core.umath_tests.inner1d(arr1, arr2)``

    Examples:
    - To compute square norm of vector i.e. 1D array
    inner1d(arr) = np.einsum('i,i->', arr, arr)
                 = np.dot(arr, arr)
                 = (arr**2).sum()

    - To compute vector inner product i.e. 2 1D arrays
    inner1d(arr1, arr2) = np.einsum('i,i->', arr1, arr2)
                        = np.dot(arr1, arr2)
                        = (arr1*arr2).sum()

    - To compute square norm of cols/rows of 2D array
    inner1d(arr, axis=0/1)
        = np.einsum('ij,ij->j/i', arr, arr)
        = (arr**2).sum(axis=0/1)

    - To compute inner product between cols/rows of 2 arrays
    inner1d(arr1, arr2, axis=0/1)
        = np.einsum('ij,ij->j/i', arr1, arr2)
        = (arr1*arr2).sum(axis=0/1)
    """

    # if (arr2 is not None) and (arr1.shape != arr2.shape):
    #     raise ValueError('...with shapes {} {}'
    #                      .format(arr1.shape, arr2.shape))

    ndim = arr1.ndim
    sym = ascii_lowercase[:ndim]
    subscripts = sym + ',' + sym + '->' + sym.replace(sym[axis], '')

    if arr2 is None:
        return np.einsum(subscripts, arr1, arr1)
    else:
        return np.einsum(subscripts, arr1, arr2)


def det_ST(array, S, T=None):
    """ Compute :math:`\\det M_{S, T} = \\det [M_{ij}]_{i\\inS, j\\in T}`

    :param array:
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
        - if `T is None` return :math:`\\det M_{S, S}`
        - else return :math:`\\det M_{S, T}`. If S=T=[], numpy convention = 1.0
    :rtype:
        float
    """

    if T is None:  # det M_SS = det M_S
        return det(array[np.ix_(S, S)])

    else:  # det M_ST, numpy deals with det M_[][] = 1.0
        return det(array[np.ix_(S, T)])


def is_symmetric(array):
    # Cheap test to check symmetry M^T = M

    if array is None:
        return None

    indx = np.arange(min(20, array.shape[0]))
    M = array[np.ix_(indx, indx)]
    if np.allclose(M.T, M):
        return array
    else:
        raise ValueError('array not symmetric: M.T != M')


def is_projection(array):
    # Cheap test to check reproducing property: M^2 = M

    if array is None:
        return None

    indx = np.arange(min(5, array.shape[0]))
    M_j = array[:, indx]
    Mjj = array[indx, indx]

    if np.allclose(inner1d(M_j), Mjj):
        return array
    else:
        raise ValueError('array not seem to be a projection: M^2 != M')


def is_orthonormal(array):
    # Cheap test for checking orthonormality array columns: M.T M = I

    if array is None:
        return None

    indx = np.arange(np.min([5, array.shape[1]]))
    U = array[:, indx]

    if np.allclose(U.T.dot(U), np.eye(indx.size)):
        return array
    else:
        raise ValueError('array does not seem orthonormal: M.T M != I')


def is_equal_to_O_or_1(array, tol=1e-8):
    """Check if entries are **all** in :math:`\\{0, 1\\}`, for a given tolerance"""

    if array is None:
        return None

    equal_0 = np.abs(array) <= tol
    equal_1 = np.abs(1 - array) <= tol
    equal_0_or_1 = equal_0 ^ equal_1  # ^ = xor

    if np.all(equal_0_or_1):
        return array
    else:
        raise ValueError('array with entries not all in {0,1}')


def is_in_01(array, tol=1e-8):
    """Check if entries are **all** in :math:`[0, 1]`, for a given tolerance"""

    if array is None:
        return None
    elif np.all((-tol <= array) & (array <= 1.0 + tol)):
        return array
    else:
        raise ValueError('array with entries not all in [0,1]')


def is_geq_0(array, tol=1e-8):
    """Check if entries are **all** :math:`\\geq0`, for a given tolerance"""

    if array is None:
        return None
    elif np.all(array >= -tol):
        return array
    else:
        raise ValueError('array with entries not all >= 0')


def is_full_row_rank(array):
    # Check rank(M) = #rows

    if array is None:
        return None

    d, N = array.shape
    err_print = 'array (size = dxN) is not full row rank'

    if d > N:
        raise ValueError(err_print + 'd(={}) > N(={})'.format(d, N))
    else:
        rank = matrix_rank(array)
        if rank == d:
            return array
        else:
            raise ValueError(err_print + 'd(={}) != rank(={})'.format(d, rank))
