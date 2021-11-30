# coding: utf8
""" Implementation of finite DPP MCMC samplers:

- `add_exchange_delete_sampler`
- `add_delete_sampler`
- `exchange_sampler`
- `zonotope_sampler`

.. seealso:

    `Documentation on ReadTheDocs <https://dppy.readthedocs.io/en/latest/finite_dpps/mcmc_sampling.html>`_
"""

import time

import numpy as np
import scipy.linalg as la

from dppy.utils import check_random_state, det_ST


############################################
# Approximate samplers for projection DPPs #
############################################
def dpp_sampler_mcmc(kernel, mode="AED", **params):
    """Interface function with initializations and samplers for MCMC schemes.

    .. seealso::

        - :ref:`finite_dpps_mcmc_sampling_add_exchange_delete`
        - :func:`add_exchange_delete_sampler <add_exchange_delete_sampler>`
        - :func:`initialize_AED_sampler <initialize_AED_sampler>`
        - :func:`add_delete_sampler <add_delete_sampler>`
        - :func:`exchange_sampler <exchange_sampler>`
        - :func:`initialize_AD_and_E_sampler <initialize_AD_and_E_sampler>`
    """

    rng = check_random_state(params.get("random_state", None))

    s_init = params.get("s_init", None)
    nb_iter = params.get("nb_iter", 10)
    T_max = params.get("T_max", None)
    size = params.get("size", None)  # = Tr(K) for projection correlation K

    if mode == "AED":  # Add-Exchange-Delete S'=S+t, S-t+u, S-t
        if s_init is None:
            s_init = initialize_AED_sampler(kernel, random_state=rng)
        sampl = add_exchange_delete_sampler(
            kernel, s_init, nb_iter, T_max, random_state=rng
        )

    elif mode == "AD":  # Add-Delete S'=S+t, S-t
        if s_init is None:
            s_init = initialize_AD_and_E_sampler(kernel, random_state=rng)
        sampl = add_delete_sampler(kernel, s_init, nb_iter, T_max, random_state=rng)

    elif mode == "E":  # Exchange S'=S-t+u
        if s_init is None:
            s_init = initialize_AD_and_E_sampler(kernel, size, random_state=rng)
        sampl = basis_exchange_sampler(kernel, s_init, nb_iter, T_max, random_state=rng)

    return sampl


def initialize_AED_sampler(kernel, random_state=None):
    """
    .. seealso::
        - :func:`add_delete_sampler <add_delete_sampler>`
        - :func:`exchange_sampler <exchange_sampler>`
        - :func:`initialize_AED_sampler <initialize_AED_sampler>`
        - :func:`add_exchange_delete_sampler <add_exchange_delete_sampler>`
    """
    rng = check_random_state(random_state)

    N = kernel.shape[0]
    ground_set = np.arange(N)

    S0, det_S0 = [], 0.0
    nb_trials = 100
    tol = 1e-9

    for _ in range(nb_trials):
        if det_S0 > tol:
            break
        else:
            T = rng.choice(2 * N, size=N, replace=False)
            S0 = np.intersect1d(T, ground_set, assume_unique=True)
            det_S0 = det_ST(kernel, S0)
    else:
        err_str = [
            "Initialization terminated unsuccessfully.",
            "After {} random trials, no initial set S0 satisfies det L_S0 > {}.".format(
                nb_trials, tol
            ),
            'You may consider passing your own initial state **{"s_init": S0}.',
        ]
        raise ValueError("\n".join(err_str))

    return S0.tolist()


def initialize_AD_and_E_sampler(kernel, size=None, random_state=None):
    """
    .. seealso::

        - :func:`add_delete_sampler <add_delete_sampler>`
        - :func:`exchange_sampler <exchange_sampler>`
        - :func:`initialize_AED_sampler <initialize_AED_sampler>`
        - :func:`add_exchange_delete_sampler <add_exchange_delete_sampler>`
    """
    rng = check_random_state(random_state)

    N = kernel.shape[0]

    S0, det_S0 = [], 0.0
    nb_trials = 100
    tol = 1e-9

    for _ in range(nb_trials):
        if det_S0 > tol:
            break
        else:
            S0 = rng.choice(
                N, size=size if size else rng.randint(1, N + 1), replace=False
            )
            det_S0 = det_ST(kernel, S0)
    else:
        err_str = [
            "Initialization terminated unsuccessfully.",
            "After {} random trials, no initial set S0 satisfies det L_S0 > {}.".format(
                nb_trials, tol
            ),
            "If you are sampling from a k-DPP, make sure k <= rank(L)." if size else "",
            'You may consider passing your own initial state **{"s_init": S0}.',
        ]
        raise ValueError("\n".join(err_str))

    return S0.tolist()


def add_exchange_delete_sampler(
    kernel, s_init=None, nb_iter=10, T_max=None, random_state=None
):
    """MCMC sampler for generic DPPs, it is a mix of add/delete and basis exchange MCMC samplers.

    :param kernel:
        Kernel matrix
    :type kernel:
        array_like

    :param s_init:
        Initial sample.
    :type s_init:
        list

    :param nb_iter:
        Maximum number of iterations performed by the the algorithm.
        Default is 10.
    :type nb_iter:
        int

    :param T_max:
        Maximum running time of the algorithm (in seconds).
    :type T_max:
        float

    :param random_state:
    :type random_state:
        None, np.random, int, np.random.RandomState

    :return:
        list of `nb_iter` approximate samples of DPP(kernel)
    :rtype:
        array_like

    .. seealso::

        Algorithm 3 in :cite:`LiJeSr16c`
    """
    rng = check_random_state(random_state)

    # Initialization
    N = kernel.shape[0]
    ground_set = np.arange(N)

    S0, det_S0 = s_init, det_ST(kernel, s_init)
    size_S0 = len(S0)  # Size of the current sample
    chain = [S0]  # Initialize the collection (list) of sample

    # Evaluate running time...
    t_start = time.time() if T_max else 0

    for _ in range(nb_iter):

        ratio = size_S0 / N  # Proportion of items in current sample
        S1 = S0.copy()  # S1 = S0

        # Pick one element s in S_0 by index uniformly at random
        s_ind = rng.randint(0, size_S0) if size_S0 else 0
        # Unif t in [N]-S0
        t = rng.choice(np.delete(ground_set, S0))

        U = rng.rand()
        # Add: S1 = S0 + t
        if U < 0.5 * (1 - ratio) ** 2:
            S1.append(t)  # S1 = S0 + t
            # Accept_reject the move
            det_S1 = det_ST(kernel, S1)  # det K_S1
            if rng.rand() < det_S1 / det_S0 * (size_S0 + 1) / (N - size_S0):
                S0, det_S0 = S1, det_S1
                size_S0 += 1

        # Exchange: S1 = S0 - s + t
        elif (0.5 * (1 - ratio) ** 2 <= U) & (U < 0.5 * (1 - ratio)):
            del S1[s_ind]  # S1 = S0 - s
            S1.append(t)  # S1 = S1 + t = S0 - s + t
            # Accept_reject the move
            det_S1 = det_ST(kernel, S1)  # det K_S1
            if rng.rand() < (det_S1 / det_S0):
                S0, det_S0 = S1, det_S1

        # Delete: S1 = S0 - s
        elif (0.5 * (1 - ratio) <= U) & (U < 0.5 * (ratio ** 2 + (1 - ratio))):
            del S1[s_ind]  # S0 - s
            # Accept_reject the move
            det_S1 = det_ST(kernel, S1)  # det K_S1
            if rng.rand() < det_S1 / det_S0 * size_S0 / (N - (size_S0 - 1)):
                S0, det_S0 = S1, det_S1
                size_S0 -= 1

        chain.append(S0)

        if T_max and (time.time() - t_start < T_max):
            break

    return chain


def add_exchange_delete_sampler_refactored(
    kernel, s_init=None, nb_iter=10, T_max=None, random_state=None
):
    """MCMC sampler for generic DPPs, it is a mix of add/delete and basis exchange MCMC samplers.

    :param kernel:
        Kernel matrix
    :type kernel:
        array_like

    :param s_init:
        Initial sample.
    :type s_init:
        list

    :param nb_iter:
        Maximum number of iterations performed by the the algorithm.
        Default is 10.
    :type nb_iter:
        int

    :param T_max:
        Maximum running time of the algorithm (in seconds).
    :type T_max:
        float

    :param random_state:
    :type random_state:
        None, np.random, int, np.random.RandomState

    :return:
        list of `nb_iter` approximate samples of DPP(kernel)
    :rtype:
        array_like

    .. seealso::

        Algorithm 3 in :cite:`LiJeSr16c`
    """
    rng = check_random_state(random_state)

    # Initialization, the ground set of items is devided into two: items forming the current samples and the rest.
    N = kernel.shape[0]
    items = s_init + [i for i in range(N) if i not in s_init]

    det_S0 = det_ST(kernel, s_init)
    size = len(s_init)
    chain = [s_init]

    t_start = time.time() if T_max else 0

    for _ in range(nb_iter):

        # Pick the indices of one element s inside and t outside of the current sample uniformly at random
        s, t = rng.randint(0, size), rng.randint(size, N)

        U, V = rng.rand(), rng.rand()
        ratio = size / N  # Proportion of items in current sample

        # Add: S += t
        if U < 0.5 * (1 - ratio) ** 2:
            items[t], items[size] = items[size], items[t]
            # Accept_reject the move
            det_S1 = det_ST(kernel, items[: size + 1])
            if V < det_S1 / det_S0 * (size + 1) / (N - size):
                det_S0 = det_S1
                size += 1

        # Exchange: S = S - s + t
        elif (0.5 * (1 - ratio) ** 2 <= U) & (U < 0.5 * (1 - ratio)):
            items[s], items[t] = items[t], items[s]
            # Accept_reject the move, size stays the same
            det_S1 = det_ST(kernel, items[:size])
            if V < det_S1 / det_S0:
                det_S0 = det_S1
            else:
                items[s], items[t] = items[t], items[s]

        # Delete: S -= s
        elif (0.5 * (1 - ratio) <= U) & (U < 0.5 * (ratio ** 2 + (1 - ratio))):
            items[s], items[size - 1] = items[size - 1], items[s]
            # Accept_reject the move
            det_S1 = det_ST(kernel, items[: size - 1])
            if V < det_S1 / det_S0 * size / (N - (size - 1)):
                det_S0 = det_S1
                size -= 1

        chain.append(items[:size])

        if T_max and (time.time() - t_start < T_max):
            break

    return chain


def add_delete_sampler(kernel, s_init, nb_iter=10, T_max=None, random_state=None):
    """MCMC sampler for generic DPP(kernel), it performs local moves by removing/adding one element at a time.

    :param kernel:
        Kernel matrix
    :type kernel:
        array_like

    :param s_init:
        Initial sample.
    :type s_init:
        list

    :param nb_iter:
        Maximum number of iterations performed by the the algorithm.
        Default is 10.
    :type nb_iter:
        int

    :param T_max:
        Maximum running time of the algorithm (in seconds).
        Default is None.
    :type T_max:
        float

    :param random_state:
    :type random_state:
        None, np.random, int, np.random.RandomState

    :return:
        list of `nb_iter` approximate samples of DPP(kernel)
    :rtype:
        array_like

    .. seealso::

        Algorithm 1 in :cite:`LiJeSr16c`
    """
    rng = check_random_state(random_state)

    # Initialization
    N = kernel.shape[0]  # Number of elements

    # Initialization
    S0, det_S0 = s_init, det_ST(kernel, s_init)
    chain = [S0]  # Initialize the collection (list) of sample

    # Evaluate running time...
    t_start = time.time() if T_max else 0

    for _ in range(nb_iter):

        # With proba 1/2 try to add/delete an element
        if rng.rand() < 0.5:

            # Perform the potential add/delete move S1 = S0 +/- s
            S1 = S0.copy()  # S1 = S0
            s = rng.choice(N)  # Uniform item in [N]
            if s in S1:
                S1.remove(s)  # S1 = S0 - s
            else:
                S1.append(s)  # S1 = SO + s

            # Accept_reject the move
            det_S1 = det_ST(kernel, S1)  # det K_S1
            if rng.rand() < det_S1 / det_S0:
                S0, det_S0 = S1, det_S1

        chain.append(S0)

        if T_max and (time.time() - t_start < T_max):
            break

    return chain


def add_delete_sampler_refactored(
    kernel, s_init, nb_iter=10, T_max=None, random_state=None
):
    """MCMC sampler for generic DPP(kernel), it performs local moves by removing/adding one element at a time.

    :param kernel:
        Kernel matrix
    :type kernel:
        array_like

    :param s_init:
        Initial sample.
    :type s_init:
        list

    :param nb_iter:
        Maximum number of iterations performed by the the algorithm.
        Default is 10.
    :type nb_iter:
        int

    :param T_max:
        Maximum running time of the algorithm (in seconds).
        Default is None.
    :type T_max:
        float

    :param random_state:
    :type random_state:
        None, np.random, int, np.random.RandomState

    :return:
        list of `nb_iter` approximate samples of DPP(kernel)
    :rtype:
        list of lists

    .. seealso::

        Algorithm 1 in :cite:`LiJeSr16c`
    """

    # Initialization
    rng = check_random_state(random_state)

    N = kernel.shape[0]
    items = s_init + [i for i in range(N) if i not in s_init]

    det_S0, size, add_or_del = det_ST(kernel, s_init), len(s_init), 0
    chain = [s_init]

    t_start = time.time() if T_max else 0

    for _ in range(nb_iter):

        # With proba 1/2 try to add/delete an element
        if rng.rand() < 0.5:

            s = rng.randint(0, N)  # Uniform item in [N]
            if s >= size:  # S += s
                items[s], items[size] = items[size], items[s]
                add_or_del = 1
            else:  # S -= s
                items[s], items[size - 1] = items[size - 1], items[s]
                add_or_del = -1

            # Accept_reject the move
            det_S1 = det_ST(kernel, items[: size + add_or_del])
            if rng.rand() < det_S1 / det_S0:
                det_S0 = det_S1
                size += add_or_del

        chain.append(items[:size])

        if T_max and (time.time() - t_start < T_max):
            break

    return chain


def basis_exchange_sampler(kernel, s_init, nb_iter=10, T_max=None, random_state=None):
    """MCMC sampler for projection DPPs, based on the basis exchange property.

    :param kernel:
        Feature vector matrix, feature vectors are stacked columnwise.
        It is assumed to be full row rank.
    :type kernel:
        array_like

    :param s_init:
        Initial sample.
    :type s_init:
        list

    :param nb_iter:
        Maximum number of iterations performed by the the algorithm.
        Default is 10.
    :type nb_iter:
        int

    :param T_max:
        Maximum running time of the algorithm (in seconds).
        Default is None.
    :type T_max:
        float

    :param random_state:
    :type random_state:
        None, np.random, int, np.random.RandomState

    :return:
        MCMC chain of approximate sample (stacked row_wise i.e. nb_iter rows).
    :rtype:
        list of lists

    .. seealso::

        Algorithm 2 in :cite:`LiJeSr16c`
    """
    rng = check_random_state(random_state)

    # Initialization
    N = kernel.shape[0]  # Number of elements
    ground_set = np.arange(N)  # Ground set

    size = len(s_init)  # Size of the sample (cardinality is fixed)
    # Initialization
    S0, det_S0 = s_init, det_ST(kernel, s_init)

    chain = np.zeros((nb_iter, size), dtype=int)
    chain[0] = S0

    # Evaluate running time...
    t_start = time.time() if T_max else 0

    for it in range(1, nb_iter):

        # With proba 1/2 try to swap 2 elements
        if rng.rand() < 0.5:

            # Perform the potential exchange move S1 = S0 - s + t
            S1 = S0.copy()  # S1 = S0
            # Pick one element s in S0 by index uniformly at random
            s_ind = rng.choice(size)
            # Pick one element t in [N]\S0 uniformly at random
            t = rng.choice(np.delete(ground_set, S0))
            S1[s_ind] = t  # S_1 = S0 - S0[s_ind] + t

            det_S1 = det_ST(kernel, S1)  # det K_S1

            # Accept_reject the move w. proba
            if rng.rand() < det_S1 / det_S0:
                S0, det_S0 = S1, det_S1
                chain[it] = S1

            else:  # if reject, stay in the same state
                chain[it] = S0

        else:
            chain[it] = S0

        if T_max:
            if time.time() - t_start < T_max:
                break

    return chain.tolist()


def exchange_sampler_refactored(
    kernel, s_init, nb_iter=10, T_max=None, random_state=None
):
    """MCMC sampler for projection DPPs, based on the basis exchange property.

    :param kernel:
        Feature vector matrix, feature vectors are stacked columnwise.
        It is assumed to be full row rank.
    :type kernel:
        array_like

    :param s_init:
        Initial sample.
    :type s_init:
        list

    :param nb_iter:
        Maximum number of iterations performed by the the algorithm.
        Default is 10.
    :type nb_iter:
        int

    :param T_max:
        Maximum running time of the algorithm (in seconds).
        Default is None.
    :type T_max:
        float

    :param random_state:
    :type random_state:
        None, np.random, int, np.random.RandomState

    :return:
        MCMC chain of approximate sample (stacked row_wise i.e. nb_iter rows).
    :rtype:
        list of lists

    .. seealso::

        Algorithm 2 in :cite:`LiJeSr16c`
    """

    # Initialization
    rng = check_random_state(random_state)

    N = kernel.shape[0]
    items = s_init + [i for i in range(N) if i not in s_init]

    det_S0, size = det_ST(kernel, s_init), len(s_init)

    chain = np.zeros((nb_iter, size), dtype=int)
    chain[0] = s_init

    t_start = time.time() if T_max else 0

    for it in range(1, nb_iter):

        if rng.rand() < 0.5:

            s, t = rng.randint(0, size), rng.randint(size, N)
            items[s], items[t] = items[t], items[s]

            det_S1 = det_ST(kernel, items[:size])
            if rng.rand() < det_S1 / det_S0:
                det_S0 = det_S1
            else:
                items[s], items[t] = items[t], items[s]

        chain[it] = items[:size]

        if T_max and time.time() - t_start < T_max:
            break

    return chain.tolist()


def exchange_sampler_gauss_quadrature(
    kernel, s_init, nb_iter=10, T_max=None, random_state=None
):

    rng = check_random_state(random_state)

    N = kernel.shape[0]
    items = s_init + [x for x in range(N) if x not in s_init]

    size = len(s_init)

    chain = np.zeros((nb_iter, size), dtype=int)
    chain[0] = s_init

    t_start = time.time() if T_max else 0

    for it in range(1, nb_iter):

        ind_x, ind_y = rng.randint(0, size), rng.randint(size, N)
        x, y = items[ind_x], items[ind_y]

        u = rng.rand()
        if judge_exchange_gauss_quadrature(
            unif=u, kernel=kernel, sample=items[:size], x_del=x, y_add=y
        ):
            items[ind_x], items[ind_y] = y, x

        chain[it] = items[:k]

        if T_max and time.time() - t_start < T_max:
            break

    return chain.tolist()


############
# ZONOTOPE #
############
def extract_basis(y_sol, eps=1e-5):
    """Subroutine of :func:`zonotope_sampler <zonotope_sampler>` to extract the tile of the zonotope
    in which a point lies. It extracts the indices of entries of the solution
    of LP :eq:`eq:Px` that are in (0,1).

    :param y_sol:
        Optimal solution of LP :eq:`eq:Px`
    :type y_sol:
        list

    :param eps:
        Tolerance :math:`y_i^* \\in (\\epsilon, 1-\\epsilon), \\quad \\epsilon \\geq 0`
    :eps type:
        float

    :return:
        Indices of the feature vectors spanning the tile in which the point is lies.
        :math:`B_{x} = \\left\\{ i \\, ; \\, y_i^* \\in (0,1) \\right\\}`
    :rtype:
        list

    .. seealso::

        Algorithm 3 in :cite:`GaBaVa17`

        - :func:`zonotope_sampler <zonotope_sampler>`
    """

    basis = np.where((eps < y_sol) & (y_sol < 1 - eps))[0]

    return basis


def zonotope_sampler(A_zono, **params):
    """MCMC based sampler for projection DPPs.
    The similarity matrix is the orthogonal projection matrix onto
    the row span of the feature vector matrix.
    Samples are of size equal to the ransampl_size of the projection matrix
    also equal to the rank of the feature matrix (assumed to be full row rank).

    :param A_zono:
        Feature vector matrix, feature vectors are stacked columnwise.
        It is assumed to be full row rank.
    :type A_zono:
        array_like

    :param params: Dictionary containing the parameters

        - ``'lin_obj'`` (list): Linear objective (:math:`c`) of the linear program used to identify the tile in which a point lies. Default is a random Gaussian vector.
        - ``'x_0'` (list): Initial point.
        - ``'nb_iter'`` (int): Number of iterations of the MCMC chain. Default is 10.
        - ``'T_max'`` (float): Maximum running time of the algorithm (in seconds).
        Default is None.
        - ``'random_state`` (default None)
    :type params: dict

    :return:
        MCMC chain of approximate samples (stacked row_wise i.e. nb_iter rows).
    :rtype:
        list of lists

    .. seealso::

        Algorithm 5 in :cite:`GaBaVa17`

        - :func:`extract_basis <extract_basis>`
        - :func:`exchange_sampler <exchange_sampler>`
    """
    # For zonotope sampler
    try:
        from cvxopt import matrix, solvers, spmatrix
    except ImportError:
        raise ValueError(
            "The cvxopt package is required to use the zonotype sampler (see setup.py)."
        )

    solvers.options["show_progress"] = params.get("show_progress", False)
    solvers.options["glpk"] = {"msg_lev": params.get("show_progress", "GLP_MSG_OFF")}

    rng = check_random_state(params.get("random_state", None))

    r, N = A_zono.shape  # Sizes of r=samples=rank(A_zono), N=ground set
    # Linear objective
    c = matrix(params.get("lin_obj", rng.randn(N)))
    # Initial point x0 = A*u, u~U[0,1]^n
    x0 = matrix(params.get("x_0", A_zono.dot(rng.rand(N))))

    nb_iter = params.get("nb_iter", 10)
    T_max = params.get("T_max", None)

    ###################
    # Linear problems #
    ###################
    # Canonical form
    # min       c.T*x         min     c.T*x
    # s.t.  G*x <= h    <=>   s.t.    G*x + s = h
    #        A*x = b                      A*x = b
    #                                      s >= 0
    # CVXOPT
    # =====> solvers.lp(c, G, h, A, b, solver='glpk')
    #################################################

    # To access the tile Z(B_x)
    # Solve P_x(A,c)
    ######################################################
    # y^* =
    # argmin  c.T*y               argmin  c.T*y
    # s.t.  A*y = x         <=>   s.t.  A  *y  = x
    #       0 <= y <= 1             [ I_n] *y <= [1^n]
    #                               [-I_n]       [0^n]
    ######################################################
    # Then B_x = \{ i ; y_i^* \in ]0,1[ \}

    A = spmatrix(0.0, [], [], (r, N))
    A[:, :] = A_zono

    G = spmatrix(0.0, [], [], (2 * N, N))
    G[:N, :] = spmatrix(1.0, range(N), range(N))
    G[N:, :] = spmatrix(-1.0, range(N), range(N))

    # Endpoints of segment
    # D_x \cap Z(A) = [x+alpha_m*d, x-alpha_M*d]
    ###########################################################################
    # alpha_m/_M = argmin  +/-alpha      argmin [+/-1 0^N].T * [alpha,lambda]
    # s.t.    x + alpha d = A lambda <=> s.t.  [-d A] *[alpha, lambda] = x
    #         0 <= lambda <= 1             [0^N I_N] *[alpha, lambda] <= [1^N]
    #                                      [0^N -I_N]                    [0^N]
    ##########################################################################

    c_mM = matrix(0.0, (N + 1, 1))
    c_mM[0] = 1.0

    A_mM = spmatrix(0.0, [], [], (r, N + 1))
    A_mM[:, 1:] = A

    G_mM = spmatrix(0.0, [], [], (2 * N, N + 1))
    G_mM[:, 1:] = G

    # Common h to both kind of LP
    # cf. 0 <= y <= 1 and 0 <= lambda <= 1
    h = matrix(0.0, (2 * N, 1))
    h[:N, :] = 1.0

    ##################
    # Initialization #
    ##################
    B_x0 = []
    while len(B_x0) != r:
        # Initial tile B_x0
        # Solve P_x0(A,c)
        y_star = solvers.lp(c, G, h, A, x0, solver="glpk")["x"]
        # Get the tile
        B_x0 = extract_basis(np.asarray(y_star))

    # Initialize sequence of sample
    chain = np.zeros((nb_iter, r), dtype=int)
    chain[0] = B_x0

    # Compute the det of the tile (Vol(B)=abs(det(B)))
    det_B_x0 = la.det(A_zono[:, B_x0])

    t_start = time.time() if T_max else 0

    for it in range(1, nb_iter):

        # Take uniform direction d defining D_x0
        d = matrix(rng.randn(r, 1))

        # Define D_x0 \cap Z(A) = [x0 + alpha_m*d, x0 - alpha_M*d]
        # Update the constraint [-d A] * [alpha,lambda] = x
        A_mM[:, 0] = -d
        # Find alpha_m/M
        alpha_m = solvers.lp(c_mM, G_mM, h, A_mM, x0, solver="glpk")["x"][0]
        alpha_M = solvers.lp(-c_mM, G_mM, h, A_mM, x0, solver="glpk")["x"][0]

        # Propose x1 ~ U_{[x0+alpha_m*d, x0-alpha_M*d]}
        x1 = x0 + (alpha_m + (alpha_M - alpha_m) * rng.rand()) * d
        # Proposed tile B_x1
        # Solve P_x1(A,c)
        y_star = solvers.lp(c, G, h, A, x1, solver="glpk")["x"]
        # Get the tile
        B_x1 = extract_basis(np.asarray(y_star))

        # Accept/Reject the move with proba Vol(B1)/Vol(B0)
        if len(B_x1) != r:  # if extract_basis returned smtg ill conditioned
            chain[it] = B_x0
        else:
            det_B_x1 = la.det(A_zono[:, B_x1])
            if rng.rand() < abs(det_B_x1 / det_B_x0):
                x0, B_x0, det_B_x0 = x1, B_x1, det_B_x1
                chain[it] = B_x1
            else:
                chain[it] = B_x0

        if T_max and time.time() - t_start < T_max:
            break

    return chain.tolist()
