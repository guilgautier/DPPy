import time

import numpy as np

from dppy.judge_ratio_of_determinants import judge_exchange_gauss_quadrature
from dppy.utils import check_random_state, det_ST


def exchange_sampler(dpp, random_state=None, **params):
    r"""MCMC based sampler for sampling approximately from a DPP, by performing `exchange (swap) moves <finite_dpps_mcmc_sampling_add_exchange_delete>`_.

    See also :py:func:`~dppy.finite.mcmc_samplers.exchange_sampler.exchange_sampler_core`.

    :param dpp:
        Finite DPP
    :type dpp:
        :py:class:`~dppy.finite.dpp.FiniteDPP`

    :param random_state:
        random number generator or seed, defaults to None
    :type random_state:
        optional

    :return:
        MCMC chain of approximate samples (stacked row_wise i.e. max_iter rows).
    :rtype:
        list of lists
    """
    rng = check_random_state(random_state)
    s0 = params.pop("s_init", None)
    size = params.pop("size", None if s0 is None else len(s0))
    if size is None:
        raise ValueError(
            "Exchange sampler cannot be initialized, consider passing s_init or size keyword argument."
        )
    kernel = get_exchange_sampler_kernel(dpp, size)
    if s0 is None:
        s0 = initialize_exchange_sampler(kernel, size, rng, **params)
    return exchange_sampler_core(kernel, s0, rng, **params)


def get_exchange_sampler_kernel(dpp, size):
    if dpp.kernel_type == "correlation" and dpp.projection:
        dpp.compute_K()
        rank = np.rint(np.trace(dpp.K)).astype(int)
        if size != rank:
            raise ValueError(
                "size={} must be equal to rank={} for DPP(kernel_type='correlation', projection=True, ...)".format(
                    size, rank
                )
            )
        return dpp.K
    dpp.compute_L()
    return dpp.L


def exchange_sampler_core(
    kernel, s_init, random_state=None, nb_iter=10, T_max=None, **kwargs
):
    """MCMC based sampler for sampling approximately from a DPP, by performing exhange (swap) moves.

    This function implements Algorithm 2 in :cite:`LiJeSr16c`.

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

        chain[it] = items[:size]

        if T_max and time.time() - t_start < T_max:
            break

    return chain.tolist()


def initialize_exchange_sampler(
    kernel, size, random_state=None, nb_trials=100, tol=1e-9, **kwargs
):
    r"""Initialize the exchange (swap) Markov chain with a sample :math:`X_0` with cardinality ``size``, such that :math:`\det K_{X_0} >` ``tol``.

    :param kernel: Kernel matrix :math:`K`
    :type kernel: array_like

    :param random_state: Random number generator, defaults to None
    :type random_state: optional

    :param size: size of the initial sample, defaults to None
    :type size: int, optional

    :param nb_trials: Maximum number of proposed initial samples, defaults to 100. If no proposed sample satisfies the above condition, an error is raised.
    :type nb_trials: int, optional

    :param tol: Threshold such that :math:`\det K_{X_0} >` ``tol``, defaults to 1e-9
    :type tol: float, optional

    :return: initial sample :math:`X_0`
    :rtype: list
    """
    rng = check_random_state(random_state)
    N = kernel.shape[0]

    for _ in range(nb_trials):
        S0 = rng.choice(N, size=size, replace=False)
        det_S0 = det_ST(kernel, S0)
        if det_S0 > tol:
            return S0.tolist()

    raise ValueError(
        "Failed to initialize exchange sampler. After {} random trials, no initial set S0 satisfies det L_S0 > {}. If you are sampling from a k-DPP, make sure size k <= rank(L). You may consider passing your own initial state s_init.".format(
            nb_trials, tol
        )
    )
