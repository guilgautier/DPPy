import time

from dppy.utils import check_random_state, det_ST


def add_delete_sampler(dpp, random_state=None, **params):
    r"""MCMC based sampler for sampling approximately from a DPP, by performing `single addition or single deletion moves <finite_dpps_mcmc_sampling_add_exchange_delete>`_.

    See also :py:func:`~dppy.finite.mcmc_samplers.add_delete_sampler.add_delete_sampler_core`.

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
    dpp.compute_L()
    kernel = dpp.L
    rng = check_random_state(random_state)
    s0 = params.pop("s_init", None)
    if s0 is None:
        s0 = initialize_add_delete_sampler(kernel, rng, **params)
    return add_delete_sampler_core(kernel, s0, rng, **params)


def add_delete_sampler_core(
    kernel, s_init, random_state=None, nb_iter=10, T_max=None, **kwargs
):
    """MCMC based sampler for sampling approximately from a DPP, by performing single addition or single deletion moves.

    This function implements Algorithm 1 in :cite:`LiJeSr16c`.

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


def initialize_add_delete_sampler(
    kernel, random_state=None, size=None, nb_trials=100, tol=1e-9, **kwargs
):
    r"""Initialize the add-delete Markov chain with a sample :math:`X_0` with cardinality ``size``, such that :math:`\det K_{X_0} >` ``tol``.

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
        _size = rng.randint(1, N + 1) if size is None else size
        S0 = rng.choice(N, size=_size, replace=False)
        det_S0 = det_ST(kernel, S0)
        if det_S0 > tol:
            return S0.tolist()

    raise ValueError(
        "Failed to initialize add-delete sampler. After {} random trials, no initial set S0 satisfies det L_S0 > {}. If you are sampling from a k-DPP, make sure size k <= rank(L). You may consider passing your own initial state s_init.".format(
            nb_trials, tol
        )
    )
