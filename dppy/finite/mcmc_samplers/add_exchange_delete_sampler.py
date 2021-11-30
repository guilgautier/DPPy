import time

import numpy as np

from dppy.utils import check_random_state, det_ST


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
