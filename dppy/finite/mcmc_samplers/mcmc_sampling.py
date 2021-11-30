# coding: utf8
""" Implementation of finite DPP MCMC samplers:

- `add_exchange_delete_sampler`
- `add_delete_sampler`
- `exchange_sampler`
- `zonotope_sampler`

.. seealso:

    `Documentation on ReadTheDocs <https://dppy.readthedocs.io/en/latest/finite_dpps/mcmc_sampling.html>`_
"""

import numpy as np

from dppy.finite.mcmc_samplers.add_delete_sampler import add_delete_sampler
from dppy.finite.mcmc_samplers.add_exchange_delete_sampler import (
    add_exchange_delete_sampler,
)
from dppy.finite.mcmc_samplers.exchange_sampler import basis_exchange_sampler
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


def initialize_AED_sampler(kernel, random_state=None, nb_trials=100, tol=1e-9):
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

    for _ in range(nb_trials):
        if det_S0 > tol:
            return S0.tolist()
        T = rng.choice(2 * N, size=N, replace=False)
        S0 = np.intersect1d(T, ground_set, assume_unique=True)
        det_S0 = det_ST(kernel, S0)
    raise ValueError(
        "Unsuccessful initialization of add-exchange-delete sampler. After {} random trials, no initial set S0 satisfies det L_S0 > {}. You may consider passing your own initial state s_init.".format(
            nb_trials, tol
        )
    )


def initialize_AD_and_E_sampler(
    kernel, size=None, random_state=None, nb_trials=100, tol=1e-9
):
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

    for _ in range(nb_trials):
        if det_S0 > tol:
            return S0.tolist()
        S0 = rng.choice(N, size=size if size else rng.randint(1, N + 1), replace=False)
        det_S0 = det_ST(kernel, S0)

    raise ValueError(
        "Unsuccessful initialization of add-delete or exchange sampler. After {} random trials, no initial set S0 satisfies det L_S0 > {}. If you are sampling from a k-DPP, make sure size k <= rank(L). You may consider passing your own initial state s_init.".format(
            nb_trials, tol
        )
    )
