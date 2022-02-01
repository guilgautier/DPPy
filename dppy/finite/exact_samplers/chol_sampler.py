from dppy.finite.exact_samplers.projection_kernel_samplers import (
    projection_kernel_sampler,
)
from dppy.finite.exact_samplers.sequential_samplers import generic_sampler


def chol_sampler(dpp, random_state=None, **params):
    r"""Generate an exact sample from
    ``dpp`` using the :ref:`generic method <finite_dpps_exact_sampling_sequential_methods>`.

        :param dpp: Finite DPP
        :type dpp: :py:class:`~dppy.finite.dpp.FiniteDPP`

        :param random_state: random number generator or seed, defaults to None
        :type random_state: optional

        Keyword arguments:
            - mode (str): select the variant of the sampler, see :py:func:`~dppy.finite.exact_samplers.projection_kernel_samplers.projection_kernel_sampler` if else :py:func:`~dppy.finite.exact_samplers.sequential_samplers.generic_sampler`

        :return: list
        :rtype: sample
    """
    params["mode"] = "chol"

    cond_K = dpp.kernel_type == "correlation"
    cond_L = dpp.kernel_type == "likelihood" and params.get("size")

    if dpp.projection and (cond_K or cond_L):
        sampler = projection_kernel_sampler
    else:
        sampler = generic_sampler

    sample = sampler(dpp, random_state=random_state, **params)
    return sample


def chol_sampler_k_dpp(dpp, size, random_state=None, **params):
    params["size"] = size
    return chol_sampler(dpp, random_state=None, **params)
