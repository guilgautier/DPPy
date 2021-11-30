from dppy.finite.exact_samplers.generic_samplers import generic_sampler
from dppy.finite.exact_samplers.projection_kernel_samplers import (
    projection_kernel_sampler,
)


def chol_sampler(dpp, random_state, **params):
    params["mode"] = "Chol"

    cond_K = dpp.kernel_type == "correlation"
    cond_L = dpp.kernel_type == "likelihood" and params.get("size")

    if dpp.projection and (cond_K or cond_L):
        sampler = projection_kernel_sampler
    else:
        sampler = generic_sampler

    sample = sampler(dpp, random_state=random_state, **params)
    return sample
