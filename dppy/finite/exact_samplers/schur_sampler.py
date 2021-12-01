from dppy.finite.exact_samplers.projection_kernel_samplers import (
    projection_kernel_sampler,
)


def schur_sampler(dpp, random_state=None, **params):
    params["mode"] = "Schur"
    return projection_kernel_sampler(dpp, random_state=random_state, **params)


def schur_sampler_k_dpp(dpp, size, random_state=None, **params):
    params["size"] = size
    return schur_sampler(dpp, random_state=None, **params)
