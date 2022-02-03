from dppy.finite.exact_samplers.intermediate_sampler import (
    intermediate_sampler_dpp,
    intermediate_sampler_k_dpp,
)
from dppy.finite.exact_samplers.projection_sampler_kernel import (
    projection_sampler_kernel,
)
from dppy.finite.exact_samplers.sequential_samplers import generic_sampler
from dppy.finite.exact_samplers.spectral_sampler import (
    spectral_sampler_dpp,
    spectral_sampler_k_dpp,
)


def select_sampler_exact_dpp(dpp, method):
    samplers = {
        "spectral": spectral_sampler_dpp,
        "projection": projection_sampler_kernel,
        "intermediate": intermediate_sampler_dpp,
        "generic": generic_sampler,
    }
    default = samplers["spectral" if dpp.hermitian else "generic"]
    return samplers.get(method.lower(), default)


def select_sampler_exact_k_dpp(dpp, method):
    samplers = {
        "spectral": spectral_sampler_k_dpp,
        "projection": projection_sampler_kernel,
        "intermediate": intermediate_sampler_k_dpp,
    }
    default = samplers["spectral"]
    return samplers.get(method.lower(), default)
