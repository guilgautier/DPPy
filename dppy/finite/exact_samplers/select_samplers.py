from dppy.finite.exact_samplers.alpha_samplers import (
    alpha_sampler_dpp,
    alpha_sampler_k_dpp,
)
from dppy.finite.exact_samplers.projection_sampler_kernel import (
    projection_sampler_kernel,
)
from dppy.finite.exact_samplers.sequential_samplers import generic_sampler
from dppy.finite.exact_samplers.spectral_sampler import (
    spectral_sampler_dpp,
    spectral_sampler_k_dpp,
)
from dppy.finite.exact_samplers.vfx_samplers import vfx_sampler_dpp, vfx_sampler_k_dpp


def select_sampler_exact_dpp(dpp, method):
    samplers = {
        "spectral": spectral_sampler_dpp,
        "projection": projection_sampler_kernel,
        "vfx": vfx_sampler_dpp,
        "alpha": alpha_sampler_dpp,
        "generic": generic_sampler,
    }
    default = "spectral" if dpp.hermitian else "generic"
    return samplers.get(method.lower(), samplers[default])


def select_sampler_exact_k_dpp(dpp, method):
    samplers = {
        "spectral": spectral_sampler_k_dpp,
        "projection": projection_sampler_kernel,
        "vfx": vfx_sampler_k_dpp,
        "alpha": alpha_sampler_k_dpp,
    }
    default = "spectral" if dpp.hermitian else "schur"
    return samplers.get(method.lower(), samplers[default])
