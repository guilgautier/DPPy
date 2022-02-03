from dppy.finite.exact_samplers.intermediate_sampler_alpha import (
    intermediate_sampler_alpha_dpp,
    intermediate_sampler_alpha_k_dpp,
)
from dppy.finite.exact_samplers.intermediate_sampler_vfx import (
    intermediate_sampler_vfx_dpp,
    intermediate_sampler_vfx_k_dpp,
)


def intermediate_sampler_dpp(dpp, random_state=None, **kwargs):
    assert dpp.hermitian
    mode = kwargs.get("mode", "")
    sampler = select_intermediate_sampler_dpp(mode)
    return sampler(dpp, random_state=random_state, **kwargs)


def select_intermediate_sampler_dpp(mode):
    samplers = {
        "alpha": intermediate_sampler_alpha_dpp,
        "vfx": intermediate_sampler_vfx_dpp,
    }
    default = samplers["vfx"]
    return samplers.get(mode.lower(), default)


def intermediate_sampler_k_dpp(dpp, size=None, random_state=None, **kwargs):
    assert dpp.hermitian
    mode = kwargs.get("mode", "")
    sampler = select_intermediate_sampler_k_dpp(mode)
    return sampler(dpp, size=size, random_state=random_state, **kwargs)


def select_intermediate_sampler_k_dpp(mode):
    samplers = {
        "alpha": intermediate_sampler_alpha_k_dpp,
        "vfx": intermediate_sampler_vfx_k_dpp,
    }
    default = samplers["vfx"]
    return samplers.get(mode.lower(), default)
