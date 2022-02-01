from dppy.finite.exact_samplers.alpha_samplers import (
    alpha_sampler_dpp,
    alpha_sampler_k_dpp,
)
from dppy.finite.exact_samplers.chol_sampler import chol_sampler, chol_sampler_k_dpp
from dppy.finite.exact_samplers.schur_sampler import schur_sampler, schur_sampler_k_dpp
from dppy.finite.exact_samplers.sequential_samplers import generic_sampler
from dppy.finite.exact_samplers.spectral_sampler_dpp import spectral_sampler
from dppy.finite.exact_samplers.spectral_sampler_k_dpp import spectral_sampler_k_dpp
from dppy.finite.exact_samplers.vfx_samplers import vfx_sampler_dpp, vfx_sampler_k_dpp


def select_sampler_exact_dpp(dpp, method):
    samplers = {
        "spectral": spectral_sampler,
        "vfx": vfx_sampler_dpp,
        "alpha": alpha_sampler_dpp,
        "schur": schur_sampler,
        "chol": chol_sampler,
        "generic": generic_sampler,
    }
    default = "spectral" if dpp.hermitian else "generic"
    return samplers.get(method.lower(), samplers[default])


def select_sampler_exact_k_dpp(dpp, method):
    samplers = {
        "spectral": spectral_sampler_k_dpp,
        "vfx": vfx_sampler_k_dpp,
        "alpha": alpha_sampler_k_dpp,
        "schur": schur_sampler_k_dpp,
        "chol": chol_sampler_k_dpp,
    }
    default = "spectral" if dpp.hermitian else "schur"
    return samplers.get(method.lower(), samplers[default])
