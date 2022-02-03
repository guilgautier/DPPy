from dppy.finite.mcmc_samplers.add_delete_sampler import add_delete_sampler
from dppy.finite.mcmc_samplers.add_exchange_delete_sampler import (
    add_exchange_delete_sampler,
)
from dppy.finite.mcmc_samplers.exchange_sampler import exchange_sampler
from dppy.finite.mcmc_samplers.zonotope_sampler import zonotope_sampler


def select_sampler_mcmc_dpp(dpp, method):
    samplers = {
        "aed": add_exchange_delete_sampler,
        "ad": add_delete_sampler,
        "e": exchange_sampler,
        "zonotope": zonotope_sampler,
    }
    default = samplers["aed"]
    return samplers.get(method.lower(), default)


def select_sampler_mcmc_k_dpp(dpp, method):
    samplers = {
        "e": exchange_sampler,
    }
    default = "e"
    return samplers.get(method.lower(), samplers[default])
