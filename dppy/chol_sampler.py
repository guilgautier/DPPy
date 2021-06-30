from .projection_kernel_sampler import projection_kernel_sampler
from .generic_kernel_sampler import dpp_sampler_generic_kernel


def chol_sampler(dpp, random_state, **params):
    if dpp.kernel_type == "correlation" and dpp.hermitian and dpp.projection:
        return projection_kernel_sampler(
            dpp, mode="Chol", random_state=random_state, **params
        )
    else:
        dpp.compute_K()
        sample, _ = dpp_sampler_generic_kernel(dpp.K, random_state=random_state)
        return sample
