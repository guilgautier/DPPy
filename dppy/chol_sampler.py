from .exact_sampling import (
    proj_dpp_sampler_kernel, dpp_sampler_generic_kernel)


def chol_sampler(dpp, rng, **params):
    dpp.compute_K()
    if dpp.kernel_type == 'correlation' and dpp.projection:
        return proj_dpp_sampler_kernel(
            dpp.K, dpp.sampling_mode, random_state=rng)
    else:
        sample, _ = dpp_sampler_generic_kernel(dpp.K, random_state=rng)
        return sample
