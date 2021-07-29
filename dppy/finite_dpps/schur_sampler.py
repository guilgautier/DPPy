from dppy.finite_dpps.projection_kernel_sampler import projection_kernel_sampler


def schur_sampler(dpp, random_state, **params):
    return projection_kernel_sampler(
        dpp, mode="Schur", random_state=random_state, **params
    )
