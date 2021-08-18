from dppy.finite_dpps.projection_kernel_sampler import projection_kernel_sampler


def schur_sampler(dpp, random_state, **params):
    params["mode"] = "Schur"
    return projection_kernel_sampler(dpp, random_state=random_state, **params)
