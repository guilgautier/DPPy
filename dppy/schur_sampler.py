from .exact_sampling import proj_dpp_sampler_kernel

def schur_sampler(dpp, rng):
    if dpp.kernel_type == 'correlation' and dpp.projection:
        dpp.compute_K()
        return proj_dpp_sampler_kernel(
            dpp.K, dpp.sampling_mode, random_state=rng)
    else:
        err_print = ""
        raise ValueError('\n'.join(err_print))


if __name__ == "__main__":
    # execute only if run as a script

    from numpy.random import rand, randn
    from scipy.linalg import qr

    from .finite_dpps import FiniteDPP

    r, N = 6, 10
    eig_vecs, _ = qr(randn(N, r), mode='economic')
    # eig_vals = rand(r)  # 0< <1
    eig_vals = np.ones(r)

    my_dpp = FiniteDPP('correlation', True, K_eig_dec=(eig_vals, eig_vecs))

    sample = my_dpp.sample_exact(mode="Schur")
    print(sample)
