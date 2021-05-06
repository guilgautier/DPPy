import numpy as np
import numpy.linalg as la

from scipy.linalg import qr
from warnings import warn

from .exact_sampling import (
    dpp_eig_vecs_selector, proj_dpp_sampler_eig, proj_dpp_sampler_kernel)
from .utils import (check_in_01, check_geq_0)


def spectral_sampler(dpp, rng, **params):

    compute_spectral_sampler_parameters(dpp)
    return do_spectral_sampler(dpp, rng)


def do_spectral_sampler(dpp, rng):

    if dpp.K_eig_vals is not None:
        # Phase 1
        if dpp.kernel_type == 'correlation' and dpp.projection:
            V = dpp.eig_vecs[:, dpp.K_eig_vals > 0.5]
        else:
            V = dpp_eig_vecs_selector(dpp.K_eig_vals, dpp.eig_vecs,
                                      random_state=rng)
        # Phase 2
        return proj_dpp_sampler_eig(V, dpp.sampling_mode,
                                    random_state=rng)

    # # If DPP defined via projection correlation kernel K
    # # no eigendecomposition required
    elif dpp.K is not None and dpp.projection:
        return proj_dpp_sampler_kernel(dpp.K, dpp.sampling_mode,
                                       random_state=rng)
    else:
        raise ValueError("Invalid dpp parameters. Spectral sampler requires spec")


def compute_spectral_sampler_parameters(dpp):
    """Compute eigenvalues and eigenvectors of correlation kernel K from various parametrizations of ``dpp``

    :param dpp: ``FiniteDPP`` object
    :type dpp: ``FiniteDPP`` object
    """
    while compute_spectral_sampler_parameters_step(dpp):
        pass


def compute_spectral_sampler_parameters_step(dpp):
    """
    Returns
    ``False`` if the right parameters are indeed computed
    ``True`` if extra computations are required

    Note: Sort of fixed point algorithm to find dpp.K_eig_vals and dpp.eig_vecs
    """
    if dpp.K_eig_vals is not None:
        return False

    if dpp.L_eig_vals is not None:
        dpp.K_eig_vals = dpp.L_eig_vals / (1.0 + dpp.L_eig_vals)
        return False

    elif dpp.K is not None:
        dpp.K_eig_vals, dpp.eig_vecs = la.eigh(dpp.K)
        check_in_01(dpp.K_eig_vals)
        return False

    elif dpp.L_dual is not None:
        phi = dpp.L_gram_factor
        # L_dual = Phi * Phi.T = W Theta W.T
        # L = Phi.T Phi = V Gamma V
        # implies Gamma = Theta and V = Phi.T W Theta^{-1/2}
        dpp.L_eig_vals, eig_vecs = la.eigh(dpp.L_dual)
        check_geq_0(dpp.L_eig_vals)
        dpp.L_eig_vals = np.maximum(dpp.L_eig_vals, 0.0)
        dpp.eig_vecs = phi.T.dot(eig_vecs / np.sqrt(dpp.L_eig_vals))
        return True

    elif dpp.L is not None:
        dpp.L_eig_vals, dpp.eig_vecs = la.eigh(dpp.L)
        check_geq_0(dpp.L_eig_vals)
        return True

    # If DPP defined through correlation kernel with parameter 'A_zono'
    # a priori you wish to use the zonotope approximate sampler
    elif dpp.A_zono is not None:
        warn('DPP defined via `A_zono`, apriori you want to use `sample_mcmc`, but you have called `sample_exact`')

        dpp.K_eig_vals = np.ones(dpp.A_zono.shape[0])
        dpp.eig_vecs, _ = qr(dpp.A_zono.T, mode='economic')
        return False

    elif dpp.eval_L is not None and dpp.X_data is not None:
        dpp.compute_L()
        return True

    else:
        raise ValueError('None of the available samplers could be used based on the current DPP representation. This should never happen, please consider rasing an issue on github at https://github.com/guilgautier/DPPy/issues')
