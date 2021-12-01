import numpy as np
import scipy.linalg as la

from dppy.finite.exact_samplers.projection_eigen_samplers import (
    select_sampler_eigen_projection,
)
from dppy.utils import check_random_state


def spectral_sampler(dpp, random_state=None, **params):
    assert dpp.hermitian
    compute_spectral_sampler_parameters_dpp(dpp)
    return do_spectral_sampler_dpp(dpp, random_state, **params)


def do_spectral_sampler_dpp(dpp, random_state=None, **params):
    rng = check_random_state(random_state)
    eig_vals, eig_vecs = dpp.K_eig_vals, dpp.eig_vecs
    V = select_eigen_vectors_dpp(eig_vals, eig_vecs, random_state=rng)
    sampler = select_sampler_eigen_projection(params.get("mode"))
    return sampler(V, size=params.get("size"), random_state=rng)


def compute_spectral_sampler_parameters_dpp(dpp):
    """Compute eigenvalues and eigenvectors of correlation kernel K from various parametrizations of ``dpp``

    :param dpp: ``FiniteDPP`` object
    :type dpp: FiniteDPP
    """
    while compute_spectral_sampler_parameters_dpp_step(dpp):
        pass


def compute_spectral_sampler_parameters_dpp_step(dpp):
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

    if dpp.K is not None:  # 0 <= K <= I
        eig_vals, dpp.eig_vecs = la.eigh(dpp.K)
        np.clip(eig_vals, 0.0, 1.0, out=eig_vals)
        dpp.K_eig_vals = eig_vals
        return False

    if dpp.L_gram_factor is not None:
        # L_dual = Phi * Phi.T = W Theta W.T
        # L = Phi.T Phi = V Gamma V
        # then Gamma = Theta and V = Phi.T W Theta^{-1/2}
        Phi = dpp.L_gram_factor
        d, N = Phi.shape
        if d >= N:
            dpp.L = Phi.T.dot(Phi)
            return True

        L_dual = Phi.dot(Phi.T)
        eig_vals, eig_vecs = la.eigh(L_dual)
        np.fmax(eig_vals, 0.0, out=eig_vals)
        dpp.L_eig_vals = eig_vals
        dpp.eig_vecs = Phi.T.dot(eig_vecs / np.sqrt(eig_vals))
        return True

    if dpp.L is not None:
        eig_vals, dpp.eig_vecs = la.eigh(dpp.L)
        np.fmax(eig_vals, 0.0, out=eig_vals)
        dpp.L_eig_vals = eig_vals
        return True

    if dpp.A_zono is not None:  # K = A.T (A A.T)^-1 A (orthogonal projection)
        A = dpp.A_zono
        dpp.K_eig_vals = np.ones(len(A), dtype=float)
        dpp.eig_vecs, *_ = la.qr(A.T, mode="economic")
        return False

    if dpp.eval_L is not None and dpp.X_data is not None:
        dpp.compute_L()
        return True

    raise ValueError(
        "Failed to compute spectral sampler parameters (K eigenvalues and eigenvectors). This should never happen, please consider rasing an issue on github at https://github.com/guilgautier/DPPy/issues"
    )


# Phase 1
def select_eigen_vectors_dpp(bernoulli_params, eig_vecs, random_state=None):
    """Select columns of ``eig_vecs`` by sampling Bernoulli variables with parameters ``bernoulli_params``.

    :param bernoulli_params:
        Parameters of Bernoulli variables
    :type bernoulli_params:
        array_like, shape (r,)

    :param eig_vecs:
        Eigenvectors, stored as columns of a 2d array
    :type eig_vecs:
        array_like, shape (N, r)

    :return:
        Selected eigenvectors
    :rtype:
        array_like

    .. seealso::

        - :func:`dpp_sampler_eig <dpp_sampler_eig>`
    """
    rng = check_random_state(random_state)
    mask = rng.rand(bernoulli_params.size) < bernoulli_params
    return eig_vecs[:, mask]
