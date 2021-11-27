import numpy as np
import scipy.linalg as la

from dppy.finite.sampling.projection_eigen_samplers import (
    projection_eigen_sampler_GS,
    projection_eigen_sampler_GS_bis,
    projection_eigen_sampler_KuTa12,
)
from dppy.utils import check_random_state, elementary_symmetric_polynomials


def spectral_sampler(dpp, random_state=None, **params):
    assert dpp.hermitian
    compute_spectral_sampler_parameters(dpp)
    return do_spectral_sampler(dpp, random_state, **params)


def do_spectral_sampler(dpp, random_state=None, **params):
    eig_vals, eig_vecs = dpp.K_eig_vals, dpp.eig_vecs
    V = select_eigen_vectors(eig_vals, eig_vecs, random_state=random_state)
    sampler = select_projection_eigen_sampler(params.get("method"))
    return sampler(V, size=params.get("size"), random_state=random_state)


def select_projection_eigen_sampler(method):
    samplers = {
        "GS": projection_eigen_sampler_GS,
        "GS_bis": projection_eigen_sampler_GS_bis,
        "KuTa12": projection_eigen_sampler_KuTa12,
    }
    default = samplers["GS"]
    return samplers.get(method, default)


def compute_spectral_sampler_parameters(dpp):
    """Compute eigenvalues and eigenvectors of correlation kernel K from various parametrizations of ``dpp``

    :param dpp: ``FiniteDPP`` object
    :type dpp: FiniteDPP
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

    if dpp.K is not None:  # 0 <= K <= I
        eig_vals, dpp.eig_vecs = la.eigh(dpp.K)
        np.clip(eig_vals, 0.0, 1.0, out=eig_vals)
        dpp.K_eig_vals = eig_vals
        return False

    if dpp.L_dual is not None:
        # L_dual = Phi * Phi.T = W Theta W.T
        # L = Phi.T Phi = V Gamma V
        # then Gamma = Theta and V = Phi.T W Theta^{-1/2}
        eig_vals, eig_vecs = la.eigh(dpp.L_dual)
        np.fmax(eig_vals, 0.0, out=eig_vals)
        dpp.L_eig_vals = eig_vals
        dpp.eig_vecs = dpp.L_gram_factor.T.dot(eig_vecs / np.sqrt(eig_vals))
        return True

    if dpp.L is not None:
        eig_vals, dpp.eig_vecs = la.eigh(dpp.L)
        np.fmax(eig_vals, 0.0, out=eig_vals)
        dpp.L_eig_vals = eig_vals
        return True

    if dpp.A_zono is not None:  # K = A.T (A A.T)^-1 A (orthogonal projection)
        A = dpp.A_zono
        dpp.K_eig_vals = np.ones(len(A), dtype=float)
        dpp.eig_vecs, _ = la.qr(A.T, mode="economic")
        return False

    if dpp.eval_L is not None and dpp.X_data is not None:
        dpp.compute_L()
        return True

    raise ValueError(
        "Failed to compute spectral sampler parameters (K eigenvalues and eigenvectors). This should never happen, please consider rasing an issue on github at https://github.com/guilgautier/DPPy/issues"
    )


# Phase 1
def select_eigen_vectors(bernoulli_params, eig_vecs, random_state=None):
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


def k_dpp_eig_vecs_selector(eig_vals, eig_vecs, size, esp=None, random_state=None):
    """Select columns of ``eig_vecs`` by sampling Bernoulli variables with parameters derived from the computation of elementary symmetric polynomials ``esp`` of order 0 to ``size`` evaluated in ``eig_vals``.
    This corresponds to :cite:`KuTa12` Algorithm 8.

    :param eig_vals:
        Collection of eigenvalues (assumed non-negetive)
    :type eig_vals:
        array_like

    :param eig_vecs:
        Matrix of eigenvectors stored columnwise
    :type eig_vecs:
        array_like

    :param size:
        Number of eigenvectors to be selected
    :type size:
        int

    :param esp:
        Computation of the elementary symmetric polynomials previously evaluated in ``eig_vals`` and returned by :py:func:`elementary_symmetric_polynomials <elementary_symmetric_polynomials>`, default to None.
    :type esp:
        array_like

    :return:
        Selected eigenvectors
    :rtype:
        array_like

    .. seealso::

        - :cite:`KuTa12` Algorithm 8
        - :func:`elementary_symmetric_polynomials <elementary_symmetric_polynomials>`
    """

    rng = check_random_state(random_state)

    # Size of: ground set / sample
    N, k = eig_vecs.shape[0], size

    # as in np.linalg.matrix_rank
    tol = np.max(eig_vals) * N * np.finfo(float).eps
    rank = np.count_nonzero(eig_vals > tol)
    if k > rank:
        raise ValueError("size k={} > rank={}".format(k, rank))

    if esp is None:
        esp = elementary_symmetric_polynomials(eig_vals, k)

    mask = np.zeros(k, dtype=int)
    for n in range(eig_vals.size, 0, -1):
        if rng.rand() < eig_vals[n - 1] * esp[k - 1, n - 1] / esp[k, n]:
            k -= 1
            mask[k] = n - 1
            if k == 0:
                break

    return eig_vecs[:, mask]
