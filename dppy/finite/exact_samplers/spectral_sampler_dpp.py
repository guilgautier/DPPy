import numpy as np
import scipy.linalg as la

from dppy.finite.exact_samplers.projection_eigen_samplers import (
    select_sampler_eigen_projection,
)
from dppy.utils import check_random_state


def spectral_sampler(dpp, random_state=None, **params):
    r"""Generate an exact sample from an hermitian ``dpp`` using the :ref:`spectral method <finite_dpps_exact_sampling_spectral_method>`.

    The precomputation cost of generating the first sample involves computing the eigenvalues and eigenvectors of the likelihood kernel :math:`\mathbf{K}` from the current parametrization of ``dpp`` and stored in the ``dpp.K_eig_vals`` and ``dpp.eig_vecs`` attributes.

    :param dpp:
        Finite DPP
    :type dpp:
        :py:class:`~dppy.finite.dpp.FiniteDPP`

    :param random_state:
        random number generator or seed, defaults to None
    :type random_state:
        optional

    Keyword arguments:
        - mode (str): select the variant of the sampler used in the second step which boils down to sampling from a projection DPP, see :py:func:`~dppy.finite.exact_samplers.projection_eigen_samplers.select_sampler_eigen_projection`

    :return: sample
    :rtype: list
    """
    assert dpp.hermitian
    compute_spectral_sampler_parameters_dpp(dpp)
    return do_spectral_sampler_dpp(dpp, random_state, **params)


def do_spectral_sampler_dpp(dpp, random_state=None, **params):
    """Perform the main steps of the :ref:`spectral method <finite_dpps_exact_sampling_spectral_method>` to generate an exact sample from ``dpp``.

    :param dpp: Finite DPP
    :type dpp: :py:class:`~dppy.finite.dpp.FiniteDPP`

    :param random_state: random number generator or seed, defaults to None
    :type random_state: optional

    :return: sample
    :rtype: list
    """
    rng = check_random_state(random_state)
    eig_vals, eig_vecs = dpp.K_eig_vals, dpp.eig_vecs
    V = select_eigen_vectors_dpp(eig_vals, eig_vecs, random_state=rng)
    sampler = select_sampler_eigen_projection(params.get("mode"))
    return sampler(V, random_state=rng)


def compute_spectral_sampler_parameters_dpp(dpp):
    r"""Compute eigenvalues and eigenvectors of correlation kernel :math:`\mathbf{K}` from the current parametrization of ``dpp``. These values are stored in the ``dpp.K_eig_vals`` and ``dpp.eig_vecs`` attributes.

    :param dpp: Finite hermitian DPP
    :type dpp: :py:class:`~dppy.finite.dpp.FiniteDPP`
    """
    while compute_spectral_sampler_parameters_dpp_step(dpp):
        pass


def compute_spectral_sampler_parameters_dpp_step(dpp):
    r"""Compute eigenvalues and eigenvectors of correlation kernel :math:`\mathbf{K}` from the current parametrization of ``dpp``. These values are stored in the ``dpp.K_eig_vals`` and ``dpp.eig_vecs`` attributes.

    This corresponds to a sort of fixed point algorithm to compute eigenvalues and eigenvectors.

    :return: ``False`` if the right parameters are indeed computed, ``True`` if extra computations are required.
    :rtype: bool
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
