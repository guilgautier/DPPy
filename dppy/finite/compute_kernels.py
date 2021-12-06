from warnings import warn

import numpy as np
import scipy.linalg as la


def compute_correlation_kernel(dpp):
    while _compute_correlation_kernel_step(dpp):
        continue
    return dpp.K


def _compute_correlation_kernel_step(dpp):
    """Return
    ``False`` if the right parameters are indeed computed
    ``True`` if extra computations are required
    """
    if dpp.K is not None:
        return False

    if dpp.K_eig_vals is not None:
        lambda_, U = dpp.K_eig_vals, dpp.eig_vecs
        dpp.K = (U * lambda_).dot(U.T)
        return False

    if dpp.A_zono is not None:
        rank = dpp.A_zono.shape[0]
        dpp.K_eig_vals = np.ones(rank)
        dpp.eig_vecs, *_ = la.qr(dpp.A_zono.T, mode="economic")
        return True

    if dpp.L_eig_vals is not None:
        gamma = dpp.L_eig_vals
        dpp.K_eig_vals = gamma / (1.0 + gamma)
        return True

    if dpp.L is not None:
        # todo separate (non)hermitian cases K = L(L+I)-1
        dpp.L_eig_vals, dpp.eig_vecs = la.eigh(dpp.L)
        return True

    compute_likelihood_kernel(dpp)
    return True


def compute_likelihood_kernel(dpp):
    while _compute_likelihood_kernel_step(dpp):
        continue
    return dpp.L


def _compute_likelihood_kernel_step(dpp):

    """Return
    ``False`` if the right parameters are indeed computed
    ``True`` if extra computations are required
    """
    if dpp.L is not None:
        return False

    if dpp.projection and dpp.kernel_type == "correlation":
        raise ValueError(
            "Likelihood kernel L cannot be computed as L = K (I - K)^-1 since projection kernel K has some eigenvalues equal 1"
        )

    if dpp.L_eig_vals is not None:
        gamma, V = dpp.L_eig_vals, dpp.eig_vecs
        dpp.L = (V * gamma).dot(V.T)
        return False

    if dpp.L_gram_factor is not None:
        Phi = dpp.L_gram_factor
        dpp.L = Phi.T.dot(Phi)
        return False

    if dpp.eval_L is not None:
        warn_print = [
            "Weird setting:",
            "FiniteDPP(.., **{'L_eval_X_data': (eval_L, X_data)})",
            "When using 'L_eval_X_data', you are a priori working with a big `X_data` and not willing to compute the full likelihood kernel L",
            "Right now, the computation of L=eval_L(X_data) is performed but might be very expensive, this is at your own risk!",
            "You might also use FiniteDPP(.., **{'L': eval_L(X_data)})",
        ]
        warn("\n".join(warn_print))
        dpp.L = dpp.eval_L(dpp.X_data)
        return False

    if dpp.K_eig_vals is not None:
        try:  # to compute eigenvalues of kernel L = K(I-K)^-1
            np.seterr(divide="raise")
            dpp.L_eig_vals = dpp.K_eig_vals / (1.0 - dpp.K_eig_vals)
            return True
        except FloatingPointError:
            err_print = [
                "Eigenvalues of the likelihood L kernel cannot be computed as eig_L = eig_K / (1 - eig_K).",
                "K kernel has some eig_K very close to 1. Hint: `K` kernel might be a projection",
            ]
            raise FloatingPointError("\n".join(err_print))

    if dpp.K is not None:
        # todo separate (non)hermitian cases L = K(K-I)-1
        eig_vals, dpp.eig_vecs = la.eigh(dpp.K)
        np.clip(eig_vals, 0.0, 1.0, out=eig_vals)  # 0 <= K <= I
        dpp.K_eig_vals = eig_vals
        return True

    compute_correlation_kernel(dpp)
    return True
