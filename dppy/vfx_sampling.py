# MIT License
#
# Copyright (c) 2017 Laboratory for Computational and Statistical Learning
#
# authors: Daniele Calandriello
# email:   daniele.calandriello@iit.it
# Website: http://lcsl.mit.edu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dppy.bless import bless, Dictionary, estimate_rls
from dppy.utils import stable_invert_root, evaluate_L_diagonal, get_progress_bar, check_random_state
import numpy as np
from dppy.finite_dpps import FiniteDPP
from scipy.optimize import brentq
from collections import namedtuple


def compute_nystrom_dict(X,
                         eval_L,
                         rls_oversample_bless,
                         rls_oversample_dppvfx,
                         rng,
                         H_bless=None):
    """ Computes the initial dictionary necessary for the algorithm. Internally invoke BLESS.

    :param X: points

    :param eval_L: likelihood function

    :param rls_oversample_bless: qbar for BLESS, should be >= log(n)^2, empirically works well even as a constant

    :param rls_oversample_dppvfx: qbar for dpp-fx, should be >= deff^2, empirically works well even as a constant

    :param rng:

    :param H_bless:  iterations for BLESS, if None it is set to log(n)

    :return:

    .. todo::

        - docstring: continue description of params and return, add types
    """
    n, _ = X.shape

    # Phase 1: compute initial dictionary D_bless with small rls_oversample_bless
    # D_bless is used only to estimate all RLS

    dict_bless = bless(X, eval_L, 1.0, rls_oversample_bless, rng, H=H_bless)

    bless_rls_estimate = estimate_rls(dict_bless, X, eval_L, 1)

    # Phase 2: use estimate RLS to sample the dict_dppvfx dictionary, i.e. the one used to construct A
    # here theory says that to have high acceptance probability we need the oversampling factor to be ~deff^2
    # but even with constant oversampling factor we seem to accept fast

    probs = np.minimum(rls_oversample_dppvfx * bless_rls_estimate, 1.0)
    if not np.all(probs >= 0.0):
        raise ValueError('Some estimated RLS is negative, this should never happen. Min prob: {}'.format(np.min(probs)))

    selected = rng.rand(n) <= probs
    s = selected.sum()

    if not s > 0:
        raise ValueError('No point selected during RLS sampling step, try to increase qbar. '
                         'Expected number of points: {:.3f}'.format(probs.sum()))

    dict_dppvfx = Dictionary(idx=selected.nonzero()[0],
                             X=X[selected, :],
                             probs=probs[selected],
                             lam=1,
                             qbar=rls_oversample_dppvfx)

    return dict_dppvfx


def estimate_rls_with_embedding(eigvec, eigvals,
                                B_bar_T, diag_L, diag_L_hat, alpha_star):
    """ Given embedded points, and a decomposition of embedded covariance matrix, estimate RLS.
    Note that this is a different estimator than the one used in BLESS (i.e. estimate_rls), which we use here for efficiency because we can recycle already embedded points and eigendecomp.
    .. todo::

        - docstring: description of params and return, add types
    """

    U_DD, S_root_inv_DD = stable_invert_root(eigvec, np.maximum(eigvals, 0))

    E = S_root_inv_DD * U_DD.T

    X_precond = E.dot(B_bar_T)

    rls_estimate =\
        diag_L * alpha_star\
        - diag_L_hat * alpha_star\
        + np.square(X_precond, out=X_precond).sum(axis=0)

    return rls_estimate


_PrecomputeState =\
    namedtuple('_PrecomputeState',
               ['alpha_star', 'logdet_I_A', 'q', 's', 'z', 'rls_estimate'])


def vfx_sampling_precompute_constants(X,
                                      eval_L,
                                      rng,
                                      desired_s=None,
                                      rls_oversample_dppvfx=4,
                                      rls_oversample_bless=4,
                                      H_bless=None):
    """
    .. todo::

        - docstring: description of params and return, add types
    """
    diag_L = evaluate_L_diagonal(eval_L, X)
    trace_L = diag_L.sum()

    # Phase 1: get all necessary prerequisite for preprocessing
    # construct D_A dictionary used to construct A and L_hat
    D_A = compute_nystrom_dict(X,
                               eval_L,
                               rls_oversample_bless,
                               rls_oversample_dppvfx,
                               rng,
                               H_bless=H_bless)

    # Phase 2: pre-compute L_hat, B_bar, l_i, det(I + L_hat), etc.
    U_DD, S_DD, _ = np.linalg.svd(eval_L(D_A.X, D_A.X))
    U_DD, S_root_inv_DD = stable_invert_root(U_DD, S_DD)
    m = U_DD.shape[1]

    E = S_root_inv_DD * U_DD.T

    # The _T indicates that B_bar_T is the transpose of B_bar,
    # we keep it that way for efficiency reasons
    B_bar_T = E.dot(eval_L(D_A.X, X))
    diag_L_hat = np.square(B_bar_T).sum(axis=0)
    trace_L_hat = diag_L_hat.sum()

    # While we have L_hat = B_bar_T.T * B_bar_T, we do not want to compute explicitly the (n x n) matrix
    # instead we reason in terms of B_bar_T * B_bar_T.T which is a (m x m) matrix. We call this matrix A_mm.
    # I_A_mm indicates I + A_mm (i.e. A_mm with identity added)
    I_A_mm = B_bar_T.dot(B_bar_T.T)
    I_A_mm[np.diag_indices(m)] += 1.0

    # we now need to compute the l_i estimates using L_hat, it is more efficient to do it in terms of
    # B_bar_T and I_A_mm
    # in particular, we will use the diag(L - L_hat + L_hat(L_hat + I)^-1) estimator
    # but we must first tune L to obtain a desired s
    # we can use the fact the the non-zero eigenvalues of I + L_hat and I_A_mm are equal
    eigvals, eigvec = np.linalg.eigh(I_A_mm)

    if not np.all(eigvals >= 0.0):
        raise ValueError('Some eigenvalues of L_hat are negative, this should never happen. '
                         'Minimum eig: {}'.format(np.min(eigvals)))

    natural_s = trace_L - trace_L_hat + np.sum((eigvals - 1.0) / eigvals)

    if not natural_s >= 0.0:
        raise ValueError('natural_s is negative, this should never happen. '
                         'natural_s: {}'.format(natural_s))

    # s might naturally be too large, but we can rescale L to shrink it
    # if we rescale alpha * L by a constant alpha < 1,
    # s is now trace(alpha * L - alpha * L_hat + L_hat(L_hat + I / alpha)^-1)
    if desired_s is None or natural_s <= desired_s:
        alpha_star = 1.0
    else:
        # since this is monotone in alpha, we can simply use Brent's algorithm (bisection + tricks)
        # it is a root finding algorithm so we must create a function with a root in desired_s
        def f_opt(x):
            """
            .. todo::

                - docstring: description of function, params and return, add types
            """
            return (
                    x * trace_L
                    - x * trace_L_hat
                    + np.sum((eigvals - 1.0) / (eigvals - 1.0 + 1.0 / x))
                    - desired_s
                   )

        alpha_star, opt_result = brentq(f_opt,
                                        a=10.0 * np.finfo(np.float).eps,
                                        b=1.0,
                                        full_output=True)

        if not opt_result.converged:
            raise ValueError('Could not find an appropriate rescaling for desired_s.'
                             '(Flag, Iter, Root): {}'.format((opt_result.flag,
                                                              opt_result.iterations,
                                                              opt_result.root)))

    # adjust from I + A to I / alpha_star + A
    I_A_mm[np.diag_indices(m)] += 1.0 / alpha_star - 1.0
    eigvals += 1.0 / alpha_star - 1.0
    rls_estimate = estimate_rls_with_embedding(eigvec,
                                               eigvals,
                                               B_bar_T,
                                               diag_L,
                                               diag_L_hat,
                                               alpha_star)

    if not np.all(rls_estimate >= 0.0):
        raise ValueError('Some estimate l_i is negative, this should never happen. '
                         'Minimum l_i: {}'.format(np.min(rls_estimate)))

    # s is simply the sum of l_i
    s = np.sum(rls_estimate)
    if not s >= 0.0:
        raise ValueError('s is negative, this should never happen. '
                         's: {}'.format(s))

    # we need to compute z and logDet(I + L_hat)
    z = np.sum((eigvals - alpha_star) / eigvals)

    # we need logdet(I + alpha * A) and we have eigvals(I / alpha_star + A) we can adjust using sum of logs
    logdet_I_A = np.sum(np.log(alpha_star * eigvals))

    if not logdet_I_A >= 0.0:
        raise ValueError('logdet_I_A is negative, this should never happen. '
                         's: {}'.format(logdet_I_A))

    result = _PrecomputeState(alpha_star=alpha_star,
                              logdet_I_A=logdet_I_A,
                              q=-1,
                              s=s,
                              z=z,
                              rls_estimate=rls_estimate)

    return result


def vfx_sampling_do_sampling_loop(X,
                                  eval_L,
                                  precomputed_state: _PrecomputeState,
                                  rng,
                                  verbose=True):
    """
    .. todo::

        - docstring: continue description of params and return, add types
        - put a maximal number of rejection steps and potentially replace while
    """
    n, d = X.shape

    pc_state = precomputed_state

    # Phase 3: rejection sampling loop
    accept = False

    # count how many times we reject for debug purpose
    rej_count = 0

    with get_progress_bar(disable=not verbose) as prog_bar:
        while not accept:
            rej_count = rej_count + 1

            # sample t
            lam = np.ceil(pc_state.q * np.exp(pc_state.s / pc_state.q))
            t = rng.poisson(lam=lam.astype('int'))

            # sample sigma subset
            sigma = rng.choice(n, size=t, p=pc_state.rls_estimate / pc_state.s)
            X_sigma = X[sigma, :]

            # compute log(Det(I + \tilda{L}_sigma)) = log(Det(I + W*L_sigma*W))
            # with W_ii = ( s / (q * l_i) )^1/2
            # this is done by computing
            # log(Det(I + W*L_sigma*W))
            # = log(Det(W * (W^-2 + L_sigma) * W))
            # = log(Det(W) * Det(W^-2 + L_sigma) * Det(W))
            # = log(Det(W)) + log(Det(W^-2 + L_sigma)) + log(Det(W))
            # = log(Det(W^2)) + log(Det(W^-2 + L_sigma))
            # = -log(Det(W^-2)) + log(Det(W^-2 + L_sigma))

            W_square_inv = pc_state.q * pc_state.rls_estimate[sigma] / pc_state.s

            I_L_sigma = (pc_state.alpha_star * eval_L(X_sigma, X_sigma)
                         + np.diag(W_square_inv))

            s_logdet, logdet_I_L_sigma = np.linalg.slogdet(I_L_sigma)
            if not s_logdet >= 0.0:
                raise ValueError('logdet_I_L_sigma is negative, this should never happen. '
                                 's: {}'.format(s_logdet))

            logdet_W_square_inv = np.sum(np.log(W_square_inv))

            acc_thresh = (pc_state.z
                          + logdet_I_L_sigma
                          - logdet_W_square_inv
                          - pc_state.logdet_I_A
                          - t * pc_state.s / pc_state.q).item()

            accept = np.log(rng.rand()) <= acc_thresh

            prog_bar.set_postfix(acc_thresh=acc_thresh,
                                 e_acc_thresh=np.exp(acc_thresh),
                                 t=t,
                                 rej_count=rej_count)
            prog_bar.update()

    # Phase 4: use L_tilda to perform exact DPP sampling
    # compute alpha_star * L_tilda = alpha_star * W*L_sigma*W
    W = (np.sqrt(pc_state.s)
         / np.sqrt(pc_state.q * pc_state.rls_estimate[sigma]).reshape(-1, 1))

    L_tilda = pc_state.alpha_star * W.T * eval_L(X_sigma, X_sigma) * W

    E, U = np.linalg.eigh(L_tilda)

    DPP = FiniteDPP(kernel_type='likelihood', L_eig_dec=(E, U))
    DPP.sample_exact(random_state=rng)

    S_tilda = np.array(DPP.list_of_samples)
    S = sigma[S_tilda].tolist()

    return S, rej_count
