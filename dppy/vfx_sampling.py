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

from dppy.bless import bless, CentersDictionary, estimate_rls_bless
from dppy.utils import stable_invert_root, evaluate_L_diagonal, get_progress_bar, check_random_state
import numpy as np
from scipy.optimize import brentq
from collections import namedtuple


def compute_nystrom_dict(X_data, eval_L, rls_oversample_bless, rls_oversample_dppvfx, rng, nb_iter_bless=None, verbose=True):
    """ Computes the initial dictionary necessary for the algorithm. Internally invoke BLESS.

    :param array_like X_data: dataset that we must approximate
    :param callable eval_L: likelihood function
    :param float rls_oversample_bless: see :func:`vfx_sampling_precompute_constants`
    :param float rls_oversample_dppvfx: see :func:`vfx_sampling_precompute_constants`
    :param RandomState rng: random source used for sampling
    :param int nb_iter_bless:  iterations for BLESS, if None it is set to log(n)
    :param bool verbose: controls verbosity of debug output, including progress bars.
        the progress bar reports:
        - lam: lambda value of the current iteration
        - m: current size of the dictionary (number of centers contained)
        - m_expected: expected size of the dictionary before sampling
        - probs_dist: (mean, max, min) of the approximate rlss at the current iteration
    :return: an (eps, lambda)-accurate dictionary for Nystrom approximation
    :rtype:
        CentersDictionary

    """
    n, _ = X_data.shape

    # Phase 1: compute initial dictionary D_bless with small rls_oversample_bless
    # D_bless is used only to estimate all RLS

    dict_bless = bless(X_data, eval_L, 1.0, rls_oversample_bless, rng, nb_iter_bless=nb_iter_bless, verbose=verbose)

    bless_rls_estimate = estimate_rls_bless(dict_bless, X_data, eval_L, 1.0)

    # Phase 2: use estimate RLS to sample the dict_dppvfx dictionary, i.e. the one used to construct A
    # here theory says that to have high acceptance probability we need the oversampling factor to be ~deff^2
    # but even with constant oversampling factor we seem to accept fast

    probs = np.minimum(rls_oversample_dppvfx * bless_rls_estimate, 1.0)
    if not np.all(probs >= 0.0):
        raise ValueError('Some estimated RLS is negative, this should never happen. Min prob: {}'.format(np.min(probs)))

    selected = rng.rand(n) <= probs
    s = selected.sum()

    if not s > 0:
        raise ValueError('No point selected during RLS sampling step, try to increase rls_oversample_bless. '
                         'Expected number of points: {:.3f}'.format(probs.sum()))

    dict_dppvfx = CentersDictionary(idx=selected.nonzero()[0],
                                    X=X_data[selected, :],
                                    probs=probs[selected],
                                    lam=1,
                                    rls_oversample=rls_oversample_dppvfx)

    return dict_dppvfx


def estimate_rls_from_embedded_points(eigvec, eigvals, B_bar_T, diag_L, diag_L_hat, alpha_star):
    """ Given embedded points, and a decomposition of embedded covariance matrix, estimate RLS.
    Note that this is a different estimator than the one used in BLESS (i.e. :func:`dppy.bless.estimate_rls_bless`),
    which we use here for efficiency because we can recycle already embedded points and eigen-decomposition.

    :param array_like eigvec: eigenvectors of I_A_mm = B_bar_T*B_bar_T.T + lam I, see :func:`vfx_sampling_precompute_constants`
    :param array_like eigvals: eigenvalues of I_A_mm = B_bar_T*B_bar_T.T + lam I, see :func:`vfx_sampling_precompute_constants`
    :param array_like B_bar_T: (m x n) transposed matrix of n points embedded using a dictionary with m centers
    :param array_like diag_L: diagonal of L
    :param array_like diag_L_hat: diagonal of L_hat, the Nystrom approximation of L
    :param float alpha_star: a rescaling factor used to adjust the expected size of the DPP sample
    :return: RLS estimates for all rows in B_bar_T
    :rtype:
        array_like
    """

    U_DD, S_root_inv_DD = stable_invert_root(eigvec, np.maximum(eigvals, 0))

    E = S_root_inv_DD * U_DD.T

    X_precond = E.dot(B_bar_T)

    rls_estimate =\
        diag_L * alpha_star\
        - diag_L_hat * alpha_star\
        + np.square(X_precond, out=X_precond).sum(axis=0)

    return rls_estimate


_IntermediateSampleInfo =\
    namedtuple('_IntermediateSampleInfo',
               ['alpha_star', 'logdet_I_A', 'q', 's', 'z', 'rls_estimate'])


def vfx_sampling_precompute_constants(X_data,
                                      eval_L,
                                      rng,
                                      desired_expected_size=None,
                                      rls_oversample_dppvfx=4.0,
                                      rls_oversample_bless=4.0,
                                      nb_iter_bless=None,
                                      verbose=True):
    """Pre-compute quantities necessary for the vfx rejection sampling loop, such as the inner Nystrom approximation,
    and the RLS of all elements in L.

        :param array_like X_data: dataset such that L = eval_L(X_data), out of which we aresampling objects according
        to a DPP
        :param callable eval_L: likelihood function. Given two sets of n points X and m points Y, eval_L(X, Y) should
        compute the (n x m) matrix containing the likelihood between points. The function should also
        accept a single argument X and return eval_L(X) = eval_L(X, X).
        As an example, see the implementation of any of the kernels provided by scikit-learn
        (e.g. sklearn.gaussian_process.kernels.PairwiseKernel).
        :param RandomState rng: random source used for sampling
        :param desired_expected_size: desired expected sample size for the DPP. If None, use the natural DPP expected
        sample size. The vfx sampling algorithm can approximately adjust the expected sample size of the DPP by
        rescaling the L matrix with a scalar alpha_star <= 1. Adjusting the expected sample size can be useful to
        control downstream complexity, and it is necessary to improve the probability of drawing a sample with
        exactly k elements when using vfx for k-DPP sampling. Currently only reducing the sample size is supported,
        and the sampler will return an exception if the DPP sample has already a natural expected size
        smaller than desired_expected_size.
        :type desired_expected_size:
            float or None, default None
        :param rls_oversample_dppvfx: Oversampling parameter used to construct dppvfx's internal Nystrom approximation.
        The rls_oversample_dppvfx >= 1 parameter is used to increase the rank of the approximation by
        a rls_oversample_dppvfx factor. This makes each rejection round slower and more memory intensive,
        but reduces variance and the number of rounds of rejections, so the actual runtime might increase or decrease.
        Empirically, a small factor rls_oversample_dppvfx = [2,10] seems to work. It is suggested to start with
        a small number and increase if the algorithm fails to terminate.
        :type rls_oversample_dppvfx:
            float, default 4.0
        :param rls_oversample_bless: Oversampling parameter used during bless's internal Nystrom approximation.
        Note that this is a different Nystrom approximation than the one related to :func:`rls_oversample_dppvfx`,
        and can be tuned separately.
        The rls_oversample_bless >= 1 parameter is used to increase the rank of the approximation by
        a rls_oversample_bless factor. This makes the one-time pre-processing slower and more memory intensive,
        but reduces variance and the number of rounds of rejections, so the actual runtime might increase or decrease.
        Empirically, a small factor rls_oversample_bless = [2,10] seems to work. It is suggested to start with
        a small number and increase if the algorithm fails to terminate or is not accurate.
        :type rls_oversample_bless:
            float, default 4.0
        :param int nb_iter_bless:  iterations for BLESS, if None it is set to log(n)
        :type nb_iter_bless:
            int or None, default None
        :param bool verbose: controls verbosity of debug output, including progress bars.
        The progress bar reports the inner execution of the bless algorithm, showing:
            - lam: lambda value of the current iteration
            - m: current size of the dictionary (number of centers contained)
            - m_expected: expected size of the dictionary before sampling
            - probs_dist: (mean, max, min) of the approximate rlss at the current iteration

        :return: Pre-computed information necessary for the vfx rejection sampling loop with fields
        - result.alpha_star: appropriate rescaling such that the expected sample size of DPP(alpha_star * L) is equal
        to a user-indicated constant desired_expected_size, or 1.0 if no such constant was specified by the user.
        - result.logdet_I_A: log determinant of the Nystrom approximation of L + I
        - result.q: placeholder q constant used for vfx sampling, to be replaced by the user before the sampling loop
        - result.s and result.z: approximations of the expected sample size of DPP(alpha_star * L) to be used in
        the sampling loop. For more details see [DeCaVa19]
        - result.rls_estimate: approximations of the RLS of all elements in X (i.e. in L)
        :rtype: _IntermediateSampleInfo

    """
    diag_L = evaluate_L_diagonal(eval_L, X_data)
    trace_L = diag_L.sum()

    # Phase 1: get all necessary prerequisite for preprocessing
    # construct D_A dictionary used to construct A and L_hat
    D_A = compute_nystrom_dict(X_data,
                               eval_L,
                               rls_oversample_bless,
                               rls_oversample_dppvfx,
                               rng,
                               nb_iter_bless=nb_iter_bless,
                               verbose=verbose)

    # Phase 2: pre-compute L_hat, B_bar, l_i, det(I + L_hat), etc.
    U_DD, S_DD, _ = np.linalg.svd(eval_L(D_A.X, D_A.X))
    U_DD, S_root_inv_DD = stable_invert_root(U_DD, S_DD)
    m = U_DD.shape[1]

    E = S_root_inv_DD * U_DD.T

    # The _T indicates that B_bar_T is the transpose of B_bar,
    # we keep it that way for efficiency reasons
    B_bar_T = E.dot(eval_L(D_A.X, X_data))
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

    natural_expected_size = trace_L - trace_L_hat + np.sum((eigvals - 1.0) / eigvals)

    if not natural_expected_size >= 0.0:
        raise ValueError('natural_expected_size is negative, this should never happen. '
                         'natural_expected_size: {}'.format(natural_expected_size))

    # s might naturally be too large, but we can rescale L to shrink it
    # if we rescale alpha * L by a constant alpha,
    # s is now trace(alpha * L - alpha * L_hat + L_hat(L_hat + I / alpha)^-1)
    if desired_expected_size is None:
        alpha_star = 1.0
    elif natural_expected_size <= desired_expected_size:
        raise ValueError('The expected sample size is smaller than the desired sample size or k (if sampling from'
                         'a k-DPP).\n'
                         'This is unusual (i.e. you are trying to select more than the overall amount of diversity '
                         'in your set.\n'
                         'Increasing the expected sample size is currently not supported (only decreasing).\n'
                         'Please consider decreasing your k={} or changing L.'
                         ' Estimated mean cardinality: {}'.format(desired_expected_size,
                                                                  natural_expected_size))
    else:
        # since this is monotone in alpha, we can simply use Brent's algorithm (bisection + tricks)
        # it is a root finding algorithm so we must create a function with a root in desired_expected_size
        def temp_func_with_root_in_desired_expected_size(x):
            return (
                    x * trace_L
                    - x * trace_L_hat
                    + np.sum((eigvals - 1.0) / (eigvals - 1.0 + 1.0 / x))
                    - desired_expected_size
                   )

        alpha_star, opt_result = brentq(temp_func_with_root_in_desired_expected_size,
                                        a=10.0 * np.finfo(np.float).eps,
                                        b=1.0,
                                        full_output=True)

        if not opt_result.converged:
            raise ValueError('Could not find an appropriate rescaling for desired_expected_size.'
                             '(Flag, Iter, Root): {}'.format((opt_result.flag,
                                                              opt_result.iterations,
                                                              opt_result.root)))

    # adjust from I + A to I / alpha_star + A
    I_A_mm[np.diag_indices(m)] += 1.0 / alpha_star - 1.0
    eigvals += 1.0 / alpha_star - 1.0
    rls_estimate = estimate_rls_from_embedded_points(eigvec,
                                                     eigvals,
                                                     B_bar_T,
                                                     diag_L,
                                                     diag_L_hat,
                                                     alpha_star)

    if not np.all(rls_estimate >= 0.0):
        raise ValueError('Some estimate l_i is negative, this should never happen. '
                         'Minimum l_i: {}'.format(np.min(rls_estimate)))

    # s is simply the sum of l_i, and it is our proxy for the expected sample size. If desired_expected_size
    # is set, s should be very close to it
    s = np.sum(rls_estimate)
    if not s >= 0.0:
        raise ValueError('s is negative, this should never happen. '
                         's: {}'.format(s))

    # we need to compute z and logDet(I + L_hat)
    z = np.sum((eigvals - 1/alpha_star) / eigvals)

    # we need logdet(I + alpha * A) and we have eigvals(I / alpha_star + A) we can adjust using sum of logs
    logdet_I_A = np.sum(np.log(alpha_star * eigvals))

    if not logdet_I_A >= 0.0:
        raise ValueError('logdet_I_A is negative, this should never happen. '
                         's: {}'.format(logdet_I_A))

    result = _IntermediateSampleInfo(alpha_star=alpha_star,
                                     logdet_I_A=logdet_I_A,
                                     q=-1,
                                     s=s,
                                     z=z,
                                     rls_estimate=rls_estimate)

    return result


def vfx_sampling_do_sampling_loop(X_data, eval_L, intermediate_sample_info, rng, max_iter=1000, verbose=True, **kwargs):
    """Given pre-computed information, run a rejection sampling loop to generate DPP samples.
        :param array_like X_data: dataset such that L = eval_L(X_data), out of which we are sampling objects
        according to a DPP
        :param callable eval_L: likelihood function. Given two sets of n points X and m points Y, eval_L(X, Y) should
        compute the (n x m) matrix containing the likelihood between points. The function should also
        accept a single argument X and return eval_L(X) = eval_L(X, X).
        As an example, see the implementation of any of the kernels provided by scikit-learn
        (e.g. sklearn.gaussian_process.kernels.PairwiseKernel).
        :param _IntermediateSampleInfo: Pre-computed information necessary for the vfx rejection sampling loop,
        as returned by :func:`vfx_sampling_precompute_constants`
        :param RandomState rng: random source used for sampling
        :param max_iter:
        :type max_iter:
            int, default 1000
        :param bool verbose: controls verbosity of debug output, including progress bars.
        The progress bar reports the execution of the rejection sampling loop, showing:
            - acc_thresh: latest computed probability of acceptance
            - rej_iter: iteration of the rejection sampling loop (i.e. rejections so far)
        :type verbose:
            bool, default True
        :param dict kwargs: we add a unused catch all kwargs argument to make sure that the user can pass the
        same set of parameters to both vfx_sampling_precompute_constants and vfx_sampling_do_sampling_loop. This
        way if there is any spurious non-shared parameter (e.g. rls_oversample_bless) we simply ignore it.
        :return: Sample from a DPP (as a list) and number of rejections as int
        :rtype: tuple(list, int)
    """
    #TODO: taking as input a catch-all kwargs can be misleading for the user. e.g. if there is a typo in a paremater
    # it will silently ignore it and use the default instead

    n, d = X_data.shape

    # rename it to pre-computed state for shortness
    pc_state = intermediate_sample_info

    # Phase 3: rejection sampling loop

    with get_progress_bar(disable=not verbose) as prog_bar:
        for rej_iter in range(max_iter):
            # sample t
            lam = np.ceil(pc_state.q * np.exp(pc_state.s / pc_state.q))
            t = rng.poisson(lam=lam.astype('int'))

            # sample sigma subset
            sigma = rng.choice(n, size=t, p=pc_state.rls_estimate / pc_state.s, replace=True)
            sigma_uniq, sigma_uniq_count = np.unique(sigma, return_counts=True)
            X_sigma_uniq = X_data[sigma_uniq, :]

            # compute log(Det(I + \tilda{L}_sigma)) = log(Det(I + W*L_sigma*W))
            # with W_ii = ( s / (q * l_i) )^1/2
            # this is done by computing
            # log(Det(I + W*L_sigma*W))
            # = log(Det(W * (W^-2 + L_sigma) * W))
            # = log(Det(W) * Det(W^-2 + L_sigma) * Det(W))
            # = log(Det(W)) + log(Det(W^-2 + L_sigma)) + log(Det(W))
            # = log(Det(W^2)) + log(Det(W^-2 + L_sigma))
            # = -log(Det(W^-2)) + log(Det(W^-2 + L_sigma))

            W_square_inv = pc_state.q * pc_state.rls_estimate[sigma_uniq] / (pc_state.s * sigma_uniq_count)

            I_L_sigma = (pc_state.alpha_star * eval_L(X_sigma_uniq, X_sigma_uniq)
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

            prog_bar.set_postfix(acc_thresh=np.exp(acc_thresh), rej_count=rej_iter)
            prog_bar.update()

            if accept:
                break
        else:
            raise ValueError('The vfx sampler reached the maximum number of rejections allowed '
                             'for the intermediate sample selection ({}), try to increase the q factor '
                             '(see q_func parameter) or the Nystrom approximation accuracy '
                             'see rls_oversample_* parameters).'.format(max_iter))

    # Phase 4: use L_tilda to perform exact DPP sampling
    # compute alpha_star * L_tilda = alpha_star * W*L_sigma*W
    W = (np.sqrt(pc_state.s * sigma_uniq_count)
         / np.sqrt(pc_state.q * pc_state.rls_estimate[sigma_uniq]).reshape(-1, 1))

    L_tilda = pc_state.alpha_star * W.T * eval_L(X_sigma_uniq, X_sigma_uniq) * W

    E, U = np.linalg.eigh(L_tilda)

    # this has to be here rather than at the top to avoid circular dependencies
    # TODO: maybe refactor to avoid this
    from dppy.finite_dpps import FiniteDPP
    DPP = FiniteDPP(kernel_type='likelihood', L_eig_dec=(E, U))
    S_tilda = np.array(DPP.sample_exact(random_state=rng), dtype=int)

    S = sigma_uniq[S_tilda].ravel().tolist()

    return S, rej_iter
