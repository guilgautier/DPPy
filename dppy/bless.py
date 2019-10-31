# MIT License
#
# Copyright (c) 2017 Laboratory for Computational and Statistical Learning
#
# authors: Daniele Calandriello, Luigi Carratino
# email:   daniele.calandriello@iit.it
# Website: http://lcsl.mit.edu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from collections import namedtuple
from .utils import check_random_state, stable_invert_root, get_progress_bar, evaluate_L_diagonal

CentersDictionary = namedtuple('CentersDictionary', ('idx', 'X', 'probs', 'lam', 'rls_oversample'))


def estimate_rls_bless(D, X, eval_L, lam_new):
    """Given a previously computed (eps, lambda)-accurate dictionary, it computes estimates
    of all RLS using the estimator from :cite:`CaLaVa17`
    :param CentersDictionary D: an (eps, lambda) accurate dictionary, see :ref:`bless`
    :param array_like X: samples whose RLS we must approximate
    :param callable eval_L: likelihood function
    :param float lam_new: lambda regularization to use for the RLSs
    :return: array of estimated RLS
    :rtype:
        array_like
    """

    diag_norm = evaluate_L_diagonal(eval_L, X)

    # (m x n) kernel matrix between samples in dictionary and dataset X
    K_DU = eval_L(D.X, X)

    # the estimator proposed in Calandriello et al. 2017 is
    # diag(XX' - XX'S(SX'XS + lam*I)^(-1)SXX')/lam
    # here for efficiency we collect an S inside the inverse and compute
    # diag(XX' - XX'(X'X + lam*S^(-2))^(-1)XX')/lam
    # note that in the second term we take care of dropping the rows/columns of X associated
    # with 0 entries in S
    U_DD, S_DD, _ = np.linalg.svd(eval_L(D.X, D.X) + lam_new * np.diag(D.probs))
    U_DD, S_root_inv_DD = stable_invert_root(U_DD, S_DD)

    E = S_root_inv_DD * U_DD.T

    # compute (X'X + lam*S^(-2))^(-1/2)XX'
    X_precond = E.dot(K_DU)

    # the diagonal entries of XX'(X'X + lam*S^(-2))^(-1)XX' are just the squared
    # ell-2 norm of the columns of (X'X + lam*S^(-2))^(-1/2)XX'
    rls_estimate = (diag_norm - np.square(X_precond, out=X_precond).sum(axis=0)) / lam_new

    if not np.all(rls_estimate  >= 0.):
        raise ValueError('Some estimated RLS is negative, this should never happen.'
                         ' Min prob: {}'.format(np.min(rls_estimate)))

    return rls_estimate


def reduce_lambda(X_data, eval_L, intermediate_dict_bless, lam_new, rng, rls_oversample_parameter=None):
    """Given a previously computed (eps, lambda)-accurate dictionary and a lambda' < lambda parameter,
     it constructs an (eps, lambda')-accurate dictionary using approximate RLS sampling.
    :param array_like X_data: dataset that we must approximate
    :param callable eval_L: likelihood function
    :param CentersDictionary intermediate_dict_bless: an (eps, lambda) accurate dictionary, see :ref:`bless`
    :param float lam_new: lambda regularization for the new dictionary
    :param RandomState rng: rng for sampling
    :param rls_oversample_parameter: Oversampling parameter to increase success probability, see :ref:`bless`
    :return: An (eps, lam_new)-accurate dictionary with high probability
    :rtype:
        CentersDictionary
     """

    n, d = X_data.shape

    if rls_oversample_parameter is None:
        rls_oversample_parameter = intermediate_dict_bless.rls_oversample

    red_ratio = intermediate_dict_bless.lam / lam_new

    if not red_ratio >= 1.0:
        raise ValueError(str(red_ratio))

    diag = np.asarray(evaluate_L_diagonal(eval_L, X_data))

    # compute upper confidence bound on RLS of each sample, overestimate (oversample) by a rls_oversample factor
    # to boost success probability at the expenses of a larger sample (dictionary)
    ucb = np.minimum(rls_oversample_parameter * diag / (diag + lam_new), 1.)

    U = np.asarray(rng.rand(n)) <= ucb
    u = U.sum()

    if not u > 0:
        raise ValueError('No point selected during uniform sampling step, try to increase rls_oversample_bless. '
                         'Expected number of points: {:.3f}'.format(n * ucb.mean()))

    X_U = X_data[U, :]

    rls_estimate = estimate_rls_bless(intermediate_dict_bless, X_U, eval_L, lam_new)

    # RLS should always be smaller than 1
    rls_estimate = np.minimum(rls_estimate, 1.0)

    # same as before, oversample by a rls_oversample factor
    probs = np.minimum(rls_oversample_parameter * rls_estimate, ucb[U]) / ucb[U]

    if not np.all(probs >= 0.0):
        raise ValueError('Some estimated probability is negative, this should never happen. '
                         'Min prob: {}'.format(np.min(probs)))

    deff_estimate = probs.sum() / rls_oversample_parameter

    if not rls_oversample_parameter * deff_estimate >= 1.0:
        raise ValueError('Estimated deff is smaller than 1, you might want to reconsider your kernel. '
                         'deff_estimate: {:.3f}'.format(rls_oversample_parameter * deff_estimate))

    selected = np.asarray(rng.rand(u)) <= probs

    s = selected.sum()

    if not s > 0:
        raise ValueError('No point selected during RLS sampling step, try to increase rls_oversample_bless. '
                         'Expected number of points (rls_oversample_bless*deff): {:.3f}'.format(np.sum(probs)))

    intermediate_dict_bless_new = CentersDictionary(idx=U.nonzero()[0][selected.nonzero()[0]],
                                                    X=X_U[selected, :],
                                                    probs=probs[selected],
                                                    lam=lam_new,
                                                    rls_oversample=rls_oversample_parameter)

    return intermediate_dict_bless_new


def bless(X_data, eval_L, lam_final, rls_oversample_param, random_state=None, nb_iter_bless=None, verbose=True):
    """Returns a (eps, lambda)-accurate dictionary of Nystrom centers sampled according to approximate RLS.

    Given data X, a similarity function, and its related similarity matrix similarity_function(X, X),
    an (eps, lambda)-accurate dictionary approximates all principal components of the similarity matrix
    with a singular value larger than lambda, up to a (1+eps) multiplicative error.

    The algorithm is introduced and analyzed in :cite:`RuCaCaRo18`, for a more formal
    definition of (eps, lambda)-accuracy and other potential uses see :cite:`CaLaVa17`.

    :param array_like X_data: input data, as an ndarray-like (n x m) object

    :param callable eval_L: likelihood function between points.
        If L is the associated likelihood matrix, it must satisfy the interface
        eval_L(X_1) = K(X_1, X_1)
        eval_L(X_1, X_2) = K(X_1, X_2)
        This interface is inspired by scikit-learn's implementation of kernel functions in Gaussian Processes.
        Any of the kernels provided by sklearn (e.g. sklearn.gaussian_process.kernels.RBF or
        sklearn.gaussian_process.kernels.PairwiseKernel) should work out of the box.

    :param float lam_final: final lambda (i.e. as in (eps, lambda)-accuracy) desired.
        Roughly, the final dictionary will approximate all principal components with a singular value
        larger than lam_final, and therefore smaller lam_final creates larger, more accurate dictionaries.

    :param int rls_oversample_param: Oversampling parameter used during BLESS's step of random RLS sampling.
        The rls_oversample >= 1 parameter is used to increase the sampling probabilities and sample size by a
        rls_oversample factor. This linearly increases the size of the output dictionary, making the algorithm
        less memory and time efficient, but reduces variance and the negative effects of randomness on the accuracy
        of the algorithm. Empirically, a small factor rls_oversample = [2,10] seems to work.
        It is suggested to start with a small number and increase if the algorithm fails to terminate or is inaccurate.

    :param random_state: Random number generator (RNG) used for the algorithm.
        By default, if random_state is not provided or is None, a numpy's RandomState with default seeding is used.
        If a numpy's RandomState is passed, it is used as RNG. If an int is passed, it is used to seed a RandomState
    :type random_state:
        np.random.RandomState or int or None, default None

    :param nb_iter_bless: number of iterations, defaults to log(n) if None
    :type nb_iter_bless:
        int or None, default None

    :param bool verbose: Controls verbosity of debug output, including progress bars.
        The progress bar reports:
        - lam: lambda value of the current iteration
        - m: current size of the dictionary (number of centers contained)
        - m_expected: expected size of the dictionary before sampling
        - probs_dist: (mean, max, min) of the approximate RLSs at the current iteration

    :return: An (eps, lambda)-accurate dictionary centers_dict (with high probability).
        If centers_dict contains m entries then the output fields are as follow

        centers_dict.idx`: the indices of the m selected samples in the input dataset `X`
        centers_dict.X': the (m x d) numpy.ndarray containing the selected samples
        centers_dict.probs: the probabilities (i.e. approximate RLSs) used to sample the dictionary
        lam: the final lambda accuracy
        rls_oversample: the rls_oversample used to sample the dictionary, as a proxy for the `eps`-accuracy
    :rtype:
        CentersDictionary
    """

    n, d = X_data.shape

    nb_iter_bless = nb_iter_bless if nb_iter_bless is not None else np.ceil(np.log(n)).astype('int')

    diag_norm = np.asarray(evaluate_L_diagonal(eval_L, X_data))
    lam_init = n
    ucb_init = rls_oversample_param * diag_norm / (diag_norm + lam_init)
    while ucb_init.sum() <= 10:
        lam_init = lam_init / 1.25
        ucb_init = rls_oversample_param * diag_norm / (diag_norm + lam_init)

    rng = check_random_state(random_state)

    selected_init = rng.rand(n) <= ucb_init

    # force at least one sample to be selected
    selected_init[0] = 1

    intermediate_dict_bless = CentersDictionary(idx=selected_init.nonzero(),
                                                X=X_data[selected_init, :],
                                                probs=np.ones(np.sum(selected_init)) * ucb_init[selected_init],
                                                lam=lam_init,
                                                rls_oversample=rls_oversample_param)

    lam_sequence = list(np.geomspace(lam_final, lam_init, nb_iter_bless))

    # discard lam_init from the list, we already used it to initialize
    lam_sequence.pop()

    with get_progress_bar(total=len(lam_sequence), disable=not verbose) as t:
        while len(lam_sequence) > 0:
            lam_new = lam_sequence.pop()
            intermediate_dict_bless = reduce_lambda(X_data, eval_L, intermediate_dict_bless, lam_new, rng)
            t.set_postfix(lam=int(lam_new),
                          m=len(intermediate_dict_bless.probs),
                          m_expected=int(intermediate_dict_bless.probs.mean() * n),
                          probs_dist="({:.4}, {:.4}, {:.4})".format(intermediate_dict_bless.probs.mean(),
                                                                    intermediate_dict_bless.probs.max(),
                                                                    intermediate_dict_bless.probs.min()))
            t.update()

    return intermediate_dict_bless
