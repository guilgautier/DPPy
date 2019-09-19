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
    tau = (diag_norm - np.square(X_precond, out=X_precond).sum(axis=0)) / lam_new

    if not np.all(tau >= 0.):
        raise ValueError('Some estimated RLS is negative, this should never happen. Min prob: {}'.format(np.min(tau)))

    return tau


def reduce_lambda(X, eval_L, D: CentersDictionary, lam_new: float, rng, rls_oversample=None):
    """Given a previously computed (eps, lambda)-accurate dictionary and a lambda' < lambda parameter,
     it constructs an (eps, lambda')-accurate dictionary using approximate RLS sampling.
    :param array_like X: dataset that we must approximate
    :param callable eval_L: likelihood function
    :param CentersDictionary D: an (eps, lambda) accurate dictionary, see :ref:`bless`
    :param float lam_new: lambda regularization for the new dictionary
    :param RandomState rng: rng for sampling
    :param rls_oversample: Oversampling parameter to increase success probability, see :ref:`bless`
    :return: An (eps, lam_new)-accurate dictionary with high probability
    :rtype:
        CentersDictionary
     """

    n, d = X.shape

    if rls_oversample is None:
        rls_oversample = D.rls_oversample

    red_ratio = D.lam / lam_new

    if not red_ratio >= 1.0:
        raise ValueError(str(red_ratio))

    diag = np.asarray(evaluate_L_diagonal(eval_L, X))

    # compute upper confidence bound on RLS of each sample, overestimate (oversample) by a rls_oversample factor
    # to boost success probability at the expenses of a larger sample (dictionary)
    ucb = np.minimum(rls_oversample * diag / (diag + lam_new), 1.)

    U = np.asarray(rng.rand(n)) <= ucb
    u = U.sum()

    if not u > 0:
        raise ValueError('No point selected during uniform sampling step, try to increase rls_oversample_bless. '
                         'Expected number of points: {:.3f}'.format(n * ucb))

    X_U = X[U, :]

    # taus are RLS
    tau = estimate_rls_bless(D, X_U, eval_L, lam_new)

    # RLS should always be smaller than 1
    tau = np.minimum(tau, 1.0)

    # same as before, oversample by a rls_oversample factor
    probs = np.minimum(rls_oversample * tau, ucb[U]) / ucb[U]

    if not np.all(probs >= 0.0):
        raise ValueError('Some estimated probability is negative, this should never happen. '
                         'Min prob: {}'.format(np.min(probs)))

    deff_estimate = probs.sum()/rls_oversample

    if not rls_oversample*deff_estimate >= 1.0:
        raise ValueError('Estimated deff is smaller than 1, you might want to reconsider your kernel. '
                         'deff_estimate: {:.3f}'.format(rls_oversample*deff_estimate))

    selected = np.asarray(rng.rand(u)) <= probs

    s = selected.sum()

    if not s > 0:
        raise ValueError('No point selected during RLS sampling step, try to increase rls_oversample_bless. '
                         'Expected number of points (rls_oversample_bless*deff): {:.3f}'.format(np.sum(probs)))

    D_new = CentersDictionary(idx=U.nonzero()[0][selected.nonzero()[0]],
                              X=X_U[selected, :],
                              probs=probs[selected],
                              lam=lam_new,
                              rls_oversample=rls_oversample)

    return D_new


def bless(X, eval_L, lam_final, rls_oversample, random_state=None, H=None, verbose=True):
    """Returns a (eps, lambda)-accurate dictionary of Nystrom centers sampled according to approximate RLS.

    Given data X, a similarity function, and its related similarity matrix similarity_function(X, X),
    an (eps, lambda)-accurate dictionary approximates all principal components of the similarity matrix
    with a singular value larger than lambda, up to a (1+eps) multiplicative error.

    The algorithm is introduced and analyzed in :cite:`RuCaCaRo18`, for a more formal
    definition of (eps, lambda)-accuracy and other potential uses see :cite:`CaLaVa17`.

    :param array_like X: input data, as an ndarray-like (n x m) object

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

    :param int rls_oversample: Oversampling parameter used during BLESS's step of random RLS sampling.
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

    :param H: number of iterations, defaults to log(n) if None
    :type H:
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

    n, d = X.shape

    H = H if H is not None else np.ceil(np.log(n)).astype('int')

    diag_norm = np.asarray(evaluate_L_diagonal(eval_L, X))
    ucb_init = rls_oversample * diag_norm / n

    rng = check_random_state(random_state)

    selected_init = rng.rand(n) <= ucb_init

    # force at least one sample to be selected
    selected_init[0] = 1

    D = CentersDictionary(idx=selected_init.nonzero(),
                          X=X[selected_init, :],
                          probs=np.ones(np.sum(selected_init)) * ucb_init[selected_init],
                          lam=n,
                          rls_oversample=rls_oversample)

    lam_sequence = list(np.geomspace(lam_final, n, H))

    # discard n from the list, we already used it to initialize
    lam_sequence.pop()

    with get_progress_bar(total=len(lam_sequence), disable=not verbose) as t:
        while len(lam_sequence) > 0:
            lam_new = lam_sequence.pop()
            D = reduce_lambda(X, eval_L, D, lam_new, rng)
            t.set_postfix(lam=int(lam_new),
                          m=len(D.probs),
                          m_expected=int(D.probs.mean() * n),
                          probs_dist="({:.4}, {:.4}, {:.4})".format(D.probs.mean(), D.probs.max(), D.probs.min()))
            t.update()

    return D
