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

Dictionary = namedtuple('Dictionary', ('idx', 'X', 'probs', 'lam', 'qbar'))


def estimate_rls(D, X, eval_L, lam_new):
    """Given a previosuly computed (eps, lambda)-accurate dictionary, it computes estimates
    of all RLS using the estimator from Calandriello et al. 2017

    .. todo::

        - asserts -> throw
        - create bib entry for Calandriello et al. 2017
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

    assert np.all(tau >= 0.), ("Some estimated RLS is negative, this should never happen."
                               "Min prob: {}".format(np.min(tau)))

    return tau


def reduce_lambda(X, eval_L, D: Dictionary, lam_new: float, rng, qbar=None):
    """
    .. todo::

        - write docstring
        - asserts -> throw
    """
    n, d = X.shape

    if qbar is None:
        qbar = D.qbar

    red_ratio = D.lam / lam_new

    assert red_ratio >= 1.

    diag = np.asarray(evaluate_L_diagonal(eval_L, X))

    # compute upper confidence bound on RLS of each sample, overestimate (oversample) by a qbar factor
    # to boost success probability at the expenses of a larger sample (dictionary)
    ucb = np.minimum(qbar * diag / (diag + lam_new), 1.)

    U = np.asarray(rng.rand(n)) <= ucb
    u = U.sum()

    assert u > 0, ("No point selected during uniform sampling step, try to increase qbar."
                   "Expected number of points: {:.3}".format(n * ucb))

    X_U = X[U, :]

    # taus are RLS
    tau = estimate_rls(D, X_U, eval_L, lam_new)

    # RLS should always be smaller than 1
    tau = np.minimum(tau, 1.)

    # same as before, oversample by a qbar factor
    probs = np.minimum(qbar * tau, ucb[U]) / ucb[U]

    assert np.all(probs >= 0.), ("Some estimated probability is negative, this should never happen."
                                 "Min prob: {}".format(np.min(probs)))

    deff_estimate = probs.sum()/qbar
    assert qbar*deff_estimate >= 1., ("Estimated deff is smaller than 1, you might want to reconsider your kernel."
                                      "deff_estimate: {:.3}".format(qbar*deff_estimate))

    selected = np.asarray(rng.rand(u)) <= probs

    s = selected.sum()

    assert s > 0, ("No point selected during RLS sampling step, try to increase qbar. "
                   "Expected number of points (qbar*deff): {:.3}".format(np.sum(probs)))

    D_new = Dictionary(idx=U.nonzero()[0][selected.nonzero()[0]],
                       X=X_U[selected, :],
                       probs=probs[selected],
                       lam=lam_new,
                       qbar=qbar)

    return D_new


def bless(X, eval_L, lam_final, qbar, random_state=None, H=None, verbose=True):
    """ Generates dictionary sampled according to RLS. In a nutshell, Rudi et al 2018 shows that uniform sampling is equivalent to RLS sampling with lambda=n.
    Therefore, we can initialize BLESS's dictionary using uniform sampling, and over H iteration alternate between reducing lambda_h, and re-sampling using the more accurate lambda_h-RLS.
    When we reach a desired final lambda_H we terminate

    :param X: input data, as an ndarray-like (n x m) object

    :param eval_L: likelihood function between points.
    If L is the associated likelihood matrix, it must satisfy the interface eval_L(X) = K(X,X) eval_L(X_1, X_2) = K(X_1, X_2)

    :param lam_final: final lambda (i.e. as in (eps, lambda)-accuracy) desired.
    Roughly, the final dictionary will approximate all eigenvalues larger than lambda, and therefore smaller lambda creates larger, more accurate dictionaries.

    :param qbar: BLESS generates a dictionary with :math:`\\tilde{O}(qbar*deff)` points, and the qbar >= 1 parameter controls the oversampling.
    Larger qbar make the algorithm succeed with higher probability and generate larger but more accurate dictionary.

    :param random_state:

    :param H: number of iterations, defaults to log(n) if None

    :param verbose: control debug output

    :return: A Dictionary
    """

    n, d = X.shape

    H = H if H is not None else np.ceil(np.log(n)).astype('int')

    diag_norm = np.asarray(evaluate_L_diagonal(eval_L, X))
    ucb_init = qbar * diag_norm / n

    rng = check_random_state(random_state)

    selected_init = rng.rand(n) <= ucb_init

    # force at least one sample to be selected
    selected_init[0] = 1

    D = Dictionary(idx=selected_init.nonzero(),
                   X=X[selected_init, :],
                   probs=np.ones(np.sum(selected_init)) * ucb_init[selected_init],
                   lam=n,
                   qbar=qbar)

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
