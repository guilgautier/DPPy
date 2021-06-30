# coding: utf8
"""Implementation of finite DPP exact samplers derived from:

- the raw **projection** correlation :math:`K` kernel (no need for eigendecomposition)
- the eigendecomposition of the correlation :math:`K` kernel

.. seealso:

    `Documentation on ReadTheDocs <https://dppy.readthedocs.io/en/latest/finite_dpps/exact_sampling.html>`_
"""

import numpy as np
from ..utils import check_random_state

##########
# k-DPPs #
##########


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


def elementary_symmetric_polynomials(x, k):
    """Evaluate the `elementary symmetric polynomials <https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial>`_ :math:`[e_i(x_1, \\dots, x_m)]_{i=0, m=1}^{k, n}`.

    :param x:
        Points at which the elementary symmetric polynomials will be evaluated
    :type x:
        array_like

    :param k:
        Maximum degree of the elementary symmetric polynomials to be evaluated
    :type k:
        int

    :return:
        Matrix of size :math:`(k+1, n)` containing the evaluation of the elementary symmetric polynomials :math:`[e_i(x_1, \\dots, x_m)]_{i=0, m=1}^{k, n}`
    :rtype:
        array_like

    .. seealso::

        - :cite:`KuTa12` Algorithm 7
        - `Wikipedia <https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial>`_
    """

    # Initialize output array
    n = x.size
    E = np.zeros((k + 1, n + 1), dtype=float)
    E[0, :] = 1.0

    # Recursive evaluation
    for i in range(1, k + 1):
        for m in range(0, n):
            E[i, m + 1] = E[i, m] + x[m] * E[i - 1, m]

    return E
