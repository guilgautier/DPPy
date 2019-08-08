# coding: utf8
""" Implementation of finite DPP exact samplers derived from:

- the raw **projection** correlation :math:`K` kernel (no need for eigendecomposition)
- the eigendecomposition of the correlation :math:`K` kernel

.. seealso:

    `Documentation on ReadTheDocs <https://dppy.readthedocs.io/en/latest/finite_dpps/exact_sampling.html>`_
"""

import numpy as np
import scipy.linalg as la
from dppy.utils import inner1d, check_random_state


#####################
# Projection kernel #
#####################
# Sample projection DPP from kernel
def proj_dpp_sampler_kernel(kernel, mode='GS', size=None, random_state=None):
    """
    .. seealso::
        - :func:`proj_dpp_sampler_kernel_GS <proj_dpp_sampler_kernel_GS>`
        - :func:`proj_dpp_sampler_kernel_Schur <proj_dpp_sampler_kernel_Schur>`
        - :func:`proj_dpp_sampler_kernel_Chol <proj_dpp_sampler_kernel_Chol>`
    """

    rng = check_random_state(random_state)

    if size:
        rank = int(np.round(np.trace(kernel)))
        if size > rank:
            raise ValueError('size k={} > rank={}'. format(size, rank))

    # Sample from orthogonal projection kernel K = K^2 = K.H K
    if mode == 'GS':  # Gram-Schmidt equiv Cholesky
        sampl = proj_dpp_sampler_kernel_GS(kernel, size, rng)

    elif mode == 'Chol':  # Cholesky updates of Pou19
        sampl = proj_dpp_sampler_kernel_Chol(kernel, size, rng)[0]

    elif mode == 'Schur':  # Schur complement
        sampl = proj_dpp_sampler_kernel_Schur(kernel, size, rng)

    else:
        str_list = ['Invalid sampling mode, choose among:',
                    '- "GS (default)',
                    '- "Chol"',
                    '- "Schur"',
                    'Given {}'.format(mode)]
        raise ValueError('\n'.join(str_list))

    return sampl


def proj_dpp_sampler_kernel_Chol(K, size=None, random_state=None):
    """ Sample from:

    - :math:`\\operatorname{DPP}(K)` with orthogonal projection **correlation** kernel :math:`K` if ``size`` is not provided
    - :math:`\\operatorname{k-DPP}` with orthogonal projection **likelihood** kernel :math:`K` with :math:`k=` ``size`` is not provided

    Chain rule is applied by performing Cholesky updates of :math:`K`.

    :param K:
        Orthogonal projection kernel.
    :type K:
        array_like

    :param k:
        Size of the sample.
        Default is :math:`k=\\operatorname{trace}(K)=\\operatorname{rank}(K)`.
    :type k:
        int

    :return:
        If ``size`` is not provided (None),
            A sample :math:`\\mathcal{X}` from :math:`\\operatorname{DPP}(K)`.
        If ``size`` is provided,
            A sample :math:`\\mathcal{X}` from :math:`\\operatorname{k-DPP}(K)`.
        along with in-place Cholesky factorization of :math:`\\mathbf{K}_{\\mathcal{X} }`
    :rtype:
        list and array_like

    .. caution::

        The current implementation is an attempt of @guilgautier to reproduce the original C implementation of `catamari <https://gitlab.com/hodge_star/catamari>`_

    .. seealso::

        - :cite:`Pou19` Algorithm 3 and :ref:`catamari code <https://gitlab.com/hodge_star/catamari/blob/38718a1ea34872fb6567e019ece91fbeb5af5be1/include/catamari/dense_dpp/elementary_hermitian_dpp-impl.hpp#L37>`_ for the Hermitian swap routine.
        - :func:`proj_dpp_sampler_kernel_GS <proj_dpp_sampler_kernel_GS>`
        - :func:`proj_dpp_sampler_kernel_Schur <proj_dpp_sampler_kernel_Schur>`
    """

    rng = check_random_state(random_state)

    hermitian = True if K.dtype.kind == 'c' else False

    N, rank = len(K), np.round(np.trace(K)).astype(int)
    if size is None:  # full projection DPP
        size = rank
    # else: k-DPP with k = size

    A = K.copy()
    d = np.diagonal(A).astype(float)

    orig_indices = np.arange(N)

    for j in range(size):

        # Sample from pivot index and permute
        t = rng.choice(range(j, N), p=d[j:] / (rank - j))

        # Hermitian swap of indices j and t of A (may be written in a function)
        # bottom swap
        A[t + 1:, [j, t]] = A[t + 1:, [t, j]]
        # inner swap
        tmp = A[j + 1:t, j].copy()
        np.conj(A[t, j + 1:t], out=A[j + 1:t, j])
        np.conj(tmp, out=A[t, j + 1:t])
        # corner swap
        A[t, j] = A[t, j].conj()
        # diagonal swap
        A[[j, t], [j, t]] = A[[t, j], [t, j]].real
        # left swap
        A[[j, t], :j] = A[[t, j], :j]

        # Swap positions j and t of orig_indices and d
        orig_indices[[j, t]] = orig_indices[[t, j]]
        d[[j, t]] = d[[t, j]]

        A[j, j] = np.sqrt(d[j])

        if j == size - 1:
            break

        # Form new column and update diagonal
        A[j + 1:, j] -= A[j + 1:, :j].dot(A[j, :j].conj())
        A[j + 1:, j] /= A[j, j]

        if hermitian:
            d[j + 1:] -= A[j + 1:, j].real**2 + A[j + 1:, j].imag**2
        else:
            d[j + 1:] -= A[j + 1:, j]**2

    return orig_indices[:size].tolist(), A[:size, :size]


def proj_dpp_sampler_kernel_GS(K, size=None, random_state=None):
    """ Sample from:

    - :math:`\\operatorname{DPP}(K)` with orthogonal projection **correlation** kernel :math:`K` if ``size`` is not provided
    - :math:`\\operatorname{k-DPP}` with orthogonal projection **likelihood** kernel :math:`K` with :math:`k=` ``size`` is not provided

    Chain rule is applied by performing sequential Gram-Schmidt orthogonalization or equivalently Cholesky decomposition updates of :math:`K`.

    :param K:
        Orthogonal projection kernel.
    :type K:
        array_like

    :param k:
        Size of the sample.
        Default is :math:`k=\\operatorname{trace}(K)=\\operatorname{rank}(K)`.
    :type k:
        int

    :return:
        If ``size`` is not provided (None),
            A sample from :math:`\\operatorname{DPP}(K)`.
        If ``size`` is provided,
            A sample from :math:`\\operatorname{k-DPP}(K)`.
    :rtype:
        array_like

    .. seealso::

        - cite:`TrBaAm18` Algorithm 3, :cite:`Gil14` Algorithm 2
        - :func:`proj_dpp_sampler_kernel_Schur <proj_dpp_sampler_kernel_Schur>`
        - :func:`proj_dpp_sampler_kernel_Chol <proj_dpp_sampler_kernel_Chol>`
    """

    rng = check_random_state(random_state)

    # Initialization
    # ground set size / rank(K) = Tr(K)
    N, rank = len(K), np.round(np.trace(K)).astype(int)
    if size is None:  # full projection DPP
        size = rank
    # else: k-DPP with k = size

    ground_set = np.arange(N)
    sampl = np.zeros(size, dtype=int)  # sample list
    avail = np.ones(N, dtype=bool)  # available items

    c = np.zeros((N, size))
    norm_2 = K.diagonal().copy()  # residual norm^2

    for it in range(size):
        j = rng.choice(ground_set[avail],
                       p=np.abs(norm_2[avail]) / (rank - it))

        sampl[it] = j
        if it == size - 1:
            break
        # Update the Cholesky factor
        avail[j] = False
        c[avail, it] = (K[avail, j] - c[avail, :it].dot(c[j, :it]))\
                       / np.sqrt(norm_2[j])

        norm_2[avail] -= c[avail, it]**2

    return sampl


def proj_dpp_sampler_kernel_Schur(K, size=None, random_state=None):
    """ Sample from:

    - :math:`\\operatorname{DPP}(K)` with orthogonal projection **correlation** kernel :math:`K` if ``size`` is not provided
    - :math:`\\operatorname{k-DPP}` with orthogonal projection **likelihood** kernel :math:`K` with :math:`k=` ``size``

    Chain rule is applied by computing the Schur complements.

    :param K:
        Orthogonal projection kernel.
    :type K:
        array_like
    :param size:
        Size of the sample.
        Default is :math:`k=\\operatorname{trace}(K)=\\operatorname{rank}(K)`.
    :type size:
        int

    :return:
        If ``size`` is not provided (None),
            A sample from :math:`\\operatorname{DPP}(K)`.
        If ``size`` is provided,
            A sample from :math:`\\operatorname{k-DPP}(K)`.
    :rtype:
        array_like

    .. seealso::
        - :func:`proj_dpp_sampler_kernel_GS <proj_dpp_sampler_kernel_GS>`
        - :func:`proj_dpp_sampler_kernel_Chol <proj_dpp_sampler_kernel_Chol>`
    """

    rng = check_random_state(random_state)

    # Initialization
    # ground set size / rank(K) = Tr(K)
    N, rank = len(K), np.round(np.trace(K)).astype(int)
    if size is None:  # full projection DPP
        size = rank
    # else: k-DPP with k = size

    ground_set = np.arange(N)
    sampl = np.zeros(size, dtype=int)  # sample list
    avail = np.ones(N, dtype=bool)  # available items

    # Schur complement list i.e. residual norm^2
    schur_comp = K.diagonal().copy()
    K_inv = np.zeros((size, size))

    for it in range(size):
        # Pick a new item proportionally to residual norm^2
        j = rng.choice(ground_set[avail],
                             p=np.abs(schur_comp[avail]) / (rank - it))
        # store the item and make it unavailable
        sampl[it], avail[j] = j, False

        # Update Schur complements K_ii - K_iY (K_Y)^-1 K_Yi
        #
        # 1) use Woodbury identity to update K[Y,Y]^-1 to K[Y+j,Y+j]^-1
        # K[Y+j,Y+j]^-1 =
        # [ K[Y,Y]^-1 + (K[Y,Y]^-1 K[Y,j] K[j,Y] K[Y,Y]^-1)/schur_j,
        #      -K[Y,Y]^-1 K[Y,j]/schur_j]
        # [ -K[j,Y] K[Y,Y]^-1/schur_j,
        #      1/schur_j]
        if it == 0:
            K_inv[0, 0] = 1.0 / K[j, j]

        elif it == 1:
            i = sampl[0]
            K_inv[:2, :2] = np.array([[K[j, j], -K[j, i]],
                                      [-K[j, i], K[i, i]]])\
                            / (K[i, i] * K[j, j] - K[j, i]**2)

        elif it < size - 1:
            temp = K_inv[:it, :it].dot(K[sampl[:it], j])  # K_Y^-1 K_Yj
            # K_jj - K_jY K_Y^-1 K_Yj
            schur_j = K[j, j] - K[j, sampl[:it]].dot(temp)

            K_inv[:it, :it] += np.outer(temp, temp / schur_j)
            K_inv[:it, it] = - temp / schur_j
            K_inv[it, :it] = K_inv[:it, it]
            K_inv[it, it] = 1.0 / schur_j

        else:  # it == size-1
            break  # no need to update for nothing

        # 2) update Schur complements
        # K_ii - K_iY (K_Y)^-1 K_Yi for Y <- Y+j
        K_iY = K[np.ix_(avail, sampl[:it + 1])]
        schur_comp[avail] = K[avail, avail]\
                        - inner1d(K_iY.dot(K_inv[:it+1, :it+1]), K_iY, axis=1)

    return sampl


##################
# Generic kernel #
##################

# Directly from correlation kernel, without spectral decomposition
##################################################################
def dpp_sampler_generic_kernel(K, random_state=None):
    """ Sample from generic :math:`\\operatorname{DPP}(\\mathbf{K})` with potentially non hermitian correlation kernel :math:`\\operatorname{DPP}(\\mathbf{K})` based on :math:`LU` factorization procedure.

    :param K:
        Correlation kernel (potentially non hermitian).
    :type K:
        array_like

    :return:
        A sample :math:`\\mathcal{X}` from :math:`\\operatorname{DPP}(K)` and
        the in-place :math:`LU factorization of :math:`K − I_{\\mathcal{X}^{c}}` where :math:`I_{\\mathcal{X}^{c}}` is the diagonal indicator matrix for the entries not in the sample :math:`\\mathcal{X}`.
    :rtype:
        list and array_like

    .. seealso::

        - :cite:`Pou19` Algorithm 1
    """

    rng = check_random_state(random_state)

    A = K.copy()
    sample = []

    for j in range(len(A)):

        if rng.rand() < A[j, j]:
            sample.append(j)
        else:
            A[j, j] -= 1

        A[j+1:, j] /= A[j, j]
        A[j+1:, j+1:] -= np.outer(A[j+1:, j], A[j, j+1:])
#         A[j+1:, j+1:] -=  np.einsum('i,j', A[j+1:, j], A[j, j+1:])

    return sample, A

# From spectral decomposition
#############################

# Phase 1: subsample eigenvectors by drawing independent Bernoulli variables with parameter the eigenvalues of the correlation kernel K.
def dpp_eig_vecs_selector(ber_params, eig_vecs,
                          random_state=None):
    """ Phase 1 of exact sampling procedure. Subsample eigenvectors :math:`V` of the initial kernel (correlation :math:`K`, resp. likelihood :math:`L`) to build a projection DPP with kernel :math:`V V^{\\top}` from which sampling is easy.
    The selection is made based on a realization of Bernoulli variables with parameters related to the eigenvalues of :math:`K`, resp. :math:`L`.

    :param ber_params:
        Parameters of Bernoulli variables
        :math:`\\lambda^K=\\lambda^L/(1+\\lambda^L)
    :type ber_params:
        list, array_like

    :param eig_vecs:
        Collection of eigenvectors of the kernel :math:`K`, resp. :math:`L`
    :type eig_vecs:
        array_like

    :return:
        selected eigenvectors
    :rtype:
        array_like

    .. seealso::

        - :func:`dpp_sampler_eig <dpp_sampler_eig>`
    """
    rng = check_random_state(random_state)

    # Realisation of Bernoulli random variables with params ber_params
    ind_sel = rng.rand(ber_params.size) < ber_params

    return eig_vecs[:, ind_sel]


def dpp_eig_vecs_selector_L_dual(eig_vals, eig_vecs, gram_factor,
                                 random_state=None):
    """ Subsample eigenvectors :math:`V` of likelihood kernel :math:`L=\\Phi^{\\top} \\Phi` based on the eigendecomposition dual kernel :math:`L'=\\Phi \\Phi^{\\top}`. Note that :math:`L'` and :math:`L'` share the same nonzero eigenvalues.

    :param eig_vals:
        Collection of eigenvalues of :math:`L_dual` kernel.
    :type eig_vals:
        list, array_like

    :param eig_vecs:
        Collection of eigenvectors of :math:`L_dual` kernel.
    :type eig_vecs:
        array_like

    :param gram_factor:
        Feature matrix :math:`\\Phi`
    :type gram_factor:
        array_like

    :return:
        selected eigenvectors
    :rtype:
        array_like

    .. see also::

        Phase 1:

        - :func:`dpp_eig_vecs_selector <dpp_eig_vecs_selector>`

        Phase 2:

        - :func:`proj_dpp_sampler_eig_GS <proj_dpp_sampler_eig_GS>`
        - :func:`proj_dpp_sampler_eig_GS_bis <proj_dpp_sampler_eig_GS_bis>`
        - :func:`proj_dpp_sampler_eig_KuTa12 <proj_dpp_sampler_eig_KuTa12>`
    """

    rng = check_random_state(random_state)

    # Realisation of Bernoulli random variables with params eig_vals
    ind_sel = rng.rand(eig_vals.size) < eig_vals / (1.0 + eig_vals)

    return gram_factor.T.dot(eig_vecs[:, ind_sel] / np.sqrt(eig_vals[ind_sel]))

# Phase 2:
# Sample projection kernel VV.T where V are the eigvecs selected in Phase 1.
def proj_dpp_sampler_eig(eig_vecs, mode='GS', size=None,
                         random_state=None):
    """ Sample from projection :math:`\\operatorname{DPP}(K)` using the eigendecomposition of the projection kernel :math:`K=VV^{\top}` where :math:`V^{\top}V = I_r` and :math:`r=\\operatorname{rank}(\\mathbf{K})`.

    .. seealso::

        Phase 1:

        - :func:`dpp_eig_vecs_selector <dpp_eig_vecs_selector>`
        - :func:`dpp_eig_vecs_selector_gram_factor <dpp_eig_vecs_selector_gram_factor>`

        Phase 2:

        - :func:`proj_dpp_sampler_eig_GS <proj_dpp_sampler_eig_GS>`
        - :func:`proj_dpp_sampler_eig_GS_bis <proj_dpp_sampler_eig_GS_bis>`
        - :func:`proj_dpp_sampler_eig_KuTa12 <proj_dpp_sampler_eig_KuTa12>`
    """

    rng = check_random_state(random_state)

    if eig_vecs.shape[1]:
        # Phase 2: Sample from projection kernel VV.T
        # Chain rule, conditionals are updated using:
        if mode == 'GS':  # Gram-Schmidt
            sampl = proj_dpp_sampler_eig_GS(eig_vecs, size, rng)

        elif mode == 'GS_bis':  # Slight modif of 'GS'
            sampl = proj_dpp_sampler_eig_GS_bis(eig_vecs, size, rng)

        elif mode == 'KuTa12':  # cf Kulesza-Taskar
            sampl = proj_dpp_sampler_eig_KuTa12(eig_vecs, size, rng)

        else:
            str_list = ['Invalid sampling mode, choose among:',
                        '- "GS" (default)',
                        '- "GS_bis"',
                        '- "KuTa12"',
                        'Given {}'.format(mode)]
            raise ValueError('\n'.join(str_list))
    else:
        sampl = []

    return sampl


# Using Gram-Schmidt orthogonalization
def proj_dpp_sampler_eig_GS(eig_vecs, size=None,
                            random_state=None):
    """ Sample from projection :math:`\\operatorname{DPP}(K)` using the eigendecomposition of the projection kernel :math:`K=VV^{\top}` where :math:`V^{\top}V = I_r` and :math:`r=\\operatorname{rank}(\\mathbf{K})`.
    It performs sequential update of Cholesky decomposition, which is equivalent to Gram-Schmidt orthogonalization of the rows of the eigenvectors.

    :param eig_vecs:
        Eigenvectors used to form projection kernel :math:`K=VV^{\top}`.
    :type eig_vecs:
        array_like

    :return:
        A sample from projection :math:`\\operatorname{DPP}(K)`.
    :rtype:
        list, array_like

    .. seealso::

        - cite:`TrBaAm18` Algorithm 3, :cite:`Gil14` Algorithm 2
        - :func:`proj_dpp_sampler_eig_GS_bis <proj_dpp_sampler_eig_GS_bis>`
        - :func:`proj_dpp_sampler_eig_KuTa12 <proj_dpp_sampler_eig_KuTa12>`
    """

    rng = check_random_state(random_state)

    # Initialization
    V = eig_vecs

    N, rank = V.shape  # ground set size / rank(K)
    if size is None:  # full projection DPP
        size = rank
    # else: k-DPP with k = size

    ground_set = np.arange(N)
    sampl = np.zeros(size, dtype=int)  # sample list
    avail = np.ones(N, dtype=bool)  # available items

    # Phase 1: Already performed!
    # Select eigvecs with Bernoulli variables with parameter = eigvals of K.

    # Phase 2: Chain rule
    # Use Gram-Schmidt recursion to compute the Vol^2 of the parallelepiped spanned by the feature vectors associated to the sample

    c = np.zeros((N, size))
    norms_2 = inner1d(V, axis=1)  # ||V_i:||^2

    for it in range(size):
        # Pick an item \propto this squred distance
        j = rng.choice(ground_set[avail],
                             p=np.abs(norms_2[avail]) / (rank - it))
        sampl[it] = j
        if it == size - 1:
            break
        # Cancel the contribution of V_j to the remaining feature vectors
        avail[j] = False
        c[avail, it] =\
            (V[avail, :].dot(V[j, :]) - c[avail, :it].dot(c[j, :it]))\
            / np.sqrt(norms_2[j])

        norms_2[avail] -= c[avail, it]**2  # update residual norm^2

    return sampl


# Slight modif of Gram-Schmidt above
def proj_dpp_sampler_eig_GS_bis(eig_vecs, size=None, random_state=None):
    """ Sample from projection :math:`\\operatorname{DPP}(K)` using the eigendecomposition of the projection kernel :math:`K=VV^{\top}` where :math:`V^{\top}V = I_r` and :math:`r=\\operatorname{rank}(\\mathbf{K})`.
    It performs sequential Gram-Schmidt orthogonalization of the rows of the eigenvectors.

    :param eig_vecs:
        Eigenvectors used to form projection kernel :math:`K=VV^{\top}`.
    :type eig_vecs:
        array_like

    :return:
        A sample from projection :math:`\\operatorname{DPP}(K)`.
    :rtype:
        list, array_like

    .. seealso::

        - This is a slight modification of :func:`proj_dpp_sampler_eig_GS <proj_dpp_sampler_eig_GS>`
        - :func:`proj_dpp_sampler_eig_KuTa12 <proj_dpp_sampler_eig_KuTa12>`
    """

    rng = check_random_state(random_state)

    # Initialization
    V = eig_vecs.copy()

    N, rank = V.shape  # ground set size / rank(K)
    if size is None:  # full projection DPP
        size = rank
    # else: k-DPP with k = size

    ground_set = np.arange(N)
    sampl = np.zeros(size, dtype=int)  # sample list
    avail = np.ones(N, dtype=bool)  # available items

    # Phase 1: Already performed!
    # Select eigvecs with Bernoulli variables with parameter = eigvals of K.

    # Phase 2: Chain rule
    # Use Gram-Schmidt recursion to compute the Vol^2 of the parallelepiped spanned by the feature vectors associated to the sample

    # Matrix of the contribution of remaining vectors
    # <V_i, P_{V_Y}^{orthog} V_j>
    contrib = np.zeros((N, size))

    norms_2 = inner1d(V, axis=1)  # ||V_i:||^2

    for it in range(size):

        # Pick an item proportionally to the residual norm^2
        # ||P_{V_Y}^{orthog} V_j||^2
        j = rng.choice(ground_set[avail],
                             p=np.abs(norms_2[avail]) / (rank - it))
        sampl[it] = j
        if it == size - 1:
            break
        # Update the residual norm^2
        #
        # |P_{V_Y+j}^{orthog} V_i|^2
        #                                    <V_i,P_{V_Y}^{orthog} V_j>^2
        #     =  |P_{V_Y}^{orthog} V_i|^2 -  ----------------------------
        #                                      |P_{V_Y}^{orthog} V_j|^2
        #
        # 1) Orthogonalize V_j w.r.t. orthonormal basis of Span(V_Y)
        #    V'_j = P_{V_Y}^{orthog} V_j
        #         = V_j - <V_j,sum_Y V'_k>V'_k
        #         = V_j - sum_Y <V_j, V'_k> V'_k
        # Note V'_j is not normalized
        avail[j] = False
        V[j, :] -= contrib[j, :it].dot(V[sampl[:it], :])

        # 2) Compute <V_i, V'_j> = <V_i, P_{V_Y}^{orthog} V_j>
        contrib[avail, it] = V[avail, :].dot(V[j, :])

        # 3) Normalize V'_j with norm^2 and not norm
        #              V'_j         P_{V_Y}^{orthog} V_j
        #    V'_j  =  -------  =  --------------------------
        #             |V'j|^2      |P_{V_Y}^{orthog} V_j|^2
        #
        # in preparation for next orthogonalization in 1)
        V[j, :] /= norms_2[j]

        # 4) Update the residual norm^2: cancel contrib of V_i onto V_j
        #
        # |P_{V_Y+j}^{orthog} V_i|^2
        #   = |P_{V_Y}^{orthog} V_i|^2 - <V_i,V'_j>^2 / |V'j|^2
        #                                  <V_i,P_{V_Y}^{orthog} V_j>^2
        #   =  |P_{V_Y}^{orthog} V_i|^2 -  ----------------------------
        #                                   |P_{V_Y}^{orthog} V_j|^2
        norms_2[avail] -= contrib[avail, it]**2 / norms_2[j]

    return sampl


def proj_dpp_sampler_eig_KuTa12(eig_vecs, size=None, random_state=None):
    """ Sample from :math:`\\operatorname{DPP}(K)` using the eigendecomposition of the similarity kernel :math:`K`.
    It is based on the orthogonalization of the selected eigenvectors.

    :param eig_vals:
        Collection of eigen values of the similarity kernel :math:`K`.
    :type eig_vals:
        list

    :param eig_vecs:
        Eigenvectors of the similarity kernel :math:`K`.
    :type eig_vecs:
        array_like

    :return:
        A sample from :math:`\\operatorname{DPP}(K)`.
    :rtype:
        list

    .. seealso::

        - :cite:`KuTa12` Algorithm 1
        - :func:`proj_dpp_sampler_eig_GS <proj_dpp_sampler_eig_GS>`
        - :func:`proj_dpp_sampler_eig_GS_bis <proj_dpp_sampler_eig_GS_bis>`
    """

    rng = check_random_state(random_state)

    # Initialization
    V = eig_vecs.copy()

    N, rank = V.shape  # ground set size / rank(K)
    if size is None:  # full projection DPP
        size = rank
    # else: k-DPP with k = size

    sampl = np.zeros(size, dtype=int)  # sample list

    # Phase 1: Already performed!
    # Select eigvecs with Bernoulli variables with parameter the eigvals

    # Phase 2: Chain rule
    norms_2 = inner1d(V, axis=1)  # ||V_i:||^2

    # Following [Algo 1, KuTa12], the aim is to compute the orhto complement of the subspace spanned by the selected eigenvectors to the canonical vectors \{e_i ; i \in Y\}. We proceed recursively.
    for it in range(size):

        j = rng.choice(N, p=np.abs(norms_2) / (rank - it))
        sampl[it] = j
        if it == size - 1:
            break

        # Cancel the contribution of e_i to the remaining vectors that is, find the subspace of V that is orthogonal to \{e_i ; i \in Y\}
        # Take the index of a vector that has a non null contribution on e_j
        k = np.where(V[j, :] != 0)[0][0]
        # Cancel the contribution of the remaining vectors on e_j, but stay in the subspace spanned by V i.e. get the subspace of V orthogonal to \{e_i ; i \in Y\}
        V -= np.outer(V[:, k] / V[j, k], V[j, :])
        # V_:j is set to 0 so we delete it and we can derive an orthononormal basis of the subspace under consideration
        V, _ = la.qr(np.delete(V, k, axis=1), mode='economic')

        norms_2 = inner1d(V, axis=1)  # ||V_i:||^2

    return sampl


##########
# k-DPPs #
##########

def k_dpp_eig_vecs_selector(eig_vals, eig_vecs, size,
                            E_poly=None, random_state=None):
    """ Subsample eigenvectors V of the 'L' kernel to build a projection DPP with kernel V V.T from which sampling is easy. The selection is made based a realization of Bernoulli variables with parameters the eigenvalues of 'L' and evalutations of the elementary symmetric polynomials.

    :param eig_vals:
        Collection of eigen values of 'L' (likelihood) kernel.
    :type eig_vals:
        list, array_like

    :param eig_vecs:
        Collection of eigenvectors of 'L' kernel.
    :type eig_vecs:
        array_like

    :param size:
        Size :math:`k` of :math:`k`-DPP
    :type size:
        int

    :param E_poly:
        Evaluation of symmetric polynomials in the eigenvalues
    :type E_poly:
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
    tol = np.max(eig_vals) * N * np.finfo(np.float).eps
    rank = np.count_nonzero(eig_vals > tol)
    if k > rank:
        raise ValueError('size k={} > rank={}'.format(k, rank))

    if E_poly is None:
        E_poly = elementary_symmetric_polynomials(eig_vals, k)

    ind_selected = np.zeros(k, dtype=int)
    for n in range(eig_vals.size, 0, -1):

        if rng.rand() < eig_vals[n-1] * E_poly[k-1, n-1] / E_poly[k, n]:
            k -= 1
            ind_selected[k] = n - 1
            if k == 0:
                break

    return eig_vecs[:, ind_selected]


# Evaluate the elementary symmetric polynomials
def elementary_symmetric_polynomials(eig_vals, size):
    """ Evaluate the elementary symmetric polynomials :math:`e_k` in the eigenvalues :math:`(\\lambda_1, \\cdots, \\lambda_N)`.

    :param eig_vals:
        Collection of eigenvalues :math:`(\\lambda_1, \\cdots, \\lambda_N)` of the similarity kernel :math:`L`.
    :type eig_vals:
        list

    :param size:
        Maximum degree of elementary symmetric polynomial.
    :type size:
        int

    :return:
        :math:`[E_{kn}]_{k=0, n=0}^{\text{size}, N}`
        :math:`E_{kn} = e_k(\\lambda_1, \\cdots, \\lambda_n)`
    :rtype:
        array_like

    .. seealso::

        - :cite:`KuTa12` Algorithm 7
        - `Wikipedia <https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial>`_
    """

    # Initialize output array
    N = eig_vals.size
    E_poly = np.zeros((size + 1, N + 1))
    E_poly[0, :] = 1.0

    # Recursive evaluation
    for l in range(1, size + 1):
        for n in range(1, N + 1):
            E_poly[l, n] = E_poly[l, n-1] + eig_vals[n-1] * E_poly[l-1, n-1]

    return E_poly
