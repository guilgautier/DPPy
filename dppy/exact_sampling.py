# -*- coding: utf-8 -*-
""" Implementation of finite DPP exact samplers derived from:

- the raw **projection** inclusion :math:`K` kernel (no need for eigendecomposition)
- the eigendecomposition of the inclusion :math:`K` kernel

.. seealso:

    `Documentation on ReadTheDocs <https://dppy.readthedocs.io/en/latest/finite_dpps/exact_sampling.html>`_
"""

import numpy as np
import scipy.linalg as la
from dppy.utils import inner1d


#####################
# Projection kernel #
#####################
# Sample projection DPP from kernel
def proj_dpp_sampler_kernel(kernel, mode='GS'):
    """
    .. seealso::
        - :func:`proj_dpp_sampler_kernel_GS <proj_dpp_sampler_kernel_GS>`
    """

    # Phase 1: Select eigenvectors
    # No need for eigendecomposition

    # Phase 2: Sample from orthogonal projection kernel K = K^2 = K.T K
    # Chain rule, conditionals are updated using:
    if mode == 'GS':  # Gram-Schmidt equiv Cholesky
        sampl = proj_dpp_sampler_kernel_GS(kernel)

    elif mode == 'Schur':  # Schur complement
        sampl = proj_dpp_sampler_kernel_Schur(kernel)

    else:
        str_list = ['Invalid sampling mode, choose among:',
                    '- "GS" (default)',
                    '- "Schur"',
                    'Given {}'.format(mode)]
        raise ValueError('\n'.join(str_list))

    return sampl


def proj_dpp_sampler_kernel_GS(K, size=None):
    """ Sample from :math:`\operatorname{DPP}(K)` with :math:`K` orthogonal projection matrix.
    It performs sequential Gram-Schmidt orthogonalization or equivalently Cholesky decomposition updates of :math:`K`.

    :param K:
        Orthogonal projection kernel.
    :type K:
        array_like

    :param k:
        Size of the sample.
        Default is :math:`k=\operatorname{Tr}(K)=\operatorname{rank}(K)`.
    :type k:
        int

    :return:
        A sample from :math:`\operatorname{DPP}(K)`.
    :rtype:
        list

    .. seealso::

        - cite:`TrBaAm18` Algorithm 3, :cite:`Gil14` Algorithm 2

    #   - :func:`proj_dpp_sampler_kernel_Schur <proj_dpp_sampler_kernel_Schur>`
    """

    # Initialization
    # ground set size / rank(K) = Tr(K)
    N, rank = K.shape[0], int(np.round(np.trace(K)))
    ground_set = np.arange(N)

    size = rank if size is None else size  # Full projection DPP or k-DPP
    sampl = np.zeros(size, dtype=int)  # sample list

    avail = np.ones(N, dtype=bool)  # available items

    c = np.zeros((N, size))
    norm_2 = K.diagonal().copy()  # residual norm^2

    for it in range(size):
        j = np.random.choice(ground_set[avail],
                             size=1,
                             p=np.abs(norm_2[avail]) / (rank - it))[0]
        sampl[it] = j
        if it == size - 1:
            break
        # Update the Cholesky factor
        avail[j] = False
        c[avail, it] = (K[avail, j] - c[avail, :it].dot(c[j, :it]))\
                       / np.sqrt(norm_2[j])

        norm_2[avail] -= c[avail, it]**2

    return sampl


def proj_dpp_sampler_kernel_Schur(K, size=None):
    """ Sample from :math:`\operatorname{k-DPP}(K)` where the similarity kernel :math:`K`
    is an orthogonal projection matrix.
    It sequentially updates the Schur complement by updating the inverse of the matrix involved.
    :param K:
        Orthogonal projection kernel.
    :type K:
        array_like
    :param size:
        Size of the sample.
        Default is :math:`k=\operatorname{Tr}(K)=\operatorname{rank}(K)`.
    :type size:
        int

    :return:
        If ``size`` is not provided (None),
            A sample from :math:`\operatorname{DPP}(K)`.
        If ``size`` is provided,
            A sample from :math:`\operatorname{k-DPP}(K)`.
    :rtype:
        list

    .. seealso::
        - :func:`projection_dpp_sampler_GS_bis <projection_dpp_sampler_GS_bis>`
    """

    # Initialization
    # ground set size / rank(K) = Tr(K)
    N, rank = K.shape[0], int(np.round(np.trace(K)))
    ground_set = np.arange(N)

    size = rank if size is None else size  # Full projection DPP or k-DPP
    sampl = np.zeros(size, dtype=int)  # sample list

    avail = np.ones(N, dtype=bool)  # available items

    K_diag = K.diagonal()
    schur_comp = K_diag.copy()  # Schur complement list i.e. residual norm^2

    for it in range(size):
        # Pick a new item proportionally to residual norm^2
        j = np.random.choice(ground_set[avail],
                             size=1,
                             p=np.abs(schur_comp[avail]) / (rank - it))[0]
        # store the item and make it unavailable
        sampl[it], avail[j] = j, False

        # sampl = Y + j
        # Update Schur complements K_ii - K_iY (K_Y)^-1 K_Yi
        #
        # 1) use Woodbury identity to update K[Y,Y]^-1 to K[Y+j,Y+j]^-1
        # K[Y+j,Y+j]^-1 =
        # [ K[Y,Y]^-1 + (K[Y,Y]^-1 K[Y,j] K[j,Y] K[Y,Y]^-1)/schur_j,
        #      -K[Y,Y]^-1 K[Y,j]/schur_j]
        # [ -K[j,Y] K[Y,Y]^-1/schur_j,
        #      1/schur_j]
        if it == 0:
            K_inv = 1.0 / K[j, j]

        elif it == 1:
            Y = sampl[0]
            K_inv = np.array([[K[j, j], -K[j, Y]],
                             [-K[j, Y], K[Y, Y]]])
            K_inv /= K[Y, Y] * K[j, j] - K[j, Y]**2

        elif it < size - 1:
            Y = sampl[:it]
            temp = K_inv.dot(K[Y, j])  # K_Y^-1 K_Yj
            schur_j = K[j, j] - K[j, Y].dot(temp)  # K_jj-K_jY K_Y^-1 K_Yj

            K_inv = np.lib.pad(K_inv, (0, 1),
                               'constant',
                               constant_values=1.0 / schur_j)

            K_inv[:-1, :-1] += np.outer(temp, temp / schur_j)
            K_inv[:-1, -1] *= -temp
            K_inv[-1, :-1] = K_inv[:-1, -1]

        else:  # it == size-1
            break  # no need to update for nothing

        # 2) update Schur complements
        # K_ii - K_iY (K_Y)^-1 K_Yi for Y <- Y+j
        K_iY = K[np.ix_(avail, sampl[:it + 1])]
        schur_comp[avail] =\
            K_diag[avail] - inner1d(K_iY.dot(K_inv), K_iY, axis=1)

    return sampl


##################
# Generic kernel #
##################
# Phase 1: subsample eigenvectors by drawing independent Bernoulli variables with parameter the eigenvalues of the inclusion kernel K.
def dpp_eig_vecs_selector(ber_params, eig_vecs):
    """ Phase 1 of exact sampling procedure. Subsample eigenvectors :math:`V` of the initial kernel (inclusion :math:`K`, resp. marginal :math:`L`) to build a projection DPP with kernel :math:`V V^{\top}` from which sampling is easy.
    The selection is made based on a realization of Bernoulli variables with parameters related to the eigenvalues of :math:`K`, resp. :math:`L`.

    :param ber_params:
        Parameters of Bernoulli variables
        :math:`\lambda^K=\lambda^L/(1+\lambda^L)
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

    # Realisation of Bernoulli random variables with params ber_params
    ind_sel = np.random.rand(ber_params.size) < ber_params

    return eig_vecs[:, ind_sel]


def dpp_eig_vecs_selector_L_dual(eig_vals, eig_vecs, gram_factor):
    """ Subsample eigenvectors :math:`V` of marginal kernel :math:`L=\Phi^{\top} \Phi` based on the eigendecomposition dual kernel :math:`L'=\Phi \Phi^{\top}`. Note that :math:`L'` and :math:`L'` share the same nonzero eigenvalues.

    :param eig_vals:
        Collection of eigenvalues of :math:`L_dual` kernel.
    :type eig_vals:
        list, array_like

    :param eig_vecs:
        Collection of eigenvectors of :math:`L_dual` kernel.
    :type eig_vecs:
        array_like

    :param gram_factor:
        Feature matrix :math:`\Phi`
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

    # Realisation of Bernoulli random variables with params eig_vals
    ind_sel = np.random.rand(eig_vals.size) < eig_vals / (1.0 + eig_vals)

    return gram_factor.T.dot(eig_vecs[:, ind_sel] / np.sqrt(eig_vals[ind_sel]))


# Phase 2:
# Sample projection kernel VV.T where V are the eigvecs selected in Phase 1.
def proj_dpp_sampler_eig(eig_vecs, mode='GS'):
    """ Sample from projection :math:`\operatorname{DPP}(K)` using the eigendecomposition of the projection kernel :math:`K=VV^{\top}` where :math:`V^{\top}V = I_r` and :math:`r=\operatorname{rank}(\mathbf{K})`.

    .. seealso::

        Phase 1:

        - :func:`dpp_eig_vecs_selector <dpp_eig_vecs_selector>`
        - :func:`dpp_eig_vecs_selector_gram_factor <dpp_eig_vecs_selector_gram_factor>`

        Phase 2:

        - :func:`proj_dpp_sampler_eig_GS <proj_dpp_sampler_eig_GS>`
        - :func:`proj_dpp_sampler_eig_GS_bis <proj_dpp_sampler_eig_GS_bis>`
        - :func:`proj_dpp_sampler_eig_KuTa12 <proj_dpp_sampler_eig_KuTa12>`
    """
    if eig_vecs.shape[1]:
        # Phase 2: Sample from projection kernel VV.T
        # Chain rule, conditionals are updated using:
        if mode == 'GS':  # Gram-Schmidt
            sampl = proj_dpp_sampler_eig_GS(eig_vecs)

        elif mode == 'GS_bis':  # Slight modif of 'GS'
            sampl = proj_dpp_sampler_eig_GS_bis(eig_vecs)

        elif mode == 'KuTa12':  # cf Kulesza-Taskar
            sampl = proj_dpp_sampler_eig_KuTa12(eig_vecs)

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
def proj_dpp_sampler_eig_GS(eig_vecs, size=None):
    """ Sample from projection :math:`\operatorname{DPP}(K)` using the eigendecomposition of the projection kernel :math:`K=VV^{\top}` where :math:`V^{\top}V = I_r` and :math:`r=\operatorname{rank}(\mathbf{K})`.
    It performs sequential update of Cholesky decomposition, which is equivalent to Gram-Schmidt orthogonalization of the rows of the eigenvectors.

    :param eig_vecs:
        Eigenvectors used to form projection kernel :math:`K=VV^{\top}`.
    :type eig_vecs:
        array_like

    :return:
        A sample from projection :math:`\operatorname{DPP}(K)`.
    :rtype:
        list, array_like

    .. seealso::

        - :func:`proj_dpp_sampler_eig_GS_bis <proj_dpp_sampler_eig_GS_bis>`
        - :func:`proj_dpp_sampler_eig_KuTa12 <proj_dpp_sampler_eig_KuTa12>`
    """

    # Initialization
    V = eig_vecs

    N, rank = V.shape  # ground set size / rank(K)
    ground_set = np.arange(N)

    size = rank if size is None else size  # Full projection DPP or k-DPP
    sampl = np.zeros(size, dtype=int)

    avail = np.ones(N, dtype=bool)  # available items

    # Phase 1: Already performed!
    # Select eigvecs with Bernoulli variables with parameter = eigvals of K.

    # Phase 2: Chain rule
    # Use Gram-Schmidt recursion to compute the Vol^2 of the parallelepiped spanned by the feature vectors associated to the sample

    c = np.zeros((N, size))
    norms_2 = inner1d(V, axis=1)  # ||V_i:||^2

    for it in range(size):
        # Pick an item \propto this squred distance
        j = np.random.choice(ground_set[avail],
                             size=1,
                             p=np.abs(norms_2[avail]) / (rank - it))[0]
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
def proj_dpp_sampler_eig_GS_bis(eig_vecs, size=None):
    """ Sample from projection :math:`\operatorname{DPP}(K)` using the eigendecomposition of the projection kernel :math:`K=VV^{\top}` where :math:`V^{\top}V = I_r` and :math:`r=\operatorname{rank}(\mathbf{K})`.
    It performs sequential Gram-Schmidt orthogonalization of the rows of the eigenvectors.

    :param eig_vecs:
        Eigenvectors used to form projection kernel :math:`K=VV^{\top}`.
    :type eig_vecs:
        array_like

    :return:
        A sample from projection :math:`\operatorname{DPP}(K)`.
    :rtype:
        list, array_like

    .. seealso::

        - This is a slight modification of :func:`proj_dpp_sampler_eig_GS <proj_dpp_sampler_eig_GS>`
        - :func:`proj_dpp_sampler_eig_KuTa12 <proj_dpp_sampler_eig_KuTa12>`
    """

    # Initialization
    V = eig_vecs.copy()
    N, rank = V.shape  # ground set size / rank(K)
    ground_set = np.arange(N)

    # Sample
    size = rank if size is None else size  # Full projection DPP or k-DPP
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
        j = np.random.choice(ground_set[avail],
                             size=1,
                             p=np.abs(norms_2[avail]) / (rank - it))[0]
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
        norms_2[avail] -= (contrib[avail, it]**2) / norms_2[j]

    return sampl


def proj_dpp_sampler_eig_KuTa12(eig_vecs, size=None):
    """ Sample from :math:`\operatorname{DPP}(K)` using the eigendecomposition of the similarity kernel :math:`K`.
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
        A sample from :math:`\operatorname{DPP}(K)`.
    :rtype:
        list

    .. seealso::

        - Algorithm 1 in :cite:`KuTa12`
        - :func:`proj_dpp_sampler_eig_GS <proj_dpp_sampler_eig_GS>`
        - :func:`proj_dpp_sampler_eig_GS_bis <proj_dpp_sampler_eig_GS_bis>`
    """

    # Initialization
    V = eig_vecs.copy()
    N, rank = V.shape  # ground set size / rank(K)

    size = rank if size is None else size  # Full projection DPP or k-DPP
    sampl = np.zeros(size, dtype=int)  # sample list

    # Phase 1: Already performed!
    # Select eigvecs with Bernoulli variables with parameter the eigvals

    # Phase 2: Chain rule
    norms_2 = inner1d(V, axis=1)  # ||V_i:||^2

    # Following [Algo 1, KuTa12], the aim is to compute the orhto complement of the subspace spanned by the selected eigenvectors to the canonical vectors \{e_i ; i \in Y\}. We proceed recursively.
    for it in range(size):

        j = np.random.choice(N, size=1, p=np.abs(norms_2) / (rank - it))[0]
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

# def k_dpp_sampler(kernel, size, projection=False, mode='GS'):
#   """ Sample from :math:`\operatorname{DPP}(K)`, where :math:`K` is real symmetric with eigenvalues in :math:`[0,1]`.

#   :param kernel: Real symmetric kernel with eigenvalues in :math:`[0,1]`
#   :type kernel:
#       array_like

#   :param projection:
#       Indicate :math:`K` is an orthogonal projection kernel.
#       If ``projection=True``, diagonalization of :math:`K` is not necessary, thus not performed.
#   :type projection:
#       bool, default 'False'

#   :param mode:
#       Indicate how the conditional probabilities i.e. the ratio of 2 determinants must be updated.

#       If ``projection=True``:
#           - 'GS' (default): Gram-Schmidt on the columns of :math:`K`
#           # - 'Schur': Schur complement updates

#       If ``projection=False``:
#           - 'GS' (default): Gram-Schmidt on the columns of :math:`K` equiv
#           - 'GS_bis': Slight modif of 'GS'
#           - 'KuTa12': Algorithm 1 in :cite:`KuTa12`
#   :type mode:
#       string, default 'GS_bis'

#   :return:
#       A sample from :math:`\operatorname{DPP}(K)`.
#   :rtype:
#       list

#   .. seealso::

#       - :func:`proj_k_dpp_sampler <proj_k_dpp_sampler>`
#       - :func:`k_dpp_sampler_eig <k_dpp_sampler_eig>`
#   """

#   if projection:
#       sampl = proj_k_dpp_sampler_kernel(kernel, size, mode)

#   else:
#       eig_vecs, eig_vals = la.eigh(kernel)
#       sampl = k_dpp_sampler_eig(eig_vals, eig_vecs, size, mode)

#   return sampl

# #########################
# ### Projection kernel ###
# #########################
# def proj_k_dpp_sampler_kernel(kernel, size, mode='GS'):
#   """
#   .. seealso::
#       - :func:`proj_dpp_sampler_kernel_GS_bis <proj_dpp_sampler_kernel_GS_bis>`
#       # - :func:`proj_dpp_sampler_kernel_Schur <proj_dpp_sampler_kernel_Schur>`
#   """

#   #### Phase 1: Select eigenvectors: No need for eigendecomposition!

#   #### Phase 2: Sample from orthogonal projection kernel K = K^2 = K.T K
#   # Chain rule, conditionals are updated using:
#   if mode == 'GS': # Gram-Schmidt equiv Cholesky
#       sampl = proj_dpp_sampler_kernel_GS(kernel, size)

#   # elif mode == 'Shur': # Schur complement
#   #   sampl = proj_dpp_sampler_kernel_Schur(kernel, size)

#   else:
#       str_list = ['Invalid `mode` parameter, choose among:',
#                               '- `GS` (default)',
#                               # '- 'Schur'',
#                               'Given `mode` = {}'.format(mode)]
#       raise ValueError('\n'.join(str_list))

#   return sampl

# #######################################################
# # From the eigen decomposition of the kernel :math:`K`

# ######################
# ### Generic kernel ###
# ######################

# def k_dpp_sampler_eig(eig_vals, eig_vecs, size, mode='GS',
#                                           el_sym_pol_eval=None):
#   """
#   .. seealso::

#       Phase 1:

#       - :func:`k_dpp_eig_vecs_selector <k_dpp_eig_vecs_selector>`

#       Phase 2:

#       - :func:`proj_dpp_sampler_eig_GS_bis <proj_dpp_sampler_eig_GS_bis>`
#       - :func:`proj_dpp_sampler_eig_GS <proj_dpp_sampler_eig_GS>`
#       - :func:`proj_dpp_sampler_eig_KuTa12 <proj_dpp_sampler_eig_KuTa12>`
#   """
#   #### Phase 1: Select eigenvectors
#   eig_vecs_sel = k_dpp_eig_vecs_selector(eig_vals, eig_vecs, size, el_sym_pol_eval)

#   #### Phase 2: Sample from projection kernel VV.T
#   # Chain rule, conditionals are updated using:

#   if mode == 'GS': # Gram-Schmidt
#       sampl = proj_dpp_sampler_eig_GS(eig_vecs_sel)

#   elif mode == 'GS_bis': # Slight modif of 'GS'
#       sampl = proj_dpp_sampler_eig_GS_bis(eig_vecs_sel)

#   elif mode == 'KuTa12': # cf Kulesza-Taskar
#       sampl = proj_dpp_sampler_eig_KuTa12(eig_vecs_sel)

#   else:
#       str_list = ['Invalid `mode` parameter, choose among:',
#                               '- `GS` (default)',
#                               '- `GS_bis`',
#                               '- `KuTa12`',
#                               'Given {}'.format(mode)]
#       raise ValueError('\n'.join(str_list))

#   return sampl

# def k_dpp_eig_vecs_selector(eig_vals, eig_vecs, size, el_sym_pol_eval=None):
#   """ Subsample eigenvectors V of the initial kernel ('K' or equivalently 'L') to build a projection DPP with kernel V V.T from which sampling is easy. The selection is made based a realization of Bernoulli variables with parameters the eigenvalues of 'K'.

#   :param eig_vals:
#       Collection of eigen values of 'K' (inclusion) kernel.
#   :type eig_vals:
#       list, array_like

#   :param eig_vecs:
#       Collection of eigenvectors of 'K' (or equiv 'L') kernel.
#   :type eig_vals:
#       array_like

#   :return:
#       Selected eigenvectors
#   :rtype:
#       array_like

#   .. seealso::

#       Algorithm 8 in :cite:`KuTa12`
#   """

#   # Size of the ground set
#   nb_items = eig_vecs.shape[0]

#   # Evaluate the elem symm polys in the eigenvalues
#   if el_sym_pol_eval is None:
#       E = elem_symm_poly(eig_vals, size)
#   else:
#       E = el_sym_pol_eval

#   ind_selected = []
#   for n in range(nb_items,0,-1):
#       if size == 0:
#           break

#       if np.random.rand() < eig_vals[n-1]*(E[size-1, n-1]/E[size, n]):
#           ind_selected.append(n-1)
#           size -= 1

#   return eig_vecs[:, ind_selected]

# # Evaluate the elementary symmetric polynomials
# def elem_symm_poly(eig_vals, size):
#   """ Evaluate the elementary symmetric polynomials in the eigenvalues.

#   :param eig_vals:
#       Collection of eigen values of the similarity kernel :math:`K`.
#   :type eig_vals:
#       list

#   :param size:
#       Maximum degree of elementary symmetric polynomial.
#   :type size:
#       int

#   :return:
#       poly(size, N) = :math:`e_size(\lambda_1, \cdots, \lambda_N)`
#   :rtype:
#       array_like

#   .. seealso::

#       Algorithm 7 in :cite:`KuTa12`
#   """

#  Number of variables for the elementary symmetric polynomials to be evaluated
#   N = eig_vals.shape[0]
#   # Initialize output array
#   poly = np.zeros((size+1, N+1))
#   poly[0, :] = 1

#   # Recursive evaluation
#   for l in range(1, size+1):
#       for n in range(1, N+1):
#           poly[l, n] = poly[l, n-1] + eig_vals[n-1] * poly[l-1, n-1]

#   return poly
