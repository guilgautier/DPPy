# coding: utf-8

import numpy as np
from numpy.core.umath_tests import inner1d as np_inner1d
import scipy.linalg as la

########
# DPPs #
########


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
        str_list = ['Invalid `mode` parameter, choose among:',
                    '- `GS` (default)',
                    # '- 'Schur'',
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
    # Size of the ground set
    N = K.shape[0]
    # Maximal size of the sample: Tr(K)=rank(K)
    rank = int(np.round(np.trace(K)))
    # Size of the sample
    if size is None:
        size = rank  # Full projection DPP
    else:
        pass  # projection k-DPP
    # Sample
    Y = []

    ground_set, rem_set = np.arange(N), np.full(N, True)

    c = np.zeros((N, size))
    d_2 = K.diagonal().copy()

    for it in range(size):

        j = np.random.choice(ground_set[rem_set],
                             size=1,
                             p=np.fabs(d_2[rem_set]) / (rank - it))[0]
        # Add the item to the sample
        rem_set[j] = False
        Y.append(j)

        # Update the Cholesky factor
        c[rem_set, it] = K[rem_set, j] - c[rem_set, :it].dot(c[j, :it])
        c[rem_set, it] /= np.sqrt(d_2[j])

        d_2[rem_set] -= c[rem_set, it]**2

    return Y


def proj_dpp_sampler_kernel_Schur(K, size=None):
    """ Sample from :math:`\operatorname{k-DPP}(K)` where the similarity kernel :math:`K`
    is an orthogonal projection matrix.
    It sequentially updates the Schur complement by updating the inverse of the matrix involved.
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
        If ``k`` is not provided (None),
            A sample from :math:`\operatorname{DPP}(K)`.
        If ``k`` is provided,
            A sample from :math:`\operatorname{k-DPP}(K)`.
    :rtype:
        list
    .. seealso::
        - :func:`projection_dpp_sampler_GS_bis <projection_dpp_sampler_GS_bis>`
    """

    # Initialization
    # Size of the ground set
    N = K.shape[0]
    # Maximal size of the sample: Tr(K)=rank(K)
    rank = int(np.round(np.trace(K)))
    # Size of the sample
    if size is None:
        size = rank  # Full projection DPP
    else:
        pass  # projection k-DPP
    ground_set, rem_set = np.arange(N), np.full(N, True)
    # Sample
    Y = []

    K_diag = K.diagonal()  # Used to compute the first term of Schur complement
    schur_comp = K_diag.copy()

    for it in range(size):
        # Pick a new item
        j = np.random.choice(ground_set[rem_set],
                             size=1,
                             p=np.fabs(schur_comp[rem_set]) / (rank - it))[0]

        # Update Schur complements K_ii - K_iY (K_Y)^-1 K_Yi for Y <- Y+j
        #
        # 1) use Woodbury identity to update K[Y,Y]^-1 to K[Y+j,Y+j]^-1
        # K[Y+j,Y+j]^-1 =
        # [ K[Y,Y]^-1 + (K[Y,Y]^-1 K[Y,j] K[j,Y] K[Y,Y]^-1)/schur_j,
        #       -K[Y,Y]^-1 K[Y,j]/schur_j]
        # [ -K[j,Y] K[Y,Y]^-1/schur_j,
        #       1/schur_j]

        if it == 0:
            K_inv = 1.0 / K[j, j]
        elif it == 1:
            K_inv = np.array([[K[j, j], -K[j, Y]],
                             [-K[j, Y], K[Y, Y]]])\
                / (K[Y, Y] * K[j, j] - K[j, Y]**2)
        else:
            schur_j = K[j, j] - K[j, Y].dot(K_inv.dot(K[Y, j]))
            temp = K_inv.dot(K[Y, j])

            K_inv = np.lib.pad(K_inv, (0, 1),
                               'constant',
                               constant_values=1.0 / schur_j)

            K_inv[:-1, :-1] += np.outer(temp, temp / schur_j)
            K_inv[:-1, -1] *= -temp
            K_inv[-1, :-1] = K_inv[:-1, -1]
            # K_inv[-1,-1] = 1.0/schur_j

        # Add the item to the sample
        rem_set[j] = False
        Y.append(j)

        # 2) update Schur complements
        # K_ii - K_iY (K_Y)^-1 K_Yi for Y <- Y+j
        schur_comp[rem_set] =\
            K_diag[rem_set] - np_inner1d(K[np.ix_(rem_set, Y)],
                                         K[np.ix_(rem_set, Y)].dot(K_inv))

    return Y


##################
# Generic kernel #
##################


# Phase 1: subsample eigenvectors by drawing independent Bernoulli variables with parameter the eigenvalues of the inclusion kernel K.
def dpp_eig_vecs_selector(ber_params, eig_vecs):
    """ Phase 1 of exact sampling procedure. Subsample eigenvectors :math:`V` of the initial kernel (inclusion :math:`K`, resp. marginal :math:`L`) to build a projection DPP with kernel :math:`V V^{\top}` from which sampling is easy.
    The selection is made based on a realization of Bernoulli variables with parameters related to the eigenvalues of :math:`K`, resp. :math:`L`.

    :param ber_params:
        Parameters of Bernoulli variables:
        .. math::

            \lambda^K=\lambda^L/(1+\lambda^L)
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
    ind_sel = np.random.rand(len(ber_params)) < ber_params

    return eig_vecs[:, ind_sel]


def dpp_eig_vecs_selector_L_dual(eig_vals, eig_vecs, gram_factor):
    """ Subsample eigenvectors :math:`V` of marginal kernel :math:`L=\Phi \Phi^{\top}` based on the eigendecomposition dual kernel :math:`L'=\Phi \Phi^{\top}`.

    :param eig_vals:
        Collection of eigenvalues of :math:`L` or :math:`L_dual` kernel.
    :type eig_vals:
        list, array_like

    :param eig_vecs:
        Collection of eigenvectors of :math:`L_dual` kernel.
    :type eig_vecs:
        array_like

    :param gram_factor:
        Feature vectors
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
    ind_sel = np.random.rand(len(eig_vals)) < eig_vals / (1.0 + eig_vals)

    return gram_factor.T.dot(eig_vecs[:, ind_sel] / np.sqrt(eig_vals[ind_sel]))


# Phase 2: sample projection kernel VV.T where V are the eigenvectors selected in Phase 1.
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
            str_list = ['Invalid `mode` parameter, choose among:',
                        '- `GS` (default)',
                        '- `GS_bis`',
                        '- `KuTa12`',
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
    V = eig_vecs  # Eigenvectors
    N, rank = V.shape  # N = size of the ground set, r = rank(K)
    # Size of the sample
    if size is None:
        size = rank  # Full projection DPP
    else:
        pass  # projection k-DPP
    ground_set, rem_set = np.arange(N), np.full(N, True)
    Y = []  # sample

    # Phase 1: Select eigenvectors with Bernoulli variables with parameter the eigenvalues. Already performed!

    # Phase 2: Chain rule
    # To compute the squared volume of the parallelepiped spanned by the feature vectors defining the sample
    # use Gram-Schmidt recursion aka Base x Height formula.

    # Initially this corresponds to the squared norm of the feature vectors
    c = np.zeros((N, size))
    norms_2 = np_inner1d(V, V)

    for it in range(size):

        # Pick an item \propto this squred distance
        j = np.random.choice(ground_set[rem_set],
                             size=1,
                             p=np.fabs(norms_2[rem_set]) / (rank - it))[0]

        # Add the item just picked
        rem_set[j] = False
        Y.append(j)

        # Cancel the contribution of V_j to the remaining feature vectors
        c[rem_set, it] = \
            V[rem_set, :].dot(V[j, :]) - c[rem_set, :it].dot(c[j, :it])
        c[rem_set, it] /= np.sqrt(norms_2[j])

        # Compute the square distance of the feature vectors to Span(V_Y:)
        norms_2[rem_set] -= c[rem_set, it]**2

    return Y


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
    V = eig_vecs  # Eigenvectors
    N, rank = V.shape  # N = size of the ground set, r = rank(K)
    # Size of the sample
    if size is None:
        size = rank  # Full projection DPP
    else:
        pass  # projection k-DPP
    ground_set, rem_set = np.arange(N), np.full(N, True)
    # Sample
    Y = []

    ##### Phase 1: Select eigenvectors with Bernoulli variables with parameter the eigenvalues. Already performed!

    # Phase 2: Chain rule
    # To compute the squared volume of the parallelepiped spanned by the feature vectors defining the sample
    # use Gram-Schmidt recursion aka Base x Height formula.

    # Matrix of the contribution of remaining vectors V_i onto the orthonormal basis {e_j}_Y of V_Y
    # <V_i,P_{V_Y}^{orthog} V_j>
    contrib = np.zeros((N, size))

    # Residual square norm
    # ||P_{V_Y}^{orthog} V_j||^2
    norms_2 = np_inner1d(V, V)

    for it in range(size):

        # Pick an item proportionally to the residual square norm
        # ||P_{V_Y}^{orthog} V_j||^2
        j = np.random.choice(ground_set[rem_set],
                             size=1,
                             p=np.fabs(norms_2[rem_set]) / (rank - it))[0]

        # Update the residual square norm
        #
        # |P_{V_Y+j}^{orthog} V_i|^2
        #                                    <V_i,P_{V_Y}^{orthog} V_j>^2
        #     =  |P_{V_Y}^{orthog} V_i|^2 -  ----------------------------
        #                                      |P_{V_Y}^{orthog} V_j|^2

        # 1) Orthogonalize V_j w.r.t. orthonormal basis of Span(V_Y)
        #    V'_j = P_{V_Y}^{orthog} V_j
        #         = V_j - <V_j,sum_Y V'_k>V'_k
        # Note V'_j is not normalized
        V[j, :] -= contrib[j, :it].dot(V[Y, :])

        # Make the item selected unavailable
        rem_set[j] = False
        # Add the item to the sample
        Y.append(j)

        # 2) Compute <V_i,V'_j> = <V_i,P_{V_Y}^{orthog} V_j>
        contrib[rem_set, it] = V[rem_set, :].dot(V[j, :])

        # 3) Normalize V'_j with norm^2 and not norm
        #              V'_j         P_{V_Y}^{orthog} V_j
        #    V'_j  =  -------  =  --------------------------
        #             |V'j|^2      |P_{V_Y}^{orthog} V_j|^2
        V[j, :] /= norms_2[j]
        # for next orthogonalization in 1)
        #  <V_i,P_{V_Y}^{orthog} V_j> P_{V_Y}^{orthog} V_j
        #  V_i - <V_i,V'_j>V'_j = V_i - |P_{V_Y}^{orthog} V_j|^2
        #
        #
        # 4) Update the residual norm^2: cancel contribution of V_i onto V_j
        #
        # |P_{V_Y+j}^{orthog} V_i|^2
        #   = |P_{V_Y}^{orthog} V_i|^2 - <V_i,V'_j>^2 / |V'j|^2
        #                                  <V_i,P_{V_Y}^{orthog} V_j>^2
        #   =  |P_{V_Y}^{orthog} V_i|^2 -  ----------------------------
        #                                   |P_{V_Y}^{orthog} V_j|^2

        norms_2[rem_set] -= (contrib[rem_set, it]**2) / norms_2[j]

    return Y


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
    V = eig_vecs  # Eigenvectors
    N, rank = V.shape  # N = size of the ground set, r = rank(K)
    # Size of the sample
    if size is None:
        size = rank  # Full projection DPP
    else:
        pass  # projection k-DPP
    # Sample
    Y = []

    # Phase 1: Select eigenvectors with Bernoulli variables with parameter the eigenvalues. Already performed!

    # Phase 2: Chain rule
    # Initialize the sample
    norms_2 = np_inner1d(V, V)
    # Pick an item
    i = np.random.choice(N, size=1, p=np.fabs(norms_2) / rank)[0]
    # Add the item just picked
    Y.append(i)  # sample

    # Following [Algo 1, KuTa12], the aim is to compute the orhto complement of the subspace spanned by the selected eigenvectors to the canonical vectors \{e_i ; i \in Y\}. We proceed recursively.
    for it in range(1, size):

        # Cancel the contribution of e_i to the remaining vectors that is, find the subspace of V that is orthogonal to \{e_i ; i \in Y\}

        # Take the index of a vector that has a non null contribution along e_i
        j = np.where(V[i, :] != 0)[0][0]
        # Cancel the contribution of the remaining vectors along e_i, but stay in the subspace spanned by V i.e. get the subspace of V orthogonal to \{e_i ; i \in Y\}
        V -= np.outer(V[:, j] / V[i, j], V[i, :])
        # V_:j is set to 0 so we delete it and we can derive an orthononormal basis of the subspace under consideration
        V, _ = la.qr(np.delete(V, j, axis=1), mode='economic')

        norms_2 = np_inner1d(V, V)
        # Pick an item
        i = np.random.choice(N, size=1, p=np.fabs(norms_2) / (rank - it))[0]
        # Add the item just picked
        Y.append(i)

    return Y


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
