import numpy as np
import scipy.linalg as la

from dppy.utils import check_random_state, inner1d


def select_projection_sampler_eigen(mode):
    r"""Select the variant of the spectral method applied to projection :math:`\operatorname{DPP}(\mathbf{K})` with :math:`\mathbf{K} = U U^{*}`.

    :param mode: variant name, default "GS"
    :type mode: str

    :return: sampler selected by ``mode``

        - ``"GS"`` :py:func:`~dppy.finite.exact_samplers.projection_sampler_eigen.projection_eigen_sampler_GS`
        - ``"GS_bis"`` :py:func:`~dppy.finite.exact_samplers.projection_sampler_eigen.projection_eigen_sampler_GS_bis`
        - ``"KuTa12"`` :py:func:`~dppy.finite.exact_samplers.projection_sampler_eigen.projection_eigen_sampler_KuTa12`

    :rtype: callable
    """
    samplers = {
        "GS": projection_eigen_sampler_GS,
        "GS_bis": projection_eigen_sampler_GS_bis,
        "KuTa12": projection_eigen_sampler_KuTa12,
    }
    default = samplers["GS"]
    return samplers.get(mode, default)


# Phase 2: sample from the projection DPP selected in phase 1
def projection_eigen_sampler_GS(eig_vecs, size=None, random_state=None):
    r"""Generate an exact sample from projection :math:`\operatorname{DPP}(K)` with orthogonal projection kernel :math:`K=VV^{\top}` where :math:`V=` ``eig_vecs`` such that :math:`V^{\top}V = I_r` and :math:`r=\operatorname{rank}(K)`.

    :param eig_vecs:
        Eigenvectors :math:`V` of :math:`K=VV^{\top}`.
    :type eig_vecs:
        array_like

    :param size:
        Size of the output sample (if ``size`` is provided), otherwise :math:`k=\operatorname{trace}(K)=\operatorname{rank}(K)=r`.
    :type size:
        int

    :return:
        An exact sample from the corresponding :math:`\operatorname{DPP}(K)` or :math:`\operatorname{k-DPP}(K)`.
    :rtype:
        list

    .. seealso::

        - :cite:`Gil14` Algorithm 2 or :cite:`TrBaAm18` Algorithm 3
        - :func:`projection_eigen_sampler_GS_bis <projection_eigen_sampler_GS_bis>`
        - :func:`projection_eigen_sampler_KuTa12 <projection_eigen_sampler_KuTa12>`
    """
    if not eig_vecs.shape[1]:
        return []  # np.empty((0,), dtype=int)

    rng = check_random_state(random_state)

    V = eig_vecs
    N, rank = V.shape  # ground set size / rank(K)
    if size is None:  # full projection DPP else k-DPP with k = size
        size = rank
    assert size <= rank

    ground_set = np.arange(N)
    sample = np.zeros(size, dtype=int)
    rem = np.full(N, fill_value=True, dtype=bool)

    # Phase 1: Already performed!
    # Select eigvecs with Bernoulli variables with parameter = eigvals of K.

    # Phase 2: Chain rule
    # Use Gram-Schmidt recursion to compute the Vol^2 of the parallelepiped spanned by the feature vectors associated to the sample

    c = np.zeros((N, size))
    norms_2 = inner1d(V, axis=1)  # ||V_i:||^2

    for it in range(size):
        # Pick an item \propto this squared distance
        j = rng.choice(ground_set[rem], p=np.abs(norms_2[rem]) / (rank - it))
        sample[it] = j
        rem[j] = False

        if it == size - 1:
            break

        # Cancel the contribution of V_j to the remaining feature vectors
        c[rem, it] = V[rem, :].dot(V[j, :]) - c[rem, :it].dot(c[j, :it])
        c[rem, it] /= np.sqrt(norms_2[j])

        norms_2[rem] -= c[rem, it] ** 2  # update residual norm^2

    # log_likelihood = np.sum(np.log(norm_2[sample]))
    return sample.tolist()  # , log_likelihood


def projection_eigen_sampler_GS_bis(eig_vecs, size=None, random_state=None):
    r"""Generate an exact sample from projection :math:`\operatorname{DPP}(K)` with orthogonal projection kernel :math:`K=VV^{\top}` where :math:`V=` ``eig_vecs`` such that :math:`V^{\top}V = I_r` and :math:`r=\operatorname{rank}(K)`.

    This function is a slight modification of :func:`projection_eigen_sampler_GS <projection_eigen_sampler_GS>`.

    :param eig_vecs:
        Eigenvectors :math:`V` of :math:`K=VV^{\top}`.
    :type eig_vecs:
        array_like

    :param size:
        Size of the output sample (if ``size`` is provided), otherwise :math:`k=\operatorname{trace}(K)=\operatorname{rank}(K)=r`.
    :type size:
        int

    :return:
        An exact sample from the corresponding :math:`\operatorname{DPP}(K)` or :math:`\operatorname{k-DPP}(K)`.
    :rtype:
        list

    .. seealso::

        - :func:`projection_eigen_sampler_GS <projection_eigen_sampler_GS>`
        - :func:`projection_eigen_sampler_KuTa12 <projection_eigen_sampler_KuTa12>`
    """
    if not eig_vecs.shape[1]:
        return []  # np.empty((0,), dtype=int)

    rng = check_random_state(random_state)

    V = eig_vecs.copy()
    N, rank = V.shape  # ground set size / rank(K)
    if size is None:  # full projection DPP else k-DPP with k = size
        size = rank
    assert size <= rank

    ground_set = np.arange(N)
    sample = np.zeros(size, dtype=int)
    rem = np.full(N, fill_value=True, dtype=bool)

    # Chain rule
    # Use Gram-Schmidt recursion to compute the Vol^2 of the parallelepiped spanned by the feature vectors associated to the sample

    # Matrix of the contribution of remaining vectors
    # <V_i, P_{V_Y}^{orthog} V_j>
    c = np.zeros((N, size), dtype=float)
    norms_2 = inner1d(V, axis=1)  # ||V_i:||^2

    for it in range(size):

        # Pick an item proportionally to the residual norm^2
        # ||P_{V_Y}^{orthog} V_j||^2
        j = rng.choice(ground_set[rem], p=np.abs(norms_2[rem]) / (rank - it))
        sample[it] = j
        if it == size - 1:
            break
        # Update the residual norm^2
        #
        # |P_{V_Y+j}^{orthog} V_i|^2
        #                                    <V_i,P_{V_Y}^{orthog} V_j>^2
        #     =  |P_{V_Y}^{orthog} V_i|^2 -  ----------------------------
        #                                      |P_{V_Y}^{orthog} V_j|^2
        #
        # 1) Orthogonal part of V_j w.r.t. orthonormal basis of Span(V_Y)
        #    V'_j = P_{V_Y}^{orthog} V_j
        #         = V_j - <V_j,sum_Y V'_k>V'_k
        #         = V_j - sum_Y <V_j, V'_k> V'_k
        # Note V'_j is not normalized
        rem[j] = False
        V[j, :] -= c[j, :it].dot(V[sample[:it], :])

        # 2) Compute <V_i, V'_j> = <V_i, P_{V_Y}^{orthog} V_j>
        c[rem, it] = V[rem, :].dot(V[j, :])

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
        norms_2[rem] -= c[rem, it] ** 2 / norms_2[j]

    # log_likelihood = np.sum(np.log(norm_2[sample]))
    return sample.tolist()  # , log_likelihood


def projection_eigen_sampler_KuTa12(eig_vecs, size=None, random_state=None):
    r"""Generate an exact sample from projection :math:`\operatorname{DPP}(K)` with orthogonal projection kernel :math:`K=VV^{\top}` where :math:`V=` ``eig_vecs`` such that :math:`V^{\top}V = I_r` and :math:`r=\operatorname{rank}(K)`.

    This function implements Algorithm 1 :cite:`KuTa12`.

    :param eig_vecs:
        Eigenvectors :math:`V` of :math:`K=VV^{\top}`.
    :type eig_vecs:
        array_like

    :param size:
        Size of the output sample (if ``size`` is provided), otherwise :math:`k=\operatorname{trace}(K)=\operatorname{rank}(K)=r`.
    :type size:
        int

    :return:
        An exact sample from the corresponding :math:`\operatorname{DPP}(K)` or :math:`\operatorname{k-DPP}(K)`.
    :rtype:
        list

    .. seealso::

        - :func:`projection_eigen_sampler_GS <projection_eigen_sampler_GS>`
        - :func:`projection_eigen_sampler_GS_bis <projection_eigen_sampler_GS_bis>`
    """
    if not eig_vecs.shape[1]:
        return []  # np.empty((0,), dtype=int)

    rng = check_random_state(random_state)

    V = eig_vecs.copy()
    N, rank = V.shape  # ground set size / rank(K)
    if size is None:  # full projection DPP else k-DPP with k = size
        size = rank
    assert size <= rank

    sample = np.full(size, fill_value=0, dtype=int)
    norms_2 = inner1d(V, axis=1)  # ||V_i:||^2

    # Following [Algo 1, KuTa12], the aim is to compute the ortho complement of the subspace spanned by the selected eigenvectors to the canonical vectors \{e_i ; i \in Y\}. We proceed recursively.
    for it in range(size):

        j = rng.choice(N, p=np.abs(norms_2) / (rank - it))
        sample[it] = j
        if it == size - 1:
            break

        # Cancel the contribution of e_i to the remaining vectors that is, find the subspace of V that is orthogonal to \{e_i ; i \in Y\}
        # Take the index of a vector that has a non null contribution on e_j
        k = np.where(V[j, :] != 0)[0][0]
        # Cancel the contribution of the remaining vectors on e_j, but stay in the subspace spanned by V i.e. get the subspace of V orthogonal to \{e_i ; i \in Y\}
        V -= np.outer(V[:, k] / V[j, k], V[j, :])
        # V_:j is set to 0 so we delete it and we can derive an orthononormal basis of the subspace under consideration
        V, _ = la.qr(np.delete(V, k, axis=1), mode="economic")

        norms_2 = inner1d(V, axis=1)  # ||V_i:||^2

    # log_likelihood = np.sum(np.log(norm_2[sample]))
    return sample.tolist()  # , log_likelihood
