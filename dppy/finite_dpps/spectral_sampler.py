import numpy as np
import scipy.linalg as la

from ..utils import check_random_state, inner1d


def spectral_sampler(dpp, random_state=None, **params):
    compute_spectral_sampler_parameters(dpp)
    return do_spectral_sampler(dpp, random_state, **params)


def do_spectral_sampler(dpp, random_state=None, **params):
    eig_vals, eig_vecs = dpp.K_eig_vals, dpp.eig_vecs
    V = select_eigen_vectors(eig_vals, eig_vecs, random_state=random_state)
    sampler = select_projection_dpp_eigen_sampler(params.get("mode"))
    return sampler(V, random_state=random_state)


def select_projection_dpp_eigen_sampler(name):
    samplers = {
        "GS": proj_dpp_sampler_eig_GS,
        "GS_bis": proj_dpp_sampler_eig_GS_bis,
        "KuTa12": proj_dpp_sampler_eig_KuTa12,
    }
    default = samplers["GS"]
    return samplers.get(name, default)


def compute_spectral_sampler_parameters(dpp):
    """Compute eigenvalues and eigenvectors of correlation kernel K from various parametrizations of ``dpp``

    :param dpp: ``FiniteDPP`` object
    :type dpp: ``FiniteDPP`` object
    """
    while compute_spectral_sampler_parameters_step(dpp):
        pass


def compute_spectral_sampler_parameters_step(dpp):
    """
    Returns
    ``False`` if the right parameters are indeed computed
    ``True`` if extra computations are required

    Note: Sort of fixed point algorithm to find dpp.K_eig_vals and dpp.eig_vecs
    """
    if dpp.K_eig_vals is not None:
        return False

    if dpp.L_eig_vals is not None:
        dpp.K_eig_vals = dpp.L_eig_vals / (1.0 + dpp.L_eig_vals)
        return False

    elif dpp.K is not None:  # 0 <= K <= I
        eig_vals, dpp.eig_vecs = la.eigh(dpp.K)
        np.clip(eig_vals, 0.0, 1.0, out=eig_vals)
        dpp.K_eig_vals = eig_vals
        return False

    elif dpp.L_dual is not None:
        # L_dual = Phi * Phi.T = W Theta W.T
        # L = Phi.T Phi = V Gamma V
        # then Gamma = Theta and V = Phi.T W Theta^{-1/2}
        eig_vals, eig_vecs = la.eigh(dpp.L_dual)
        np.fmax(eig_vals, 0.0, out=eig_vals)
        dpp.L_eig_vals = eig_vals
        dpp.eig_vecs = dpp.L_gram_factor.T.dot(eig_vecs / np.sqrt(eig_vals))
        return True

    elif dpp.L is not None:
        eig_vals, dpp.eig_vecs = la.eigh(dpp.L)
        np.fmax(eig_vals, 0.0, out=eig_vals)
        dpp.L_eig_vals = eig_vals
        return True

    elif dpp.A_zono is not None:  # K = A.T (A A.T)^-1 A (orthogonal projection)
        A = dpp.A_zono
        dpp.K_eig_vals = np.ones(len(A), dtype=float)
        dpp.eig_vecs, _ = la.qr(A.T, mode="economic")
        return False

    elif dpp.eval_L is not None and dpp.X_data is not None:
        dpp.compute_L()
        return True

    else:
        raise ValueError(
            "None of the available samplers could be used based on the current DPP representation. This should never happen, please consider rasing an issue on github at https://github.com/guilgautier/DPPy/issues"
        )


# Phase 1
def select_eigen_vectors(bernoulli_params, eig_vecs, random_state=None):
    """Select columns of ``eig_vecs`` by sampling Bernoulli variables with parameters ``bernoulli_params``.

    :param bernoulli_params:
        Parameters of Bernoulli variables
    :type bernoulli_params:
        array_like, shape (r,)

    :param eig_vecs:
        Eigenvectors, stored as columns of a 2d array
    :type eig_vecs:
        array_like, shape (N, r)

    :return:
        Selected eigenvectors
    :rtype:
        array_like

    .. seealso::

        - :func:`dpp_sampler_eig <dpp_sampler_eig>`
    """
    rng = check_random_state(random_state)
    mask = rng.rand(bernoulli_params.size) < bernoulli_params
    return eig_vecs[:, mask]


# Phase 2: sample from the projection DPP selected in phase 1
def proj_dpp_sampler_eig_GS(eig_vecs, size=None, random_state=None):
    """Generate an exact sample from projection :math:`\\operatorname{DPP}(K)` with orthogonal projection kernel :math:`K=VV^{\\top}` where :math:`V=` ``eig_vecs`` such that :math:`V^{\\top}V = I_r` and :math:`r=\\operatorname{rank}(\\mathbf{K})`.
    Performs sequential Gram-Schmidt (GS) orthogonalization of the rows of the eigenvectors corresponding to the sampled items.

    :param eig_vecs:
        Eigenvectors used to form projection kernel :math:`K=VV^{\\top}`.
    :type eig_vecs:
        array_like

    :return:
        A sample from projection :math:`\\operatorname{DPP}(K)`.
    :rtype:
        list

    .. seealso::

        - :cite:`Gil14` Algorithm 2 or :cite:`TrBaAm18` Algorithm 3
        - :func:`proj_dpp_sampler_eig_GS_bis <proj_dpp_sampler_eig_GS_bis>`
        - :func:`proj_dpp_sampler_eig_KuTa12 <proj_dpp_sampler_eig_KuTa12>`
    """
    if not eig_vecs.shape[1]:
        return []  # np.empty((0,), dtype=int)

    rng = check_random_state(random_state)

    V = eig_vecs
    N, rank = V.shape  # ground set size / rank(K)
    if size is None:  # full projection DPP else k-DPP with k = size
        size = rank

    ground_set = np.arange(N)
    sample = np.zeros(size, dtype=int)
    avail = np.full(N, fill_value=True, dtype=bool)

    # Phase 1: Already performed!
    # Select eigvecs with Bernoulli variables with parameter = eigvals of K.

    # Phase 2: Chain rule
    # Use Gram-Schmidt recursion to compute the Vol^2 of the parallelepiped spanned by the feature vectors associated to the sample

    c = np.zeros((N, size))
    norms_2 = inner1d(V, axis=1)  # ||V_i:||^2

    for it in range(size):
        # Pick an item \propto this squared distance
        j = rng.choice(ground_set[avail], p=np.abs(norms_2[avail]) / (rank - it))
        sample[it] = j
        avail[j] = False

        if it == size - 1:
            break

        # Cancel the contribution of V_j to the remaining feature vectors
        c[avail, it] = (
            V[avail, :].dot(V[j, :]) - c[avail, :it].dot(c[j, :it])
        ) / np.sqrt(norms_2[j])

        norms_2[avail] -= c[avail, it] ** 2  # update residual norm^2

    return sample.tolist()


def proj_dpp_sampler_eig_GS_bis(eig_vecs, size=None, random_state=None):
    """Sample from projection :math:`\\operatorname{DPP}(K)` using the eigendecomposition of the orthogonal projection kernel :math:`K=VV^{\\top}` where :math:`V^{\\top}V = I_r` and :math:`r=\\operatorname{rank}(\\mathbf{K})`.
    Sequential Gram-Schmidt orthogonalization is performed on the rows of the matrix of eigenvectors corresponding to the sampled items.
    This is a slight modification of :func:`proj_dpp_sampler_eig_GS <proj_dpp_sampler_eig_GS>`.

    :param eig_vecs:
        Eigenvectors of the projection kernel :math:`K=VV^{\\top}`.
    :type eig_vecs:
        array_like

    :return:
        A sample from projection :math:`\\operatorname{DPP}(K)`.
    :rtype:
        list

    .. seealso::

        - :func:`proj_dpp_sampler_eig_GS <proj_dpp_sampler_eig_GS>`
        - :func:`proj_dpp_sampler_eig_KuTa12 <proj_dpp_sampler_eig_KuTa12>`
    """
    if not eig_vecs.shape[1]:
        return []  # np.empty((0,), dtype=int)

    rng = check_random_state(random_state)

    V = eig_vecs.copy()
    N, rank = V.shape  # ground set size / rank(K)
    if size is None:  # full projection DPP else k-DPP with k = size
        size = rank

    ground_set = np.arange(N)
    sample = np.zeros(size, dtype=int)
    avail = np.full(N, fill_value=True, dtype=bool)

    # Chain rule
    # Use Gram-Schmidt recursion to compute the Vol^2 of the parallelepiped spanned by the feature vectors associated to the sample

    # Matrix of the contribution of remaining vectors
    # <V_i, P_{V_Y}^{orthog} V_j>
    contrib = np.zeros((N, size), dtype=float)
    norms_2 = inner1d(V, axis=1)  # ||V_i:||^2

    for it in range(size):

        # Pick an item proportionally to the residual norm^2
        # ||P_{V_Y}^{orthog} V_j||^2
        j = rng.choice(ground_set[avail], p=np.abs(norms_2[avail]) / (rank - it))
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
        avail[j] = False
        V[j, :] -= contrib[j, :it].dot(V[sample[:it], :])

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
        norms_2[avail] -= contrib[avail, it] ** 2 / norms_2[j]

    return sample.tolist()


def proj_dpp_sampler_eig_KuTa12(eig_vecs, size=None, random_state=None):
    """Generate an exact sample from projection :math:`\\operatorname{DPP}(K)` with orthogonal projection kernel :math:`K=VV^{\\top}` where :math:`V=` ``eig_vecs`` such that :math:`V^{\\top}V = I_r` and :math:`r=\\operatorname{rank}(\\mathbf{K})`.
    This corresponds to :cite:`KuTa12` Algorithm 1.

    :param eig_vals:
        Collection of eigenvalues of the similarity kernel :math:`K`.
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
    if not eig_vecs.shape[1]:
        return []  # np.empty((0,), dtype=int)

    rng = check_random_state(random_state)

    V = eig_vecs.copy()
    N, rank = V.shape  # ground set size / rank(K)
    if size is None:  # full projection DPP else k-DPP with k = size
        size = rank

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

    return sample.tolist()
