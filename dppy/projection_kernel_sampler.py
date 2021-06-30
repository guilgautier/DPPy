import numpy as np
from .utils import check_random_state, inner1d


def projection_kernel_sampler(dpp, random_state=None, **params):
    assert dpp.projection
    dpp.compute_K()
    sampler = select_projection_dpp_kernel_sampler(params.get("mode"))
    return sampler(dpp.K, params.get("size"), random_state)


def select_projection_dpp_kernel_sampler(name):
    """Select a sampler for projection DPP define via its correlation kernel

    - :func:`proj_dpp_sampler_kernel_GS <proj_dpp_sampler_kernel_GS>`
    - :func:`proj_dpp_sampler_kernel_Schur <proj_dpp_sampler_kernel_Schur>`
    - :func:`proj_dpp_sampler_kernel_Chol <proj_dpp_sampler_kernel_Chol>`
    """
    samplers = {
        "GS": proj_dpp_sampler_kernel_GS,
        "Chol": proj_dpp_sampler_kernel_Chol,
        "Schur": proj_dpp_sampler_kernel_Schur,
    }
    default = samplers["GS"]
    return samplers.get(name, default)


def proj_dpp_sampler_kernel_Chol(K, size=None, random_state=None):
    """Sample from:

    - :math:`\\operatorname{DPP}(K)` with orthogonal projection **correlation** kernel :math:`K` if ``size`` is not provided
    - :math:`\\operatorname{k-DPP}` with orthogonal projection **likelihood** kernel :math:`K` with :math:`k=` ``size`` is provided

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

        #L37>`_ for the Hermitian swap routine.
        - :cite:`Pou19` Algorithm 3 and :ref:`catamari code <https://gitlab.com/hodge_star/catamari/blob/38718a1ea34872fb6567e019ece91fbeb5af5be1/include/catamari/dense_dpp/elementary_hermitian_dpp-impl.hpp
        - :func:`proj_dpp_sampler_kernel_GS <proj_dpp_sampler_kernel_GS>`
        - :func:`proj_dpp_sampler_kernel_Schur <proj_dpp_sampler_kernel_Schur>`
    """

    rng = check_random_state(random_state)

    N = len(K)
    rank = np.rint(np.trace(K)).astype(int)
    if size is None:  # full projection DPP else k-DPP with k = size
        size = rank
    assert size <= rank

    A = K.copy()
    d = np.diagonal(A).astype(float)

    ground_set = np.arange(N)

    for j in range(size):

        # Sample from pivot index and permute
        t = rng.choice(range(j, N), p=np.abs(d[j:]) / (rank - j))

        # Hermitian swap of indices j and t of A (may be written in a function)
        # bottom swap
        A[t + 1 :, [j, t]] = A[t + 1 :, [t, j]]
        # inner swap
        tmp = A[j + 1 : t, j].copy()
        np.conj(A[t, j + 1 : t], out=A[j + 1 : t, j])
        np.conj(tmp, out=A[t, j + 1 : t])
        # corner swap
        A[t, j] = A[t, j].conj()
        # diagonal swap
        A[[j, t], [j, t]] = A[[t, j], [t, j]].real
        # left swap
        A[[j, t], :j] = A[[t, j], :j]

        # Swap positions j and t of ground_set and d
        ground_set[[j, t]] = ground_set[[t, j]]
        d[[j, t]] = d[[t, j]]

        A[j, j] = np.sqrt(d[j])

        if j == size - 1:
            break

        # Form new column and update diagonal
        A[j + 1 :, j] -= A[j + 1 :, :j].dot(A[j, :j].conj())
        A[j + 1 :, j] /= A[j, j]

        if np.iscomplexobj(A):
            d[j + 1 :] -= A[j + 1 :, j].real ** 2 + A[j + 1 :, j].imag ** 2
        else:
            d[j + 1 :] -= A[j + 1 :, j] ** 2

    sample = ground_set[:size].tolist()
    log_likelihood = np.sum(np.log(np.diagonal(A[:size, :size]).real))

    return sample  # , log_likelihood


def proj_dpp_sampler_kernel_GS(K, size=None, random_state=None):
    """Sample from:

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

        - :cite:`TrBaAm18` Algorithm 3, :cite:`Gil14` Algorithm 2
        - :func:`proj_dpp_sampler_kernel_Schur <proj_dpp_sampler_kernel_Schur>`
        - :func:`proj_dpp_sampler_kernel_Chol <proj_dpp_sampler_kernel_Chol>`
    """

    rng = check_random_state(random_state)

    N = len(K)  # ground set size
    rank = np.rint(np.trace(K)).astype(int)  # rank(K) = Tr(K)
    if size is None:  # full projection DPP else k-DPP with k = size
        size = rank
    assert size <= rank

    ground_set = np.arange(N)
    sample = np.zeros(size, dtype=int)
    avail = np.full(N, fill_value=True, dtype=bool)

    c = np.zeros((N, size))
    norm_2 = K.diagonal().copy()  # residual norm^2

    for it in range(size):
        j = rng.choice(ground_set[avail], p=np.abs(norm_2[avail]) / (rank - it))

        sample[it] = j
        if it == size - 1:
            break
        # Update the Cholesky factor
        avail[j] = False
        c[avail, it] = (K[avail, j] - c[avail, :it].dot(c[j, :it])) / np.sqrt(norm_2[j])

        norm_2[avail] -= c[avail, it] ** 2

    return sample.tolist()

    # log_likelihood = np.sum(np.log(norm_2[sample]))
    # return sample.tolist(), log_likelihood


def proj_dpp_sampler_kernel_Schur(K, size=None, random_state=None):
    """Sample from:

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

    N = len(K)  # ground set size
    rank = np.rint(np.trace(K)).astype(int)  # rank(K) = Tr(K)
    if size is None:  # full projection DPP else k-DPP with k = size
        size = rank
    assert size <= rank

    ground_set = np.arange(N)
    sample = np.zeros(size, dtype=int)
    avail = np.full(N, fill_value=True, dtype=bool)

    # Schur complement list i.e. residual norm^2
    schur_comp = K.diagonal().copy()
    K_inv = np.zeros((size, size), dtype=float)

    for it in range(size):
        j = rng.choice(ground_set[avail], p=np.abs(schur_comp[avail]) / (rank - it))
        sample[it] = j
        avail[j] = False

        # Update Schur complements K_ii - K_iY (K_Y)^-1 K_Yi
        # 1. Use Woodbury identity to compute K[Y+j,Y+j]^-1 from K[Y,Y]^-1
        if it == 0:
            K_inv[0, 0] = 1.0 / K[j, j]

        elif it == 1:
            i = sample[0]
            K_inv[:2, :2] = np.array([[K[j, j], -K[j, i]], [-K[j, i], K[i, i]]])
            K_inv[:2, :2] /= K[i, i] * K[j, j] - K[j, i] ** 2

        elif it < size - 1:
            temp = K_inv[:it, :it].dot(K[sample[:it], j])  # K_Y^-1 K_Yj
            # K_jj - K_jY K_Y^-1 K_Yj
            schur_j = K[j, j] - K[j, sample[:it]].dot(temp)

            K_inv[:it, :it] += np.outer(temp, temp / schur_j)
            K_inv[:it, it] = -temp / schur_j
            K_inv[it, :it] = K_inv[:it, it]
            K_inv[it, it] = 1.0 / schur_j

        else:  # it == size-1
            break  # no need to update for nothing

        # 2. Update Schur complements
        # K_ii - K_iY (K_Y)^-1 K_Yi for Y <- Y+j
        K_iY = K[np.ix_(avail, sample[: it + 1])]
        schur_comp[avail] = K[avail, avail] - inner1d(
            K_iY.dot(K_inv[: it + 1, : it + 1]), K_iY, axis=1
        )

    return sample.tolist()

    # log_likelihood = np.sum(np.log(schur_comp[sample]))
    # return sample.tolist(), log_likelihood
