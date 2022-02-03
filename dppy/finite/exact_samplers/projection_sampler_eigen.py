import numpy as np

from dppy.utils import check_random_state, log_binom


def projection_sampler_eigen(dpp, size=None, random_state=None, **kwargs):
    r"""Generate an exact sample from ``dpp`` using the :ref:`projection method <finite_dpps_exact_sampling_projection_methods>`.
    The attribute ``dpp.projection`` must be True and the underlying kernel must be a orthogonal projection matrix with eigenvectors :math:`U=` ``dpp.eig_vecs``.

    If the attribute ``dpp.kernel_type`` is ``"likelihood"``, sample from :math:`\operatorname{k-DPP}(\mathbf{L}=UU^{*})` where ``size`` :math:`=k` must be provided. Denote :math:`r=\operatorname{rank}(\mathbf{L})`, the number of columns of :math:`U`.

    .. math::

        \mathbb{P}\!\left[ \mathcal{X} = X \right]
        = \frac{1}{\binom{r}{k}} \det \mathbf{L}_X ~ 1_{|X|=k}.

    If the attribute ``dpp.kernel_type`` is ``"correlation"``, sample from the projection :math:`\operatorname{DPP}(\mathbf{K}=UU^{*})` where ``size`` must be equal to :math:`r=\operatorname{rank}(\mathbf{K})`, the number of columns of :math:`U`.

    .. math::

        \mathbb{P}\!\left[ \mathcal{X} = X \right]
        = \det \mathbf{K}_X ~ 1_{|X|=r}.

    :param dpp:
        Finite DPP
    :type dpp:
        :py:class:`~dppy.finite.dpp.FiniteDPP`

    :param size:
        If None, it is set to :math:`r`, otherwise it defines the size :math:`k\leq r` of the output :math:`\operatorname{k-DPP}` sample, defaults to None.
    :type size:
        int, optional

    :param random_state:
        random number generator or seed, defaults to None
    :type random_state:
        optional

    :return:
        Exact sample :math:`X` and its log-likelihood.
    :rtype:
        tuple(list, float)

    :Keyword arguments:

        - **mode** (str): select the variant of the sampler, see :py:func:`~dppy.finite.exact_samplers.projection_sampler_eigen.select_projection_sampler_eigen`
    """
    assert dpp.projection and dpp.hermitian and dpp.eig_vecs is not None

    if dpp.kernel_type == "likelihood" and not size:
        raise ValueError(
            "'size' argument (k) required for sampling k-DPP(L) with projection likelihood kernel L."
        )
    if dpp.kernel_type == "correlation" and size:
        rank_K = dpp.eig_vecs.shape[1]
        if size != rank_K:
            raise ValueError(
                f"size={size} argument must be equal to rank(K)={rank_K} for projection DPP(K)"
            )
    mode = kwargs.get("mode", "")
    sampler = select_projection_sampler_eigen(mode)
    return sampler(dpp.eig_vecs, random_state=random_state, **kwargs)


def select_projection_sampler_eigen(mode):
    r"""Select the variant of the spectral method applied to a projection kernel given its factorization.

    :param mode: variant name, defaults to "gs".
    :type mode: str

    :return: sampler selected by ``mode``

        - ``"gs"`` :py:func:`~dppy.finite.exact_samplers.projection_sampler_eigen.projection_sampler_eigen_gs`
        - ``"gs-perm"`` :py:func:`~dppy.finite.exact_samplers.projection_sampler_eigen.projection_sampler_eigen_gs_perm`

    :rtype: callable
    """
    samplers = {
        "gs": projection_sampler_eigen_gs,
        "gs-perm": projection_sampler_eigen_gs_perm,
    }
    default = samplers["gs"]
    return samplers.get(mode.lower(), default)


def projection_sampler_eigen_gs(eig_vecs, size=None, random_state=None, **kwargs):
    r"""Generate an exact sample from :math:`\operatorname{k-DPP}(\mathbf{L})` with :math:`k=` ``size`` (if ``size`` is provided), where :math:`\mathbf{L}=UU^{*}` is an orthogonal projection matrix given :math:`U=` ``eig_vecs`` such that :math:`U^{*}U = I_r` and denote :math:`r=\operatorname{rank}(\mathbf{L})`. If ``size`` is None (default), it is set to :math:`k=r`, this also corresponds to sampling from the projection :math:`\operatorname{DPP}(\mathbf{K}=UU^{*})`.

    The likelihood of the output sample is given by

    .. math::

        \mathbb{P}\!\left[ \mathcal{X} = X \right]
        = \frac{1}{\binom{r}{k}} \det [UU^*]_X ~ 1_{|X|=k}.

    :param eig_vecs:
        Eigenvectors :math:`U` of :math:`\mathbf{K}=UU^{*}`.
    :type eig_vecs:
        array_like

    :param size:
        If None, it is set to :math:`r`, otherwise it defines the size :math:`k\leq r` of the output :math:`\operatorname{k-DPP}` sample, defaults to None.
    :type size:
        int, optional

    :param random_state:
        random number generator or seed, defaults to None
    :type random_state:
        optional

    :return:
        Exact sample :math:`X` and its log-likelihood.
    :rtype:
        tuple(list, float)

    .. seealso::

        - :ref:`finite_dpps_exact_sampling_projection_methods`
        - :cite:`Gil14` Algorithm 2 or :cite:`TrBaAm18` Algorithm 3
        - :py:func:`~dppy.finite.exact_samplers.projection_sampler_eigen.projection_sampler_eigen_gs_perm`
        - :py:func:`~dppy.finite.exact_samplers.projection_sampler_kernel.projection_sampler_kernel_cho`
    """
    rng = check_random_state(random_state)

    N, rank = eig_vecs.shape
    if rank == 0:
        return []

    if size is None:  # full projection DPP else k-DPP with k = size
        size = rank
    assert size <= rank

    Q = eig_vecs
    is_complex = np.iscomplexobj(Q)

    R = np.zeros((size, N))
    d2 = np.linalg.norm(Q, axis=1) ** 2

    items = np.arange(N)
    Xc = np.full(N, fill_value=True, dtype=bool)

    for i in range(size):
        # Pick an item \propto this squared distance
        xi = rng.choice(items[Xc], p=d2[Xc] / (rank - i))
        Xc[xi] = False

        R[i, xi] = np.sqrt(d2[xi])

        if i == size - 1:
            break

        R[i, Xc] = Q[Xc, :].dot(Q[xi, :].conj())
        R[i, Xc] -= R[:i, xi].dot(R[:i, Xc])
        R[i, Xc] /= R[i, xi]

        if is_complex:
            d2[Xc] -= np.abs(R[i, Xc]) ** 2
        else:
            d2[Xc] -= R[i, Xc] ** 2
        np.fmax(d2, 0.0, where=Xc, out=d2)

    sample = items[~Xc].tolist()
    log_likelihood = np.sum(np.log(d2[sample])) - log_binom(rank, size)
    return sample  # , log_likelihood


def projection_sampler_eigen_gs_perm(
    eig_vecs, size=None, random_state=None, overwrite=False, **kwargs
):
    """Variant of :py:func:`~dppy.finite.exact_samplers.projection_sampler_eigen.projection_sampler_eigen_gs` involving permutations of the rows of ``eig_vecs``.

    If ``overwrite`` is True, ``eig_vecs`` is permuted inplace.
    """

    rng = check_random_state(random_state)

    N, rank = eig_vecs.shape
    if rank == 0:
        return []

    if size is None:  # full projection DPP else k-DPP with k = size
        size = rank
    assert size <= rank

    Q = eig_vecs
    is_complex = np.iscomplexobj(Q)

    R = np.zeros((size, N), dtype=Q.dtype)
    d2 = np.linalg.norm(Q, axis=1) ** 2

    items = np.arange(N)

    for i in range(size):
        J = range(i, N)
        j = rng.choice(J, p=d2[J] / (rank - i))

        # swap
        i_j = [i, j]
        j_i = [j, i]

        Q[i_j] = Q[j_i]
        I1 = slice(0, i)
        R[I1, i_j] = R[I1, j_i]

        items[i], items[j] = items[j], items[i]
        d2[i], d2[j] = d2[j], d2[i]

        R[i, i] = np.sqrt(d2[i])

        if i == size - 1:
            break

        I1, I2 = slice(0, i), slice(i + 1, N)
        R[i, I2] = Q[I2, :].dot(Q[i, :].conj())
        R[i, I2] -= R[I1, i].dot(R[I1, I2])
        R[i, I2] /= R[i, i]

        if is_complex:
            d2[I2] -= np.abs(R[i, I2]) ** 2
        else:
            d2[I2] -= R[i, I2] ** 2
        np.fmax(d2[I2], 0.0, out=d2[I2])

    if not overwrite:
        perm = np.empty_like(items)
        perm[items] = np.arange(items.size)
        eig_vecs[:] = Q[perm]

    S = range(0, size)
    sample = items[S].tolist()
    log_likelihood = np.sum(np.log(d2[S])) - log_binom(rank, size)
    return sample  # , log_likelihood


# def projection_sampler_eigen_mgs(U, size=None, random_state=None, overwrite=False):

#     rng = check_random_state(random_state)

#     N, rank = U.shape
#     if rank == 0:
#         return []

#     if size is None:  # full projection DPP else k-DPP with k = size
#         size = rank
#     assert size <= rank

#     Q = U if overwrite else U.copy()
#     is_complex = np.iscomplexobj(Q)

#     R = np.zeros((size, N), dtype=Q.dtype)
#     d2 = np.linalg.norm(Q, axis=1) ** 2

#     items = np.arange(N)  # ground set
#     Xc = np.full(N, fill_value=True, dtype=bool)

#     for i in range(size):
#         xi = rng.choice(items[Xc], p=d2[Xc] / (rank - i))
#         Xc[xi] = False

#         R[i, xi] = np.sqrt(d2[xi])

#         if i == size - 1:
#             break

#         Q[xi, :] /= R[i, xi]
#         R[i, Xc] = Q[Xc, :].dot(Q[i, :].conj())
#         Q[Xc, :] -= np.outer(R[i, Xc], Q[i, :])

#         if is_complex:
#             d2[Xc] -= np.abs(R[i, Xc]) ** 2
#         else:
#             d2[Xc] -= R[i, Xc] ** 2
#         np.fmax(d2[Xc], 0.0, out=d2[Xc])

#     X = items[~Xc].tolist()
#     likelihood = np.prod(d2[X])  # np.prod(np.square(R[X, range(0, size)]))
#     return X, likelihood


# def projection_sampler_eigen_mgs_perm(
#     U, size=None, random_state=None, overwrite=False
# ):

#     rng = check_random_state(random_state)

#     N, rank = U.shape
#     if rank == 0:
#         return []

#     if size is None:  # full projection DPP else k-DPP with k = size
#         size = rank
#     assert size <= rank

#     Q = U if overwrite else U.copy()
#     is_complex = np.iscomplexobj(Q)

#     R = np.zeros((size, N), dtype=Q.dtype)
#     d2 = np.linalg.norm(Q, axis=1) ** 2

#     items = np.arange(N)

#     for i in range(size):
#         J = range(i, N)
#         j = rng.choice(J, p=d2[J] / (rank - i))

#         # swap
#         i_j = [i, j]
#         j_i = [j, i]

#         Q[i_j] = Q[j_i]
#         I1 = slice(0, i)
#         R[I1, i_j] = R[I1, j_i]

#         items[i], items[j] = items[j], items[i]
#         d2[i], d2[j] = d2[j], d2[i]

#         R[i, i] = np.sqrt(d2[i])

#         if i == size - 1:
#             break

#         I2 = slice(i + 1, N)
#         Q[i, :] /= R[i, i]
#         R[i, I2] = Q[I2, :].dot(Q[i, :].conj())
#         Q[I2, :] -= np.outer(R[i, I2], Q[i, :])

#         if is_complex:
#             d2[I2] -= np.abs(R[i, I2]) ** 2
#         else:
#             d2[I2] -= R[i, I2] ** 2
#         np.fmax(d2[I2], 0.0, out=d2[I2])

#     S = range(0, size)
#     sample = items[S].tolist()
#     likelihood = np.prod(d2[S])  # np.prod(np.square(R[S, S]))

#     return sample, likelihood
