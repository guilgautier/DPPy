import numpy as np

from dppy.utils import check_random_state, log_binom


def projection_sampler_kernel(dpp, size=None, random_state=None, **kwargs):
    r"""Generate an exact sample from ``dpp`` using the :ref:`projection method <finite_dpps_exact_sampling_projection_methods>`. The attribute ``dpp.projection`` must be True.

    If the attribute ``dpp.kernel_type`` is ``"likelihood"``, sample from :math:`\operatorname{k-DPP}(\mathbf{L})` where ``size`` :math:`=k` must be provided. Denote :math:`r=\operatorname{rank}(\mathbf{L})`.

    .. math::

        \mathbb{P}\!\left[ \mathcal{X} = X \right]
        = \frac{1}{\binom{r}{k}} \det \mathbf{L}_X ~ 1_{|X|=k}.

    If the attribute ``dpp.kernel_type`` is ``"correlation"``, sample from the projection :math:`\operatorname{DPP}(\mathbf{K})` where ``size`` must be equal to :math:`r=\operatorname{rank}(\mathbf{K})`.

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

        - **mode** (str): select the variant of the sampler, see :py:func:`~dppy.finite.exact_samplers.projection_sampler_kernel.select_projection_sampler_kernel`
    """
    assert dpp.projection

    mode = kwargs.get("mode", "")
    sampler = select_projection_sampler_kernel(mode, dpp.hermitian)

    if dpp.kernel_type == "likelihood":
        if not size:
            raise ValueError(
                "'size' argument (k) required for sampling k-DPP(L) with projection likelihood kernel L."
            )
        dpp.compute_likelihood_kernel()
        return sampler(dpp.L, size=size, random_state=random_state, **kwargs)

    if dpp.kernel_type == "correlation":
        dpp.compute_correlation_kernel()
        if size:
            rank_K = np.rint(np.trace(dpp.K)).astype(int)
            if size != rank_K:
                raise ValueError(
                    "'size' argument {} != {} = rank(K) for sampling projection DPP(K)".format(
                        size, rank
                    )
                )
        return sampler(dpp.K, size=size, random_state=random_state, **kwargs)


def select_projection_sampler_kernel(mode, hermitian):
    r"""Select the variant of the :ref:`projection method <finite_dpps_exact_sampling_projection_methods>`.

    :param mode: variant name, defaults to ``"cho"`` if ``hermitian`` is True, otherwise "lu".
    :type mode: str

    :return: sampler selected by ``mode``

        - ``"lu"`` (default) :py:func:`~dppy.finite.exact_samplers.projection_sampler_kernel.projection_sampler_kernel_lu`
        - ``"cho"`` (default) :py:func:`~dppy.finite.exact_samplers.projection_sampler_kernel.projection_sampler_kernel_cho`
    :rtype: callable
    """
    samplers = {
        "lu": projection_sampler_kernel_lu,
        "cho": projection_sampler_kernel_cho,
    }
    if mode == "cho":
        assert hermitian
    default = samplers["cho" if hermitian else "lu"]
    return samplers.get(mode.lower(), default)


def projection_sampler_kernel_lu(
    K,
    size=None,
    random_state=None,
    overwrite=False,
    log_likelihood=False,
    **kwargs,
):
    """Variant of :py:func:`~dppy.finite.exact_samplers.projection_sampler_kernel.projection_sampler_kernel_cho` where LU updates are performed instead of Cholesky updates."""
    rng = check_random_state(random_state)

    rank = np.rint(np.trace(K)).astype(int)
    if size is None:
        size = rank
    assert size <= rank

    A = K if overwrite else K.copy()
    d2 = np.copy(A.diagonal().real)

    N = len(A)
    items = np.arange(N)

    for i in range(size):

        R = range(i, N)
        # Sample from pivot index and itemsute
        j = rng.choice(R, p=d2[R] / (rank - i))

        swap(A, i, j)
        items[i], items[j] = items[j], items[i]
        d2[i], d2[j] = d2[j], d2[i]

        A[i, i] = d2[i]

        if i == size - 1:
            break

        # outer mode O(N^2 size)
        # J4 = slice(j + 1, N)
        # A[J4, j] /= A[j, j]
        # A[J4, J4] -= np.outer(A[J4, j], A[j, J4])
        # R = range(j + 1, N)
        # d[R] = np.fmax(A[R, R], 0)

        # Gaxpy-like mode O(N size^2)
        I, J = slice(0, i), slice(i + 1, N)
        A[J, i] -= A[J, I].dot(A[I, i])
        A[J, i] /= A[i, i]
        A[i, J] -= A[i, I].dot(A[I, J])

        d2[J] -= A[i, J] * A[J, i]
        d2[J] = np.fmax(d2[J], 0.0)

    # return items[:size].tolist()

    S = range(0, size)
    sample = items[S].tolist()
    if log_likelihood:
        log_lik = np.sum(np.log(d2[S])) - log_binom(rank, size)
        return sample, log_lik
    return sample


def projection_sampler_kernel_cho(
    K,
    size=None,
    random_state=None,
    overwrite=False,
    log_likelihood=False,
    **kwargs,
):
    r"""Generate an exact sample from :math:`\operatorname{k-DPP}(K)` with :math:`k=` ``size`` (if ``size`` is provided), where ``K`` is an orthogonal projection matrix. If ``size`` is None (default), it is set to :math:`k=\operatorname{rank}(K)\triangleq r`, this also corresponds to sampling from the projection :math:`\operatorname{DPP}(K)`.

    This function implements :cite:`Pou19` Algorithm 3, where updates of the conditionals driving the chain rule are performed via Cholesky updates. This can also be viewed as randomized a Gram-Schmidt orthogalization procedure applied on the rows or columns of the kernel.

    :param K:
        Orthogonal projection kernel :math:`K=K^*=K^2`.
    :type K:
        array_like

    :param size:
        If None, it is set to :math:`r`, otherwise it defines the size :math:`k\leq r` of the output :math:`\operatorname{k-DPP}` sample, defaults to None.
    :type size:
        int, optional

    :param overwrite:
        If True, ``K`` is overwritten otherwise a copy of ``K`` is made, defaults to True.
    :type overwrite:
        bool, optional

    :param random_state:
        random number generator or seed, defaults to None
    :type random_state:
        optional

    :param log_likelihood:
        If True, log-likelihood of the sample is also returned.
    :type log_likelihood:
        bool

    :return:
        Exact sample :math:`X` and its log-likelihood according to ``log_likelihood``.
    :rtype:
        list or tuple(list, float)

    .. seealso::

        - :ref:`finite_dpps_exact_sampling_projection_methods`
        - :cite:`Pou19` Algorithm 3
        - :py:func:`~dppy.finite.exact_samplers.projection_sampler_kernel.projection_sampler_kernel_lu`
        - :py:func:`~dppy.finite.exact_samplers.projection_sampler_eigen.projection_sampler_eigen_gs_perm`
    """
    rng = check_random_state(random_state)

    rank = np.rint(np.trace(K)).astype(int)
    if size is None:
        size = rank
    assert size <= rank

    A = K if overwrite else K.copy()
    d2 = np.copy(A.diagonal().real)

    N = len(A)
    items = np.arange(N)

    for i in range(size):

        J = range(i, N)
        j = rng.choice(J, p=d2[J] / (rank - i))

        swap_hermitian(A, i, j)
        items[i], items[j] = items[j], items[i]
        d2[i], d2[j] = d2[j], d2[i]

        A[i, i] = np.sqrt(d2[i])

        if i == size - 1:
            break

        # Form new column and update diagonal
        I1, I2 = slice(0, i), slice(i + 1, N)
        A[I2, i] -= A[I2, I1].dot(A[i, I1].conj())
        A[I2, i] /= A[i, i]

        if np.iscomplexobj(A):
            d2[I2] -= np.abs(A[I2, i]) ** 2
        else:
            d2[I2] -= A[I2, i] ** 2
        d2[I2] = np.fmax(d2[I2], 0.0)

    S = range(0, size)
    sample = items[S].tolist()

    if log_likelihood:
        log_lik = np.sum(np.log(d2[S])) - log_binom(rank, size)
        return sample, log_lik
    return sample


def swap(A, i, j):
    # idx = np.arange(len(A))
    # idx[i], idx[j] = idx[j], idx[i]
    # A = A[np.ix_(idx, idx)]
    # Swap of indices j and t
    i_j = [i, j]
    j_i = [j, i]
    # top, inner, bottom swap
    for I in [slice(0, i), slice(i + 1, j), slice(j + 1, len(A))]:
        A[I, i_j] = A[I, j_i]
        A[i_j, I] = A[j_i, I]
    # diagonal swap
    A[i_j, i_j] = A[j_i, j_i]
    # corner swap
    A[i_j, j_i] = A[j_i, i_j]

    return None


def swap_hermitian(A, i, j):
    # Hermitian swap of indices i, j of A
    # See `catamari code <https://gitlab.com/hodge_star/catamari/blob/38718a1ea34872fb6567e019ece91fbeb5af5be1/include/catamari/dense_dpp/elementary_hermitian_dpp-impl.hpp#L37>`_ for the Hermitian swap routine.
    i_j = [i, j]
    j_i = [j, i]
    I1, I2, I3 = slice(0, i), slice(i + 1, j), slice(j + 1, len(A))

    # left swap
    A[i_j, I1] = A[j_i, I1]
    # inner swap
    tmp = A[I2, i].copy()
    np.conj(A[j, I2], out=A[I2, i])
    np.conj(tmp, out=A[j, I2])
    # bottom swap
    A[I3, i_j] = A[I3, j_i]
    # corner swap
    A[i, j] = np.conj(A[i, j])
    # diagonal swap
    A[i_j, i_j] = A[j_i, j_i]

    return None
