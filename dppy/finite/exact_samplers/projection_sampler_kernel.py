import numpy as np

from dppy.utils import check_random_state, log_binom


def projection_sampler_kernel(dpp, size=None, random_state=None, **kwargs):
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

    # if dpp.kernel_type == "correlation":
    dpp.compute_correlation_kernel()
    return sampler(dpp.K, size=size, random_state=random_state, **kwargs)


def select_projection_sampler_kernel(mode, hermitian=False):
    samplers = {
        "lu": projection_sampler_kernel_lu,
        "cho": projection_sampler_kernel_cho,
    }
    default = samplers["cho" if hermitian else "lu"]
    return samplers.get(mode.lower(), default)


def projection_sampler_kernel_lu(
    K, size=None, random_state=None, overwrite=False, **kwargs
):
    """Variant of :py:func:`~dppy.finite.exact_samplers.projection_samplers_eigen.projection_sampler_kernel_cho` where LU updates are performed instead of Cholesky updates."""
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
    log_likelihood = np.sum(np.log(d2[S])) - log_binom(rank, size)
    return sample  # , log_likelihood


def projection_sampler_kernel_cho(
    K, size=None, random_state=None, overwrite=False, **kwargs
):
    r"""Generate an exact sample from :math:`\operatorname{DPP}(\mathbf{K})`, or :math:`\operatorname{k-DPP}(\mathbf{K})` with :math:`k=` ``size`` (if ``size`` is provided), where :math:`\mathbf{K}` is an orthogonal projection `kernel` and denote :math:`r=\operatorname{rank}(\mathbf{K})`.
    If ``size=None`` (default), it is set to :math:`k=r`, this also corresponds to sampling from the projection :math:`\operatorname{DPP}(\mathbf{K}=UU^{*})`.

    This function implements :cite:`Pou19` Algorithm 3, where updates of the conditionals driving the chain rule are performed via Cholesky updates.

    The likelihood of the output sample is given by

    .. math::

        \mathbb{P}\!\left[ \mathcal{X} = X \right]
        = \frac{1}{\binom{r}{k}} \det K_X ~ 1_{|X|=k}.

    :param K:
        Orthogonal projection kernel :math:`\mathbf{K}=\mathbf{K}^*=\mathbf{K}^2`.
    :type K:
        array_like

    :param size:
        If None, it is set to :math:`r`, otherwise it defines the size :math:`k\leq r` of the output :math:`\operatorname{k-DPP} sample, defaults to None.
    :type size:
        int, optional

    :param overwrite:
        If True, ``K`` is overwritten otherwise a copy of ``K`` is made, defaults to True.
    :type overwrite:
        bool, optional

    :return:
        Exact sample :math:`X` and its log-likelihood.
    :rtype:
        tuple(list, float)

    .. seealso::

        - :ref:`finite_dpps_exact_sampling_projection_dpp`
        - :cite:`Pou19` Algorithm 3
        - :py:func:`~dppy.finite.exact_samplers.projection_sampler_kernel.projection_sampler_kernel_lu`
        - :py:func:`~dppy.finite.exact_samplers.projection_samplers_eigen.projection_sampler_eigen_gs_perm`
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

    log_likelihood = np.sum(np.log(d2[S])) - log_binom(rank, size)
    return sample  # , log_likelihood


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
