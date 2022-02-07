import numpy as np

from dppy.utils import check_random_state


def sequential_sampler(dpp, random_state=None, **kwargs):
    r"""Generate an exact sample from ``dpp`` using the :ref:`sequential method <finite_dpps_exact_sampling_sequential_methods>`.

    The correlation kernel :math:`\mathbf{K}` is computed from the current parametrization of ``dpp``, see :py:meth:`~dppy.finite.dpp.FiniteDPP.compute_correlation_kernel`.

    :param dpp: Finite DPP
    :type dpp: :py:class:`~dppy.finite.dpp.FiniteDPP`

    :param random_state:
        random number generator or seed, defaults to None
    :type random_state:
        optional

    :Keyword arguments:

        - **mode** (str): select the variant of the sampler, see :py:func:`~dppy.finite.exact_samplers.sequential_sampler.select_sequential_sampler`

    :return: sample
    :rtype: list
    """
    dpp.compute_correlation_kernel()
    mode = kwargs.get("mode", "")
    sampler = select_sequential_sampler(mode, dpp.hermitian)
    return sampler(dpp.K, random_state=random_state, **kwargs)


def select_sequential_sampler(mode, hermitian):
    r"""Select the variant of the :ref:`sequential method <finite_dpps_exact_sampling_sequential_methods>`.

    :param mode:
        Select the variant among

        - ``"lu"`` (default) :py:func:`~dppy.finite.exact_samplers.sequential_sampler.sequential_sampler_lu`
        - ``"ldl"`` (default) :py:func:`~dppy.finite.exact_samplers.sequential_sampler.sequential_sampler_ldl`

    :type mode: str
    """
    samplers = {
        "lu": sequential_sampler_lu,
        "ldl": sequential_sampler_ldl,
    }
    if mode == "ldl":
        assert hermitian
    default = samplers["ldl" if hermitian else "lu"]
    return samplers.get(mode.lower(), default)


def sequential_sampler_lu(
    K,
    random_state=None,
    overwrite=False,
    log_likelihood=False,
    **kwargs,
):
    r"""Generate an exact sample from generic :math:`\operatorname{DPP}(\mathbf{K})` with potentially non hermitian correlation kernel :math:`\mathbf{K}`.

    This function implements :cite:`Pou19` Algorithm 1 based on a LU-type factorization procedure.

    :param K:
        Correlation kernel (potentially non hermitian).
    :type K:
        array_like

    :param random_state:
        random number generator or seed, defaults to None
    :type random_state:
        optional

    :return:
        An exact sample :math:`X \sim \operatorname{DPP}(\mathbf{K})`.
    :rtype:
        list
    """
    rng = check_random_state(random_state)
    A = K if overwrite else K.copy()
    N = len(A)
    sample = []
    for i in range(N):
        if rng.rand() < A[i, i]:
            sample.append(i)
        else:
            A[i, i] -= 1.0
        I = slice(i + 1, N)
        A[I, i] /= A[i, i]
        A[I, I] -= np.outer(A[I, i], A[i, I])

    if log_likelihood:
        log_lik = np.sum(np.log(np.abs(A.diagonal())))
        return sample, log_lik
    return sample


def sequential_sampler_ldl(
    K,
    random_state=None,
    overwrite=False,
    log_likelihood=False,
    **kwargs,
):
    r"""Generate an exact sample from a hermitian :math:`\operatorname{DPP}(\mathbf{K})`.

    This function implements the hermitian version of :cite:`Pou19` Algorithm 1. It is based on a :math:`LDL^h`-type factorization procedure.

    :param K:
        Hermitian correlation kernel :math:`\mathbf{K}=\mathbf{K}^*`.
    :type K:
        array_like

    :param random_state:
        random number generator or seed, defaults to None
    :type random_state:
        optional

    :return:
        An exact sample :math:`X \sim \operatorname{DPP}(\mathbf{K})`.
    :rtype:
        list
    """
    rng = check_random_state(random_state)
    A = K if overwrite else K.copy()
    hermitian = np.iscomplexobj(A)
    N = len(A)
    v = np.zeros(N, dtype=(A.dtype))
    d = np.copy(A.diagonal().real)
    sample = []
    for i in range(N):
        I = range(0, i)

        if hermitian:
            v[I] = np.conj(A[i, I] * d[I])
            d[i] -= np.real(A[i, I].dot(v[I]))
        else:
            v[I] = A[i, I] * d[I]
            d[i] -= A[i, I].dot(v[I])

        if rng.rand() < d[i]:
            sample.append(i)
        else:
            d[i] -= 1.0

        A[i, i] = d[i]
        I, J = slice(0, i), slice(i + 1, N)
        A[J, i] -= A[J, I].dot(v[I])
        A[J, i] /= d[i]

    if log_likelihood:
        log_lik = np.sum(np.log(np.abs(d)))
        return sample, log_lik
    return sample
