import numpy as np

from dppy.utils import check_random_state


def sequential_sampler(dpp, random_state=None, **params):
    r"""Generate an exact sample from ``dpp`` using the :ref:`sequential method <finite_dpps_exact_sampling_sequential_methods>`.

    The correlation kernel :math:`\mathbf{K}` is computed from the current parametrization of ``dpp``, see :py:meth:`~dppy.finite.dpp.FiniteDPP.compute_K`.

    :param dpp: Finite DPP
    :type dpp: :py:class:`~dppy.finite.dpp.FiniteDPP`

    :param random_state: random number generator or seed, defaults to None
    :type random_state: optional

    Keyword arguments:

        - **mode** (str): select the variant of the sampler, see :py:func:`~dppy.finite.exact_samplers.sequential_sampler.select_sequential_sampler`

    :return: sample
    :rtype: list
    """
    dpp.compute_K()
    mode = params.get("mode", "")
    sampler = select_sequential_sampler(mode, dpp.hermitian)
    return sampler(dpp.K, random_state=random_state, **params)


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


def sequential_sampler_lu(K, random_state=None, **params):
    r"""Generate an exact sample from generic :math:`\operatorname{DPP}(\mathbf{K})` with potentially non hermitian correlation kernel :math:`\mathbf{K}`.

    This function implements :cite:`Pou19` Algorithm 1 based on a LU-type factorization procedure.

    :param K:
        Correlation kernel (potentially non hermitian).
    :type K:
        array_like

    :return:
        An exact sample :math:`X \sim \operatorname{DPP}(\mathbf{K})`.
    :rtype:
        list

    .. note::

        The likelihood of the output sample :math:`X` is given by

        .. math::

            \mathbb{P}\!\left[ \mathcal{ X } = X \right]
            = \det \left[ \mathbf{K} − I^{X^{c}} \right]

    """
    rng = check_random_state(random_state)
    A = K.copy()
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
        # A[j+1:, j+1:] -=  np.einsum('i,j', A[j+1:, j], A[j, j+1:])
    # log_likelihood = np.sum(np.log(np.diagonal(A)))
    return sample  # , A , log_likelihood


def sequential_sampler_ldl(K, random_state=None, **params):
    r"""Generate an exact sample from a hermitian :math:`\operatorname{DPP}(\mathbf{K})`.

    This function implements the hermitian version of :cite:`Pou19` Algorithm 1. It is based on a LDL^h-type factorization procedure.

    :param K:
        Hermitian correlation kernel :math:`\mathbf{K}`.
    :type K:
        array_like

    :return:
        An exact sample :math:`X \sim \operatorname{DPP}(\mathbf{K})`.
    :rtype:
        list
    """
    rng = check_random_state(random_state)
    A = K.copy()
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
        A[i, I] = d[i]
        I, J = slice(0, i), slice(i + 1, N)
        A[J, i] -= A[J, I].dot(v[I])
        A[J, i] /= d[i]

    return sample  # , A
