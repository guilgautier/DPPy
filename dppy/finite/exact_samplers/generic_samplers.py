import numpy as np

from dppy.utils import check_random_state


def generic_sampler(dpp, random_state=None, **params):
    dpp.compute_K()
    mode = params.get("mode", "")
    sampler = select_generic_sampler(mode)
    return sampler(dpp.K, random_state=random_state, **params)


def select_generic_sampler(mode):
    samplers = {
        "lu": generic_correlation_kernel_sampler_lu,
    }
    default = samplers["lu"]
    return samplers.get(mode.lower(), default)


def generic_correlation_kernel_sampler_lu(K, random_state=None, **params):
    r"""Generate an exact sample from generic :math:`\operatorname{DPP}(\mathbf{K})` with potentially non hermitian correlation kernel :math:`\mathbf{K}` based on LU factorization procedure.

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
            = \det \left[ K − I_{X^{c}} \right]

    .. seealso::

        - :cite:`Pou19` Algorithm 1
    """
    rng = check_random_state(random_state)
    A = K.copy()
    N = len(A)
    sample = []
    for j in range(N):
        if rng.rand() < A[j, j]:
            sample.append(j)
        else:
            A[j, j] -= 1.0
        J1 = slice(j + 1, N)
        A[J1, j] /= A[j, j]
        A[J1, J1] -= np.outer(A[J1, j], A[j, J1])
        # A[j+1:, j+1:] -=  np.einsum('i,j', A[j+1:, j], A[j, j+1:])
    # log_likelihood = np.sum(np.log(np.diagonal(A)))
    return sample  # , A , log_likelihood
