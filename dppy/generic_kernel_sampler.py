import numpy as np
from .utils import check_random_state


def dpp_sampler_generic_kernel(K, random_state=None):
    """Generate an exact sample from generic :math:`\\operatorname{DPP}(\\mathbf{K})` with potentially non hermitian correlation kernel :math:`\\operatorname{DPP}(\\mathbf{K})` based on LU factorization procedure.

    :param K:
        Correlation kernel (potentially non hermitian).
    :type K:
        array_like

    :return:
        An exact sample :math:`X \sim \\operatorname{DPP}(K)` and
        the in-place LU factorization of :math:`K − I_{X^{c}}` where :math:`I_{X^{c}}` is the diagonal indicator matrix for the entries not in the sample :math:`X`.
    :rtype:
        list and array_like

    .. seealso::

        - :cite:`Pou19` Algorithm 1
    """
    rng = check_random_state(random_state)
    A = K.copy()
    sample = []
    for j in range(len(A)):
        if rng.rand() < A[j, j]:
            sample.append(j)
        else:
            A[j, j] -= 1.0
        A[j + 1 :, j] /= A[j, j]
        A[j + 1 :, j + 1 :] -= np.outer(A[j + 1 :, j], A[j, j + 1 :])
        # A[j+1:, j+1:] -=  np.einsum('i,j', A[j+1:, j], A[j, j+1:])
    return sample, A
