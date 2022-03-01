import itertools as itt

import numpy as np


def compute_ordering(N, d):
    r"""Compute the ordering of the multi-indices :math:`\in\mathbb{N}^d` defining the order between the multivariate monomials as described in Section 2.1.3 of :cite:`BaHa16`.

    :param N:
        Number of polynomials :math:`(P_k)` considered to build the kernel :py:meth:`~dppy.continuous.jacobi.JacobiProjectionDPP.correlation_kernel` (number of points of the corresponding :py:class:`JacobiProjectionDPP`)
    :type N:
        int

    :param d:
        Size of the multi-indices :math:`k\in \mathbb{N}^d` characterizing the _degree_ of :math:`P_k` (ambient dimension of the points x_{1}, \dots, x_{N} \in [-1, 1]^d)
    :type d:
        int

    :return:
        Array of size :math:`N\times d` containing the first :math:`N` multi-indices :math:`\in\mathbb{N}^d` in the order prescribed by the ordering :math:`\mathfrak{b}` :cite:`BaHa16` Section 2.1.3
    :rtype:
        array_like

    For instance, for :math:`N=12, d=2`

    .. code:: python

        [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (0, 2),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
            (0, 3),
            (1, 3),
            (2, 3),
        ]

    .. seealso::

        - :cite:`BaHa16` Section 2.1.3
    """
    layer_max = np.floor(N ** (1.0 / d)).astype(np.int16)

    ordering = itt.chain.from_iterable(
        filter(lambda x: m in x, itt.product(range(m + 1), repeat=d))
        for m in range(layer_max + 1)
    )

    return list(ordering)[:N]
