import numpy as np
import scipy.linalg as la

from dppy.utils import check_random_state


def ginibre_sampler_full(n, random_state=None):
    r"""Generate a sample from the complex Ginibre ensemble of size :math:`n`, by computing the eigenvalues of a random complex standard Gaussian matrix.

    The output sample :math:`x=\left(x_{1}, \dots, x_{n}\right)` has joint density proportional to

    .. math::

        \Delta(x_{1}, \dots, x_{n})^{\beta}
        \prod_{i=1}^{n} \exp( -\frac{x_i^2}{4} ).

    :param n: Size :math:`n` of the output sample, i.e., size of the matrix to be diagonalized.
    :type n: int

    :param random_state: _description_, defaults to None
    :type random_state: _type_, optional

    :return: Vector of size :math:`n` representing the output sample.
    :rtype: numpy.ndarray

    .. note::

        The rescaled version :math:`\frac{x}{\sqrt{n}}` of the sample has limiting distribution the uniform distribution on the unit disk.
    """
    rng = check_random_state(random_state)
    A = rng.randn(n, n) + 1j * rng.randn(n, n)
    return la.eigvals(A) / np.sqrt(2.0)
