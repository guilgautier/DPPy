import numpy as np
import scipy.linalg as la
from scipy.stats import semicircular

from dppy.utils import check_random_state


def sampler_hermite_full(beta, n, random_state=None):
    r"""Generate a sample from the Hermite ``beta=1, 2, 4`` ensemble of size :math:`n`, by computing the eigenvalues of the **symmetric full** random  matrix model :math:`\frac{1}{\sqrt{2}}(A + A^*)`.

    The output sample :math:`x=\left(x_{1}, \dots, x_{n}\right)` has joint density proportional to

    .. math::

        \Delta(x_{1}, \dots, x_{n})^{\beta}
        \prod_{i=1}^{n} \exp( -\frac{x_i^2}{4} ).

    See :py:func:`~dppy.beta_ensembles.hermite.sampler_hermite_tridiagonal` (with the default arguments) for the tridiagonal counter part.

    :param beta: :math:`\beta \in \{1, 2, 4\}` parameter of the ensemble.
    :type beta: int
    :param n: Size :math:`n` of the output sample, i.e., size of the matrix to be diagonalized.
    :type n: int

    :param random_state: _description_, defaults to None
    :type random_state: _type_, optional

    :return: Vector of size :math:`n` representing the output sample.
    :rtype: numpy.ndarray

    .. note::

        The rescaled version :math:`\frac{x}{\sqrt{\beta n}}` of the sample has limiting distribution the semicircle distribution supported on :math:`[-2, 2]`.

    .. seealso::

        :cite:`DuEd02` II-C
    """
    rng = check_random_state(random_state)

    if beta == 1:
        A = rng.randn(n, n)
        A += A.T

    elif beta == 2:
        A = rng.randn(n, n) + 1j * rng.randn(n, n)
        A += A.conj().T

    elif beta == 4:
        X = rng.randn(n, n) + 1j * rng.randn(n, n)
        Y = rng.randn(n, n) + 1j * rng.randn(n, n)
        A = np.block([[X, Y], [-Y.conj(), X.conj()]])
        A += A.conj().T

    else:
        raise ValueError("beta argument must be 0, 1, 2 or 4.")

    return la.eigvalsh(A) / np.sqrt(2.0)


def sampler_hermite_tridiagonal(
    beta, size, loc=0.0, scale=np.sqrt(2), random_state=None
):
    r"""Generate a sample from the Hermite ``beta`` ensemble of size :math:`n` equal to ``size``, with reference measure :math:`\mathcal{n}(\mu, \sigma^2)` or equivalently with potential :math:`V(x) = -\frac{(x_i-\mu)^2}{2\sigma^2}`. The default arguments make the correspondence with :py:func:`~dppy.beta_ensembles.hermite.sampler_hermite_full`.

    This is done by computing the eigenvalues of a **symmetric tridiagonal** random matrix. The output sample :math:`x=\left(x_{1}, \dots, x_{n}\right)` has joint density proportional to

    .. math::

        \Delta(x_{1}, \dots, x_{n})^{\beta}
        \prod_{i=1}^{n} \exp( -\frac{(x_i-\mu)^2}{2\sigma^2} ).

    :param beta: :math:`\beta \geq 0` parameter of the ensemble.
    :type beta: int
    :param n: Size :math:`n` of the output sample, i.e., size of the matrix to be diagonalized.
    :type n: int
    :param loc: Mean :math:`\mu` of the Gaussian reference distribution, defaults to 0.0
    :type loc: float, optional
    :param scale: Standard deviation :math:`\sigma` of the Gaussian reference distribution, defaults to np.sqrt(2)
    :type scale: float, optional

    :param random_state: _description_, defaults to None
    :type random_state: _type_, optional

    :return: Vector of size :math:`n` representing the output sample.
    :rtype: numpy.ndarray

    .. note::

        The rescaled version :math:`\frac{1}{\sqrt{\beta n}} \frac{\sqrt{2} (x - \mu)}{\sigma}` of the sample has limiting distribution the semicircle distribution supported on :math:`[-2, 2]`.

    .. seealso::

        :cite:`DuEd02` II-C
    """
    if not (beta >= 0):
        raise ValueError("beta must be non-negative (beta >= 0).")

    rng = check_random_state(random_state)

    if beta == 0:  # Answer issue #28 raised by @rbardenet
        return rng.normal(loc=loc, scale=scale, size=size)

    alpha_coef = rng.normal(loc=loc, scale=scale, size=size)
    # beta/2*[n-1, n-2, ..., 1]
    b_2_Ni = 0.5 * beta * np.arange(size - 1, 0, step=-1)
    beta_coef = rng.gamma(shape=b_2_Ni, scale=scale ** 2)

    return la.eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))


def semi_circle_density(x, loc=0.0, scale=1.0):
    return semicircular.pdf(x, loc=loc, scale=scale)
