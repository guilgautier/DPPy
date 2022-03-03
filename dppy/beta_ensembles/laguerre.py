import numpy as np
import scipy.linalg as la

from dppy.utils import check_random_state


def sampler_laguerre_full(beta, n, m, random_state=None):
    r"""Generate a sample from the Laguerre ``beta=1, 2, 4`` ensemble by computing the eigenvalues of the **symmetric full** random matrix model :math:`A A^*`, where :math:`A` has size :math:`n \times m`.

    The output sample :math:`x=\left(x_{1}, \dots, x_{n}\right)` has joint density proportional to

    .. math::

        \Delta(x_{1}, \dots, x_{n})^{\beta}
        \prod_{i=1}^{n}
            x_i^{\frac{\beta}{2}(m - n + 1) - 1}
            \exp( -\frac{1}{2} x_i ).

    The equivalent banded model :py:func:`~dppy.beta_ensembles.laguerre.sampler_laguerre_tridiagonal` can be called as
    ``sampler_laguerre_tridiagonal(beta=beta, size=n, shape=beta / 2 * (m - n + 1), scale=2)``.

    :param beta: :math:`\beta \in \{1, 2, 4\}` parameter of the ensemble.
    :type beta: int

    :param n: Size :math:`n` of the output sample, i.e., size of the matrix to be diagonalized.
    :type n: int

    :param m: Size :math:`n` of the output sample, i.e., size of the matrix to be diagonalized.
    :type m: int

    :param random_state: _description_, defaults to None
    :type random_state: _type_, optional

    :return: Vector of size :math:`n` representing the output sample.
    :rtype: numpy.ndarray

    .. note::

        The rescaled version :math:`\frac{x}{\beta m}` of the sample has limiting distribution the Marcenko-Pastur distribution.

    .. seealso::

        :cite:`DuEd02` II-C
    """
    rng = check_random_state(random_state)

    if beta == 1:
        A = rng.randn(n, m)

    elif beta == 2:
        A = rng.randn(n, m) + 1j * rng.randn(n, m)

    elif beta == 4:
        X = rng.randn(n, m) + 1j * rng.randn(n, m)
        Y = rng.randn(n, m) + 1j * rng.randn(n, m)
        A = np.block([[X, Y], [-Y.conj(), X.conj()]])

    else:
        raise ValueError("beta argument must be 1, 2 or 4.")

    return la.eigvalsh(A.dot(A.conj().T))


def sampler_laguerre_tridiagonal(beta, size, shape=1.0, scale=1.0, random_state=None):
    r"""Generate a sample from the Laguerre ``beta`` ensemble of size :math:`n` equal to ``size``, with reference measure :math:`\Gamma(k, \theta)` or equivalently with potential :math:`V(x) = -(k-1)\log(x) + x / \theta`.

    This is done by computing the eigenvalues of a **symmetric tridiagonal** random matrix. The output sample :math:`x=\left(x_{1}, \dots, x_{n}\right)` has joint density proportional to

    .. math::

        \Delta(x_{1}, \dots, x_{n})^{\beta}
        \prod_{i=1}^{n} x_i^{k-1} \exp( -\frac{x_i}{\theta} ).

    :param beta: :math:`\beta \geq 0` parameter of the ensemble.
    :type beta: int

    :param size: Size :math:`n` of the output sample, i.e., size of the matrix to be diagonalized.
    :type size: int
    :param shape: Shape :math:`k` of the Gamma reference distribution, defaults to 0.0

    :param shape: Shape parameter :math:`k` of the Gamma reference distribution, defaults to 0.0
    :type shape: float, optional

    :param scale: Scale parameter :math:`\theta` of the Gamma reference distribution, defaults to 1.0
    :type scale: float, optional

    :param random_state: _description_, defaults to None
    :type random_state: _type_, optional

    :return: Vector of size :math:`n` representing the output sample.
    :rtype: numpy.ndarray

    .. note::

        The rescaled version :math:`\frac{2}{\theta} \frac{1}{\beta m} x` of the sample (with :math:`m = \lfloor \frac{2}{\beta} \shape + n - 1 \rfloor`) has limiting distribution the Marcenko-Pastur distribution.

    .. seealso::

        - :ref:`Tridiagonal matrix model <Laguerre_banded_matrix_model>` associated to the Laguerre ensemble
        - :cite:`DuEd02` III-B
    """

    rng = check_random_state(random_state)

    if not (beta >= 0):
        raise ValueError("beta argument must be non negative (beta >= 0).")

    if beta == 0:
        return rng.gamma(shape=shape, scale=scale)  # even

    # beta/2*[n-1, n-2, ..., 1, 0]
    b_2_Ni = 0.5 * beta * np.arange(size - 1, -1, step=-1)

    # xi_odd = xi_1, ... , xi_2N-1
    xi_odd = rng.gamma(shape=b_2_Ni + shape, scale=scale)  # odd

    # xi_even = xi_0=0, xi_2, ... ,xi_2N-2
    xi_even = np.zeros(size)
    xi_even[1:] = rng.gamma(shape=b_2_Ni[:-1], scale=scale)  # even

    # alpha_i = xi_2i-2 + xi_2i-1, xi_0 = 0
    alpha_coef = xi_even + xi_odd
    # beta_i+1 = xi_2i-1 * xi_2i
    beta_coef = xi_odd[:-1] * xi_even[1:]

    return la.eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))
