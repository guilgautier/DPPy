import numpy as np
import scipy.linalg as la

from dppy.utils import check_random_state


def sampler_jacobi_full(beta, n, m1, m2, random_state=None):
    r"""Generate a sample from the Jacobi ``beta=1, 2, 4`` ensemble by computing the eigenvalues of the **full** matrix model :math:`A A^* (A A^* + B B^*)^{-1}`, where :math:`A` has size :math:`n \times m_1` and :math:`B` has size :math:`n \times m_2`.

    The output sample :math:`x=\left(x_{1}, \dots, x_{n}\right)` has joint density proportional to

    .. math::

        \Delta(x_{1}, \dots, x_{n})^{\beta}
        \\prod_{n=1}^{n}
            x_i^{\frac{\beta}{2}(m_1 - n + 1) - 1}
            (1-x_i)^{\frac{\beta}{2}(m_2 - n + 1) - 1}

    The equivalent banded model :py:func:`~dppy.beta_ensembles.jacobi.sampler_jacobi_tridiagonal` can be called as
    ``sampler_jacobi_tridiagonal(beta=beta, size=n, a=a, b=b)`` with ``a = beta / 2 * (m1 - n + 1)`` and  ``b = beta / 2 * (m1 - n + 1)``.

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

        The output sample has limiting distribution the Wachter distribution.
    """
    rng = check_random_state(random_state)

    if beta == 1:
        X = rng.randn(n, m1)
        Y = rng.randn(n, m2)

    elif beta == 2:
        X = rng.randn(n, m1) + 1j * rng.randn(n, m1)
        Y = rng.randn(n, m2) + 1j * rng.randn(n, m2)

    elif beta == 4:
        X_1 = rng.randn(n, m1) + 1j * rng.randn(n, m1)
        X_2 = rng.randn(n, m1) + 1j * rng.randn(n, m1)

        Y_1 = rng.randn(n, m2) + 1j * rng.randn(n, m2)
        Y_2 = rng.randn(n, m2) + 1j * rng.randn(n, m2)

        X = np.block([[X_1, X_2], [-X_2.conj(), X_1.conj()]])
        Y = np.block([[Y_1, Y_2], [-Y_2.conj(), Y_1.conj()]])

    else:
        raise ValueError("beta argument must be 1, 2 or 4.")

    X_tmp = X.dot(X.conj().T)
    Y_tmp = Y.dot(Y.conj().T)

    return la.eigvals(X_tmp.dot(la.inv(X_tmp + Y_tmp))).real


def sampler_jacobi_tridiagonal(beta, size, a=1.0, b=1.0, random_state=None):
    r"""Generate a sample from the Jacobi ``beta`` ensemble of size :math:`n` equal to ``size``, with reference measure :math:`\Beta(a, b)` or equivalently with potential :math:`V(x) = -(a-1)\log(x) -(b-1)\log(1-x)`.

    This is done by computing the eigenvalues of a **symmetric tridiagonal** random matrix. The output sample :math:`x=\left(x_{1}, \dots, x_{n}\right)` has joint density proportional to

    .. math::

        \Delta(x_{1}, \dots, x_{n})^{\beta}
        \prod_{i=1}^{n} x_i^{k-1} \exp( -\frac{x_i}{\theta} ).

    :param beta: :math:`\beta \geq 0` parameter of the ensemble.
    :type beta: int

    :param size: Size :math:`n` of the output sample, i.e., size of the matrix to be diagonalized.
    :type size: int

    :param a: Positive shape parameter :math:`a > 0` of the Beta reference distribution, defaults to 1.0.
    :type a: float, optional

    :param b: Positive shape parameter :math:`b > 0` of the Beta reference distribution, defaults to 1.0.
    :type b: float, optional

    :param random_state: _description_, defaults to None
    :type random_state: _type_, optional

    :return: Vector of size :math:`n` representing the output sample.
    :rtype: numpy.ndarray

    .. note::

        The output sample has limiting distribution the Wachter distribution.

    .. seealso::

        :cite:`KiNe04` Theorem 2
    """

    rng = check_random_state(random_state)

    if not (beta >= 0):
        raise ValueError("beta argument must be non negative (beta >= 0).")

    if beta == 0:
        return rng.beta(a, b, size=size)

    # beta/2*[n-1, n-2, ..., 1, 0]
    b_2_Ni = 0.5 * beta * np.arange(size - 1, -1, step=-1)

    # c_odd = c_1, c_3, ..., c_2N-1
    c_odd = rng.beta(b_2_Ni + a, b_2_Ni + b)

    # c_even = c_0, c_2, c_2N-2
    c_even = np.zeros(size)
    c_even[1:] = rng.beta(b_2_Ni[:-1], b_2_Ni[1:] + a + b)

    # xi_odd = xi_2i-1 = (1-c_2i-2) c_2i-1
    xi_odd = (1 - c_even) * c_odd

    # xi_even = xi_0=0, xi_2, xi_2N-2
    # xi_2i = (1-c_2i-1)*c_2i
    xi_even = np.zeros(size)
    xi_even[1:] = (1 - c_odd[:-1]) * c_even[1:]

    # alpha_i = xi_2i-2 + xi_2i-1
    # alpha_1 = xi_0 + xi_1 = xi_1
    alpha_coef = xi_even + xi_odd
    # beta_i+1 = xi_2i-1 * xi_2i
    beta_coef = xi_odd[:-1] * xi_even[1:]

    return la.eigvalsh_tridiagonal(alpha_coef, np.sqrt(beta_coef))
