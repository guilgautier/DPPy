import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from dppy.utils import check_random_state


def sampler_circular_full(beta, n, random_state=None):
    r"""Generate a sample from the Circular ``beta=1, 2`` ensemble by computing the eigenvalues of a random :math:`n \times n` orthogonal (\beta=1), resp. unitary (\beta=2) matrix of size :math:`n \times n` drawn from the Haar measure on the associate group.

    The output sample :math:`x=\left(e^{i \theta_{1}}, \dots, e^{i \theta_{n}} \right)` has joint density proportional to

    .. math::

        \Delta(e^{i \theta_{1}}, \dots, e^{i \theta_{n})^{\beta}
        \\prod_{n=1}^{n}
            1_{[0, 2\pi]}(\theta_i)

    The equivalent banded model :py:func:`~dppy.beta_ensembles.jacobi.sampler_circular_quindiagonal` can be called as
    ``sampler_circular_quindiagonal(beta=beta, size=n)``.

    :param beta: :math:`\beta \in \{1, 2, 4\}` parameter of the ensemble.
    :type beta: int

    :param n: Size :math:`n` of the output sample, i.e., size of the matrix to be diagonalized.
    :type n: int

    :param random_state: _description_, defaults to None
    :type random_state: _type_, optional

    :return: Vector of size :math:`n` representing the output sample.
    :rtype: numpy.ndarray

    .. seealso::

        - :ref:`Full matrix model <circular_full_matrix_model>` associated to the Circular ensemble
        - :cite:`Mez06` Section 5
    """
    rng = check_random_state(random_state)

    if beta not in [1, 2]:
        raise ValueError("beta argument must be 1 or 2.")

    A = np.zeros((n, n), dtype=float if beta == 1 else complex)
    if beta == 1:  # COE
        A[:, :] = rng.randn(n, n)

    elif beta == 2:  # CUE
        A.real = rng.randn(n, n)
        A.imag = rng.randn(n, n)

    Q, R = la.qr(A, pivoting=False)
    d = np.diagonal(R)
    U = np.multiply(Q, d / np.abs(d))

    return la.eigvals(U)


def sampler_circular_quindiagonal(beta, size, random_state=None):
    r"""Generate a sample from the Circular ``beta \geq 0`` ensemble by computing the eigenvalues of a random quindiagonal :math:`n \times n` matrix.

    The output sample :math:`x=\left(e^{i \theta_{1}}, \dots, e^{i \theta_{n}} \right)` has joint density proportional to

    .. math::

        \Delta(e^{i \theta_{1}}, \dots, e^{i \theta_{n})^{\beta}
        \\prod_{n=1}^{n}
            1_{[0, 2\pi]}(\theta_i)

    :param beta: :math:`\beta \geq 0` parameter of the ensemble.
    :type beta: int

    :param n: Size :math:`n` of the output sample, i.e., size of the matrix to be diagonalized.
    :type n: int

    :param random_state: _description_, defaults to None
    :type random_state: _type_, optional

    :return: Vector of size :math:`n` representing the output sample.
    :rtype: numpy.ndarray

    .. seealso::

        - :ref:`Quindiagonal matrix model <circular_banded_matrix_model>` associated to the Circular ensemble
        - :cite:`KiNe04` Theorem 1
    """

    rng = check_random_state(random_state)

    if not (isinstance(beta, int) and (beta >= 0)):
        raise ValueError("beta argument must be non negative integer.")

    if beta == 0:  # Answer issue #28 raised by @rbardenet
        # i.i.d. points uniformly on the circle
        theta = rng.uniform(0.0, 2 * np.pi, size=size)
        return np.exp(1j * theta)

    alpha = np.zeros(size, dtype=complex)

    # nu = 1 + beta*(N-1, N-2, ..., 0)
    for i, nu in enumerate(1 + beta * np.arange(size - 1, -1, step=-1)):
        gauss_vec = rng.randn(nu + 1)
        alpha[i] = (gauss_vec[0] + 1j * gauss_vec[1]) / la.norm(gauss_vec)

    rho = np.sqrt(1 - np.abs(alpha[:-1]) ** 2)

    xi = np.zeros((size - 1, 2, 2), dtype=complex)  # xi[0,..,N-1]
    xi[:, 0, 0], xi[:, 0, 1] = alpha[:-1].conj(), rho
    xi[:, 1, 0], xi[:, 1, 1] = rho, -alpha[:-1]

    # L = diag(xi_0, xi_2, ...)
    # M = diag(1, xi_1, x_3, ...)
    # xi[N-1] = alpha[N-1].conj()
    if size % 2 == 0:  # even
        L = sp.block_diag(xi[::2, :, :], dtype=complex)
        M = sp.block_diag([1.0, *xi[1::2, :, :], alpha[-1].conj()], dtype=complex)
    else:  # odd
        L = sp.block_diag([*xi[::2, :, :], alpha[-1].conj()], dtype=complex)
        M = sp.block_diag([1.0, *xi[1::2, :, :]], dtype=complex)

    return la.eigvals(L.dot(M).toarray())
