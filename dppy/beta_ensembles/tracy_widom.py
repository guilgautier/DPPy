import numpy as np
import scipy.linalg as la
from black import out
from scipy.special import airy


class TracyWidom(object):
    r"""Implements computation of the cumulative distribution function (cdf) of the Tracy Widom :math:`TW_{\beta}` distribution for :math:`\beta = 1, 2, 4`.

    .. seealso::

        - `TracyWidow.jl from RandomMatrices.jl <https://github.com/JuliaMath/RandomMatrices.jl/blob/f893fb395af78b7063b9a26032640341744922b3/src/densities/TracyWidom.jl>`_
        - `TracyWidom of yymao <https://github.com/yymao/TracyWidom>`_
    """

    def __init__(self):
        self.quad_order = None
        self.x_quad, self.w_quad = None, None

    def cdf(self, x, beta=2, quad_order=50):
        r"""Compute cumulative distribution function :math:`F_{TW_{\beta}}` of the Tracy-Widom :math:`TW_{\beta}` distribution.

        :param x: Evalutation points
        :type x: float

        :param beta: :math:`\beta` parameter of the distribution (1, 2, 4), defaults to 2.
        :type beta: int

        :param quad_order: order (degree) of the quadrature used to compute the underlying Fredholm determinant.
        :type quad_order: int
        """
        assert beta in (1, 2, 4)

        if self.quad_order != quad_order:
            self.quad_order = quad_order
            self.x_quad, self.w_quad = quadrature_legendre_01(quad_order)

        x_quad, w_quad = self.x_quad, self.w_quad

        if beta == 1:
            K1 = _kernel1(x)
            return fredholm_determinant(K1, x_quad, w_quad)

        if beta == 2:
            K2 = _kernel2(x)
            return fredholm_determinant(K2, x_quad, w_quad)

        if beta == 4:
            K1 = _kernel1(x * np.sqrt(2))
            K2 = _kernel2(x * np.sqrt(2))
            det_I_K1 = fredholm_determinant(K1, x_quad, w_quad)
            det_I_K2 = fredholm_determinant(K2, x_quad, w_quad)
            return 0.5 * (det_I_K1 + det_I_K2 / det_I_K1)


def _change_of_variable(s):
    """Bornemann Equation 7.5"""

    def phi(x):
        return s + 10.0 * np.tan(0.5 * np.pi * x)

    def d_phi(x):
        return 5.0 * np.pi * (1.0 + np.tan(0.5 * np.pi * x) ** 2)

    return phi, d_phi


def _kernel1(s):
    phi, d_phi = _change_of_variable(s)

    def K_s(x, y):
        K = 0.5 * airy(0.5 * np.add.outer(phi(x), phi(y)))[0]
        K *= np.outer(np.sqrt(d_phi(x)), np.sqrt(d_phi(y)))
        return K

    return K_s


def _kernel2(s):
    phi, d_phi = _change_of_variable(s)

    def K_s(x, y):
        K = airy_kernel(phi(x), phi(y))
        K *= np.outer(np.sqrt(d_phi(x)), np.sqrt(d_phi(y)))
        return K

    return K_s


def airy_kernel(x, y):
    """Evaluate the `Airy kernel <https://en.wikipedia.org/wiki/Tracy%E2%80%93Widom_distribution>`_."""
    # Ai_x, dAi_x, _, _ = airy(x)
    # if x == y:
    #     # L'Hopital's rule + d2 Ai - x Ai = 0
    #     return dAi_x ** 2 - x * Ai_x ** 2
    # Ai_y, dAi_y, _, _ = airy(y)
    # return (Ai_x * dAi_y - dAi_x * Ai_y) / (x - y)

    # Vectorized version
    Ai_x, dAi_x, _, _ = airy(x)
    Ai_y, dAi_y, _, _ = airy(y)

    K_xy = np.outer(Ai_x, dAi_y)
    K_xy -= np.outer(dAi_x, Ai_y)

    x_y = np.subtract.outer(x, y)

    if y is x or np.all(y == x):
        # avoid division by 0 on the diagonal
        np.fill_diagonal(x_y, 1.0)
        K_xy /= x_y
        # L'Hopital's rule + d2 Ai - x Ai = 0
        A_xx = dAi_x ** 2 - x * Ai_x ** 2
        np.fill_diagonal(K_xy, A_xx)
    else:
        K_xy /= x_y

    return K_xy.squeeze()


def quadrature_legendre_01(order):
    x, w = np.polynomial.legendre.leggauss(order)
    return 0.5 * (x + 1), 0.5 * w


def fredholm_determinant(kernel, x_quad, w_quad):
    """This implements the numerical evaluation of Fredholm determinants following `Borneman [2010, Equation 6.1.] <doi.org/10.1090/S0025-5718-09-02280-7>`_"""
    w = np.sqrt(w_quad)
    K = kernel(x_quad, x_quad)
    K *= np.outer(w, w)  # w[:, None] * K * w
    I = np.eye(*K.shape)
    return la.det(I - K)
