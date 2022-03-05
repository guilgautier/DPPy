"""Tests of the TracyWidow class based on the work of Bornemann 2010 `https://arxiv.org/pdf/0804.2543.pdf <https://arxiv.org/pdf/0804.2543.pdf>`_"""

import numpy as np
import pytest

from dppy.beta_ensembles.beta_ensemble_polynomial_potential_core import TracyWidom

TW = TracyWidom()


def test_kernel_example_bornemann_fredholm_determinant_should_equal_sin1():
    """Equation 5.8 Bornemann"""

    def K_Green(x, y):
        Y, X = np.meshgrid(x, y)
        return np.where(X <= Y, X * (1 - Y), Y * (1 - X))

    quad_order = 50
    x_quad, w_quad = TW.compute_quadrature(quad_order)
    approx = TW.fredholm_determinant(K_Green, x_quad, w_quad)
    expected = np.sin(1)

    np.testing.assert_almost_equal(approx, expected, decimal=5)


@pytest.mark.parametrize(
    "x, y",
    [
        (np.linspace(0, 1, 10)[:-1], np.linspace(0, 1, 10)[1:]),
    ],
)
def test_change_of_variables_from_0_1_to_s_oo_should_be_increasing(x, y):
    s = -1
    phi, *_ = TW.change_of_variable(s)
    np.testing.assert_array_less(phi(x), phi(y))


@pytest.mark.parametrize("x", np.linspace(0, 1, 10, endpoint=False))
def test_change_of_variables_from_0_1_to_s_oo_derivative_is_correct(x):
    s = -1
    phi, d_phi = TW.change_of_variable(s)

    eps = 1e-7
    approx = (phi(x + eps) - phi(x)) / eps
    computed = d_phi(x)

    np.testing.assert_almost_equal(approx, computed, decimal=2)


@pytest.mark.parametrize(
    "x, cdf_x",
    [
        [-3.0, 0.080361],
        [-2.5, 0.212392],
        [-2.0, 0.413256],
        [-1.5, 0.631401],
        [-1.0, 0.807225],
        [-0.5, 0.916070],
        [0.0, 0.969375],
        [0.5, 0.990545],
        [1.0, 0.997506],
        [1.5, 0.999432],
        [2.0, 0.999888],
    ],
)
def test_evaluation_Tracy_Widom_cdf(x, cdf_x):
    """evalution points obtained from Table 5. in *LARGEST EIGENVALUES AND SAMPLE COVARIANCE MATRICES*, ANDREI IU. BEJAN
    https://pdfs.semanticscholar.org/ca19/3484415f374d8fb02e7fbdad72b99727b41f.pdf?_ga=2.251544262.1964171041.1570206947-237360766.1567514713
    """
    quad_order = 50
    cdf_x_computed = TW.cdf(x, quad_order)
    np.testing.assert_almost_equal(cdf_x_computed, cdf_x, decimal=4)
