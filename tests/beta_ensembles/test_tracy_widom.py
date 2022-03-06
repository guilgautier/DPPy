"""Tests of the TracyWidow cumulative distribution function based on the work of Bornemann [2010] doi.org/10.1090/S0025-5718-09-02280-7 `https://arxiv.org/pdf/0804.2543.pdf <https://arxiv.org/pdf/0804.2543.pdf>`_"""

import numpy as np
import pytest

from dppy.beta_ensembles.tracy_widom import (
    TracyWidom,
    _change_of_variable,
    fredholm_determinant,
    quadrature_legendre_01,
)


def test_kernel_example_bornemann_fredholm_determinant_should_equal_sin1():
    """Equation 5.8 Bornemann doi.org/10.1090/S0025-5718-09-02280-7 or https://arxiv.org/abs/0804.2543"""

    def K_Green(x, y):
        Y, X = np.meshgrid(x, y)
        return np.where(X <= Y, X * (1 - Y), Y * (1 - X))

    quad_order = 50
    x_quad, w_quad = quadrature_legendre_01(quad_order)
    approx = fredholm_determinant(K_Green, x_quad, w_quad)
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
    phi, *_ = _change_of_variable(s)
    np.testing.assert_array_less(phi(x), phi(y))


@pytest.mark.parametrize(
    "x, cdf_x",
    [
        [-3.5, 0.0258941469930181],
        [-3.0, 0.06960011886736989],
        [-2.5, 0.1513777291781468],
        [-2.0, 0.2743201979092179],
        [-1.5, 0.4262258019375784],
        [-1.0, 0.5837898955197323],
        [-0.5, 0.7236910515483814],
        [0.0, 0.8319080662029519],
        [0.5, 0.9059711835032338],
        [1.0, 0.9514212369115508],
        [1.5, 0.9767136486166604],
        [2.0, 0.989597571084827],
        [2.5, 0.9956520227045179],
        [3.0, 0.9982934803498806],
    ],
)
def test_evaluation_Tracy_Widom_beta_1_cdf(x, cdf_x):
    # See relevant table at http://www-m5.ma.tum.de/KPZ
    TW = TracyWidom()
    cdf_x_computed = TW.cdf(x, beta=1, quad_order=50)
    np.testing.assert_almost_equal(cdf_x_computed, cdf_x, decimal=10)


@pytest.mark.parametrize(
    "x, cdf_x",
    [
        [-3.5, 0.02096769149276654],
        [-3.0, 0.08031955293933454],
        [-2.5, 0.2123514258195901],
        [-2.0, 0.4132241425051226],
        [-1.5, 0.631380876420726],
        [-1.0, 0.8072142419992853],
        [-0.5, 0.9160651890092872],
        [0.0, 0.9693728283552626],
        [0.5, 0.9905446073837164],
        [1.0, 0.9975054381493893],
        [1.5, 0.9994322343111557],
        [2.0, 0.9998875536983092],
    ],
)
def test_evaluation_Tracy_Widom_beta_2_cdf(x, cdf_x):
    # See relevant table at http://www-m5.ma.tum.de/KPZ
    TW = TracyWidom()
    cdf_x_computed = TW.cdf(x, beta=2, quad_order=50)
    np.testing.assert_almost_equal(cdf_x_computed, cdf_x, decimal=10)


@pytest.mark.parametrize(
    "x, cdf_x",
    [
        [-3.5, 0.043396],
        [-3.0, 0.167754],
        [-2.5, 0.404066],
        [-2.0, 0.673527],
        [-1.5, 0.867876],
        [-1.0, 0.960754],
        [-0.5, 0.991370],
        [0.0, 0.998574],
        [0.5, 0.999820],
    ],
)
def test_evaluation_Tracy_Widom_beta_4_cdf(x, cdf_x):
    # See relevant table at
    # Largest eigenvalues and sample covariance matrices. tracy-widom and painlevé ii: computational aspects and realization in s-plus with applications
    # Andrei Bejan
    # https://pdfs.semanticscholar.org/ca19/3484415f374d8fb02e7fbdad72b99727b41f.pdf?_ga=2.251544262.1964171041.1570206947-237360766.1567514713
    TW = TracyWidom()
    cdf_x_computed = TW.cdf(x, beta=4, quad_order=50)
    np.testing.assert_almost_equal(cdf_x_computed, cdf_x, decimal=4)
