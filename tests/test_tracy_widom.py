# coding: utf8

import unittest

import numpy as np

import sys
sys.path.append('..')
from dppy.beta_ensemble_polynomial_potential_core import TracyWidom


class TestTracyWidom(unittest.TestCase):
    """ Based on the work of Bornemann 2010 `https://arxiv.org/pdf/0804.2543.pdf <https://arxiv.org/pdf/0804.2543.pdf>`_
    """

    TW = TracyWidom()

    def test_kernel_example_bornemann_fredholm_determinant_should_equal_sin1(self):
        """ Equation 5.8 Bornemann
        """

        def K_Green(x, y):
            Y, X = np.meshgrid(x, y)
            return np.where(X <= Y, X * (1 - Y), Y * (1 - X))

        quad_order = 50
        x_quad, w_quad = self.TW.compute_quadrature(quad_order)
        fred_det_K_approx = self.TW.fredholm_determinant(K_Green,
                                                         x_quad,
                                                         w_quad)
        fred_det_K_theo = np.sin(1)

        self.assertAlmostEqual(fred_det_K_approx, fred_det_K_theo,
                               msg=(fred_det_K_approx, fred_det_K_theo),
                               delta=1e-5)

    def test_change_of_variables_from_0_1_to_s_oo_should_be_increasing(self):
        """
        .. todo::

            Add refer to increasing choice
        """
        points = np.linspace(0, 1, 10)
        s = -1
        phi, d_phi = self.TW.change_of_variable(s)

        for x, y in zip(points[:-1], points[1:]):
            with self.subTest(x=x, y=y):
                self.assertLessEqual(phi(x), phi(y))

    def test_change_of_variables_from_0_1_to_s_oo_derivative_is_correct(self):

        points = np.linspace(0, 1, 10, endpoint=False)
        s = -1
        phi, d_phi = self.TW.change_of_variable(s)

        eps = 1e-7

        for x in points:
            with self.subTest(x=x):
                d_phi_x_approx = (phi(x + eps) - phi(x)) / eps
                d_phi_x = d_phi(x)

                self.assertAlmostEqual(d_phi_x_approx, d_phi_x,
                                       msg=(x, d_phi_x_approx, d_phi_x),
                                       delta=1e-2)

    def test_evaluation_Tracy_Widom_cdf(self):
        """ evalution points obtained from Table 5. in *LARGEST EIGENVALUES AND SAMPLE COVARIANCE MATRICES*, ANDREI IU. BEJAN
        https://pdfs.semanticscholar.org/ca19/3484415f374d8fb02e7fbdad72b99727b41f.pdf?_ga=2.251544262.1964171041.1570206947-237360766.1567514713
        """
        points = np.array([[-3.0, 0.080361],
                           [-2.5, 0.212392],
                           [-2.0, 0.413256],
                           [-1.5, 0.631401],
                           [-1.0, 0.807225],
                           [-0.5, 0.916070],
                           [0.0, 0.969375],
                           [0.5, 0.990545],
                           [1.0, 0.997506],
                           [1.5, 0.999432],
                           [2.0, 0.999888]])

        quad_order = 50

        tol = 1e-4

        cdf_s_approx = self.TW.cdf(points[:, 0], quad_order)
        self.assertTrue(np.allclose(cdf_s_approx, points[:, 1], atol=tol))


def main():

    unittest.main()


if __name__ == '__main__':
    main()
