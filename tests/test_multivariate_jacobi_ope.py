# coding: utf8
""" Unit tests:

- :class:`TestMultivariateJacobiOPE` check correct implementation of the corresponding class.
"""

import unittest
import numpy as np

from scipy.integrate import quad
from scipy.special import eval_jacobi

import sys
sys.path.append('..')

from dppy.multivariate_jacobi_ope import (MultivariateJacobiOPE,
                                          compute_ordering_BaHa16,
                                          compute_Gautschi_bounds)

from dppy.utils import inner1d, check_random_state


class TestMultivariateJacobiOPE(unittest.TestCase):
    """
    """

    seed = 0

    def test_ordering(self):
        """Make sure the ordering of multi-indices respects the one prescirbed by :cite:`BaHa16` Section 2.1.3
        """

        ord_d2_N16 = [(0, 0),
                      (0, 1), (1, 0), (1, 1),
                      (0, 2), (1, 2), (2, 0), (2, 1), (2, 2),
                      (0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]

        ord_d3_N27 = [(0, 0, 0),
                      (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
                      (0, 0, 2), (0, 1, 2), (0, 2, 0), (0, 2, 1), (0, 2, 2), (1, 0, 2), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2), (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 2, 0), (2, 2, 1), (2, 2, 2)]

        for ord_to_check in (ord_d2_N16, ord_d3_N27):

            N, d = len(ord_to_check), len(ord_to_check[0])

            self.assertTrue(compute_ordering_BaHa16(N, d), ord_to_check)

    def test_square_norms(self):

        N = 100
        dims = np.arange(2, 5)

        max_deg = 50  # to avoid quad warning in dimension 1
        for d in dims:

            jacobi_params = 0.5 - np.random.rand(d, 2)
            jacobi_params[0, :] = -0.5

            dpp = MultivariateJacobiOPE(N, jacobi_params)
            pol_2_eval = dpp.poly_1D_degrees[:max_deg]

            quad_square_norms =\
                [[quad(lambda x:
                        (1-x)**a * (1+x)**b * eval_jacobi(n, a, b, x)**2,
                        -1, 1)[0]
                        for n, a, b in zip(deg,
                                            dpp.jacobi_params[:, 0],
                                            dpp.jacobi_params[:, 1])]
                 for deg in pol_2_eval]

            self.assertTrue(np.allclose(
                            dpp.poly_1D_square_norms[pol_2_eval,
                                                     range(dpp.dim)],
                            quad_square_norms))

    def test_Gautschi_bounds(self):
        """Test if bounds computed w/wo log scale coincide"""

        N = 100
        dims = np.arange(2, 5)

        for d in dims:

            jacobi_params = 0.5 - np.random.rand(d, 2)
            jacobi_params[0, :] = -0.5

            dpp = MultivariateJacobiOPE(N, jacobi_params)

            with_log_scale = compute_Gautschi_bounds(dpp.jacobi_params,
                                                     dpp.ordering,
                                                     log_scale=True)

            without_log_scale = compute_Gautschi_bounds(dpp.jacobi_params,
                                                        dpp.ordering,
                                                        log_scale=False)

            self.assertTrue(np.allclose(with_log_scale, without_log_scale))

    def test_kernel_symmetry(self):
        """
            K(x) == K(x, x)
            K(x, y) == K(y, x)
            K(x, Y) == K(Y, x) = [K(x, y) for y in Y]
            K(X, Y) == [K(x, y) for x, y in zip(X, Y)]
        """
        N = 100
        dims = np.arange(2, 5)

        for d in dims:

            jacobi_params = 0.5 - np.random.rand(d, 2)
            jacobi_params[0, :] = -0.5

            dpp = MultivariateJacobiOPE(N, jacobi_params)

            x, y = np.random.rand(d), np.random.rand(d)
            X, Y = np.random.rand(5, d), np.random.rand(5, d)

            self.assertTrue(np.allclose(dpp.K(x),
                                        dpp.K(x, x)))

            self.assertTrue(
                np.allclose(np.array([dpp.K(x) for x in X]),
                            np.concatenate([dpp.K(x, x) for x in X])))

            self.assertTrue(np.allclose(dpp.K(X),
                                        dpp.K(X, X)))

            self.assertTrue(np.allclose(dpp.K(x, y),
                                        dpp.K(y, x)))

            self.assertTrue(
                np.allclose(dpp.K(x, Y),
                            np.concatenate([dpp.K(x, y) for y in Y])))

            self.assertTrue(np.allclose(dpp.K(x, Y),
                                        dpp.K(Y, x)))

            self.assertTrue(np.allclose(dpp.K(X, Y),
                                        dpp.K(Y, X)))

    def test_kernel_gram_matrix(self):
        """
            K(x) == phi_x.dot(phi_x)
            K(x, y) == phi_x.dot(phi_y)
            K(x, Y) == K(Y, x) = phi_x.dot(phi_y)
            K(X, Y) == [phi_x.dot(phi_y) for x, y in zip(X, Y)]
        """
        N = 100
        dims = np.arange(2, 5)

        for d in dims:

            jacobi_params = 0.5 - np.random.rand(d, 2)
            jacobi_params[0, :] = -0.5

            dpp = MultivariateJacobiOPE(N, jacobi_params)

            x, y = np.random.rand(d), np.random.rand(d)
            phi_x = dpp.eval_poly_multiD(x, normalize='norm')
            phi_y = dpp.eval_poly_multiD(y, normalize='norm')

            X, Y = np.random.rand(5, d), np.random.rand(5, d)
            phi_X = dpp.eval_poly_multiD(X, normalize='norm').T
            phi_Y = dpp.eval_poly_multiD(Y, normalize='norm').T

            self.assertTrue(np.allclose(dpp.K(x),
                                        inner1d(phi_x)))

            self.assertTrue(np.allclose(dpp.K(X),
                                        inner1d(phi_X)))

            self.assertTrue(np.allclose(dpp.K(x, y),
                                        inner1d(phi_x, phi_y)))

            self.assertTrue(np.allclose(dpp.K(x, Y),
                                        phi_x.dot(phi_Y)))

            self.assertTrue(np.allclose(dpp.K(X, Y),
                                        inner1d(phi_X, phi_Y)))

    def test_sample_1D(self):

        N, d = 20, 1
        jacobi_params = - 0.5 * np.ones((d, 2))

        dpp = MultivariateJacobiOPE(N, jacobi_params)
        sampl = dpp.sample(random_state=self.seed)
        # seed = 0
        expected_sample = np.array([[0.9995946],
                                    [0.98944808],
                                    [0.97485733],
                                    [0.86576265],
                                    [0.7958162],
                                    [0.64406931],
                                    [0.53459294],
                                    [0.4259159],
                                    [0.1784497],
                                    [0.12319757],
                                    [-0.13340743],
                                    [-0.28758726],
                                    [-0.40275405],
                                    [-0.68282936],
                                    [-0.76523971],
                                    [-0.82355336],
                                    [-0.88258742],
                                    [-0.94587727],
                                    [-0.96426474],
                                    [-0.99658163]])

        self.assertTrue(np.allclose(sampl, expected_sample))

    def test_sample_2D(self):

        N, d = 20, 2
        jacobi_params = - 0.5 * np.ones((d, 2))

        dpp = MultivariateJacobiOPE(N, jacobi_params)

        sampl = dpp.sample(random_state=self.seed)
        # seed = 0
        expected_sample = np.array([[-0.44929357, -0.92988338],
                                    [-0.38287518, 0.20141371],
                                    [-0.82081218, 0.71370281],
                                    [0.54434591, -0.40548681],
                                    [0.83560033, 0.99975326],
                                    [0.58479869, -0.97294791],
                                    [-0.97140965, -0.9958691],
                                    [0.73176809, 0.0462124],
                                    [0.99999994, 0.62478638],
                                    [-0.995107, 0.6146506],
                                    [0.99956026, -0.54696107],
                                    [-0.73400973, 0.99976412],
                                    [-0.9940169, 0.16811198],
                                    [0.26219327, 0.71667124],
                                    [-0.76730512, -0.05402772],
                                    [0.99910531, 0.90729924],
                                    [0.99984566, -0.95942833],
                                    [0.99996511, -0.01959666],
                                    [0.05053165, -0.40778628],
                                    [0.11004152, 0.99337928]])

        self.assertTrue(np.allclose(sampl, expected_sample))


def main():

    unittest.main()


if __name__ == '__main__':
    main()
