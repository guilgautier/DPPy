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
                                          compute_ordering,
                                          compute_rejection_bounds)

from dppy.utils import is_symmetric


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

        orderings = [ord_d2_N16, ord_d3_N27]

        for idx, ord_to_check in enumerate(orderings):
            with self.subTest(idx=idx):

                N, d = len(ord_to_check), len(ord_to_check[0])

                self.assertTrue(compute_ordering(N, d), ord_to_check)

    def test_norms_of_multiD_polynomials(self):

        N = 100
        dims = np.arange(2, 5)

        max_deg = 50  # to avoid quad warning in dimension 1
        for d in dims:
            with self.subTest(dimension=d):

                jacobi_params = 0.5 - np.random.rand(d, 2)
                jacobi_params[0, :] = -0.5

                dpp = MultivariateJacobiOPE(N, jacobi_params)
                pol_2_eval = dpp.degrees_1D_polynomials[:max_deg]

                quad_square_norms =\
                    [[quad(lambda x:
                            (1-x)**a * (1+x)**b * eval_jacobi(n, a, b, x)**2,
                            -1, 1)[0]
                            for n, a, b in zip(deg,
                                               dpp.jacobi_params[:, 0],
                                               dpp.jacobi_params[:, 1])]
                     for deg in pol_2_eval]

                self.assertTrue(
                    np.allclose(
                        dpp.norms_1D_polynomials[pol_2_eval, range(dpp.dim)],
                        np.sqrt(quad_square_norms)))

    def test_Gautschi_bounds(self):
        """Test if bounds computed w/wo log scale coincide"""

        N = 100
        dims = np.arange(2, 5)

        for d in dims:
            with self.subTest(dimension=d):

                jacobi_params = 0.5 - np.random.rand(d, 2)
                jacobi_params[0, :] = -0.5

                dpp = MultivariateJacobiOPE(N, jacobi_params)

                with_log_scale = compute_rejection_bounds(dpp.jacobi_params,
                                                          dpp.ordering,
                                                          log_scale=True)

                without_log_scale = compute_rejection_bounds(dpp.jacobi_params,
                                                             dpp.ordering,
                                                             log_scale=False)

                self.assertTrue(np.allclose(with_log_scale, without_log_scale))

    def test_kernel_evaluations(self):
        N = 100
        dims = np.arange(2, 5)

        for d in dims:
            with self.subTest(dimension=d):

                jacobi_params = 0.5 - np.random.rand(d, 2)
                jacobi_params[0, :] = -0.5

                dpp = MultivariateJacobiOPE(N, jacobi_params)

                X = np.random.rand(20, d)
                Y = np.random.rand(20, d)

                K_XX = is_symmetric(dpp.K(X, X))
                K_xx = np.diag(K_XX)
                K_xy = np.ravel([dpp.K(x, y) for x, y in zip(X, Y)])

                checks = ((dpp.K(X), K_XX),
                          (dpp.K(X, X, eval_pointwise=True), K_xx),
                          (dpp.K(X, Y, eval_pointwise=True), K_xy))

                for idx, (a, b) in enumerate(checks):
                    with self.subTest(idx=idx):
                        self.assertTrue(np.allclose(a, b),
                                        'a={}, b={}'.format(a, b))

    def test_sample_1D(self):

        N, d = 20, 1
        jacobi_params = - 0.5 * np.ones((d, 2))

        dpp = MultivariateJacobiOPE(N, jacobi_params)
        sampl = dpp.sample(random_state=self.seed)  # seed = 0
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

        sampl = dpp.sample(random_state=self.seed)  # seed = 0

        expected_sample = np.array([[-0.44929357, -0.92988338],
                                    [0.07128896, -0.98828901],
                                    [-0.43895328, -0.64850438],
                                    [-0.56491996, 0.43632636],
                                    [0.33859341, 0.6642957],
                                    [-0.89437538, -0.98384996],
                                    [0.93451148, -0.42788073],
                                    [-0.81846092, 0.57000777],
                                    [-0.42084694, 0.98065145],
                                    [0.97651548, 0.94243444],
                                    [0.11753084, 0.96240585],
                                    [-0.12183308, -0.14093164],
                                    [-0.9940169, 0.16811198],
                                    [-0.76730512, -0.05402772],
                                    [0.99984566, -0.95942833],
                                    [0.99996511, -0.01959666],
                                    [0.05053165, -0.40778628],
                                    [0.82158181, 0.58501064],
                                    [-0.97396649, 0.90805501],
                                    [-0.99808676, -0.49690354]])

        self.assertTrue(np.allclose(sampl, expected_sample))


def main():

    unittest.main()


if __name__ == '__main__':
    main()
