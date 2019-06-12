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

from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE


class TestMultivariateJacobiOPE(unittest.TestCase):
    """
    """

    N = 100
    dims = np.arange(2, 5)

    def test_kernel_symmetry(self):
        """
            K(x) == K(x, x)
            K(x, y) == K(y, x)
            K(x, Y) == K(Y, x) = [K(x, y) for y in Y]
            K(X, Y) == [K(x, y) for x, y in zip(X, Y)]
        """

        for d in self.dims:

            jacobi_params = 0.5 - np.random.rand(d, 2)
            jacobi_params[0, :] = -0.5

            dpp = MultivariateJacobiOPE(self.N, jacobi_params)

            x, y = np.random.rand(d), np.random.rand(d)
            X, Y = np.random.rand(5, d), np.random.rand(5, d)

            self.assertTrue(np.allclose(dpp.K(x),
                                        dpp.K(x, x)))

            self.assertTrue(
                np.allclose(np.array(
                                [dpp.K(x)for x in X]),
                            np.concatenate(
                                [dpp.K(x, x) for x in X])))

            self.assertTrue(np.allclose(dpp.K(X),
                                        dpp.K(X, X)))

            self.assertTrue(np.allclose(dpp.K(x, y),
                                        dpp.K(y, x)))

            self.assertTrue(np.allclose(
                                dpp.K(x, Y),
                            np.concatenate(
                                [dpp.K(x, y) for y in Y])))

            self.assertTrue(np.allclose(dpp.K(x, Y),
                                        dpp.K(Y, x)))

            self.assertTrue(np.allclose(dpp.K(X, Y),
                                        dpp.K(Y, X)))

    def test_square_norms(self):

        max_deg = 50  # to avoid quad warning in dimension 1
        for d in self.dims:

            jacobi_params = 0.5 - np.random.rand(d, 2)
            jacobi_params[0, :] = -0.5

            dpp = MultivariateJacobiOPE(self.N, jacobi_params)
            pol_2_eval = dpp.poly_degrees[:max_deg]

            quad_square_norms =\
                [[quad(lambda x:
                        (1-x)**a * (1+x)**b * eval_jacobi(n, a, b, x)**2,
                        -1, 1)[0]
                        for n, a, b in zip(deg,
                                            dpp.jacobi_params[:, 0],
                                            dpp.jacobi_params[:, 1])]
                 for deg in pol_2_eval]

            self.assertTrue(np.allclose(
                                dpp.square_norms[pol_2_eval, range(dpp.dim)],
                                quad_square_norms))


def main():

    unittest.main()


if __name__ == '__main__':
    main()
