# coding: utf8
""" Unit tests:

- :class:`TestRSKCorrespondence`
"""


import unittest

import numpy as np
import numpy.linalg as la
import numpy.random as rndm

from scipy.linalg import qr

import sys
sys.path.append('..')
from dppy import utils


class TestUtils(unittest.TestCase):
    """ Test
    """
    def test_det_ST(self):
        """Test determinant
            - det_ST(arr, S) = det(arr[S, S])
            - det_ST(arr, S, T) = det(arr[S, T])
        """

        shapes = [10, 50, 100, 300]
        nb_minors = 10

        for sh in shapes:

            arr = rndm.rand(sh, sh)
            size_minors = sh // 3

            for _ in range(nb_minors):
                S, T = rndm.choice(sh, size=(2, size_minors))

                self.assertTrue(
                    np.allclose(utils.det_ST(arr, S),
                                la.det(arr[np.ix_(S, S)])))

                self.assertTrue(
                    np.allclose(utils.det_ST(arr, S, T),
                                la.det(arr[np.ix_(S, T)])))

    def test_inner_products_and_square_norms(self):

        X = rndm.rand(10, 20, 30, 40)
        Y = rndm.rand(*X.shape)

        for ax in range(X.ndim):

            # inner product
            self.assertTrue(
                np.allclose(utils.inner1d(X, Y, axis=ax),
                            (X * Y).sum(axis=ax)))
            # square norm
            self.assertTrue(
                np.allclose(utils.inner1d(X, axis=ax),
                            (X**2).sum(axis=ax)))

    def test_symmetry(self):

        X = rndm.randn(20, 20)
        sym_part = 0.5 * (X + X.T)
        self.assertTrue(
            np.allclose(utils.is_symmetric(sym_part),
                        sym_part))

        with self.assertRaises(ValueError) as context:
            utils.is_symmetric(X)

        self.assertTrue('M.T != M' in str(context.exception))

        Y = rndm.randn(20, 30)
        cov = Y.T.dot(Y)
        self.assertTrue(
            np.allclose(utils.is_symmetric(cov),
                        cov))

        with self.assertRaises(ValueError) as context:
            utils.is_symmetric(Y)

        self.assertTrue('M.T != M' in str(context.exception))

    def test_projection(self):

        N, rank = 30, 10

        # First test
        X = rndm.randn(N, rank)
        K_proj_1 = X.dot(la.inv(X.T.dot(X)).dot(X.T))

        self.assertTrue(
            np.allclose(utils.is_projection(K_proj_1),
                        K_proj_1))

        with self.assertRaises(ValueError) as context:
            utils.is_projection(X.dot(X.T))

        self.assertTrue('M^2 != M' in str(context.exception))

        # Second test
        Y = rndm.randn(N, rank)
        eig_vecs, _ = qr(Y, mode="economic")
        K_proj_2 = eig_vecs.dot(eig_vecs.T)

        self.assertTrue(
            np.allclose(utils.is_projection(K_proj_2),
                        K_proj_2))

        with self.assertRaises(ValueError) as context:
            utils.is_projection(Y.dot(Y.T))

        self.assertTrue('M^2 != M' in str(context.exception))

    def test_orthonormal(self):

        N, rank = 30, 10
        X = rndm.randn(N, rank)
        eig_vecs, _ = qr(X, mode="economic")

        self.assertTrue(
            np.allclose(utils.is_orthonormal(eig_vecs),
                        eig_vecs))

        with self.assertRaises(ValueError) as context:
            utils.is_orthonormal(X)

        self.assertTrue('M.T M != I' in str(context.exception))

    def test_is_equal_to_O_or_1(self):

        N, p = 100, 0.5

        bernoullis = rndm.rand(N) < p
        self.assertTrue(
            np.allclose(utils.is_equal_to_O_or_1(bernoullis),
                        bernoullis))

        gaussian = rndm.randn(N)
        gaussian[:N // 2] = bernoullis[:N // 2]
        with self.assertRaises(ValueError) as context:
            utils.is_equal_to_O_or_1(gaussian)

        self.assertTrue('not all in {0,1}' in str(context.exception))

    def test_is_in_01(self):

        N = 100

        unif = rndm.rand(N)
        self.assertTrue(
            np.allclose(utils.is_in_01(unif),
                        unif))

        gaussian = 5 * rndm.randn(N)
        gaussian[:N // 2] = unif[:N // 2]
        with self.assertRaises(ValueError) as context:
            utils.is_in_01(gaussian)

        self.assertTrue('not all in [0,1]' in str(context.exception))

    def test_is_geq_0(self):

        N, lam = 100, 4

        poisson = rndm.poisson(lam=lam, size=N)
        self.assertTrue(
            np.allclose(utils.is_geq_0(poisson),
                        poisson))

        gaussian = 5 * rndm.randn(N)
        gaussian[:N // 2] = poisson[:N // 2]
        with self.assertRaises(ValueError) as context:
            utils.is_geq_0(gaussian)

        self.assertTrue('not all >= 0' in str(context.exception))

    def test_is_full_row_rank(self):

        N, rank = 30, 10
        X = rndm.randn(rank, N)

        self.assertTrue(
            np.allclose(utils.is_full_row_rank(X),
                        X))

        Y = np.vstack([X, rndm.randn(rank).dot(X)])
        with self.assertRaises(ValueError) as context:
            utils.is_full_row_rank(Y)

        self.assertTrue('not full row rank' in str(context.exception))


def main():

    unittest.main()


if __name__ == '__main__':
    main()
