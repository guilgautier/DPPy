# coding: utf8
""" Unit tests:

- :class:`TestUtils`
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
    def test_inner1D_to_compute_inner_product_and_square_norms(self):

        shape = (10, 20, 30, 40)
        X = rndm.rand(*shape)
        Y = rndm.rand(*shape)

        for ax in range(len(shape)):
            with self.subTest(axis=ax):

                for test_inner1D in ['inner_prod', 'sq_norm']:
                    with self.subTest(test_inner1D=test_inner1D):

                        if test_inner1D == 'inner_prod':

                            self.assertTrue(
                                np.allclose(utils.inner1d(X, Y, axis=ax),
                                            (X * Y).sum(axis=ax)))

                        if test_inner1D == 'sq_norm':

                            self.assertTrue(
                                np.allclose(utils.inner1d(X, axis=ax),
                                            (X**2).sum(axis=ax)))

    def test_det_ST(self):
        """Test determinant
            - det_ST(arr, S) = det(arr[S, S])
            - det_ST(arr, S, T) = det(arr[S, T])
        """

        shapes = [10, 50, 100, 300]
        nb_minors = 10

        for sh in shapes:
            with self.subTest(axis=sh):

                arr = rndm.rand(sh, sh)
                size_minors = sh // 3

                for idx in range(nb_minors):
                    with self.subTest(idx=idx):

                        S, T = rndm.choice(sh,
                                           size=(2, size_minors),
                                           replace=False)

                        for test_det_ST in ['SS', 'ST']:
                            with self.subTest(test_det_ST=test_det_ST):

                                if test_det_ST == 'SS':

                                    self.assertTrue(
                                        np.allclose(utils.det_ST(arr, S),
                                                    la.det(arr[np.ix_(S, S)])))

                                if test_det_ST == 'ST':

                                    self.assertTrue(
                                        np.allclose(utils.det_ST(arr, S, T),
                                                    la.det(arr[np.ix_(S, T)])))

    def test_symmetric(self):

        N = 20
        X = rndm.randn(N, N)

        list_of_inputs = [(True, None),
                          (False, X),
                          (True, X.T + X),
                          (True, X.T.dot(X))]

        for idx, (flag, _input) in enumerate(list_of_inputs):
            with self.subTest(index=idx, is_symmetric=flag):

                if flag:
                    self.assertTrue(
                        utils.is_symmetric(_input) is _input)
                else:
                    with self.assertRaises(ValueError) as context:
                        utils.is_symmetric(_input)

                    self.assertIn('M.T != M', str(context.exception))

    def test_is_projection(self):

        N, rank = 30, 10

        X = rndm.randn(N, rank)
        e_vecs, _ = qr(X, mode='economic')

        list_of_inputs = [(True, None),
                          (False, X.dot(X.T)),
                          (True, X.dot(la.inv(X.T.dot(X)).dot(X.T))),
                          (True, e_vecs.dot(e_vecs.T))]

        for idx, (flag, _input) in enumerate(list_of_inputs):
            with self.subTest(index=idx, is_projection=flag):

                if flag:
                    self.assertTrue(
                        utils.is_projection(_input) is _input)
                else:
                    with self.assertRaises(ValueError) as context:
                        utils.is_projection(_input)

                    self.assertIn('M^2 != M', str(context.exception))

    def test_is_orthonormal_columns(self):

        N, rank = 30, 10

        X = rndm.randn(N, rank)
        e_vecs, _ = qr(X, mode='economic')

        list_of_inputs = [(True, None),
                          (False, X),
                          (True, e_vecs)]

        for idx, (flag, _input) in enumerate(list_of_inputs):
            with self.subTest(index=idx, is_orthonormal_columns=flag):

                if flag:
                    self.assertTrue(
                        utils.is_orthonormal_columns(_input) is _input)
                else:
                    with self.assertRaises(ValueError) as context:
                        utils.is_orthonormal_columns(_input)

                    self.assertIn('M.T M != I', str(context.exception))

    def test_is_equal_to_O_or_1(self):

        N, tol = 100, 1e-8

        list_of_inputs = [(True, None),
                          (True, np.ones(N)),
                          (True, np.zeros(N)),
                          (True, -tol * np.ones(N)),
                          (True, tol + np.ones(N)),
                          (False, -2 * tol * np.ones(N)),
                          (False, np.ones(N) + 2 * tol),
                          (False, rndm.rand(N)),
                          (False, rndm.rand(N, N)),
                          (False, 1 - rndm.rand(N)),
                          (False, 1 + rndm.rand(N)),
                          (False, -tol + rndm.rand(N))]

        for idx, (flag, _input) in enumerate(list_of_inputs):
            with self.subTest(index=idx, is_equal_to_O_or_1=flag):

                if flag:
                    self.assertTrue(
                        utils.is_equal_to_O_or_1(_input, tol) is _input)
                else:
                    with self.assertRaises(ValueError) as context:
                        utils.is_equal_to_O_or_1(_input, tol)

                    self.assertIn('not all in {0,1}', str(context.exception))

    def test_is_in_01(self):

        N, tol = 100, 1e-8

        list_of_inputs = [(True, None),
                          (True, np.ones(N)),
                          (True, np.zeros(N)),
                          (True, -tol * np.ones(N)),
                          (True, tol + np.ones(N)),
                          (False, -2 * tol * np.ones(N)),
                          (False, np.ones(N) + 2 * tol),
                          (True, rndm.rand(N)),
                          (True, rndm.rand(N, N)),
                          (True, 1 - rndm.rand(N)),
                          (False, 1 + rndm.rand(N)),
                          (True, -tol + rndm.rand(N)),
                          (True, rndm.rand(N, N))]

        for idx, (flag, _input) in enumerate(list_of_inputs):
            with self.subTest(index=idx, is_in_01=flag):

                if flag:
                    self.assertTrue(utils.is_in_01(_input, tol) is _input)
                else:
                    with self.assertRaises(ValueError) as context:
                        utils.is_in_01(_input, tol)

                    self.assertIn('not all in [0,1]', str(context.exception))

    def test_is_geq_0(self):

        N, lam = 100, 4
        tol = 1e-8

        poisson = rndm.poisson(lam=lam, size=N)

        pm_poisson = poisson.copy()
        pm_poisson[:N // 2] *= -1

        list_of_inputs = [(True, None),
                          (True, np.zeros(N)),
                          (True, np.ones(N)),
                          (False, -np.ones(N)),
                          (True, - tol * np.ones(N)),
                          (False, - 2 * tol * np.ones(N)),
                          (True, poisson),
                          (False, pm_poisson)]

        for idx, (flag, _input) in enumerate(list_of_inputs):
            with self.subTest(index=idx, is_geq_0=flag):

                if flag:
                    self.assertTrue(utils.is_geq_0(_input, tol) is _input)
                else:
                    with self.assertRaises(ValueError) as context:
                        utils.is_geq_0(_input, tol)

                    self.assertIn('not all >= 0', str(context.exception))

    def test_is_full_row_rank(self):

        N, rank = 30, 10

        list_of_inputs = [(True, None),
                          (True, rndm.randn(rank, N)),
                          (False, rndm.randn(N + 1, N)),
                          (False, np.zeros((rank, N))),
                          (False, np.ones((rank, N)))]

        for idx, (flag, _input) in enumerate(list_of_inputs):
            with self.subTest(index=idx, is_full_row_rank=flag):

                if flag:
                    self.assertTrue(utils.is_full_row_rank(_input) is _input)
                else:
                    with self.assertRaises(ValueError) as context:
                        utils.is_full_row_rank(_input)

                    self.assertIn('not full row rank', str(context.exception))

    def test_evaluate_L_diagonal(self):
        """ checking np.diag(dpp.L) = utils.evaluate_L_diagonal(eval_L, X_data)
        """

        X = rndm.randn(100, 20)

        np.testing.assert_almost_equal(
          np.diag(utils.example_eval_L_linear(X)),
          utils.evaluate_L_diagonal(utils.example_eval_L_linear, X))

        X = rndm.rand(100, 1)

        np.testing.assert_almost_equal(
          np.diag(utils.example_eval_L_min_kern(X)),
          utils.evaluate_L_diagonal(utils.example_eval_L_min_kern, X))

        pass


def main():

    unittest.main()


if __name__ == '__main__':
    main()
