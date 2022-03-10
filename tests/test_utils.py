# coding: utf8
""" Unit tests:

- :class:`TestUtils`
"""

import unittest

import numpy as np
import numpy.linalg as la
import numpy.random as rndm

from dppy.utils import det_ST, eval_kernel_diagonal, kernel_linear, kernel_minimum


class TestUtils(unittest.TestCase):
    """Test"""

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

                        S, T = rndm.choice(sh, size=(2, size_minors), replace=False)

                        for test_det_ST in ["SS", "ST"]:
                            with self.subTest(test_det_ST=test_det_ST):

                                if test_det_ST == "SS":

                                    self.assertTrue(
                                        np.allclose(
                                            det_ST(arr, S),
                                            la.det(arr[np.ix_(S, S)]),
                                        )
                                    )

                                if test_det_ST == "ST":

                                    self.assertTrue(
                                        np.allclose(
                                            det_ST(arr, S, T),
                                            la.det(arr[np.ix_(S, T)]),
                                        )
                                    )

    def test_evaluate_L_diagonal(self):
        """checking np.diag(dpp.L) = evaluate_L_diagonal(eval_L, X_data)"""

        X = rndm.randn(100, 20)

        np.testing.assert_almost_equal(
            np.diag(kernel_linear(X)),
            eval_kernel_diagonal(kernel_linear, X),
        )

        X = rndm.rand(100, 1)

        np.testing.assert_almost_equal(
            np.diag(kernel_minimum(X)),
            eval_kernel_diagonal(kernel_minimum, X),
        )


def main():

    unittest.main()


if __name__ == "__main__":
    main()
