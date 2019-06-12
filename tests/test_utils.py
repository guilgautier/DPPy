# coding: utf8
""" Unit tests:

- :class:`TestRSKCorrespondence`
"""


import unittest

from numpy import allclose, ix_
from numpy.linalg import det
from numpy.random import rand, choice

import sys
sys.path.append('..')
from dppy.utils import inner1d, det_ST


class TestUtils(unittest.TestCase):
    """ Test
    """

    def test_inner_products_and_square_norms(self):

        X = rand(10, 20, 30, 40)
        Y = rand(*X.shape)

        for ax in range(X.ndim):

            # inner product
            self.assertTrue(allclose(inner1d(X, Y, axis=ax),
                                     (X * Y).sum(axis=ax)))
            # square norm
            self.assertTrue(allclose(inner1d(X, axis=ax),
                                     (X**2).sum(axis=ax)))

    def test_det_ST(self):
        """Test determinant
            - det_ST(arr, S) = det(arr[S, S])
            - det_ST(arr, S, T) = det(arr[S, T])
        """

        shapes = [10, 50, 100, 300]
        nb_minors = 10

        for sh in shapes:

            arr = rand(sh, sh)
            size_minors = sh // 3

            for _ in range(nb_minors):
                S, T = choice(sh, size=(2, size_minors))

                self.assertTrue(allclose(det_ST(arr, S),
                                         det(arr[ix_(S, S)])))

                self.assertTrue(allclose(det_ST(arr, S, T),
                                         det(arr[ix_(S, T)])))

def main():

    unittest.main()


if __name__ == '__main__':
    main()
