# coding: utf8
""" Unit tests:

- :class:`TestRSKCorrespondence`
"""


import unittest

import sys
sys.path.append('..')

from numpy import allclose
from numpy.random import rand
from dppy.utils import inner1d, det_ST


class TestUtils(unittest.TestCase):
    """ Test
    """

    def test_inner_products_and_square_norms(self):

        X = rand(10, 20, 30, 40)
        Y = rand(*X.shape)

        for ax in range(X.ndim):

            self.assertTrue(allclose(inner1d(X, Y, axis=ax),
                                     (X * Y).sum(axis=ax)))
            self.assertTrue(allclose(inner1d(X, axis=ax),
                                     (X**2).sum(axis=ax)))

    def test_det_ST(self):
        """ Compute :math:`\det M_{S, T} = \det [M_{ij}]_{i\inS, j\in T}`
        """


        if T is None:  # det M_SS = det M_S
            return det(array[np.ix_(S, S)])

        else:  # det M_ST, numpy deals with det M_[][] = 1.0
            return det(array[np.ix_(S, T)])

def main():

    unittest.main()


if __name__ == '__main__':
    main()
