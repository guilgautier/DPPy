# coding: utf8
""" Unit tests:

- :class:`TestRSKCorrespondence`
"""


import unittest

import sys
sys.path.append('..')

from numpy import allclose
from numpy.random import rand
from dppy.utils import inner1d


class TestInner1d(unittest.TestCase):
    """ Test
    """

    X = rand(10, 20, 30, 40)
    Y = rand(*X.shape)

    def test_inner_products_and_square_norms(self):

        for ax in range(self.X.ndim):

            self.assertTrue(allclose(inner1d(self.X, self.Y, axis=ax),
                                     (self.X * self.Y).sum(axis=ax)))
            self.assertTrue(allclose(inner1d(self.X, axis=ax),
                                     (self.X**2).sum(axis=ax)))


def main():

    unittest.main()


if __name__ == '__main__':
    main()
