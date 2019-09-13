# coding: utf8
""" Tests:

- :class:`TestDescentProcesses` check that the marginal probability of selecting any integer is indeed given by the ``_bernoulli_param`` attribute
"""

import unittest

import numpy as np

import sys
sys.path.append('..')

from dppy.exotic_dpps import CarriesProcess, DescentProcess, VirtualDescentProcess


class TestDescentProcesses(unittest.TestCase):
    """ Check that the marginal probability of selecting any integer is indeed given by the ``_bernoulli_param`` attribute
    """

    size = 10000
    tol = 1e-2

    def marginal_adequation(self, process):

        process.sample(size=self.size)

        estim = len(process.list_of_samples[-1]) / self.size

        self.assertTrue(np.abs(estim - process._bernoulli_param) < self.tol)

    def test_carries_process(self):
        process = CarriesProcess(base=10)
        self.marginal_adequation(process)

    def test_descent_process(self):
        process = DescentProcess()
        self.marginal_adequation(process)

    def test_virtual_descent_process(self):
        process = VirtualDescentProcess(x_0=0.5)
        self.marginal_adequation(process)


def main():

    unittest.main()


if __name__ == '__main__':
    main()
