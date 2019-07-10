# coding: utf8
""" Tests:

- :class:`MarginalProbaDescentProcesses` check that the marginal probability of selecting any integer is indeed given by the ``_bernoulli_param`` attribute
"""

import unittest

import numpy as np
import numpy.random as rndm

from scipy.linalg import qr, eigh

import sys
sys.path.append('..')

from dppy.exotic_dpps import CarriesProcess, DescentProcess, VirtualDescentProcess


class MarginalProbaDescentProcesses(unittest.TestCase):
    """ Check that the marginal probability of selecting any integer is indeed given by the ``_bernoulli_param`` attribute
    """

    size = 10000
    tol = 1e-2

    def test_carries_process(self):

        cp = CarriesProcess(base=10)
        cp.sample(size=self.size)

        estim = len(cp.list_of_samples[-1]) / self.size

        self.assertTrue(np.abs(estim - cp._bernoulli_param) < self.tol)

    def test_carries_process(self):

        dp = DescentProcess()
        dp.sample(size=self.size)

        estim = len(dp.list_of_samples[-1]) / self.size

        self.assertTrue(np.abs(estim - dp._bernoulli_param) < self.tol)

    def test_carries_process(self):

        vdp = VirtualDescentProcess(x_0=0.5)
        vdp.sample(size=self.size)

        estim = len(vdp.list_of_samples[-1]) / self.size

        self.assertTrue(np.abs(estim - vdp._bernoulli_param) < self.tol)


def main():

    unittest.main()


if __name__ == '__main__':
    main()
