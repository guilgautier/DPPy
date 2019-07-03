# coding: utf8
""" Unit tests:

- :class:`InclusionProbabilitiesProjectionDPP` to check that exact samplers for finite DPPs have the right (at least 1 and 2) inclusion probabilities
"""

import unittest

import numpy as np
import numpy.random as rndm

from scipy.linalg import qr
from scipy.stats import chisquare

from itertools import chain  # to flatten list of samples

import sys
sys.path.append('..')


from dppy.finite_dpps import FiniteDPP
from dppy.utils import det_ST


class InstanciationOfFiniteDppWithCorrelationKernel(unittest.TestCase):
    """
    """

    def test_instanciation_from_eig_vals_in_01(self):
        rank, N = 6, 10

        eig_vals = rndm.rand(rank)
        eig_vecs, _ = qr(rndm.randn(N, rank), mode="economic")

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=False,
                        **{'K_eig_dec':(eig_vals, eig_vecs)})

        dpp.compute_K()
        K = (eig_vecs*eig_vals).dot(eig_vecs.T)

        self.assertTrue(np.allclose(dpp.K, K))

        dpp.compute_L()
        L = (eig_vecs*(eig_vals/(1.0-eig_vals))).dot(eig_vecs.T)

        self.assertTrue(np.allclose(dpp.L, L))

    def test_instanciation_from_eig_vals_equal_01(self):
        rank, N = 6, 10

        eig_vals = np.ones(rank)
        eig_vecs, _ = qr(rndm.randn(N, rank), mode="economic")

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=True,
                        **{'K_eig_dec':(eig_vals, eig_vecs)})

        dpp.compute_K()
        K = (eig_vecs*eig_vals).dot(eig_vecs.T)

        self.assertTrue(np.allclose(dpp.K, K))

        with self.assertRaises(ValueError) as context:
            dpp.compute_L()

        self.assertTrue('cannot be computed' in str(context.exception))



def main():

    unittest.main()


if __name__ == '__main__':
    main()
