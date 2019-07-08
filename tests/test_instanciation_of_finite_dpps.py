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
    """ Test the instanciation of :py:class:`~dppy.finite_dpps.FiniteDPP` defined through its correlation kernel :math:`K`, which must satisfy :math:`0 \\preceq K \\preceq I`
    """

    def test_instanciation_from_eig_vals_equal_01(self):

        rank, N = 6, 10

        eig_vals = np.ones(rank)
        eig_vecs, _ = qr(rndm.randn(N, rank), mode="economic")

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=True,
                        **{'K_eig_dec': (eig_vals, eig_vecs)})

        dpp.compute_K()
        K = (eig_vecs * eig_vals).dot(eig_vecs.T)

        self.assertTrue(np.allclose(dpp.K, K))

        with self.assertRaises(ValueError) as context:
            dpp.compute_L()

        self.assertTrue('cannot be computed' in str(context.exception))

    def test_instanciation_from_eig_vals_in_01(self):
        rank, N = 6, 10

        eig_vals = rndm.rand(rank)
        eig_vecs, _ = qr(rndm.randn(N, rank), mode="economic")

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=False,
                        **{'K_eig_dec': (eig_vals, eig_vecs)})

        dpp.compute_K()
        K = (eig_vecs * eig_vals).dot(eig_vecs.T)

        self.assertTrue(np.allclose(dpp.K, K))

        dpp.compute_L()
        L = (eig_vecs * (eig_vals / (1.0 - eig_vals))).dot(eig_vecs.T)

        self.assertTrue(np.allclose(dpp.L, L))

    def test_instanciation_from_A_zono(self):
        rank, N = 6, 10

        A = rndm.randn(rank, N)

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=True,
                        **{'A_zono': A})

        dpp.compute_K()
        K = A.T.dot(np.linalg.inv(A.dot(A.T))).dot(A)

        self.assertTrue(np.allclose(dpp.K, K))

        with self.assertRaises(ValueError) as context:

            FiniteDPP(kernel_type='correlation',
                      projection=True,
                      **{'A_zono': rndm.randn(N, rank)})

        self.assertTrue('not full row rank' in str(context.exception))


class InstanciationOfFiniteDppWithLikelihoodKernel(unittest.TestCase):
    """ Test the instanciation of :py:class:`~dppy.finite_dpps.FiniteDPP` defined through its likelyhood kernel :math:`L`, which must satisfy :math:`L \\succeq 0`
    """

    def test_instanciation_from_eig_vals_geq_0(self):
        rank, N = 6, 10

        eig_vals = 1 + rndm.geometric(p=0.5, size=rank)
        eig_vecs, _ = qr(rndm.randn(N, rank), mode="economic")

        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=False,
                        **{'L_eig_dec': (eig_vals, eig_vecs)})

        dpp.compute_L()
        L = (eig_vecs * eig_vals).dot(eig_vecs.T)

        self.assertTrue(np.allclose(dpp.L, L))

        dpp.compute_K()
        K = (eig_vecs * (eig_vals / (1.0 + eig_vals))).dot(eig_vecs.T)

        self.assertTrue(np.allclose(dpp.K, K))

    def test_instanciation_from_eig_vals_equal_01(self):
        rank, N = 6, 10

        eig_vals = np.ones(rank)
        eig_vecs, _ = qr(rndm.randn(N, rank), mode="economic")

        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=True,
                        **{'L_eig_dec': (eig_vals, eig_vecs)})

        dpp.compute_L()
        L = (eig_vecs * eig_vals).dot(eig_vecs.T)

        self.assertTrue(np.allclose(dpp.L, L))

    def test_instanciation_from_L_gram_factor(self):
        rank, N = 6, 10

        phi = rndm.randn(rank, N)

        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=True,
                        **{'L_gram_factor': phi})

        dpp.compute_L()
        L = phi.T.dot(phi)

        self.assertTrue(np.allclose(dpp.L, L))


def main():

    unittest.main()


if __name__ == '__main__':
    main()
