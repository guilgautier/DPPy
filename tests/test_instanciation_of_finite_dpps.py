# coding: utf8
""" Unit tests:

- :class:`InstanciationOfFiniteDppWithCorrelationKernel`
- :class:`InstanciationOfFiniteDppWithLikelihoodKernel`

to check that instanciation of FiniteDPP in the various settings works well
"""

import unittest

import numpy as np
import numpy.random as rndm

from scipy.linalg import qr, eigh

import sys
sys.path.append('..')

from dppy.finite_dpps import FiniteDPP


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

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=False,
                        **{'K_eig_dec': (eig_vals, eig_vecs)})

        dpp.compute_L()
        L = (eig_vecs * (eig_vals / (1.0 - eig_vals))).dot(eig_vecs.T)

        self.assertTrue(np.allclose(dpp.L, L))

    def test_instanciation_from_kernel(self):
        rank, N = 6, 10

        eig_vals = rndm.rand(rank)
        eig_vecs, _ = qr(rndm.randn(N, rank), mode="economic")

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=False,
                        **{'K': (eig_vecs * eig_vals).dot(eig_vecs.T)})

        dpp.compute_L()
        L = (eig_vecs * (eig_vals / (1.0 - eig_vals))).dot(eig_vecs.T)

        self.assertTrue(np.allclose(dpp.L, L))


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

        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=False,
                        **{'L_eig_dec': (eig_vals, eig_vecs)})

        dpp.compute_K()
        K = (eig_vecs * (eig_vals / (1.0 + eig_vals))).dot(eig_vecs.T)

        self.assertTrue(np.allclose(dpp.K, K))

    def test_instanciation_from_kernel(self):
        rank, N = 6, 10

        eig_vals = 1 + rndm.geometric(p=0.5, size=rank)
        eig_vecs, _ = qr(rndm.randn(N, rank), mode="economic")

        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=False,
                        **{'L': (eig_vecs * eig_vals).dot(eig_vecs.T)})

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

    def test_instanciation_from_L_gram_factor_wide(self):
        rank, N = 6, 10

        phi = rndm.randn(rank, N)

        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=False,
                        **{'L_gram_factor': phi})

        dpp.compute_L()
        L = phi.T.dot(phi)

        self.assertTrue(np.allclose(dpp.L, L))

        dpp = FiniteDPP(kernel_type='likelihood',
                projection=False,
                **{'L_gram_factor': phi})

        dpp.compute_K()
        eig_vals, eig_vecs = eigh(L)
        K = (eig_vecs * (eig_vals / (1.0 + eig_vals))).dot(eig_vecs.T)

        self.assertTrue(np.allclose(dpp.K, K))

    def test_instanciation_from_L_gram_factor_tall(self):
        rank, N = 6, 10

        phi = rndm.randn(rank, N).T

        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=False,
                        **{'L_gram_factor': phi})

        dpp.compute_L()
        L = phi.T.dot(phi)

        self.assertTrue(np.allclose(dpp.L, L))

        dpp = FiniteDPP(kernel_type='likelihood',
                projection=False,
                **{'L_gram_factor': phi})

        dpp.compute_K()
        eig_vals, eig_vecs = eigh(L)
        K = (eig_vecs * (eig_vals / (1.0 + eig_vals))).dot(eig_vecs.T)

        self.assertTrue(np.allclose(dpp.K, K))


def main():

    unittest.main()


if __name__ == '__main__':
    main()
