#!/usr/bin/env python3
# coding=utf-8

import unittest

import numpy as np
from numpy.random import rand, randn
from scipy.linalg import qr

from dppy.finite_dpps import FiniteDPP


class TestSpectralSampler(unittest.TestCase):

    r, N = 6, 10

    def test_likelihood_eig_vals_to_correlation_eig_vals(self):
        eig_vecs, _ = qr(randn(self.N, self.r), mode="economic")
        eig_vals = np.arange(1, self.r+1)

        dpp = FiniteDPP("likelihood", False,
                        L_eig_dec=(eig_vals, eig_vecs))

        dpp.sample_exact(mode="GS")

        self.assertTrue(np.allclose(
            dpp.K_eig_vals, eig_vals / (1.0 + eig_vals)))

    def test_correlation_eig_vals_not_in_01_from_K_raises_error(self):
        eig_vecs, _ = qr(randn(self.N, self.r), mode="economic")
        eig_vals = np.arange(1, self.r+1)

        K = (eig_vecs * eig_vals).dot(eig_vecs.T)

        dpp = FiniteDPP("correlation", False, K=K)

        with self.assertRaises(ValueError):
            dpp.sample_exact(mode="GS")

    def test_correlation_eig_vals_in_01_from_K(self):
        eig_vecs, _ = qr(randn(self.N, self.N), mode="economic")
        eig_vals = np.sort(rand(self.N))

        K = (eig_vecs * eig_vals).dot(eig_vecs.T)

        dpp = FiniteDPP("correlation", False, K=K)

        dpp.sample_exact(mode="GS")

        self.assertTrue(np.allclose(dpp.K_eig_vals, eig_vals))

    def test_likelihood_eig_vals_initialization_is_valid(self):
        eig_vecs, _ = qr(randn(self.N, self.r), mode="economic")
        eig_vals = np.arange(1, self.r+1)

        dpp = FiniteDPP("likelihood", False,
                        L_eig_dec=(eig_vals, eig_vecs))

        self.assertTrue(np.allclose(dpp.L_eig_vals, eig_vals))

    def test_likelihood_eig_vals_not_geq_0_from_L_raises_error(self):
        eig_vecs, _ = qr(randn(self.N, self.r), mode="economic")
        eig_vals = np.arange(1, self.r+1)
        eig_vals[0] = -5.0

        L = (eig_vecs * eig_vals).dot(eig_vecs.T)

        dpp = FiniteDPP("likelihood", False, L=L)

        with self.assertRaises(ValueError):
            dpp.sample_exact(mode="GS")

    def test_likelihood_eig_vals_geq_0_from_L(self):
        eig_vecs, _ = qr(randn(self.N, self.N), mode="economic")
        eig_vals = 10.0 * np.sort(rand(self.N))

        L = (eig_vecs * eig_vals).dot(eig_vecs.T)

        dpp = FiniteDPP("likelihood", False, L=L)

        dpp.sample_exact(mode="GS")

        self.assertTrue(np.allclose(dpp.L_eig_vals, eig_vals))

    def test_correlation_eig_vals_from_A_zono_equal_1(self):
        A_zono = randn(self.r, self.N)

        dpp = FiniteDPP("correlation", True, A_zono=A_zono)

        dpp.sample_exact(mode="GS")

        self.assertTrue(np.allclose(dpp.K_eig_vals, np.ones(self.r)))

    # def test_likelihood_dual_eig_vals_to_likelihood_eig_vals(self):
    #     phi = randn(self.r, self.N)
    #     L_dual = phi.dot(phi.T)

    #     dpp = FiniteDPP("likelihood", False,
    #                     L_dual=L_dual)

    #     dpp.sample_exact(mode="GS")

    #     self.assertTrue(np.allclose(
    #         dpp.K_eig_vals, eig_vals / (1.0 + eig_vals)))

    # def test_likelihood_dual_eig_vals_to_likelihood_eig_vals(self):
    #     eig_vecs, _ = qr(randn(self.N, self.r), mode="economic")
    #     eig_vals = np.arange(1, self.r+1)

    #     dpp = FiniteDPP("likelihood", False,
    #                     L_eig_dec=(eig_vals, eig_vecs))

    #     dpp.sample_exact(mode="GS")

    #     self.assertTrue(np.allclose(
    #         dpp.K_eig_vals, eig_vals / (1.0 + eig_vals)))


def main():

    unittest.main()
