#!/usr/bin/env python3
# coding=utf-8

import unittest

import numpy as np
from numpy.random import rand, randn
from scipy.linalg import qr

from dppy.finite_dpps import FiniteDPP

from dppy.utils import example_eval_L_linear


class ExactSamplerOutputSomething(unittest.TestCase):

    r, N = 6, 10

    def test_schur_sampler(self):
        eig_vecs, _ = qr(randn(self.N, self.r), mode="economic")
        # eig_vals = rand(r)  # 0< <1
        eig_vals = np.ones(self.r)

        dpp = FiniteDPP("correlation", True,
                        K_eig_dec=(eig_vals, eig_vecs))

        sample = dpp.sample_exact(mode="Schur")
        self.assertTrue(isinstance(sample, list), sample)

    def test_chol_sampler(self):
        eig_vecs, _ = qr(randn(self.N, self.r), mode="economic")
        # eig_vals = rand(r)  # 0< <1
        eig_vals = np.ones(self.r)

        dpp = FiniteDPP("correlation", True,
                        K_eig_dec=(eig_vals, eig_vecs))

        sample = dpp.sample_exact(mode="Chol")
        self.assertTrue(isinstance(sample, list), sample)

    def test_vfx_sampler(self):
        X = randn(self.N, self.r)
        dpp = FiniteDPP("likelihood", False,
                        L_eval_X_data=(example_eval_L_linear, X))

        sample = dpp.sample_exact(mode="vfx")
        self.assertTrue(isinstance(sample, list), sample)

    def test_alpha_sampler(self):
        X = randn(self.N, self.r)

        dpp = FiniteDPP("likelihood", False,
                        L_eval_X_data=(example_eval_L_linear, X))

        sample = dpp.sample_exact(mode="alpha")
        self.assertTrue(isinstance(sample, list), sample)

    def test_spectral_sampler(self):
        eig_vecs, _ = qr(randn(self.N, self.r), mode="economic")
        # eig_vals = rand(self.r)  # 0< <1
        eig_vals = np.ones(self.r)

        dpp = FiniteDPP("correlation", True, K_eig_dec=(eig_vals, eig_vecs))

        sample = dpp.sample_exact(mode="GS")
        self.assertTrue(isinstance(sample, list), sample)


def main():

    unittest.main()
