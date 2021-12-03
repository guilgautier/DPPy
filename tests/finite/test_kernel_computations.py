import unittest

import numpy as np
import numpy.random as rndm
import scipy.linalg as la

from dppy.finite.dpp import FiniteDPP
from dppy.utils import example_eval_L_linear, example_eval_L_min_kern


class TestFiniteDppInstantiationFromCorrelationKernel(unittest.TestCase):
    r"""Tests on :py:class:`~dppy.finite_dpps.FiniteDPP` defined through different parametrizations its correlation kernel :math:`\mathbf{K}`."""

    kernel_type = "correlation"

    rank, N = 6, 10
    nb_samples = 1000

    lambd_eq_01 = np.ones(rank)
    lambd_in_01 = np.random.rand(rank)

    A_zono = rndm.randn(rank, N)
    U, _ = la.qr(A_zono.T, mode="economic")

    K_proj_hermitian = U.dot(U.T)
    K_non_proj_hermitian = (U * lambd_in_01).dot(U.T)
    L_non_proj_hermitian = (U * (lambd_in_01 / (1.0 - lambd_in_01))).dot(U.T)

    # projection, param
    params_expected_kernels = [
        {
            "params": {"K": K_proj_hermitian, "projection": True, "hermitian": True},
            "expected_K": K_proj_hermitian,
            "expected_L": None,
        },
        {
            "params": {"K": K_proj_hermitian, "projection": False, "hermitian": True},
            "expected_K": K_proj_hermitian,
            "expected_L": None,
        },
        {
            "params": {
                "K": K_non_proj_hermitian,
                "projection": False,
                "hermitian": True,
            },
            "expected_K": K_non_proj_hermitian,
            "expected_L": L_non_proj_hermitian,
        },
        {
            "params": {
                "K_eig_dec": (lambd_eq_01, U),
                "projection": True,
                "hermitian": True,
            },
            "expected_K": K_proj_hermitian,
            "expected_L": None,
        },
        {
            "params": {
                "K_eig_dec": (lambd_eq_01, U),
                "projection": False,
                "hermitian": True,
            },
            "expected_K": K_proj_hermitian,
            "expected_L": None,
        },
        {
            "params": {
                "K_eig_dec": (lambd_in_01, U),
                "projection": False,
                "hermitian": True,
            },
            "expected_K": K_non_proj_hermitian,
            "expected_L": L_non_proj_hermitian,
        },
        {
            "params": {"A_zono": A_zono, "projection": True, "hermitian": True},
            "expected_K": K_proj_hermitian,
            "expected_L": None,
        },
        {
            "params": {"A_zono": U.T, "projection": True, "hermitian": True},
            "expected_K": K_proj_hermitian,
            "expected_L": None,
        },
    ]

    def test_compute_correlation_kernel_from_correlation_parametrization(self):
        for idx, _dict in enumerate(self.params_expected_kernels):
            params, K, L = _dict.values()
            with self.subTest(index=idx, params=params):
                dpp = FiniteDPP(self.kernel_type, **params)
                dpp.compute_K()
                self.assertTrue(np.allclose(dpp.K, K))

    def test_compute_likehood_kernel_from_correlation_parametrization(self):
        for idx, _dict in enumerate(self.params_expected_kernels):
            params, K, L = _dict.values()
            with self.subTest(index=idx, params=params):
                dpp = FiniteDPP(self.kernel_type, **params)
                try:
                    dpp.compute_L()
                    self.assertTrue(np.allclose(dpp.L, L))
                except (ValueError, FloatingPointError) as e:
                    self.assertTrue(True)


class TestFiniteDppInstantiationFromLikelihoodKernel(unittest.TestCase):
    r"""Tests on :py:class:`~dppy.finite_dpps.FiniteDPP` defined through different parametrizations of its likelihood kernel :math:`\mathbf{L}`"""

    kernel_type = "likelihood"
    rank, N = 6, 10

    # From gram factor L = Phi.T Phi
    L_to_K_generic = lambda L: L.dot(np.linalg.inv(L + np.eye(*L.shape)))

    Phi = rndm.randn(rank, N)
    L_phi = Phi.T.dot(Phi)
    K_phi = L_to_K_generic(L_phi)

    # From eigen-decomposition L = V gamma V.T
    L_to_K_eigs = lambda x, V: (V * (x / (1.0 + x))).dot(V.T)

    gamma_eq_01 = np.ones(rank)
    gamma_geq_0 = 1 + rndm.geometric(p=0.5, size=rank)
    V, _ = la.qr(Phi.T, mode="economic")

    L_proj_hermitian = V.dot(V.T)
    K_proj_hermitian = L_to_K_eigs(gamma_eq_01, V)

    L_non_proj_hermitian = (V * gamma_geq_0).dot(V.T)
    K_non_proj_hermitian = L_to_K_eigs(gamma_geq_0, V)

    # From kernel function L(x, y) and data
    X_data_randn = rndm.rand(N, rank)
    L_linear_X_randn = example_eval_L_linear(X_data_randn)
    K_linear_X_randn = L_to_K_generic(L_linear_X_randn)

    X_data_unif01 = rndm.rand(N, 1)
    L_min_X_unif01 = example_eval_L_min_kern(X_data_unif01)
    K_min_X_unif01 = L_to_K_generic(L_min_X_unif01)

    params_expected_kernels = [
        {
            "params": {"L": L_proj_hermitian, "projection": True, "hermitian": True},
            "expected_K": K_proj_hermitian,
            "expected_L": L_proj_hermitian,
        },
        {
            "params": {"L": L_proj_hermitian, "projection": False, "hermitian": True},
            "expected_K": K_proj_hermitian,
            "expected_L": L_proj_hermitian,
        },
        {
            "params": {
                "L": L_non_proj_hermitian,
                "projection": False,
                "hermitian": True,
            },
            "expected_K": K_non_proj_hermitian,
            "expected_L": L_non_proj_hermitian,
        },
        {
            "params": {
                "L_eig_dec": (gamma_eq_01, V),
                "projection": True,
                "hermitian": True,
            },
            "expected_K": K_proj_hermitian,
            "expected_L": L_proj_hermitian,
        },
        {
            "params": {
                "L_eig_dec": (gamma_eq_01, V),
                "projection": False,
                "hermitian": True,
            },
            "expected_K": K_proj_hermitian,
            "expected_L": L_proj_hermitian,
        },
        {
            "params": {
                "L_eig_dec": (gamma_geq_0, V),
                "projection": False,
                "hermitian": True,
            },
            "expected_K": K_non_proj_hermitian,
            "expected_L": L_non_proj_hermitian,
        },
        {
            "params": {"L_gram_factor": Phi, "projection": False, "hermitian": True},
            "expected_K": K_phi,
            "expected_L": L_phi,
        },
        {
            "params": {
                "L_eval_X_data": (example_eval_L_linear, X_data_randn),
                "projection": False,
                "hermitian": True,
            },
            "expected_K": K_linear_X_randn,
            "expected_L": L_linear_X_randn,
        },
        {
            "params": {
                "L_eval_X_data": (example_eval_L_min_kern, X_data_unif01),
                "projection": False,
                "hermitian": True,
            },
            "expected_K": K_min_X_unif01,
            "expected_L": L_min_X_unif01,
        },
    ]

    def test_compute_correlation_kernel_from_correlation_parametrization(self):
        for idx, _dict in enumerate(self.params_expected_kernels):
            params, K, L = _dict.values()
            with self.subTest(index=idx, params=params):
                dpp = FiniteDPP(self.kernel_type, **params)
                dpp.compute_K()
                self.assertTrue(np.allclose(dpp.K, K))

    def test_compute_likehood_kernel_from_correlation_parametrization(self):
        for idx, _dict in enumerate(self.params_expected_kernels):
            params, K, L = _dict.values()
            with self.subTest(index=idx, params=params):
                dpp = FiniteDPP(self.kernel_type, **params)
                try:
                    dpp.compute_L()
                    self.assertTrue(np.allclose(dpp.L, L))
                except (ValueError, FloatingPointError) as e:
                    self.assertTrue(True)
