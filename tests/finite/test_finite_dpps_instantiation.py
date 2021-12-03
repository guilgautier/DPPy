import unittest

import numpy as np
import numpy.random as rndm
import scipy.linalg as la

from dppy.finite.dpp import FiniteDPP
from dppy.utils import example_eval_L_linear, example_eval_L_min_kern


class TestFiniteDppInstantiationFromCorrelationKernel(unittest.TestCase):
    r"""Tests on :py:class:`~dppy.finite_dpps.FiniteDPP` defined through different parametrizations its correlation kernel :math:`\mathbf{K}`."""

    rank, N = 6, 10
    nb_samples = 1000

    lambd_eq_01 = np.ones(rank)
    lambd_in_01 = np.random.rand(rank)

    U, _ = la.qr(rndm.randn(N, rank), mode="economic")

    K_proj_hermitian = (U * lambd_eq_01).dot(U.T)
    K_non_proj_hermitian = (U * lambd_in_01).dot(U.T)

    A_zono = rndm.randn(rank, N)

    # projection, param
    list_of_valid_params = [
        {"K": K_proj_hermitian, "projection": True, "hermitian": True},
        {"K": K_proj_hermitian, "projection": False, "hermitian": True},
        {"K": K_non_proj_hermitian, "projection": False, "hermitian": True},
        {"K_eig_dec": (lambd_eq_01, U), "projection": True, "hermitian": True},
        {"K_eig_dec": (lambd_eq_01, U), "projection": False, "hermitian": True},
        {"K_eig_dec": (lambd_in_01, U), "projection": False, "hermitian": True},
        {"A_zono": A_zono, "projection": True, "hermitian": True},
    ]

    def test_smoke_instantiation_from_valid_parameters(self):
        for idx, params in enumerate(self.list_of_valid_params):
            k_type = "correlation"
            with self.subTest(index=idx, kernel_type=k_type, params=params):
                FiniteDPP(k_type, **params)
                self.assertTrue(True)

    def test_instantiation_from_invalid_parameter_key(self):
        k_type = "correlation"
        for idx, key in enumerate(["L", "L_eig_dec", "L_gram_factor", "random_key"]):
            params = {key: 0}
            with self.subTest(index=idx, kernel_type=k_type, params=params):
                with self.assertRaises(ValueError):
                    FiniteDPP(k_type, **params)


class TestFiniteDppInstantiationFromLikelihoodKernel(unittest.TestCase):
    r"""Tests on :py:class:`~dppy.finite_dpps.FiniteDPP` defined through different parametrizations of its likelihood kernel :math:`\mathbf{L}`"""

    rank, N = 6, 10

    gamma_eq_01 = np.ones(rank)
    gamma_geq_0 = 1 + rndm.geometric(p=0.5, size=rank)

    V, _ = la.qr(rndm.randn(N, rank), mode="economic")

    Phi = rndm.randn(rank, N)

    L_proj_hermitian = (V * gamma_eq_01).dot(V.T)
    L_non_proj_hermitian = (V * gamma_geq_0).dot(V.T)

    X_data_randn = rndm.rand(N, rank)
    X_data_in_01 = rndm.rand(N, 1)

    list_of_valid_params = [
        {"L": L_proj_hermitian, "projection": True, "hermitian": True},
        {"L": L_proj_hermitian, "projection": False, "hermitian": True},
        {"L": L_non_proj_hermitian, "projection": False, "hermitian": True},
        {"L": L_proj_hermitian, "projection": True, "hermitian": True},
        {"L_eig_dec": (gamma_eq_01, V), "projection": False, "hermitian": True},
        {"L_eig_dec": (gamma_geq_0, V), "projection": False, "hermitian": True},
        {"L_eig_dec": (gamma_eq_01, V), "projection": True, "hermitian": True},
        {"L_gram_factor": Phi, "projection": False, "hermitian": True},
        {"L_gram_factor": Phi.T, "projection": False, "hermitian": True},
        {
            "L_eval_X_data": (example_eval_L_linear, X_data_randn),
            "projection": False,
            "hermitian": True,
        },
        {
            "L_eval_X_data": (example_eval_L_min_kern, X_data_in_01),
            "projection": False,
            "hermitian": True,
        },
    ]

    def test_smoke_instantiation_from_valid_parameters(self):
        for idx, params in enumerate(self.list_of_valid_params):
            k_type = "likelihood"
            with self.subTest(index=idx, kernel_type=k_type, params=params):
                FiniteDPP(k_type, **params)
                self.assertTrue(True)

    def test_instantiation_from_invalid_parameter_key(self):
        k_type = "likelihood"
        for idx, key in enumerate(["K", "K_eig_dec", "A_zono", "random_key"]):
            params = {key: 0}
            with self.subTest(index=idx, kernel_type=k_type, params=params):
                with self.assertRaises(ValueError):
                    FiniteDPP(k_type, **params)
