# coding: utf8
""" Unit tests:

- :class:`TestFiniteDppSamplers`
"""

import unittest
from itertools import chain  # to flatten list of samples

import numpy as np
import numpy.random as rndm
import scipy.linalg as la
from scipy.stats import chisquare

from dppy.finite.dpp import FiniteDPP
from dppy.finite.utils import loglikelihood
from dppy.utils import det_ST, kernel_linear, kernel_minimum


class Configuration(object):
    def __init__(self, dpp_params, sampler_type, method_params):
        self.sampler_type = sampler_type
        self.method_params = method_params
        self.dpp_params = dpp_params

    def create_dpp(self):
        return FiniteDPP(**self.dpp_params)

    def get_samples(self, dpp, nb_exact_samples):
        if self.sampler_type.startswith("exact"):
            samples = []
            for _ in range(nb_exact_samples):
                if self.sampler_type == "exact_dpp":
                    X = dpp.sample_exact(**self.method_params)
                elif self.sampler_type == "exact_k_dpp":
                    X = dpp.sample_exact_k_dpp(**self.method_params)
                else:
                    raise ValueError(self.sampler_type)
                samples.append(X)
            return samples

        elif self.sampler_type.startswith("mcmc"):
            if self.sampler_type == "mcmc_dpp":
                dpp.sample_mcmc(**self.method_params)
            elif self.sampler_type == "mcmc_k_dpp":
                dpp.sample_mcmc_k_dpp(**self.method_params)
            else:
                raise ValueError(self.sampler_type)
            return dpp.list_of_samples[-1]


def singleton_adequation(dpp, samples, tol=0.05, **kwargs):
    """Perform chi-square test to check that
    P[{i} C X] = K_ii
    """

    if dpp.size_k_dpp:
        adeq = True
        msg = "We do not check inclusion probabilities for k-DPPs"
    else:
        dpp.compute_correlation_kernel()
        N = len(dpp.K)

        f_th = np.diag(dpp.K) / np.trace(dpp.K)

        samples_as_singletons = list(chain.from_iterable(samples))
        f_emp, _ = np.histogram(samples_as_singletons, bins=range(N + 1), density=True)

        _, pval = chisquare(f_obs=f_emp, f_exp=f_th)

        adeq = pval > tol
        msg = "pval = {}, emp = {}, th = {}".format(pval, f_emp, f_th)

    return adeq, msg


def doubleton_adequation(dpp, samples, tol=0.05, **kwargs):
    """Perform chi-square test to check that
    P[{i, j} C X] = det [[K_ii, K_ij], [K_ji, K_jj]]
    """
    if dpp.size_k_dpp:
        return True, "We do not check inclusion probabilities for k-DPPs"
    else:
        samples = list(map(set, samples))
        dpp.compute_correlation_kernel()
        N = len(dpp.K)

        nb_doubletons = 10
        doubletons = [
            set(rndm.choice(N, size=2, replace=False)) for _ in range(nb_doubletons)
        ]

        # det [[K_ii, K_ij], [K_ji, K_jj]]
        f_th = np.array([det_ST(dpp.K, list(d)) for d in doubletons])

        counts = np.array(
            [sum(doubl.issubset(sampl) for sampl in samples) for doubl in doubletons]
        )
        f_emp = counts / len(samples)

        f_emp /= f_emp.sum()
        f_th /= f_th.sum()
        _, pval = chisquare(f_obs=f_emp, f_exp=f_th)

        adeq = pval > tol
        msg = "pval = {}, emp = {}, th = {}".format(pval, f_emp, f_th)

        return adeq, msg


def uniqueness_of_items(dpp, samples, **kwargs):
    """Check that each sample is made of unique items (no duplicates)"""

    adeq = all(len(set(x)) == len(x) for x in samples)
    msg = "Some samples contain duplicated items, while each item must appear only once"

    return adeq, msg


def cardinality_adequation(dpp, samples, **kwargs):
    """Check that the empirical cardinality of the samples is within a standard deviation to the true E[|X|] = Trace(K).
    For k-DPP, simply check that the samples have the prescribed cadinality"""

    card_emp = np.array([len(sample) for sample in samples])
    mean_card_emp = np.mean(card_emp)

    if dpp.size_k_dpp:
        adeq = np.all(card_emp == dpp.size_k_dpp)
        msg = "|X|_emp = {}, |X|_th = {}".format(mean_card_emp, dpp.size_k_dpp)

    else:
        dpp.compute_correlation_kernel()
        # E[|X|] = Trace(K), Var[|X|] = Trace(K - K^2)
        mean_card_th = np.trace(dpp.K)
        if dpp.kernel_type == "correlation" and dpp.projection:
            # E[|X|] = |X| = Trace(K) |X| = rank(K)
            mean_card_th = np.rint(mean_card_th).astype(int)

        std_card_th = np.sqrt(np.abs(np.trace(dpp.K - dpp.K.dot(dpp.K))))

        adeq = np.abs(mean_card_emp - mean_card_th) <= std_card_th
        msg = "E_emp = {}, E_th = {}, Std_th = {}".format(
            mean_card_emp, mean_card_th, std_card_th
        )

    return adeq, msg


def log_likelihood_adequation(dpp, samples, k_dpp=False, **kwargs):
    log_lik_emp = np.zeros(len(samples))
    log_lik_theo = np.zeros(len(samples))
    for i, (X, log_likelihood) in enumerate(samples):
        log_lik_emp[i] = log_likelihood
        log_lik_theo[i] = loglikelihood(dpp, X, k_dpp=k_dpp)

    adeq = np.allclose(log_lik_emp, log_lik_theo)
    msg = "{}".format(np.vstack((log_lik_emp, log_lik_theo)))

    return adeq, msg


def select_adequation(name):
    adequations = {
        "uniqueness_of_items": uniqueness_of_items,
        "cardinality": cardinality_adequation,
        "singleton": singleton_adequation,
        "doubleton": doubleton_adequation,
        "log_likelihood": log_likelihood_adequation,
    }
    return adequations.get(name)


class TestAdequationOfFiniteDppSamplers(unittest.TestCase):
    """For various settings, test that the"""

    rank, N = 6, 10
    nb_exact_samples = 1000
    nb_iter_mcmc = nb_exact_samples

    e_vals_eq_01 = np.ones(rank)
    e_vals_in_01 = np.random.rand(rank)
    e_vals_geq_0 = 1 + rndm.geometric(p=0.5, size=rank)

    e_vecs, *_ = la.qr(rndm.randn(N, rank), mode="economic")

    A_zono = rndm.randn(rank, N)

    phi = rndm.randn(rank, N)

    def iter_configurations(self, list_dpp_params, sampler_method_params):
        for sampler_type, list_method_params in sampler_method_params.items():
            for method_params in list_method_params:
                for dpp_params in list_dpp_params:
                    # define a dpp for each sampling type and sampling method
                    yield Configuration(dpp_params, sampler_type, method_params)

    def run_adequation_tests(
        self,
        dpp_params,
        sampler_method_params,
        adequations_to_check,
    ):

        for config in self.iter_configurations(dpp_params, sampler_method_params):
            # Initialize DPP and generate samples a single time
            # before performing checks for performance reasons (sampling is the bottleneck)
            dpp = config.create_dpp()
            samples = config.get_samples(dpp, self.nb_exact_samples)
            for (name, adeq_params) in adequations_to_check.items():
                with self.subTest(
                    config=config.__dict__,
                    adequation=name,
                ):
                    adequation = select_adequation(name)
                    adeq, msg = adequation(dpp, samples, **adeq_params)
                    self.assertTrue(adeq, msg)

    def test_adequation_of_projection_dpp_K_zonotope_sampler(self):

        dpp_params = [
            {
                "kernel_type": "correlation",
                "projection": True,
                "A_zono": self.A_zono,
            },
        ]

        sampler_method_params = {
            "mcmc_dpp": (
                {
                    "method": "zonotope",
                },
            ),
        }

        adequations_to_check = {
            "uniqueness_of_items": {},
            "cardinality": {},
            "singleton": {},
            "doubleton": {},
        }

        self.run_adequation_tests(
            dpp_params, sampler_method_params, adequations_to_check
        )

    def test_dpp_adequation_with_projection_correlation_kernel(self):

        U = self.e_vecs
        dpp_params = [
            {
                "kernel_type": "correlation",
                "projection": True,
                "K": (U * self.e_vals_eq_01).dot(U.T),
            },
            {
                "kernel_type": "correlation",
                "projection": True,
                "K_eig_dec": (self.e_vals_eq_01, U),
            },
            {
                "kernel_type": "correlation",
                "projection": True,
                "A_zono": self.A_zono,
            },
        ]

        k = self.rank
        sampler_method_params = {
            "exact_dpp": (
                {"method": "spectral", "mode": "gs"},
                {"method": "projection", "mode": "cho"},
                {"method": "projection", "mode": "lu"},
                {"method": "sequential", "mode": "lu"},
                {"method": "sequential", "mode": "ldl"},
            ),
            "exact_k_dpp": (
                {"method": "spectral", "mode": "gs", "size": k},
                {"method": "spectral", "mode": "gs-perm", "size": k},
                {"method": "projection", "mode": "cho", "size": k},
                {"method": "projection", "mode": "lu", "size": k},
            ),
            "mcmc_dpp": ({"method": "E", "size": k, "nb_iter": self.nb_iter_mcmc},),
            "mcmc_k_dpp": ({"method": "E", "size": k, "nb_iter": self.nb_iter_mcmc},),
        }

        adequations_to_check = {
            "uniqueness_of_items": {},
            "cardinality": {},
            "singleton": {},
            "doubleton": {},
        }

        self.run_adequation_tests(
            dpp_params, sampler_method_params, adequations_to_check
        )

    def test_dpp_adequation_with_non_projection_correlation_kernel(self):

        U = self.e_vecs
        dpp_params = [
            {
                "kernel_type": "correlation",
                "projection": False,
                "K": (U * self.e_vals_in_01).dot(U.T),
            },
            {
                "kernel_type": "correlation",
                "projection": False,
                "K_eig_dec": (self.e_vals_in_01, U),
            },
        ]

        k = self.rank // 2

        sampler_method_params = {
            "exact_dpp": (
                {"method": "spectral", "mode": "gs"},
                {"method": "spectral", "mode": "gs-perm"},
                {"method": "sequential", "mode": "lu"},
                {"method": "sequential", "mode": "ldl"},
            ),
            "exact_k_dpp": (
                {"method": "spectral", "mode": "gs", "size": k},
                {"method": "spectral", "mode": "gs-perm", "size": k},
            ),
            "mcmc_dpp": (
                {"method": "AD", "nb_iter": self.nb_iter_mcmc},
                {"method": "AED", "nb_iter": self.nb_iter_mcmc},
            ),
            "mcmc_k_dpp": ({"method": "E", "size": k, "nb_iter": self.nb_iter_mcmc},),
        }

        adequations_to_check = {
            "uniqueness_of_items": {},
            "cardinality": {},
            "singleton": {},
            "doubleton": {},
        }

        self.run_adequation_tests(
            dpp_params, sampler_method_params, adequations_to_check
        )

    def test_dpp_adequation_with_projection_likelihood_kernel(self):

        U = self.e_vecs
        dpp_params = [
            {
                "kernel_type": "likelihood",
                "projection": True,
                "L": (U * self.e_vals_eq_01).dot(U.T),
            },
            {
                "kernel_type": "likelihood",
                "projection": True,
                "L_eig_dec": (self.e_vals_eq_01, U),
            },
        ]

        k = self.rank
        sampler_method_params = {
            "exact_dpp": (
                {"method": "spectral", "mode": "gs"},
                {"method": "spectral", "mode": "gs-perm"},
                {"method": "sequential", "mode": "lu"},
                {"method": "sequential", "mode": "ldl"},
            ),
            "exact_k_dpp": (
                {"method": "spectral", "mode": "gs", "size": k},
                {"method": "spectral", "mode": "gs-perm", "size": k},
                {"method": "projection", "mode": "cho", "size": k},
                {"method": "projection", "mode": "lu", "size": k},
            ),
            "mcmc_dpp": (
                {"method": "AD", "nb_iter": self.nb_iter_mcmc},
                {"method": "AED", "nb_iter": self.nb_iter_mcmc},
            ),
            "mcmc_k_dpp": ({"method": "E", "size": k, "nb_iter": self.nb_iter_mcmc},),
        }

        adequations_to_check = {
            "uniqueness_of_items": {},
            "cardinality": {},
            "singleton": {},
            "doubleton": {},
        }

        self.run_adequation_tests(
            dpp_params, sampler_method_params, adequations_to_check
        )

    def test_dpp_adequation_with_non_projection_likelihood_kernel(self):

        U = self.e_vecs
        dpp_params = [
            {
                "kernel_type": "likelihood",
                "projection": False,
                "L": (U * self.e_vals_eq_01).dot(U.T),
            },
            {
                "kernel_type": "likelihood",
                "projection": False,
                "L_eig_dec": (self.e_vals_eq_01, U),
            },
            {
                "kernel_type": "likelihood",
                "projection": False,
                "L": (U * self.e_vals_geq_0).dot(U.T),
            },
            {
                "kernel_type": "likelihood",
                "projection": False,
                "L_eig_dec": (self.e_vals_geq_0, U),
            },
            {
                "kernel_type": "likelihood",
                "projection": False,
                "L_gram_factor": self.phi,
            },
        ]

        k = self.rank // 2
        sampler_method_params = {
            "exact_dpp": (
                {"method": "spectral", "mode": "gs"},
                {"method": "spectral", "mode": "gs-perm"},
                {"method": "sequential", "mode": "lu"},
                {"method": "sequential", "mode": "ldl"},
            ),
            "exact_k_dpp": (
                {"method": "spectral", "mode": "gs", "size": k},
                {"method": "spectral", "mode": "gs-perm", "size": k},
            ),
            "mcmc_dpp": (
                {"method": "AD", "nb_iter": self.nb_iter_mcmc},
                {"method": "AED", "nb_iter": self.nb_iter_mcmc},
            ),
            "mcmc_k_dpp": ({"method": "E", "size": k, "nb_iter": self.nb_iter_mcmc},),
        }

        adequations_to_check = {
            "uniqueness_of_items": {},
            "cardinality": {},
            "singleton": {},
            "doubleton": {},
        }

        self.run_adequation_tests(
            dpp_params, sampler_method_params, adequations_to_check
        )

    def test_adequation_intermediate_sampler(self):

        # def compute_expected_cardinality(L_xy, X):
        #     L = L_xy(X)
        #     I_L_min = L + np.eye(*L.shape)
        #     exp_card = np.sum(np.diag(L.dot(np.linalg.inv(I_L_min))))
        #     return np.floor(exp_card).astype(int)

        N = 100
        dpp_params = [
            {
                "kernel_type": "likelihood",
                "projection": False,
                "L_eval_X_data": (kernel_linear, rndm.rand(N, 6)),
            },
            {
                "kernel_type": "likelihood",
                "projection": False,
                "L_eval_X_data": (kernel_minimum, rndm.rand(N, 1)),
            },
        ]
        sampler_method_params = {
            "exact_dpp": (
                {
                    "method": "intermediate",
                    "mode": "vfx",
                    "verbose": False,
                    "rls_oversample_bless": 5,
                },
                {
                    "method": "intermediate",
                    "mode": "alpha",
                    "verbose": False,
                    "rls_oversample_bless": 5,
                },
            ),
            # todo k_dpp samplers need tunning, unconvenient to test
            # "exact_k_dpp": (
            #     {
            #         "method": "intermediate",
            #         "mode": "vfx",
            #         "size": 3,
            #         "verbose": False,
            #         "rls_oversample_bless": 5,
            #     },
            #     {
            #         "method": "intermediate",
            #         "mode": "alpha",
            #         "size": 3,
            #         "verbose": False,
            #         "rls_oversample_bless": 5,
            #     },
            # ),
        }

        adequations_to_check = {
            "uniqueness_of_items": {},
            "cardinality": {},
            "singleton": {},
            "doubleton": {},
        }

        self.run_adequation_tests(
            dpp_params, sampler_method_params, adequations_to_check
        )

    def test_log_likelihood_projection_sampler_eigen_dpp(self):

        e_vals = self.e_vals_eq_01
        U = self.e_vecs
        dpp_params = [
            {
                "kernel_type": "correlation",
                "projection": True,
                "K_eig_dec": (e_vals, U),
            },
        ]

        sampler_method_params = {
            "exact_dpp": (
                {"method": "projection", "mode": "gs", "log_likelihood": True},
                {"method": "projection", "mode": "gs-perm", "log_likelihood": True},
            ),
        }

        adequations_to_check = {"log_likelihood": {}}

        self.run_adequation_tests(
            dpp_params, sampler_method_params, adequations_to_check
        )

    def test_log_likelihood_projection_sampler_kernel_dpp(self):

        U = self.e_vecs
        dpp_params = [
            {"kernel_type": "correlation", "projection": True, "K": U.dot(U.T)},
        ]

        sampler_method_params = {
            "exact_dpp": (
                {"method": "projection", "mode": "cho", "log_likelihood": True},
                {"method": "projection", "mode": "lu", "log_likelihood": True},
            ),
        }

        adequations_to_check = {"log_likelihood": {}}

        self.run_adequation_tests(
            dpp_params, sampler_method_params, adequations_to_check
        )

    def test_log_likelihood_projection_sampler_eigen_k_dpp(self):

        r = self.rank
        e_vals = self.e_vals_eq_01
        U = self.e_vecs
        dpp_params = [
            {
                "kernel_type": "likelihood",
                "projection": True,
                "L_eig_dec": (e_vals, U),
            },
        ]

        sampler_method_params = {
            "exact_k_dpp": (
                {
                    "size": r // 2,
                    "method": "projection",
                    "mode": "gs",
                    "log_likelihood": True,
                },
                {
                    "size": r // 2,
                    "method": "projection",
                    "mode": "gs-perm",
                    "log_likelihood": True,
                },
            ),
        }

        adequations_to_check = {"log_likelihood": {"k_dpp": True}}

        self.run_adequation_tests(
            dpp_params, sampler_method_params, adequations_to_check
        )

    def test_log_likelihood_projection_sampler_kernel_k_dpp(self):

        r = self.rank
        U = self.e_vecs
        dpp_params = [
            {"kernel_type": "likelihood", "projection": True, "L": U.dot(U.T)},
        ]

        sampler_method_params = {
            "exact_k_dpp": (
                {
                    "size": r // 2,
                    "method": "projection",
                    "mode": "cho",
                    "log_likelihood": True,
                },
                {
                    "size": r // 2,
                    "method": "projection",
                    "mode": "lu",
                    "log_likelihood": True,
                },
            ),
        }

        adequations_to_check = {"log_likelihood": {"k_dpp": True}}

        self.run_adequation_tests(
            dpp_params, sampler_method_params, adequations_to_check
        )

    def test_log_likelihood_sequential_sampler(self):

        U = self.e_vecs
        dpp_params = [
            {"kernel_type": "correlation", "projection": True, "K": U.dot(U.T)},
        ]

        sampler_method_params = {
            "exact_dpp": (
                {"method": "sequential", "mode": "lu", "log_likelihood": True},
                {"method": "sequential", "mode": "ldl", "log_likelihood": True},
            ),
        }

        adequations_to_check = {"log_likelihood": {}}

        self.run_adequation_tests(
            dpp_params, sampler_method_params, adequations_to_check
        )


def main():

    unittest.main()


if __name__ == "__main__":
    main()
