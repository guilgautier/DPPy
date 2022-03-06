# coding: utf8
""" Tests:

- :class:`TestAdequationOfBetaEnsembleSamplers` to check that for a large enough number of points, the empirical distribution of the rescaled points is close to the expected limiting (equilibrium) distribution
"""

import sys
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import chisquare

sys.path.append("..")

import dppy.beta_ensembles.abstract_beta_ensemble as be
import dppy.random_matrices as rm


class TestAdequationOfBetaEnsembleSamplers(unittest.TestCase):

    N = 300
    nb_samples = 5

    @staticmethod
    def limiting_distribution_adequation(
        samples, limiting_distribution, support, tol=0.05
    ):
        """Perform chi-square test to check that for a large enough number of points, the empirical distribution of the rescaled points is close to the expected limiting (equilibrium) distribution"""

        bins = np.linspace(support[0] + tol, support[1] - tol, 20)

        vals, _ = np.histogram(samples, bins=bins, density=True)

        f_emp = vals * np.diff(bins)
        f_th = np.array(
            [quad(limiting_distribution, a, b)[0] for a, b in zip(bins[:-1], bins[1:])]
        )
        f_emp /= f_emp.sum()
        f_th /= f_th.sum()

        _, pval = chisquare(f_obs=f_emp, f_exp=f_th)

        adequation = pval > tol
        message = "pval = {}".format(pval)

        return adequation, message

    def run_adequation_test(
        self,
        point_process,
        limiting_distribution,
        support,
        list_of_beta,
        dict_sampler_param,
        process_samples=lambda x: x,
    ):

        for beta in list_of_beta:
            with self.subTest(point_process=point_process.__name__, beta=beta):
                point_proc = point_process(beta)
                for sampler, params in dict_sampler_param.items():
                    with self.subTest(sampler=sampler, pro_samples=process_samples):
                        point_proc.flush_samples()
                        samples = process_samples(
                            self.get_samples(point_proc, sampler, **params)
                        )

                        adeq, msg = self.limiting_distribution_adequation(
                            samples, limiting_distribution, support
                        )
                        self.assertTrue(adeq, msg)

    def get_samples(self, point_proc, sampler, **params):

        for _ in range(self.nb_samples):
            if sampler == "full":
                point_proc.sample_full_model(**params)
            elif sampler == "banded":
                point_proc.sample_banded_model(**params)

            return point_proc.normalize_points(point_proc.list_of_samples[-1])

    def test_semi_circle_for_hermite_ensemble(self):

        point_process = be.HermiteBetaEnsemble

        limiting_distribution = rm.semi_circle_density
        support = [-2.0, 2.0]

        list_of_beta = [1, 2, 4]

        sampler_params = {"N": self.N}
        dict_sampler_param = {"full": sampler_params, "banded": sampler_params}

        self.run_adequation_test(
            point_process,
            limiting_distribution,
            support,
            list_of_beta,
            dict_sampler_param,
        )

    def test_marcenko_pastur_for_laguerre_ensemble(self):

        point_process = be.LaguerreBetaEnsemble

        N, M = self.N, 2 * self.N

        def limiting_distribution(x):
            return rm.marcenko_pastur_density(x, M, N)

        support = (1 + np.array([-1, 1]) * np.sqrt(N / M)) ** 2

        list_of_beta = [1, 2, 4]

        sampler_params = {"N": N, "M": M}
        dict_sampler_param = {"full": sampler_params, "banded": sampler_params}

        self.run_adequation_test(
            point_process,
            limiting_distribution,
            support,
            list_of_beta,
            dict_sampler_param,
        )

    def test_wachter_for_jacobi_ensemble(self):

        point_process = be.JacobiBetaEnsemble

        N, M_1, M_2 = self.N, 2 * self.N, 3 * self.N

        def limiting_distribution(x):
            return rm.wachter_density(x, M_1, M_2, N)

        a, b = M_1 / N, M_2 / N
        support = np.sqrt(a * (a + b - 1)) + np.array([-1, 1]) * np.sqrt(b)
        support /= a + b
        support **= 2

        list_of_beta = [1, 2, 4]

        sampler_params = {"N": N, "M1": M_1, "M2": M_2}
        dict_sampler_param = {"full": sampler_params, "banded": sampler_params}

        self.run_adequation_test(
            point_process,
            limiting_distribution,
            support,
            list_of_beta,
            dict_sampler_param,
        )

    def test_uniformity_of_angles_for_circular_ensemble(self):

        point_process = be.CircularBetaEnsemble

        def limiting_distribution(theta):
            return 1.0 / 2 * np.pi

        support = [0, 2 * np.pi]

        list_of_beta = [1, 2, 4]
        sampler_params = {"N": self.N}
        dict_sampler_param = {"full": sampler_params, "banded": sampler_params}

        self.run_adequation_test(
            point_process,
            limiting_distribution,
            support,
            list_of_beta,
            dict_sampler_param,
            process_samples=np.angle,
        )

    def test_linearity_of_radius_for_ginibre_ensemble(self):

        point_process = be.GinibreEnsemble

        def limiting_distribution(r):
            return 2 * r

        support = [0.0, 1.0]

        list_of_beta = [2]
        sampler_params = {"N": self.N}
        dict_sampler_param = {"full": sampler_params}

        self.run_adequation_test(
            point_process,
            limiting_distribution,
            support,
            list_of_beta,
            dict_sampler_param,
            process_samples=np.abs,
        )


def main():

    unittest.main()


if __name__ == "__main__":
    main()