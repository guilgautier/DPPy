# coding: utf8
""" Tests:

- :class:`EquilibriumDistributionOfBetaEnsembles` to check that for a large enough number of points, the empirical distribution of the rescaled points is close to the expected limiting (equilibrium) distribution
"""

import unittest

import numpy as np

from scipy.integrate import quad
from scipy.stats import chisquare

import sys
sys.path.append('..')

import dppy.random_matrices as rm
import dppy.beta_ensembles as be


class EquilibriumDistributionOfBetaEnsembles(unittest.TestCase):
    """
    """

    N = 300
    nb_samples = 10

    def adequation(self, f_emp, f_theo, tol=0.05):
        """Perform chi-square test"""

        _, pval = chisquare(f_obs=f_emp, f_exp=f_theo)
        print(pval)

        return pval > tol

    # Semi circle and Hermite ensemble
    def test_semi_circle_distribution_for_hermite_ensemble_full_model(self):
        """
        """
        tol = 1e-2
        bins = np.linspace(-2.0 + tol, 2.0 - tol, 10)

        f_theo = [quad(rm.semi_circle_law, a, b)[0]
                  for a, b in zip(bins[:-1], bins[1:])]

        for beta in (1, 2, 4):

            he = be.HermiteEnsemble(beta)

            for _ in range(self.nb_samples):
                he.sample_full_model(size_N=self.N)

                vals, _ =\
                    np.histogram(he.normalize_points(he.list_of_samples[-1]),
                                 bins=bins,
                                 density=True)

                f_emp = vals * np.diff(bins)

                self.assertTrue(self.adequation(f_emp, f_theo))

    def test_semi_circle_for_hermite_ensemble_banded_model(self):
        """
        """
        supp = [-2.0, 2.0]
        tol = 1e-2
        bins = np.linspace(supp[0] + tol, supp[1] - tol, 10)

        f_theo = [quad(rm.semi_circle_law, a, b)[0]
                  for a, b in zip(bins[:-1], bins[1:])]

        for beta in (1, 2, 4):

            he = be.HermiteEnsemble(beta)

            for _ in range(self.nb_samples):
                he.sample_banded_model(size_N=self.N)

                vals, _ =\
                    np.histogram(he.normalize_points(he.list_of_samples[-1]),
                                 bins=bins,
                                 density=True)

                f_emp = vals * np.diff(bins)

                self.assertTrue(self.adequation(f_emp, f_theo))

    # Marcenko-Pastur and Laguerre ensemble
    def test_marcenko_pastur_for_laguerre_ensemble_full_model(self):
        """
        """

        M = 2 * self.N
        supp = (1 + np.array([-1, 1]) * np.sqrt(self.N / M))**2

        tol = 1e-2
        bins = np.linspace(supp[0] + tol, supp[1] - tol, 20)

        f_theo = [quad(lambda x: rm.marcenko_pastur_law(x, M, self.N), a, b)[0]
                  for a, b in zip(bins[:-1], bins[1:])]

        for beta in (1, 2, 4):

            le = be.LaguerreEnsemble(beta)

            for _ in range(self.nb_samples):
                le.sample_full_model(size_N=self.N, size_M=M)

                vals, _ =\
                    np.histogram(le.normalize_points(le.list_of_samples[-1]),
                                 bins=bins,
                                 density=True)

                f_emp = vals * np.diff(bins)
                print(f_emp)

                self.assertTrue(self.adequation(f_emp, f_theo))

    def test_marcenko_pastur_for_laguerre_ensemble_banded_model(self):
        """
        """

        M = 2 * self.N
        supp = (1 + np.array([-1, 1]) * np.sqrt(self.N / M))**2

        tol = 1e-2
        bins = np.linspace(supp[0] + tol, supp[1] - tol, 20)

        f_theo = [quad(lambda x: rm.marcenko_pastur_law(x, M, self.N), a, b)[0]
                  for a, b in zip(bins[:-1], bins[1:])]

        for beta in (1, 2, 4):

            le = be.LaguerreEnsemble(beta)

            for _ in range(self.nb_samples):
                le.sample_banded_model(size_N=self.N, size_M=M)

                vals, _ =\
                    np.histogram(le.normalize_points(le.list_of_samples[-1]),
                                 bins=bins,
                                 density=True)

                f_emp = vals * np.diff(bins)
                print(f_emp)

                self.assertTrue(self.adequation(f_emp, f_theo))

    # Wachter and Jacobi ensemble
    def test_wachter_for_jacobi_ensemble_full_model(self):
        """
        """

        M_1, M_2 = 2 * self.N, 4 * self.N
        a, b = M_1 / self.N, M_2 / self.N
        supp = ((np.sqrt(a * (a + b - 1)) + np.array([-1, 1]) * np.sqrt(b)) / (a + b))**2

        tol = 1e-2
        bins = np.linspace(supp[0] + tol, supp[1] - tol, 20)

        f_theo = [quad(lambda x: rm.wachter_law(x, M_1, M_2, self.N), a, b)[0]
                  for a, b in zip(bins[:-1], bins[1:])]

        for beta in (1, 2, 4):

            je = be.JacobiEnsemble(beta)

            for _ in range(self.nb_samples):
                je.sample_full_model(size_N=self.N,
                                     size_M1=M_1,
                                     size_M2=M_2)

                vals, _ =\
                    np.histogram(je.list_of_samples[-1],
                                 bins=bins,
                                 density=True)

                f_emp = vals * np.diff(bins)
                print(f_emp)

                self.assertTrue(self.adequation(f_emp, f_theo))

    def test_wachter_for_jacobi_ensemble_banded_model(self):
        """
        """

        M_1, M_2 = 2 * self.N, 4 * self.N
        a, b = M_1 / self.N, M_2 / self.N
        supp = ((np.sqrt(a * (a + b - 1)) + np.array([-1, 1]) * np.sqrt(b)) / (a + b))**2

        tol = 1e-2
        bins = np.linspace(supp[0] + tol, supp[1] - tol, 20)

        f_theo = [quad(lambda x: rm.wachter_law(x, M_1, M_2, self.N), a, b)[0]
                  for a, b in zip(bins[:-1], bins[1:])]

        for beta in (1, 2, 4):

            je = be.JacobiEnsemble(beta)

            for _ in range(self.nb_samples):
                je.sample_banded_model(size_N=self.N,
                                       size_M1=M_1,
                                       size_M2=M_2)

                vals, _ =\
                    np.histogram(je.list_of_samples[-1],
                                 bins=bins,
                                 density=True)

                f_emp = vals * np.diff(bins)
                print(f_emp)

                self.assertTrue(self.adequation(f_emp, f_theo))

    # Uniformity of angles and Circular ensemble
    def test_uniformity_of_angles_for_circular_ensemble_full_model(self):
        """
        """

        supp = [0, 2 * np.pi]

        tol = 1e-2
        bins = np.linspace(supp[0] + tol, supp[1] - tol, 20)

        f_theo = np.diff(bins) / (supp[1] - supp[0])

        for beta in (1, 2, 4):

            ce = be.CircularEnsemble(beta)

            for _ in range(self.nb_samples):
                ce.sample_full_model(size_N=self.N)

                vals, _ =\
                    np.histogram(np.angle(ce.list_of_samples[-1]),
                                 bins=bins,
                                 density=True)

                f_emp = vals * np.diff(bins)
                print(f_emp)

                self.assertTrue(self.adequation(f_emp, f_theo))

    def test_uniformity_of_angles_for_circular_ensemble_banded_model(self):
        """
        """

        supp = [0, 2 * np.pi]

        tol = 1e-2
        bins = np.linspace(supp[0] + tol, supp[1] - tol, 20)

        f_theo = np.diff(bins) / (supp[1] - supp[0])

        for beta in (1, 2, 4):

            ce = be.CircularEnsemble(beta)

            for _ in range(self.nb_samples):
                ce.sample_banded_model(size_N=self.N)

                vals, _ =\
                    np.histogram(np.angle(ce.list_of_samples[-1]),
                                 bins=bins,
                                 density=True)

                f_emp = vals * np.diff(bins)
                print(f_emp)

                self.assertTrue(self.adequation(f_emp, f_theo))

    # Linearity of radii and Ginibre ensemble
    def test_linear_radius_distribution_of_ginibre_ensemble(self):
        """
        """

        supp = [0.0, 1.0]

        tol = 1e-2
        bins = np.linspace(supp[0] + tol, supp[1] - tol, 20)

        f_theo = [quad(lambda x: 2 * x, a, b)[0]
                  for a, b in zip(bins[:-1], bins[1:])]

        beta = 2
        ge = be.GinibreEnsemble(beta)

        for _ in range(self.nb_samples):
            ge.sample_full_model(size_N=self.N)

            vals, _ =\
                np.histogram(
                    np.abs(ge.normalize_points(ge.list_of_samples[-1])),
                    bins=bins,
                    density=True)

            f_emp = vals * np.diff(bins)
            print(f_emp)

            self.assertTrue(self.adequation(f_emp, f_theo))


def main():

    unittest.main()


if __name__ == '__main__':
    main()
