# coding: utf8
""" Unit tests:

Simply check that the various samplers for :class:`FiniteDPP` generate samples with the right cardinality
"""

import unittest

import numpy as np
import numpy.random as rndm

from scipy.linalg import qr

import sys
sys.path.append('..')

from dppy.finite_dpps import FiniteDPP


class CardinalityOfFiniteDPPs(unittest.TestCase):
    """ E[|X|] = \\sum_{n=1}^{N} \\lambda_n(K)
        Var[|X|] = \\sum_{n=1}^{N} \\lambda_n(K)(1-\\lambda_n(K))
    """

    nb_samples = 1000
    rank, N = 6, 10

    def check_right_cardinality(self, dpp, samples):

        dpp.compute_K()

        mean_card_theo = np.matrix.trace(dpp.K)

        card_emp = np.array(list(map(len, samples)))

        if dpp.kernel_type == 'correlation' and dpp.projection:

            mean_card_theo = np.round(mean_card_theo).astype(int)
            return self.assertTrue(np.all(card_emp == mean_card_theo))

        else:
            var_card_theo = mean_card_theo\
                            - np.matrix.trace(np.linalg.matrix_power(dpp.K, 2))
            std_card_theo = np.sqrt(var_card_theo)

            mean_card_emp = np.mean(card_emp)

            tol = mean_card_theo + np.array([-1, 1]) * std_card_theo

            return self.assertTrue(tol[0] <= mean_card_emp <= tol[1])

    # Correlation kernel
    def test_correlation_kernel_projection_kernel(self):

        eig_vals = np.ones(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode='economic')

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=True,
                        **{'K': (eig_vecs * eig_vals).dot(eig_vecs.T)})

        for mode in ('GS', 'Chol', 'Schur'):

            dpp.flush_samples()
            for _ in range(self.nb_samples):
                dpp.sample_exact(mode)

            self.check_right_cardinality(dpp, dpp.list_of_samples)

        for mode in ('E'):

            dpp.flush_samples()
            dpp.sample_mcmc(mode, **{'size': self.rank, 'nb_iter': self.nb_samples})

            self.check_right_cardinality(dpp, dpp.list_of_samples[0])

    def test_correlation_kernel_projection_kernel_eig(self):

        eig_vals = np.ones(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode='economic')

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=True,
                        **{'K_eig_dec': (eig_vals, eig_vecs)})

        for mode in ('GS', 'GS_bis', 'KuTa12'):

            dpp.flush_samples()
            for _ in range(self.nb_samples):
                dpp.sample_exact(mode)

            self.check_right_cardinality(dpp, dpp.list_of_samples)

        for mode in ('E'):

            dpp.flush_samples()
            dpp.sample_mcmc(mode, **{'size': self.rank, 'nb_iter': self.nb_samples})

            self.check_right_cardinality(dpp, dpp.list_of_samples[0])

    def test_correlation_kernel_projection_A_zono(self):
        A = rndm.randn(self.rank, self.N)

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=True,
                        **{'A_zono': A})

        for mode in ('GS', 'GS_bis', 'KuTa12'):

            dpp.flush_samples()
            for _ in range(self.nb_samples):
                dpp.sample_exact(mode)

            self.check_right_cardinality(dpp, dpp.list_of_samples)

        for mode in ('zonotope', 'E'):

            dpp.flush_samples()
            dpp.sample_mcmc(mode, **{'size': self.rank, 'nb_iter': self.nb_samples})

            self.check_right_cardinality(dpp, dpp.list_of_samples[0])

    def test_kernel_eig(self):

        eig_vals = rndm.rand(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode='economic')

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=False,
                        **{'K_eig_dec': (eig_vals, eig_vecs)})

        for mode in ('GS', 'GS_bis', 'KuTa12'):

            dpp.flush_samples()
            for _ in range(self.nb_samples):
                dpp.sample_exact(mode)

            self.check_right_cardinality(dpp, dpp.list_of_samples)

        for mode in ('AED', 'AD'):

            dpp.flush_samples()
            dpp.sample_mcmc(mode, **{'nb_iter': self.nb_samples})

            self.check_right_cardinality(dpp, dpp.list_of_samples[0])

    def test_kernel(self):

        eig_vals = rndm.rand(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode='economic')

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=False,
                        **{'K': (eig_vecs * eig_vals).dot(eig_vecs.T)})

        for mode in ('GS', 'GS_bis', 'KuTa12'):

            dpp.flush_samples()
            for _ in range(self.nb_samples):
                dpp.sample_exact(mode)

            self.check_right_cardinality(dpp, dpp.list_of_samples)

        for mode in ('AED', 'AD'):

            dpp.flush_samples()
            dpp.sample_mcmc(mode, **{'nb_iter': self.nb_samples})

            self.check_right_cardinality(dpp, dpp.list_of_samples[0])

    # Likelihood kernel
    def test_likelihood_kernel_eig(self):

        eig_vals = 1 + rndm.geometric(p=0.5, size=self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode='economic')

        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=False,
                        **{'L_eig_dec': (eig_vals, eig_vecs)})

        for mode in ('GS', 'GS_bis', 'KuTa12'):

            dpp.flush_samples()
            for _ in range(self.nb_samples):
                dpp.sample_exact(mode)

            self.check_right_cardinality(dpp, dpp.list_of_samples)

        for mode in ('AED', 'AD'):

            dpp.flush_samples()
            dpp.sample_mcmc(mode, **{'nb_iter': self.nb_samples})

            self.check_right_cardinality(dpp, dpp.list_of_samples[0])

    def test_likelihood_kernel(self):

        eig_vals = 1 + rndm.geometric(p=0.5, size=self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode='economic')

        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=False,
                        **{'L': (eig_vecs * eig_vals).dot(eig_vecs.T)})

        for mode in ('GS', 'GS_bis', 'KuTa12'):

            dpp.flush_samples()
            for _ in range(self.nb_samples):
                dpp.sample_exact(mode)

            self.check_right_cardinality(dpp, dpp.list_of_samples)

        for mode in ('AED', 'AD'):

            dpp.flush_samples()
            dpp.sample_mcmc(mode, **{'nb_iter': self.nb_samples})

            self.check_right_cardinality(dpp, dpp.list_of_samples[0])

    def test_likelihood_kernel_L_gram_factor(self):

        phi = rndm.randn(self.rank, self.N)

        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=False,
                        **{'L_gram_factor': phi})

        for mode in ('GS', 'GS_bis', 'KuTa12'):

            dpp.flush_samples()
            for _ in range(self.nb_samples):
                dpp.sample_exact(mode)

            self.check_right_cardinality(dpp, dpp.list_of_samples)

        for mode in ('AED', 'AD'):

            dpp.flush_samples()
            dpp.sample_mcmc(mode, **{'nb_iter': self.nb_samples})

            self.check_right_cardinality(dpp, dpp.list_of_samples[0])

class CardinalityOfFinitekDPPs(unittest.TestCase):
    """
    """

    nb_samples = 1000
    rank, N = 6, 10
    sizes = (1, 3, 5)

    def check_right_cardinality(self, dpp, samples):

        card_emp = np.array(list(map(len, samples)))

        return self.assertTrue(np.all(card_emp == dpp.size_k_dpp))

    # Correlation kernel
    def test_kernel_eig(self):

        eig_vals = rndm.rand(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode='economic')

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=False,
                        **{'K_eig_dec': (eig_vals, eig_vecs)})

        for size in self.sizes:

            for mode in ('GS', 'GS_bis', 'KuTa12'):

                dpp.flush_samples()
                for _ in range(self.nb_samples):
                    dpp.sample_exact_k_dpp(size, mode)

                self.check_right_cardinality(dpp, dpp.list_of_samples)

            for mode in ('AED', 'AD'):

                dpp.flush_samples()
                dpp.sample_mcmc_k_dpp(size, **{'nb_iter': self.nb_samples})

                self.check_right_cardinality(dpp, dpp.list_of_samples[0])

    def test_kernel(self):

        eig_vals = rndm.rand(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode='economic')

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=False,
                        **{'K': (eig_vecs * eig_vals).dot(eig_vecs.T)})

        for size in self.sizes:

            for mode in ('GS', 'GS_bis', 'KuTa12'):

                dpp.flush_samples()
                for _ in range(self.nb_samples):
                    dpp.sample_exact_k_dpp(size, mode)

                self.check_right_cardinality(dpp, dpp.list_of_samples)

            for mode in ('AED', 'AD'):

                dpp.flush_samples()
                dpp.sample_mcmc_k_dpp(size, **{'nb_iter': self.nb_samples})

                self.check_right_cardinality(dpp, dpp.list_of_samples[0])

    # Likelihood kernel
    def test_likelihood_kernel_eig(self):

        eig_vals = 1 + rndm.geometric(p=0.5, size=self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode='economic')

        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=False,
                        **{'L_eig_dec': (eig_vals, eig_vecs)})

        for size in self.sizes:

            for mode in ('GS', 'GS_bis', 'KuTa12'):

                dpp.flush_samples()
                for _ in range(self.nb_samples):
                    dpp.sample_exact_k_dpp(size, mode)

                self.check_right_cardinality(dpp, dpp.list_of_samples)

            for mode in ('AED', 'AD'):

                dpp.flush_samples()
                dpp.sample_mcmc_k_dpp(size, **{'nb_iter': self.nb_samples})

                self.check_right_cardinality(dpp, dpp.list_of_samples[0])

    def test_likelihood_kernel(self):

        eig_vals = 1 + rndm.geometric(p=0.5, size=self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode='economic')

        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=False,
                        **{'L': (eig_vecs * eig_vals).dot(eig_vecs.T)})

        for size in self.sizes:

            for mode in ('GS', 'GS_bis', 'KuTa12'):

                dpp.flush_samples()
                for _ in range(self.nb_samples):
                    dpp.sample_exact_k_dpp(size, mode)

                self.check_right_cardinality(dpp, dpp.list_of_samples)

            for mode in ('AED', 'AD'):

                dpp.flush_samples()
                dpp.sample_mcmc_k_dpp(size, **{'nb_iter': self.nb_samples})

                self.check_right_cardinality(dpp, dpp.list_of_samples[0])

    def test_likelihood_kernel_L_gram_factor(self):

        phi = rndm.randn(self.rank, self.N)

        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=False,
                        **{'L_gram_factor': phi})

        for size in self.sizes:

            for mode in ('GS', 'GS_bis', 'KuTa12'):

                dpp.flush_samples()
                for _ in range(self.nb_samples):
                    dpp.sample_exact_k_dpp(size, mode)

                self.check_right_cardinality(dpp, dpp.list_of_samples)

            for mode in ('AED', 'AD'):

                dpp.flush_samples()
                dpp.sample_mcmc_k_dpp(size, **{'nb_iter': self.nb_samples})

                self.check_right_cardinality(dpp, dpp.list_of_samples[0])

def main():

    unittest.main()


if __name__ == '__main__':
    main()
