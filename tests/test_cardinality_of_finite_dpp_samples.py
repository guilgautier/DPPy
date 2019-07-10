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
    """ Projection DPP(K) generate fixed size cardinality samples, equal to the rank of the kernel K
    """

    nb_samples = 1000

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
    def test_correlation_projection_kernel_eig(self):

        rank, N = 6, 10

        eig_vals = np.ones(rank)
        eig_vecs, _ = qr(rndm.randn(N, rank), mode="economic")

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
            dpp.sample_mcmc(mode, **{'size': rank, 'nb_iter': self.nb_samples})

            self.check_right_cardinality(dpp, dpp.list_of_samples[0])

    def test_correlation_projection_kernel(self):

        rank, N = 6, 10

        eig_vals = np.ones(rank)
        eig_vecs, _ = qr(rndm.randn(N, rank), mode="economic")

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
            dpp.sample_mcmc(mode, **{'size': rank, 'nb_iter': self.nb_samples})

            self.check_right_cardinality(dpp, dpp.list_of_samples[0])

    def test_correlation_projection_A_zono(self):

        rank, N = 6, 10
        A = rndm.randn(rank, N)

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
            dpp.sample_mcmc(mode, **{'size': rank, 'nb_iter': self.nb_samples})

            self.check_right_cardinality(dpp, dpp.list_of_samples[0])

    def test_correlation_kernel_eig(self):

        rank, N = 6, 10

        eig_vals = rndm.rand(rank)
        eig_vecs, _ = qr(rndm.randn(N, rank), mode="economic")

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

    def test_correlation_kernel(self):

        rank, N = 6, 10

        eig_vals = rndm.rand(rank)
        eig_vecs, _ = qr(rndm.randn(N, rank), mode="economic")

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

        rank, N = 6, 10

        eig_vals = 1 + rndm.geometric(p=0.5, size=rank)
        eig_vecs, _ = qr(rndm.randn(N, rank), mode="economic")

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

        rank, N = 6, 10

        eig_vals = 1 + rndm.geometric(p=0.5, size=rank)
        eig_vecs, _ = qr(rndm.randn(N, rank), mode="economic")

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

        rank, N = 6, 10

        phi = rndm.randn(rank, N)

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

def main():

    unittest.main()


if __name__ == '__main__':
    main()
