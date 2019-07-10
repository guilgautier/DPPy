# coding: utf8
""" Unit tests:

- :class:`InclusionProbabilitiesProjectionDPP` to check that exact samplers for finite DPPs have the right (at least 1 and 2) inclusion probabilities
"""

import unittest

import numpy as np
import numpy.random as rndm

from scipy.linalg import qr
from scipy.stats import chisquare

from itertools import chain  # to flatten list of samples

import sys
sys.path.append('..')

from dppy.finite_dpps import FiniteDPP
from dppy.exact_sampling import dpp_sampler_generic_kernel
from dppy.utils import det_ST


class InclusionProbabilitiesProjectionDPP(unittest.TestCase):
    """Check that exact samplers for finite DPPs have the right (at least 1 and 2) inclusion probabilities

    .. math::

        \\mathbb{P}[S\\subset \\mathcal{X}] = \\det K_S
    """

    rank, N = 6, 10
    nb_samples = 100

    def singleton_adequation(self, dpp, samples, tol=0.05):
        """Perform chi-square test"""
        dpp.compute_K()

        singletons = list(chain.from_iterable(samples))

        freq, _ = np.histogram(singletons, bins=range(self.N + 1), density=True)
        marg_theo = np.diag(dpp.K) / self.rank

        _, pval = chisquare(f_obs=freq, f_exp=marg_theo)

        print(freq)
        print(marg_theo)
        return pval > tol

    def doubleton_adequation(self, dpp, samples, tol=0.05):
        """Perform chi-square test"""

        samples = list(map(set, samples))
        dpp.compute_K()

        nb_doubletons_to_check = 10
        doubletons = [set(rndm.choice(self.N,
                                      size=2,
                                      p=np.diag(dpp.K) / self.rank,
                                      replace=False))
                      for _ in range(nb_doubletons_to_check)]

        counts = [sum([doubl.issubset(sampl) for sampl in samples])
                  for doubl in doubletons]
        freq = np.array(counts) / len(samples)
        marg_theo = [det_ST(dpp.K, list(d)) for d in doubletons]

        print(freq)
        print(marg_theo)

        _, pval = chisquare(f_obs=freq, f_exp=marg_theo)

        return pval > tol

    # From eigendecomposition
    def test_proj_dpp_sampler_from_eigdec_mode_GS(self):
        """ Test whether 'GS' sampling mode generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by orthogonal projection correlation kernel K from its eigendecomposition

        Complexity :math:`\\mathcal{O}(N rank^2)`

        This is the default sampler when calling `.sample_exact()`
        """
        eig_vals = np.ones(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode="economic")
        dpp = FiniteDPP(kernel_type='correlation',
                        projection=True,
                        **{'K_eig_dec': (eig_vals, eig_vecs)})

        dpp.flush_samples()
        for _ in range(self.nb_samples):
            dpp.sample_exact(mode='GS')


        self.assertTrue(self.singleton_adequation(dpp, dpp.list_of_samples))
        self.assertTrue(self.doubleton_adequation(dpp, dpp.list_of_samples))

    def test_proj_dpp_sampler_from_eigdec_mode_GS_bis(self):
        """ Test whether 'GS_bis' sampling mode generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by orthogonal projection correlation kernel K from its eigendecomposition

        Complexity :math:`\\mathcal{O}(N rank^2)`

        Evaluate the conditionals using an alternative GS
        """
        eig_vals = np.ones(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode="economic")
        dpp = FiniteDPP(kernel_type='correlation',
                        projection=True,
                        **{'K_eig_dec': (eig_vals, eig_vecs)})

        dpp.flush_samples()
        for _ in range(self.nb_samples):
            dpp.sample_exact(mode='GS_bis')


        self.assertTrue(self.singleton_adequation(dpp, dpp.list_of_samples))
        self.assertTrue(self.doubleton_adequation(dpp, dpp.list_of_samples))

    def test_proj_dpp_sampler_from_eigdec_mode_KuTa12(self):
        """ Test whether 'KuTa12' sampling mode generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by orthogonal projection correlation kernel K from its eigendecomposition

        Complexity :math:`\\mathcal{O}(N rank^3)`
        """
        eig_vals = np.ones(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode="economic")
        dpp = FiniteDPP(kernel_type='correlation',
                        projection=True,
                        **{'K_eig_dec': (eig_vals, eig_vecs)})

        dpp.flush_samples()
        for _ in range(self.nb_samples):
            dpp.sample_exact(mode='KuTa12')


        self.assertTrue(self.singleton_adequation(dpp, dpp.list_of_samples))
        self.assertTrue(self.doubleton_adequation(dpp, dpp.list_of_samples))

    # From kernel
    def test_proj_dpp_sampler_from_eigdec_mode_Chol(self):
        """ Test whether 'Chol' sampling mode generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by orthogonal projection correlation kernel K from its eigendecomposition.

        Complexity :math:`\\mathcal{O}(N rank^2)`

        .. seealso::

            - :cite:`Pou19` Algorithm 1
        """
        eig_vals = np.ones(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode="economic")
        dpp = FiniteDPP(kernel_type='correlation',
                        projection=True,
                        **{'K': (eig_vecs * eig_vals).dot(eig_vecs.T)})

        dpp.flush_samples()
        for _ in range(self.nb_samples):
            dpp.sample_exact(mode='Chol')

        self.assertTrue(self.singleton_adequation(dpp, dpp.list_of_samples))
        self.assertTrue(self.doubleton_adequation(dpp, dpp.list_of_samples))

    def test_proj_dpp_sampler_generic_kernel(self):
        """ Test whether 'Chol' sampling mode generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by orthogonal projection correlation kernel K from its eigendecomposition.
        The ``projection`` argument is set to ``False`` to make sure the :py:func:`~dppy.exact_sampling.dpp_sampler_generic_kernel` is used

        This is the default sampler when calling `.sample_exact()`
        """
        eig_vals = np.ones(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode="economic")
        dpp = FiniteDPP(kernel_type='correlation',
                        projection=False,
                        **{'K': (eig_vecs * eig_vals).dot(eig_vecs.T)})

        dpp.flush_samples()
        for _ in range(self.nb_samples):
            dpp.sample_exact(mode='Chol')

        self.assertTrue(self.singleton_adequation(dpp, dpp.list_of_samples))
        self.assertTrue(self.doubleton_adequation(dpp, dpp.list_of_samples))

    def test_proj_dpp_sampler_from_kernel_mode_GS(self):
        """ Test whether 'GS' sampling mode generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by orthogonal projection correlation kernel K from its eigendecomposition

        Complexity :math:`\\mathcal{O}(N rank^2)`

        This is the default sampler when calling `.sample_exact()`
        """
        eig_vals = np.ones(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode="economic")
        dpp = FiniteDPP(kernel_type='correlation',
                        projection=True,
                        **{'K': (eig_vecs * eig_vals).dot(eig_vecs.T)})

        dpp.flush_samples()
        for _ in range(self.nb_samples):
            dpp.sample_exact(mode='GS')

        # dpp.compute_K()

        self.assertTrue(self.singleton_adequation(dpp, dpp.list_of_samples))
        self.assertTrue(self.doubleton_adequation(dpp, dpp.list_of_samples))

    def test_proj_dpp_sampler_from_kernel_mode_Schur(self):
        """ Test whether 'Schur' sampling mode generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by orthogonal projection correlation kernel K from its eigendecomposition

        Evaluate the conditionals using the Schur complement updates
        """
        eig_vals = np.ones(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode="economic")
        dpp = FiniteDPP(kernel_type='correlation',
                        projection=True,
                        **{'K': (eig_vecs * eig_vals).dot(eig_vecs.T)})

        dpp.flush_samples()
        for _ in range(self.nb_samples):
            dpp.sample_exact(mode='Schur')

        self.assertTrue(self.singleton_adequation(dpp, dpp.list_of_samples))
        self.assertTrue(self.doubleton_adequation(dpp, dpp.list_of_samples))

    def test_proj_dpp_sampler_as_kDPP_with_correlation(self):
        """ Test whether projection DPP sampled as a k-DPP with k=rank(K) generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by orthogonal projection correlation kernel K from its eigendecomposition
        """

        eig_vals = np.ones(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode="economic")
        dpp = FiniteDPP(kernel_type='correlation',
                        projection=True,
                        **{'K_eig_dec': (eig_vals, eig_vecs)})

        dpp.flush_samples()
        for _ in range(self.nb_samples):
            dpp.sample_exact_k_dpp(self.rank)

        self.assertTrue(self.singleton_adequation(dpp, dpp.list_of_samples))
        self.assertTrue(self.doubleton_adequation(dpp, dpp.list_of_samples))

    def test_proj_dpp_sampler_as_kDPP_with_likelihood(self):
        """ Test whether projection DPP sampled as a k-DPP with k=rank(K)  generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by orthogonal projection likelihood kernel L from its eigendecomposition
        """

        eig_vals = np.ones(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode="economic")
        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=True,
                        **{'L_eig_dec': (eig_vals, eig_vecs)})

        dpp.flush_samples()
        for _ in range(self.nb_samples):
            dpp.sample_exact_k_dpp(self.rank)

        dpp.compute_L()
        dpp.K = dpp.L

        self.assertTrue(self.singleton_adequation(dpp, dpp.list_of_samples))
        self.assertTrue(self.doubleton_adequation(dpp, dpp.list_of_samples))

    def test_proj_dpp_sampler_as_kDPP_with_likelihood(self):
        """ Test whether projection DPP sampled as a k-DPP with k=rank(K)  generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by orthogonal projection likelihood kernel L from its eigendecomposition and projection is set to False in order to go through the computation of elementary symmetric polynomials etc
        """
        eig_vals = np.zeros(self.N)
        eig_vals[:self.rank] = 1.0

        eig_vecs, _ = qr(rndm.randn(self.N, self.N), mode="economic")
        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=False,
                        **{'L_eig_dec': (eig_vals, eig_vecs)})

        dpp.flush_samples()
        for _ in range(self.nb_samples):
            dpp.sample_exact_k_dpp(self.rank)

        dpp.compute_L()
        dpp.K = dpp.L

        self.assertTrue(self.singleton_adequation(dpp, dpp.list_of_samples))
        self.assertTrue(self.doubleton_adequation(dpp, dpp.list_of_samples))


    def test_mcmc_sampler_zonotope(self):
        """ Test whether 'zonotope' MCMC sampling mode generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by orthogonal projection correlation kernel K = A.T (A A.T)^-1 A
        """
        A = rndm.randn(self.rank, self.N)

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=True,
                        **{'A_zono': A})

        dpp.sample_mcmc(mode='zonotope', **{'nb_iter': 1000})

        self.assertTrue(self.singleton_adequation(dpp, dpp.list_of_samples[0]))
        self.assertTrue(self.doubleton_adequation(dpp, dpp.list_of_samples[0]))

    def test_mcmc_sampler_basis_exchange(self):
        """ Test whether 'E' (basis_exchange) MCMC sampling mode generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by orthogonal projection correlation kernel K from its eigendecomposition
        """

        eig_vals = np.ones(self.rank)
        eig_vecs, _ = qr(rndm.randn(self.N, self.rank), mode="economic")

        dpp = FiniteDPP(kernel_type='correlation',
                        projection=True,
                        **{'K_eig_dec': (eig_vals, eig_vecs)})

        dpp.sample_mcmc(mode='E', **{'size': self.rank, 'nb_iter': 1000})

        self.assertTrue(self.singleton_adequation(dpp, dpp.list_of_samples[0]))
        self.assertTrue(self.doubleton_adequation(dpp, dpp.list_of_samples[0]))


def main():

    unittest.main()


if __name__ == '__main__':
    main()
