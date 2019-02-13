# coding: utf8
""" Unit tests:

- :class:`MarginalsProjectionDPP` to check that exact samplers for finite DPPs have the right (at least 1 and 2) inclusion probabilities
"""

import unittest

from numpy import array, arange, diag, histogram
from numpy.random import randn, choice

from scipy.linalg import qr
from scipy.stats import chisquare

from itertools import chain  # to flatten list of samples

import sys
sys.path.append('..')

from dppy.exact_sampling import proj_dpp_sampler_kernel_GS as kernel_GS
from dppy.exact_sampling import proj_dpp_sampler_kernel_Schur as kernel_Schur
from dppy.exact_sampling import proj_dpp_sampler_eig_GS as eig_GS
from dppy.exact_sampling import proj_dpp_sampler_eig_GS_bis as eig_GS_bis
from dppy.exact_sampling import proj_dpp_sampler_eig_KuTa12 as eig_KuTa12

from dppy.utils import det_ST


class InclusionProbabilitiesProjectionDPP(unittest.TestCase):
    """Check that exact samplers for finite DPPs have the right (at least 1 and 2) inclusion probabilities

    .. math::

        \mathbb{P}[S\subset \mathcal{X}] = \det K_S
    """

    rank, N = 6, 10
    # eig_vals = ones(rank)
    eig_vecs, _ = qr(randn(N, rank), mode="economic")
    K = eig_vecs.dot(eig_vecs.T)

    nb_samples = 100
    list_of_samples = []

    def singleton_adequation(self, tol=0.05):
        """Perform chi-square test"""

        singletons = list(chain.from_iterable(self.list_of_samples))

        freq, _ = histogram(singletons, bins=arange(self.N + 1), density=True)
        marg_theo = diag(self.K) / self.rank

        _, pval = chisquare(f_obs=freq, f_exp=marg_theo)

        return pval > tol

    def doubleton_adequation(self, tol=0.05):
        """Perform chi-square test"""

        samples = list(map(set, self.list_of_samples))

        nb_doubletons_to_check = 10
        doubletons = [set(choice(self.N, size=2, p=diag(self.K) / self.rank,
                                 replace=False))
                      for _ in range(nb_doubletons_to_check)]

        counts = [sum([doubl.issubset(sampl) for sampl in samples])
                  for doubl in doubletons]
        freq = array(counts) / self.nb_samples
        marg_theo = [det_ST(self.K, list(d)) for d in doubletons]

        _, pval = chisquare(f_obs=freq, f_exp=marg_theo)

        return pval > tol

    def test_kernel_GS(self):
        """ Test whether 'GS' sampling mode generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by projection inclusion kernel K.
        """
        self.list_of_samples = [kernel_GS(self.K)
                                for _ in range(self.nb_samples)]

        self.assertTrue(self.singleton_adequation())
        self.assertTrue(self.doubleton_adequation())

    def test_kernel_Schur(self):
        """ Test whether 'Schur' sampling mode generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by projection inclusion kernel K.
        """

        self.list_of_samples = [kernel_Schur(self.K)
                                for _ in range(self.nb_samples)]

        self.assertTrue(self.singleton_adequation())
        self.assertTrue(self.doubleton_adequation())

    def test_eig_GS(self):
        """ Test whether 'GS' sampling mode generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by its eigendecomposition
        """

        self.list_of_samples = [eig_GS(self.eig_vecs)
                                for _ in range(self.nb_samples)]

        self.assertTrue(self.singleton_adequation())
        self.assertTrue(self.doubleton_adequation())

    def test_eig_GS_bis(self):
        """ Test whether 'GS_bis' sampling mode generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by its eigendecomposition
        """

        self.list_of_samples = [eig_GS_bis(self.eig_vecs)
                                for _ in range(self.nb_samples)]

        self.assertTrue(self.singleton_adequation())
        self.assertTrue(self.doubleton_adequation())

    def test_eig_KuTa12(self):
        """ Test whether 'KuTa12' sampling mode generates samples with the right 1 and 2 points inclusion probabilities when DPP defined by its eigendecomposition
        """

        self.list_of_samples = [eig_KuTa12(self.eig_vecs)
                                for _ in range(self.nb_samples)]

        self.assertTrue(self.singleton_adequation())
        self.assertTrue(self.doubleton_adequation())


def main():

    unittest.main()


if __name__ == '__main__':
    main()
