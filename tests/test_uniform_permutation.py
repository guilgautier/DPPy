import unittest
from collections import Counter
from math import factorial

import numpy as np
from scipy.stats import chisquare

from dppy.utils import uniform_permutation


class UniformityOfPermutation(unittest.TestCase):
    """Test uniformity of the implementation of `uniform_permutation`"""

    N = 5
    nb_perms = factorial(N)
    nb_samples = 5000

    @staticmethod
    def sample_to_label(samples):

        return "".join(map(str, samples))

    def test_uniform_permutation_sampler(self):
        """ """
        samples = [uniform_permutation(self.N) for _ in range(self.nb_samples)]

        self.assertTrue(self.uniformity_adequation(samples))

    def uniformity_adequation(self, samples, tol=0.05):
        """Perform chi-square test"""

        counter = Counter(map(self.sample_to_label, samples))

        freq = np.array(list(counter.values())) / self.nb_samples
        theo = np.ones(self.nb_perms) / self.nb_perms

        _, pval = chisquare(f_obs=freq, f_exp=theo)

        return pval > tol
