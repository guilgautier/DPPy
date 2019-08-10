# coding: utf8
""" Unit tests:

- :class:`TestElementarySymmetricPolynomials`
"""


import unittest

import numpy as np
from itertools import combinations

import sys
sys.path.append('..')

from dppy.exact_sampling import elementary_symmetric_polynomials as e_k


class TestElementarySymmetricPolynomials(unittest.TestCase):
    """ Test the recursive evaluations of the elementary symmetric polynomials used for sampling k-DPPs :ref:`finite_dpps_exact_sampling_k_dpps`
    """
    def test_elementary_symmetric_polynomials(self):
        """
        .. math::

            e_k(\\lambda{1:r}=1, \\lambda{r+1:N}=0)
            = \\sum_{\\substack{S \\subset [N]\\\\|S|=k}}
                \\prod_{s\\in S} \\lambda{s}
        """
        N = 10
        E = np.zeros((N + 1, N + 1))
        E[0, :] = 1.0

        lmbda = np.random.randn(N)

        for k in range(1, N + 1):

            for n in range(1, N + 1):

                E[k, n] = np.sum([np.prod(lmbda_s)
                                  for lmbda_s in combinations(lmbda[:n], k)])

        self.assertTrue(np.allclose(E, e_k(lmbda, N)))


def main():

    unittest.main()


if __name__ == '__main__':
    main()
