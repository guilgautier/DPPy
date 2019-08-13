# coding: utf8
""" Unit tests:

- :class:`TestUniformityUniformSpanningTreeSampler` to check that the different procedures used for uniform spanning trees of a graph actually sample spanning trees uniformly at random.
"""
import unittest

import itertools as itt
from collections import Counter

import numpy as np
from scipy.stats import chisquare

from networkx import erdos_renyi_graph, is_connected, incidence_matrix

import sys
sys.path.append('..')

from dppy.utils import det_ST
from dppy.exotic_dpps import UST
from dppy.finite_dpps import FiniteDPP


class TestUniformityUniformSpanningTreeSampler(unittest.TestCase):
    """ Test uniformity the sampling procedures for sampling spanning trees of a graph uniformly at random on an Erdos-Renyi graph :math:`G(n,p)`
    """

    def __init__(self, *args, **kwargs):
        super(TestUniformityUniformSpanningTreeSampler,
              self).__init__(*args, **kwargs)

        # Sample a connected Erdos-Renyi graph
        n, p = 5, 0.4
        nb_st_min, nb_st_max = 5, 10

        it_max = 100
        for _ in range(it_max):

            g = erdos_renyi_graph(n, p)

            if is_connected(g):
                A = incidence_matrix(g, oriented=True)[:-1, :].toarray()

                potential_st = itt.combinations(range(g.number_of_edges()),
                                                n - 1)
                list_st = [st for st in potential_st
                           if det_ST(A, range(n - 1), st)]

                if nb_st_min <= len(list_st) <= nb_st_max:
                    break
        else:
            raise ValueError('No satisfactory Erdos-Renyi graph found')

        self.nb_spanning_trees = len(list_st)

        self.dpp = UST(g)

        self.nb_samples = 1000

    @staticmethod
    def sample_to_label(graph):
        """Join egdes of a sample to from the ID of the corresponding spanning tree
        Ex:
        [(3, 2), (2, 0), (2, 1), (0, 4)] -> '[0, 2][0, 4][1, 2][2, 3]'
        """
        return ''.join(map(str, sorted(map(sorted, graph.edges()))))

    def uniformity_adequation(self, tol=0.05):
        """Perform chi-square test"""

        counter = Counter(map(self.sample_to_label, self.dpp.list_of_samples))

        freq = np.array(list(counter.values())) / self.nb_samples
        theo = np.ones(self.nb_spanning_trees) / self.nb_spanning_trees

        _, pval = chisquare(f_obs=freq, f_exp=theo)

        return pval > tol

    def test_dpp_exact_projection_dpp_sampler_markov_chain(self):

        for mode in self.dpp._sampling_modes['markov-chain']:

            print('mode: {}'.format(mode))

            self.dpp.flush_samples()
            for _ in range(self.nb_samples):
                self.dpp.sample(mode=mode)

            self.assertTrue(self.uniformity_adequation())

    def test_dpp_exact_projection_dpp_sampler_eig_vecs(self):

        for mode in self.dpp._sampling_modes['spectral-method']:

            print('mode: {}'.format(mode))

            self.dpp.flush_samples()
            for _ in range(self.nb_samples):
                self.dpp.sample(mode=mode)

            self.assertTrue(self.uniformity_adequation())

    def test_dpp_exact_projection_dpp_sampler_correlation_kernel(self):

        for mode in self.dpp._sampling_modes['projection-K-kernel']:

            print('mode: {}'.format(mode))

            self.dpp.flush_samples()
            for _ in range(self.nb_samples):
                self.dpp.sample(mode=mode)

            self.assertTrue(self.uniformity_adequation())

    def test_projection_kernel_computation(self):
        """UST is a DPP associated to the projection kernel onto the row span of the vertex-edge-incidence matrix
        """
        inc = incidence_matrix(self.dpp.graph, oriented=True).todense()
        expected_kernel = np.linalg.pinv(inc).dot(inc)

        self.dpp.compute_kernel()

        self.assertTrue(np.allclose(self.dpp.kernel, expected_kernel))


def main():

    unittest.main()


if __name__ == '__main__':
    main()
