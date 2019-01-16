# -*- coding: utf-8 -*-
""" Unit tests:

- :class:`MarginalsProjectionDPP` to check that exact samplers for finite DPPs have the right (at least 1 and 2) inclusion probabilities
"""

import unittest

from itertools import combinations
from collections import Counter

from numpy import array, ones
from numpy.linalg import det
from scipy.stats import chisquare

from networkx import erdos_renyi_graph, is_connected, incidence_matrix

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from dppy.exotic_dpps_core import ust_sampler_wilson as wilson
from dppy.exotic_dpps_core import ust_sampler_aldous_broder as aldous_broder


class TestUniformityUniformSpanningTreeSampler(unittest.TestCase):
    """ Test uniformity of Wilson and Aldous-Broder procedures for sampling spanning trees of a graph uniformly at random.
    The 2 procedures are applied to an Erdos-Renyi graph G(n,p)
    """

    # Sample an Erdos-Renyi graph
    def __init__(self, *args, **kwargs):
        super(TestUniformityUniformSpanningTreeSampler, self).__init__(*args, **kwargs)

        n, p = 5, 0.4
        nb_st_min, nb_st_max = 5, 10

        it_max = 100
        for _ in range(it_max):

            g = erdos_renyi_graph(n, p)

            if is_connected(g):
                A = incidence_matrix(g, oriented=True)[:-1, :].toarray()

                potential_st = combinations(range(g.number_of_edges()), n - 1)
                list_st = [st for st in potential_st if det(A[:, st])]

                if nb_st_min <= len(list_st) <= nb_st_max:
                    break
        else:
            raise ValueError('No satisfactory Erdos-Renyi graph found')

        self.g = g
        self.list_of_neighbors = [list(self.g.neighbors(v))
                                  for v in self.g.nodes()]

        self.nb_spanning_trees = len(list_st)

        g_edges_str = [str(set(edge)) for edge in self.g.edges()]
        self.dict_edge_label = dict(zip(g_edges_str,
                                    range(self.g.number_of_edges())))

        self.nb_samples = 1000
        self.list_of_samples = []

    def edges_to_labels(self, graph):

        edges_to_str = [str(set(edge)) for edge in graph.edges()]

        return tuple(sorted(self.dict_edge_label[e] for e in edges_to_str))

    def uniformity_adequation(self, tol=0.05):
        """Perform chi-square test"""

        edges = list(map(self.edges_to_labels, self.list_of_samples))

        counter = Counter(edges)
        freq = array(list(counter.values())) / self.nb_samples
        theo = ones(self.nb_spanning_trees) / self.nb_spanning_trees

        _, pval = chisquare(f_obs=freq, f_exp=theo)

        return pval > tol

    def test_wilson(self):
        """ Test whether wilson procedure generate uniform spanning trees uniformly at random
        """

        self.list_of_samples = [wilson(self.list_of_neighbors)
                                for _ in range(self.nb_samples)]

        self.assertTrue(self.uniformity_adequation())

    def test_aldous_broder(self):
        """ Test whether wilson procedure generate uniform spanning trees uniformly at random
        """

        self.list_of_samples = [aldous_broder(self.list_of_neighbors)
                                for _ in range(self.nb_samples)]

        self.assertTrue(self.uniformity_adequation())


def main():

    unittest.main()


if __name__ == '__main__':
    main()
